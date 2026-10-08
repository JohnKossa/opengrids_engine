"""Comp discovery (peel-back), ranking and pinned-comp search expansion.

This is a port of comp_adj_app's ``CompFinder``. The app compiled each
filter to a SQL ``WHERE`` clause, while this version compiles it to a pandas
boolean mask over the in-memory sales table. SQL NULL semantics are kept
(see :mod:`opengrids._util`).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from ._util import between_mask, canon, cmp_mask, coerce_scalar, eq_mask, in_mask, is_missing, like_mask, py
from .config import VALID_ABSOLUTE_OPS, _safe_float
from .dataset import Dataset, row_to_dict
from .filters import HARD_ITER_CAP

logger = logging.getLogger("opengrids")

ParamCount = Union[int, Callable[[pd.DataFrame], int]]


def _get(subject: dict, field: Optional[str]) -> Any:
    return None if field is None else py(subject.get(field))


class CompFinder:
    def __init__(self, dataset: Dataset):
        self.ds = dataset

    # ---------------------------------------------------------------- helpers
    @property
    def _sales(self) -> pd.DataFrame:
        return self.ds.sales

    def _col_eq(self, col: str, value: Any) -> pd.Series:
        s = self._sales[col]
        cs = self.ds.canon_column(col) if s.dtype == object else None
        return eq_mask(s, value, cs)

    def _absolute_mask(self, field: str, op: str, value: Any, value2: Any = None) -> Optional[pd.Series]:
        """One absolute filter as a mask, or ``None`` when its value(s) don't
        support the operator (the filter is then skipped, as in the app)."""
        s = self._sales[field]
        cs = self.ds.canon_column(field) if s.dtype == object else None
        if s.dtype == object:
            # Text column: compare the text as typed. (comp_adj_app turned
            # "09" into 9 first, so a leading-zero code could never match.)
            coerce = lambda v: v.strip() if isinstance(v, str) else v  # noqa: E731
        else:
            coerce = coerce_scalar
        if op == "is_null":
            return s.isna()
        if op == "is_not_null":
            return s.notna()
        if op in ("is_true", "is_false"):
            return eq_mask(s, 1 if op == "is_true" else 0, cs)
        if op in ("contains", "starts_with"):
            if value is None:
                return None
            text = str(value)
            return like_mask(s, f"%{text}%" if op == "contains" else f"{text}%")
        if op in ("in", "not_in"):
            if isinstance(value, list):
                values = value
            else:
                values = [v.strip() for v in str(value).split(",") if v.strip()] if value is not None else []
            if not values:
                return None
            vals = [coerce(v) for v in values]
            m = in_mask(s, vals, cs)
            return (s.notna() & ~m) if op == "not_in" else m
        if op == "between":
            v, v2 = coerce(value), coerce(value2)
            if v is None or v2 is None or v == "" or v2 == "":
                return None
            return between_mask(s, v, v2)
        v = coerce(value)
        if v is None or v == "":
            return None
        return cmp_mask(s, op, v, cs)

    def _dedup(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Latest sale per property key."""
        if not candidates.empty:
            k, d = self.ds.key, self.ds.sale_date_field
            candidates[d] = pd.to_datetime(candidates[d])
            candidates = candidates.sort_values([k, d], ascending=[True, False])
            candidates = candidates.drop_duplicates(subset=[k], keep="first")
        return candidates

    def _query(self, subject_key: str, masks: list[pd.Series]) -> pd.DataFrame:
        m = self._sales[self.ds.key] != str(subject_key)
        for x in masks:
            m &= x
        return self._dedup(self._sales[m].copy())

    def lookup_subject(self, key: Any) -> Optional[dict]:
        """Universe row for ``key``, falling back to its sale row when the
        dataset has no universe."""
        try:
            return self.ds.subject(key)
        except KeyError:
            return self.ds.sales_row(key)

    # ------------------------------------------------------------ peel engine
    def _init_work(self, filter_stack: list[dict]) -> list[dict]:
        work = []
        for f in filter_stack:
            w = dict(f)
            w["dropped"] = False
            w["current"] = dict(f.get("params", {}))
            peel = f.get("peel", {})
            w["_behavior"] = peel.get("behavior", "hold")
            w["_step"] = peel.get("step", {})
            w["_max_value"] = None
            w["_relax_key"] = None
            kind = f.get("kind")
            if kind == "spatial":
                ladder = peel.get("ladder") or ([{"field": f["field"], "clean_to_next": False}] if f.get("field") else [])
                w["_ladder"] = ladder
                w["_ladder_pos"] = 0
                w["active_levels"] = [ladder[0]["field"]] if ladder else ([f["field"]] if f.get("field") else [])
            elif kind == "geo_radius":
                base = float(f.get("params", {}).get("radius", 0) or 0)
                extra = float(peel.get("max_relax", {}).get("units", 0) or 0)
                w["_max_value"] = base + extra
                w["_relax_key"] = "radius"
            elif kind == "relative" and f.get("params", {}).get("type") in ("pct", "units"):
                base = float(f.get("params", {}).get("value", 0) or 0)
                mr = peel.get("max_relax", {})
                extra = float(mr.get("pct", mr.get("units", 0)) or 0)
                w["_max_value"] = base + extra
                w["_relax_key"] = "value"
            work.append(w)
        return work

    def _compile(self, work: list[dict], subject: dict) -> list[pd.Series]:
        """Masks for the working stack's current (possibly relaxed) state."""
        sales = self._sales
        cols = set(sales.columns)
        lat_f, lng_f = self.ds.lat_field, self.ds.lng_field
        masks: list[pd.Series] = []
        for f in work:
            if not f.get("enabled", True) or f.get("dropped"):
                continue
            kind = f.get("kind")
            cur = f.get("current", {})
            if kind == "geo_radius":
                lat0, lng0 = _get(subject, lat_f), _get(subject, lng_f)
                if lat0 is None or lng0 is None or lat_f not in cols or lng_f not in cols:
                    continue
                lat0, lng0 = float(lat0), float(lng0)
                r = float(cur.get("radius", 0) or 0) / 69.0  # degrees, roughly
                masks.append(between_mask(sales[lat_f], lat0 - r, lat0 + r)
                             & between_mask(sales[lng_f], lng0 - r, lng0 + r))
            elif kind == "relative":
                field = f.get("field")
                if not field or field not in subject or field not in cols:
                    continue
                val = _get(subject, field)
                if val is None:
                    continue
                t = cur.get("type")
                if t == "exact":
                    masks.append(self._col_eq(field, val))
                elif t == "pct":
                    v = float(val)
                    pct = float(cur.get("value", 0) or 0) / 100.0
                    masks.append(between_mask(sales[field], v * (1 - pct), v * (1 + pct)))
                elif t == "units":
                    v = float(val)
                    u = float(cur.get("value", 0) or 0)
                    masks.append(between_mask(sales[field], v - u, v + u))
            elif kind == "spatial":
                level_masks = []
                for level in f.get("active_levels", []):
                    if level and level in cols and level in subject and _get(subject, level) is not None:
                        level_masks.append(self._col_eq(level, _get(subject, level)))
                if level_masks:
                    m = level_masks[0]
                    for lm in level_masks[1:]:
                        m = m | lm
                    masks.append(m)
            elif kind == "absolute":
                field = f.get("field")
                op = str(cur.get("op", "") or "").strip().lower()
                if field and field in cols and op in VALID_ABSOLUTE_OPS:
                    m = self._absolute_mask(field, op, cur.get("value"), cur.get("value2"))
                    if m is not None:
                        masks.append(m)
        return masks

    @staticmethod
    def _next_relaxation(work: list[dict]) -> Optional[dict]:
        """First eligible relaxation, scanning from the bottom of the stack up."""
        for f in reversed(work):
            if f.get("dropped"):
                continue
            behavior = f.get("_behavior", "hold")
            if behavior == "hold":
                continue
            if f.get("kind") == "spatial":
                ladder = f.get("_ladder", [])
                if behavior in ("relax", "relax_and_drop") and f.get("_ladder_pos", 0) < len(ladder) - 1:
                    return {"f": f, "action": "relax"}
                if behavior in ("drop", "relax_and_drop"):
                    return {"f": f, "action": "drop"}
                continue
            if behavior in ("relax", "relax_and_drop") and f.get("_relax_key"):
                cur = float(f["current"].get(f["_relax_key"], 0) or 0)
                mx = f.get("_max_value")
                if mx is not None and cur < mx - 1e-9:
                    return {"f": f, "action": "relax"}
                if behavior == "relax_and_drop":
                    return {"f": f, "action": "drop"}
                continue
            if behavior in ("drop", "relax_and_drop"):
                return {"f": f, "action": "drop"}
        return None

    @staticmethod
    def _apply_step(step: dict) -> dict:
        """Apply one relaxation and return a structured record of the change."""
        f = step["f"]
        base = {"kind": f.get("kind"), "id": f.get("id"), "field": f.get("field")}
        if step["action"] == "drop":
            f["dropped"] = True
            return {"action": "drop", **base}
        if f.get("kind") == "spatial":
            ladder = f.get("_ladder", [])
            pos = f.get("_ladder_pos", 0)
            clean = bool(ladder[pos].get("clean_to_next")) if pos < len(ladder) else False
            prev, nxt = ladder[pos]["field"], ladder[pos + 1]["field"]
            f["active_levels"] = [nxt] if clean else list(f.get("active_levels", [])) + [nxt]
            f["_ladder_pos"] = pos + 1
            return {"action": "ladder", **base, "from": prev, "to": nxt, "mode": "replace" if clean else "union"}
        key = f["_relax_key"]
        stepvals = f.get("_step", {})
        inc = float(stepvals.get("units", stepvals.get("pct", 0)) or 0)
        cur = float(f["current"].get(key, 0) or 0)
        newv = cur + inc
        if f.get("_max_value") is not None:
            newv = min(newv, f["_max_value"])
        f["current"][key] = newv
        if f.get("kind") == "geo_radius":
            unit = "mi"
        elif (f.get("params") or {}).get("type") == "pct":
            unit = "%"
        else:
            unit = "units"
        return {"action": "relax", **base, "from": cur, "to": newv, "unit": unit}

    def find_candidates_peeled(self, subject: dict, filter_stack: list[dict], policy: dict,
                               param_count: ParamCount = 0) -> tuple[pd.DataFrame, list[dict]]:
        """Query, and if the pool is short of target, relax the next filter
        and query again. Returns ``(candidates, audit)``. ``audit`` has one
        entry per iteration recording the pool size, the target and what
        changed."""
        subject_key = str(subject.get(self.ds.key))
        work = self._init_work(filter_stack)
        target = (policy or {}).get("target", {})
        min_comps = int(target.get("min_comps", 0) or 0)
        min_cpp = int(target.get("min_comps_per_param", 0) or 0)

        def _needed(c: pd.DataFrame) -> tuple[int, int]:
            pc = param_count(c) if callable(param_count) else int(param_count or 0)
            return max(min_comps, (min_cpp * pc) if pc else 0), pc

        audit: list[dict] = []
        last_change: dict = {"action": "base"}
        candidates = pd.DataFrame()
        for i in range(HARD_ITER_CAP):
            candidates = self._query(subject_key, self._compile(work, subject))
            needed, pc = _needed(candidates)
            audit.append({"iter": i, "comps": int(len(candidates)), "param_count": int(pc),
                          "needed": int(needed), "change": last_change})
            if len(candidates) >= needed:
                audit[-1]["stop"] = "target met"
                break
            step = self._next_relaxation(work)
            if step is None:
                audit[-1]["stop"] = "exhausted"
                break
            last_change = self._apply_step(step)
        return candidates, audit

    # ------------------------------------------- legacy criteria (pinned comps)
    def build_search_criteria(self, cfg: dict, subject: dict) -> dict:
        """The flat criteria dict that the pinned-comp ``expand_search`` path
        uses. Spatial matching becomes an exact match on the finest ladder
        level."""
        criteria: dict[str, Any] = {}
        absolute_filters = []
        ladder = self.ds.spatial_ladder
        for f in cfg.get("filters", []):
            if not f.get("enabled", True):
                continue
            kind = f.get("kind")
            params = f.get("params", {}) or {}
            if kind == "geo_radius":
                r = _safe_float(params.get("radius"), None)
                if r is not None:
                    criteria["radius"] = r
            elif kind == "spatial":
                if ladder:
                    finest = ladder[0].field
                    if finest and finest in subject and _get(subject, finest) is not None:
                        criteria[finest] = {"exact": True}
            elif kind == "relative":
                field = f.get("field")
                if not field:
                    continue
                t = params.get("type")
                val = _safe_float(params.get("value"), 0.0)
                if t == "pct":
                    criteria[field] = {"pct": val / 100.0}
                elif t == "exact":
                    criteria[field] = {"exact": True}
                else:
                    criteria[field] = {"units": val}
            elif kind == "absolute":
                field, op = f.get("field"), params.get("op")
                if field and op:
                    item = {"field": field, "op": op}
                    if "value" in params:
                        item["value"] = params.get("value")
                    if "value2" in params:
                        item["value2"] = params.get("value2")
                    absolute_filters.append(item)
        if absolute_filters:
            criteria["absolute_filters"] = absolute_filters
        return criteria

    def find_candidates(self, subject: Optional[dict], criteria: dict) -> pd.DataFrame:
        """Unpeeled candidate query from a flat criteria dict."""
        if subject is None:
            return pd.DataFrame()
        sales = self._sales
        cols = set(sales.columns)
        masks: list[pd.Series] = []
        for filt in criteria.get("absolute_filters", []) or []:
            if not isinstance(filt, dict):
                continue
            field = filt.get("field")
            op = str(filt.get("op", "")).strip().lower()
            if not field or field not in cols or op not in VALID_ABSOLUTE_OPS:
                continue
            m = self._absolute_mask(field, op, filt.get("value"), filt.get("value2"))
            if m is not None:
                masks.append(m)
        lat_f, lng_f = self.ds.lat_field, self.ds.lng_field
        if "radius" in criteria and lat_f in subject and lng_f in subject:
            lat0, lng0 = _get(subject, lat_f), _get(subject, lng_f)
            if lat0 is not None and lng0 is not None:
                r = criteria["radius"] / 69.0
                masks.append(between_mask(sales[lat_f], lat0 - r, lat0 + r))
                masks.append(between_mask(sales[lng_f], lng0 - r, lng0 + r))
        for field, limit in criteria.items():
            if field in ("radius", "absolute_filters") or field not in subject or field not in cols:
                continue
            val = _get(subject, field)
            if val is None or not isinstance(limit, dict):
                continue
            if "pct" in limit:
                masks.append(between_mask(sales[field], val * (1 - limit["pct"]), val * (1 + limit["pct"])))
            elif "units" in limit:
                masks.append(between_mask(sales[field], val - limit["units"], val + limit["units"]))
            elif "exact" in limit:
                masks.append(self._col_eq(field, val))
        return self._query(str(subject.get(self.ds.key)), masks)

    # ---------------------------------------------------------------- ranking
    def rank_candidates(self, subject: dict, candidates: pd.DataFrame, ranking_config: Optional[dict] = None,
                        unit_adjs: Optional[dict] = None, limit: int = 10) -> pd.DataFrame:
        if candidates.empty:
            return candidates.head(0)
        candidates = candidates.copy()
        method = (ranking_config or {}).get("method", "closest_price")
        if method == "closest_price":
            subject_val = (ranking_config or {}).get("subject_prediction_cost_of_sale_adj")
            if subject_val is None:
                subject_val = _get(subject, self.ds.prediction_field)
            subject_val = np.nan if subject_val is None else subject_val
            candidates["ranking_score"] = (candidates[self.ds.sale_price_time_adj_field] - subject_val).abs()
        elif method == "smallest_adj" and unit_adjs:
            total = pd.Series(0.0, index=candidates.index)
            for feat, rate_or_map in unit_adjs.items():
                if feat not in candidates.columns:
                    continue
                if isinstance(rate_or_map, dict):
                    s_rate = rate_or_map.get(canon(subject.get(feat)), 0)
                    total += candidates[feat].map(lambda x: s_rate - rate_or_map.get(canon(x), 0)).abs()
                else:
                    sv = _get(subject, feat)
                    sv = np.nan if sv is None else sv
                    total += ((sv - candidates[feat]) * rate_or_map).abs()
            candidates["ranking_score"] = total
        elif method == "weighted_euclidean":
            weights = (ranking_config or {}).get("weights", {}) or {}
            feats = list(weights.keys()) if weights else ["bldg_area_finished_sqft", "bldg_year_built", "land_area_sqft"]
            feats = [f for f in feats if f in candidates.columns and f in subject]
            if feats:
                sub = np.array([np.nan if _get(subject, f) is None else _get(subject, f) for f in feats], dtype=float)
                cand = candidates[feats].values.astype(float)
                w = np.array([float(weights.get(f, 1.0)) for f in feats])
                candidates["ranking_score"] = np.sqrt(np.sum(((cand - sub) * w) ** 2, axis=1))
            else:
                candidates["ranking_score"] = 0
        else:
            candidates["ranking_score"] = 0
        # Secondary sort on key keeps ties reproducible from run to run.
        return candidates.sort_values(["ranking_score", self.ds.key]).head(limit).copy()

    def calculate_distances(self, subject: dict, candidates: pd.DataFrame) -> pd.DataFrame:
        """Haversine distance from the subject, in miles."""
        lat_f, lng_f = self.ds.lat_field, self.ds.lng_field
        lat0, lng0 = _get(subject, lat_f), _get(subject, lng_f)
        if (lat0 is not None and lng0 is not None and not candidates.empty
                and lat_f in candidates.columns and lng_f in candidates.columns):
            slat, slon = np.radians(lat0), np.radians(lng0)
            clats, clons = np.radians(candidates[lat_f]), np.radians(candidates[lng_f])
            a = np.sin((clats - slat) / 2) ** 2 + np.cos(slat) * np.cos(clats) * np.sin((clons - slon) / 2) ** 2
            candidates["distance_miles"] = 3956 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        else:
            candidates["distance_miles"] = 0.0
        return candidates

    def _records(self, ranked: pd.DataFrame) -> list[dict]:
        if ranked.empty:
            return []
        out = ranked[[self.ds.key, "distance_miles"]].copy()
        out[self.ds.key] = out[self.ds.key].astype(str)
        return [{"key": r[self.ds.key], "distance_miles": r["distance_miles"]} for r in out.to_dict("records")]

    def find_comps(self, subject: dict, criteria: dict, ranking_config: Optional[dict] = None,
                   unit_adjs: Optional[dict] = None, limit: int = 10) -> list[dict]:
        candidates = self.find_candidates(subject, criteria)
        if candidates.empty:
            return []
        ranked = self.rank_candidates(subject, candidates, ranking_config, unit_adjs, limit=limit)
        return self._records(self.calculate_distances(subject, ranked))

    def expand_search(self, subject: dict, custom_comp_id: str, method: str, criteria: dict,
                      ranking_config: Optional[dict] = None, unit_adjs: Optional[dict] = None
                      ) -> tuple[list[dict], dict]:
        """Rebuild the ranked pool so that a hand-picked comp can be included.

        Methods: ``none`` (plain search), ``second_subject`` (union with a
        search centred on the custom comp), ``expanding_radius`` (widen the
        radius until it reaches the comp), ``stretching_range`` (widen each
        relative band until it covers the comp).
        """
        if method == "second_subject":
            cand_subject = self.find_candidates(subject, criteria)
            cand_custom = self.find_candidates(self.lookup_subject(custom_comp_id), criteria)
            parts = [c for c in (cand_subject, cand_custom) if not c.empty]
            candidates = (pd.concat(parts).drop_duplicates(subset=[self.ds.key]) if parts else pd.DataFrame())
            if candidates.empty:
                return [], {"method": "second_subject", "second_subject_id": str(custom_comp_id)}
            ranked = self.rank_candidates(subject, candidates, ranking_config, unit_adjs)
            ranked = self.calculate_distances(subject, ranked)
            return self._records(ranked), {"method": "second_subject", "second_subject_id": str(custom_comp_id)}

        if method == "none":
            return self.find_comps(subject, criteria, ranking_config, unit_adjs), {"method": "none"}

        comp_row = self._first_sale_row(custom_comp_id)
        lat_f, lng_f = self.ds.lat_field, self.ds.lng_field
        if method == "expanding_radius":
            if (comp_row is not None and _get(subject, lat_f) is not None and _get(subject, lng_f) is not None
                    and comp_row.get(lat_f) is not None and comp_row.get(lng_f) is not None):
                max_dist = max(abs(subject[lat_f] - comp_row[lat_f]), abs(subject[lng_f] - comp_row[lng_f]))
                required = round((max_dist * 69.0) + 0.01, 4)
                try:
                    current = float(criteria.get("radius", 0) or 0)
                except (TypeError, ValueError):
                    current = 0.0
                effective = max(current, required)
                new_criteria = {**criteria, "radius": effective}
                info = {"method": "expanding_radius", "original_radius": criteria.get("radius"),
                        "effective_radius": effective}
                return self.find_comps(subject, new_criteria, ranking_config, unit_adjs), info

        elif method == "stretching_range":
            if comp_row is None:
                return self.find_comps(subject, criteria, ranking_config, unit_adjs), {}
            new_criteria = dict(criteria)
            stretched: dict[str, dict] = {}
            for field, limit in criteria.items():
                if field == "radius" or not isinstance(limit, dict):
                    continue
                s_val, c_val = _get(subject, field), comp_row.get(field)
                if s_val is None or c_val is None:
                    continue
                if "pct" in limit:
                    diff = abs(s_val - c_val)
                    required_pct = diff / float(s_val) if s_val != 0 else 0
                    if required_pct > limit["pct"]:
                        new_criteria[field] = {"pct": required_pct + 0.001}
                        stretched[field] = {"original_pct": round(limit["pct"] * 100, 2),
                                            "effective_pct": round((required_pct + 0.001) * 100, 2)}
                elif "units" in limit:
                    diff = abs(s_val - c_val)
                    if diff > limit["units"]:
                        new_criteria[field] = {"units": round(diff + 0.001, 4)}
                        stretched[field] = {"original_units": limit["units"],
                                            "effective_units": round(diff + 0.001, 4)}
            info = {"method": "stretching_range", "stretched_fields": stretched}
            return self.find_comps(subject, new_criteria, ranking_config, unit_adjs), info

        return self.find_comps(subject, criteria, ranking_config, unit_adjs), {}

    def _first_sale_row(self, key: Any) -> Optional[dict]:
        rows = self._sales[self._sales[self.ds.key] == str(key)]
        return None if rows.empty else row_to_dict(rows.iloc[0])
