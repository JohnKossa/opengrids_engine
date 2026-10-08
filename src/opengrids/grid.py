"""Assembling a full-detail comp grid: discovery, adjustment, ranking, selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from ._util import canon, json_safe
from .adjustments import compute_adjustments
from .dataset import Dataset
from .errors import GridConfigError
from .fidelity import surrogate_fidelity
from .filters import build_filter_stack, peel_policy
from .finder import CompFinder

logger = logging.getLogger("opengrids")


@dataclass
class CompGrid:
    """A finished grid.

    ``comps`` are dicts holding the full sales row plus:
    ``distance_miles``, ``time_adj_pct``, ``sale_price_cost_of_sale_adj``
    (the baseline that adjustments are added to), ``adjustments``
    (``{field: $}``), ``total_adj``, ``final_price`` and ``is_user_comp``.
    They're ordered by sale date, newest first.
    """

    subject: dict
    comps: list[dict]
    features: list[dict]
    unit_adjs: dict
    raw_unit_adjs: dict
    clamped_fields: list[dict]
    rate_sources: dict
    model_rates: dict
    adj_fields: list[str]
    modeled_fields: list[str]
    fwl_fields: list[str]
    candidate_ids: list[str]
    excluded_comp_count: int
    peel_audit: list[dict]
    surrogate_fidelity: Optional[dict]
    use_cost_of_sale: bool
    valuation_date: Any
    generated_at: datetime
    config: dict
    schema: dict = field(default_factory=dict)

    @property
    def subject_key(self) -> str:
        return str(self.subject.get(self.schema.get("key", "key")))

    @property
    def subject_value(self) -> Optional[float]:
        """The subject's model value on the same basis as ``final_price``."""
        v = self.subject.get("prediction_cost_of_sale_adj")
        return v if v is not None else self.subject.get(self.schema.get("prediction_field", "prediction"))

    def metrics(self) -> dict:
        from .metrics import grid_metrics
        return grid_metrics(self)

    def rates(self) -> pd.DataFrame:
        """One row per adjustment field: the applied rate, what the data alone
        says (the fitted rate before any sign clamp or override), and where the
        applied rate came from."""
        rows = []
        for f in self.features:
            n = f["name"]
            applied = self.unit_adjs.get(n)
            raw = self.model_rates.get(n, self.raw_unit_adjs.get(n))
            fmt = lambda r: ({k: round(float(v), 2) for k, v in r.items()} if isinstance(r, dict)  # noqa: E731
                             else None if r is None else float(r))
            rows.append({"field": n, "label": f["label"], "rate": fmt(applied), "fitted_rate": fmt(raw),
                         "source": f.get("rate_source")})
        return pd.DataFrame(rows).set_index("field")

    def to_frame(self) -> pd.DataFrame:
        """The grid as a table: one column for the subject and one per comp,
        with a value row and a ``$`` adjustment row for each field."""
        s = self.schema
        cols = {"subject": {}}
        for c in self.comps:
            cols[str(c.get(s["key"]))] = {}
        rows: list[str] = []

        def put(row: str, subj: Any, per_comp) -> None:
            rows.append(row)
            cols["subject"][row] = subj
            for c in self.comps:
                cols[str(c.get(s["key"]))][row] = per_comp(c)

        put("sale_date", None, lambda c: c.get("sale_date"))
        put("sale_price", None, lambda c: c.get(s["sale_price_field"]))
        put("sale_price_time_adj", None, lambda c: c.get(s["sale_price_time_adj_field"]))
        put("time_adj_pct", None, lambda c: c.get("time_adj_pct"))
        if self.use_cost_of_sale:
            put("sale_price_cost_of_sale_adj", None, lambda c: c.get("sale_price_cost_of_sale_adj"))
        put("distance_miles", None, lambda c: c.get("distance_miles"))
        for f in self.features:
            name = f["name"]
            put(f["label"], self.subject.get(name), lambda c, n=name: c.get(n))
            put(f"{f['label']} adj", None, lambda c, n=name: c["adjustments"].get(n))
        put("total_adj", None, lambda c: c.get("total_adj"))
        put("adjusted_price", self.subject_value, lambda c: c.get("final_price"))
        return pd.DataFrame(cols).reindex(rows)

    def to_dict(self, diagnostics: bool = False) -> dict:
        """A narrow, JSON-safe projection suitable for correspondence or an
        API response. Set ``diagnostics=True`` to include the peel trace,
        surrogate fidelity, clamped fields and grid metrics."""
        s = self.schema
        names = [f["name"] for f in self.features]
        subject_out = {
            "id": self.subject.get(s["key"]),
            "prediction": self.subject.get(s["prediction_field"]),
            "prediction_cost_of_sale_adj": self.subject.get("prediction_cost_of_sale_adj"),
            "features": {f["name"]: {"label": f["label"], "value": self.subject.get(f["name"]),
                                     "is_integer": f["is_integer"]} for f in self.features},
        }
        if "address" in self.subject:
            subject_out["address"] = self.subject.get("address")
        comps_out = []
        for c in self.comps:
            co = {
                "id": c.get(s["key"]),
                "sale_date": c.get("sale_date"),
                "sale_price": c.get(s["sale_price_field"]),
                "sale_price_time_adj": c.get(s["sale_price_time_adj_field"]),
                "time_adj_pct": c.get("time_adj_pct"),
                "sale_price_cost_of_sale_adj": c.get("sale_price_cost_of_sale_adj"),
                "distance_miles": c.get("distance_miles"),
                "features": {n: c.get(n) for n in names},
                "adjustments": c.get("adjustments", {}),
                "total_adjustment": c.get("total_adj"),
                "adjusted_price": c.get("final_price"),
                "is_user_comp": c.get("is_user_comp", False),
            }
            if "address" in c:
                co["address"] = c.get("address")
            comps_out.append(co)
        out = {
            "parcel_id": self.subject_key,
            "valuation_date": self.valuation_date,
            "use_cost_of_sale": self.use_cost_of_sale,
            "generated_at": self.generated_at,
            "subject": subject_out,
            "adjustment_features": [
                {**{k: f[k] for k in ("name", "label", "category", "impact", "rate_source")},
                 "unit_adj": self.unit_adjs.get(f["name"])}
                for f in self.features
            ],
            "comps": comps_out,
        }
        if diagnostics:
            out["diagnostics"] = {
                "metrics": self.metrics(),
                "peel_audit": self.peel_audit,
                "surrogate_fidelity": self.surrogate_fidelity,
                "clamped_fields": self.clamped_fields,
                "raw_unit_adjs": self.raw_unit_adjs,
                "modeled_fields": self.modeled_fields,
                "fwl_fields": self.fwl_fields,
                "candidate_ids": self.candidate_ids,
                "excluded_comp_count": self.excluded_comp_count,
            }
        return json_safe(out)


def _validate(ds: Dataset, adj_fields: list[str], subject: dict) -> list[dict]:
    problems = []
    for f in adj_fields:
        missing_from = [name for name, ok in (("sales", f in ds.sales.columns), ("subject", f in subject)) if not ok]
        if missing_from:
            problems.append({"section": "adj_fields", "field": f,
                             "reason": f"Column not found in {' or '.join(missing_from)}.",
                             "suggestion": f'Remove "{f}" from adj_fields or add the column.'})
    if problems:
        return problems
    for f in adj_fields:
        if f in ds.contributions.columns and not pd.api.types.is_numeric_dtype(ds.contributions[f].dtype):
            problems.append({"section": "adj_fields", "field": f,
                             "reason": "Its SHAP contribution column is not numeric.",
                             "suggestion": "Contribution dollars must be numeric, even for a categorical field."})
            continue
        s = ds.sales[f]
        if (not ds.is_categorical(f) and s.dtype == object
                and pd.to_numeric(s.dropna(), errors="coerce").isna().any()):
            problems.append({"section": "adj_fields", "field": f,
                             "reason": f'"{f}" holds text but is not marked categorical.',
                             "suggestion": "Mark it categorical in FieldMeta or remove it from adj_fields."})
    return problems


def build_grid(ds: Dataset, subject: dict, cfg: dict, user_comp_id: Optional[str] = None) -> CompGrid:
    """Build one grid. ``cfg`` must already be normalized, with
    ``adj_fields`` resolved to a list."""
    subject = dict(subject)
    subject_key = str(subject.get(ds.key))
    adj_fields_requested = list(cfg.get("adj_fields") or [])
    problems = _validate(ds, adj_fields_requested, subject)
    if problems:
        raise GridConfigError(problems)

    cos = ds.cost_of_sale
    finder = CompFinder(ds)
    try:
        criteria = finder.build_search_criteria(cfg, subject)

        # 1. Peel-back discovery of the pool the surrogate is fitted on.
        stack = build_filter_stack(cfg, ds.spatial_ladder)
        policy = peel_policy(cfg)

        def _param_count(cands: pd.DataFrame) -> int:
            n = 0
            for f in adj_fields_requested:
                if ds.is_categorical(f):
                    n += (int(cands[f].nunique(dropna=True)) or 1) if (not cands.empty and f in cands.columns) else 1
                else:
                    n += 1
            return n

        candidates, audit = finder.find_candidates_peeled(subject, stack, policy, param_count=_param_count)
        cand_ids = candidates[ds.key].astype(str).tolist() if not candidates.empty else []
        subset_ids = [subject_key] + cand_ids

        # 2. Unit adjustments over that pool.
        adj = compute_adjustments(ds, subject_key, subset_ids, cfg)
        unit_adjs, effective = adj["unit_adjs"], adj["effective_unit_adjs"]
        adj_fields = adj["adj_fields"]

        pred = subject.get(ds.prediction_field)
        if pred is None:
            subject["prediction_cost_of_sale_adj"] = None
        elif cos is not None and not cos.model_trained_on_cost_of_sale:
            subject["prediction_cost_of_sale_adj"] = pred * cos.factor
        else:
            subject["prediction_cost_of_sale_adj"] = pred

        # 3. Rank a deeper bench than the grid shows, for comp-quality backfill.
        grid_cfg = cfg.get("grid", {}) or {}
        ranking_config = dict(cfg.get("ranking", {}) or {})
        ranking_config["subject_prediction_cost_of_sale_adj"] = subject["prediction_cost_of_sale_adj"]
        if not candidates.empty:
            ranked = finder.rank_candidates(subject, candidates, ranking_config, unit_adjs,
                                            limit=int(grid_cfg.get("pool_size", 25)))
            ranked = finder.calculate_distances(subject, ranked)
            comp_results = finder._records(ranked)
        else:
            comp_results = []

        fidelity = None
        try:
            fidelity = surrogate_fidelity(ds, cand_ids, adj["modeled_fields"])
        except Exception:  # diagnostic only; never fail a grid over it
            logger.exception("surrogate fidelity failed for %s", subject_key)

        # 4. A pinned comp always takes a slot and skips the quality checks.
        if user_comp_id is not None:
            user_comp_id = str(user_comp_id)
            if ds.has_sale(user_comp_id):
                if user_comp_id not in [r["key"] for r in comp_results]:
                    comp_results, _ = finder.expand_search(subject, user_comp_id, cfg.get("expansion", "none"),
                                                           criteria, ranking_config, unit_adjs)
            else:
                user_comp_id = None

        distance_lookup = {r["key"]: r["distance_miles"] for r in comp_results}
        pool_keys = [r["key"] for r in comp_results]
        rows_by_key: dict[str, dict] = {}
        for k in dict.fromkeys(([user_comp_id] if user_comp_id else []) + pool_keys):
            row = ds.sales_row(k)
            if row is not None:
                if ds.sale_date_field != "sale_date":
                    row["sale_date"] = row.get(ds.sale_date_field)
                rows_by_key[k] = row

        max_adj_pct = grid_cfg.get("max_adjustment_pct")

        def _score(key: str) -> dict:
            comp = rows_by_key[key]
            comp["distance_miles"] = distance_lookup.get(key)
            raw = comp.get(ds.sale_price_field)
            tadj = comp.get(ds.sale_price_time_adj_field)
            comp["time_adj_pct"] = (tadj - raw) / raw * 100 if raw and tadj is not None else None
            if cos is not None and cos.source == "field" and cos.field:
                comp["sale_price_cost_of_sale_adj"] = comp.get(cos.field)
            elif cos is not None:
                comp["sale_price_cost_of_sale_adj"] = tadj * cos.factor
            else:
                comp["sale_price_cost_of_sale_adj"] = tadj
            comp["adjustments"] = {}
            total = 0
            for feat in adj_fields:
                rate = effective.get(feat, 0)
                if isinstance(rate, dict):
                    sr = rate.get(canon(subject.get(feat)), 0)
                    cr = rate.get(canon(comp.get(feat)), 0)
                    sr = 0 if pd.isna(sr) else sr
                    cr = 0 if pd.isna(cr) else cr
                    a = sr - cr
                elif subject.get(feat) is None or comp.get(feat) is None:
                    a = 0
                else:
                    a = (subject[feat] - comp[feat]) * rate
                comp["adjustments"][feat] = a
                total += a
            comp["total_adj"] = total
            comp["final_price"] = comp["sale_price_cost_of_sale_adj"] + total
            comp["is_user_comp"] = key == user_comp_id
            return comp

        chosen: dict[str, dict] = {}
        if user_comp_id and user_comp_id in rows_by_key:
            chosen[user_comp_id] = _score(user_comp_id)

        excluded = 0
        target = int(grid_cfg.get("size", 4))

        def _passes(comp: dict) -> bool:
            nonlocal excluded
            if comp["final_price"] < 0:
                excluded += 1
                return False
            if max_adj_pct is not None and comp["sale_price_cost_of_sale_adj"]:
                if abs(comp["total_adj"]) / comp["sale_price_cost_of_sale_adj"] * 100 > max_adj_pct:
                    excluded += 1
                    return False
            return True

        if (cfg.get("ranking", {}) or {}).get("prefer_bracketed"):
            passing = []
            for k in pool_keys:
                if k in chosen or k not in rows_by_key:
                    continue
                comp = _score(k)
                if _passes(comp):
                    passing.append((k, comp))
            basis = subject.get("prediction_cost_of_sale_adj", subject.get(ds.prediction_field))
            slots = target - len(chosen)
            picked: set[str] = set()
            if basis is not None and slots > 0:
                above = next((kc for kc in passing if kc[1]["final_price"] > basis), None)
                if above:
                    chosen[above[0]] = above[1]; picked.add(above[0]); slots -= 1
            if basis is not None and slots > 0:
                below = next((kc for kc in passing if kc[1]["final_price"] < basis and kc[0] not in picked), None)
                if below:
                    chosen[below[0]] = below[1]; picked.add(below[0]); slots -= 1
            for k, comp in passing:
                if slots <= 0:
                    break
                if k in picked:
                    continue
                chosen[k] = comp; picked.add(k); slots -= 1
        else:
            for k in pool_keys:
                if len(chosen) >= target:
                    break
                if k in chosen or k not in rows_by_key:
                    continue
                comp = _score(k)
                if _passes(comp):
                    chosen[k] = comp

        comps = sorted(chosen.values(), key=lambda c: _date_key(c.get("sale_date")), reverse=True)
    except GridConfigError:
        raise
    except KeyError as e:
        missing = e.args[0] if e.args else str(e)
        raise GridConfigError([{"section": "unknown", "field": str(missing),
                                "reason": f'Column "{missing}" was referenced while building the grid but is not in the data.',
                                "suggestion": "Check filters, ranking weights and adj_fields for this column."}]) from e
    except (TypeError, ValueError) as e:
        suspects = ", ".join(f'"{f}"' for f in adj_fields_requested) or "(no adjustment fields configured)"
        raise GridConfigError([{"section": "adj_fields", "field": suspects,
                                "reason": f"A calculation error occurred while computing adjustments ({e}). "
                                          f"An adjustment field likely holds missing or non-numeric data.",
                                "suggestion": "Remove adjustment fields one at a time to isolate the culprit."}]) from e

    integer_fields = ds.integer_fields()
    clamped_names = {c["name"] for c in adj["clamped_fields"]}
    features = []
    for f in adj_fields:
        m = ds.meta(f)
        features.append({"name": f, "label": m.label or f, "category": m.category or "unknown",
                         "impact": m.impact or "mixed", "is_categorical": bool(m.is_categorical),
                         "is_boolean": bool(m.is_boolean), "warning": f in clamped_names,
                         "rate_source": adj["rate_sources"].get(f),
                         "is_integer": f in integer_fields})

    return CompGrid(
        subject=subject, comps=comps, features=features, unit_adjs=effective, raw_unit_adjs=unit_adjs,
        clamped_fields=adj["clamped_fields"], rate_sources=adj["rate_sources"],
        model_rates=adj["model_rates"], adj_fields=adj_fields, modeled_fields=adj["modeled_fields"],
        fwl_fields=adj["fwl_fields"], candidate_ids=cand_ids, excluded_comp_count=excluded,
        peel_audit=audit, surrogate_fidelity=fidelity, use_cost_of_sale=cos is not None,
        valuation_date=ds.valuation_date, generated_at=datetime.now(), config=cfg,
        schema={"key": ds.key, "sale_price_field": ds.sale_price_field,
                "sale_price_time_adj_field": ds.sale_price_time_adj_field,
                "prediction_field": ds.prediction_field},
    )


def _date_key(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return v.isoformat()
    return str(v)
