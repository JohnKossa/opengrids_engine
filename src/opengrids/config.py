"""Search configuration: which comps qualify, how they're ranked, what gets adjusted.

A config is a plain dict (JSON-serializable), so it can live in a file, a
database row or a web form. It uses the same schema comp_adj_app
stored in ``dataset_registry.default_search_config``, extended with a
``grid`` block for knobs the app hard-coded. Use :func:`normalize_config`
to validate a partial or hand-written config and fill in defaults.

Shape::

    {
      "filters": [               # ORDERED: top = most protected, peeled last
        {"id": "spatial",    "kind": "spatial",    "enabled": True, "field": None, "params": {}, "peel": {}},
        {"id": "geo_radius", "kind": "geo_radius", "enabled": True, "field": None, "params": {"radius": 1.0}, "peel": {}},
        {"id": "rel_gla",    "kind": "relative",   "enabled": True, "field": "bldg_area_finished_sqft",
         "params": {"type": "pct", "value": 20.0}, "peel": {}},
        {"id": "yr",         "kind": "absolute",   "enabled": True, "field": "sale_year",
         "params": {"op": ">=", "value": 2024}, "peel": {"behavior": "hold"}},
      ],
      "peel_policy": {"enabled": True, "min_comps": 10, "min_comps_per_param": 2},
      "adj_fields": [...],                  # None/missing -> every modeled feature
      "adjustment_overrides": {"garage_sqft": 75.0, "pool:1": 25000.0},
      "ranking": {"method": "smallest_adj", "weights": {}, "prefer_bracketed": False},
      "expansion": "none",                  # pinned-comp search expansion
      "fwl": {"mi_shrinkage": True},        # see PORTING_NOTES.md before relying on FWL rates
      "grid": {"size": 4, "pool_size": 25, "max_adjustment_pct": None},
    }

Filter ``kind`` values:

* ``spatial``: the comp must share the subject's value on the finest level
  of the dataset's spatial ladder. Peeling climbs the ladder one level at
  a time.
* ``geo_radius``: bounding box of ``radius`` miles around the subject.
* ``relative``: a band around the subject's own value, either ``pct`` (a
  percent), ``units`` (plus or minus a fixed amount) or ``exact``.
* ``absolute``: a fixed predicate, e.g. ``sale_year >= 2024``. Operators:
  ``= != > >= < <= between in not_in contains starts_with is_true is_false
  is_null is_not_null``.

Per-filter ``peel.behavior`` is one of ``hold | relax | drop | relax_and_drop``.
Leave it empty to get the kind's default (see :mod:`opengrids.filters`).
"""
from __future__ import annotations

import copy
from typing import Any, Optional

VALID_FILTER_KINDS = {"geo_radius", "spatial", "relative", "absolute"}
VALID_PEEL_BEHAVIORS = {"relax", "relax_and_drop", "drop", "hold"}
VALID_FILTER_TYPES = {"pct", "units", "exact"}
VALID_EXPANSION_METHODS = {"second_subject", "expanding_radius", "stretching_range", "none"}
VALID_RANKING_METHODS = {"closest_price", "smallest_adj", "weighted_euclidean"}
VALID_ABSOLUTE_OPS = {
    "=", "!=", ">", ">=", "<", "<=", "between", "in", "not_in",
    "contains", "starts_with", "is_true", "is_false", "is_null", "is_not_null",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "filters": [
        {"id": "spatial", "kind": "spatial", "enabled": True, "field": None, "params": {}, "peel": {}},
        {"id": "geo_radius", "kind": "geo_radius", "enabled": True, "field": None, "params": {"radius": 1.0}, "peel": {}},
        {"id": "rel_gla", "kind": "relative", "enabled": True, "field": "bldg_area_finished_sqft",
         "params": {"type": "pct", "value": 20.0}, "peel": {}},
        {"id": "rel_year", "kind": "relative", "enabled": True, "field": "bldg_year_built",
         "params": {"type": "units", "value": 5.0}, "peel": {}},
    ],
    "peel_policy": {"enabled": True, "min_comps": 10, "min_comps_per_param": 2},
    "adjustment_overrides": {},
    "expansion": "none",
    "adj_fields": None,
    "ranking": {"method": "smallest_adj", "weights": {}, "prefer_bracketed": False},
    "fwl": {
        # Shrink each SHAP rate in the FWL linearization by the normalized
        # mutual information between the unmodeled field and that feature.
        # On by default to match comp_adj_app. It can bias the unmodeled
        # field's rate when the two are correlated; see PORTING_NOTES.md.
        "mi_shrinkage": True,
    },
    "grid": {
        # Comps shown on the grid (comp_adj_app: 4).
        "size": 4,
        # Depth of the ranked bench that comp-quality backfill draws from.
        "pool_size": 25,
        # Drop comps whose |net adjustment| exceeds this percent of their
        # (cost-of-sale-adjusted) sale price. None = no cap. Comps whose
        # adjusted price would go negative are always dropped.
        "max_adjustment_pct": None,
    },
}


def _safe_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_filter_entry(item: Any, idx: int) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    kind = item.get("kind")
    if kind not in VALID_FILTER_KINDS:
        return None
    params = item.get("params") if isinstance(item.get("params"), dict) else {}
    entry = {"id": str(item.get("id") or f"flt_{idx}"), "kind": kind,
             "enabled": bool(item.get("enabled", True)), "field": None, "params": {}, "peel": {}}

    if kind == "geo_radius":
        entry["params"] = {"radius": _safe_float(params.get("radius"), 1.0)}
    elif kind == "relative":
        f, t = item.get("field"), params.get("type")
        if not f or t not in VALID_FILTER_TYPES:
            return None
        entry["field"] = f
        entry["params"] = {"type": t, "value": _safe_float(params.get("value"), 0.0)}
    elif kind == "absolute":
        f = item.get("field")
        op = str(params.get("op", "")).strip().lower()
        if not f or op not in VALID_ABSOLUTE_OPS:
            return None
        entry["field"] = f
        p: dict[str, Any] = {"op": op}
        if "value" in params:
            p["value"] = params.get("value")
        if "value2" in params:
            p["value2"] = params.get("value2")
        entry["params"] = p

    peel = item.get("peel") if isinstance(item.get("peel"), dict) else {}
    npeel: dict[str, Any] = {}
    if peel.get("behavior") in VALID_PEEL_BEHAVIORS:
        npeel["behavior"] = peel["behavior"]
    if isinstance(peel.get("max_relax"), dict):
        clean = {k: _safe_float(v, None) for k, v in peel["max_relax"].items()}
        clean = {k: v for k, v in clean.items() if v is not None}
        if clean:
            npeel["max_relax"] = clean
    if isinstance(peel.get("step"), dict):
        clean = {k: _safe_float(v, None) for k, v in peel["step"].items()}
        clean = {k: v for k, v in clean.items() if v is not None}
        if clean:
            npeel["step"] = clean
    entry["peel"] = npeel
    return entry


def normalize_config(cfg: Optional[dict] = None, base: Optional[dict] = None) -> dict:
    """Validate ``cfg`` and fill in any missing values from ``base``
    (default: :data:`DEFAULT_CONFIG`).

    Invalid entries are dropped quietly rather than raising, so a
    partly-bad config still produces a grid. A missing ``adj_fields``
    resolves to every modeled feature when the grid is built.
    """
    base = base if base is not None else DEFAULT_CONFIG
    cfg = cfg or {}
    out = copy.deepcopy(base)
    for k, v in DEFAULT_CONFIG.items():
        out.setdefault(k, copy.deepcopy(v))

    filters = cfg.get("filters")
    if isinstance(filters, list):
        out["filters"] = [e for e in (_normalize_filter_entry(it, i) for i, it in enumerate(filters)) if e]

    policy = cfg.get("peel_policy")
    if isinstance(policy, dict):
        bp = dict(out.get("peel_policy") or {})
        if "enabled" in policy:
            bp["enabled"] = bool(policy.get("enabled"))
        for k in ("min_comps", "min_comps_per_param"):
            v = _safe_float(policy.get(k), None)
            if v is not None and v >= 0:
                bp[k] = int(v)
        out["peel_policy"] = bp

    if cfg.get("expansion") in VALID_EXPANSION_METHODS:
        out["expansion"] = cfg["expansion"]

    if "adj_fields" in cfg:
        af = cfg.get("adj_fields")
        out["adj_fields"] = None if af is None else [f for f in af if isinstance(f, str) and f]

    ranking = cfg.get("ranking") if isinstance(cfg.get("ranking"), dict) else {}
    out["ranking"] = dict(out.get("ranking") or {})
    if ranking.get("method") in VALID_RANKING_METHODS:
        out["ranking"]["method"] = ranking["method"]
    if "prefer_bracketed" in ranking:
        out["ranking"]["prefer_bracketed"] = bool(ranking.get("prefer_bracketed"))
    if isinstance(ranking.get("weights"), dict):
        out["ranking"]["weights"] = {
            f: _safe_float(w, 1.0) for f, w in ranking["weights"].items() if isinstance(f, str) and f
        }

    if isinstance(cfg.get("adjustment_overrides"), dict):
        out["adjustment_overrides"] = {
            f: _safe_float(r, 0.0) for f, r in cfg["adjustment_overrides"].items() if isinstance(f, str) and f
        }

    fwl = cfg.get("fwl") if isinstance(cfg.get("fwl"), dict) else {}
    out["fwl"] = dict(out.get("fwl") or {})
    if "mi_shrinkage" in fwl:
        out["fwl"]["mi_shrinkage"] = bool(fwl["mi_shrinkage"])

    grid = cfg.get("grid") if isinstance(cfg.get("grid"), dict) else {}
    out["grid"] = dict(out.get("grid") or {})
    for k in ("size", "pool_size"):
        v = _safe_float(grid.get(k), None)
        if v is not None and v >= 1:
            out["grid"][k] = int(v)
    if "max_adjustment_pct" in grid:
        v = _safe_float(grid.get("max_adjustment_pct"), None)
        out["grid"]["max_adjustment_pct"] = v if v and v > 0 else None

    # Drop app-specific bookkeeping keys if a comp_adj_app config was pasted in.
    out.pop("_schema", None)
    out.pop("_default_version", None)
    return out
