"""Turning a config into a filter stack, plus the peel-back relaxation policy.

The candidate pool is the sample the local surrogate (the per-feature
adjustment rates) is fitted on. Filters drawn too tight starve that fit,
leaving several FWL parameters supported by four comps. Peel-back relaxes
the filter stack from the bottom up until the pool reaches::

    needed = max(min_comps, min_comps_per_param * param_count)

``param_count`` is categorical-aware. Each continuous adjustment field costs
one parameter. Each categorical field costs one parameter per distinct
variant present in the pool, so the count rises as relaxation pulls in more
variants. Those are the degrees of freedom the surrogate actually has to
estimate.
"""
from __future__ import annotations

from typing import Optional

from .fields import SpatialLevel

MIN_COMPS = 10
MIN_COMPS_PER_PARAM = 2

#: geo_radius widens by this many miles per step, up to RADIUS_MAX_EXTRA beyond
#: its base, and by default is never dropped: once the spatial filter is gone,
#: it's the only geographic bound left.
RADIUS_STEP = 0.5
RADIUS_MAX_EXTRA = 4.0

#: Safety cap on peel iterations. A normal peel takes a handful.
HARD_ITER_CAP = 40


def _relative_step_and_max(f_type: Optional[str], value) -> tuple[dict, dict]:
    """Default step for a relative band: widen by about 1/4 of the base band
    per step, up to double the base band in total."""
    base = abs(float(value)) if value else 0.0
    if f_type == "pct":
        return {"pct": max(1.0, round(base / 4.0, 2))}, {"pct": base}
    step = max(base / 4.0, 1.0) if base else 1.0
    return {"units": step}, {"units": base}


def resolve_peel(entry: dict, spatial_ladder: list[SpatialLevel]) -> dict:
    """Fill each filter's peel behavior, step and max with the kind's
    defaults, keeping any values the user set."""
    kind = entry.get("kind")
    override = entry.get("peel") or {}
    beh = override.get("behavior")
    params = entry.get("params", {}) or {}

    if kind == "geo_radius":
        return {"behavior": beh or "relax",
                "step": override.get("step") or {"units": RADIUS_STEP},
                "max_relax": override.get("max_relax") or {"units": RADIUS_MAX_EXTRA}}
    if kind == "relative":
        if params.get("type") == "exact":
            return {"behavior": beh or "drop"}
        step, default_max = _relative_step_and_max(params.get("type"), params.get("value"))
        return {"behavior": beh or "relax_and_drop",
                "step": override.get("step") or step,
                "max_relax": override.get("max_relax") or default_max}
    if kind == "spatial":
        return {"behavior": beh or "relax_and_drop",
                "ladder": [{"field": l.field, "clean_to_next": bool(l.clean_to_next)} for l in spatial_ladder]}
    # absolute: held by default; only hold/drop are meaningful.
    return {"behavior": beh or "hold"}


def build_filter_stack(cfg: dict, spatial_ladder: list[SpatialLevel]) -> list[dict]:
    """The enabled filters, in order, with peel defaults resolved."""
    stack = []
    for entry in cfg.get("filters", []) or []:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        kind = entry.get("kind")
        if kind not in ("geo_radius", "spatial", "relative", "absolute"):
            continue
        stack.append({
            "id": entry.get("id") or kind,
            "kind": kind,
            "enabled": True,
            "field": entry.get("field"),
            "params": dict(entry.get("params", {}) or {}),
            "peel": resolve_peel(entry, spatial_ladder),
        })
    return stack


def peel_policy(cfg: Optional[dict]) -> dict:
    """Peel target derived from ``cfg['peel_policy']``. When peeling is
    disabled, the target is zero, so only the base filters apply."""
    policy = (cfg or {}).get("peel_policy") or {}
    if policy.get("enabled", True) is False:
        return {"target": {"min_comps": 0, "min_comps_per_param": 0}}
    return {"target": {
        "min_comps": int(policy.get("min_comps", MIN_COMPS)),
        "min_comps_per_param": int(policy.get("min_comps_per_param", MIN_COMPS_PER_PARAM)),
    }}
