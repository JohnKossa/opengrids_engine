"""Summary statistics for a finished grid (spread, bracketing, peel depth, fidelity)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .grid import CompGrid


def quality_summary(grid: "CompGrid") -> dict:
    audit = grid.peel_audit or []
    fid = grid.surrogate_fidelity or {}
    return {
        "surrogate_mape": fid.get("mape"),
        "surrogate_mdape": fid.get("mdape"),
        "surrogate_n": fid.get("n"),
        "peel_iters": len(audit),
        "peel_final_comps": audit[-1].get("comps") if audit else None,
        "peel_relaxed": len(audit) > 1,
        "peel_spatial_dropped": any(
            isinstance(a.get("change"), dict) and a["change"].get("kind") == "spatial"
            and a["change"].get("action") == "drop" for a in audit),
        "peel_exhausted": bool(audit) and audit[-1].get("stop") == "exhausted",
    }


def grid_metrics(grid: "CompGrid") -> dict:
    """Grid-level diagnostics.

    * ``price_spread_pct``: (max - min) / median of adjusted comp prices
    * ``avg_adj_pct``: mean |net adjustment| / comp sale price
    * ``comp_match_pct``: (mean adjusted price - subject value) / subject value
    * ``is_bracketed``: at least one adjusted price above the subject value
      and at least one below it
    * plus peel-back and surrogate-fidelity summaries
    """
    comps = grid.comps
    if not comps:
        return {"comp_count": 0, "price_spread_pct": 0, "avg_adj_pct": 0, "avg_distance": 0,
                "comp_match_pct": 0, "comps_above": 0, "comps_below": 0, "is_bracketed": False,
                **quality_summary(grid)}

    final = [c["final_price"] for c in comps]
    base = [c.get("sale_price_cost_of_sale_adj") for c in comps]
    adjs = [abs(c["total_adj"]) for c in comps]
    dists = [c.get("distance_miles") for c in comps if c.get("distance_miles") is not None]

    med = np.median(final)
    spread = (max(final) - min(final)) / med if med > 0 else 0
    adj_pcts = [a / p for a, p in zip(adjs, base) if p and p > 0]
    pred = grid.subject_value
    match = (np.mean(final) - pred) / pred if pred and pred > 0 else 0
    above = sum(1 for p in final if pred and pred > 0 and p > pred)
    below = sum(1 for p in final if pred and pred > 0 and p < pred)
    return {
        "comp_count": len(comps),
        "price_spread_pct": float(spread),
        "avg_adj_pct": float(np.mean(adj_pcts)) if adj_pcts else 0,
        "avg_distance": float(np.mean(dists)) if dists else 0,
        "comp_match_pct": float(match),
        "comps_above": above,
        "comps_below": below,
        "is_bracketed": above > 0 and below > 0,
        **quality_summary(grid),
    }
