"""Field-by-field comparison of an app grid payload and an opengrids CompGrid."""
from __future__ import annotations

import math
from typing import Any

REL, ABS = 1e-6, 1e-4


def _num_eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return math.isclose(fa, fb, rel_tol=REL, abs_tol=ABS)


def _rate_eq(a: Any, b: Any) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        keys = set(a) | set(b)
        return all(_num_eq(a.get(k, 0), b.get(k, 0)) for k in keys)
    return _num_eq(a, b)


def compare(app: dict | None, lib) -> list[str]:
    """List of human-readable differences (empty = identical)."""
    diffs: list[str] = []
    if app is None:
        return ["app returned no grid"]
    if app["candidate_ids"] != lib.candidate_ids:
        diffs.append(f"candidate pool differs: app {len(app['candidate_ids'])} vs lib {len(lib.candidate_ids)}")
    a_audit = [(x["comps"], x["needed"], x.get("stop"), x["change"].get("action"), x["change"].get("id"))
               for x in app["peel_audit"]]
    l_audit = [(x["comps"], x["needed"], x.get("stop"), x["change"].get("action"), x["change"].get("id"))
               for x in lib.peel_audit]
    if a_audit != l_audit:
        diffs.append(f"peel audit differs: {a_audit} vs {l_audit}")
    for f in app["adj_fields"]:
        if not _rate_eq(app["unit_adjs"].get(f), lib.unit_adjs.get(f)):
            diffs.append(f"effective rate {f}: {app['unit_adjs'].get(f)} vs {lib.unit_adjs.get(f)}")
    if sorted(c["name"] for c in app["clamped_fields"]) != sorted(c["name"] for c in lib.clamped_fields):
        diffs.append("clamped fields differ")
    a_keys = [str(c["key"]) for c in app["comps"]]
    l_keys = [str(c["key"]) for c in lib.comps]
    if a_keys != l_keys:
        diffs.append(f"comps differ: {a_keys} vs {l_keys}")
    else:
        for ac, lc in zip(app["comps"], lib.comps):
            for f in app["adj_fields"]:
                if not _num_eq(ac["adjustments"].get(f), lc["adjustments"].get(f)):
                    diffs.append(f"comp {ac['key']} adj {f}: {ac['adjustments'].get(f)} vs {lc['adjustments'].get(f)}")
            for k in ("total_adj", "final_price", "distance_miles", "sale_price_cost_of_sale_adj", "time_adj_pct"):
                if not _num_eq(ac.get(k), lc.get(k)):
                    diffs.append(f"comp {ac['key']} {k}: {ac.get(k)} vs {lc.get(k)}")
    if app["excluded_comp_count"] != lib.excluded_comp_count:
        diffs.append(f"excluded count {app['excluded_comp_count']} vs {lib.excluded_comp_count}")
    af, lf = app.get("surrogate_fidelity"), lib.surrogate_fidelity
    if (af is None) != (lf is None) or (af and not all(_num_eq(af[k], lf[k]) for k in ("mape", "mdape", "n"))):
        diffs.append(f"fidelity {af} vs {lf}")
    return diffs
