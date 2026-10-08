"""Parity check: opengrids vs. comp_adj_app on the same database.

    python tests/parity/run_parity.py --app ../comp_adj_app --n 200

Exits non-zero if any grid differs. Each scenario runs both engines over
the same random sample of subjects.
"""
from __future__ import annotations

import argparse
import warnings
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from opengrids import CompGridEngine  # noqa: E402
from opengrids.compat.comp_adj_app import load  # noqa: E402
from tests.parity.app_reference import AppReference  # noqa: E402
from tests.parity.compare import compare  # noqa: E402

FWL_FIELDS = ["bedrooms", "bathrooms", "bldg_year_built", "has_swimming_pool"]

SCENARIOS = [
    ("office default", {}, {}),
    ("closest_price", {"ranking": {"method": "closest_price", "weights": {}, "prefer_bracketed": False}}, {}),
    ("weighted_euclidean", {"ranking": {"method": "weighted_euclidean", "prefer_bracketed": False,
                                        "weights": {"bldg_area_finished_sqft": 1.0, "land_area_sqft": 0.05,
                                                    "bldg_effective_year_built": 50}}}, {}),
    ("prefer_bracketed", {"ranking": {"method": "smallest_adj", "weights": {}, "prefer_bracketed": True}}, {}),
    ("max_adjustment_pct=10", {}, {"max_adj": 10.0}),
    ("peel disabled", {"peel_policy": {"enabled": False, "min_comps": 12, "min_comps_per_param": 2}}, {}),
    ("tight filters + absolute", {"filters": [
        {"id": "spatial", "kind": "spatial", "enabled": True, "field": None, "params": {}, "peel": {}},
        {"id": "geo_radius", "kind": "geo_radius", "enabled": True, "field": None, "params": {"radius": 0.5}, "peel": {}},
        {"id": "yr", "kind": "absolute", "enabled": True, "field": "sale_year", "params": {"op": ">=", "value": "2025"},
         "peel": {"behavior": "drop"}},
        {"id": "pool", "kind": "relative", "enabled": True, "field": "has_swimming_pool",
         "params": {"type": "exact", "value": 0}, "peel": {}},
        {"id": "rel_gla", "kind": "relative", "enabled": True, "field": "bldg_area_finished_sqft",
         "params": {"type": "pct", "value": 10.0}, "peel": {}},
    ]}, {}),
    ("session overrides", {"adjustment_overrides": {"land_area_sqft": 2.5, "outdoor_living_sqft": 20}}, {}),
    ("FWL fields (params->sales)", {"adj_fields": None}, {"fwl": True}),
    ("pinned comp / none", {"expansion": "none"}, {"pin": True}),
    ("pinned comp / expanding_radius", {"expansion": "expanding_radius"}, {"pin": True}),
    ("pinned comp / stretching_range", {"expansion": "stretching_range"}, {"pin": True}),
    ("pinned comp / second_subject", {"expansion": "second_subject"}, {"pin": True}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="path to the comp_adj_app checkout")
    ap.add_argument("--prefix", default="sf")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default=None, help="substring filter on scenario names")
    ap.add_argument("--show", type=int, default=3, help="print up to N diffs per scenario")
    args = ap.parse_args()
    logging.getLogger("app").setLevel(logging.ERROR)
    warnings.simplefilter("ignore", FutureWarning)

    t = time.time()
    ds, office_cfg = load(str(Path(args.app) / "comp_adj.db"), args.prefix)
    print(f"loaded dataset in {time.time() - t:.1f}s: {len(ds.sales)} sales, {len(ds._universe)} parcels")
    app = AppReference(args.app, args.prefix)
    app_fwl = None

    rng = random.Random(args.seed)
    with_pred = ds._universe[ds._universe["prediction"].notna()]
    subjects = rng.sample(list(with_pred.index), args.n)
    sale_keys = sorted(set(ds.sales[ds.key]))
    multi_sale = set(ds.sales[ds.key][ds.sales[ds.key].duplicated()])

    failures = 0
    for name, overrides, opts in SCENARIOS:
        if args.only and args.only not in name:
            continue
        ref = app
        if opts.get("fwl"):
            app_fwl = app_fwl or AppReference(args.app, args.prefix, fwl_params_from_sales=True)
            ref = app_fwl
            base = ref.config()
            overrides = {"adj_fields": list(base["adj_fields"]) + FWL_FIELDS}
        app_cfg = ref.config(overrides)
        lib_cfg = {k: v for k, v in app_cfg.items() if not k.startswith("_")}
        if opts.get("max_adj"):
            lib_cfg["grid"] = {"max_adjustment_pct": opts["max_adj"]}
        engine = CompGridEngine(ds, lib_cfg)

        results = Counter()
        shown = 0
        t_app = t_lib = 0.0
        for subj in subjects:
            pin = rng.choice(sale_keys) if opts.get("pin") else None
            t0 = time.time()
            try:
                a = ref.grid(subj, app_cfg, user_comp_id=pin, max_adjustment_pct=opts.get("max_adj"))
                a_err = None
            except Exception as e:  # noqa: BLE001
                a, a_err = None, type(e).__name__
            t1 = time.time()
            try:
                g = engine.grid(subj, user_comp=pin)
                l_err = None
            except Exception as e:  # noqa: BLE001
                g, l_err = None, type(e).__name__
            t2 = time.time()
            t_app += t1 - t0
            t_lib += t2 - t1

            if a_err or l_err:
                if a_err and l_err:
                    results["both raised"] += 1
                    continue
                if a_err and not l_err:
                    # e.g. second_subject with an empty union: the app's
                    # rank_candidates returns a list and the next call crashes.
                    results[f"app raised {a_err}, lib ok"] += 1
                    continue
                diffs = [f"app raised {a_err}" if a_err else f"lib raised {l_err}"]
            else:
                diffs = compare(a, g)
            if opts.get("fwl") and diffs and set(a["candidate_ids"] + [subj]) & multi_sale if a else False:
                # The app's FWL query cartesian-joins multi-sale parcels; the
                # library aligns per sale. Expected to differ; counted separately.
                results["differs (multi-sale pool, expected)"] += 1
                continue
            if (diffs and opts.get("fwl") and all(d.startswith("comps differ") for d in diffs)
                    and app_cfg["ranking"]["method"] == "smallest_adj"):
                # Rates, pool and peel all match; only the pick differs. The app's
                # smallest_adj ranking looks up categorical variant rates with
                # raw (int) values against str keys, so it silently scores every
                # categorical field as $0. The library canonicalizes keys.
                results["differs (categorical ranking, expected)"] += 1
                continue
            if diffs:
                results["DIFFERS"] += 1
                if shown < args.show:
                    shown += 1
                    print(f"  [{name}] subject {subj} pin={pin}: " + "; ".join(diffs[:4]))
            else:
                results["identical"] += 1
        bad = results["DIFFERS"]
        failures += bad
        print(f"{'FAIL' if bad else 'ok  '} {name:32s} {dict(results)}  app {t_app / len(subjects) * 1000:.0f} ms/grid,"
              f" lib {t_lib / len(subjects) * 1000:.0f} ms/grid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
