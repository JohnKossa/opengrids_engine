"""Quickstart: build a comp grid from the bundled synthetic dataset.

    pip install -e .
    python examples/quickstart.py

With your own data, swap make_demo() for Dataset.from_files(...); see README.md.
"""
import pandas as pd

from opengrids import CompGridEngine
from opengrids.demo import make_demo

pd.set_option("display.width", 160)

dataset, universe = make_demo()

engine = CompGridEngine(dataset, config={
    "filters": [
        {"kind": "spatial"},                                   # same neighborhood -> district -> anywhere
        {"kind": "geo_radius", "params": {"radius": 1.0}},
        {"kind": "relative", "field": "bldg_area_finished_sqft", "params": {"type": "pct", "value": 20}},
        {"kind": "absolute", "field": "sale_year", "params": {"op": ">=", "value": 2024},
         "peel": {"behavior": "hold"}},
    ],
    "peel_policy": {"min_comps": 12, "min_comps_per_param": 2},
    "ranking": {"method": "smallest_adj", "prefer_bracketed": True},
    "grid": {"size": 4, "max_adjustment_pct": 25},
})

print(engine.check())   # pre-flight data report

subject_key = universe["key"].iloc[42]
grid = engine.grid(subject_key)

print(grid.to_frame().round(2).to_string())
print()
print("per-unit rates:", {k: (round(float(v), 2) if not isinstance(v, dict) else {vk: round(float(r), 2) for vk, r in v.items()}) for k, v in grid.unit_adjs.items()})
print("metrics:", grid.metrics())
print("peel trace:", [(a["comps"], a["change"].get("action"), a["change"].get("field")) for a in grid.peel_audit])
