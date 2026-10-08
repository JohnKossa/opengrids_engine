"""Choosing which fields the grid adjusts for.

    python examples/change_adjustment_fields.py

By default the grid adjusts for every feature the model uses. Usually you
want a shorter, defensible list: the characteristics a taxpayer would
recognize on a grid. A field you leave out gets no adjustment line, so
differences in it are accepted as they are. Fewer fields also means fewer
parameters to estimate, so peel-back needs fewer comps to reach its target.
"""
import pandas as pd

from opengrids import CompGridEngine
from opengrids.demo import make_demo

pd.set_option("display.width", 160)
dataset, universe = make_demo()
subject = universe["key"].iloc[42]

# --- 1. The default: every modeled feature -------------------------------------
engine = CompGridEngine(dataset)
print("default adj_fields:", engine.config["adj_fields"])

# --- 2. Set the list for every grid this engine builds -------------------------
engine = CompGridEngine(dataset, config={
    "adj_fields": ["bldg_area_finished_sqft", "land_area_sqft", "bldg_year_built", "garage_sqft", "has_pool"],
})
grid = engine.grid(subject)
print("\nfive-field grid:")
print(grid.to_frame().round(0).to_string())

# --- 3. Change it for one grid only --------------------------------------------
# config= on .grid() is merged over the engine's config for that one call.
one_off = engine.grid(subject, config={"adj_fields": ["bldg_area_finished_sqft", "garage_sqft"]})
print("\ntwo-field grid, net adjustments:", [round(c["total_adj"]) for c in one_off.comps])

# ...or derive a second engine and keep both around.
quality_engine = engine.with_config({"adj_fields": engine.config["adj_fields"] + ["building_quality_score"]})
print("with quality added:", quality_engine.config["adj_fields"])

# --- 4. Fewer fields -> fewer comps needed -------------------------------------
# The peel target is max(min_comps, min_comps_per_param x parameters), where a
# categorical field counts one parameter per variant in the pool.
policy = {"peel_policy": {"min_comps": 0, "min_comps_per_param": 4}}
for fields in (["bldg_area_finished_sqft"], None):
    cfg = {**policy, "adj_fields": fields}
    _, audit = engine.candidates(subject, config=cfg)
    n = len(engine.with_config(cfg).config["adj_fields"])
    print(f"{n} field(s): {audit[-1]['param_count']} parameters -> needs {audit[-1]['needed']} comps; "
          f"relaxed {len(audit) - 1} filter step(s) to reach a pool of {audit[-1]['comps']}")

# --- 5. Typos get caught before they cost you a grid ----------------------------
print()
print(engine.check({"adj_fields": ["bldg_area_finished_sqft", "garage_sq_ft"]}))
