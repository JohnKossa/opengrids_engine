"""Assessor overrides: replacing a computed rate with one you choose.

    python examples/assessor_overrides.py

Three places a rate can come from, highest priority first:

  1. this search:    config["adjustment_overrides"]     (one grid, one engine)
  2. office-wide:    FieldMeta(override_rate=...)        (every grid on this dataset)
  3. computed:       the model's SHAP slope, or FWL for unmodeled fields

Overrides are exempt from the sign clamp. grid.rates() shows the applied
rate, what the data alone says, and which of these three it came from, which
is the paper trail you want at a hearing.
"""
from dataclasses import replace

import pandas as pd

from opengrids import CompGridEngine
from opengrids.demo import make_demo

pd.set_option("display.width", 160)
dataset, universe = make_demo()
subject = universe["key"].iloc[42]
engine = CompGridEngine(dataset)

# --- 1. One search: garage at $75/sqft ------------------------------------------
grid = engine.grid(subject, config={"adjustment_overrides": {"garage_sqft": 75.0}})
print(grid.rates().loc[["garage_sqft"]].to_string())

# --- 2. Office-wide: set it on the field's metadata -------------------------------
dataset.fields["garage_sqft"] = replace(dataset.meta("garage_sqft"), override_rate=60.0)
print("\noffice-wide:", engine.grid(subject).rates().loc["garage_sqft", ["rate", "source"]].to_dict())
# ...and a per-search override still wins over the office-wide one:
g = engine.grid(subject, config={"adjustment_overrides": {"garage_sqft": 75.0}})
print("search wins:", g.rates().loc["garage_sqft", ["rate", "source"]].to_dict())
dataset.fields["garage_sqft"] = replace(dataset.meta("garage_sqft"), override_rate=None)   # undo

# --- 3. Categorical fields: override variants, not the difference -----------------
# A categorical field's rates are levels per variant, and an adjustment is
# rate[subject] - rate[comp]. To say "a pool is worth $30,000", set BOTH sides.
# Overriding only "has_pool:1" would leave the model's level for "0" in place.
pool = {"has_pool:1": 30_000.0, "has_pool:0": 0.0}
g = engine.grid(subject, config={"adjustment_overrides": pool})
print("\n", g.rates().loc[["has_pool"]].to_string())
for c in g.comps:
    if c["has_pool"] != g.subject["has_pool"]:
        print(f"comp {c['key']} (pool={c['has_pool']}) pool adjustment: ${c['adjustments']['has_pool']:,.0f}")
# Office-wide equivalent: FieldMeta(..., is_categorical=True, variant_overrides={1: 30000, 0: 0})

# --- 4. Unmodeled (FWL) fields can be overridden too -----------------------------
# The FWL fit is skipped for that field; the other fields are unaffected.
g = engine.grid(subject, config={"adj_fields": dataset.modeled_features + ["bedrooms"],
                                 "adjustment_overrides": {"bedrooms": 5_000.0}})
print("\n", g.rates().loc[["bedrooms"]].to_string())

# --- 5. Overrides vs. the sign clamp ------------------------------------------------
# If a field's fitted rate has the "wrong" sign for its expected impact, it is
# clamped to $0 and listed in grid.clamped_fields. An override replaces the
# clamp. That's the usual fix when the local fit disagrees with market evidence.
dataset.fields["land_area_sqft"] = replace(dataset.meta("land_area_sqft"), impact="negative")  # wrong on purpose
g = engine.grid(subject)
print("\nclamped:", [c["name"] for c in g.clamped_fields], "|", g.rates().loc["land_area_sqft", "source"])
g = engine.grid(subject, config={"adjustment_overrides": {"land_area_sqft": 1.25}})
print("override:", g.rates().loc["land_area_sqft", ["rate", "source"]].to_dict())
dataset.fields["land_area_sqft"] = replace(dataset.meta("land_area_sqft"), impact="positive")   # undo
