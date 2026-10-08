"""Adjusting for a field the model never saw (Frisch-Waugh-Lovell).

    python examples/add_unmodeled_field.py

Sometimes a characteristic matters to buyers, or to a hearing, but isn't in
the model: bedrooms, a view, a recent remodel. Add it to ``adj_fields``. Any
adjustment field that isn't one of the model's features gets its rate from
the Frisch-Waugh-Lovell theorem:

  1. Build the model's linearized opinion of each sale in the candidate pool.
  2. Remove that opinion from the sale prices, and from the new field.
  3. Regress what's left of the price on what's left of the field.

The result is the field's dollar value *on top of* what the model already
explains. It is fitted on sale prices, so it needs real variation in the
field across the pool. A field that is nearly constant among the comps
gives an unstable rate.

In the synthetic market, bedrooms add $7,500 each but are left out of the
model, so you can compare the estimate with the truth.
"""
import pandas as pd

from opengrids import CompGridEngine, FieldMeta
from opengrids.demo import TRUE_EFFECTS, make_demo

pd.set_option("display.width", 160)
dataset, universe = make_demo()
subject = universe["key"].iloc[42]

# Optional: give it a label and an expected direction (enables the sign clamp).
dataset.fields["bedrooms"] = FieldMeta("Bedrooms", "improvement", impact="positive")

engine = CompGridEngine(dataset)
cfg = {"adj_fields": dataset.modeled_features + ["bedrooms"]}

# The data check confirms which fields will go through FWL (and flags nulls in them).
print(engine.check(cfg), "\n")

grid = engine.grid(subject, config=cfg)
print("FWL fields:", grid.fwl_fields)
print(grid.rates().to_string(), "\n")
print(grid.to_frame().loc[["Bedrooms", "Bedrooms adj", "total_adj", "adjusted_price"]].round(0).to_string())

# --- How good is the estimate? -----------------------------------------------------
# The synthetic market knows the answer. FWL is fitted on sales, so a bigger pool
# steadies it. Bedrooms are also correlated with living area, which exposes a
# known issue: the default MI-shrinkage step takes part of living area's value
# out of the model's opinion, and bedrooms soak it up. Turning it off removes
# that bias. See "Known limitations" in the README.
print(f"\ntrue effect: ${TRUE_EFFECTS['bedrooms']:,.0f} per bedroom")
pools = {"local pool (default filters)": {},
         "whole county (no filters)": {"filters": [], "peel_policy": {"enabled": False}}}
for label, pool_cfg in pools.items():
    for mi in (True, False):
        g = engine.grid(subject, config={**cfg, **pool_cfg, "fwl": {"mi_shrinkage": mi}})
        print(f"{label:30s} {len(g.candidate_ids):5d} sales, mi_shrinkage={str(mi):5s} -> "
              f"${float(g.unit_adjs['bedrooms']):,.0f} per bedroom")

# Text categories work too: mark them categorical (FieldMeta(is_categorical=True))
# and each variant gets its own FWL rate; an adjustment is
# rate[subject variant] - rate[comp variant].
