"""Bring your own model: from a fitted scikit-learn model to comp grids.

    pip install -e .[models]
    python examples/from_model.py

The same steps work with your own sales and universe tables. Fit any model
on your sales (an MRA-style linear model, a log-price model, LightGBM,
XGBoost, a random forest), then hand it to Dataset.from_model.
"""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from opengrids import CompGridEngine, Dataset, FieldMeta, SpatialLevel
from opengrids.demo import make_demo

pd.set_option("display.width", 160)

# Stand-ins for your own tables. Only the raw columns are used here, not the
# demo's built-in SHAP table.
demo, universe = make_demo()
sales = demo.sales
universe = universe.drop(columns=["prediction"])     # pretend the model hasn't scored it yet

numeric = ["bldg_area_finished_sqft", "land_area_sqft", "bldg_year_built", "garage_sqft",
           "building_quality_score", "bedrooms"]
categorical = ["neighborhood"]

# 1. A multiplicative (log-price) MRA, the most common form in assessment offices.
model = TransformedTargetRegressor(
    make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), categorical),
                                (StandardScaler(), numeric)),
        LinearRegression(),
    ),
    func=np.log, inverse_func=np.exp,
)
model.fit(sales[numeric + categorical], sales["sale_price_time_adj"])

# 2. Explain it and build the dataset. Contributions come out in dollars and
#    add up to the dollar prediction; neighborhood is marked categorical
#    automatically.
ds = Dataset.from_model(
    model, sales, universe,
    fields={"bldg_area_finished_sqft": FieldMeta("Living Area", impact="positive"),
            "garage_sqft": FieldMeta("Garage", impact="positive")},
    spatial_ladder=[SpatialLevel("neighborhood", clean_to_next=True), SpatialLevel("district")],
)
print("largest additivity error ($):", float(ds.contributions["check_delta"].abs().max()))

# 3. Grids, as usual.
engine = CompGridEngine(ds, {"adj_fields": numeric + categorical})
grid = engine.grid(universe["key"].iloc[42])
print(grid.to_frame().round(0).to_string())

# In a multiplicative model the dollar value of a square foot depends on the
# rest of the house, so the rate differs from subject to subject:
for k in universe["key"].iloc[[42, 7, 300]]:
    print(k, "living area $/sqft:", round(float(engine.grid(k).unit_adjs["bldg_area_finished_sqft"]), 2))
