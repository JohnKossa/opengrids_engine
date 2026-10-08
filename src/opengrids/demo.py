"""A synthetic jurisdiction with a known model, for examples and tests.

The "AVM" is an additive model (a GAM): value = base + Σ_j f_j(x_j). For
an additive model with independent features, the interventional SHAP value
is exact and needs no sampling: φ_j(x) = f_j(x_j) - E[f_j(X_j)]. That makes
the demo dependency-free, and it gives the true per-unit effects to test
against. ``bedrooms`` moves sale prices but is left out of the model, so it
can be used to exercise the FWL path.

Every number here is invented. Nothing comes from any real jurisdiction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import Dataset
from .fields import FieldMeta, SpatialLevel

#: True marginal effects baked into the synthetic market.
TRUE_EFFECTS = {
    "garage_sqft": 40.0,          # $/sqft, linear
    "has_pool": 25_000.0,         # categorical bump
    "bedrooms": 7_500.0,          # NOT in the model: recoverable only via FWL
}


def _shape(n: int, rng: np.random.Generator) -> pd.DataFrame:
    district = rng.integers(0, 4, n)
    nbhd_in_district = rng.integers(0, 5, n)
    neighborhood = district * 10 + nbhd_in_district
    lat = 26.5 + district * 0.03 + nbhd_in_district * 0.006 + rng.normal(0, 0.002, n)
    lng = -81.9 + nbhd_in_district * 0.006 + district * 0.01 + rng.normal(0, 0.002, n)
    gla = rng.lognormal(np.log(1800), 0.25, n).round()
    land = (gla * rng.uniform(4, 8, n)).round()
    year = rng.integers(1960, 2025, n)
    garage = np.where(rng.random(n) < 0.85, rng.normal(450, 120, n).clip(150, 900).round(), 0.0)
    pool = (rng.random(n) < 0.4).astype(int)
    quality = rng.integers(2, 6, n)
    bedrooms = np.clip(np.round(gla / 600 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    return pd.DataFrame({
        "key": [f"{i:08d}" for i in range(1, n + 1)],
        "address": [f"{100 + i} Example St" for i in range(n)],
        "latitude": lat, "longitude": lng,
        "district": district.astype(str), "neighborhood": neighborhood.astype(str),
        "bldg_area_finished_sqft": gla, "land_area_sqft": land, "bldg_year_built": year,
        "garage_sqft": garage, "has_pool": pool, "building_quality_score": quality, "bedrooms": bedrooms,
    })


def _components(df: pd.DataFrame) -> pd.DataFrame:
    """f_j(x_j) for each modeled feature (dollars)."""
    nbhd_effect = {str(d * 10 + k): 30_000 * d + 8_000 * k for d in range(4) for k in range(5)}
    return pd.DataFrame({
        "bldg_area_finished_sqft": 160 * df["bldg_area_finished_sqft"] - 0.012 * df["bldg_area_finished_sqft"] ** 2,
        "land_area_sqft": 9_000 * np.log1p(df["land_area_sqft"]),
        "bldg_year_built": 1_400 * (df["bldg_year_built"] - 1960) - 9 * (df["bldg_year_built"] - 1960) ** 2,
        "garage_sqft": TRUE_EFFECTS["garage_sqft"] * df["garage_sqft"],
        "has_pool": TRUE_EFFECTS["has_pool"] * df["has_pool"],
        "building_quality_score": 18_000 * df["building_quality_score"],
        "neighborhood": df["neighborhood"].map(nbhd_effect).astype(float),
    }, index=df.index)


def make_demo(n_parcels: int = 4000, sale_rate: float = 0.35, seed: int = 0) -> tuple[Dataset, pd.DataFrame]:
    """Return ``(dataset, universe)``. The dataset already includes the universe."""
    rng = np.random.default_rng(seed)
    uni = _shape(n_parcels, rng)
    comp = _components(uni)
    centered = comp - comp.mean()          # exact interventional SHAP for an additive model
    base_value = 60_000 + float(comp.sum(axis=1).mean())
    uni["prediction"] = base_value + centered.sum(axis=1)

    sold = uni.sample(frac=sale_rate, random_state=seed)
    resold = sold.sample(frac=0.08, random_state=seed + 1)
    sales = pd.concat([sold.assign(_n=0), resold.assign(_n=1)]).sort_values(["key", "_n"]).reset_index(drop=True)
    days = rng.integers(0, 730, len(sales))
    sales["sale_date"] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d")
    sales["sale_year"] = sales["sale_date"].str[:4].astype(int)
    truth = sales["prediction"] + TRUE_EFFECTS["bedrooms"] * (sales["bedrooms"] - uni["bedrooms"].mean())
    market = truth * rng.lognormal(0, 0.04, len(sales))
    trend = 1 + 0.03 * (730 - days) / 365.0      # prices rose ~3%/yr; time-adjust to valuation date
    sales["sale_price"] = (market / trend).round(-2)
    sales["sale_price_time_adj"] = sales["sale_price"] * trend
    sales["key_sale"] = [f"S{i:07d}" for i in range(1, len(sales) + 1)]
    sales = sales.drop(columns=["_n"])

    contrib = centered.loc[uni.index].copy()
    contrib.insert(0, "key", uni["key"])
    contrib_sales = sales[["key", "key_sale"]].merge(contrib, on="key", how="left")
    contrib_sales["base_value"] = base_value
    feats = list(centered.columns)
    contrib_sales["contribution_sum"] = contrib_sales[feats].sum(axis=1)
    contrib_sales["prediction"] = base_value + contrib_sales["contribution_sum"]

    fields = {
        "bldg_area_finished_sqft": FieldMeta("Living Area (sqft)", "improvement", "positive"),
        "land_area_sqft": FieldMeta("Land Area (sqft)", "land", "positive"),
        "bldg_year_built": FieldMeta("Year Built", "improvement", "positive"),
        "garage_sqft": FieldMeta("Garage (sqft)", "improvement", "positive"),
        "has_pool": FieldMeta("Pool", "improvement", "positive", is_categorical=True, is_boolean=True),
        "building_quality_score": FieldMeta("Quality", "improvement", "positive"),
        "neighborhood": FieldMeta("Neighborhood", "land", is_categorical=True),
        "district": FieldMeta("District", "land", is_categorical=True),
        "bedrooms": FieldMeta("Bedrooms", "improvement", "positive"),
    }
    ds = Dataset(
        sales, contrib_sales, uni,
        fields=fields,
        spatial_ladder=[SpatialLevel("neighborhood", clean_to_next=True), SpatialLevel("district")],
        valuation_date="2026-01-01",
    )
    return ds, uni
