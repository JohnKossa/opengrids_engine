import itertools
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from opengrids import CompGridEngine, Dataset
from opengrids.contributions import UnsupportedModelError, explain, from_coefficients, from_statsmodels
from opengrids.contributions._link import _exp_shapley, to_dollars

NUM = ["bldg_area_finished_sqft", "land_area_sqft", "bldg_year_built", "garage_sqft", "building_quality_score",
       "bedrooms"]
CAT = ["neighborhood"]


def _pre():
    return make_column_transformer((OneHotEncoder(handle_unknown="ignore"), CAT), (StandardScaler(), NUM))


@pytest.fixture(scope="module")
def sales(ds):
    return ds.sales


@pytest.fixture(scope="module")
def ridge(sales):
    return make_pipeline(_pre(), Ridge(1.0)).fit(sales[NUM + CAT], sales["sale_price_time_adj"])


def _additive(c):
    assert np.allclose(c["base_value"] + c["contribution_sum"], c["prediction"], rtol=1e-9, atol=1e-6)
    assert np.abs(c["check_delta"]).max() < 1e-6 * c["prediction"].abs().max()


# ---------------------------------------------------------------- dollarization
@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_exp_shapley_matches_brute_force(n):
    rng = np.random.default_rng(n)
    a, b = rng.normal(0, 0.7, (4, n)), rng.normal(12, 0.2, 4)
    fast = _exp_shapley(b, a)
    for r in range(4):
        for j in range(n):
            others = [i for i in range(n) if i != j]
            want = 0.0
            for s in range(n):
                w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                for S in itertools.combinations(others, s):
                    want += w * (math.exp(b[r] + a[r, list(S) + [j]].sum()) - math.exp(b[r] + a[r, list(S)].sum()))
            assert fast[r, j] == pytest.approx(want, rel=1e-10)


def test_permutation_dollarization_is_efficient():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 0.5, (5, 6)), rng.normal(3, 0.1, 5)
    base_d, phi_d = to_dollars(b, a, lambda z: z ** 3, n_permutations=8)
    assert np.allclose(base_d + phi_d.sum(1), (b + a.sum(1)) ** 3)


# ---------------------------------------------------------------------- linear
def test_linear_pipeline_is_exact_and_grouped(ridge, sales):
    c = explain(ridge, sales)
    _additive(c)
    assert set(NUM + CAT) <= set(c.columns) and "key_sale" in c.columns
    # one-hot columns collapse back into one 'neighborhood' column
    assert not any(col.startswith("neighborhood_") for col in c.columns)


def test_grid_rates_reproduce_mra_coefficients(ridge, sales, ds, universe):
    """For a linear model, the engine's local SHAP slopes are exactly the coefficients."""
    dsm = Dataset.from_model(ridge, sales, universe, spatial_ladder=ds.spatial_ladder)
    g = CompGridEngine(dsm, {"adj_fields": NUM}).grid(universe["key"].iloc[42])
    names = list(ridge[0].get_feature_names_out())
    sc = ridge[0].named_transformers_["standardscaler"]
    for i, f in enumerate(NUM):
        beta = ridge[-1].coef_[names.index(f"standardscaler__{f}")] / sc.scale_[i]
        assert g.raw_unit_adjs[f] == pytest.approx(beta, rel=1e-8)


def test_from_model_fills_prediction_and_marks_categoricals(ridge, sales, universe, ds):
    dsm = Dataset.from_model(ridge, sales, universe.drop(columns=["prediction"]), spatial_ladder=ds.spatial_ladder)
    assert dsm.is_categorical("neighborhood")
    s = dsm.subject(universe["key"].iloc[0])
    assert s["prediction"] == pytest.approx(ridge.predict(universe[NUM + CAT].iloc[[0]])[0])


def test_log_target_regressor_in_dollars(sales):
    m = TransformedTargetRegressor(make_pipeline(_pre(), LinearRegression()), func=np.log, inverse_func=np.exp)
    m.fit(sales[NUM + CAT], sales["sale_price_time_adj"])
    c = explain(m, sales)
    _additive(c)
    # base is the geometric-mean (log-mean) parcel's value
    lin = m.regressor_
    logpred = lin.predict(sales[NUM + CAT])
    assert c["base_value"].iloc[0] == pytest.approx(np.exp(logpred.mean()), rel=1e-9)


def test_manual_log_target(sales):
    m = make_pipeline(_pre(), LinearRegression()).fit(sales[NUM + CAT], np.log(sales["sale_price_time_adj"]))
    c = explain(m, sales, target="log")
    _additive(c)
    assert c["prediction"].iloc[0] == pytest.approx(np.exp(m.predict(sales[NUM + CAT].iloc[[0]])[0]))


def test_mixing_transformer_is_refused(sales):
    m = make_pipeline(make_column_transformer((PolynomialFeatures(2), NUM[:3])), LinearRegression())
    m.fit(sales[NUM[:3]], sales["sale_price_time_adj"])
    with pytest.raises(UnsupportedModelError):
        explain(m, sales, method="linear")


# ------------------------------------------------------- coefficient tables
def test_from_coefficients_groups_terms(sales):
    design = pd.DataFrame({"ln_gla": np.log(sales["bldg_area_finished_sqft"]),
                           "age": 2025 - sales["bldg_year_built"],
                           "age_sq": (2025 - sales["bldg_year_built"]) ** 2})
    c = from_coefficients({"ln_gla": 0.6, "age": -0.004, "age_sq": 0.00002}, design, intercept=8.5,
                          data=sales, groups={"ln_gla": "bldg_area_finished_sqft", "age": "bldg_year_built",
                                              "age_sq": "bldg_year_built"}, target="log")
    _additive(c)
    assert [x for x in c.columns if x in NUM] == ["bldg_area_finished_sqft", "bldg_year_built"]


def test_from_statsmodels_log_formula(sales):
    smf = pytest.importorskip("statsmodels.formula.api")
    res = smf.ols("np.log(sale_price_time_adj) ~ np.log(bldg_area_finished_sqft) + land_area_sqft"
                  " + I(2025 - bldg_year_built) + garage_sqft + C(neighborhood)", data=sales).fit()
    c = from_statsmodels(res, sales)
    _additive(c)
    assert c["prediction"].iloc[0] == pytest.approx(np.exp(res.fittedvalues.iloc[0]))
    assert {"bldg_area_finished_sqft", "bldg_year_built", "neighborhood"} <= set(c.columns)


# ------------------------------------------------------------------ trees
def test_lightgbm_native(sales):
    lgb = pytest.importorskip("lightgbm")
    X = sales[NUM + CAT].assign(neighborhood=sales["neighborhood"].astype("category"))
    m = lgb.LGBMRegressor(n_estimators=100, verbose=-1).fit(X, sales["sale_price_time_adj"])
    _additive(explain(m, sales.assign(neighborhood=X["neighborhood"])))


def test_xgboost_log_native(sales):
    xgb = pytest.importorskip("xgboost")
    X = sales[NUM + CAT].assign(neighborhood=sales["neighborhood"].astype("category"))
    m = TransformedTargetRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=4, enable_categorical=True),
                                   func=np.log, inverse_func=np.exp).fit(X, sales["sale_price_time_adj"])
    c = explain(m, sales.assign(neighborhood=X["neighborhood"]))
    assert np.abs(c["check_delta"] / c["prediction"]).max() < 1e-4   # xgboost works in float32


def test_sklearn_forest_via_shap(sales):
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestRegressor

    m = make_pipeline(make_column_transformer((OneHotEncoder(handle_unknown="ignore"), CAT), remainder="passthrough"),
                      RandomForestRegressor(30, min_samples_leaf=5, random_state=0))
    m.fit(sales[NUM + CAT], sales["sale_price_time_adj"])
    _additive(explain(m, sales))


# ------------------------------------------------------------- permutation
def test_permutation_is_efficient_for_any_model(sales):
    m = make_pipeline(_pre(), KNeighborsRegressor(10)).fit(sales[NUM + CAT], sales["sale_price_time_adj"])
    c = explain(m, sales.iloc[:15], method="permutation", background=sales, background_size=20, n_permutations=4)
    _additive(c)


def test_permutation_is_exact_for_additive_models(ridge, sales):
    small = sales.iloc[:40]
    exact = explain(ridge, small, background=small)
    perm = explain(ridge, small, background=small, method="permutation", background_size=len(small), n_permutations=2)
    assert np.allclose(exact[NUM + CAT].to_numpy(), perm[NUM + CAT].to_numpy(), rtol=1e-8, atol=1e-6)
