# opengrids-engine

Comparable-sales grids whose adjustments agree with your AVM.

Give it your sales, your model's per-sale SHAP contributions and a search
config. For each subject it:

1. **finds comps**, relaxing ("peeling back") filters from the bottom of your
   priority list until the local fit has enough support;
2. **derives per-unit adjustments** that are a local linearization of *your
   model*, not a separate regression that can contradict the roll value;
3. **ranks, quality-checks and selects** comps, then returns the full
   adjustment grid with diagnostics.

It works with any model (GBM, random forest, neural net, ensemble, or a
linear model) because it never calls the model itself. It reads only the
model's SHAP decomposition.

> Extracted from the comp adjustment application built at the Lee County
> (FL) Property Appraiser's office. On that office's production data it
> reproduces the original application's grids exactly (see
> [Parity](#parity-with-comp_adj_app)).

Licensed under the [Apache License 2.0](LICENSE).

---

## Install

```bash
pip install -e .            # numpy, pandas, scikit-learn
pip install -e .[parquet]   # + pyarrow, for Parquet input
pip install -e .[models]    # + shap, statsmodels, lightgbm, xgboost (explaining your model)
pip install -e .[dev]       # + pytest and the above
```

Python 3.10+.

## Quickstart

```python
from opengrids import CompGridEngine, Dataset, FieldMeta, SpatialLevel

ds = Dataset.from_files(
    sales="sales.parquet",                 # one row per sale
    contributions="shap_sales.parquet",    # one row per sale: SHAP $ per feature
    universe="universe.parquet",           # optional: only used to look up subjects by key
    universe_columns=["prediction", "latitude", "longitude", "bldg_area_finished_sqft", ...],
    fields={
        "bldg_area_finished_sqft": FieldMeta("Living Area", "improvement", impact="positive"),
        "garage_sqft":             FieldMeta("Garage", impact="positive", override_rate=75.0),
        "has_pool":                FieldMeta("Pool", is_categorical=True),
    },
    spatial_ladder=[SpatialLevel("neighborhood"), SpatialLevel("market_area")],
)

engine = CompGridEngine(ds, config={"grid": {"size": 4}})

grid = engine.grid("10168801")   # or engine.grid(subject_row_dict)
grid.to_frame()                  # subject + comps, value and $-adjustment rows per field
grid.to_dict()                   # JSON-safe; to_dict(diagnostics=True) adds peel trace, fidelity, clamps
grid.metrics()                   # spread, bracketing, avg adjustment %, peel depth, surrogate fidelity
grid.rates()                     # per field: applied rate, fitted rate, source (SHAP / FWL / override / clamped)
```

Run `python examples/quickstart.py` for a working example on a bundled
synthetic dataset (`opengrids.demo.make_demo()`). [`examples/`](examples/README.md)
also covers the common tasks:

| task | how | example |
|---|---|---|
| choose which fields are adjusted | `config={"adj_fields": [...]}` on the engine, or per call in `engine.grid(..., config=...)` | `change_adjustment_fields.py` |
| adjust for a field the model doesn't have | add it to `adj_fields`; it's fitted with FWL automatically | `add_unmodeled_field.py` |
| override a rate for one search | `config={"adjustment_overrides": {"garage_sqft": 75, "has_pool:1": 30000, "has_pool:0": 0}}` | `assessor_overrides.py` |
| override a rate office-wide | `dataset.fields["garage_sqft"] = FieldMeta(..., override_rate=75)` | `assessor_overrides.py` |
| see where every rate came from | `grid.rates()`: applied rate, what the data alone says, and its source | all of the above |

Other entry points:

```python
engine.grid(subject, user_comp="10295521")        # pin a hand-picked comp
engine.grid(subject, config={"ranking": {...}})   # per-call config, merged over the engine's
engine.candidates(subject)                        # just the peeled pool + trace
engine.adjustments(subject, comp_keys)            # rates over a pool you choose
for subj, g in engine.grids(keys, errors="return"): ...   # batch
```

## Check your data first

```python
report = engine.check()        # or dataset.check(config)
print(report)
report.raise_if_errors()       # for scripts and CI
report.to_frame()              # one row per issue
```

```
opengrids data check: 1 error(s), 2 warning(s), 1 note(s)
  ERROR [garage_sqft] 312 of 26,099 (1.2%) sales have no value. Any subject whose candidate pool
         includes one of them will fail to compute this rate (the grid raises GridConfigError).
         e.g. 10338345, 10337723, ...
         fix: fill the values, drop those sales, or remove the field from adj_fields
  WARN  [sfla] filter 'flt_4' (relative) uses 'sfla', which is not a sales column, so the filter is skipped.
  ...
```

What it looks for:

* **Model contributions:** sales with no contribution row, orphan or
  duplicate rows, missing or non-numeric contribution values, and whether
  `base_value + Σ contributions = prediction`.
* **Adjustment-field types:** numbers stored as text, text not marked
  categorical, a field typed differently in sales and universe, dates, and
  categoricals with so many variants that most rates rest on one or two sales.
* **Nulls in adjustment fields:** in sales (a null in a continuous modeled
  field makes the rate fit fail for any pool containing that sale) and in the
  universe (that field's adjustment becomes $0 for the subject).
* **Schema and config:** required columns, and filters, ranking weights or
  overrides that name a column that doesn't exist. Without the check these
  are skipped without a word.

Errors mean some grids will fail or be wrong. Warnings mean grids run but
something degrades. Notes are informational, e.g. which fields go through
FWL.

## No SHAP values? Start from your model

You don't need to know anything about SHAP. Pass the model you already have:

```python
from opengrids import Dataset, CompGridEngine

ds = Dataset.from_model(model, sales, universe)     # explains the sales, scores the universe
engine = CompGridEngine(ds)
```

`opengrids.contributions.explain(model, sales)` does the same thing if you
only want the contribution table. Each row comes out in dollars and passes
an additivity check: `base_value + Σ contributions = prediction`, with any
miss reported in `check_delta`.

| your model | how it's explained | extra install |
|---|---|---|
| Linear / MRA in scikit-learn (`LinearRegression`, `Ridge`, `Lasso`, …), including `Pipeline` + `ColumnTransformer` with one-hot encoders, scalers and imputers | exact: β·(x − mean), one-hot columns summed back into their feature | none |
| Log-price (multiplicative) models: `TransformedTargetRegressor(func=np.log, …)`, or `target="log"` for a model you fit on log price yourself | log-space attributions converted to dollars with the exact Shapley decomposition (closed form, no sampling) | none |
| statsmodels formula OLS/WLS, e.g. `np.log(price) ~ np.log(sfla) + C(nbhd)` | `from_statsmodels(results, sales)`: each term attributed to the column it reads | `statsmodels` |
| Coefficients exported from SAS/SPSS/R/Excel/CAMA | `from_coefficients({term: β}, design, intercept=…, groups={term: feature}, target="log")` | none |
| LightGBM, XGBoost | exact TreeSHAP, built in | the model's own package |
| scikit-learn random forest / gradient boosting, CatBoost | exact TreeSHAP via `shap.TreeExplainer` | `shap` |
| Anything else with `.predict` (KNN, SVR, neural nets, stacked ensembles) | `method="permutation"`: sampled interventional Shapley; still adds up exactly | none (slower) |

Things worth knowing:

* **Linear models reproduce MRA.** For a linear model the grid's per-unit
  rates *are* the regression coefficients, exactly. The grid becomes a
  transparent restatement of an MRA you already trust, which makes a good
  first test on your own data.
* **Multiplicative models give subject-specific rates.** In a log model a
  square foot is worth more on a better house, so the rate moves with the
  subject. That's the model's actual opinion, and the grid follows it.
* **Background.** `base_value` is measured against a reference population
  (`background=`, default: your universe). Use the same one for every call.
  For log models the base is the geometric-mean parcel's value. The grid
  never uses the base directly, only how each contribution changes with
  its feature.
* **Interactions.** Pipelines that build feature crosses
  (`PolynomialFeatures`, PCA) can't be attributed column by column and are
  refused. Use `method="permutation"`.

See [`examples/from_model.py`](examples/from_model.py).

## Data contract

All three inputs are pandas DataFrames, or CSV/Parquet paths via
`Dataset.from_files`. Column roles can be renamed with keyword arguments
(`key=`, `sale_key=`, `lat_field=`, `sale_price_field=`,
`sale_price_time_adj_field=`, `sale_date_field=`, `prediction_field=`,
`base_value_field=`). The defaults follow
[OpenAVMKit](https://github.com/larsiusprime/openavmkit) naming.

| table | grain | required columns | notes |
|---|---|---|---|
| `sales` | one row per sale | `key`, `key_sale`, `sale_price`, `sale_price_time_adj`, `sale_date`, `latitude`, `longitude` | plus every field used in filters, ranking or adjustments |
| `contributions` | one row per sale | `key`, `key_sale`, one numeric column **per model feature** (SHAP $), `base_value`, `prediction` | rows align to `sales` by `key_sale` |
| `universe` *(optional)* | one row per parcel | `key`, `prediction`, the adjustment fields | used only to look up a subject by key |

Notes:

* Keys are compared as strings, so leading zeros are preserved.
* SHAP values must be in the target's units (dollars) and additive:
  `base_value + Σ contributions = prediction`.
* A field listed in `adj_fields` that is *not* a contributions column is a
  **FWL field**: something you want to adjust for that the model never saw.
* Only the sales-side tables are scanned per grid. On a county with 26k sales
  a grid takes about 80 ms, and the 277k-parcel universe never has to be in
  memory; stream subjects in batches if you like.

## How the adjustments work

Formal names are given so you can find the literature.

**SHAP linearization (modeled fields).** For each modeled feature *j*, the
SHAP contributions φ_j are regressed on the feature values x_j over the
candidate pool (subject plus peeled candidates, every sale of each). The
slope is the dollar rate per unit. This is a *local linear surrogate* of the
model fitted to exact Shapley attributions (compare LIME's local surrogate),
so the grid adjusts the way the model values the subject's neighborhood of
feature space. Categorical features get a mean φ per variant, and an
adjustment is `rate[subject variant] − rate[comp variant]`.

**Frisch–Waugh–Lovell (unmodeled fields).** A linearized prediction
ẑ = b₀ + Σ rate_j·x_j is partialled out of both the sale price and each
unmodeled field. Residual price is then regressed on the residual fields
(the FWL theorem: the same coefficient a joint regression would give). The
result is each field's marginal value *holding the model's opinion fixed*.
Before that step, each SHAP rate in ẑ is shrunk by the normalized mutual
information `I(x_fwl; x_j) / H(x_fwl)` between the unmodeled field and the
modeled one. This is on by default to match comp_adj_app. Turn it off with
`config={"fwl": {"mi_shrinkage": False}}`. It biases the unmodeled field's
rate when that field is correlated with a modeled one; see
[PORTING_NOTES.md](PORTING_NOTES.md).

**Sign clamp.** A continuous field with `impact="positive"` whose fitted
rate came out negative (or the reverse) is set to $0 and reported in
`grid.clamped_fields`. Overrides are exempt.

**Overrides.** Precedence: per-call `adjustment_overrides`, then
`FieldMeta.override_rate` / `variant_overrides`, then the computed rate.
Categorical variants use `"field:variant"` keys.

**Cost of sale.** `CostOfSale(pct=15)` moves sale prices onto the
assessment basis. If the model was trained on full market value
(`model_trained_on_cost_of_sale=False`), the SHAP contributions, intercept
and prediction are scaled by the same factor before any regression runs,
so rates and baselines stay on one basis.

**Peel-back.** The filter list is ordered by priority, and peeling runs
bottom-up. Each filter has a behavior (`hold`, `relax`, `drop`,
`relax_and_drop`); defaults depend on the filter kind. Spatial filters
climb a ladder of membership fields (finest to coarsest). A *clean* step
(the finer level nests inside the coarser one) replaces the active level;
an unclean step unions with it, so relaxing can never drop a comp that
already qualified. The target is
`max(min_comps, min_comps_per_param × parameters)`, where a categorical
field counts one parameter per variant present. Every iteration is recorded
in `grid.peel_audit`.

**Selection.** Candidates are ranked (`smallest_adj`, `closest_price` or
`weighted_euclidean`), then scored in rank order. A comp is skipped and
backfilled from the bench if its adjusted price is negative or its |net
adjustment| exceeds `grid.max_adjustment_pct`. `prefer_bracketed` reserves
one slot above the subject value and one below.

**Local fidelity.** `grid.surrogate_fidelity` is an out-of-sample (k-fold
by parcel) MAPE/MdAPE of a linear fit of the model's prediction on the
modeled features across the pool. A high value means the model is
far from linear here, and no set of linear adjustments can represent it
faithfully.

## Configuration

A plain, JSON-serializable dict. The full schema is in
[`src/opengrids/config.py`](src/opengrids/config.py), and
`opengrids.DEFAULT_CONFIG` shows every key. It's a superset of
comp_adj_app's saved search-config format, so an exported app config can
be used directly.

## Coming from comp_adj_app

```python
from opengrids.compat.comp_adj_app import load
dataset, config = load("comp_adj.db", prefix="sf")    # opened read-only
engine = CompGridEngine(dataset, config)
```

## Tests

```bash
pytest                      # unit tests on the synthetic dataset (model tests skip if lightgbm/xgboost/shap/statsmodels are absent)
```

## Not included (by design)

The web UI, sessions, PDF/batch rendering, photos, static maps, and the
"low detail" Inferior/Similar/Superior view. Those belong in an
application built on top of this library.

## License

Apache License 2.0. See [LICENSE](LICENSE). You may use, modify and
redistribute this library, including in commercial and government
products, as long as you keep the license and any notices with it.

## Special Thanks

* Lee County (FL) Property Appraiser's office for the original comp_adj_app
  and the synthetic dataset.