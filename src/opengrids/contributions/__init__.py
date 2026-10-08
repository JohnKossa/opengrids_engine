"""Build the SHAP contribution table from a fitted model.

Most offices don't have SHAP values sitting around. This module produces
them from the model you already have, in exactly the shape
:class:`~opengrids.Dataset` expects: one row per sale, one dollar column per
input feature, plus ``base_value``, ``prediction`` and ``check_delta``.

    from opengrids.contributions import explain
    contrib = explain(model, sales)                    # sklearn model or Pipeline
    ds = Dataset(sales, contrib, universe)

    # or in one step (also fills universe["prediction"]):
    ds = Dataset.from_model(model, sales, universe)

Supported models, and how each one is explained:

``method="linear"`` (exact, no extra packages)
    Anything with ``coef_`` / ``intercept_``: ``LinearRegression``, ``Ridge``,
    ``Lasso``, ``ElasticNet``, ``HuberRegressor``, ``SGDRegressor`` and so on,
    including inside a ``Pipeline`` with a ``ColumnTransformer`` of encoders,
    scalers and imputers. φ_j = Σ_{columns c from j} β_c (z_c − mean z_c).
    This is the exact interventional SHAP value of a linear model.

``method="tree"`` (exact TreeSHAP)
    LightGBM and XGBoost use their built-in TreeSHAP and need no extra
    package. scikit-learn forests and gradient boosting (and CatBoost) use
    ``shap.TreeExplainer`` and need ``pip install shap``.

``method="permutation"`` (model-agnostic, no extra packages)
    Any model with ``predict``, e.g. KNN, SVR, neural nets, stacked
    ensembles, or a pipeline with feature crosses. Slower, and the split
    between features is sampled, but the contributions still add up to the
    prediction exactly.

``statsmodels`` / plain coefficient tables
    :func:`from_statsmodels` for formula-API OLS/WLS results;
    :func:`from_coefficients` for coefficients exported from any stats package.

**Multiplicative (log-price) models.** If the model predicts ``log(price)``
(through ``TransformedTargetRegressor(func=np.log, inverse_func=np.exp)``,
a statsmodels formula like ``np.log(sale_price) ~ ...``, or
``target="log"`` for a model you fit on log price yourself), the log-space
attributions are converted to dollars with the exact Shapley decomposition
in :mod:`opengrids.contributions._link`. The dollar contributions add up to
the dollar prediction.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._groups import UnsupportedModelError, output_groups
from ._link import Target, inverse_for, to_dollars

__all__ = ["explain", "predict_dollars", "from_coefficients", "from_statsmodels", "UnsupportedModelError"]

logger = logging.getLogger("opengrids")


# --------------------------------------------------------------------- helpers
def _unwrap_target(model: Any, target: Union[str, Target]) -> tuple[Any, Target]:
    """Peel a TransformedTargetRegressor; resolve ``target='auto'``."""
    if type(model).__name__ == "TransformedTargetRegressor":
        inner = model.regressor_
        if target == "auto":
            func, inv = getattr(model, "func", None), getattr(model, "inverse_func", None)
            if func is np.log and inv in (np.exp, None):
                target = "log"
            elif func is np.log1p and inv in (np.expm1, None):
                target = "log1p"
            elif func is np.log10:
                target = "log10"
            elif inv is not None:
                target = inv
            elif getattr(model, "transformer_", None) is not None:
                tr = model.transformer_
                target = lambda z, tr=tr: tr.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()  # noqa: E731
            else:
                target = "identity"
        return inner, target
    return model, ("identity" if target == "auto" else target)


def _split_pipeline(model: Any) -> tuple[Any, Any]:
    if type(model).__name__ == "Pipeline":
        return (model[:-1] if len(model.steps) > 1 else None), model.steps[-1][1]
    return None, model


def _features(model: Any, data: pd.DataFrame, features: Optional[Sequence[str]]) -> list[str]:
    if features is not None:
        return list(features)
    names = getattr(model, "feature_names_in_", None)
    if names is None and type(model).__name__ == "TransformedTargetRegressor":
        names = getattr(model.regressor_, "feature_names_in_", None)
    if names is None:
        raise ValueError("can't infer the model's input columns; pass features=[...] "
                         "(fit on a DataFrame and scikit-learn records them for you)")
    return [str(n) for n in names]


def _dense(z: Any) -> np.ndarray:
    if hasattr(z, "toarray"):
        z = z.toarray()
    if isinstance(z, pd.DataFrame):
        z = z.to_numpy()
    return np.asarray(z, dtype=float)


def _kind(est: Any) -> str:
    name = type(est).__name__
    mod = type(est).__module__
    if name.startswith("LGBM") or mod.startswith("lightgbm"):
        return "lightgbm"
    if name.startswith("XGB") or mod.startswith("xgboost"):
        return "xgboost"
    if mod.startswith("catboost"):
        return "catboost"
    if hasattr(est, "coef_") and hasattr(est, "intercept_"):
        return "linear"
    if hasattr(est, "estimators_") or hasattr(est, "tree_") or name.startswith("HistGradientBoosting") \
            or hasattr(est, "_predictors"):
        return "sklearn_tree"
    return "other"


def predict_dollars(model: Any, X: pd.DataFrame, target: Union[str, Target] = "auto") -> np.ndarray:
    """``model.predict`` in dollars. Applies the inverse target transform
    only when the model predicts in link space (``target`` given explicitly
    for a model you fit on log price yourself)."""
    pred = np.asarray(model.predict(X), dtype=float).ravel()
    if type(model).__name__ == "TransformedTargetRegressor" or target in ("auto", "identity", None):
        return pred
    g = inverse_for(target)
    return g(pred) if g else pred


def _assemble(data: pd.DataFrame, features: list[str], base: np.ndarray, phi: np.ndarray,
              prediction: np.ndarray, key: str, sale_key: str) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for c in (key, sale_key):
        if c in data.columns:
            out[c] = data[c].values
    for j, f in enumerate(features):
        out[f] = phi[:, j]
    out["base_value"] = base
    out["contribution_sum"] = phi.sum(axis=1)
    out["prediction"] = prediction
    out["check_delta"] = prediction - (base + phi.sum(axis=1))
    rel = np.nanmax(np.abs(out["check_delta"]) / np.maximum(np.abs(prediction), 1.0)) if len(out) else 0.0
    if rel > 1e-4:
        logger.warning("contributions miss the prediction by up to %.4f%%; see the check_delta column", rel * 100)
    return out.reset_index(drop=True)


def _group_sum(phi_t: np.ndarray, groups: list[str], features: list[str]) -> np.ndarray:
    idx = {f: i for i, f in enumerate(features)}
    out = np.zeros((phi_t.shape[0], len(features)))
    for c, g in enumerate(groups):
        if g not in idx:
            raise UnsupportedModelError(f"transformed column {c} maps to {g!r}, which is not one of the features")
        out[:, idx[g]] += phi_t[:, c]
    return out


# ------------------------------------------------------------------ explainers
def _linear(pre: Any, est: Any, X: pd.DataFrame, bg: pd.DataFrame, features: list[str]):
    Z = _dense(pre.transform(X)) if pre is not None else _dense(X[features])
    Zb = _dense(pre.transform(bg)) if pre is not None else _dense(bg[features])
    coef = np.asarray(est.coef_, dtype=float).ravel()
    if coef.size != Z.shape[1]:
        raise UnsupportedModelError("multi-output linear models are not supported")
    intercept = float(np.ravel(est.intercept_)[0]) if np.ndim(est.intercept_) else float(est.intercept_)
    mean = np.nanmean(Zb, axis=0)
    phi_t = coef * (Z - mean)
    base = np.full(len(X), intercept + float(coef @ mean))
    groups = output_groups(pre, features) if pre is not None else features
    return base, _group_sum(phi_t, groups, features)


def _tree(pre: Any, est: Any, X: pd.DataFrame, bg: pd.DataFrame, features: list[str], kind: str,
          interventional: bool):
    if pre is not None:
        Zt = pre.transform(X)
        try:
            names = [str(n) for n in pre.get_feature_names_out(features)]
        except Exception:  # noqa: BLE001
            names = None
        Z = pd.DataFrame(_dense(Zt), columns=names) if names else _dense(Zt)
        groups = output_groups(pre, features)
    else:
        Z = X[features]
        groups = features

    if kind == "lightgbm":
        raw = np.asarray(est.predict(Z, pred_contrib=True), dtype=float)
        phi_t, base = raw[:, :-1], raw[:, -1]
    elif kind == "xgboost":
        import xgboost as xgb

        booster = est.get_booster() if hasattr(est, "get_booster") else est
        dm = xgb.DMatrix(Z, enable_categorical=True)
        raw = np.asarray(booster.predict(dm, pred_contribs=True), dtype=float)
        phi_t, base = raw[:, :-1], raw[:, -1]
    else:
        try:
            import shap
        except ImportError as e:  # pragma: no cover
            raise ImportError("explaining scikit-learn/CatBoost tree models needs the shap package: "
                              "pip install shap   (or pip install opengrids-engine[shap])") from e
        if interventional:
            bgz = pre.transform(bg) if pre is not None else bg[features]
            bgz = _dense(bgz) if pre is not None else bgz
            ex = shap.TreeExplainer(est, data=shap.sample(bgz, min(len(bg), 200), random_state=0),
                                    feature_perturbation="interventional")
        else:
            ex = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
        phi_t = np.asarray(ex.shap_values(Z, check_additivity=False), dtype=float)
        base = np.full(len(X), float(np.ravel(ex.expected_value)[0]))
    return base, _group_sum(phi_t, groups, features)


def _permutation(model: Any, X: pd.DataFrame, bg: pd.DataFrame, features: list[str], target: Target,
                 background_size: int, seed: int, n_permutations: int, chunk_rows: int = 64):
    """Interventional Shapley values by permutation sampling (model-agnostic).

    For each sampled feature order, start every row at the background sample
    and switch features to the row's own values one at a time. The change in
    mean prediction at each switch is that feature's marginal contribution.
    Each permutation telescopes to ``f(x) - mean f(background)``, so the
    result is exactly efficient. Orders are drawn in antithetic pairs (a
    permutation and its reverse) to reduce variance. Works on raw DataFrames,
    so text and categorical columns are fine.
    """
    g = None if type(model).__name__ == "TransformedTargetRegressor" else inverse_for(target)

    def f(df: pd.DataFrame) -> np.ndarray:
        p = np.asarray(model.predict(df), dtype=float).ravel()
        return g(p) if g else p

    rng = np.random.default_rng(seed)
    B = bg[features].sample(min(len(bg), background_size), random_state=seed).reset_index(drop=True)
    K, n = len(B), len(features)
    base_val = float(f(B).mean())
    half = max(1, n_permutations // 2)
    orders = []
    for _ in range(half):
        o = rng.permutation(n)
        orders += [o, o[::-1]]
    Xf = X[features].reset_index(drop=True)
    phi = np.zeros((len(Xf), n))
    for start in range(0, len(Xf), chunk_rows):
        rows = Xf.iloc[start:start + chunk_rows]
        R = len(rows)
        tiled_bg = pd.concat([B] * R, ignore_index=True)
        row_idx = np.repeat(np.arange(R), K)
        for order in orders:
            z = tiled_bg.copy()
            prev = f(z).reshape(R, K).mean(axis=1)
            for j in order:
                col = features[j]
                z[col] = rows[col].to_numpy()[row_idx]
                cur = f(z).reshape(R, K).mean(axis=1)
                phi[start:start + R, j] += cur - prev
                prev = cur
    phi /= len(orders)
    return np.full(len(Xf), base_val), phi


# ---------------------------------------------------------------------- public
def explain(model: Any, data: pd.DataFrame, features: Optional[Sequence[str]] = None, *,
            background: Optional[pd.DataFrame] = None, method: str = "auto",
            target: Union[str, Target] = "auto", key: str = "key", sale_key: str = "key_sale",
            interventional: bool = False, background_size: int = 50, n_permutations: int = 10,
            seed: int = 0) -> pd.DataFrame:
    """Contribution table for ``data`` (usually your sales).

    Args:
        model: A *fitted* model, Pipeline or TransformedTargetRegressor.
        data: Rows to explain. ``key`` / ``key_sale`` columns are carried through.
        features: Model input columns (default: ``model.feature_names_in_``).
        background: The reference population the contributions are measured
            against. ``base_value`` is the model's average over it. Use your
            universe or training set. Default: ``data``. Pass the *same*
            background whenever you explain different row sets, so they
            share one baseline.
        method: ``auto`` | ``linear`` | ``tree`` | ``permutation``.
        target: ``auto`` (detect TransformedTargetRegressor), ``identity``, or
            ``log`` / ``log1p`` / ``log10`` / an inverse callable for a model
            whose ``predict`` returns transformed prices.
        interventional: Trees only. Use interventional TreeSHAP against
            ``background`` instead of the faster path-dependent default
            (sklearn/CatBoost via shap only; LightGBM/XGBoost always use
            path-dependent).
        background_size, n_permutations, seed: Permutation method only. Cost is
            about ``rows × n_permutations × (features + 1) × background_size``
            model evaluations. Raise both for more stable splits.
    """
    feats = _features(model, data, features)
    missing = [f for f in feats if f not in data.columns]
    if missing:
        raise ValueError(f"data is missing model input column(s): {missing}")
    bg = background if background is not None else data
    inner, tgt = _unwrap_target(model, target)
    pre, est = _split_pipeline(inner)
    kind = _kind(est)
    if method == "auto":
        method = {"linear": "linear", "lightgbm": "tree", "xgboost": "tree", "catboost": "tree",
                  "sklearn_tree": "tree"}.get(kind, "permutation")
        logger.info("explaining %s with method=%s", type(est).__name__, method)

    X = data
    if method == "linear":
        if kind != "linear":
            raise UnsupportedModelError(f"{type(est).__name__} is not a linear model")
        base, phi = _linear(pre, est, X, bg, feats)
    elif method == "tree":
        if kind not in ("lightgbm", "xgboost", "catboost", "sklearn_tree"):
            raise UnsupportedModelError(f"{type(est).__name__} is not a supported tree model")
        base, phi = _tree(pre, est, X, bg, feats, kind, interventional)
    elif method == "permutation":
        base, phi = _permutation(model, X, bg, feats, tgt, background_size, seed, n_permutations)
        pred = predict_dollars(model, X[feats], target if target != "auto" else "auto")
        return _assemble(data, feats, base, phi, pred, key, sale_key)
    else:
        raise ValueError(f"unknown method {method!r}")

    base_d, phi_d = to_dollars(base, phi, tgt, seed=seed)
    pred = predict_dollars(model, X[feats], target if target != "auto" else "auto")
    return _assemble(data, feats, base_d, phi_d, pred, key, sale_key)


def from_coefficients(coefficients: Mapping[str, float], design: pd.DataFrame, *, intercept: float = 0.0,
                      data: Optional[pd.DataFrame] = None, groups: Optional[Mapping[str, str]] = None,
                      target: Target = "identity", background: Optional[pd.DataFrame] = None,
                      key: str = "key", sale_key: str = "key_sale") -> pd.DataFrame:
    """Contributions for a linear model given as a coefficient table.

    Use this for MRA output from SAS/SPSS/R/Excel or a CAMA system.

    Args:
        coefficients: ``{term: β}`` (exclude the intercept).
        design: One row per sale, one column per term, as the model saw it
            (e.g. ``ln_sfla``, ``nbhd_12`` dummies, ``age_sq``).
        intercept: The model constant.
        data: Optional frame to copy ``key`` / ``key_sale`` from (default: ``design``).
        groups: ``{term: input feature}``, e.g. ``{"ln_sfla": "sfla",
            "nbhd_12": "nbhd", "age": "age", "age_sq": "age"}``. Terms not
            listed keep their own name. Feature names must match your sales
            columns so the grid can compare subject and comp values.
        target: ``"log"`` (or ``log1p`` / ``log10``) for a multiplicative
            model fitted on log price.
        background: Design rows defining the baseline (default: ``design``).
    """
    terms = [t for t in coefficients if t in design.columns]
    lost = [t for t in coefficients if t not in design.columns]
    if lost:
        raise ValueError(f"design is missing term column(s): {lost}")
    groups = dict(groups or {})
    feats = list(dict.fromkeys(groups.get(t, t) for t in terms))
    Z = _dense(design[terms])
    Zb = _dense((background if background is not None else design)[terms])
    beta = np.array([coefficients[t] for t in terms], dtype=float)
    mean = np.nanmean(Zb, axis=0)
    phi = _group_sum(beta * (Z - mean), [groups.get(t, t) for t in terms], feats)
    base = np.full(len(design), intercept + float(beta @ mean))
    link_pred = intercept + Z @ beta
    g = inverse_for(target)
    pred = g(link_pred) if g else link_pred
    base_d, phi_d = to_dollars(base, phi, target)
    return _assemble(data if data is not None else design, feats, base_d, phi_d, pred, key, sale_key)


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def from_statsmodels(results: Any, data: pd.DataFrame, *, groups: Optional[Mapping[str, str]] = None,
                     target: Union[str, Target] = "auto", background: Optional[pd.DataFrame] = None,
                     key: str = "key", sale_key: str = "key_sale") -> pd.DataFrame:
    """Contributions for a statsmodels formula-API linear model (OLS, WLS, GLS).

    Each formula term is attributed to the data column it reads:
    ``C(nbhd)`` → ``nbhd``, ``np.log(sfla)`` → ``sfla``. An interaction
    such as ``sfla:quality`` reads two columns and needs an explicit
    ``groups={"sfla:quality": "sfla"}``. ``target="auto"`` detects a
    logged left-hand side, e.g. ``np.log(sale_price) ~ ...``.
    """
    mdata = results.model.data
    spec = getattr(mdata, "design_info", None) or getattr(mdata, "model_spec", None)
    if spec is None:
        raise UnsupportedModelError("results were not fitted with the formula API; use from_coefficients()")
    params = results.params
    intercept = float(params.get("Intercept", 0.0))
    cols = set(map(str, data.columns))

    if hasattr(spec, "term_name_slices"):            # patsy DesignInfo
        from patsy import NAAction, dmatrix

        design = dmatrix(spec, data, return_type="dataframe", NA_action=NAAction(NA_types=[]))
        bdesign = (dmatrix(spec, background, return_type="dataframe", NA_action=NAAction(NA_types=[]))
                   if background is not None else None)
        term_cols = {t: list(design.columns[sl]) for t, sl in spec.term_name_slices.items()}
        term_codes = {t: [f.name() for f in spec.terms[spec.term_names.index(t)].factors] for t in term_cols}
    else:                                              # formulaic ModelSpec
        design = spec.get_model_matrix(data, na_action="ignore")
        design = pd.DataFrame(np.asarray(design), columns=list(design.columns), index=data.index)
        bdesign = None
        if background is not None:
            bm = spec.get_model_matrix(background, na_action="ignore")
            bdesign = pd.DataFrame(np.asarray(bm), columns=list(bm.columns))
        term_cols = {str(t): [design.columns[i] for i in idx] for t, idx in spec.term_indices.items()}
        term_codes = {str(t): [str(f) for f in t.factors] for t in spec.term_indices}

    term_groups: dict[str, str] = {}
    for term, tcols in term_cols.items():
        if term in ("Intercept", "1"):
            continue
        if groups and term in groups:
            owner = groups[term]
        else:
            idents = {tok for code in term_codes[term] for tok in _IDENT.findall(code)}
            hits = sorted(idents & cols)
            if len(hits) != 1:
                raise UnsupportedModelError(
                    f"formula term {term!r} reads {hits or 'no data columns'}; "
                    f"pass groups={{{term!r}: '<feature>'}} to say which feature owns it")
            owner = hits[0]
        for col in tcols:
            term_groups[col] = owner
    coefs = {c: float(params[c]) for c in term_groups}
    if target == "auto":
        lhs = str(results.model.endog_names).replace(" ", "")
        target = ("log1p" if "log1p(" in lhs else "log10" if "log10(" in lhs else "log" if "log(" in lhs
                  else "identity")
    return from_coefficients(coefs, design, intercept=intercept, data=data, groups=term_groups,
                             target=target, background=bdesign, key=key, sale_key=sale_key)
