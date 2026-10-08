"""Unit adjustments: SHAP linearization, MI shrinkage, FWL, and the sign clamp.

Two sources of per-unit rates, both fitted over the candidate pool
(subject + peeled candidates, using every sale of those properties):

**Modeled fields** (present in the SHAP contributions). The model's own
attribution φ_j for feature j is regressed on the feature's value x_j
across the pool, and the slope is the dollar rate per unit. This is a
local, model-coherent linearization of the model around the subject (the
same idea as LIME's local surrogate, but fitted to exact Shapley
attributions instead of perturbation samples). For a categorical field,
the "rate" is the mean φ_j per variant, and an adjustment is
``rate[subject_variant] - rate[comp_variant]``.

**Unmodeled fields** (requested adjustments the model never saw). These go
through Frisch-Waugh-Lovell. The linearized model prediction ẑ = b0 +
Σ rate_j·x_j is partialled out of both the sale price and each unmodeled
field. The residuals of price are then regressed on the residuals of the
fields, and the coefficients are each field's marginal value *holding the
model's opinion fixed*. Before that, each SHAP rate is shrunk by
``1 - I(x_fwl; x_j) / H(x_fwl)`` (the max over unmodeled fields), so a
modeled feature that mostly re-encodes an unmodeled one doesn't
double-count its value in ẑ.

Finally, the **sign clamp**: a continuous field whose expected impact is
``positive`` (``negative``) but whose fitted rate came out negative
(positive) is set to 0 and reported in ``clamped_fields``, unless an
override rate is set for it.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression

from ._util import canon
from .dataset import Dataset

logger = logging.getLogger("opengrids")


def entropy_from_samples(x: np.ndarray, bins: int = 30) -> float:
    """Plug-in Shannon entropy (nats) from a 30-bin histogram, used to
    normalize mutual information for a continuous field."""
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log(p)))


def field_variance(vals: pd.Series) -> float:
    """Coefficient of variation around the median (diagnostic only)."""
    vals = vals.dropna()
    if len(vals) == 0:
        return 0.0
    median = np.median(vals)
    if median == 0:
        return 0.0
    return float(np.std(vals) / median)


class AdjustmentEngine:
    """Computes unit adjustments for one candidate pool.

    ``overrides`` is the merged override map. Keys are either ``field``
    (continuous rate) or ``field:variant`` (one categorical variant).
    Per-search overrides take priority over dataset-level
    :class:`FieldMeta` overrides; build the map with
    :func:`merged_overrides`.
    """

    def __init__(self, dataset: Dataset, overrides: Optional[dict] = None, mi_shrinkage: bool = True):
        self.ds = dataset
        self.overrides = dict(overrides or {})
        self.mi_shrinkage = mi_shrinkage
        cos = dataset.cost_of_sale
        self.model_shrink_factor = cos.model_shrink_factor if cos else 1.0

    # ------------------------------------------------------------- data access
    def _subset(self, frame: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
        """Rows for ``ids`` ordered by key, then by original row order.
        Ordering is deterministic so results are reproducible run to run."""
        k = self.ds.key
        sub = frame[frame[k].isin(set(map(str, ids)))]
        return sub.sort_values(k, kind="mergesort")

    def _aligned(self, ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(sales rows, contribution rows) for ``ids``, aligned one-to-one by sale."""
        ids = list(ids)
        s = self._subset(self.ds.sales, ids)
        c = self._subset(self.ds.contributions, ids)
        sk = self.ds.sale_key
        on = sk if (sk in s.columns and sk in c.columns) else self.ds.key
        left = pd.DataFrame({"_on": c[on].astype(str).values, "_c": c.index.values})
        right = pd.DataFrame({"_on": s[on].astype(str).values, "_s": s.index.values})
        pairs = left.merge(right, on="_on", how="inner")
        return (self.ds.sales.loc[pairs["_s"].values].reset_index(drop=True),
                self.ds.contributions.loc[pairs["_c"].values].reset_index(drop=True))

    def price_basis(self, sales_rows: pd.DataFrame) -> pd.Series:
        """The sale price the FWL regression treats as the target."""
        cos = self.ds.cost_of_sale
        tadj = sales_rows[self.ds.sale_price_time_adj_field]
        if cos is not None and cos.source == "field" and cos.field:
            return pd.to_numeric(sales_rows[cos.field], errors="coerce")
        if cos is not None:
            return tadj * cos.factor
        return tadj

    # ------------------------------------------------------------------ SHAP
    def shap_unit_adjustments(self, ids: Iterable[str], features: list[str],
                              mi_weights: Optional[dict] = None) -> dict[str, Any]:
        mi_weights = mi_weights or {}
        sales_rows, contrib_rows = self._aligned(ids)
        out: dict[str, Any] = {}
        for feature in features:
            categorical = self.ds.is_categorical(feature)
            if feature in self.overrides and not categorical:
                out[feature] = self.overrides[feature]
                continue
            if feature not in contrib_rows.columns or feature not in sales_rows.columns:
                continue
            vals = sales_rows[feature]
            contribs = contrib_rows[feature]
            if self.model_shrink_factor != 1.0:
                contribs = contribs * self.model_shrink_factor

            if categorical:
                variant_map = {}
                for v in vals.unique():
                    vk = canon(v)
                    if f"{feature}:{vk}" in self.overrides:
                        variant_map[vk] = self.overrides[f"{feature}:{vk}"]
                        continue
                    mask = vals.isna() if pd.isna(v) else (vals == v)
                    avg = contribs[mask].mean()
                    if pd.isna(avg):
                        avg = 0.0
                    if feature in mi_weights:
                        avg *= (1 - mi_weights[feature])
                    variant_map[vk] = avg
                out[feature] = variant_map
                continue

            if field_variance(vals) < 1.0:
                logger.debug("field %s has low variance (cv=%r)", feature, field_variance(vals))
            deltas = vals - vals.median()
            mask = deltas != 0
            if not mask.any():
                out[feature] = 0.0
                continue
            if len(vals[mask]) > 2:
                rate = LinearRegression().fit(vals.values.reshape(-1, 1), contribs).coef_[0]
            else:
                rate = (contribs[mask] / deltas[mask]).median()
            if feature in mi_weights:
                rate *= (1 - mi_weights[feature])
            out[feature] = rate
        return out

    @staticmethod
    def linearized_prediction(df: pd.DataFrame, shap_unit_adjs: dict, intercept: float) -> pd.Series:
        preds = pd.Series(intercept, index=df.index, dtype=float)
        for feat, adj in shap_unit_adjs.items():
            if feat in df.columns:
                if isinstance(adj, dict):
                    preds += df[feat].map(lambda x: adj.get(canon(x), np.nan)).astype(float).fillna(0)
                else:
                    preds += df[feat] * adj
        return preds

    # ------------------------------------------------------------------- FWL
    def fwl_unit_adjustments(self, ids: Iterable[str], shap_features: list[str], fwl_features: list[str],
                             intercept: float) -> dict[str, Any]:
        if not fwl_features:
            return {}
        ids = list(ids)
        intercept = intercept * self.model_shrink_factor

        out: dict[str, Any] = {}
        remaining = []
        for f in fwl_features:
            if self.ds.is_categorical(f):
                remaining.append(f)
            elif f in self.overrides:
                out[f] = self.overrides[f]
            else:
                remaining.append(f)
        if not remaining:
            return out

        # One row per sale: price basis plus feature values, both from the sales table.
        rows = self._subset(self.ds.sales, ids)
        cols = [c for c in dict.fromkeys([*shap_features, *fwl_features]) if c in rows.columns]
        df = rows[[self.ds.key, *cols]].reset_index(drop=True).copy()
        df["sale_price_time_adj"] = self.price_basis(rows).values

        mi_weights = self._mi_weights(df, shap_features, remaining) if self.mi_shrinkage else {}
        shap_adjs = self.shap_unit_adjustments(ids, shap_features, mi_weights=mi_weights)
        df["linear_pred"] = self.linearized_prediction(df, shap_adjs, intercept)

        df_fwl = df.dropna(subset=[*remaining, "linear_pred", "sale_price_time_adj"])
        if df_fwl.empty or len(df_fwl) < 2:
            for f in remaining:
                out[f] = {} if self.ds.is_categorical(f) else 0.0
            return out

        y = df_fwl["sale_price_time_adj"]
        z = df_fwl[["linear_pred"]]
        y_star = y - LinearRegression().fit(z, y).predict(z)

        x_stars: dict[str, np.ndarray] = {}
        dummy_cols: dict[str, dict[str, str]] = {}
        for feat in remaining:
            if self.ds.is_categorical(feat):
                dummies = pd.get_dummies(df_fwl[feat])
                dummy_cols[feat] = {}
                for val in dummies.columns:
                    col = f"{feat}_{canon(val)}"
                    dummy_cols[feat][canon(val)] = col
                    x = dummies[[val]].astype(float)
                    x_stars[col] = x.values.flatten() - LinearRegression().fit(z, x).predict(z).flatten()
            else:
                x = df_fwl[[feat]].apply(pd.to_numeric, errors="coerce").fillna(0)
                x_stars[feat] = x.values.flatten() - LinearRegression().fit(z, x).predict(z).flatten()

        X_star = pd.DataFrame(x_stars)
        coefs = dict(zip(X_star.columns, LinearRegression(fit_intercept=False).fit(X_star, y_star).coef_))

        for feat in remaining:
            if self.ds.is_categorical(feat):
                v_map = {}
                for v in df_fwl[feat].unique():
                    vk = canon(v)
                    if f"{feat}:{vk}" in self.overrides:
                        v_map[vk] = self.overrides[f"{feat}:{vk}"]
                    else:
                        v_map[vk] = coefs.get(dummy_cols[feat].get(vk, ""), 0.0)
                out[feat] = v_map
            else:
                out[feat] = coefs.get(feat, 0.0)
        return out

    def _mi_weights(self, df: pd.DataFrame, shap_features: list[str], fwl_features: list[str]) -> dict[str, float]:
        """Shrinkage weight per SHAP feature: the max over FWL features of
        I(fwl; shap) / H(fwl), capped at 1."""
        weights: dict[str, float] = {}
        if not (shap_features and fwl_features and len(df) >= 4):
            return weights
        for fwl_feat in fwl_features:
            if fwl_feat not in df.columns:
                continue
            eligible = [s for s in shap_features if s in df.columns]
            if not eligible:
                continue
            data = df[[fwl_feat] + eligible].dropna()
            if len(data) < 4:
                continue
            y = data[fwl_feat]
            X = data[eligible].copy()
            discrete = []
            for col in eligible:
                if self.ds.is_categorical(col) or X[col].dtype == object or X[col].dtype == bool:
                    X[col], _ = pd.factorize(X[col])
                    discrete.append(True)
                else:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                    discrete.append(False)
            X = X.fillna(0)
            if self.ds.is_categorical(fwl_feat) or y.dtype == object or y.dtype == bool:
                y_enc, _ = pd.factorize(y)
                mi = mutual_info_classif(X, y_enc, discrete_features=discrete, random_state=0)
                p = pd.Series(y_enc).value_counts(normalize=True)
                hz = float(-np.sum(p * np.log(p)))
            else:
                y = pd.to_numeric(y, errors="coerce")
                valid = y.notna()
                y, Xn = y[valid], X[valid]
                if len(y) < 4:
                    continue
                mi = mutual_info_regression(Xn, y, discrete_features=discrete, random_state=0)
                hz = entropy_from_samples(y.values)
            for shap_f, m in zip(eligible, mi):
                w = min(m / hz, 1.0) if hz > 0 else 0.0
                weights[shap_f] = max(weights.get(shap_f, 0), w)
        return weights


def merged_overrides(dataset: Dataset, search_overrides: Optional[dict] = None) -> dict:
    """Dataset-level overrides from :class:`FieldMeta`, then per-search
    overrides on top (``field`` or ``field:variant`` keys)."""
    out: dict[str, float] = {}
    for name, meta in dataset.fields.items():
        if meta.override_rate is not None and not (isinstance(meta.override_rate, float) and math.isnan(meta.override_rate)):
            out[name] = meta.override_rate
        for variant, rate in (meta.variant_overrides or {}).items():
            out[f"{name}:{canon(variant)}"] = rate
    for k, v in (search_overrides or {}).items():
        if ":" in k:
            field, variant = k.split(":", 1)
            out[f"{field}:{canon(variant)}"] = v
        else:
            out[k] = v
    return out


def compute_adjustments(dataset: Dataset, subject_key: str, subset_ids: list[str], cfg: dict) -> dict:
    """The full adjustment pipeline for one subject and candidate pool.

    Returns a dict with:
      * ``unit_adjs``: raw rates (before the sign clamp)
      * ``effective_unit_adjs``: rates after the sign clamp (what the grid applies)
      * ``clamped_fields``: ``[{name, label, original_rate, impact}]``
      * ``adj_fields``, ``modeled_fields`` (SHAP), ``fwl_fields`` (FWL)
      * ``intercept``: the SHAP base value used for the FWL linearization
      * ``model_rates``: for overridden fields only, the rate the data alone gives
      * ``rate_sources``: ``{field: "model (SHAP)" | "fitted to sales (FWL)" |
        "override (this search)" | "override (field metadata)" | "..., clamped to $0 ..."}``
    """
    adj_fields = list(cfg.get("adj_fields") or [])
    contrib = dataset.contributions
    first = contrib[contrib[dataset.key] == str(subject_key)]
    intercept = 0
    if not first.empty and dataset.base_value_field in contrib.columns:
        bv = first.iloc[0][dataset.base_value_field]
        intercept = 0 if pd.isna(bv) else bv

    modeled = [f for f in adj_fields if f in contrib.columns]
    fwl = [f for f in adj_fields if f not in modeled]
    search_overrides = cfg.get("adjustment_overrides", {}) or {}
    mi = bool((cfg.get("fwl") or {}).get("mi_shrinkage", True))
    engine = AdjustmentEngine(dataset, merged_overrides(dataset, search_overrides), mi_shrinkage=mi)

    shap_adjs = engine.shap_unit_adjustments(subset_ids, modeled)
    fwl_adjs = engine.fwl_unit_adjustments(subset_ids, modeled, fwl, intercept)
    unit_adjs = {**shap_adjs, **fwl_adjs}

    clamped, effective = [], {}
    for f in adj_fields:
        meta = dataset.meta(f)
        rate = unit_adjs.get(f, {} if meta.is_categorical else 0)
        if not meta.is_categorical:
            impact = meta.impact or "mixed"
            if (meta.override_rate is None and f not in search_overrides
                    and ((impact == "positive" and rate < 0) or (impact == "negative" and rate > 0))):
                clamped.append({"name": f, "label": meta.label or f, "original_rate": rate, "impact": impact})
                rate = 0
        effective[f] = rate

    # What the data alone says for overridden fields ("model: $40/sqft, set to $75"),
    # computed without any overrides.
    merged = engine.overrides
    overridden = [f for f in adj_fields if f in merged or any(k.startswith(f"{f}:") for k in merged)]
    model_rates: dict[str, Any] = {}
    if overridden:
        plain = AdjustmentEngine(dataset, {}, mi_shrinkage=mi)
        o_mod = [f for f in overridden if f in modeled]
        if o_mod:
            model_rates.update(plain.shap_unit_adjustments(subset_ids, o_mod))
        if any(f in fwl for f in overridden):
            model_rates.update({f: r for f, r in plain.fwl_unit_adjustments(subset_ids, modeled, fwl, intercept).items()
                                if f in overridden})

    # Where each applied rate came from, in words an appraiser can repeat.
    clamped_names = {c["name"] for c in clamped}
    sources: dict[str, str] = {}
    for f in adj_fields:
        meta = dataset.meta(f)
        fitted = "model (SHAP)" if f in modeled else "fitted to sales (FWL)"
        if meta.is_categorical:
            n_var = sum(1 for k in merged if k.startswith(f"{f}:"))
            sources[f] = f"{fitted}, {n_var} variant override(s)" if n_var else fitted
        elif f in search_overrides:
            sources[f] = "override (this search)"
        elif meta.override_rate is not None:
            sources[f] = "override (field metadata)"
        elif f in clamped_names:
            sources[f] = f"{fitted}, clamped to $0 (sign contradicts expected impact)"
        else:
            sources[f] = fitted

    return {
        "rate_sources": sources,
        "model_rates": model_rates,
        "unit_adjs": unit_adjs,
        "effective_unit_adjs": effective,
        "clamped_fields": clamped,
        "adj_fields": adj_fields,
        "modeled_fields": modeled,
        "fwl_fields": fwl,
        "intercept": intercept,
    }
