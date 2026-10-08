"""Local fidelity: how close to linear the model is near the subject.

The grid's per-feature rates form a local linear surrogate of the model. If
the model is far from linear across the candidate pool, no set of linear
rates can represent it faithfully, and the adjustments will disagree with
the roll value they're supposed to explain. This is the same idea as
LIME's local-fidelity check.

The test: fit a k-fold linear model of the model's own ``prediction`` on the
numeric modeled features across the comps, then score it on held-out comps.
Folds are by property key, so a property with several sales never appears
in both train and test. Scoring out of sample avoids a saturation trap: an
in-sample fit looks best exactly when the sample is thinnest.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .dataset import Dataset


def surrogate_fidelity(dataset: Dataset, comp_ids: list[str], modeled_fields: list[str], k: int = 5) -> Optional[dict]:
    """``{'mape', 'mdape', 'n'}`` as fractions, or ``None`` when there are
    too few comps or numeric features to fit a meaningful model."""
    ids = list(dict.fromkeys(str(c) for c in comp_ids))
    numeric = [f for f in (modeled_fields or []) if not dataset.is_categorical(f)]
    if len(ids) < 5 or not numeric:
        return None
    sk, key, pf = dataset.sale_key, dataset.key, dataset.prediction_field
    sales, contrib = dataset.sales, dataset.contributions
    if sk not in sales.columns or sk not in contrib.columns or pf not in contrib.columns:
        return None
    feats = [f for f in numeric if f in sales.columns]
    if not feats:
        return None

    c = contrib[contrib[key].isin(set(ids))].sort_values(key, kind="mergesort")
    c = pd.DataFrame({key: c[key].values, "prediction": c[pf].values, "_sk": c[sk].astype(str).values})
    s = sales[[sk, *feats]].copy()
    s["_sk"] = s[sk].astype(str)
    s = s.drop_duplicates(subset=["_sk"], keep="first").drop(columns=[sk])
    df = c.merge(s, on="_sk", how="inner").drop(columns=["_sk"])
    if df.empty:
        return None

    for col in ["prediction", *feats]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction", *feats])
    df = df[df["prediction"] != 0]
    if len(df) < 5:
        return None

    keys_unique = sorted(df[key].unique())
    kk = min(k, len(keys_unique))
    if kk < 2:
        return None
    fold = df[key].map({kk_: i % kk for i, kk_ in enumerate(keys_unique)})

    apes = []
    for fi in range(kk):
        tr = (fold != fi).values
        te = (fold == fi).values
        if tr.sum() <= len(feats) + 1 or te.sum() == 0:
            continue
        model = LinearRegression().fit(df.loc[tr, feats].values, df.loc[tr, "prediction"].values)
        pred = model.predict(df.loc[te, feats].values)
        for p, a in zip(pred, df.loc[te, "prediction"].values):
            if a != 0 and np.isfinite(p):
                apes.append(abs(p - a) / abs(a))
    if not apes:
        return None
    arr = np.array(apes, dtype=float)
    return {"mape": float(np.mean(arr)), "mdape": float(np.median(arr)), "n": int(arr.size)}
