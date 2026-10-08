"""Map the columns a preprocessing pipeline produces back to the input features.

If a pipeline one-hot encodes ``neighborhood`` into 40 columns and scales
``gla``, the model's attributions are over 41 columns. The grid needs one
contribution per *input* feature, so the attributions are summed within each
group. Summing is exact for linear models, because the encoding of one input
contributes additively. For tree models it is the usual convention, and
additivity still holds.

Supported: ``Pipeline``, ``ColumnTransformer`` (including remainder and
drop), and any transformer whose outputs map to its inputs either one to
one (scalers, imputers, ordinal encoders, power and quantile transforms,
element-wise ``FunctionTransformer``) or by name prefix (``OneHotEncoder``,
``SplineTransformer``, one-hot ``KBinsDiscretizer``,
``SimpleImputer(add_indicator=True)``). Transformers that mix inputs, such
as ``PolynomialFeatures`` or ``PCA``, can't be attributed to single inputs
and raise :class:`UnsupportedModelError`. Use the model-agnostic explainer
for those.
"""
from __future__ import annotations

from typing import Any, Sequence


class UnsupportedModelError(ValueError):
    """The model or pipeline can't be explained with the requested method."""


_MIXING = ("PolynomialFeatures", "PCA", "TruncatedSVD", "Nystroem", "RBFSampler", "KernelPCA",
           "FeatureUnion", "SplineInteraction")


def output_groups(transformer: Any, inputs: Sequence[str]) -> list[str]:
    """For each output column of a *fitted* transformer, the input it came from."""
    inputs = list(inputs)
    name = type(transformer).__name__
    if transformer is None or transformer == "passthrough":
        return inputs
    if transformer == "drop":
        return []
    if name in _MIXING:
        raise UnsupportedModelError(
            f"{name} mixes input features, so its outputs can't be attributed to single inputs. "
            "Use explain(..., method='permutation') for this model.")
    if name == "Pipeline":
        names, origins = inputs, inputs
        for _, step in transformer.steps:
            if step is None or step == "passthrough":
                continue
            sub = output_groups(step, names)              # each output -> one of `names`
            lookup = dict(zip(names, origins))
            origins = [lookup[s] for s in sub]
            out_names = _feature_names_out(step, names)
            names = out_names if out_names is not None and len(out_names) == len(origins) \
                else [f"__c{i}" for i in range(len(origins))]
            if len(set(names)) != len(names):
                names = [f"{n}__{i}" for i, n in enumerate(names)]
                # positional fallback keeps origins aligned
        return origins
    if name == "ColumnTransformer":
        return _column_transformer_groups(transformer, inputs)
    if name == "SimpleImputer" and getattr(transformer, "add_indicator", False):
        ind = transformer.indicator_
        return inputs + [inputs[i] for i in ind.features_]
    if name == "MissingIndicator":
        return [inputs[i] for i in transformer.features_]

    out_names = _feature_names_out(transformer, inputs)
    if out_names is None or out_names == inputs:
        return inputs  # element-wise transform (scaler, imputer, FunctionTransformer, ...)
    # prefix mapping: "<input>_<suffix>"
    ordered = sorted(inputs, key=len, reverse=True)
    groups = []
    for o in out_names:
        hit = next((i for i in ordered if o == i or o.startswith(f"{i}_")), None)
        if hit is None:
            if len(out_names) == len(inputs):
                return inputs  # renamed one-to-one outputs
            raise UnsupportedModelError(
                f"can't tell which input produced column {o!r} of {name}; "
                "use explain(..., method='permutation') or pass groups= explicitly")
        groups.append(hit)
    return groups


def _feature_names_out(transformer: Any, inputs: list[str]):
    if not hasattr(transformer, "get_feature_names_out"):
        return None
    try:
        return [str(x) for x in transformer.get_feature_names_out(inputs)]
    except Exception:  # noqa: BLE001 - some transformers reject explicit names
        try:
            return [str(x) for x in transformer.get_feature_names_out()]
        except Exception:  # noqa: BLE001
            return None


def _resolve_columns(cols: Any, inputs: list[str]) -> list[str]:
    import numpy as np

    if isinstance(cols, str):
        return [cols]
    if isinstance(cols, slice):
        return inputs[cols]
    if callable(cols):
        raise UnsupportedModelError("unresolved callable column selector")
    cols = list(cols) if not isinstance(cols, np.ndarray) else cols.tolist()
    if cols and isinstance(cols[0], (bool, np.bool_)):
        return [c for c, keep in zip(inputs, cols) if keep]
    return [inputs[c] if isinstance(c, (int, np.integer)) else str(c) for c in cols]


def _column_transformer_groups(ct: Any, inputs: list[str]) -> list[str]:
    groups: list[str] = []
    slices = getattr(ct, "output_indices_", None)
    pieces = []
    for name, trans, cols in ct.transformers_:
        if trans == "drop":
            continue
        cols = _resolve_columns(cols, inputs)
        if not cols:
            continue
        sub = output_groups(trans, cols)
        pieces.append((name, sub))
    if slices:
        total = max((s.stop for s in slices.values()), default=0)
        out = [None] * total
        for name, sub in pieces:
            s = slices[name]
            if s.stop - s.start != len(sub):
                raise UnsupportedModelError(f"column count mismatch in transformer {name!r}")
            out[s.start:s.stop] = sub
        groups = [g for g in out if g is not None]
    else:
        for _, sub in pieces:
            groups.extend(sub)
    return groups
