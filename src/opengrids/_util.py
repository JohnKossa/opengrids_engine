"""Small value-handling helpers shared across the engine.

Filters follow SQL semantics so they behave the way assessors' database
queries do: NULL never matches a comparison, ``LIKE``-style text matching is
case-insensitive, and a value is coerced to the column's type before
comparing.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


def is_missing(v: Any) -> bool:
    """True for None, NaN, NaT and pandas NA."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def py(v: Any) -> Any:
    """numpy scalar -> native Python scalar; NaN/NA -> None."""
    if is_missing(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def canon(v: Any) -> str:
    """Canonical string key for a categorical value.

    Categorical adjustments are stored as ``{variant: rate}`` maps keyed by
    string. The same variant can show up as ``9``, ``9.0``, ``np.int64(9)``
    or ``True``/``1`` depending on how a column was loaded (an integer column
    with nulls becomes float in pandas, for example), so every lookup goes
    through this function. Integral floats collapse to their integer form and
    booleans to ``'1'``/``'0'``. Missing values become ``'None'``.
    """
    v = py(v)
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def coerce_scalar(raw: Any) -> Any:
    """Best-effort typing of a raw filter value (often a string from a form).

    ``'12'`` -> ``12``, ``'1.5'`` -> ``1.5``, any other text stays text, and
    blank -> ``None``.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None
        try:
            return float(text) if "." in text else int(text)
        except ValueError:
            return text
    return raw


def value_for_column(value: Any, series: pd.Series) -> Any:
    """Coerce a scalar to something comparable with ``series``.

    This mirrors SQLite's affinity conversion for a bound parameter. Returns
    ``_NOMATCH`` when the value can't be represented in the column's type, in
    which case the comparison matches nothing.
    """
    value = py(value)
    if value is None:
        return _NOMATCH
    if pd.api.types.is_bool_dtype(series.dtype):
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"1", "true", "t", "y", "yes"}:
                return True
            if low in {"0", "false", "f", "n", "no"}:
                return False
            return _NOMATCH
        return bool(value)
    if pd.api.types.is_numeric_dtype(series.dtype):
        if isinstance(value, (int, float)):
            return value
        try:
            return float(str(value).strip())
        except ValueError:
            return _NOMATCH
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return _NOMATCH
    # object / string column
    return value


class _NoMatch:
    def __repr__(self) -> str:  # pragma: no cover
        return "<no match>"


_NOMATCH = _NoMatch()


def _object_equals(series: pd.Series, value: Any, canon_series: pd.Series | None = None) -> pd.Series:
    """Equality on an object/string column, compared via ``canon`` on both sides.

    ``canon_series`` is an optional precomputed ``series.map(canon)``; the
    engine caches one per column so that peel iterations stay vectorized.
    """
    if canon_series is None:
        canon_series = series.map(canon)
    return ((canon_series == canon(value)) & series.notna()).astype(bool)


def eq_mask(series: pd.Series, value: Any, canon_series: pd.Series | None = None) -> pd.Series:
    """``series = value`` with SQL NULL semantics (null rows never match)."""
    v = value_for_column(value, series)
    if v is _NOMATCH:
        return pd.Series(False, index=series.index)
    if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
        return _object_equals(series, v, canon_series)
    return (series == v).fillna(False).astype(bool)


def cmp_mask(series: pd.Series, op: str, value: Any, canon_series: pd.Series | None = None) -> pd.Series:
    """``series <op> value`` for op in =, !=, >, >=, <, <= with SQL NULL semantics."""
    if op == "=":
        return eq_mask(series, value, canon_series)
    notnull = series.notna()
    if op == "!=":
        return notnull & ~eq_mask(series, value, canon_series)
    v = value_for_column(value, series)
    if v is _NOMATCH:
        return pd.Series(False, index=series.index)
    s = series
    if (series.dtype == object) and isinstance(v, (int, float)):
        s = pd.to_numeric(series, errors="coerce")
    try:
        if op == ">":
            out = s > v
        elif op == ">=":
            out = s >= v
        elif op == "<":
            out = s < v
        elif op == "<=":
            out = s <= v
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"unsupported operator {op!r}")
    except TypeError:
        return pd.Series(False, index=series.index)
    return (out.fillna(False) & notnull).astype(bool)


def between_mask(series: pd.Series, lo: Any, hi: Any) -> pd.Series:
    lo_v = value_for_column(lo, series)
    hi_v = value_for_column(hi, series)
    if lo_v is _NOMATCH or hi_v is _NOMATCH:
        return pd.Series(False, index=series.index)
    s = series
    if series.dtype == object and isinstance(lo_v, (int, float)):
        s = pd.to_numeric(series, errors="coerce")
    try:
        return s.between(lo_v, hi_v).fillna(False).astype(bool)
    except TypeError:
        return pd.Series(False, index=series.index)


def in_mask(series: pd.Series, values: list, canon_series: pd.Series | None = None) -> pd.Series:
    out = pd.Series(False, index=series.index)
    for v in values:
        out |= eq_mask(series, v, canon_series)
    return out


def like_mask(series: pd.Series, pattern: str) -> pd.Series:
    """SQLite ``LIKE``: case-insensitive, ``%`` = any run, ``_`` = one char."""
    rx = "".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern)
    regex = re.compile(rf"^{rx}$", re.IGNORECASE | re.DOTALL)
    return series.map(lambda x: (not is_missing(x)) and bool(regex.match(str(x)))).astype(bool)


def json_safe(obj: Any) -> Any:
    """Recursively coerce numpy/pandas/datetime values to JSON-able primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)) and not isinstance(obj, np.generic):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Decimal):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    if is_missing(obj):
        return None
    return str(obj)
