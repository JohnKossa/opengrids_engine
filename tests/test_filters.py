import numpy as np
import pandas as pd

from opengrids import Dataset
from opengrids.finder import CompFinder


def _ds():
    sales = pd.DataFrame({
        "key": ["1", "2", "3", "4", "5"],
        "key_sale": ["a", "b", "c", "d", "e"],
        "sale_date": ["2025-01-01"] * 5,
        "sale_price_time_adj": [1.0] * 5,
        "code": ["Pool", "pool", None, "SPA", "09"],
        "n": [1.0, 2.0, np.nan, 4.0, 9.0],
        "flag": [1, 0, 1, 0, 1],
    })
    contrib = pd.DataFrame({"key": sales["key"], "key_sale": sales["key_sale"]})
    return Dataset(sales, contrib)


def _keys(ds, field, op, value=None, value2=None):
    m = CompFinder(ds)._absolute_mask(field, op, value, value2)
    return list(ds.sales.loc[m, "key"])


def test_sql_null_semantics():
    ds = _ds()
    assert _keys(ds, "n", "!=", 2) == ["1", "4", "5"]          # NULL row excluded
    assert _keys(ds, "code", "not_in", "SPA") == ["1", "2", "5"]
    assert _keys(ds, "n", "is_null") == ["3"]


def test_like_is_case_insensitive():
    ds = _ds()
    assert _keys(ds, "code", "contains", "pOOl") == ["1", "2"]
    assert _keys(ds, "code", "starts_with", "s") == ["4"]


def test_type_coercion():
    ds = _ds()
    assert _keys(ds, "n", ">=", "4") == ["4", "5"]             # text value vs numeric column
    assert _keys(ds, "n", "=", 9) == ["5"]
    assert _keys(ds, "code", "=", "09") == ["5"]
    assert _keys(ds, "n", "between", "2", "4") == ["2", "4"]
    assert _keys(ds, "flag", "is_true") == ["1", "3", "5"]
    assert _keys(ds, "n", "in", "1, 9") == ["1", "5"]
