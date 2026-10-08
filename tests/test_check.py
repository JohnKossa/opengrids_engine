import numpy as np
import pandas as pd
import pytest

from opengrids import CompGridEngine, DataCheckError, Dataset, FieldMeta, GridConfigError


def _rebuild(ds, sales=None, contrib=None, universe=None, fields=None):
    return Dataset(ds.sales if sales is None else sales,
                   ds.contributions if contrib is None else contrib,
                   ds._universe.reset_index(drop=True) if universe is None else universe,
                   fields=ds.fields if fields is None else fields, spatial_ladder=ds.spatial_ladder)


def _codes(report, severity=None):
    return {(i.code, i.field) for i in report.issues if severity is None or i.severity == severity}


def test_clean_demo_passes(ds):
    r = ds.check()
    assert r.ok and not r.warnings, str(r)


def test_missing_contributions(ds):
    c = ds.contributions
    r = _rebuild(ds, contrib=c.iloc[: int(len(c) * 0.9)]).check()
    assert ("missing_contributions", "key_sale") in _codes(r, "warning")
    r = _rebuild(ds, contrib=c.iloc[: int(len(c) * 0.3)]).check()
    assert ("missing_contributions", "key_sale") in _codes(r, "error")


def test_duplicate_and_orphan_rows(ds):
    c = ds.contributions
    extra = c.iloc[[0]].assign(key_sale="NOPE")
    r = _rebuild(ds, contrib=pd.concat([c, c.iloc[[5]], extra])).check()
    assert ("duplicate_rows", "key_sale") in _codes(r, "error")
    assert ("orphan_contributions", "key_sale") in _codes(r, "warning")


def test_numbers_stored_as_text(ds):
    s = ds.sales.assign(garage_sqft=ds.sales["garage_sqft"].astype(str))
    u = ds._universe.reset_index(drop=True).assign(garage_sqft=lambda d: d["garage_sqft"].astype(str))
    r = _rebuild(ds, sales=s, universe=u).check()
    issue = next(i for i in r.issues if i.field == "garage_sqft" and i.severity == "error")
    assert issue.code == "text_not_categorical" and "numbers stored as text" in issue.message


def test_text_not_categorical_and_type_mismatch(ds):
    fields = {**ds.fields, "neighborhood": FieldMeta("Neighborhood")}           # forgot is_categorical
    r = _rebuild(ds, fields=fields).check()
    assert ("text_not_categorical", "neighborhood") in _codes(r, "error")
    u = ds._universe.reset_index(drop=True).assign(building_quality_score=lambda d: "Q" + d["building_quality_score"].astype(str))
    r = _rebuild(ds, universe=u).check()
    assert ("type_mismatch", "building_quality_score") in _codes(r, "error")


def test_nulls_in_continuous_modeled_field_break_grids(ds, sold_subject):
    s = ds.sales.copy()
    nb = ds.subject(sold_subject)["neighborhood"]
    victims = s.index[(s["neighborhood"] == nb)][:3]
    s.loc[victims, "garage_sqft"] = np.nan
    broken = _rebuild(ds, sales=s)
    r = broken.check()
    issue = next(i for i in r.issues if i.code == "nulls_in_sales" and i.field == "garage_sqft")
    assert issue.severity == "error" and issue.count == 3
    # ...and the claim in the message is true: a pool containing one of them fails
    with pytest.raises(GridConfigError):
        CompGridEngine(broken, {"filters": [{"kind": "spatial"}], "peel_policy": {"enabled": False}}).grid(sold_subject)


def test_nulls_in_universe(ds):
    u = ds._universe.reset_index(drop=True)
    u.loc[u.index[:7], "land_area_sqft"] = np.nan
    r = _rebuild(ds, universe=u).check()
    issue = next(i for i in r.issues if i.code == "nulls_in_universe")
    assert issue.severity == "warning" and issue.count == 7


def test_log_space_contributions_are_flagged(ds):
    c = ds.contributions.copy()
    feats = ds.modeled_features
    c[feats] = c[feats] / c["prediction"].values[:, None]   # log-ish units, not dollars
    c["base_value"] = np.log(c["base_value"])
    r = _rebuild(ds, contrib=c).check()
    assert ("not_additive", None) in _codes(r, "error")


def test_config_references(ds, engine):
    r = engine.check({"filters": [{"kind": "relative", "field": "sfla", "params": {"type": "pct", "value": 10}}],
                      "adjustment_overrides": {"bogus:1": 5}})
    codes = _codes(r, "warning")
    assert ("unknown_filter_field", "sfla") in codes and ("override_unused", "bogus") in codes


def test_raise_if_errors_and_frame(ds):
    fields = {**ds.fields, "neighborhood": FieldMeta("Neighborhood")}
    r = _rebuild(ds, fields=fields).check()
    with pytest.raises(DataCheckError):
        r.raise_if_errors()
    df = r.to_frame()
    assert {"severity", "code", "field", "message"} <= set(df.columns)
    assert "ERROR" in str(r)
