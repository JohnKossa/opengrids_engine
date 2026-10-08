import json
import math

import numpy as np
import pandas as pd
import pytest

from opengrids import CompGridEngine, CostOfSale, Dataset, FieldMeta, GridConfigError, normalize_config
from opengrids.demo import TRUE_EFFECTS


def test_grid_arithmetic(engine, sold_subject):
    g = engine.grid(sold_subject)
    assert 1 <= len(g.comps) <= 4
    for c in g.comps:
        assert math.isclose(c["total_adj"], sum(c["adjustments"].values()))
        assert math.isclose(c["final_price"], c["sale_price_cost_of_sale_adj"] + c["total_adj"])
        for f, rate in g.unit_adjs.items():
            if isinstance(rate, dict):
                continue
            expected = (g.subject[f] - c[f]) * rate
            assert math.isclose(c["adjustments"][f], expected, rel_tol=1e-9, abs_tol=1e-9)
    dates = [c["sale_date"] for c in g.comps]
    assert dates == sorted(dates, reverse=True)


def test_shap_rates_recover_linear_effects(engine, sold_subject):
    g = engine.grid(sold_subject)
    assert g.raw_unit_adjs["garage_sqft"] == pytest.approx(TRUE_EFFECTS["garage_sqft"], rel=1e-6)
    pool = g.raw_unit_adjs["has_pool"]
    if {"0", "1"} <= set(pool):
        assert pool["1"] - pool["0"] == pytest.approx(TRUE_EFFECTS["has_pool"], rel=1e-6)


def test_categorical_adjustment_is_variant_difference(engine, universe, ds):
    # find a subject whose grid includes a comp with a different pool status
    for key in universe["key"].iloc[:200]:
        g = engine.grid(key)
        for c in g.comps:
            if c["has_pool"] != g.subject["has_pool"]:
                m = g.unit_adjs["has_pool"]
                assert c["adjustments"]["has_pool"] == pytest.approx(
                    m[str(g.subject["has_pool"])] - m[str(c["has_pool"])])
                return
    pytest.skip("no mixed-pool grid found")


def test_sign_clamp_and_override_exemption(ds, sold_subject):
    fields = dict(ds.fields)
    fields["garage_sqft"] = FieldMeta("Garage", impact="negative")  # wrong on purpose
    ds2 = Dataset(ds.sales, ds.contributions, ds._universe, fields=fields, spatial_ladder=ds.spatial_ladder)
    g = CompGridEngine(ds2).grid(sold_subject)
    assert g.unit_adjs["garage_sqft"] == 0
    assert "garage_sqft" in [c["name"] for c in g.clamped_fields]
    assert g.raw_unit_adjs["garage_sqft"] > 0
    # a per-search override is exempt from the clamp
    g2 = CompGridEngine(ds2, {"adjustment_overrides": {"garage_sqft": 55}}).grid(sold_subject)
    assert g2.unit_adjs["garage_sqft"] == 55
    assert not g2.clamped_fields or "garage_sqft" not in [c["name"] for c in g2.clamped_fields]


def test_override_precedence(ds, sold_subject):
    fields = dict(ds.fields)
    fields["garage_sqft"] = FieldMeta("Garage", impact="positive", override_rate=10.0)
    fields["has_pool"] = FieldMeta("Pool", is_categorical=True, variant_overrides={1: 30000, 0: 0})
    ds2 = Dataset(ds.sales, ds.contributions, ds._universe, fields=fields, spatial_ladder=ds.spatial_ladder)
    assert CompGridEngine(ds2).grid(sold_subject).unit_adjs["garage_sqft"] == 10.0
    g = CompGridEngine(ds2, {"adjustment_overrides": {"garage_sqft": 20.0}}).grid(sold_subject)
    assert g.unit_adjs["garage_sqft"] == 20.0
    assert g.unit_adjs["has_pool"]["1"] == 30000


def test_peel_relaxes_until_target(engine, sold_subject):
    cfg = {"filters": [
        {"kind": "spatial", "params": {}},
        {"kind": "geo_radius", "params": {"radius": 0.05}},
        {"kind": "relative", "field": "bldg_area_finished_sqft", "params": {"type": "pct", "value": 2}},
    ], "peel_policy": {"min_comps": 30, "min_comps_per_param": 0}}
    cands, audit = engine.candidates(sold_subject, config=cfg)
    assert audit[0]["change"] == {"action": "base"}
    assert len(audit) > 1
    assert audit[-1]["stop"] in ("target met", "exhausted")
    if audit[-1]["stop"] == "target met":
        assert len(cands) >= 30
    # bottom-up: the GLA band (last in list) is relaxed before the radius
    first = audit[1]["change"]
    assert first["field"] == "bldg_area_finished_sqft"


def test_peel_disabled_uses_base_filters_only(engine, sold_subject):
    _, audit = engine.candidates(sold_subject, config={"peel_policy": {"enabled": False}})
    assert len(audit) == 1 and audit[0]["stop"] == "target met"


def test_spatial_ladder_replace_on_clean_step(engine, sold_subject):
    cfg = {"filters": [{"kind": "spatial", "params": {}, "peel": {"behavior": "relax"}}],
           "peel_policy": {"min_comps": 10_000, "min_comps_per_param": 0}}
    cands, audit = engine.candidates(sold_subject, config=cfg)
    ladder = [a["change"] for a in audit if a["change"].get("action") == "ladder"]
    assert ladder and ladder[0]["mode"] == "replace" and ladder[0]["to"] == "district"
    assert audit[-1]["stop"] == "exhausted"


def test_fwl_unmodeled_field(engine, sold_subject, ds):
    cfg = {"adj_fields": ds.modeled_features + ["bedrooms"],
           "peel_policy": {"min_comps": 150, "min_comps_per_param": 0}}
    g = engine.grid(sold_subject, config=cfg)
    assert g.fwl_fields == ["bedrooms"]
    assert np.isfinite(g.unit_adjs["bedrooms"])


def test_cost_of_sale_scales_rates_and_baseline(ds, sold_subject):
    cos = CostOfSale(pct=15)
    ds2 = Dataset(ds.sales, ds.contributions, ds._universe, fields=ds.fields,
                  spatial_ladder=ds.spatial_ladder, cost_of_sale=cos)
    g0 = CompGridEngine(ds).grid(sold_subject)
    g1 = CompGridEngine(ds2).grid(sold_subject)
    assert g1.raw_unit_adjs["garage_sqft"] == pytest.approx(0.85 * g0.raw_unit_adjs["garage_sqft"])
    assert g1.subject["prediction_cost_of_sale_adj"] == pytest.approx(0.85 * g0.subject["prediction"])
    for c in g1.comps:
        assert c["sale_price_cost_of_sale_adj"] == pytest.approx(0.85 * c["sale_price_time_adj"])


def test_max_adjustment_pct_excludes_and_backfills(engine, sold_subject):
    g = engine.grid(sold_subject, config={"grid": {"max_adjustment_pct": 1.0}})
    for c in g.comps:
        assert abs(c["total_adj"]) / c["sale_price_cost_of_sale_adj"] * 100 <= 1.0
    loose = engine.grid(sold_subject)
    if any(abs(c["total_adj"]) / c["sale_price_cost_of_sale_adj"] * 100 > 1.0 for c in loose.comps):
        assert g.excluded_comp_count > 0


def test_prefer_bracketed(engine, universe):
    cfg = {"ranking": {"method": "smallest_adj", "prefer_bracketed": True}}
    hits = 0
    for key in universe["key"].iloc[:60]:
        g = engine.grid(key, config=cfg)
        m = g.metrics()
        plain = engine.grid(key).metrics()
        if plain["comps_above"] or plain["comps_below"]:
            hits += 1
            # bracketing never reduces coverage on either side
            assert (m["comps_above"] > 0) or (plain["comps_above"] == 0) or m["comp_count"] < 2
    assert hits


def test_pinned_comp(engine, sold_subject, ds):
    far = ds.sales[ds.sales["district"] != engine.grid(sold_subject).subject["district"]]["key"].iloc[0]
    for method in ("none", "expanding_radius", "stretching_range", "second_subject"):
        g = engine.grid(sold_subject, user_comp=far, config={"expansion": method})
        pinned = [c for c in g.comps if c["is_user_comp"]]
        assert len(pinned) == 1 and pinned[0]["key"] == far


def test_size_config(engine, sold_subject):
    g = engine.grid(sold_subject, config={"grid": {"size": 6}})
    assert len(g.comps) == 6


def test_outputs_serialize(engine, sold_subject):
    g = engine.grid(sold_subject)
    d = g.to_dict(diagnostics=True)
    json.dumps(d)
    assert d["comps"] and d["diagnostics"]["peel_audit"]
    f = g.to_frame()
    assert list(f.columns)[0] == "subject" and f.shape[1] == len(g.comps) + 1


def test_bad_adj_field_raises(engine, sold_subject):
    with pytest.raises(GridConfigError) as e:
        engine.grid(sold_subject, config={"adj_fields": ["not_a_column"]})
    assert e.value.problems[0]["field"] == "not_a_column"


def test_fidelity_is_near_zero_for_near_linear_model(engine, sold_subject):
    fid = engine.grid(sold_subject).surrogate_fidelity
    assert fid is not None and fid["mdape"] < 0.1


def test_subject_as_dict_matches_key(engine, sold_subject, ds):
    a = engine.grid(sold_subject)
    b = engine.grid(ds.subject(sold_subject))
    assert [c["key"] for c in a.comps] == [c["key"] for c in b.comps]


def test_normalize_config_drops_invalid():
    cfg = normalize_config({"filters": [{"kind": "bogus"}, {"kind": "relative", "field": "x", "params": {"type": "pct", "value": "5"}}],
                            "ranking": {"method": "nope"}, "_schema": 2})
    assert len(cfg["filters"]) == 1 and cfg["filters"][0]["params"]["value"] == 5.0
    assert cfg["ranking"]["method"] == "smallest_adj"
    assert "_schema" not in cfg


def test_rate_sources_and_model_rates(engine, sold_subject, ds):
    g = engine.grid(sold_subject, config={"adj_fields": ds.modeled_features + ["bedrooms"],
                                          "adjustment_overrides": {"garage_sqft": 75, "has_pool:1": 1, "has_pool:0": 0}})
    src = g.rate_sources
    assert src["garage_sqft"] == "override (this search)"
    assert src["bedrooms"] == "fitted to sales (FWL)"
    assert src["has_pool"].endswith("2 variant override(s)")
    assert src["land_area_sqft"] == "model (SHAP)"
    r = g.rates()
    assert r.loc["garage_sqft", "rate"] == 75 and r.loc["garage_sqft", "fitted_rate"] == pytest.approx(40.0)


def test_mi_shrinkage_switch(engine, sold_subject, ds):
    cfg = {"adj_fields": ds.modeled_features + ["bedrooms"], "filters": [], "peel_policy": {"enabled": False}}
    on = engine.grid(sold_subject, config=cfg).unit_adjs["bedrooms"]
    off = engine.grid(sold_subject, config={**cfg, "fwl": {"mi_shrinkage": False}}).unit_adjs["bedrooms"]
    assert on != off
    assert abs(off - 7500) < abs(on - 7500)      # the synthetic truth
