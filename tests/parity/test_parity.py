"""pytest entry for the parity check. Skipped unless COMP_ADJ_APP points at a
comp_adj_app checkout containing comp_adj.db. The full scenario matrix lives
in run_parity.py."""
import os
import random

import pytest

APP = os.environ.get("COMP_ADJ_APP")
pytestmark = [pytest.mark.parity,
              pytest.mark.skipif(not APP or not os.path.exists(os.path.join(APP or "", "comp_adj.db")),
                                 reason="set COMP_ADJ_APP to a comp_adj_app checkout to run parity tests")]


@pytest.fixture(scope="module")
def both():
    from opengrids import CompGridEngine
    from opengrids.compat.comp_adj_app import load
    from tests.parity.app_reference import AppReference

    ds, _ = load(os.path.join(APP, "comp_adj.db"))
    ref = AppReference(APP)
    cfg = ref.config()
    return ds, ref, cfg, CompGridEngine(ds, {k: v for k, v in cfg.items() if not k.startswith("_")})


def test_office_default_grids_match(both):
    from tests.parity.compare import compare

    ds, ref, cfg, engine = both
    keys = list(ds._universe[ds._universe["prediction"].notna()].index)
    for subj in random.Random(11).sample(keys, 20):
        assert compare(ref.grid(subj, cfg), engine.grid(subj)) == [], subj
