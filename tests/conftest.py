import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opengrids import CompGridEngine  # noqa: E402
from opengrids.demo import make_demo  # noqa: E402


@pytest.fixture(scope="session")
def demo():
    return make_demo()


@pytest.fixture(scope="session")
def ds(demo):
    return demo[0]


@pytest.fixture(scope="session")
def universe(demo):
    return demo[1]


@pytest.fixture()
def engine(ds):
    return CompGridEngine(ds)


@pytest.fixture(scope="session")
def sold_subject(ds, universe):
    """A parcel in a well-populated neighborhood."""
    counts = ds.sales["neighborhood"].value_counts()
    nb = counts.index[0]
    return universe[universe["neighborhood"] == nb]["key"].iloc[3]
