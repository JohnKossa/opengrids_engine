"""Every example script must run cleanly (they double as documentation)."""
import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path)])
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out.strip()
