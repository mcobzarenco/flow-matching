"""Keep the #6 rung-(c) mcselect frozen-read selftest under check.py.

fontaine/scripts/mcselect_results.py mechanizes the rung-(c) pre-reg
draft's reads (2026-08-09-prereg-subgoal-mcselect.md) and PINS the
future scorer run's dump contract; its ``--oracle`` selftest is the
planted-argmax fixture (exact paired arithmetic, tie rule, capture
fraction) + every abort branch. Running it here means a regression in
the pick logic or an abort branch going quiet fails check.py, not the
read session.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "fontaine" / "scripts" / "mcselect_results.py"
)
spec = importlib.util.spec_from_file_location("mcselect_results", SCRIPT)
assert spec is not None and spec.loader is not None
mcr = importlib.util.module_from_spec(spec)
sys.modules["mcselect_results"] = mcr
spec.loader.exec_module(mcr)


def test_oracle_selftest_green(capsys: pytest.CaptureFixture[str]) -> None:
    mcr.oracle()
    out = capsys.readouterr().out
    assert "oracle: ALL branches OK" in out
    assert "planted fixture OK" in out
    # guards the abort-branch census staying complete
    assert out.count("abort branch OK") == 10
