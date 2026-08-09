"""Keep the subgoal-swap frozen-read selftest under check.py.

fontaine/scripts/subgoal_swap_results.py mechanizes the pre-registered
reads (2026-08-09-prereg-subgoal-swap.md); its ``--oracle`` selftest is
the exact-arithmetic fixture + every abort branch. Running it here means
a regression in the paired-read math or an abort branch going quiet
fails check.py, not the read session.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "subgoal_swap_results.py"
)
spec = importlib.util.spec_from_file_location("subgoal_swap_results", SCRIPT)
assert spec is not None and spec.loader is not None
ssr = importlib.util.module_from_spec(spec)
sys.modules["subgoal_swap_results"] = ssr
spec.loader.exec_module(ssr)


def test_oracle_selftest_green(capsys: pytest.CaptureFixture[str]) -> None:
    ssr.oracle()
    out = capsys.readouterr().out
    assert "oracle: ALL branches OK" in out
    # the selftest itself asserts the planted deltas; this guards the
    # abort-branch census staying complete
    assert out.count("abort branch OK") == 10
