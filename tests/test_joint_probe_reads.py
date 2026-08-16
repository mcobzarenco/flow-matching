"""Oracles for the route-C joint-probe reads instrument
(``fontaine/scripts/grasp_sft_joint_probe_reads.py``): the decision
surfaces it bakes are the two parents' FROZEN ones (A §5 / B §3 via
the 2026-08-16 amendment §4) — a drifted threshold would bank a wrong
verdict at the endpoint boundary, so the bands are pinned here edge by
edge, along with the kept-split and the serve-head provenance guard on
synthetic probe JSONs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fontaine.scripts.grasp_sft_joint_probe_reads import (
    arm_stats,
    flow_verdict,
    token_verdict,
)

READS = Path(__file__).parent.parent / "fontaine/scripts/grasp_sft_joint_probe_reads.py"


def verdict(value: str | None) -> str:
    assert value is not None
    return value


def episode(seed: int, *, success: bool, progress: float = 3.0) -> dict:
    return {
        "seed": seed,
        "success_tick": 120 if success else None,
        "progress_final_cm": progress,
        "reset_strikes": 0,
    }


def test_flow_verdict_bands_pin_a5() -> None:
    """A §5 verbatim, overlap surfaced: >31 clean positive; 29-31 the
    frozen text's two clauses BOTH apply (flagged, never silently
    resolved); 25-28 data-limited; <25 seam."""
    assert flow_verdict(None) is None
    assert verdict(flow_verdict(32)) == (
        "TABLE_FIX_POSITIVE (>28+3: corrected lineage becomes the SFT artifact)"
    )
    assert verdict(flow_verdict(31)).startswith("TABLE_FIX_POSITIVE_WITHIN_BAND")
    assert verdict(flow_verdict(29)).startswith("TABLE_FIX_POSITIVE_WITHIN_BAND")
    assert verdict(flow_verdict(28)).startswith("DATA_LIMITED")
    assert verdict(flow_verdict(25)).startswith("DATA_LIMITED")  # 28-3 inclusive band
    assert verdict(flow_verdict(24)).startswith("SEAM_INVESTIGATION")
    assert verdict(flow_verdict(0)).startswith("SEAM_INVESTIGATION")


def test_token_verdict_bands_pin_b3() -> None:
    """B §3 verbatim: >=20 activates R2, 5-19 owner, <5 no-transfer."""
    assert token_verdict(None) is None
    assert verdict(token_verdict(20)).startswith("R2_PREMISE_HOLDS")
    assert verdict(token_verdict(19)).startswith("OWNER_DECISION")
    assert verdict(token_verdict(5)).startswith("OWNER_DECISION")
    assert verdict(token_verdict(4)).startswith("NO_TRANSFER")


def test_arm_stats_conventions() -> None:
    """Success = success_tick non-null; moved = progress > 0.5 cm —
    the step2000-probe row conventions, verbatim."""
    stats = arm_stats(
        [
            episode(3, success=True),
            episode(1, success=False, progress=0.6),
            episode(2, success=False, progress=0.2),
        ],
    )
    assert stats["n"] == 3
    assert stats["successes"] == 1
    assert stats["success_seeds"] == [3]
    assert stats["moved_gt_half_cm"] == 2  # 3.0 and 0.6; 0.2 sits still


def test_end_to_end_kept_split_and_provenance(tmp_path: Path) -> None:
    """The full CLI on synthetic legs: kept/non-kept split off the
    collect state, serve_head provenance carried per arm, missing legs
    read null (a partial boundary never crashes the instrument)."""
    probe = tmp_path / "probes"
    probe.mkdir()
    (probe / "flow_unseen.json").write_text(
        json.dumps(
            {
                "config": {"serve_head": None},
                "episodes": [episode(s, success=s < 30) for s in range(100)],
            },
        ),
    )
    (probe / "flow_train.json").write_text(
        json.dumps(
            {
                "config": {"serve_head": None},
                "episodes": [
                    episode(s, success=s in (1000, 1001, 1002))
                    for s in range(1000, 1100)
                ],
            },
        ),
    )
    (probe / "token_unseen.json").write_text(
        json.dumps(
            {
                "config": {"serve_head": "ar"},
                "episodes": [episode(s, success=s < 7) for s in range(100)],
            },
        ),
    )
    collect = tmp_path / "collect.json"
    collect.write_text(json.dumps({"kept_seeds": [1000, 1001, 1050]}))
    out = tmp_path / "analysis.json"
    subprocess.run(
        [
            sys.executable,
            str(READS),
            "--probe-dir",
            str(probe),
            "--collect",
            str(collect),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    banked = json.loads(out.read_text())
    assert banked["flow_unseen_0_99"]["successes"] == 30
    assert banked["flow_verdict_A5"].startswith("TABLE_FIX_POSITIVE_WITHIN_BAND")
    assert banked["token_unseen_0_99"]["successes"] == 7
    assert banked["token_unseen_0_99"]["serve_head"] == "ar"
    assert banked["token_verdict_B3"].startswith("OWNER_DECISION")
    # kept 1000,1001,1050: two of the three successes are kept seeds.
    assert banked["flow_train_kept"]["n"] == 3
    assert banked["flow_train_kept"]["successes"] == 2
    assert banked["flow_train_nonkept"]["n"] == 97
    assert banked["flow_train_nonkept"]["successes"] == 1
    # token_base leg missing: null arm, no delta key crash.
    assert banked["token_base_unseen_0_99"] is None
    assert "token_sft_delta" not in banked
