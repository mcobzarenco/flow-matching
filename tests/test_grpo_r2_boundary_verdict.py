"""Oracles for the R2 boundary verdict instrument
(``fontaine/scripts/grpo_r2_boundary_verdict.py``): A3.4's three endpoint
legs mechanized. The PRIMARY's paired-exact band edges are pinned pair by
pair (the minimal clean sweep to IMPROVED is 5 wins / 0 losses; each extra
loss buys the edge two more wins), the flow material-regression line is
pinned at the exact 5% tail under Bin(100, 0.44) (k <= 35), the sampled
sibling stays record-only, and the provenance guards refuse the wrong
leg's json, mixed checkpoints, the pinned base, and a non-PASS preflight.
"""

from __future__ import annotations

import math

import pytest

from fontaine.scripts.grpo_r2_boundary_verdict import (
    FLOW_ANCHOR,
    FLOW_SUCCESS_SEEDS,
    GREEDY_SUCCESS_SEEDS,
    boundary_verdict,
    sign_test_tail,
)
from fontaine.scripts.grpo_r2_preflight_verdict import binom_tail_below

BOUNDARY_CKPT = "outputs/sim/grpo_r2/loop/step_000010"


def leg(success_seeds: set[int], **config_overrides: object) -> dict:
    config: dict[str, object] = {
        "serve_head": "ar",
        "ar_temperature": None,
        "method": "heun",
        "sample_steps": 10,
        "seed": 0,
        "num_seeds": 100,
        "checkpoint": BOUNDARY_CKPT,
        "commit": "deadbee",
    }
    config.update(config_overrides)
    return {
        "config": config,
        "episodes": [
            {"seed": s, "success_tick": 240 if s in success_seeds else None}
            for s in range(100)
        ],
    }


def sampled_leg(success_seeds: set[int], **overrides: object) -> dict:
    return leg(success_seeds, ar_temperature=1.0, **overrides)


def flow_leg(success_seeds: set[int], **overrides: object) -> dict:
    config: dict[str, object] = {"serve_head": None, "method": "euler"}
    config.update(overrides)
    return leg(success_seeds, **config)


PREFLIGHT = {"verdict": "PASS", "sampled_successes": 8}


def verdict(
    greedy: set[int],
    sampled: set[int] | None = None,
    flow: set[int] | None = None,
) -> dict:
    return boundary_verdict(
        leg(greedy),
        sampled_leg(sampled if sampled is not None else greedy),
        flow_leg(flow if flow is not None else set(FLOW_SUCCESS_SEEDS)),
        PREFLIGHT,
    )


def wins(n: int) -> set[int]:
    """n never-succeeded seeds (disjoint from both anchors' success sets)."""
    fresh = [
        s
        for s in range(100)
        if s not in GREEDY_SUCCESS_SEEDS and s not in FLOW_SUCCESS_SEEDS
    ]
    assert len(fresh) >= n
    return set(fresh[:n])


def keep(n: int) -> set[int]:
    """The first n of the greedy anchor's 7 success seeds."""
    return set(sorted(GREEDY_SUCCESS_SEEDS)[:n])


def test_sign_test_exact() -> None:
    # Independent pins: pure-win tails are 2^-b; no discordance = no read.
    assert sign_test_tail(0, 0) == 1.0
    for b in range(1, 8):
        assert sign_test_tail(b, b) == pytest.approx(0.5**b, rel=1e-12)
    # One loss: P(X >= b | Bin(b+1, 1/2)) = (b+2) / 2^(b+1).
    assert sign_test_tail(7, 8) == pytest.approx(9 / 256, rel=1e-12)
    assert sign_test_tail(6, 7) == pytest.approx(8 / 128, rel=1e-12)


def test_primary_band_edges() -> None:
    # 0 losses: 4 wins FLAT (2^-4 = 0.0625), 5 wins IMPROVED (0.03125).
    assert verdict(keep(7) | wins(4))["primary"]["band"] == "FLAT"
    assert verdict(keep(7) | wins(5))["primary"]["band"] == "IMPROVED"
    # 1 loss: 6 wins FLAT (0.0625), 7 wins IMPROVED (~0.0352).
    assert verdict(keep(6) | wins(6))["primary"]["band"] == "FLAT"
    assert verdict(keep(6) | wins(7))["primary"]["band"] == "IMPROVED"
    # 2 losses: 8 wins FLAT (~0.0547), 9 wins IMPROVED (~0.0327).
    assert verdict(keep(5) | wins(8))["primary"]["band"] == "FLAT"
    assert verdict(keep(5) | wins(9))["primary"]["band"] == "IMPROVED"
    # Regression side mirrors: 4 losses FLAT, 5 losses REGRESSED.
    assert verdict(keep(3))["primary"]["band"] == "FLAT"
    assert verdict(keep(2))["primary"]["band"] == "REGRESSED"
    assert verdict(set())["primary"]["band"] == "REGRESSED"
    # Identical outcome set: no discordant pairs, FLAT with p = 1.
    same = verdict(set(GREEDY_SUCCESS_SEEDS))
    assert same["primary"]["band"] == "FLAT"
    assert same["primary"]["p_improve_exact"] == 1.0


def test_primary_receipts() -> None:
    result = verdict(keep(6) | wins(7))["primary"]
    assert result["boundary_successes"] == 13
    assert result["anchor_successes"] == 7
    assert result["wins"] == 7
    assert result["losses"] == 1
    assert result["win_seeds"] == sorted(wins(7))
    assert result["loss_seeds"] == [96]  # max of the anchor set is dropped
    assert result["held_seeds"] == sorted(keep(6))
    assert result["p_improve_exact"] == pytest.approx(9 / 256, abs=5e-7)


def test_flow_material_line() -> None:
    # The exact 5% tail under Bin(100, 0.44) admits exactly counts 0-35.
    below = [k for k in range(101) if binom_tail_below(k, 100, 0.44) < 0.05]
    assert below == list(range(36))
    at_edge = verdict(keep(7), flow=set(sorted(FLOW_SUCCESS_SEEDS)[:35]))
    over_edge = verdict(keep(7), flow=set(sorted(FLOW_SUCCESS_SEEDS)[:36]))
    assert at_edge["flow_regression"]["materially_below"] is True
    assert over_edge["flow_regression"]["materially_below"] is False
    assert at_edge["flow_regression"]["material_line_max"] == 35
    # The registered surface is stated; the call stays judged.
    assert any("judge" in note for note in at_edge["judged_separately"])


def test_overall_surface_combination() -> None:
    # IMPROVED + flow clean -> accumulation.
    assert verdict(keep(7) | wins(5))["overall_surface"] == "accumulation"
    # Flow material regression outranks token improvement (A3.4).
    hurt_flow = set(sorted(FLOW_SUCCESS_SEEDS)[:30])
    assert verdict(keep(7) | wins(5), flow=hurt_flow)["overall_surface"] == (
        "f-regression"
    )
    assert verdict(keep(2))["overall_surface"] == "f-regression"
    assert verdict(keep(7))["overall_surface"] == "f-flat"


def test_sampled_sibling_is_record_only() -> None:
    result = verdict(keep(7) | wins(5), sampled=wins(20))
    record = result["sampled_record"]
    assert record["boundary_successes"] == 20
    assert record["preflight_floor"] == 8
    assert record["delta_vs_floor"] == 12
    # Base gap = preflight 8 - greedy anchor 7; boundary gap = 20 - 12.
    assert record["decode_gap_base"] == 1
    assert record["decode_gap_boundary"] == 8
    assert record["decode_gap_movement"] == 7
    # A collapsed sampled read never touches the bands (record-only).
    collapsed = verdict(keep(7) | wins(5), sampled=set())
    assert collapsed["primary"]["band"] == "IMPROVED"
    assert collapsed["overall_surface"] == "accumulation"
    assert collapsed["sampled_record"]["decode_gap_movement"] == -13


def test_provenance_guards() -> None:
    good = keep(7)
    with pytest.raises(ValueError, match="serve-head ar"):
        boundary_verdict(
            leg(good, serve_head="flow"),
            sampled_leg(good),
            flow_leg(good),
            PREFLIGHT,
        )
    with pytest.raises(ValueError, match="GREEDY"):
        boundary_verdict(
            leg(good, ar_temperature=1.0),
            sampled_leg(good),
            flow_leg(good),
            PREFLIGHT,
        )
    with pytest.raises(ValueError, match=r"T=1\.0"):
        boundary_verdict(leg(good), leg(good), flow_leg(good), PREFLIGHT)
    with pytest.raises(ValueError, match="flow head"):
        boundary_verdict(
            leg(good),
            sampled_leg(good),
            flow_leg(good, serve_head="ar"),
            PREFLIGHT,
        )
    with pytest.raises(ValueError, match="euler-10"):
        boundary_verdict(
            leg(good),
            sampled_leg(good),
            flow_leg(good, method="heun"),
            PREFLIGHT,
        )
    with pytest.raises(ValueError, match="euler-10"):
        boundary_verdict(
            leg(good),
            sampled_leg(good),
            flow_leg(good, sample_steps=20),
            PREFLIGHT,
        )
    with pytest.raises(ValueError, match="ONE checkpoint"):
        boundary_verdict(
            leg(good),
            sampled_leg(good, checkpoint="somewhere/else"),
            flow_leg(good),
            PREFLIGHT,
        )
    for base in ("step_002000", "step_002000_v2"):
        ckpt = f"/home/ubuntu/checkpoints/finetune/joint_corrected/{base}"
        with pytest.raises(ValueError, match="pinned BASE"):
            boundary_verdict(
                leg(good, checkpoint=ckpt),
                sampled_leg(good, checkpoint=ckpt),
                flow_leg(good, checkpoint=ckpt),
                PREFLIGHT,
            )
    with pytest.raises(ValueError, match="seeds 0-99"):
        boundary_verdict(
            leg(good, num_seeds=20),
            sampled_leg(good),
            flow_leg(good),
            PREFLIGHT,
        )
    bad = leg(good)
    bad["episodes"][0]["seed"] = 5  # duplicate -> not exactly 0-99
    with pytest.raises(ValueError, match="not exactly 0-99"):
        boundary_verdict(bad, sampled_leg(good), flow_leg(good), PREFLIGHT)
    with pytest.raises(ValueError, match="not PASS"):
        boundary_verdict(
            leg(good),
            sampled_leg(good),
            flow_leg(good),
            {"verdict": "BAND", "sampled_successes": 5},
        )


def test_frozen_anchor_sets() -> None:
    # The constants must agree with the counts the pre-reg froze.
    assert len(GREEDY_SUCCESS_SEEDS) == 7
    assert len(FLOW_SUCCESS_SEEDS) == FLOW_ANCHOR == 44
    assert math.comb(4, 2) == 6  # math.comb import is load-bearing
