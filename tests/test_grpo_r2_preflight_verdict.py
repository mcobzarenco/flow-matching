"""Oracles for the R2 preflight leg-0 verdict instrument
(``fontaine/scripts/grpo_r2_preflight_verdict.py``): the A3.3
F-premise gate mechanized. The verdict edges are pinned count by count
(ABORT <= 2 at the exact 5% tail under Bin(100, 0.07), BAND 3-6, PASS
>= 7), the exact binomial arithmetic is pinned against independent
sums, and the provenance guards (serve_head, temperature, seed window)
refuse to mint a verdict from the wrong leg's json.
"""

from __future__ import annotations

import math

import pytest

from fontaine.scripts.grpo_r2_preflight_verdict import (
    GREEDY_SUCCESS_SEEDS,
    binom_tail_below,
    mixed_group_prediction,
    preflight_verdict,
)


def payload(successes: int, **config_overrides: object) -> dict:
    config: dict[str, object] = {
        "serve_head": "ar",
        "ar_temperature": 1.0,
        "seed": 0,
        "num_seeds": 100,
        "checkpoint": "step_002000_v2",
        "commit": "deadbee",
    }
    config.update(config_overrides)
    return {
        "config": config,
        "episodes": [
            {"seed": s, "success_tick": 120 if s < successes else None}
            for s in range(100)
        ],
    }


def test_binomial_tail_exact() -> None:
    # Independent pin: P(X=0) = 0.93^100, ratio recursion for the rest.
    p0 = 0.93**100
    expect = p0
    term = p0
    for k in range(1, 4):
        term *= (100 - k + 1) / k * (0.07 / 0.93)
        expect += term
        assert binom_tail_below(k, 100, 0.07) == pytest.approx(expect, rel=1e-12)
    assert binom_tail_below(100, 100, 0.07) == pytest.approx(1.0, rel=1e-12)
    # The 5% material line falls between counts 2 and 3.
    assert binom_tail_below(2, 100, 0.07) < 0.05 < binom_tail_below(3, 100, 0.07)


def test_verdict_edges() -> None:
    assert preflight_verdict(payload(0))["verdict"] == "ABORT"
    assert preflight_verdict(payload(2))["verdict"] == "ABORT"
    assert preflight_verdict(payload(3))["verdict"] == "BAND"
    assert preflight_verdict(payload(6))["verdict"] == "BAND"
    assert preflight_verdict(payload(7))["verdict"] == "PASS"
    assert preflight_verdict(payload(23))["verdict"] == "PASS"
    # The competence floor is recorded only on PASS.
    assert preflight_verdict(payload(6))["training_decode_floor"] is None
    assert preflight_verdict(payload(23))["training_decode_floor"] == 0.23


def test_verdict_receipts() -> None:
    result = preflight_verdict(payload(23))
    assert result["sampled_successes"] == 23
    assert result["greedy_anchor"] == 7
    assert result["success_seeds"] == list(range(23))
    assert result["greedy_overlap"] == []  # synthetic seeds 0-22 miss the anchors
    assert result["greedy_success_seeds"] == list(GREEDY_SUCCESS_SEEDS)
    # §1 arithmetic at p = 0.23, 8 draws.
    assert result["predicted_mixed_groups_frac"] == pytest.approx(
        1 - 0.77**8 - 0.23**8,
        abs=5e-5,
    )
    # A3.2's quoted prediction: ~44% mixed at the greedy floor 0.07.
    assert mixed_group_prediction(0.07) == pytest.approx(0.44, abs=0.005)


def test_provenance_guards() -> None:
    with pytest.raises(ValueError, match="serve-head ar"):
        preflight_verdict(payload(7, serve_head="flow"))
    with pytest.raises(ValueError, match="greedy json"):
        preflight_verdict(payload(7, ar_temperature=None))
    with pytest.raises(ValueError, match=r"T=1\.0"):
        preflight_verdict(payload(7, ar_temperature=0.7))
    with pytest.raises(ValueError, match="seeds 0-99"):
        preflight_verdict(payload(7, seed=1000))
    with pytest.raises(ValueError, match="seeds 0-99"):
        preflight_verdict(payload(7, num_seeds=20))
    bad = payload(7)
    bad["episodes"][0]["seed"] = 5  # duplicate -> not exactly 0-99
    with pytest.raises(ValueError, match="not exactly 0-99"):
        preflight_verdict(bad)


def test_alpha_line_is_conservative_against_rounding() -> None:
    # Guard the frozen constants themselves: the ABORT set under the
    # exact tail at alpha 0.05 is {0, 1, 2} and nothing else.
    aborts = [k for k in range(101) if binom_tail_below(k, 100, 0.07) < 0.05]
    assert aborts == [0, 1, 2]
    assert math.comb(100, 0) == 1  # math.comb import is load-bearing
