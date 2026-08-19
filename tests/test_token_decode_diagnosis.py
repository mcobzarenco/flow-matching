"""Oracles for the token-decode diagnosis instrument (CPU, synthetic
episodes) — the funnel, the pinch-fail taxonomy, and carry speed are
the numbers the R2 band recommendation rests on."""

from __future__ import annotations

from typing import Any

from fontaine.scripts.token_decode_diagnosis import (
    carry_speeds,
    funnel,
    pinch_fail_taxonomy,
    reach_envelope,
)


def ep(
    seed: int,
    grip: list[int],
    dist: list[float],
    success_tick: int | None = None,
    initial: float | None = None,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "grip": grip,
        "distance_cm": dist,
        "success_tick": success_tick,
        "initial_cm": dist[0] if initial is None else initial,
        "final_cm": dist[-1],
        "min_cm": min(dist),
    }


def test_funnel_counts_and_conversions() -> None:
    eps = [
        ep(0, [0, 0, 0], [10.0, 10.0, 10.0]),  # untouched
        ep(1, [0, 1, 0], [10.0, 10.0, 12.0]),  # touch, knocked
        ep(2, [0, 3, 3], [10.0, 8.0, 6.0]),  # pinch, fail
        ep(3, [0, 3, 3], [10.0, 5.0, 1.0], success_tick=2),  # success
    ]
    f = funnel(eps)
    assert (f["touch"], f["pinch"], f["success"]) == (3, 2, 1)
    assert f["knock_aways_gt1cm"] == 1
    assert f["success_seeds"] == [3]
    assert f["touch_to_pinch"] == round(2 / 3, 3)
    assert f["pinch_to_success"] == 0.5


def test_pinch_fail_taxonomy_three_classes() -> None:
    away = ep(0, [0, 3, 3, 0, 0, 0], [10.0, 10.0, 12.0, 12.0, 12.0, 12.0])
    stalled = ep(1, [0, 3, 3, 0, 0, 0], [10.0, 10.0, 7.0, 7.0, 7.0, 7.0])
    timeout = ep(2, [0, 3, 3, 3, 3, 3], [10.0, 10.0, 8.0, 7.0, 6.0, 6.0])
    succ = ep(3, [0, 3, 3, 3, 3, 3], [10.0, 6.0, 1.0, 1.0, 1.0, 1.0], success_tick=2)
    t = pinch_fail_taxonomy([away, stalled, timeout, succ])
    assert t["n"] == 3  # the success is excluded
    assert t["classes"] == {"wrong_way": 1, "stalled_carry": 1, "timeout_holding": 1}


def test_carry_speed_is_cm_per_second_over_pinch_span() -> None:
    # pinched ticks 0..60 (span 60 ticks = 2 s at 30 Hz), 4 cm recovered
    grip = [3] * 61 + [0] * 10
    dist = [10.0 - 4.0 * min(i, 60) / 60 for i in range(71)]
    cs = carry_speeds([ep(0, grip, dist)], min_ticks=30)
    assert cs["n"] == 1
    assert abs(cs["rows"][0]["cm_per_s"] - 2.0) < 1e-6
    # below the min-ticks floor -> excluded
    assert carry_speeds([ep(1, [3] * 5 + [0] * 66, dist)], min_ticks=30)["n"] == 0


def test_reach_envelope_bins_touch_and_success() -> None:
    close = ep(0, [0, 3], [7.0, 1.0], success_tick=1)
    far = ep(1, [0, 0], [12.0, 12.0])
    env = reach_envelope([close, far])
    by_bin = {tuple(b["bin"]): b for b in env}
    assert by_bin[(6.0, 8.0)] == {"bin": [6.0, 8.0], "n": 1, "touch": 1, "success": 1}
    assert by_bin[(11.0, 13.0)] == {
        "bin": [11.0, 13.0],
        "n": 1,
        "touch": 0,
        "success": 0,
    }
