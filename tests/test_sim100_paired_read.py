"""Oracles for the paired per-seed sim100 read
(fontaine/scripts/sim100_paired_read.py): seed alignment, the McNemar
table + exact p, and the count/progress bootstrap scaling, pinned on
hand-checkable inputs."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from fontaine.scripts.sim100_paired_read import (
    load_paired,
    mcnemar_exact_p,
    mcnemar_table,
    paired_read,
)


def _write_arm(
    path: Path,
    seeds: Sequence[int],
    successes: Sequence[int],
    progress: Sequence[float],
) -> Path:
    episodes = [
        {
            "seed": int(s),
            "success_tick": 100 if ok else None,
            "progress_final_cm": float(p),
            "reset_strikes": 0,
        }
        for s, ok, p in zip(seeds, successes, progress, strict=True)
    ]
    path.write_text(
        json.dumps({"config": {"checkpoint": path.stem}, "episodes": episodes}),
    )
    return path


def test_alignment_by_seed_not_order(tmp_path: Path) -> None:
    # Arm B stores the same seeds shuffled; deltas must still pair by
    # seed value.
    a = _write_arm(
        tmp_path / "a.json",
        [0, 1, 2, 3],
        [1, 1, 0, 0],
        [4.0, 3.0, 2.0, 1.0],
    )
    b = _write_arm(
        tmp_path / "b.json",
        [3, 0, 2, 1],
        [0, 1, 0, 0],
        [0.5, 1.0, 2.0, 3.0],
    )
    arm_a, arm_b = load_paired(a, b)
    read = paired_read(arm_a, arm_b)
    # seed-aligned progress deltas: 3.0, 0.0, 0.0, 0.5 -> mean 0.875
    assert read["progress"]["mean_delta_cm"] == 0.875
    assert read["progress"]["tie_rate"] == 0.5
    assert read["success"]["count_delta"] == 1
    assert read["discordant"]["a_only"] == 1
    assert read["discordant"]["b_only"] == 0


def test_seed_set_mismatch_refused(tmp_path: Path) -> None:
    a = _write_arm(tmp_path / "a.json", [0, 1, 2], [1, 0, 0], [1.0, 1.0, 1.0])
    b = _write_arm(tmp_path / "b.json", [0, 1, 9], [1, 0, 0], [1.0, 1.0, 1.0])
    with pytest.raises(AssertionError, match="seed sets differ"):
        load_paired(a, b)


def test_duplicate_seeds_refused(tmp_path: Path) -> None:
    a = _write_arm(tmp_path / "a.json", [0, 1, 1], [1, 0, 0], [1.0, 1.0, 1.0])
    with pytest.raises(AssertionError, match="duplicate seeds"):
        load_paired(a, a)


def test_mcnemar_table_hand_counts() -> None:
    succ_a = np.array([True, True, False, False, True])
    succ_b = np.array([True, False, True, False, False])
    table = mcnemar_table(succ_a, succ_b)
    assert table == {"both_succeed": 1, "a_only": 2, "b_only": 1, "both_fail": 1}


def test_mcnemar_exact_p_hand_values() -> None:
    # n10=3, n01=1: n=4, k=3 -> tail is (C(4,3)+C(4,4))/16 = 5/16,
    # which doubles to 10/16.
    assert mcnemar_exact_p(3, 1) == pytest.approx(0.625)
    # No discordant seeds: p = 1 by convention.
    assert mcnemar_exact_p(0, 0) == 1.0
    # Symmetric discordance caps at 1.
    assert mcnemar_exact_p(2, 2) == 1.0
    # One-sided extreme: n10=8, n01=0 -> 2 * 1/256 = 1/128.
    assert mcnemar_exact_p(8, 0) == pytest.approx(2 / 256)


def test_count_ci_scales_with_n_and_is_deterministic(tmp_path: Path) -> None:
    # 40 seeds, A succeeds on 30, B on 10, all discordant pairs
    # A-favoring: the count-delta CI must bracket +20 and exclude zero.
    n = 40
    succ_a = [1] * 30 + [0] * 10
    succ_b = [0] * 30 + [1] * 10
    prog = list(np.linspace(0.0, 5.0, n))
    a = _write_arm(tmp_path / "a.json", range(n), succ_a, prog)
    b = _write_arm(tmp_path / "b.json", range(n), succ_b, prog)
    read = paired_read(*load_paired(a, b))
    low, high = read["success"]["count_delta_ci95"]
    assert low < 20 < high
    assert read["success"]["ci_excludes_zero"]
    assert read["progress"]["mean_delta_cm"] == 0.0
    # Deterministic: same inputs, same CI (seeded bootstrap).
    again = paired_read(*load_paired(a, b))
    assert again["success"]["count_delta_ci95"] == [low, high]


def test_retro_shape_probe_vs_disc1000_fixture(tmp_path: Path) -> None:
    # Miniature of the registered retro read: clear A-favoring split
    # with a null progress arm -> success CI excludes zero, progress CI
    # does not.
    n = 20
    succ_a = [1] * 9 + [0] * 11
    succ_b = [1] * 2 + [0] * 18
    rng = np.random.default_rng(3)
    prog_a = rng.normal(0.0, 1.0, n).tolist()
    a = _write_arm(tmp_path / "a.json", range(n), succ_a, prog_a)
    b = _write_arm(tmp_path / "b.json", range(n), succ_b, prog_a)  # identical progress
    read = paired_read(*load_paired(a, b))
    assert read["success"]["count_delta"] == 7
    assert read["success"]["ci_excludes_zero"]
    assert read["discordant"]["a_only"] == 7
    assert read["discordant"]["b_only"] == 0
    assert not read["progress"]["ci_excludes_zero"]
    assert read["progress"]["tie_rate"] == 1.0
