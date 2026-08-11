"""Oracles for the 100-seed sim eval reads
(fontaine/scripts/sim100_reads.py): the ordering read's pair logic and
the bootstrap/Spearman arithmetic, pinned on hand-checkable inputs."""

import numpy as np

from fontaine.scripts.sim100_reads import (
    PANEL_MAE,
    bootstrap_ci,
    ordering_read,
    spearman_rho,
)


def test_ordering_all_correct() -> None:
    # Sim progress perfectly anti-ordered with panel MAE -> every pair
    # correct, rho = +1, no violations.
    progress = {r: -PANEL_MAE[r] for r in PANEL_MAE}
    read = ordering_read(progress)
    assert read["expectation_met"]
    assert read["gated_pairs_total"] == 5  # (55k,60k) gap 0.0487 not gated
    assert read["gated_pairs_correct"] == 5
    assert read["max_violation_panel_gap"] == 0.0
    assert read["spearman_rho_progress_vs_neg_panel"] == 1.0


def test_ordering_55_60_swap_is_record_only() -> None:
    # 55k beating 60k in sim swaps the one non-gated pair: expectation
    # still met.
    progress = {"er15k": 1.0, "er35k": 2.0, "er55k": 3.1, "er60k": 3.0}
    read = ordering_read(progress)
    assert read["expectation_met"]
    swapped = next(p for p in read["pairs"] if p["pair"] == ["er55k", "er60k"])
    assert not swapped["gated"] and not swapped["sim_correct"]


def test_ordering_gated_violation_fails() -> None:
    # 15k beating 35k in sim is a gated misrank: expectation fails and
    # the violation weight is that pair's panel gap.
    progress = {"er15k": 2.5, "er35k": 2.0, "er55k": 3.0, "er60k": 4.0}
    read = ordering_read(progress)
    assert not read["expectation_met"]
    assert read["max_violation_panel_gap"] == round(
        PANEL_MAE["er15k"] - PANEL_MAE["er35k"],
        4,
    )


def test_spearman_exact_values() -> None:
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(2.0, 1.0, size=100)
    low, high = bootstrap_ci(deltas)
    assert low < deltas.mean() < high
    assert (low, high) == bootstrap_ci(deltas)  # seeded resamples
    # A clearly positive effect at n=100 excludes zero.
    assert low > 0
