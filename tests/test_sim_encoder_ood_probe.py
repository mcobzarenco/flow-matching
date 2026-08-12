"""Oracles for the encoder OOD probe's rank math (fontaine/scripts/
sim_encoder_ood_probe.py): AUROC is Mann-Whitney U with ties at half
credit — the probe's headline read must not drift."""

import numpy as np

from fontaine.scripts.sim_encoder_ood_probe import auroc


def test_perfect_separation() -> None:
    assert auroc(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == 1.0


def test_reversed_separation() -> None:
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0


def test_identical_distributions_are_chance() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert auroc(values, values.copy()) == 0.5


def test_ties_get_half_credit() -> None:
    # Pairs: (1,1) tie 0.5; (1,0), (2,1), (2,0) wins -> 3.5 over 2*2.
    assert auroc(np.array([1.0, 2.0]), np.array([1.0, 0.0])) == 0.875


def test_matches_naive_pair_count() -> None:
    rng = np.random.default_rng(7)
    positive, negative = rng.normal(0.5, 1, 40), rng.normal(0, 1, 30)
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative
    )
    assert abs(auroc(positive, negative) - wins / (40 * 30)) < 1e-12
