"""Oracles for fontaine/scripts/sigma_draw_direct.py (direct σ_draw
from per-draw dumps). The script also self-oracles on every invocation;
these keep the math under check.py without touching probe data.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "sigma_draw_direct.py"
)
spec = importlib.util.spec_from_file_location("sigma_draw_direct", SCRIPT)
assert spec is not None and spec.loader is not None
sdd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdd)


def test_self_oracles_pass() -> None:
    sdd.run_oracles()


def test_synthetic_recovery_tight() -> None:
    draws, truth, valid, s0 = sdd._synthetic_world(frames=4000, s0=0.5)
    s, w, _ = sdd.draw_stats(draws, truth, valid)
    s_pooled = math.sqrt(float(np.mean(np.square(s))))
    assert abs(s_pooled - s0) / s0 < 0.05
    # constant w ⇒ primary reduces to s_pooled/√F_eff exactly
    sig = sdd.sigma_primary(s, w, 16488.5)
    assert abs(sig - s_pooled / math.sqrt(16488.5)) < 1e-12


def test_unequal_weights_exact_delta_method() -> None:
    # two frames, hand-computable: s = [1, 2], w = [1, 3]
    s = np.array([1.0, 2.0])
    w = np.array([1.0, 3.0])
    # sqrt((1·1 + 9·4)/(1 + 9)) / sqrt(F)
    expected = math.sqrt(37.0 / 10.0) / math.sqrt(100.0)
    assert abs(sdd.sigma_primary(s, w, 100.0) - expected) < 1e-12


def test_degenerate_identical_draws_zero() -> None:
    rng = np.random.default_rng(1)
    one = rng.normal(size=(50, 1, 4, 6))
    draws = np.broadcast_to(one, (50, 10, 4, 6)).copy()
    truth = np.zeros((50, 4, 6))
    valid = np.ones((50, 4), dtype=bool)
    s, w, pooled = sdd.draw_stats(draws, truth, valid)
    assert sdd.sigma_primary(s, w, 16488.5) < 1e-12
    assert float(pooled.std(ddof=1)) < 1e-12


def test_f_eff_constant_weights() -> None:
    w = np.full(37, 5.0)
    assert abs(sdd.f_eff(w) - 37.0) < 1e-12
