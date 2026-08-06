"""Oracles for fontaine/scripts/sigma_draw_finalize.py (σ_draw pin).

The script also self-oracles on every invocation (O1–O4); these tests
keep the math components under check.py without touching the report
inputs.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "sigma_draw_finalize.py"
)
spec = importlib.util.spec_from_file_location("sigma_draw_finalize", SCRIPT)
assert spec is not None and spec.loader is not None
sdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdf)


def test_ls_fit_recovers_exact_triple() -> None:
    c0, v0 = 12.34, 5.678
    pts = [(n, math.sqrt(c0 + v0 / n)) for n in (1, 5, 10)]
    c, v, resid = sdf.ls_fit_c_v(pts)
    assert abs(c - c0) < 1e-10
    assert abs(v - v0) < 1e-10
    assert resid < 1e-12


@pytest.mark.parametrize(
    "maes",
    [(6.0, 6.0, 6.0), (6.0, 6.1, 6.2)],
    ids=["flat", "inverted"],
)
def test_ls_fit_clamps_nonpositive_v(maes: tuple) -> None:
    pts = list(zip((1, 5, 10), maes, strict=False))
    _, v, _ = sdf.ls_fit_c_v(pts)
    assert v == 0.0


def test_folded_mean_matches_mc() -> None:
    rng = np.random.default_rng(3)
    mu, sigma = np.array([0.0, 0.7, -2.1, 5.0]), 1.3
    draws = np.abs(mu[None, :] + rng.normal(0.0, sigma, size=(400_000, 4)))
    np.testing.assert_allclose(
        sdf.folded_mean(mu, sigma),
        draws.mean(axis=0),
        rtol=5e-3,
    )


def test_sigma_of_g_known_case() -> None:
    # g(η) = |η| ⇒ std = √(1 − 2/π)
    got = sdf.sigma_of_g(np.abs(sdf._ETA))
    assert abs(got - math.sqrt(1.0 - 2.0 / math.pi)) < 1e-6


def test_gaussian_bias_calibration_and_closed_form() -> None:
    fam = sdf.GaussianBias(6.6239, 5.3645)
    assert abs(fam.m(1) - 6.6239) < 1e-9
    assert abs(fam.m(10) - 5.3645) < 1e-9
    # E_η[g(η)] must equal m(1) — quadrature vs closed form
    e_g = float(np.sum(fam.g(sdf._ETA) * sdf._ETA_W))
    assert abs(e_g - fam.m(1)) < 1e-4


def test_kinked_calibration_matches_endpoints() -> None:
    fam = sdf.Kinked(6.6239, 5.3645)
    assert abs(fam.m(1) - 6.6239) < 1e-6
    assert abs(fam.m(10) - 5.3645) < 1e-4
