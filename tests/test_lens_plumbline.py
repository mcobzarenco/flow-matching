"""Oracles for the plumb-line lens-fit math core
(fontaine/scripts/fit_lens_plumbline.py).

Synthetic recovery: project known-straight pinhole lines through a
known theta->r model onto the distorted image, feed the resulting
point sets to the fitter, and require it to recover the model — not
by raw parameters (center and curve trade off under limited line
orientations) but by the readout that matters: the radial placement
of rays across the field.
"""

from __future__ import annotations

import numpy as np
import pytest

from fontaine.scripts.fit_lens_plumbline import (
    F_DIST,
    chain_rms,
    fit,
    undistort,
)

RNG = np.random.default_rng(7)


def distort(pts_u: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Forward model: pinhole coords (relative to center) -> distorted
    pixel coords. Numeric inversion of `undistort` on a dense radius
    grid."""
    cx, cy, k2, k4 = params
    r_grid = np.linspace(0.0, 500.0, 20001)
    rho = r_grid / F_DIST
    theta_of_r = rho * (1 + k2 * rho**2 + k4 * rho**4)
    r_u_of_r = F_DIST * np.tan(theta_of_r)
    r_u = np.hypot(pts_u[:, 0], pts_u[:, 1])
    r_d = np.interp(r_u, r_u_of_r, r_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(r_u > 1e-9, r_d / r_u, 1.0)
    return pts_u * scale[:, None] + (cx, cy)


def synth_chains(params: np.ndarray, noise: float = 0.15) -> list[np.ndarray]:
    """Points along straight pinhole lines at varied offsets and
    orientations, pushed through the forward model."""
    chains = []
    for ang in np.linspace(0, np.pi, 12, endpoint=False):
        for off in (-160.0, -80.0, 0.0, 80.0, 160.0):
            n = np.array([np.cos(ang), np.sin(ang)])
            d = np.array([-n[1], n[0]])
            t = np.linspace(-260.0, 260.0, 240)
            pts_u = off * n + t[:, None] * d
            pts_d = distort(pts_u, params)
            keep = (
                (pts_d[:, 0] > 10)
                & (pts_d[:, 0] < 630)
                & (pts_d[:, 1] > 10)
                & (pts_d[:, 1] < 470)
            )
            if keep.sum() < 150:
                continue
            chains.append(pts_d[keep] + RNG.normal(0, noise, (int(keep.sum()), 2)))
    return chains


def radial_curve(params: np.ndarray) -> np.ndarray:
    """r_d(theta) sampled on a fixed theta grid, for model comparison."""
    r_grid = np.linspace(0.0, 420.0, 8401)
    rho = r_grid / F_DIST
    theta_of_r = rho * (1 + params[2] * rho**2 + params[3] * rho**4)
    theta_s = np.linspace(0.0, np.deg2rad(45.0), 100)
    return np.interp(theta_s, theta_of_r, r_grid)


def test_undistort_straightens_known_model() -> None:
    truth = np.array([310.0, 250.0, 0.04, 0.02])
    chains = synth_chains(truth, noise=0.0)
    assert len(chains) >= 40
    bowed = np.array([chain_rms(c - truth[:2]) for c in chains])
    straight = np.array([chain_rms(undistort(c, truth)) for c in chains])
    assert bowed.max() > 1.0  # the synthetic distortion visibly bows lines
    assert straight.max() < 0.05  # the true model flattens them exactly


def test_fit_recovers_synthetic_model() -> None:
    truth = np.array([308.0, 252.0, 0.035, 0.025])
    chains = synth_chains(truth)
    params, resid = fit(chains)
    assert resid < 0.3
    # the readout that matters: ray placement across the field agrees
    # with the truth to sub-pixel everywhere up to 45 deg
    err = np.abs(radial_curve(params) - radial_curve(truth))
    assert err.max() < 1.0
    assert abs(params[0] - truth[0]) < 3.0
    assert abs(params[1] - truth[1]) < 3.0


def test_fit_does_not_invent_distortion() -> None:
    truth = np.array([319.5, 239.5, 0.0, 0.0])
    chains = synth_chains(truth)
    params, _ = fit(chains)
    err = np.abs(radial_curve(params) - radial_curve(truth))
    assert err.max() < 0.8


def test_chain_rms_zero_on_exact_line() -> None:
    t = np.linspace(0, 100, 50)
    pts = np.stack([t, 3.0 + 0.5 * t], axis=1)
    assert chain_rms(pts) == pytest.approx(0.0, abs=1e-9)
