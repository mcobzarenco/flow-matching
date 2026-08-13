"""Plumb-line theta->r lens fit on the pinned real wrist frames
(`sim-fit-real-lens-model` leg (a), owner-adopted 22:31Z 08-12).

The v1 wrist render assumes an ideal equidistant lens: output radius
r shows the ray at theta = r / F_DIST, distortion center at the image
center. The real module is a 130-deg wide-angle lens center-cropped
to 4:3 — its true theta->r curve and optical center are unknown. The
table planks are known-straight world lines visible in every real
frame, so the classic plumb-line constraint applies: the correct lens
model is the one whose undistortion makes plank-seam edge chains
straight. No rig time, no new captures — the 150 pinned A-half
reference frames are the calibration set.

Model (Kannala-Brandt-style odd polynomial, first coefficient pinned
so magnification at the image center matches the deployed render):

    rho   = r_px / F_DIST                      (F_DIST = 52-deg pinhole focal)
    theta = rho * (1 + k2*rho^2 + k4*rho^4)
    undistorted radius r_u = F_DIST * tan(theta)

Free parameters: (cx, cy, k2, k4). k2 = k4 = 0 with the center at the
image midpoint is exactly the deployed v1 assumption, so the fit's
improvement over that start IS the model-error readout.

Pipeline: Canny edges -> junction-split chain linking -> geometric
filters (length, border margin, sagitta/chord bound to reject the
disk rim and boat contours) -> Nelder-Mead over (cx, cy, k2, k4) on
the trimmed length-weighted mean of per-chain RMS straightness
residuals -> bootstrap over frames for parameter spread.

Outputs (outputs/sim/lens_fit/):
    wrist_lens_fit.json   fitted params, residual table, bootstrap spread
    chains_debug/*.png    accepted chains overlaid on sample frames
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.so101_sim import SO101Sim

WIDTH, HEIGHT = 640, 480
F_DIST = (HEIGHT / 2.0) / np.tan(np.deg2rad(SO101Sim.V1_CENTER_FOVY) / 2.0)

# Chain acceptance: long enough to constrain curvature, clear of the
# frame border, and gently bowed (plank seams sag ~2-4% of chord;
# the disk rim at r~150 px sags far more over any long chord).
MIN_CHAIN_PX = 180
BORDER_MARGIN = 8
MAX_SAGITTA_FRAC = 0.06
TRIM_FRAC = 0.2  # drop the worst residual chains (occluders that slip the filters)
CANNY_LO, CANNY_HI = 40, 120


def extract_chains(gray: np.ndarray) -> list[np.ndarray]:
    """Canny edge map -> candidate plumb-line point sets.

    Connected components of the edge map are treated as unordered point
    sets (the straightness residual needs no ordering). A component is
    accepted only if it reads as ONE thin, gently bowed curve: in its
    PCA frame it must fit a quadratic s(t) with small residual (rejects
    branchy merges and blobs), span a long chord, and have a bounded
    sagitta/chord ratio (rejects the disk rim and boat contours, whose
    arcs no radial lens model should be asked to flatten).
    """
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), CANNY_LO, CANNY_HI)
    mask = edges > 0
    mask[:BORDER_MARGIN, :] = False
    mask[-BORDER_MARGIN:, :] = False
    mask[:, :BORDER_MARGIN] = False
    mask[:, -BORDER_MARGIN:] = False

    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(mask)
    lab = labels[ys, xs]
    order = np.argsort(lab)
    ys, xs, lab = ys[order], xs[order], lab[order]
    bounds = np.searchsorted(lab, np.arange(1, n_labels + 1))

    chains: list[np.ndarray] = []
    for i in range(n_labels - 1):
        lo, hi = bounds[i], bounds[i + 1] if i + 1 < len(bounds) else len(ys)
        if hi - lo < MIN_CHAIN_PX:
            continue
        pts = np.stack([xs[lo:hi], ys[lo:hi]], axis=1).astype(np.float64)
        c = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        t = c @ vt[0]
        s = c @ vt[1]
        chord = t.max() - t.min()
        if chord < MIN_CHAIN_PX * 0.6:
            continue
        # quadratic thinness test in the PCA frame
        tn = t / chord
        design = np.stack([np.ones_like(tn), tn, tn**2], axis=1)
        coef, *_ = np.linalg.lstsq(design, s, rcond=None)
        resid = s - design @ coef
        if np.sqrt(np.mean(resid**2)) > 1.5:
            continue
        sagitta = abs(coef[2]) / 4.0  # peak deviation of a*u^2 over u in [-.5,.5]
        if sagitta / chord > MAX_SAGITTA_FRAC:
            continue
        chains.append(pts)
    return chains


def undistort(pts: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Map distorted pixel coords -> pinhole coords under the model."""
    cx, cy, k2, k4 = params
    d = pts - (cx, cy)
    r = np.hypot(d[:, 0], d[:, 1])
    rho = r / F_DIST
    theta = rho * (1 + k2 * rho**2 + k4 * rho**4)
    theta = np.clip(theta, 0.0, np.deg2rad(85.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(r > 1e-9, F_DIST * np.tan(theta) / r, 1.0)
    return d * scale[:, None]


def chain_rms(pts_u: np.ndarray) -> float:
    """RMS perpendicular residual of a total-least-squares line fit."""
    c = pts_u - pts_u.mean(axis=0)
    cov = c.T @ c / len(c)
    _evals, evecs = np.linalg.eigh(cov)
    n = evecs[:, 0]  # normal = eigenvector of the smaller eigenvalue
    return float(np.sqrt(np.mean((c @ n) ** 2)))


def objective(
    params: np.ndarray,
    chains: list[np.ndarray],
    lengths: np.ndarray,
) -> float:
    res = np.array([chain_rms(undistort(ch, params)) for ch in chains])
    keep = np.argsort(res)[: max(1, int(len(res) * (1 - TRIM_FRAC)))]
    w = lengths[keep]
    return float((res[keep] * w).sum() / w.sum())


def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    steps: np.ndarray,
    iters: int = 400,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float]:
    n = len(x0)
    simplex = [x0.copy()] + [x0 + steps * np.eye(n)[i] for i in range(n)]
    vals = [f(x) for x in simplex]
    for _ in range(iters):
        idx = np.argsort(vals)
        simplex = [simplex[i] for i in idx]
        vals = [vals[i] for i in idx]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex = [simplex[0] + 0.5 * (s - simplex[0]) for s in simplex]
                vals = [vals[0]] + [f(x) for x in simplex[1:]]
    best = int(np.argmin(vals))
    return simplex[best], vals[best]


def fit(
    chains: list[np.ndarray],
    free: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[np.ndarray, float]:
    """Fit with a subset of (cx, cy, k2, k4) free; the rest stay at the
    deployed v1 values (image midpoint, k2 = k4 = 0)."""
    lengths = np.array([len(c) for c in chains], dtype=np.float64)
    x0 = np.array([(WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0, 0.0, 0.0])
    steps = np.array([8.0, 8.0, 0.02, 0.02])
    free_idx = np.array(free)

    def full(x_free: np.ndarray) -> np.ndarray:
        p = x0.copy()
        p[free_idx] = x_free
        return p

    x_best, f_best = nelder_mead(
        lambda x: objective(full(x), chains, lengths),
        x0[free_idx],
        steps[free_idx],
    )
    return full(x_best), f_best


def theta_r_table(params: np.ndarray, n: int = 200) -> dict[str, list[float]]:
    """Sampled r_px(theta) for fitted vs assumed-equidistant models.

    The fitted model gives theta(r); invert numerically on a dense grid.
    """
    r_grid = np.linspace(0.0, 420.0, 4201)
    rho = r_grid / F_DIST
    _, _, k2, k4 = params
    theta_of_r = rho * (1 + k2 * rho**2 + k4 * rho**4)
    theta_s = np.linspace(0.0, float(theta_of_r[-1]), n)
    r_fit = np.interp(theta_s, theta_of_r, r_grid)
    r_eq = F_DIST * theta_s
    return {
        "theta_deg": np.rad2deg(theta_s).tolist(),
        "r_fitted_px": r_fit.tolist(),
        "r_equidistant_px": r_eq.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames",
        default="outputs/sim/ood_probe_frames/real_v2/wrist",
        help="pinned real wrist frames (A-half reference = 0000..0149)",
    )
    ap.add_argument("--out", default="outputs/sim/lens_fit")
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--bootstrap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frame_dir = Path(args.frames)
    out_dir = Path(args.out)
    (out_dir / "chains_debug").mkdir(parents=True, exist_ok=True)

    per_frame: list[list[np.ndarray]] = []
    for i in range(args.n_frames):
        img = cv2.imread(str(frame_dir / f"{i:04d}.png"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        chains = extract_chains(gray)
        per_frame.append(chains)
        if i % 30 == 0:
            dbg = img.copy()
            for ch in chains:
                for x, y in ch.astype(int):
                    cv2.circle(dbg, (x, y), 0, (0, 220, 90), -1)
            cv2.imwrite(str(out_dir / "chains_debug" / f"{i:04d}.png"), dbg)

    all_chains = [c for f in per_frame for c in f]
    n_frames_used = sum(1 for f in per_frame if f)
    print(f"chains: {len(all_chains)} from {n_frames_used}/{args.n_frames} frames")
    if len(all_chains) < 30:
        raise SystemExit("too few chains — check Canny thresholds / filters")

    lengths = np.array([len(c) for c in all_chains], dtype=np.float64)
    p_pinhole = np.array([(WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0, 0.0, 0.0])

    # anchor 1: raw bow (no undistortion at all — identity mapping)
    raw = np.array([chain_rms(c) for c in all_chains])
    keep = np.argsort(raw)[: int(len(raw) * (1 - TRIM_FRAC))]
    raw_rms = float((raw[keep] * lengths[keep]).sum() / lengths[keep].sum())
    # anchor 2: the deployed v1 assumption (equidistant, centered)
    eq_rms = objective(p_pinhole, all_chains, lengths)

    params, fit_rms = fit(all_chains)
    # decompositions: how much of the improvement is the center vs the curve
    p_center, center_rms = fit(all_chains, free=(0, 1))
    p_curve, curve_rms = fit(all_chains, free=(2, 3))
    print(f"raw bow      : {raw_rms:.3f} px")
    print(f"equidistant  : {eq_rms:.3f} px  (deployed v1 assumption)")
    print(f"center-only  : {center_rms:.3f} px  cx,cy={p_center[:2].round(1)}")
    print(f"curve-only   : {curve_rms:.3f} px  k2,k4={p_curve[2:].round(4)}")
    print(f"fitted       : {fit_rms:.3f} px  params={params}")

    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.bootstrap):
        idx = rng.choice(len(per_frame), size=len(per_frame), replace=True)
        chains_b = [c for i in idx for c in per_frame[i]]
        if len(chains_b) < 30:
            continue
        pb, _ = fit(chains_b)
        boots.append(pb)
    boots_arr = np.array(boots)

    tab = theta_r_table(params)
    th = np.array(tab["theta_deg"])

    # displacement at the frame-edge (r~240) and corner (r~400) radii:
    # at the ray angle the model places at radius r_px, how far away
    # would the equidistant assumption have drawn it
    def disp_at(r_px: float, p: np.ndarray = params) -> float:
        t = theta_r_table(p)
        r_f = np.array(t["r_fitted_px"])
        r_e = np.array(t["r_equidistant_px"])
        j = int(np.argmin(np.abs(r_f - r_px)))
        return float(r_f[j] - r_e[j])

    disp_boot = {
        r: sorted(disp_at(float(r), pb) for pb in boots_arr) for r in (240, 400)
    }
    disp_ci = {
        f"at_r{r}": [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
        for r, v in disp_boot.items()
        if len(v) >= 10
    }

    result = {
        "frames": str(frame_dir),
        "n_frames": args.n_frames,
        "n_chains": len(all_chains),
        "f_dist_px": F_DIST,
        "model": "theta = rho*(1 + k2*rho^2 + k4*rho^4), rho = r_px/F_DIST",
        "params": {
            "cx": params[0],
            "cy": params[1],
            "k2": params[2],
            "k4": params[3],
        },
        "residual_px": {
            "raw_bow": raw_rms,
            "equidistant_deployed": eq_rms,
            "center_only": center_rms,
            "curve_only": curve_rms,
            "fitted": fit_rms,
        },
        "variants": {
            "center_only": {"cx": p_center[0], "cy": p_center[1]},
            "curve_only": {"k2": p_curve[2], "k4": p_curve[3]},
        },
        "bootstrap": {
            "n": len(boots),
            "std": dict(
                zip(
                    ["cx", "cy", "k2", "k4"],
                    boots_arr.std(axis=0).tolist(),
                    strict=True,
                ),
            )
            if len(boots)
            else None,
            "params": boots_arr.tolist(),
        },
        "displacement_px": {
            "at_r240_frame_edge": disp_at(240.0),
            "at_r400_corner": disp_at(400.0),
            "bootstrap_ci95": disp_ci,
        },
        "theta_r_table": tab,
        "max_theta_seen_deg": float(th[-1]),
    }
    out = out_dir / "wrist_lens_fit.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    print(
        f"radial displacement fitted-vs-equidistant: "
        f"{disp_at(240.0):+.2f} px at r=240, {disp_at(400.0):+.2f} px at r=400",
    )


if __name__ == "__main__":
    main()
