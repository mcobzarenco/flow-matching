"""fontaine — σ_draw finalization (CPU-only) from the draws-chain pooled reports.

Pins the empirical σ_draw — the std of the POOLED single-draw panel
chunk_mae under an independent full-panel noise redraw — required by two
pre-registered bands before their evals open:

  - SnapFlow endpoint adopt band: 6.6232 + max(3·σ_draw, 0.15)
    (posts/2026-08-06-prereg-snapflow-distill.md)
  - stable-noise re-bank band: 6.6232 ± 3·max(0.045, σ_draw)
    (posts/2026-08-05-noise-reseed-prereg.md)

The draws chain (runs 1–5) dumped pooled report JSONs only — no per-draw
npz (the --dump-draws instrument landed later, with the fairness probe).
So σ_draw is pinned MODEL-BASED from the mean-of-N pooled MAEs at
matched solver, and the fairness probe's per-draw dump supersedes this
pin with a direct measurement if it lands larger (both dependent evals
open after the probe).

Method, frozen here:
  1. At matched solver, pooled MAE vs draws N identifies the
     draw-averageable error component. Model families (element error =
     bias b + shared-per-frame draw noise s·η, η ~ N(0,1) i.i.d. across
     frames, rank-1 within a frame — the worst case for pooled
     variance):
       gaussian_bias : b ~ N(0, β²)  ⇒ m(N) = √(2/π)·√(β² + s²/N)
       kinked        : b = ±β equal split ⇒ frame MAE = max(β, s|η|)
       pure_noise    : β = 0
  2. Each family is calibrated on (m(1), m(10)) exactly; m(5) is the
     HELD-OUT check. A family qualifies iff |m5_pred − m5_obs|/m5_obs
     < 1%.
  3. For a calibrated family, the pooled single-draw std is
     σ_draw = std_η(g(η)) / √F_eff, with g(η) the frame MAE at draw
     noise η and F_eff = (Σw)²/Σw² over the panel's per-frame valid
     weights (banked flow-80k npz).
  4. Pin = max over qualifying families x both solvers (heun-10 draws
     disperse more than heun-30; fewer-step samplers, incl. the
     SnapFlow 1-NFE endpoint, lean toward the heun-10 side).

Conservatisms (all one-directional, stated in the amendment): rank-1
within-frame correlation is maximal (real chunk correlation < 1 lowers
σ); the max over families/solvers is taken; the direct probe
measurement supersedes if larger.

Oracles (run on every invocation, before any output is written):
  O1  LS fit recovers an exact synthetic c + v/N triple to 1e-10.
  O2  degenerate flat triple ⇒ v clamped to 0, σ contribution 0.
  O3  Monte-Carlo end-to-end on the calibrated gaussian_bias family:
      finite frames x elements x redraws reproduce the closed-form
      m(N) (<0.5%) and the analytic σ_pooled (<15%).
  O4  report loader hard-asserts the posted chain numbers.

Output: reports/analysis__sigma_draw_finalization.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
STEM = "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2"
FLOW_NPZ = (
    REPORT_DIR
    / "eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz"
)
OUT = REPORT_DIR / "analysis__sigma_draw_finalization.json"

ANCHOR = 6.6232  # owner-banked heun-30 single-draw panel anchor
NAIVE_SIGMA = 0.045  # reseed pre-reg's naive floor (mainline 5.9°/√17204)
SNAPFLOW_FLOOR = 0.15
QUALIFY_RTOL = 0.01

# O4: posted chain numbers (results in now.md / draws posts). Loader dies
# on any drift — these are the amendment's inputs, not free parameters.
POSTED = {
    "heun30": {1: 6.6239, 5: 5.5235, 10: 5.3645},
    "heun10": {1: 6.8468, 10: 5.4045},
}
TAGS = {
    ("heun30", 1): "draws1_heun30",
    ("heun30", 5): "draws5_heun30",
    ("heun30", 10): "draws10_heun30",
    ("heun10", 1): "draws1_heun10",
    ("heun10", 10): "draws10_heun10",
}


def load_pooled() -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for (solver, n), tag in TAGS.items():
        path = REPORT_DIR / f"{STEM}_{tag}.json"
        with path.open() as f:
            report = json.load(f)
        rows = [s for s in report["summaries"] if s["policy"].startswith("bijou")]
        assert len(rows) == 1, f"{path.name}: expected one bijou policy row"
        mae = rows[0]["chunk_mae"]
        posted = POSTED[solver][n]
        assert abs(mae - posted) < 5e-5, (
            f"{path.name}: chunk_mae {mae} != posted {posted} — input drift"
        )
        out.setdefault(solver, {})[n] = mae
    return out


def ls_fit_c_v(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """LS fit of m(N)² = c + v/N. Returns (c, v, max rel residual on m)."""
    x = np.array([1.0 / n for n, _ in points])
    y = np.array([m * m for _, m in points])
    v, c = np.polyfit(x, y, 1)
    v = float(v)
    v = 0.0 if v < 1e-9 else v  # O2: flat/inverted data pins v at 0, loudly
    c = float(max(c, 0.0))
    pred = np.sqrt(np.clip(c + v * x, 0.0, None))
    obs = np.array([m for _, m in points])
    resid = float(np.max(np.abs(pred - obs) / obs))
    return c, v, resid


def norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.square(x)) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def folded_mean(mu: np.ndarray, sigma: float) -> np.ndarray:
    """E|N(mu, sigma²)| elementwise."""
    if sigma == 0.0:
        return np.abs(mu)
    z = mu / sigma
    return sigma * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * z * z) + mu * (
        2.0 * norm_cdf(z) - 1.0
    )


_ETA = np.linspace(-10.0, 10.0, 40001)
_ETA_W = norm_pdf(_ETA) * (_ETA[1] - _ETA[0])


def sigma_of_g(g_vals: np.ndarray) -> float:
    """std over η ~ N(0,1) of frame MAE g(η), by quadrature."""
    mean = float(np.sum(g_vals * _ETA_W))
    var = float(np.sum(np.square(g_vals - mean) * _ETA_W))
    return math.sqrt(max(var, 0.0))


class GaussianBias:
    """b ~ N(0, β²), shared frame noise s·η: m(N)² = (2/π)(β² + s²/N)."""

    name = "gaussian_bias"

    def __init__(self, m1: float, m10: float) -> None:
        tot = m1 * m1 * math.pi / 2.0  # β² + s²
        r = (m10 / m1) ** 2
        rho = (r - 0.1) / (1.0 - r)  # β²/s²
        assert rho >= 0.0, "m10 implies negative bias share — data inconsistent"
        self.s2 = tot / (rho + 1.0)
        self.b2 = tot - self.s2

    def m(self, n: int) -> float:
        return math.sqrt(2.0 / math.pi) * math.sqrt(self.b2 + self.s2 / n)

    def g(self, eta: np.ndarray) -> np.ndarray:
        return folded_mean(math.sqrt(self.s2) * eta, math.sqrt(self.b2))


class Kinked:
    """b = ±β equal split, shared frame noise: frame MAE = max(β, s|η|)."""

    name = "kinked"

    def __init__(self, m1: float, m10: float) -> None:
        # calibrate (β, s) on m(1), m(10) by bisection on rho = β/s
        def m_ratio(rho: float) -> float:
            return self._m_unit(rho, 10) / self._m_unit(rho, 1)

        lo, hi = 0.0, 50.0
        target = m10 / m1
        assert m_ratio(lo) < target < m_ratio(hi), "kinked family cannot match"
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if m_ratio(mid) < target:
                lo = mid
            else:
                hi = mid
        self.rho = 0.5 * (lo + hi)
        self.s = m1 / self._m_unit(self.rho, 1)
        self.beta = self.rho * self.s

    @staticmethod
    def _m_unit(rho: float, n: int) -> float:
        # E max(rho, |η̄|), η̄ ~ N(0, 1/n), in units of s
        sd = 1.0 / math.sqrt(n)
        vals = np.maximum(rho, sd * np.abs(_ETA))
        return float(np.sum(vals * _ETA_W))

    def m(self, n: int) -> float:
        return self.s * self._m_unit(self.rho, n)

    def g(self, eta: np.ndarray) -> np.ndarray:
        return np.maximum(self.beta, self.s * np.abs(eta))


class PureNoise:
    """β = 0: m(N) = √(2/π)·s/√N — qualifies only if ensembling scales as √N."""

    name = "pure_noise"

    def __init__(self, m1: float, m10: float) -> None:
        self.s = m1 / math.sqrt(2.0 / math.pi)

    def m(self, n: int) -> float:
        return math.sqrt(2.0 / math.pi) * self.s / math.sqrt(n)

    def g(self, eta: np.ndarray) -> np.ndarray:
        return self.s * np.abs(eta)


FAMILIES = (GaussianBias, Kinked, PureNoise)


def effective_frames() -> float:
    z = np.load(FLOW_NPZ)
    core = z["core"]
    valid = z["valid"][core]
    w = valid.reshape(valid.shape[0], -1).sum(axis=1).astype(np.float64)
    assert w.min() > 0, "zero-valid core frame in the banked npz"
    return float(w.sum() ** 2 / np.square(w).sum())


def analyze_solver(solver: str, pooled: dict[int, float], f_eff: float) -> dict:
    points = sorted(pooled.items())
    c, v, resid = ls_fit_c_v(points)
    m1, m10 = pooled[1], pooled[10]
    families = []
    for cls in FAMILIES:
        try:
            fam = cls(m1, m10)
        except AssertionError as exc:
            families.append({"family": cls.name, "calibration_failed": str(exc)})
            continue
        entry = {
            "family": cls.name,
            "m_fit": {str(n): round(fam.m(n), 4) for n, _ in points},
        }
        if 5 in pooled:
            err5 = abs(fam.m(5) - pooled[5]) / pooled[5]
            entry["m5_heldout_rel_err"] = round(err5, 5)
            entry["qualifies"] = bool(err5 < QUALIFY_RTOL)
        else:
            # two-point solver: no held-out check possible; the family
            # verdict is inherited from the 3-point solver in main().
            entry["qualifies"] = None
        sigma_g = sigma_of_g(fam.g(_ETA))
        entry["sigma_g_deg"] = round(sigma_g, 4)
        entry["sigma_draw"] = round(sigma_g / math.sqrt(f_eff), 5)
        families.append(entry)
    return {
        "pooled_chunk_mae": {str(n): m for n, m in points},
        "ls_fit": {
            "c": round(c, 4),
            "v": round(v, 4),
            "max_rel_residual": round(resid, 5),
            "naive_sigma_sqrtv_over_sqrtFeff": round(
                math.sqrt(v) / math.sqrt(f_eff),
                5,
            ),
        },
        "families": families,
    }


# ---------------------------------------------------------------- oracles


def oracle_ls_recovery() -> None:
    c0, v0 = 27.31, 17.62
    pts = [(n, math.sqrt(c0 + v0 / n)) for n in (1, 5, 10)]
    c, v, resid = ls_fit_c_v(pts)
    assert abs(c - c0) < 1e-10 and abs(v - v0) < 1e-10 and resid < 1e-12, (
        f"O1 LS recovery failed: c={c} v={v} resid={resid}"
    )


def oracle_degenerate_flat() -> None:
    pts = [(1, 6.0), (5, 6.0), (10, 6.0)]
    _, v, _ = ls_fit_c_v(pts)
    assert v == 0.0, f"O2 flat triple must pin v=0, got {v}"
    pts_inverted = [(1, 6.0), (5, 6.1), (10, 6.2)]
    _, v_inv, _ = ls_fit_c_v(pts_inverted)
    assert v_inv == 0.0, f"O2 inverted triple must clamp v=0, got {v_inv}"


def oracle_mc_end_to_end(m1: float, m10: float) -> None:
    """Finite-sample MC of the calibrated gaussian_bias world must
    reproduce the closed-form m(N) and the analytic pooled σ."""
    fam = GaussianBias(m1, m10)
    rng = np.random.default_rng(0)
    frames, elements, redraws = 4000, 96, 600
    beta, s = math.sqrt(fam.b2), math.sqrt(fam.s2)
    b = rng.normal(0.0, beta, size=(frames, elements))
    # m(N) check: mean-of-N error = b + s·η̄, η̄ shared per frame
    for n in (1, 10):
        eta_bar = rng.normal(0.0, 1.0 / math.sqrt(n), size=(redraws, frames))
        mae = np.abs(b[None, :, :] + s * eta_bar[:, :, None]).mean(axis=2)
        m_mc = float(mae.mean())
        m_cf = fam.m(n)
        assert abs(m_mc - m_cf) / m_cf < 0.005, (
            f"O3 m({n}) MC {m_mc:.4f} vs closed-form {m_cf:.4f}"
        )
    # pooled single-draw σ check vs analytic σ_g/√F
    eta = rng.normal(0.0, 1.0, size=(redraws, frames))
    pooled = np.abs(b[None, :, :] + s * eta[:, :, None]).mean(axis=(1, 2))
    sigma_mc = float(pooled.std(ddof=1))
    sigma_an = sigma_of_g(fam.g(_ETA)) / math.sqrt(frames)
    assert abs(sigma_mc - sigma_an) / sigma_an < 0.15, (
        f"O3 pooled σ MC {sigma_mc:.5f} vs analytic {sigma_an:.5f}"
    )


def main() -> None:
    pooled = load_pooled()  # O4 asserts inside
    oracle_ls_recovery()
    oracle_degenerate_flat()
    oracle_mc_end_to_end(pooled["heun30"][1], pooled["heun30"][10])

    f_eff = effective_frames()
    result = {
        "inputs": {
            "anchor": ANCHOR,
            "f_eff": round(f_eff, 1),
            "naive_sigma_floor": NAIVE_SIGMA,
        },
        "solvers": {},
    }
    qualifying_sigmas: list[tuple[str, str, float]] = []
    heun30_verdicts: dict[str, bool] = {}
    for solver, vals in pooled.items():
        analysis = analyze_solver(solver, vals, f_eff)
        result["solvers"][solver] = analysis
        for fam in analysis["families"]:
            if "sigma_draw" not in fam:
                continue
            if solver == "heun30":
                heun30_verdicts[fam["family"]] = bool(fam["qualifies"])
    for solver, analysis in result["solvers"].items():
        for fam in analysis["families"]:
            if "sigma_draw" not in fam:
                continue
            q = fam["qualifies"]
            if q is None:  # two-point solver inherits the heun30 verdict
                q = heun30_verdicts.get(fam["family"], False)
                fam["qualifies_inherited"] = q
            if q:
                qualifying_sigmas.append((solver, fam["family"], fam["sigma_draw"]))
    assert qualifying_sigmas, "no family qualifies — pin cannot be made"

    sigma = max(s for _, _, s in qualifying_sigmas)
    pin_src = max(qualifying_sigmas, key=lambda t: t[2])
    result["pin"] = {
        "sigma_draw": round(sigma, 5),
        "source": {"solver": pin_src[0], "family": pin_src[1]},
        "qualifying": [
            {"solver": s, "family": f, "sigma_draw": sd}
            for s, f, sd in sorted(qualifying_sigmas, key=lambda t: -t[2])
        ],
        "supersession": "fairness probe --dump-draws direct measurement "
        "supersedes if larger (lands before either "
        "dependent eval opens)",
    }
    result["bands"] = {
        "snapflow_endpoint": {
            "three_sigma": round(3 * sigma, 5),
            "floor": SNAPFLOW_FLOOR,
            "floor_binds": bool(3 * sigma < SNAPFLOW_FLOOR),
            "adopt_threshold": round(ANCHOR + max(3 * sigma, SNAPFLOW_FLOOR), 4),
        },
        "stable_noise_rebank": {
            "sigma_used": round(max(NAIVE_SIGMA, sigma), 5),
            "floor_binds": bool(sigma < NAIVE_SIGMA),
            "band": [
                round(ANCHOR - 3 * max(NAIVE_SIGMA, sigma), 4),
                round(ANCHOR + 3 * max(NAIVE_SIGMA, sigma), 4),
            ],
        },
    }
    with OUT.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
