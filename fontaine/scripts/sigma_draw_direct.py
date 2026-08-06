"""Direct σ_draw measurement from the fairness probe's per-draw dumps —
the pre-declared supersession check on the model-based 0.0159 pin
(σ_draw finalization amendment, 2026-08-06: "the fairness probe's
--dump-draws npz gives the direct measurement of the same quantity...
If the direct estimate exceeds this pin, it supersedes; if it exceeds
0.045/0.05, the floor-binds verdicts are re-opened").

Quantity (identical to the amendment's definition): σ_draw = the draw-
noise std of the SINGLE-draw full-panel pooled chunk_mae,
std_η(frame-MAE)/√F_eff with F_eff = (Σw)²/Σw² over the full panel's
per-frame valid counts (16,488.5 posted).

Two estimators, both from the probe stack [frames, 10, chunk, dim]:

1. PRIMARY (frame-level, exact delta-method form): per frame i the
   across-draw std s_i (ddof=1) of the frame MAE; per frame valid count
   w_i. The pooled panel MAE is Σw·m/Σw, so under independent per-frame
   draws Var = Σw²s²/(Σw)². With the probe as an unbiased stride-7
   sample of the panel:
       σ_direct = sqrt(Σw²s²/Σw²) / sqrt(F_eff_panel).
   (s_i² is ddof=1-unbiased, so the pooled σ² is unbiased; the pooled
   estimate averages 2,458 frames → tight.)
2. CROSS-CHECK (pooled-level, assumption-light, n=10 noisy): the 10
   per-draw pooled probe MAEs directly realize the pooled number's draw
   noise at probe size; std (ddof=1) scaled by
   sqrt(F_eff_probe/F_eff_panel). Quoted with its χ²₉ 95% band
   (multipliers 0.6878 / 1.8256) — consistency check, not the pin.

Input-drift oracles (die loud): mean-of-draws pooled MAE must reproduce
the probe report JSON's bijou@80000_draws10 chunk_mae (<5e-3);
F_eff_panel must reproduce the posted 16,488.5; probe shape must be the
frozen plan's 2,458 x 10. Self-oracles O1–O4 run on every invocation.

Pure CPU. JSON out: reports/analysis__sigma_draw_direct.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draws_fairness import element_mask, frame_mae, pooled_mae

PROBE_STEM = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__"
    "panel_curated_v0_k4l2_drawsprobe_s7_draws10_heun30"
)
FLOW_NPZ = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__"
    "panel_k4l2_heun30.npz"
)
REPORT_KEY = "bijou@80000_draws10"
PIN = 0.0159  # model-based, heun-10 max (the finalization amendment)
REBANK_FLOOR = 0.045  # stable-noise re-bank band σ floor
ADOPT_FLOOR = 0.05  # SnapFlow adopt band 3σ floor equivalent (0.15/3)
F_EFF_POSTED = 16488.5
PROBE_FRAMES, PROBE_DRAWS = 2458, 10
# chi-squared(9 dof) 95% multipliers for a std from 10 samples
CHI2_9_LO, CHI2_9_HI = 0.6878, 1.8256


def draw_stats(
    draws: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (s_i, w_i) + per-draw pooled MAEs M_d for a
    [frames, N, chunk, dim] stack, frame-MAE semantics identical to the
    fairness reads (valid-element-weighted)."""
    mask = element_mask(truth, valid)
    per_draw_frame = np.stack(
        [frame_mae(draws[:, d], truth, mask) for d in range(draws.shape[1])],
    )  # [draws, frames]
    s = per_draw_frame.std(axis=0, ddof=1)
    w = mask.sum(axis=(1, 2)).astype(np.float64)
    pooled = np.array(
        [pooled_mae(draws[:, d], truth, mask) for d in range(draws.shape[1])],
    )
    return s, w, pooled


def sigma_primary(s: np.ndarray, w: np.ndarray, f_eff_panel: float) -> float:
    """sqrt(Σw²s²/Σw²)/√F_eff — exact delta-method σ of the pooled MAE
    when the probe frames sample the panel."""
    std_eta = math.sqrt(float(np.sum(np.square(w * s)) / np.sum(np.square(w))))
    return std_eta / math.sqrt(f_eff_panel)


def f_eff(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / np.square(w).sum())


# ---------------------------------------------------------------- oracles


def _synthetic_world(
    frames: int = 2000,
    s0: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rank-1 shared-noise world with analytically known σ: every element
    of frame i, draw d equals 100 + b_i + s0·η_{i,d} (offset keeps |·|
    affine), so frame MAE = 100 + b_i + s0·η exactly and its across-draw
    std is s0."""
    rng = np.random.default_rng(0)
    b = rng.normal(0.0, 2.0, size=frames)
    eta = rng.normal(0.0, 1.0, size=(frames, PROBE_DRAWS))
    vals = 100.0 + b[:, None] + s0 * eta  # [frames, draws]
    draws = np.broadcast_to(vals[:, :, None, None], (frames, PROBE_DRAWS, 4, 6)).copy()
    truth = np.zeros((frames, 4, 6))
    valid = np.ones((frames, 4), dtype=bool)
    return draws, truth, valid, s0


def run_oracles() -> None:
    # O1 — synthetic recovery: pooled s must recover s0 within 5%, and
    # the primary formula must equal s_pooled/√F_eff under constant w.
    draws, truth, valid, s0 = _synthetic_world()
    s, w, pooled = draw_stats(draws, truth, valid)
    s_pooled = math.sqrt(float(np.mean(np.square(s))))
    assert abs(s_pooled - s0) / s0 < 0.05, f"O1 recovery: {s_pooled} vs {s0}"
    sig = sigma_primary(s, w, F_EFF_POSTED)
    assert abs(sig - s_pooled / math.sqrt(F_EFF_POSTED)) < 1e-12, "O1 formula"
    # O4 — pooled-level cross-check agrees with the primary within the
    # n=10 χ² band on the same synthetic world.
    scaled = float(pooled.std(ddof=1)) * math.sqrt(f_eff(w) / F_EFF_POSTED)
    assert 0.5 < scaled / sig < 2.0, f"O4 consistency: {scaled} vs {sig}"
    # O2 — degenerate identical draws ⇒ both estimators zero (np.std's
    # pairwise-summation mean leaves ~1e-14 roundoff on identical
    # values, so the bound is 1e-12, not exact).
    draws0 = np.broadcast_to(
        draws[:, :1],
        draws.shape,
    ).copy()
    s2, w2, pooled2 = draw_stats(draws0, truth, valid)
    assert sigma_primary(s2, w2, F_EFF_POSTED) < 1e-12, "O2 primary nonzero"
    assert float(pooled2.std(ddof=1)) < 1e-12, "O2 pooled nonzero"
    # O3 — hand case: one frame, two draws with frame MAEs 1 and 3 ⇒
    # s = √2 (ddof=1) exactly.
    d = np.zeros((1, 2, 1, 1))
    d[0, 0], d[0, 1] = 1.0, 3.0
    s3, _, _ = draw_stats(d, np.zeros((1, 1, 1)), np.ones((1, 1), dtype=bool))
    assert abs(float(s3[0]) - math.sqrt(2.0)) < 1e-12, "O3 hand case"


# ------------------------------------------------------------------ main


def main() -> None:
    run_oracles()
    print("self-oracles O1–O4: OK")

    probe = np.load(PROBE_STEM + ".npz")
    core = probe["core"]
    draws = probe["draws"][core].astype(np.float64)
    truth, valid = probe["truth"][core], probe["valid"][core]
    assert draws.shape[:2] == (PROBE_FRAMES, PROBE_DRAWS), (
        f"probe shape {draws.shape[:2]} != frozen plan ({PROBE_FRAMES}, {PROBE_DRAWS})"
    )

    # Input drift: the dumped stack's mean-of-draws must reproduce the
    # probe report's pooled chunk_mae.
    report = json.loads(Path(PROBE_STEM + ".json").read_text())
    (entry,) = [s for s in report["summaries"] if s["policy"] == REPORT_KEY]
    mask = element_mask(truth, valid)
    mean_pooled = pooled_mae(draws.mean(axis=1), truth, mask)
    assert abs(mean_pooled - entry["chunk_mae"]) < 5e-3, (
        f"npz/report drift: {mean_pooled} vs {entry['chunk_mae']}"
    )

    # F_eff from the banked full-panel npz, asserted against the posted
    # value (same computation as sigma_draw_finalize.effective_frames).
    z = np.load(FLOW_NPZ)
    pv = z["valid"][z["core"]]
    w_panel = pv.reshape(pv.shape[0], -1).sum(axis=1).astype(np.float64)
    f_eff_panel = f_eff(w_panel)
    assert abs(f_eff_panel - F_EFF_POSTED) < 1.0, f"F_eff drift: {f_eff_panel}"

    s, w, pooled = draw_stats(draws, truth, valid)
    direct = sigma_primary(s, w, f_eff_panel)
    f_eff_probe = f_eff(w)
    pooled_std = float(pooled.std(ddof=1))
    scaled = pooled_std * math.sqrt(f_eff_probe / f_eff_panel)

    supersedes = direct > PIN
    sigma_final = max(direct, PIN)
    out = {
        "sigma_draw_direct": round(direct, 5),
        "sigma_draw_pin_model": PIN,
        "supersedes_pin": bool(supersedes),
        "sigma_draw_final": round(sigma_final, 5),
        "reopen_floors": bool(direct > REBANK_FLOOR or direct > ADOPT_FLOOR),
        "floors": {"rebank_sigma": REBANK_FLOOR, "adopt_sigma": ADOPT_FLOOR},
        "frame_level": {
            "pooled_std_eta_deg": round(direct * math.sqrt(f_eff_panel), 4),
            "s_frame_median": round(float(np.median(s)), 4),
            "s_frame_p90": round(float(np.quantile(s, 0.9)), 4),
        },
        "pooled_level_crosscheck": {
            "per_draw_pooled_mae": [round(float(m), 4) for m in pooled],
            "std_at_probe_size": round(pooled_std, 5),
            "scaled_to_panel": round(scaled, 5),
            "chi2_95_band": [
                round(scaled * CHI2_9_LO, 5),
                round(scaled * CHI2_9_HI, 5),
            ],
            "primary_inside_band": bool(
                scaled * CHI2_9_LO <= direct <= scaled * CHI2_9_HI,
            ),
        },
        "f_eff": {"panel": round(f_eff_panel, 1), "probe": round(f_eff_probe, 1)},
        "inputs": {
            "probe_npz": PROBE_STEM + ".npz",
            "report_chunk_mae_mean_of_draws": round(entry["chunk_mae"], 4),
        },
    }
    out_path = Path("reports/analysis__sigma_draw_direct.json")
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    verdict = (
        "SUPERSEDES the 0.0159 pin"
        if supersedes
        else "pin stands (direct <= model-based pin)"
    )
    print(f"σ_draw direct = {direct:.5f} — {verdict}")
    if out["reopen_floors"]:
        print("RE-OPEN: direct estimate exceeds a floor — amendment required")


if __name__ == "__main__":
    main()
