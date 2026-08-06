"""SnapFlow pre-launch validation gate (b): E1-style drift gate
(2026-08-06 pre-registration).

Joins a --dump-predictions npz from the φ_s-extended STEP-0 checkpoint
(Heun-30, s=t, stride-7 probe subset — see the launcher) to the banked
flow-80k panel npz on the concat ``index`` and demands the mean absolute
frame-MAE drift < 0.05. Frame-MAE semantics identical to
draws_fairness.py (masked per-frame mean absolute error).

Usage: uv run python fontaine/scripts/snapflow_drift_gate.py --probe <npz>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FLOW_NPZ = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz"
GATE = 0.05


def element_mask(truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return (valid[:, :, None] & np.isfinite(truth).all(-1, keepdims=True)).repeat(
        truth.shape[2],
        axis=2,
    )


def frame_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    err = np.abs(pred - truth) * mask
    return err.sum(axis=(1, 2)) / np.maximum(mask.sum(axis=(1, 2)), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--banked", type=Path, default=Path(FLOW_NPZ))
    args = parser.parse_args()

    probe = np.load(args.probe)
    banked = np.load(args.banked)
    pred_keys = [k for k in probe.files if k.startswith("pred:bijou")]
    if len(pred_keys) != 1:
        sys.exit(f"expected exactly one bijou prediction column, got {pred_keys}")

    core = (
        probe["core"]
        if "core" in probe.files
        else np.ones(
            probe["index"].shape,
            dtype=bool,
        )
    )
    panel_pos = {int(ix): i for i, ix in enumerate(banked["index"])}
    rows = np.array([panel_pos[int(ix)] for ix in probe["index"][core]])
    truth, valid = probe["truth"][core], probe["valid"][core]
    if not (
        np.array_equal(banked["truth"][rows], truth)
        and np.array_equal(banked["valid"][rows], valid)
    ):
        sys.exit("probe rows disagree with the banked panel rows — selection drifted")

    mask = element_mask(truth, valid)
    probe_frame = frame_mae(probe[pred_keys[0]][core], truth, mask)
    banked_frame = frame_mae(banked["pred:bijou@80000"][rows], truth, mask)
    drift = float(np.abs(probe_frame - banked_frame).mean())
    print(
        f"gate (b): {len(rows)} frames, step0-extended vs banked "
        f"frame-MAE drift {drift:.5f} (gate < {GATE})",
    )
    if drift >= GATE:
        sys.exit("VALIDATION GATE (b) FAILED — do not launch, diagnose first")
    print("GATE (b) PASSED")


if __name__ == "__main__":
    main()
