"""The three pre-declared mode-averaging fairness reads (owner 21:49Z
2026-08-05): is the panel's MAE punishing flow for sampling valid modes?

Inputs:
  --draws reports/<probe>.npz   bijou.eval --dump-draws output
                                ([frames, draws, chunk, dim] pre-average
                                stacks + truth/valid/index identity)
  plus the banked full-panel npzs (AR-100k, flow-80k single-draw) for
  the paired deficit column, joined on the corpus concat `index`
  (the flow_vs_ar_paired.py join convention — both were dumped under
  the same corpus selection).

Reads (decision semantics in the pre-reg amendment; all pooling is
valid-element-weighted, matching the report's chunk_mae):
  1. mean-of-draws MAE — ensembling manufactures the mode-averaged
     predictor; if it closes the flow-vs-AR gap, the deficit was
     punished dispersion, not worse modeling.
  2. best-of-N MAE — oracle mode-match bound: per frame the best draw
     by that frame's chunk MAE (first_mae selected independently);
     how much of the deficit is 'sampled a different valid mode'.
  3. dispersion-conditioned deficit — per-frame draw spread (masked
     mean over elements of the across-draw std) vs the paired
     flow-minus-AR deficit; the unfair-penalty signature is deficit
     concentrating in the high-dispersion quartiles.

--validate runs the same code path in the degenerate draws=1 case
(the banked flow-80k npz as a 1-draw stack): reads 1 and 2 must
reproduce the 6.6232 anchor and every dispersion must be exactly 0 —
the CPU oracle for the read implementations, runnable before any
probe data exists.

Pure CPU, read-only on the npzs. JSON out:
reports/analysis__draws_fairness_k4l2.json (or _validate.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AR_NPZ = "reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz"
FLOW_NPZ = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_k4l2_heun30.npz"
FLOW_ANCHOR = 6.6232  # owner's 12:20Z box eval, chunk_mae
FLOW_FIRST_ANCHOR = 1.9331


def element_mask(truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[frames, chunk, dim] bool — the paired-analysis masking."""
    return (valid[:, :, None] & np.isfinite(truth).all(-1, keepdims=True)).repeat(
        truth.shape[2],
        axis=2,
    )


def pooled_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(pred - truth)[mask].mean())


def frame_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    err = np.abs(pred - truth) * mask
    return err.sum(axis=(1, 2)) / np.maximum(mask.sum(axis=(1, 2)), 1)


def step_curve(err: np.ndarray, valid: np.ndarray) -> list[float]:
    """Per-horizon-step pooled MAE, the paired-analysis convention."""
    wv = valid.astype(np.float64)
    num = (err.sum(axis=2) * wv).sum(axis=0)
    den = wv.sum(axis=0) * err.shape[2]
    return (num / np.maximum(den, 1)).tolist()


def fairness_reads(
    draws: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    """The three reads on a [frames, N, chunk, dim] stack. Selection and
    pooling exactly as pre-declared; every number valid-element-weighted."""
    n_frames, n_draws = draws.shape[:2]
    mask = element_mask(truth, valid)

    # Read 1 — the ensemble (mean over draws) and the per-draw baseline.
    mean_pred = draws.mean(axis=1)
    per_draw_pooled = [pooled_mae(draws[:, d], truth, mask) for d in range(n_draws)]
    per_draw_frame = np.stack(
        [frame_mae(draws[:, d], truth, mask) for d in range(n_draws)],
    )  # [draws, frames]

    # Read 2 — oracle best draw per frame, chunk metric; first_mae picks
    # its own best draw (declared: independent selection).
    best_idx = per_draw_frame.argmin(axis=0)
    best_pred = draws[np.arange(n_frames), best_idx]
    v0 = valid[:, 0]
    first_per_draw = np.stack(
        [
            np.abs(draws[:, d, 0, :] - truth[:, 0, :]).mean(axis=1)
            for d in range(n_draws)
        ],
    )
    best_first = float(first_per_draw.min(axis=0)[v0].mean())

    # Read 3 inputs — per-frame draw dispersion (std over draws, masked
    # element mean) and its horizon profile.
    std = draws.std(axis=1)  # [frames, chunk, dim]
    dispersion = (std * mask).sum(axis=(1, 2)) / np.maximum(mask.sum(axis=(1, 2)), 1)
    dispersion_steps = step_curve(std * mask, valid)

    return {
        "n_frames": int(n_frames),
        "n_draws": int(n_draws),
        "single_draw": {
            "chunk_mae_draw0": round(per_draw_pooled[0], 4),
            "chunk_mae_per_draw_mean": round(float(np.mean(per_draw_pooled)), 4),
            "chunk_mae_per_draw_spread": round(
                float(np.max(per_draw_pooled) - np.min(per_draw_pooled)),
                4,
            ),
            "first_mae_draw0": round(float(first_per_draw[0][v0].mean()), 4),
        },
        "mean_of_draws": {
            "chunk_mae": round(pooled_mae(mean_pred, truth, mask), 4),
            "first_mae": round(
                float(
                    np.abs(mean_pred[:, 0, :] - truth[:, 0, :]).mean(axis=1)[v0].mean(),
                ),
                4,
            ),
        },
        "best_of_n": {
            "chunk_mae": round(pooled_mae(best_pred, truth, mask), 4),
            "first_mae": round(best_first, 4),
        },
        "step_curves": {
            "mean_of_draws": [
                round(v, 4) for v in step_curve(np.abs(mean_pred - truth) * mask, valid)
            ],
            "best_of_n": [
                round(v, 4) for v in step_curve(np.abs(best_pred - truth) * mask, valid)
            ],
            "draw0": [
                round(v, 4)
                for v in step_curve(np.abs(draws[:, 0] - truth) * mask, valid)
            ],
            "dispersion": [round(v, 4) for v in dispersion_steps],
        },
        "_dispersion": dispersion,  # stripped before JSON; read-3 join input
        "_frame_mae_draw0": per_draw_frame[0],
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=None, help="--dump-draws npz")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="degenerate draws=1 oracle on the banked flow-80k npz",
    )
    args = parser.parse_args()
    if (args.draws is None) == (not args.validate):
        sys.exit("exactly one of --draws / --validate")

    flow = np.load(FLOW_NPZ)
    ar = np.load(AR_NPZ)
    assert np.array_equal(flow["index"], ar["index"]), "panel npzs no longer pair"

    if args.validate:
        core = flow["core"]
        draws = flow["pred:bijou@80000"][core][:, None]  # [frames, 1, chunk, dim]
        truth, valid = flow["truth"][core], flow["valid"][core]
        reads = fairness_reads(draws, truth, valid)
        dispersion = reads.pop("_dispersion")
        reads.pop("_frame_mae_draw0")
        checks = {
            "mean_of_1_hits_anchor": abs(
                reads["mean_of_draws"]["chunk_mae"] - FLOW_ANCHOR,
            )
            < 5e-3,
            "best_of_1_hits_anchor": abs(reads["best_of_n"]["chunk_mae"] - FLOW_ANCHOR)
            < 5e-3,
            "first_mae_hits_anchor": abs(
                reads["mean_of_draws"]["first_mae"] - FLOW_FIRST_ANCHOR,
            )
            < 5e-3,
            "dispersion_exactly_zero": bool((dispersion == 0).all()),
        }
        reads["validate_checks"] = checks
        out_path = Path("reports/analysis__draws_fairness_k4l2_validate.json")
        Path(out_path).write_text(json.dumps(reads, indent=1) + "\n")
        for name, ok in checks.items():
            print(f"  {name}: {'OK' if ok else 'FAIL'}")
        print(
            f"mean-of-1 {reads['mean_of_draws']['chunk_mae']} vs anchor {FLOW_ANCHOR}; "
            f"wrote {out_path}",
        )
        if not all(checks.values()):
            sys.exit("VALIDATION FAILED — read implementations are wrong, stop")
        return

    probe = np.load(args.draws)
    core = probe["core"]
    if not core.all():
        print(f"note: {int((~core).sum())} non-core rows in probe — dropping them")
    draws = probe["draws"][core].astype(np.float64)
    truth, valid = probe["truth"][core], probe["valid"][core]
    reads = fairness_reads(draws, truth, valid)
    dispersion = reads.pop("_dispersion")
    frame_mae_draw0 = reads.pop("_frame_mae_draw0")

    # Read 3 — join the probe rows to the full-panel paired deficit.
    panel_pos = {int(ix): i for i, ix in enumerate(flow["index"])}
    rows = np.array([panel_pos[int(ix)] for ix in probe["index"][core]])
    p_truth, p_valid = flow["truth"][rows], flow["valid"][rows]
    assert np.array_equal(p_truth, truth) and np.array_equal(p_valid, valid), (
        "probe rows disagree with the banked panel rows — selection drifted"
    )
    mask = element_mask(truth, valid)
    fl_frame = frame_mae(flow["pred:bijou@80000"][rows], truth, mask)
    ar_frame = frame_mae(ar["pred:bijou@100000"][rows], truth, mask)
    deficit = fl_frame - ar_frame  # >0 = flow worse
    # Instrument drift check: probe draw 0 should re-decode the banked
    # single-draw prediction (same noise key/seed; cross-box numerics
    # may drift a little — report it, gate loosely).
    draw0_drift = float(np.abs(frame_mae_draw0 - fl_frame).mean())

    qs = np.quantile(dispersion, [0.25, 0.5, 0.75])
    bins = np.digitize(dispersion, qs)
    quartiles = {}
    for b, label in enumerate(["q1_tight", "q2", "q3", "q4_dispersed"]):
        s = bins == b
        quartiles[label] = {
            "n": int(s.sum()),
            "dispersion": round(float(dispersion[s].mean()), 4),
            "deficit": round(float(deficit[s].mean()), 4),
            "flow_win_rate": round(float((deficit[s] < 0).mean()), 4),
        }

    reads["read3_dispersion_conditioned"] = {
        "quartiles": quartiles,
        "spearman_dispersion_vs_deficit": round(spearman(dispersion, deficit), 4),
        "draw0_vs_banked_frame_mae_drift": round(draw0_drift, 4),
        "probe_pooled_deficit": round(float(deficit.mean()), 4),
    }
    out_path = Path("reports/analysis__draws_fairness_k4l2.json")
    out_path.write_text(json.dumps(reads, indent=1) + "\n")
    print(json.dumps({k: v for k, v in reads.items() if k != "step_curves"}, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
