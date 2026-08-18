"""Wear audit for the disc-1000 demosonly panel row (58.14).

Registered consumer: interpretation of the pdnorm endpoint's panel
read (posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md) — before
comparing that endpoint's panel row against this baseline, we need to
know how much of the baseline's 58.14 is the normalization WINDOW the
eval made every item wear, and how much is the weights.

Wear facts (read from the checkpoint metadata, recorded in the output
json): grasp_sft_v2_demosonly was trained with
``per_dataset_flow_norm=false`` — its flow section records the MERGED
scheme (``"q01q99"``), so at eval `flow_denormalize_chunk` never
consults any per-dataset row. Every panel item wore the ONE
recomputed-at-launch demos-only global table; the checkpoint's
`per_dataset_stats` (a single grasp_demos_v2 row) is inert here, and
"community repos absent from the table" never arises — there is no
lookup to miss.

Reads, all on the leg npz's core+valid elements against the recorded
summary anchors (refused if the headline does not reproduce):

- box audit: per-joint fraction of truth outside the worn box, the
  FLOOR chunk MAE (the best any box-confined prediction could score —
  predictions are clamp-confined by construction), and per-joint
  prediction edge-saturation (is the clamp even active?);
- re-wear counterfactuals: predictions mapped exactly back to
  normalized space through the worn table (inversion is exact inside
  the clamp; oracle-checked), then re-expressed through (a) honest
  per-repo q01/q99 rows fit on the panel's own truth — the
  pdnorm-style wear — and (b) the released source checkpoint's table.
  OUTPUT-side only: the state input stayed normalized/binned through
  the worn table when the model ran, so these bound the wear share,
  they do not re-run the model;
- demos-prior collapse probe: chunk MAE of the predictions against
  the constant demos action mean — is the model just emitting
  demos-like actions regardless of input?

Usage:
  uv run python -m fontaine.scripts.disc1000_row_audit \
      --npz reports/eval__grasp_sft_v2_demosonly_1gpu_disc__step_001000__panel_v2_k4l2_euler10_draws1_stable.npz \
      --leg-json reports/eval__grasp_sft_v2_demosonly_1gpu_disc__step_001000__panel_v2_k4l2_euler10_draws1_stable.json \
      --metadata /home/ubuntu/checkpoints/finetune/grasp_sft_v2_demosonly_1gpu_disc/step_001000/metadata.json \
      --released-metadata /home/ubuntu/checkpoints/molmoact2-so101-released/metadata.json \
      --out reports/analysis__disc1000_panel_row_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# The leg ran the flow decoder in bf16: raw predictions round-trip
# through bf16 after the fp32 denorm (molmoact2_joint.predict_flow),
# so the exact-inversion oracle carries a bf16-scale tolerance.
BF16_ATOL_DEG = 0.5
ANCHOR_ATOL = 1e-3


def chunk_mae(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> float:
    """The eval's summary semantics (metrics.summarize): total abs
    error over masked steps x all dims / (masked steps x dims).

    Shapes: pred/truth [N, T, D]; mask [N, T] bool.
    """
    diff = np.abs(pred - truth)[mask]
    return float(diff.sum() / (mask.sum() * pred.shape[-1]))


def per_joint_mae(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> list[float]:
    return [float(v) for v in np.abs(pred - truth)[mask].mean(axis=0)]


def box_bounds(q01: np.ndarray, q99: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orientation-free box: descending pairs (q01 > q99) are legal in
    the table (sign-flipped joints) but the reachable set is the same
    interval either way."""
    return np.minimum(q01, q99), np.maximum(q01, q99)


def floor_error(
    truth: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Per-element distance from truth to the box — the error the BEST
    box-confined prediction cannot avoid. Shape-preserving."""
    return np.maximum(0.0, np.maximum(lo - truth, truth - hi))


def normalize_rows(
    value: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    """fast.molmoact2.normalize_q01q99_rows, numpy form (no clamp)."""
    return 2.0 * (value - q01) / (q99 - q01) - 1.0


def unnormalize_rows(
    value: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    """fast.molmoact2.unnormalize_q01q99_rows, numpy form: clamp to
    [-1, 1] then invert (QuantileStats.denormalize's order)."""
    clamped = np.clip(value, -1.0, 1.0)
    return (clamped + 1.0) * (q99 - q01) / 2.0 + q01


def fit_repo_rows(
    truth: np.ndarray,
    valid: np.ndarray,
    repo_ids: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Honest per-repo q01/q99 rows fit on the panel's own truth —
    np.quantile (linear) over every valid chunk element of the repo,
    per joint. Chunk windows overlap across neighbouring frames, so
    elements are duplicate-weighted; recorded as the estimator, close
    enough for a wear counterfactual (a pdnorm run would fit exact
    pooled quantiles over the repo's full frame set instead).
    """
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for repo in np.unique(repo_ids):
        sel = truth[repo_ids == repo][valid[repo_ids == repo]]
        rows[repo] = (
            np.quantile(sel, 0.01, axis=0).astype(np.float32),
            np.quantile(sel, 0.99, axis=0).astype(np.float32),
        )
    return rows


def rewear(
    pred_norm: np.ndarray,
    repo_ids: np.ndarray,
    rows: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Re-express normalized predictions through per-repo rows."""
    out = np.empty_like(pred_norm)
    for repo, (q01, q99) in rows.items():
        pick = repo_ids == repo
        out[pick] = unnormalize_rows(pred_norm[pick], q01, q99)
    return out


def audit(
    npz: dict[str, np.ndarray],
    anchors: dict[str, float],
    worn_q01: np.ndarray,
    worn_q99: np.ndarray,
    worn_mean: np.ndarray,
    released_q01: np.ndarray,
    released_q99: np.ndarray,
) -> dict[str, Any]:
    """The full audit on in-memory arrays (file I/O stays in main)."""
    truth = npz["truth"].astype(np.float32)
    pred = npz["pred:bijou@1000"].astype(np.float32)
    state_copy = npz["pred:state-copy"].astype(np.float32)
    core = npz["core"].astype(bool)
    valid = npz["valid"].astype(bool)
    repo_ids = npz["repo_id"]
    mask = core[:, None] & valid

    # Anchors first: the audit is meaningless on the wrong artifact.
    got = {
        "bijou@1000": chunk_mae(pred, truth, mask),
        "state-copy": chunk_mae(state_copy, truth, mask),
    }
    for name, want in anchors.items():
        if abs(got[name] - want) > ANCHOR_ATOL:
            raise SystemExit(
                f"anchor mismatch for {name}: leg json records "
                f"{want:.6f}, npz reproduces {got[name]:.6f}",
            )

    lo, hi = box_bounds(worn_q01, worn_q99)
    truth_m = truth[mask]
    pred_m = pred[mask]
    outside = (truth_m < lo) | (truth_m > hi)
    floor = floor_error(truth_m, lo, hi)
    span = hi - lo
    edge = ((pred_m - lo) < 0.01 * span) | ((hi - pred_m) < 0.01 * span)

    # Exact inversion back to normalized space (oracle: re-wearing the
    # worn table must reproduce the raw predictions to bf16 slop).
    pred_norm = normalize_rows(pred, worn_q01, worn_q99)
    round_trip = unnormalize_rows(pred_norm, worn_q01, worn_q99)
    worst = float(np.abs(round_trip - pred)[mask].max())
    if worst > BF16_ATOL_DEG:
        raise SystemExit(
            f"inversion round-trip error {worst:.4f} deg exceeds the "
            f"bf16 tolerance {BF16_ATOL_DEG} — wrong worn table?",
        )

    repo_rows = fit_repo_rows(truth, valid, repo_ids)
    pred_repo = rewear(pred_norm, repo_ids, repo_rows)
    pred_released = unnormalize_rows(pred_norm, released_q01, released_q99)

    per_repo: list[dict[str, Any]] = []
    for repo in np.unique(repo_ids[core]):
        pick = mask & (repo_ids == repo)[:, None]
        t, p = truth[pick], pred[pick]
        per_repo.append(
            {
                "repo": str(repo),
                "elements": int(pick.sum()),
                "mae": float(np.abs(p - t).mean()),
                "floor_mae": float(floor_error(t, lo, hi).mean()),
                "rewear_repo_mae": float(np.abs(pred_repo[pick] - t).mean()),
                "truth_outside_frac": float(((t < lo) | (t > hi)).mean()),
            },
        )
    per_repo.sort(key=lambda r: r["mae"], reverse=True)

    return {
        "anchors_reproduced": got,
        "box_audit": {
            "truth_outside_frac_per_joint": dict(
                zip(MOTORS, [float(v) for v in outside.mean(axis=0)], strict=False),
            ),
            "truth_any_joint_outside_frac": float(outside.any(axis=1).mean()),
            "floor_chunk_mae": float(floor.mean()),
            "floor_per_joint": dict(
                zip(MOTORS, [float(v) for v in floor.mean(axis=0)], strict=False),
            ),
            "pred_edge_saturation_per_joint": dict(
                zip(MOTORS, [float(v) for v in edge.mean(axis=0)], strict=False),
            ),
        },
        "actual_per_joint_mae": dict(
            zip(MOTORS, per_joint_mae(pred, truth, mask), strict=False),
        ),
        "rewear": {
            "round_trip_worst_abs_deg": worst,
            "repo_rows_chunk_mae": chunk_mae(pred_repo, truth, mask),
            "repo_rows_per_joint": dict(
                zip(MOTORS, per_joint_mae(pred_repo, truth, mask), strict=False),
            ),
            "released_table_chunk_mae": chunk_mae(pred_released, truth, mask),
            "released_table_per_joint": dict(
                zip(MOTORS, per_joint_mae(pred_released, truth, mask), strict=False),
            ),
            # Null for the repo-rows read: a CONSTANT mid-box (norm 0)
            # prediction re-worn through the same rows — what knowing
            # each repo's box is worth with no model at all.
            "null_repo_midpoint_chunk_mae": chunk_mae(
                rewear(np.zeros_like(pred_norm), repo_ids, repo_rows),
                truth,
                mask,
            ),
            "n_repo_rows": len(repo_rows),
        },
        "demos_prior_collapse": {
            "pred_vs_demos_mean_mae": chunk_mae(
                np.broadcast_to(worn_mean, pred.shape),
                pred,
                mask,
            ),
            "truth_vs_demos_mean_mae": chunk_mae(
                np.broadcast_to(worn_mean, truth.shape),
                truth,
                mask,
            ),
        },
        "per_repo_worst10": per_repo[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--leg-json", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--released-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    leg = json.loads(args.leg_json.read_text())
    metadata = json.loads(args.metadata.read_text())
    released = json.loads(args.released_metadata.read_text())

    flow_section = metadata["components"]["flow_decoder"]["config"]
    wear_facts = {
        "leg_molmo_norm": leg["molmo_norm"],
        "flow_normalization_tag": flow_section["normalization"],
        "train_per_dataset_flow_norm": metadata["train_args"]["per_dataset_flow_norm"],
        "per_dataset_stats_keys": sorted(metadata["per_dataset_stats"]),
        "stats_note": metadata["stats_note"],
        "worn_table": metadata["stats"]["action"],
        "verdict": (
            "merged scheme: flow_denormalize_chunk(per_dataset=False) — "
            "every item wore the recomputed-at-launch demos-only global "
            "table; per_dataset_stats is never consulted at eval"
        ),
    }
    if flow_section["normalization"] != "q01q99":
        raise SystemExit(
            "this audit is written for the merged scheme; checkpoint "
            f"records {flow_section['normalization']!r}",
        )

    anchors = {
        s["policy"]: s["chunk_mae"]
        for s in leg["summaries"]
        if s["policy"] in ("bijou@1000", "state-copy")
    }
    result = {
        "npz": str(args.npz),
        "checkpoint": leg["checkpoint"],
        "wear_facts": wear_facts,
        "caveats": [
            (
                "re-wear counterfactuals re-express the OUTPUT side only: "
                "the state input was normalized/binned through the worn "
                "demos table when the model ran, so state-side OOD is not "
                "undone — they bound the wear share, they do not re-run "
                "the model"
            ),
            (
                "per-repo rows are fit on the panel's own truth "
                "(np.quantile linear over valid chunk elements, "
                "duplicate-weighted by window overlap) — oracle-ish rows, "
                "optimistic vs a deployment row estimate"
            ),
        ],
        **audit(
            {k: z[k] for k in z.files},
            anchors,
            np.asarray(metadata["stats"]["action"]["q01"], np.float32),
            np.asarray(metadata["stats"]["action"]["q99"], np.float32),
            np.asarray(metadata["stats"]["action"]["mean"], np.float32),
            np.asarray(released["stats"]["action"]["q01"], np.float32),
            np.asarray(released["stats"]["action"]["q99"], np.float32),
        ),
    }

    args.out.write_text(json.dumps(result, indent=1) + "\n")
    box = result["box_audit"]
    rw = result["rewear"]
    print(f"anchors reproduced: {result['anchors_reproduced']}")
    print(
        f"truth outside worn box: any-joint "
        f"{box['truth_any_joint_outside_frac']:.1%}; floor chunk MAE "
        f"{box['floor_chunk_mae']:.2f} of the actual "
        f"{result['anchors_reproduced']['bijou@1000']:.2f}",
    )
    print(
        f"re-wear: per-repo rows {rw['repo_rows_chunk_mae']:.2f}, "
        f"released table {rw['released_table_chunk_mae']:.2f} "
        f"({rw['n_repo_rows']} repo rows)",
    )
    print(
        f"demos-prior collapse probe: pred-vs-demos-mean "
        f"{result['demos_prior_collapse']['pred_vs_demos_mean_mae']:.2f} "
        f"(truth-vs-demos-mean "
        f"{result['demos_prior_collapse']['truth_vs_demos_mean_mae']:.2f})",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
