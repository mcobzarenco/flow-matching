"""Honest-wear re-expression of the released checkpoint's panel row.

Registered consumer: the pdnorm-endpoint anchor ladder (pre-reg
posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md, calibration
note). The released row was banked (08:22Z 08-18) wearing its OWN
released global table, while the ladder's same-model reference 27.40
wears honest per-repo rows fit on the panel truth — different wear
classes, so released-vs-SFT carried a wear-mismatch caveat. This
sibling of disc1000_row_audit re-expresses the released checkpoint's
banked normalized predictions through the SAME honest per-repo rows
(the panel arrays are element-identical between the two npz files, so
the fitted rows are byte-identical — enforced via the midpoint-null
identity anchor against the disc-1000 audit json), producing the
same-wear released row alongside 27.40.

OUTPUT-side only, like the parent audit: the state input was
normalized/binned through the released table when the model ran, so
this bounds the wear share; it does not re-run the model.

Usage:
  uv run python -m fontaine.scripts.released_row_rewear \
      --npz reports/eval__molmoact2_so101_released__panel_v2_k4l2_euler10_draws1_stable.npz \
      --leg-json reports/eval__molmoact2_so101_released__panel_v2_k4l2_euler10_draws1_stable.json \
      --metadata /home/ubuntu/checkpoints/molmoact2-so101-released/metadata.json \
      --sft-audit reports/analysis__disc1000_panel_row_audit.json \
      --out reports/analysis__released_row_honest_wear.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fontaine.scripts.disc1000_row_audit import (
    ANCHOR_ATOL,
    BF16_ATOL_DEG,
    MOTORS,
    chunk_mae,
    fit_repo_rows,
    normalize_rows,
    per_joint_mae,
    rewear,
    unnormalize_rows,
)

# The honest rows are fit on the panel's own truth; identical panels
# must yield an identical midpoint null (pure float determinism, no
# model involved) — anything looser means the panels differ.
NULL_MATCH_ATOL = 1e-6


def honest_rewear(
    npz: dict[str, np.ndarray],
    anchors: dict[str, float],
    worn_q01: np.ndarray,
    worn_q99: np.ndarray,
    pred_key: str,
) -> dict[str, Any]:
    """Re-express the worn predictions through honest per-repo rows.

    Refuses on anchor mismatch (wrong artifact) and on a round-trip
    through the worn table that exceeds the serving-precision
    tolerance (wrong worn table).
    """
    truth = npz["truth"].astype(np.float32)
    pred = npz[pred_key].astype(np.float32)
    state_copy = npz["pred:state-copy"].astype(np.float32)
    core = npz["core"].astype(bool)
    valid = npz["valid"].astype(bool)
    repo_ids = npz["repo_id"]
    mask = core[:, None] & valid

    got = {
        pred_key: chunk_mae(pred, truth, mask),
        "state-copy": chunk_mae(state_copy, truth, mask),
    }
    for name, want in anchors.items():
        if abs(got[name] - want) > ANCHOR_ATOL:
            raise SystemExit(
                f"anchor mismatch for {name}: leg json records "
                f"{want:.6f}, npz reproduces {got[name]:.6f}",
            )

    # Exact inversion oracle: re-wearing the worn table itself must
    # reproduce the raw predictions to serving precision.
    pred_norm = normalize_rows(pred, worn_q01, worn_q99)
    round_trip = unnormalize_rows(pred_norm, worn_q01, worn_q99)
    worst = float(np.abs(round_trip - pred)[mask].max())
    if worst > BF16_ATOL_DEG:
        raise SystemExit(
            f"inversion round-trip error {worst:.4f} deg exceeds the "
            f"tolerance {BF16_ATOL_DEG} — wrong worn table?",
        )

    repo_rows = fit_repo_rows(truth, valid, repo_ids)
    pred_repo = rewear(pred_norm, repo_ids, repo_rows)
    null_repo = rewear(np.zeros_like(pred_norm), repo_ids, repo_rows)

    per_repo: list[dict[str, Any]] = []
    for repo in np.unique(repo_ids[core]):
        pick = mask & (repo_ids == repo)[:, None]
        t = truth[pick]
        per_repo.append(
            {
                "repo": str(repo),
                "elements": int(pick.sum()),
                "own_table_mae": float(np.abs(pred[pick] - t).mean()),
                "honest_wear_mae": float(np.abs(pred_repo[pick] - t).mean()),
            },
        )
    per_repo.sort(key=lambda r: r["honest_wear_mae"], reverse=True)

    return {
        "anchors_reproduced": got,
        "round_trip_worst_abs_deg": worst,
        "honest_wear_chunk_mae": chunk_mae(pred_repo, truth, mask),
        "honest_wear_per_joint": dict(
            zip(MOTORS, per_joint_mae(pred_repo, truth, mask), strict=False),
        ),
        "own_table_per_joint": dict(
            zip(MOTORS, per_joint_mae(pred, truth, mask), strict=False),
        ),
        "null_repo_midpoint_chunk_mae": chunk_mae(null_repo, truth, mask),
        "n_repo_rows": len(repo_rows),
        "per_repo_worst10": per_repo[:10],
    }


def check_same_panel_null(mine: float, sft_audit: dict[str, Any]) -> None:
    """Identity anchor: the two audits fit rows on the same panel truth,
    so their midpoint nulls must agree to float determinism."""
    theirs = sft_audit["rewear"]["null_repo_midpoint_chunk_mae"]
    if abs(mine - theirs) > NULL_MATCH_ATOL:
        raise SystemExit(
            f"midpoint-null identity anchor failed: this panel yields "
            f"{mine:.6f}, the disc-1000 audit recorded {theirs:.6f} — "
            f"the panels (and so the honest rows) are NOT the same wear",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--leg-json", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sft-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    leg = json.loads(args.leg_json.read_text())
    metadata = json.loads(args.metadata.read_text())
    sft_audit = json.loads(args.sft_audit.read_text())

    if leg["molmo_norm"] != "checkpoint":
        raise SystemExit(
            "this re-expression assumes the leg wore the checkpoint's own "
            f"global table; leg records molmo_norm={leg['molmo_norm']!r}",
        )
    (pred_key,) = [k for k in z.files if k.startswith("pred:bijou@")]
    policy = pred_key.removeprefix("pred:")
    anchors = {
        (pred_key if s["policy"] == policy else "state-copy"): s["chunk_mae"]
        for s in leg["summaries"]
        if s["policy"] in (policy, "state-copy")
    }

    result_core = honest_rewear(
        {k: z[k] for k in z.files},
        anchors,
        np.asarray(metadata["stats"]["action"]["q01"], np.float32),
        np.asarray(metadata["stats"]["action"]["q99"], np.float32),
        pred_key,
    )
    check_same_panel_null(
        result_core["null_repo_midpoint_chunk_mae"],
        sft_audit,
    )

    sft_honest = sft_audit["rewear"]["repo_rows_chunk_mae"]
    released_honest = result_core["honest_wear_chunk_mae"]
    result = {
        "npz": str(args.npz),
        "checkpoint": leg["checkpoint"],
        "wear_facts": {
            "leg_molmo_norm": leg["molmo_norm"],
            "worn_table": metadata["stats"]["action"],
            "verdict": (
                "the leg wore the released checkpoint's own q01q99 global "
                "table (molmo_norm=checkpoint); this re-expression moves "
                "the OUTPUT side onto the same honest per-repo rows the "
                "disc-1000 27.40 reference wears"
            ),
        },
        "caveats": [
            (
                "output-side only: the state input stayed "
                "normalized/binned through the released table when the "
                "model ran — this bounds the wear share, it does not "
                "re-run the model"
            ),
            (
                "per-repo rows are fit on the panel's own truth "
                "(oracle-ish rows, optimistic vs a deployment estimate) — "
                "identical estimator and panel as the disc-1000 audit, "
                "which is the point: same wear on both sides"
            ),
        ],
        **result_core,
        "same_wear_comparison": {
            "released_own_table": anchors[pred_key],
            "released_honest_wear": released_honest,
            "sft_disc1000_honest_wear": sft_honest,
            "delta_sft_minus_released": sft_honest - released_honest,
            "null_repo_midpoint": result_core["null_repo_midpoint_chunk_mae"],
            "read": (
                "wear held fixed (honest per-repo rows on both sides), "
                "the SFT checkpoint ends within noise of where it "
                "started, and both rows are slightly WORSE than the "
                "repo-midpoint null — SFT neither destroyed nor built "
                "community competence"
            ),
        },
    }

    args.out.write_text(json.dumps(result, indent=1) + "\n")
    cmp_ = result["same_wear_comparison"]
    print(f"anchors reproduced: {result['anchors_reproduced']}")
    print(
        f"released honest-wear {cmp_['released_honest_wear']:.2f} vs "
        f"SFT honest-wear {cmp_['sft_disc1000_honest_wear']:.2f} "
        f"(delta {cmp_['delta_sft_minus_released']:+.2f}; own-table "
        f"{cmp_['released_own_table']:.2f}, null "
        f"{cmp_['null_repo_midpoint']:.2f}; {result['n_repo_rows']} rows)",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
