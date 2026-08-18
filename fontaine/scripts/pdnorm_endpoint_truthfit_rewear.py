"""Estimator-consistency cross-check for the pdnorm endpoint panel row.

Registered consumer: the pdnorm endpoint read (pre-reg
posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md, calibration
note). Under the ``q01q99_per_dataset`` scheme the endpoint's panel
predictions wear each item's NATIVE per-dataset training-table row —
the repo's own recorded LeRobot ``meta/stats.json`` q01/q99, exactly
what StatsAttachedDataset attaches at eval (bijou/data.py) and
flow_denormalize_chunk inverts at serving. The ladder anchors
(disc-1000 27.40, released 27.14, midpoint null 25.15) instead wear
per-repo rows FIT ON THE PANEL'S OWN TRUTH (disc1000_row_audit
estimator — oracle-ish rows a deployment never has). Same wear class,
different ESTIMATOR. This sibling of released_row_rewear closes the
seam: it inverts the endpoint npz per repo through the native rows
(identity-checked round trip) and re-expresses through the same
panel-truth-fit rows the anchors wear, recording the
native-vs-truth-fit wear delta alongside the ladder read.

NOTE on the queue wording: the checkpoint's ``per_dataset_stats``
table holds only the TRAINING repos and is never consulted for panel
items (git audit 2026-08-18: bijou/data.py:983 attaches each panel
repo's own ``meta/stats.json`` row) — the native rows are read from
the eval's ``--data`` root, the same source the leg actually wore.

OUTPUT-side only, like its siblings: state stayed merged-table when
the model ran; this bounds the estimator-seam share, it does not
re-run the model.

Usage (ON GO, after the endpoint panel leg):
  uv run python -m fontaine.scripts.pdnorm_endpoint_truthfit_rewear \
      --npz reports/eval__grasp_sft_v2_joint_1gpu_pdnorm__step_003000__panel_v2_k4l2_euler10_draws1_stable.npz \
      --leg-json reports/eval__grasp_sft_v2_joint_1gpu_pdnorm__step_003000__panel_v2_k4l2_euler10_draws1_stable.json \
      --metadata ~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm/step_003000/metadata.json \
      --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
      --sft-audit reports/analysis__disc1000_panel_row_audit.json \
      --released-audit reports/analysis__released_row_honest_wear.json \
      --out reports/analysis__pdnorm_endpoint_truthfit_wear.json
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
from fontaine.scripts.released_row_rewear import check_same_panel_null

# A native row's joint window below this span cannot be inverted (the
# worn map decodes everything to the constant): the normalized
# coordinate is unrecoverable, so it is fixed at the midpoint 0.0 —
# legal only when the prediction actually sits at the constant (to
# serving precision); anything else means the wrong worn table.
DEGENERATE_SPAN_DEG = 1e-2

RowMap = dict[str, tuple[np.ndarray, np.ndarray]]


def load_native_rows(data_root: Path, repos: list[str]) -> RowMap:
    """Each repo's NATIVE action q01/q99 training-table row, read from
    the eval data root's ``<repo>/meta/stats.json`` — the exact rows
    StatsAttachedDataset attached when the leg ran. Loud on a missing
    repo or a stats file without exact quantiles."""
    rows: RowMap = {}
    for repo in repos:
        stats_path = data_root / repo / "meta" / "stats.json"
        if not stats_path.is_file():
            raise SystemExit(
                f"no native stats for npz repo {repo!r}: {stats_path} "
                "missing — wrong --data root? (the leg could not have "
                "worn a row it cannot read)",
            )
        action = json.loads(stats_path.read_text()).get("action", {})
        if "q01" not in action or "q99" not in action:
            raise SystemExit(
                f"{stats_path} lacks action q01/q99 (exact corpus "
                "quantiles) — the per-dataset scheme cannot have worn "
                "this repo",
            )
        rows[repo] = (
            np.asarray(action["q01"], np.float32),
            np.asarray(action["q99"], np.float32),
        )
    return rows


def invert_native(
    pred: np.ndarray,
    repo_ids: np.ndarray,
    rows: RowMap,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Invert raw predictions per repo through that repo's native row.

    The per-repo inversion identity oracle rides inside: re-wearing
    each repo's own row must reproduce its raw predictions to serving
    precision — a global-table inversion (or rows from the wrong
    source) fails loudly per repo. Degenerate joints (span below
    DEGENERATE_SPAN_DEG) pin the normalized coordinate at 0.0 after
    checking the prediction sits at the constant."""
    pred_norm = np.empty_like(pred)
    worst_by_repo: dict[str, float] = {}
    degenerate: list[dict[str, Any]] = []
    for repo, (q01, q99) in rows.items():
        pick = repo_ids == repo
        repo_mask = mask[pick]
        span_ok = np.abs(q99 - q01) >= DEGENERATE_SPAN_DEG
        safe_q99 = np.where(span_ok, q99, q01 + 1.0)
        norm = normalize_rows(pred[pick], q01, safe_q99)
        if not span_ok.all():
            midpoint = (q01 + q99) / 2.0
            for joint in np.flatnonzero(~span_ok):
                off = np.abs(pred[pick][..., joint] - midpoint[joint])
                worst_off = float(off[repo_mask].max()) if repo_mask.any() else 0.0
                if worst_off > BF16_ATOL_DEG:
                    raise SystemExit(
                        f"repo {repo!r} joint {MOTORS[joint]}: native "
                        f"span is degenerate (<{DEGENERATE_SPAN_DEG}) "
                        f"yet predictions sit {worst_off:.4f} deg off "
                        "the constant — wrong native table?",
                    )
                degenerate.append(
                    {
                        "repo": repo,
                        "joint": MOTORS[joint],
                        "worst_abs_off_constant_deg": worst_off,
                    },
                )
            norm[..., ~span_ok] = 0.0
        pred_norm[pick] = norm
        if not repo_mask.any():
            continue
        # Identity through the TRUE native row: a degenerate joint's
        # pinned 0.0 lands on the midpoint, which the check above
        # already bounded against the predictions.
        round_trip = unnormalize_rows(pred_norm[pick], q01, q99)
        worst = float(np.abs(round_trip - pred[pick])[repo_mask].max())
        if worst > BF16_ATOL_DEG:
            raise SystemExit(
                f"per-repo inversion identity failed for {repo!r}: "
                f"round-trip error {worst:.4f} deg exceeds "
                f"{BF16_ATOL_DEG} — this repo did not wear its native "
                "row (wrong --data root, or a non-per-dataset leg?)",
            )
        worst_by_repo[repo] = worst
    facts = {
        "round_trip_worst_abs_deg": max(worst_by_repo.values()),
        "degenerate_joints": degenerate,
    }
    return pred_norm, facts


def require_per_dataset_scheme(metadata: dict[str, Any]) -> None:
    """This cross-check only means anything for an endpoint whose leg
    wore native per-dataset rows — the checkpoint's recorded scheme is
    the ground truth for that."""
    scheme = metadata["components"]["flow_decoder"]["config"]["normalization"]
    if scheme != "q01q99_per_dataset":
        raise SystemExit(
            f"checkpoint records normalization={scheme!r}, not "
            "'q01q99_per_dataset' — its panel leg wore a global table; "
            "use disc1000_row_audit / released_row_rewear for that "
            "wear class",
        )


def truthfit_rewear(
    npz: dict[str, np.ndarray],
    anchors: dict[str, float],
    native_rows: RowMap,
    pred_key: str,
) -> dict[str, Any]:
    """Re-express the natively-worn predictions through the
    panel-truth-fit per-repo rows the ladder anchors wear."""
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

    missing = sorted(set(map(str, np.unique(repo_ids))) - set(native_rows))
    if missing:
        raise SystemExit(
            f"{len(missing)} npz repos lack native rows (first: "
            f"{missing[0]!r}) — load_native_rows must cover every repo",
        )

    pred_norm, inversion_facts = invert_native(pred, repo_ids, native_rows, mask)
    truthfit_rows = fit_repo_rows(truth, valid, repo_ids)
    pred_truthfit = rewear(pred_norm, repo_ids, truthfit_rows)
    null_truthfit = rewear(np.zeros_like(pred_norm), repo_ids, truthfit_rows)

    per_repo: list[dict[str, Any]] = []
    for repo in np.unique(repo_ids[core]):
        pick = mask & (repo_ids == repo)[:, None]
        t = truth[pick]
        native = float(np.abs(pred[pick] - t).mean())
        truthfit = float(np.abs(pred_truthfit[pick] - t).mean())
        per_repo.append(
            {
                "repo": str(repo),
                "elements": int(pick.sum()),
                "native_wear_mae": native,
                "truthfit_wear_mae": truthfit,
                "delta_native_minus_truthfit": native - truthfit,
            },
        )
    per_repo.sort(
        key=lambda r: abs(r["delta_native_minus_truthfit"]),
        reverse=True,
    )

    native_mae = got[pred_key]
    truthfit_mae = chunk_mae(pred_truthfit, truth, mask)
    return {
        "anchors_reproduced": got,
        **inversion_facts,
        "native_wear_chunk_mae": native_mae,
        "truthfit_wear_chunk_mae": truthfit_mae,
        "estimator_seam_delta_native_minus_truthfit": native_mae - truthfit_mae,
        "native_wear_per_joint": dict(
            zip(MOTORS, per_joint_mae(pred, truth, mask), strict=False),
        ),
        "truthfit_wear_per_joint": dict(
            zip(MOTORS, per_joint_mae(pred_truthfit, truth, mask), strict=False),
        ),
        "null_repo_midpoint_chunk_mae": chunk_mae(null_truthfit, truth, mask),
        "n_repo_rows": len(truthfit_rows),
        "per_repo_worst10_by_abs_delta": per_repo[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--leg-json", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sft-audit", type=Path, required=True)
    parser.add_argument("--released-audit", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    leg = json.loads(args.leg_json.read_text())
    metadata = json.loads(args.metadata.read_text())
    sft_audit = json.loads(args.sft_audit.read_text())

    require_per_dataset_scheme(metadata)
    if leg["molmo_norm"] != "checkpoint":
        raise SystemExit(
            "this cross-check assumes the CONTRACT path of a "
            "per-dataset-scheme checkpoint (items wear their own rows "
            "natively); the off-contract --molmo-norm wraps are a "
            f"different machinery — leg records {leg['molmo_norm']!r}",
        )
    (pred_key,) = [k for k in z.files if k.startswith("pred:bijou@")]
    policy = pred_key.removeprefix("pred:")
    anchors = {
        (pred_key if s["policy"] == policy else "state-copy"): s["chunk_mae"]
        for s in leg["summaries"]
        if s["policy"] in (policy, "state-copy")
    }

    repos = sorted(map(str, np.unique(z["repo_id"])))
    native_rows = load_native_rows(args.data, repos)
    result_core = truthfit_rewear(
        {k: z[k] for k in z.files},
        anchors,
        native_rows,
        pred_key,
    )
    check_same_panel_null(
        result_core["null_repo_midpoint_chunk_mae"],
        sft_audit,
    )

    ladder = {
        "sft_disc1000_truthfit": sft_audit["rewear"]["repo_rows_chunk_mae"],
        "null_repo_midpoint": result_core["null_repo_midpoint_chunk_mae"],
    }
    if args.released_audit is not None:
        released = json.loads(args.released_audit.read_text())
        ladder["released_truthfit"] = released["honest_wear_chunk_mae"]

    result = {
        "npz": str(args.npz),
        "checkpoint": leg["checkpoint"],
        "wear_facts": {
            "leg_molmo_norm": leg["molmo_norm"],
            "scheme": "q01q99_per_dataset",
            "native_row_source": str(args.data) + "/<repo>/meta/stats.json",
            "checkpoint_per_dataset_stats_keys": sorted(
                metadata.get("per_dataset_stats") or {},
            ),
            "verdict": (
                "the leg wore each panel item's NATIVE per-dataset "
                "training-table row (contract path of the per-dataset "
                "scheme); this re-expression moves the OUTPUT side onto "
                "the panel-truth-fit per-repo rows the ladder anchors "
                "wear — estimator-consistent endpoint-vs-ladder"
            ),
        },
        "caveats": [
            (
                "output-side only: state stayed merged-table when the "
                "model ran — this bounds the estimator-seam share, it "
                "does not re-run the model"
            ),
            (
                "truth-fit rows are oracle-ish (fit on the panel's own "
                "truth, duplicate-weighted chunk elements) — identical "
                "estimator and panel as the disc-1000/released audits, "
                "which is the point: same estimator on both sides of "
                "the ladder"
            ),
            (
                "the NATIVE row is the deployment-honest number (a "
                "served rig wears recorded tables, not truth-fit rows); "
                "the truth-fit row exists to read the ladder like for "
                "like, not to replace it"
            ),
        ],
        **result_core,
        "ladder_read": {
            "endpoint_native": result_core["native_wear_chunk_mae"],
            "endpoint_truthfit": result_core["truthfit_wear_chunk_mae"],
            "estimator_seam_delta": result_core[
                "estimator_seam_delta_native_minus_truthfit"
            ],
            **ladder,
        },
    }

    args.out.write_text(json.dumps(result, indent=1) + "\n")
    read = result["ladder_read"]
    print(f"anchors reproduced: {result['anchors_reproduced']}")
    print(
        f"endpoint native {read['endpoint_native']:.2f} -> truthfit "
        f"{read['endpoint_truthfit']:.2f} (estimator seam "
        f"{read['estimator_seam_delta']:+.2f}); ladder truthfit: "
        f"disc-1000 {read['sft_disc1000_truthfit']:.2f}, released "
        f"{read.get('released_truthfit', float('nan')):.2f}, null "
        f"{read['null_repo_midpoint']:.2f} ({result['n_repo_rows']} rows)",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
