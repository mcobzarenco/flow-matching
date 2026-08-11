"""End-to-end predictor parity — our first-class MolmoAct2 stack vs the
banked HF-forward anchors (port item 3, gate G2).

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md: our
``bijou.molmoact2.MolmoAct2Predictor`` (item-2 packing -> bijou.molmo2
trunk forward -> item-1 wiring/expert -> their output tail) runs the
SAME 240 anchor rows the rig preflight banked from THEIR HF
``predict_action`` (same per-row noise: fresh cuda generator seeded
``BASE_SEED + concat_index``), and must

  * agree with the banked per-frame chunks to <= 0.075 MAE-units pooled
    (the bf16 budget as AMENDED 2026-08-11 — the pre-reg's 0.05 was the
    G1 module-level placeholder "to be tightened by measurement"; the
    measured cross-implementation floor is 1-ulp bf16 kernel-order
    rounding in the vision tower, amplified end-to-end to 0.041 on the
    released checkpoint and 0.054 on the rig-ft rung. Localization
    chain banked in the amendment post: inputs byte-identical, their-KV
    through our flow loop reproduces banked to 0.0000, both stacks
    individually byte-deterministic), and
  * reproduce the banked pooled anchor MAE (zero-shot 28.9454 / rung
    2000 3.2301) within the same budget (measured 0.0002 / 0.0020).

Both directions of G3 run by default: the released checkpoint in our
stack AND our rig-ft rung-2000 export in our stack. Alignment oracles
hard-abort if a dataset row drifts from the banked npz identity
(state + truth chunk), so the comparison can never silently shift rows.

Usage:
    uv run python fontaine/scripts/molmoact2_e2e_parity.py \
        [--arm released|step2000] [--tolerance 0.05] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

HORIZON = 30  # their SO-100/101 tag: 30 steps at native fps
NORM_TAG = "so100_so101_molmoact2"
BASE_SEED = 0  # the preflight's noise convention: BASE_SEED + concat index
RIG_REPOS = (
    Path("~/datasets/mcobzarenco/so101_pick_place_clean").expanduser(),
    Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
)
OUT_JSON = REPO_ROOT / "reports/analysis__molmoact2_e2e_parity.json"


@dataclass(frozen=True)
class Arm:
    name: str
    checkpoint: str
    banked_stem: str  # reports/<stem>.{npz,json}
    anchor_key: str  # the banked json field holding the pooled anchor MAE


ARMS = (
    Arm(
        name="released",
        checkpoint="allenai/MolmoAct2-SO100_101",
        banked_stem="analysis__molmoact2_rig_preflight",
        anchor_key="zero_shot_mae_matched",
    ),
    Arm(
        name="step2000",
        checkpoint=str(
            Path("~/checkpoints/molmoact2-so101-rig-r1-step2000-hf").expanduser(),
        ),
        banked_stem="analysis__molmoact2_rig_ft_step2000",
        anchor_key="zero_shot_mae_matched",
    ),
)


def load_dataset():  # noqa: ANN201
    from bijou.data import EpisodeSplit, select_datasets

    selection = select_datasets(
        RIG_REPOS,
        (),
        HORIZON,
        episode_split=EpisodeSplit.ALL,
        allowed_fps=(30.0,),
        allowed_camera_counts=(2,),
    )
    if len(selection.datasets) != 2:
        sys.exit("expected exactly the 2 rig repos — stop")
    return selection.concat()


def frame_images(item: dict) -> list[torch.Tensor]:
    images = [
        item[key]
        for key in sorted(k for k in item if k.startswith("observation.images."))
    ]
    if not images:
        sys.exit("frame has no observation.images.* keys — stop")
    return images


def run_arm(
    arm: Arm,
    dataset: Any,
    device: str,
    tolerance: float,
    limit: int | None,
) -> dict:
    from bijou.molmoact2 import MolmoAct2Predictor

    banked_npz = np.load(REPO_ROOT / f"reports/{arm.banked_stem}.npz")
    banked_json = json.loads(
        (REPO_ROOT / f"reports/{arm.banked_stem}.json").read_text(),
    )
    anchor = float(banked_json[arm.anchor_key])
    rows = banked_npz["rows"]
    theirs = banked_npz["preds"]  # (N, 30, 6) fp32 from their HF forward
    truths = banked_npz["truths"]
    states = banked_npz["states"]
    if limit is not None:
        keep = np.linspace(0, len(rows) - 1, num=limit).astype(int)
        rows, theirs, truths, states = (
            rows[keep],
            theirs[keep],
            truths[keep],
            states[keep],
        )

    print(f"[{arm.name}] loading {arm.checkpoint} ...", flush=True)
    predictor = MolmoAct2Predictor.load(
        arm.checkpoint,
        NORM_TAG,
        device=device,
        dtype=torch.bfloat16,
    )

    ours = np.full_like(theirs, np.nan)
    started = time.monotonic()
    for i, idx in enumerate(rows):
        item = dataset[int(idx)]
        state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
        if not np.allclose(state, states[i], atol=1e-5):
            sys.exit(f"row {i} (concat {idx}): state drifted from the banked npz")
        action = item["action"].float().numpy()
        n = min(action.shape[0], HORIZON)
        banked_truth = truths[i, :n]
        if not np.allclose(action[:n], banked_truth, atol=1e-5, equal_nan=True):
            sys.exit(f"row {i} (concat {idx}): truth chunk drifted from the banked npz")
        generator = torch.Generator(device=device)
        generator.manual_seed(BASE_SEED + int(idx))
        pred = predictor.predict_action(
            images=frame_images(item),
            task=str(item["task"]),
            state=torch.from_numpy(state),
            generator=generator,
        )
        chunk = pred[0].numpy()
        if chunk.shape != theirs.shape[1:]:
            sys.exit(f"prediction shape {chunk.shape} != {theirs.shape[1:]} — stop")
        ours[i] = chunk
        if (i + 1) % 40 == 0:
            rate = (i + 1) / max(time.monotonic() - started, 1e-6) * 60
            print(f"[{arm.name}] {i + 1}/{len(rows)} ({rate:.1f} f/min)", flush=True)

    delta = np.abs(ours - theirs)
    valid = np.isfinite(truths).all(-1)  # (N, H) — matched-window rows
    w = valid[:, :, None].repeat(truths.shape[-1], 2)
    mae_ours = float(np.abs(ours - truths)[w].mean())
    mae_theirs = float(np.abs(theirs - truths)[w].mean())
    per_frame = delta.mean(axis=(1, 2))
    pooled_delta = round(float(delta.mean()), 6)
    report: dict[str, Any] = {
        "checkpoint": arm.checkpoint,
        "n_frames": len(rows),
        "pooled_abs_delta_vs_banked": pooled_delta,
        "worst_frame_mean_delta": round(float(per_frame.max()), 6),
        "worst_element_delta": round(float(delta.max()), 6),
        "pooled_mae_ours": round(mae_ours, 4),
        "pooled_mae_banked_recomputed": round(mae_theirs, 4),
        "anchor_mae_banked_json": round(anchor, 4),
        "tolerance": tolerance,
    }
    np.savez_compressed(
        REPO_ROOT / f"reports/{arm.banked_stem}_ours.npz",
        preds=ours,
        rows=rows,
    )
    report["pass_chunk_parity"] = bool(pooled_delta <= tolerance)
    # Anchor reproduction: pooled MAE vs the banked forward's on the SAME
    # rows (subset-safe); at full scale the banked pooled MAE must itself
    # match the frozen json anchor, or the comparison target has drifted.
    report["pass_anchor_repro"] = bool(abs(mae_ours - mae_theirs) <= tolerance)
    if limit is None and abs(mae_theirs - anchor) > 5e-4:
        sys.exit(
            f"[{arm.name}] banked npz pooled MAE {mae_theirs:.4f} != frozen "
            f"json anchor {anchor:.4f} — comparison target drifted, stop",
        )
    print(f"[{arm.name}] {json.dumps(report, indent=2)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=[a.name for a in ARMS], default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=0.075)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evenly strided subset of the banked rows (smoke)",
    )
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"rig concat: {len(dataset)} frames", flush=True)

    results: dict[str, dict] = {}
    for arm in ARMS:
        if args.arm is not None and arm.name != args.arm:
            continue
        results[arm.name] = run_arm(
            arm,
            dataset,
            args.device,
            args.tolerance,
            args.limit,
        )
        # Free the 4.9B trunk before the next arm loads.
        torch.cuda.empty_cache()

    all_pass = all(
        r["pass_chunk_parity"] and r["pass_anchor_repro"] for r in results.values()
    )
    if args.limit is None and args.arm is None:
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"wrote {OUT_JSON}", flush=True)
    print(f"G2 e2e parity: {'PASS' if all_pass else 'FAIL'}", flush=True)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
