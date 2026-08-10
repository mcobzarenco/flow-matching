"""Rig fine-tune preflight P3 — convention/sign check, record-only anchors.

Pre-reg posts/2026-08-10-prereg-molmoact2-rig-finetune.md: run the
released ``allenai/MolmoAct2-SO100_101`` zero-shot on ~240 evenly
strided frames of the two SO-101 rig repos (v3.0) and measure, per
joint:

  * motion correlation corr(pred[t]-state, truth[t]-state) pooled over
    the 30-step window — a per-joint sign mirror between the model's
    native (v2.1-recorded) space and the rig's v3.0 space shows up as a
    strongly negative value here. HARD ABORT if any joint <= 0.
  * signed step-0 offset mean(pred[0]-truth[0]) — a convention offset
    is tens of units on ~100-unit joint ranges. HARD ABORT if
    |offset| > 0.5 * rig (q99-q01) span for that joint.
  * matched-window (30-step / 1.0 s) MAE, zero-shot vs state-copy —
    the anchors the fine-tune must beat (pre-reg expectation 2).

Reuses the panel predictor's model loading (bf16/processor patches) and
per-frame predict path verbatim; frames come from the rig repos via the
mainline loader (EpisodeSplit.ALL — no holdout; these are sanity
anchors, not generalization reads).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fontaine" / "scripts"))

import molmoact2_panel_predict as mp

RIG_REPOS = (
    Path("~/datasets/mcobzarenco/so101_pick_place_clean").expanduser(),
    Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
)
N_FRAMES = 240
HORIZON = mp.HORIZON  # 30 steps @ 30 fps = 1.0 s, their SO-100/101 tag
OUT_JSON = REPO_ROOT / "reports/analysis__molmoact2_rig_preflight.json"


def main() -> None:
    from bijou.data import EpisodeSplit, select_datasets

    selection = select_datasets(
        RIG_REPOS,
        (),
        HORIZON,
        episode_split=EpisodeSplit.ALL,
        allowed_fps=(30.0,),
        allowed_camera_counts=(2,),
    )
    dataset = selection.concat()
    print(
        f"rig selection: {len(selection.datasets)} datasets, {len(dataset)} frames",
        flush=True,
    )
    if len(selection.datasets) != 2:
        sys.exit("expected exactly the 2 rig repos — stop")

    stride = max(len(dataset) // N_FRAMES, 1)
    rows = np.arange(len(dataset))[::stride][:N_FRAMES]

    # Rig per-joint spans from the repos' own stats (count-weighted q99-q01).
    spans = np.zeros(6)
    weights = 0.0
    for repo in RIG_REPOS:
        stats = json.loads((repo / "meta/stats.json").read_text())["action"]
        count = float(np.asarray(stats["count"]).reshape(-1)[0])
        spans += count * (np.asarray(stats["q99"]) - np.asarray(stats["q01"]))
        weights += count
    spans /= weights

    model, processor, mode_kwarg = mp.load_model("cuda")

    preds = np.full((len(rows), HORIZON, 6), np.nan, np.float32)
    truths = np.full((len(rows), HORIZON, 6), np.nan, np.float32)
    states = np.zeros((len(rows), 6), np.float32)
    started = time.monotonic()
    for i, idx in enumerate(rows):
        item = dataset[int(idx)]
        state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
        action = item["action"].float().numpy()
        n = min(action.shape[0], HORIZON)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(mp.BASE_SEED + int(idx))
        with torch.inference_mode():
            out = model.predict_action(
                processor=processor,
                images=mp.frame_images(item),
                task=str(item["task"]),
                state=state,
                norm_tag=mp.NORM_TAG,
                enable_depth_reasoning=False,
                num_steps=mp.NUM_STEPS,
                generator=generator,
                normalize_language=True,
                enable_cuda_graph=True,
                **{mode_kwarg: "continuous"},
            )
        raw = out.actions if hasattr(out, "actions") else out
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (HORIZON, 6):
            sys.exit(f"prediction shape {actions.shape} != ({HORIZON}, 6) — stop")
        preds[i] = actions
        truths[i, :n] = action[:n]
        states[i] = state
        if (i + 1) % 40 == 0:
            rate = (i + 1) / max(time.monotonic() - started, 1e-6) * 60
            print(f"progress: {i + 1}/{len(rows)} ({rate:.1f} f/min)", flush=True)

    valid = np.isfinite(truths).all(-1)  # (N, H)
    report: dict = {"n_frames": len(rows), "horizon": HORIZON, "joints": []}
    hard_fail = []
    for d in range(6):
        mask = valid
        pred_motion = (preds[:, :, d] - states[:, None, d])[mask]
        true_motion = (truths[:, :, d] - states[:, None, d])[mask]
        corr = float(np.corrcoef(pred_motion, true_motion)[0, 1])
        offset0 = float(np.mean(preds[:, 0, d] - truths[:, 0, d]))
        # Convention-vs-collapse diagnostic: a v2.1/v3.0 convention offset
        # is state-independent (error ~ constant, corr(err, truth) ~ 0);
        # a model predicting a default posture has pred ~ const, so
        # corr(err, truth) ~ -1 with err std comparable to the truth std.
        err0 = preds[:, 0, d] - truths[:, 0, d]
        err_truth_corr = float(np.corrcoef(err0, truths[:, 0, d])[0, 1])
        row = {
            "joint": d,
            "motion_corr": round(corr, 4),
            "step0_signed_offset": round(offset0, 3),
            "step0_err_std": round(float(np.std(err0)), 3),
            "step0_err_truth_corr": round(err_truth_corr, 4),
            "pred0_std": round(float(np.std(preds[:, 0, d])), 3),
            "truth0_std": round(float(np.std(truths[:, 0, d])), 3),
            "rig_span_q01_q99": round(float(spans[d]), 2),
        }
        report["joints"].append(row)
        if not np.isfinite(corr) or corr <= 0.0:
            hard_fail.append(
                f"joint{d} motion corr {corr:.3f} <= 0 — sign mirror suspected",
            )
        # Amendment 1 (2026-08-10 16:4xZ, posted in-channel inside the
        # objection window): the half-span offset line is RECORD-ONLY, not
        # launch-blocking. Measured on joint1 it fired on posture-collapse
        # (pred0_std 2.0 vs truth0_std 44.8, err~truth corr -0.999) driven
        # by 97% of rig states saturating the released checkpoint's joint1
        # normalization range [43.7, 185.3] vs rig [-103, +67] — a per-joint
        # affine gap that rig-only q01/q99 absorbs, not a sign mirror.
        if abs(offset0) > 0.5 * spans[d]:
            row["offset_flag_record_only"] = (
                f"step-0 offset {offset0:.1f} exceeds half rig span {spans[d]:.1f}"
            )

    err = np.abs(preds - truths)
    copy_err = np.abs(states[:, None, :] - truths)
    w = valid[:, :, None]
    report["zero_shot_mae_matched"] = round(float(err[w.repeat(6, 2)].mean()), 4)
    report["state_copy_mae_matched"] = round(float(copy_err[w.repeat(6, 2)].mean()), 4)
    report["hard_failures"] = hard_fail

    np.savez_compressed(
        OUT_JSON.with_suffix(".npz"),
        preds=preds,
        truths=truths,
        states=states,
        rows=rows,
    )
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if hard_fail:
        sys.exit("PREFLIGHT P3 FAILED: " + "; ".join(hard_fail))
    print("PREFLIGHT P3 PASS", flush=True)


if __name__ == "__main__":
    main()
