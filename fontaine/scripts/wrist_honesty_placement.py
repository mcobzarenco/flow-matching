"""Stage-0 honesty placement — W1/W3 on the 100 manipulation-pose
slots (wrist-transfer screen, pre-reg
posts/2026-08-14-prereg-wrist-transfer-screen.md §1 + §3).

Every wrist transform is applied to the 100 manipulation-pose wrist
slots of the rollout-pose read (run-3 pass-2 pose protocol: sim arm
posed at the real held-out episodes' recorded `observation.state`,
identical (seed, appearance-draw) schedule) rendered on **the
screen's serving substrate** — production v3, `lens_model
"equidistant"` (the rollout drivers' default; the fitted lens is a
probe-side leg, and `wrist_arm_mask` is registered on the deployed
path by design) — and scored with the established er_60k knn5
harness against the 150-frame mid-band manipulation reference.
Substrate note (rides the stage-0 boundary post): the banked 0.877
manipulation anchor was measured on the FITTED leg; the placement
gate is PAIRED within-run (W3−W0, W1−W0 on identical slots), so the
in-run W0 read is the anchor that matters here and the fitted number
is context only.

Registered placement gate (§3, frozen): W1 (blackout) and W3
(arm_blur) place LESS honest than W0 (classic render) on the 100-slot
read — W3−W0 per-slot knn5 delta positive with CI95 excluding zero.
A corruption the encoder can't see isn't a treatment.

W3 uses the real per-pose `wrist_arm_mask()` at the posed state via
ArmBlurTransform (a FRESH instance per slot — the per-episode
contract), so this read scores the exact constant (sigma) stage 1
will run.

In-run anchor: W0's AUROC should sit near the banked 0.877 (recorded,
not gated — the placement deltas are paired within-run).

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/wrist_honesty_placement.py \
      --out reports/analysis__wrist_honesty_placement.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import sim_encoder_ood_probe as probe
import sim_rollout_pose_wrist_read as pose
from sim_top_gap_decomposition import arm_read, knn5, paired_read


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "~/checkpoints/er_60k/fontaine_molmo2_er_60k_ddp4/step_060000",
        ).expanduser(),
    )
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data, eps = pose.load_tables(args.v2_root)
    n_episodes = int(data.episode_index.max()) + 1
    ref_pool = pose.mid_band_pool(data, list(pose.REF_EPISODES))
    held_pool = pose.mid_band_pool(
        data,
        list(range(pose.HELD_EPISODE_MIN, n_episodes)),
    )
    ref_rows = pose.pick_evenly(ref_pool, pose.N_MANIP_REF)
    held_rows = pose.pick_evenly(held_pool, pose.N_SLOTS)
    states_deg = np.stack(held_rows["observation.state"].to_list()).astype(
        np.float64,
    )

    print("decoding exact real frames (manip ref + held) ...")
    real_ref = pose.decode_exact(args.v2_root, "wrist", eps, ref_rows)
    real_held = pose.decode_exact(args.v2_root, "wrist", eps, held_rows)

    import mujoco

    from sim.so101_sim import SO101Sim
    from sim.wrist_transform import ArmBlurTransform, make_wrist_transform

    print("serving instance (v3, equidistant wrist): manip pass + transforms ...")
    sim = SO101Sim(render_style="v3", post_backend="numpy")
    blackout = make_wrist_transform("blackout", sim)
    assert blackout is not None
    w0_frames: list[np.ndarray] = []
    w1_frames: list[np.ndarray] = []
    w3_frames: list[np.ndarray] = []
    coverage: list[float] = []
    for index in range(pose.N_SLOTS):
        seed, draw = index // pose.N_DRAWS, index % pose.N_DRAWS
        sim.reset(seed, appearance_seed=1000 * draw + seed)
        sim.data.qpos[sim._joint_qpos] = np.clip(
            np.deg2rad(states_deg[index]),
            sim._ctrl_low,
            sim._ctrl_high,
        )
        mujoco.mj_forward(sim.model, sim.data)
        obs = sim.observe()
        w0_frames.append(obs.wrist)
        w1_frames.append(blackout(obs).wrist)
        arm_blur = ArmBlurTransform(sim)  # fresh per slot (per-episode contract)
        w3_frames.append(arm_blur(obs).wrist)
        coverage.append(arm_blur.coverage[0])
    sim.renderer.close()
    print(
        f"W3 mask coverage over {pose.N_SLOTS} slots: "
        f"mean {float(np.mean(coverage)):.3f} min {min(coverage):.3f} "
        f"max {max(coverage):.3f}",
    )

    model, info = probe.from_checkpoint(args.checkpoint, device="cuda")
    vision = model.backbone.vision
    del model.decoder
    groups = {
        "real_manip_ref": real_ref,
        "real_manip_held": real_held,
        "manip_wrist_w0": w0_frames,
        "manip_wrist_w1_blackout": w1_frames,
        "manip_wrist_w3_arm_blur": w3_frames,
    }
    emb = {}
    for name, frames in groups.items():
        emb[name] = probe.embed(vision, frames)
        print(f"embedded {name}: {tuple(emb[name].shape)}")

    ref = emb["real_manip_ref"]
    held_scores = knn5(emb["real_manip_held"], ref)
    arms = {
        name: knn5(emb[f"manip_wrist_{name}"], ref)
        for name in ("w0", "w1_blackout", "w3_arm_blur")
    }
    reads = {name: arm_read(s, held_scores) for name, s in arms.items()}
    w3_vs_w0 = paired_read(arms["w3_arm_blur"], arms["w0"])
    w1_vs_w0 = paired_read(arms["w1_blackout"], arms["w0"])

    # The §3 placement gate: W3−W0 positive, CI95 excluding zero; W1
    # also places less honest than W0 (its delta positive, CI-excl-0
    # read alongside — the bracket endpoint must be a visible
    # treatment too).
    w3_pass = w3_vs_w0["ci95"][0] > 0
    w1_pass = w1_vs_w0["ci95"][0] > 0
    gate = "PASS" if (w3_pass and w1_pass) else "FAIL"

    results = {
        "arms": reads,
        "paired_w3_vs_w0": w3_vs_w0,
        "paired_w1_vs_w0": w1_vs_w0,
        "w3_mask_coverage": {
            "mean": float(np.mean(coverage)),
            "min": float(min(coverage)),
            "max": float(max(coverage)),
        },
        "placement_gate": {
            "rule": "W3-W0 per-slot knn5 delta positive CI95-excl-0 AND "
            "W1-W0 positive CI95-excl-0 (pre-reg §3: a corruption the "
            "encoder can't see isn't a treatment)",
            "w3_pass": w3_pass,
            "w1_pass": w1_pass,
            "verdict": gate,
        },
        "context": {
            "banked_manip_anchor_w0": 0.877,
            "banked_reset_fitted": 0.523,
            "real_held_mean": float(held_scores.mean()),
        },
    }
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "backbone": info.backbone,
            "protocol": "stage-0 honesty placement: run-3 pass-2 manip slots "
            "(production v3 + fitted lens + numpy post, 20x5 schedule), W1 "
            "blackout + W3 arm_blur applied to the posed wrist frames "
            "(fresh ArmBlurTransform per slot, real wrist_arm_mask), er_60k "
            "knn5 vs the 150-frame mid-band manipulation reference",
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(results["arms"], indent=1))
    print(
        f"W3-W0 CI95 {w3_vs_w0['ci95']} | W1-W0 CI95 {w1_vs_w0['ci95']} | "
        f"PLACEMENT: {gate}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
