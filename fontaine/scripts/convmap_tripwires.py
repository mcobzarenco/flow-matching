"""Pre-GPU tripwires for the release-in-sim convention-map arm
(pre-reg posts/2026-08-12-prereg-release-eval20-convmap.md; box note
fontaine/notes/molmoact2-unit-contracts-box-note.md).

Two mandatory gates before the 20-seed arm spends the GPU:

(a) WORKSPACE COVERAGE — the mapped seam workspace A(workspace) must
    land inside the release's q01/q99 box (the clamp travels with the
    model; outside = blind + unreachable). Prints the per-joint table
    plus, for any joint the gated fit leaves failing, the best-covering
    member of the discrete family — the evidence an override cites.

(b) FIRST-ACTION-VS-STATE — the note's unit-bug detector: with a correct
    map, the first decoded action sits near the current state
    (continuity of absolute-position control); a wrong sign/offset shows
    up as a ~90/180-degree single-joint error instantly. The ftrig
    contract checkpoint run through the SAME metric anchors the pass
    scale (sim-native state-copy analog).

Also prints the cross-check the box asked for: whether our seam fit
implies the same lift +180 / elbow +90 old-convention map its
fit_convention_map snapped on the curated panel.

Usage (GPU, ~2 min):
  MUJOCO_GL=egl uv run python fontaine/scripts/convmap_tripwires.py \
      --checkpoint ~/marius-convert-gate/converted/molmoact2_so100_101_release \
      --seam-stats ~/marius-convert-gate/converted/molmoact2_rig_r1_step2000 \
      --anchor-checkpoint ~/marius-convert-gate/converted/molmoact2_rig_r1_step2000 \
      [--convmap-override elbow_flex=90] [--skip-gpu]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import torch

from bijou.eval.molmo_norm import CONVENTION_OFFSETS
from bijou.rollout import SO_MOTORS
from sim.convmap import coverage_report, seam_convention_map

# The box's curated-panel snaps (the note's cross-checkable anchors):
# v3-convention data -> release table.
BOX_PANEL_SNAPS = {"shoulder_lift": 180.0, "elbow_flex": 90.0, "wrist_roll": 90.0}
# Gate (b) thresholds: fail on a single arm joint off by more than this
# (degree-family errors are 90/180 — huge), or a mean first-action
# delta more than this multiple of the contract anchor's.
JOINT_FAIL_DEG = 30.0
MEAN_FAIL_RATIO = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seam-stats", type=Path, required=True)
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        default=None,
        help="contract checkpoint run through the same first-action "
        "metric — the sim-native state-copy scale gate (b) compares to",
    )
    parser.add_argument("--convmap-override", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="tripwire (a) + cross-check only (pure CPU)",
    )
    return parser.parse_args()


def first_action_deltas(
    checkpoint: Path,
    seeds: int,
    seam_checkpoint: Path | None,
    overrides: list[str],
) -> np.ndarray:
    """[seeds, 6] |chunk[0] - state| in seam units — convmap-wrapped when
    ``seam_checkpoint`` is given, contract read otherwise."""
    from bijou.eval.molmo_norm import MolmoNorm
    from bijou.eval.policies import BijouPolicy
    from bijou.modelling.decoders.flow import SamplingMethod
    from sim.rollout_sim import STATS_REPO_ID, sim_item
    from sim.so101_sim import SO101Sim

    policy = BijouPolicy(
        checkpoint,
        device=torch.device("cuda"),
        seed=0,
        sample_steps=10,
        method=SamplingMethod.EULER,
        flow_decoder_dtype=torch.bfloat16,
        molmo_norm=(
            MolmoNorm.CONVENTION_MAP
            if seam_checkpoint is not None
            else MolmoNorm.CHECKPOINT
        ),
    )
    if seam_checkpoint is not None:
        seam = seam_convention_map(
            seam_checkpoint,
            policy.info.normalization,
            overrides,
        )
        policy._molmo_norm_maps["sim/eval100"] = seam.item_maps
        stats = seam.seam_stats
    else:
        stats = policy.info.per_dataset_normalization.get(
            STATS_REPO_ID,
            policy.info.normalization,
        )
    sim = SO101Sim()
    deltas = []
    for seed in range(seeds):
        obs = sim.reset(seed)
        item = sim_item(obs, seed, 0, stats=stats, chunk_size=policy.info.chunk_size)
        chunk = policy.predict([item], [0])[0].numpy()
        deltas.append(np.abs(chunk[0] - obs.state))
    del policy
    torch.cuda.empty_cache()
    return np.array(deltas)


def main() -> int:
    args = parse_args()
    from bijou.checkpoint import read_metadata

    table = read_metadata(args.checkpoint).stats
    seam = seam_convention_map(args.seam_stats, table, args.convmap_override)

    print("== fit ==")
    print(f"gated fit   scale {seam.fit.map.scale.tolist()}")
    print(f"gated fit   offset {seam.fit.map.offset.tolist()}")
    print(f"overrides   {seam.overrides or 'none'}")
    print(f"final map   offset {seam.map.offset.tolist()}")

    print("\n== cross-check vs box panel snaps ==")
    for joint, box_offset in BOX_PANEL_SNAPS.items():
        j = SO_MOTORS.index(joint)
        ours = float(seam.map.offset[j])
        gated = float(seam.fit.map.offset[j])
        agree = "AGREE" if abs(ours) == abs(box_offset) else "DISAGREE"
        print(
            f"{joint:13s} box panel {box_offset:+6.0f}  our final {ours:+6.0f} "
            f"(gated {gated:+6.0f})  {agree}",
        )

    print("\n== tripwire (a): workspace coverage ==")
    lines, failures = coverage_report(seam, table)
    for line in lines:
        print(line)
    assert seam.seam_stats.action_q01 is not None
    assert seam.seam_stats.action_q99 is not None
    assert table.action_q01 is not None
    assert table.action_q99 is not None
    for joint in failures:
        # The evidence an override would cite: uncovered fraction per
        # discrete-family offset (sign +1) for the failing joint.
        j = SO_MOTORS.index(joint)
        low, high = seam.seam_stats.action_q01[j], seam.seam_stats.action_q99[j]
        box = (table.action_q01[j], table.action_q99[j])
        alts = []
        for shift in CONVENTION_OFFSETS:
            m_low, m_high = low + shift, high + shift
            covered = max(0.0, min(m_high, box[1]) - max(m_low, box[0]))
            alts.append((1.0 - covered / (m_high - m_low), shift))
        ranked = ", ".join(f"{s:+.0f}: {u:.0%}" for u, s in sorted(alts))
        print(f"  {joint} family uncovered by offset -> {ranked}")

    if failures:
        print(f"\nTRIPWIRE (a) FAIL: {failures}")
        if not args.skip_gpu:
            print("(running tripwire (b) anyway — its verdict is the evidence)")
    else:
        print("\ntripwire (a) PASS")

    if args.skip_gpu:
        return 1 if failures else 0

    print(f"\n== tripwire (b): first-action-vs-state, {args.seeds} seeds ==")
    release = first_action_deltas(
        args.checkpoint,
        args.seeds,
        args.seam_stats,
        args.convmap_override,
    )
    anchor = None
    if args.anchor_checkpoint is not None:
        anchor = first_action_deltas(args.anchor_checkpoint, args.seeds, None, [])

    def report(name: str, deltas: np.ndarray) -> float:
        per_joint = deltas.mean(axis=0)
        mean_arm = float(per_joint[:5].mean())  # gripper excluded: task-driven
        print(f"{name}: per-joint mean |a0 - s| {np.round(per_joint, 2).tolist()}")
        print(f"{name}: arm-joint mean {mean_arm:.2f}")
        return mean_arm

    release_mean = report("release_convmap", release)
    verdict_fail = bool((release.mean(axis=0)[:5] > JOINT_FAIL_DEG).any())
    if anchor is not None:
        anchor_mean = report("contract anchor", anchor)
        verdict_fail |= release_mean > MEAN_FAIL_RATIO * max(anchor_mean, 1.0)
    if verdict_fail:
        print("\nTRIPWIRE (b) FAIL: unit-bug signature — do not spend the GPU")
    else:
        print("\ntripwire (b) PASS")
    return 1 if (failures or verdict_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
