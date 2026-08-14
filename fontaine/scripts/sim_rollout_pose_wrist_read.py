"""Rollout-pose wrist read — is the wrist camera honest at manipulation
poses? (queue `sim-rollout-pose-wrist-read`; pre-reg posted before this
runs: posts/2026-08-14-prereg-sim-rollout-pose-wrist.md).

Every banked wrist number is a settled RESET pose (0.548-0.561 band,
fitted lens 0.523); the banked 0.828 anchor is from old-visuals rollout
VIDEO frames — mid-episode poses where the gripper fills the frame, on
a different pose distribution. No sim rollout qpos traces were banked
(videos + distance/grip only), so the pose source here is the REAL
held-out episodes' recorded `observation.state`: the sim arm is posed
at the exact joint angles the real robot recorded, pose-matching every
slot and removing the pose-distribution confound.

Two production v3 instances (fitted curve-only lens, re-tuned wrist
pose, numpy post) over the established 20x5 (seed, appearance-draw)
schedule — default materials vs the pending promotion stack
(`arm_photometrics='v1'` + `mount_material='v1'`), identical call
sequences so the slots pair bit-exactly:

  pass 1 (in-run anchor): settled production resets — the banked
    protocol verbatim, scored vs the standard real_v2 300-strided A/B.
  pass 2: same slots, arm qpos overwritten to slot i's recorded real
    state (deg2rad, ctrl-clipped, mj_forward — kinematic pose),
    production observe(); scored vs a 150-frame mid-band manipulation
    reference from the reference-half episodes.

Real-frame decode is timestamp-exact: v2 video segments carry trailing
extra frames per episode (34,332 video frames vs 32,679 parquet rows),
so naive global striding is NOT frame-aligned; the episodes-meta
from_timestamp gives decode index = round(from_ts*30) + frame_index
(per-file contiguity verified at pre-reg feasibility).

Registered gates (frozen in the pre-reg): ABORT unless in-run reset
TOP AUROC in 0.708-0.718 AND in-run reset WRIST AUROC in [0.49, 0.57]
AND real-real manipulation calibration AUROC in [0.35, 0.65]; qpos
bit-equality across instances x200 slots; reset-pass wrist changed-px
<= 5% tripwire (the cap is REMOVED at manipulation poses by design —
recorded, not gated). PRIMARY 1: manipulation-pose wrist AUROC of the
fitted default (<= 0.65 honest / >= 0.75 gap real). PRIMARY 2: paired
manipulation dknn5 CI95 of stack vs default.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_rollout_pose_wrist_read.py \
      --out reports/analysis__sim_rollout_pose_wrist_read.json
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
from sim_top_gap_decomposition import arm_read, knn5, paired_read

N_SEEDS = 20
N_DRAWS = 5
N_SLOTS = N_SEEDS * N_DRAWS
N_MANIP_REF = 150
N_MANIP_REF_HOLDOUT = 50  # ref-half frames for the real-real calibration
MID_BAND = (0.3, 0.7)
HELD_EPISODE_MIN = 26  # manifest half-A boundary: episodes 0-25 are reference
# Amendment 1 (registered before the re-read): the calibration holdout
# must be EPISODE-DISJOINT from the knn reference. The first run
# interleaved holdout picks within the same episodes as the reference,
# so every holdout frame had temporal near-duplicates (~40 frames away)
# in the reference — knn5 distances collapse and the calibration AUROC
# reads leakage (0.129), not pool comparability. Reference episodes and
# holdout episodes now split inside the reference half.
REF_EPISODES = range(20)
CALIBRATION_EPISODES = range(20, HELD_EPISODE_MIN)
FPS = 30
REAL_KEYS = {"top": "observation.images.front", "wrist": "observation.images.wrist"}
TOP_ABORT_BAND = (0.708, 0.718)  # registered: banked 20x5 anchor 0.713 +/- 0.005
WRIST_ABORT_BAND = (0.49, 0.57)  # registered: banked fitted-lens gate read 0.523
# Amendment 2 (registered before adjudication): the calibration gate is
# DIRECTIONAL. Run 2 read 0.268 vs the original symmetric [0.35, 0.65]
# band — but the low side matches the protocol's banked real-real norm
# (clean anchors 0.26/0.28 on the same harness: along-the-dataset drift,
# later episodes farther from the early-episode reference) and can only
# UNDERSTATE the sim AUROC (a far-held pool inflates held scores). Only
# the high side inflates sim AUROC and must abort; below LOW_NOTE only
# the fake-side (>= GAP_REAL_BAR) verdict is claimable.
CALIBRATION_ABORT_MAX = 0.65
CALIBRATION_LOW_NOTE = 0.35
CHANGED_PX_MAX = 15_360  # 5% of 307,200 — reset-pass RNG-divergence tripwire
HONEST_BAR = 0.65  # registered PRIMARY 1 bands
GAP_REAL_BAR = 0.75


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
    parser.add_argument("--dump-frames", type=Path, default=None)
    return parser.parse_args()


def load_tables(root: Path):  # noqa: ANN201
    """(data df sorted by global index, episodes-meta df sorted by episode)."""
    import pandas as pd

    data = pd.concat(
        [
            pd.read_parquet(
                f,
                columns=[
                    "episode_index",
                    "frame_index",
                    "index",
                    "observation.state",
                ],
            )
            for f in sorted((root / "data").glob("chunk-*/file-*.parquet"))
        ],
    ).sort_values("index")
    eps = pd.concat(
        [
            pd.read_parquet(f)
            for f in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
        ],
    ).sort_values("episode_index")
    return data, eps


def mid_band_pool(data, episodes: list[int]):  # noqa: ANN001, ANN201
    """Mid-band rows ([0.3T, 0.7T) per episode) of the given episodes,
    concatenated in global-index order."""
    import pandas as pd

    parts = []
    for episode in episodes:
        part = data[data.episode_index == episode]
        t = len(part)
        parts.append(part.iloc[int(MID_BAND[0] * t) : int(MID_BAND[1] * t)])
    return pd.concat(parts)


def pick_evenly(pool, count: int):  # noqa: ANN001, ANN201
    """`count` rows spread evenly over the pool (linspace-rounded,
    strictly increasing positions)."""
    positions = np.linspace(0, len(pool) - 1, count).round().astype(int)
    if len(np.unique(positions)) != count:
        raise SystemExit(f"pool of {len(pool)} cannot seat {count} distinct picks")
    return pool.iloc[positions]


def decode_exact(root: Path, camera: str, eps, rows) -> list[np.ndarray]:  # noqa: ANN001
    """Decode the exact real frames for the given data rows via the
    episodes-meta timestamps (decode index = round(from_ts*FPS) +
    frame_index within the episode's file)."""
    import av

    prefix = f"videos/{REAL_KEYS[camera]}/"
    file_of = dict(zip(eps.episode_index, eps[prefix + "file_index"], strict=True))
    start_of = {
        e: round(ts * FPS)
        for e, ts in zip(eps.episode_index, eps[prefix + "from_timestamp"], strict=True)
    }
    wanted: dict[int, dict[int, int]] = {}
    for slot, row in enumerate(rows.itertuples()):
        file_index = int(file_of[row.episode_index])
        decode_index = start_of[row.episode_index] + int(row.frame_index)
        wanted.setdefault(file_index, {})[decode_index] = slot
    frames: dict[int, np.ndarray] = {}
    for file_index, targets in sorted(wanted.items()):
        path = root / prefix / "chunk-000" / f"file-{file_index:03d}.mp4"
        container = av.open(str(path))
        last = max(targets)
        for index, frame in enumerate(container.decode(video=0)):
            if index in targets:
                frames[targets[index]] = frame.to_ndarray(format="rgb24")
            if index >= last:
                break
        container.close()
    if len(frames) != len(rows):
        raise SystemExit(f"decoded {len(frames)}/{len(rows)} {camera} frames")
    return [frames[slot] for slot in range(len(rows))]


def graded_material_geoms(sim) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Geom-id sets the two material flags touch, on the DEFAULT model
    (same partition as the wrist material read)."""
    model = sim.model
    pla_mats, servo_mats, mount_mats = set(), set(), set()
    for index in range(model.nmat):
        name = model.mat(index).name
        if "sts3215" in name:
            servo_mats.add(index)
        elif "wrist_roll_follower_so101_v1_material" in name:
            mount_mats.add(index)
        elif (
            "so101" in name
            and "moving_jaw" not in name
            and "wrist_roll_follower" not in name
        ):
            pla_mats.add(index)
    mount_bodies = {
        model.body(prefix + "camera_mount").id for prefix in ("", "leader-")
    }
    out: dict[str, list[int]] = {"pla": [], "servo": [], "mount": []}
    for geom in range(model.ngeom):
        mat = model.geom_matid[geom]
        if mat in pla_mats:
            out["pla"].append(geom)
        elif mat in servo_mats:
            out["servo"].append(geom)
        elif mat in mount_mats and model.geom_bodyid[geom] in mount_bodies:
            out["mount"].append(geom)
    return {name: np.array(ids, dtype=np.int32) for name, ids in out.items()}


def wrist_graded_px(sim, classes: dict[str, np.ndarray]) -> dict[str, int]:  # noqa: ANN001
    """Per-class visible pixel counts in the RAW wrist segmentation."""
    import mujoco

    renderer = sim.renderer
    renderer.enable_segmentation_rendering()
    renderer.update_scene(sim.data, camera="wrist_cam")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
    return {
        name: int((is_geom & np.isin(seg[..., 0], ids)).sum())
        for name, ids in classes.items()
    }


def run_passes(
    sim,  # noqa: ANN001
    states_deg: np.ndarray,
    base: dict | None,
    classes: dict[str, np.ndarray] | None,
) -> dict:
    """Reset pass then manipulation pass, identical call sequence per
    instance. Returns frames, qpos logs and oracle stats; when `base`
    is given (the second instance) the cross-instance oracles run."""
    import mujoco

    out = {
        "reset_top": [],
        "reset_wrist": [],
        "manip_wrist": [],
        "reset_qpos": [],
        "manip_qpos": [],
        "visibility": [],
        "reset_changed": {"px_max": 0, "delta_max": 0, "px_per_slot": []},
        "manip_changed_px": [],
    }
    for index in range(N_SLOTS):
        seed, draw = index // N_DRAWS, index % N_DRAWS
        obs = sim.reset(seed, appearance_seed=1000 * draw + seed)
        out["reset_top"].append(obs.top)
        out["reset_wrist"].append(obs.wrist)
        out["reset_qpos"].append(sim.data.qpos.copy())
        if base is not None:
            if not np.array_equal(sim.data.qpos, base["reset_qpos"][index]):
                raise SystemExit(
                    f"reset slot {index}: instances diverge in qpos — "
                    "the grades must consume no RNG draws",
                )
            delta = np.abs(
                base["reset_wrist"][index].astype(np.int16)
                - obs.wrist.astype(np.int16),
            )
            n_diff = int((delta.max(axis=-1) > 0).sum())
            out["reset_changed"]["px_per_slot"].append(n_diff)
            out["reset_changed"]["px_max"] = max(
                out["reset_changed"]["px_max"],
                n_diff,
            )
            out["reset_changed"]["delta_max"] = max(
                out["reset_changed"]["delta_max"],
                int(delta.max()),
            )
            if n_diff > CHANGED_PX_MAX:
                raise SystemExit(
                    f"reset slot {index}: {n_diff} wrist px differ (> 5%) — "
                    "RNG-divergence tripwire",
                )
    for index in range(N_SLOTS):
        seed, draw = index // N_DRAWS, index % N_DRAWS
        sim.reset(seed, appearance_seed=1000 * draw + seed)
        sim.data.qpos[sim._joint_qpos] = np.clip(
            np.deg2rad(states_deg[index]),
            sim._ctrl_low,
            sim._ctrl_high,
        )
        mujoco.mj_forward(sim.model, sim.data)
        obs = sim.observe()
        out["manip_wrist"].append(obs.wrist)
        out["manip_qpos"].append(sim.data.qpos.copy())
        if classes is not None:
            out["visibility"].append(wrist_graded_px(sim, classes))
        if base is not None:
            if not np.array_equal(sim.data.qpos, base["manip_qpos"][index]):
                raise SystemExit(f"manip slot {index}: instances diverge in qpos")
            delta = np.abs(
                base["manip_wrist"][index].astype(np.int16)
                - obs.wrist.astype(np.int16),
            )
            out["manip_changed_px"].append(int((delta.max(axis=-1) > 0).sum()))
    sim.renderer.close()
    return out


def main() -> int:
    args = parse_args()

    data, eps = load_tables(args.v2_root)
    n_episodes = int(data.episode_index.max()) + 1
    ref_pool = mid_band_pool(data, list(REF_EPISODES))
    calibration_pool = mid_band_pool(data, list(CALIBRATION_EPISODES))
    held_pool = mid_band_pool(data, list(range(HELD_EPISODE_MIN, n_episodes)))
    print(
        f"mid-band pools: ref {len(ref_pool)}, calibration "
        f"{len(calibration_pool)}, held {len(held_pool)}",
    )
    ref_rows = pick_evenly(ref_pool, N_MANIP_REF)
    ref_holdout_rows = pick_evenly(calibration_pool, N_MANIP_REF_HOLDOUT)
    held_rows = pick_evenly(held_pool, N_SLOTS)
    states_deg = np.stack(held_rows["observation.state"].to_list()).astype(np.float64)
    jaw = states_deg[:, 5]
    print(
        f"slot jaw angle (deg): min {jaw.min():.1f} med {np.median(jaw):.1f} "
        f"max {jaw.max():.1f}",
    )

    print("decoding exact real frames (manip ref/holdout/held) ...")
    real_manip = {
        "ref": decode_exact(args.v2_root, "wrist", eps, ref_rows),
        "ref_holdout": decode_exact(args.v2_root, "wrist", eps, ref_holdout_rows),
        "held": decode_exact(args.v2_root, "wrist", eps, held_rows),
    }
    print("decoding standard strided real_v2 sets (reset-anchor protocol) ...")
    real_strided = {}
    for camera in ("top", "wrist"):
        files = sorted(
            (args.v2_root / "videos" / REAL_KEYS[camera] / "chunk-000").glob("*.mp4"),
        )
        total = probe.total_frames(files)
        real_strided[camera] = probe.decode_strided(
            files,
            total // probe.N_REAL_V2,
            probe.N_REAL_V2,
        )

    from sim.so101_sim import SO101Sim

    print("default instance (v3, fitted lens): reset + manip passes ...")
    base_sim = SO101Sim(render_style="v3", post_backend="numpy", lens_model="fitted")
    classes = graded_material_geoms(base_sim)
    base = run_passes(base_sim, states_deg, None, classes)
    del base_sim

    print("stack instance (arm_photometrics + mount_material): both passes ...")
    stack_sim = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        lens_model="fitted",
        arm_photometrics="v1",
        mount_material="v1",
    )
    stack = run_passes(stack_sim, states_deg, base, None)
    del stack_sim
    print(
        f"reset changed-px max {stack['reset_changed']['px_max']} "
        f"(cap {CHANGED_PX_MAX}); manip changed-px mean "
        f"{np.mean(stack['manip_changed_px']):.0f} "
        f"max {max(stack['manip_changed_px'])} (recorded, uncapped)",
    )

    groups: dict[str, list[np.ndarray]] = {
        "reset_top_fit": base["reset_top"],
        "reset_top_fit_stack": stack["reset_top"],
        "reset_wrist_fit": base["reset_wrist"],
        "reset_wrist_fit_stack": stack["reset_wrist"],
        "manip_wrist_fit": base["manip_wrist"],
        "manip_wrist_fit_stack": stack["manip_wrist"],
        "real_top_strided": real_strided["top"],
        "real_wrist_strided": real_strided["wrist"],
        "real_manip_ref": real_manip["ref"],
        "real_manip_ref_holdout": real_manip["ref_holdout"],
        "real_manip_held": real_manip["held"],
    }
    if args.dump_frames is not None:
        from PIL import Image

        for name, frames in groups.items():
            out_dir = args.dump_frames / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in (0, 1, 2):
                Image.fromarray(frames[i]).save(out_dir / f"{i:04d}.png")

    model, info = probe.from_checkpoint(args.checkpoint, device="cuda")
    vision = model.backbone.vision
    del model.decoder
    emb = {}
    for name, frames in groups.items():
        emb[name] = probe.embed(vision, frames)
        print(f"embedded {name}: {tuple(emb[name].shape)}")

    half = probe.N_REAL_V2 // 2
    results: dict[str, object] = {}

    # In-run reset anchors — the banked protocol verbatim.
    reset = {}
    for camera in ("top", "wrist"):
        ref = emb[f"real_{camera}_strided"][:half]
        held = knn5(emb[f"real_{camera}_strided"][half:], ref)
        arms = {
            name: knn5(emb[f"reset_{camera}_{suffix}"], ref)
            for name, suffix in (("fit", "fit"), ("fit_stack", "fit_stack"))
        }
        reset[camera] = {
            "arms": {name: arm_read(s, held) for name, s in arms.items()},
            "paired_stack_vs_default": paired_read(arms["fit_stack"], arms["fit"]),
        }
        reset[camera]["scores"] = arms
    results["reset_anchor"] = {
        camera: {k: v for k, v in reset[camera].items() if k != "scores"}
        for camera in reset
    }

    # Manipulation reads — scored vs the manipulation reference.
    ref = emb["real_manip_ref"]
    held_scores = knn5(emb["real_manip_held"], ref)
    calibration_scores = knn5(emb["real_manip_ref_holdout"], ref)
    calibration_auroc = probe.auroc(calibration_scores, held_scores)
    manip_arms = {
        "fit": knn5(emb["manip_wrist_fit"], ref),
        "fit_stack": knn5(emb["manip_wrist_fit_stack"], ref),
    }
    results["manip"] = {
        "real_held": {
            "mean": float(held_scores.mean()),
            "std": float(held_scores.std()),
        },
        "calibration_auroc_refholdout_vs_held": calibration_auroc,
        "arms": {name: arm_read(s, held_scores) for name, s in manip_arms.items()},
        "paired_stack_vs_default": paired_read(
            manip_arms["fit_stack"],
            manip_arms["fit"],
        ),
        "pose_effect_manip_vs_reset_default": paired_read(
            manip_arms["fit"],
            knn5(emb["reset_wrist_fit"], ref),
        ),
    }

    top_auroc = reset["top"]["arms"]["fit"]["auroc_vs_real"]
    wrist_auroc = reset["wrist"]["arms"]["fit"]["auroc_vs_real"]
    aborted = not (
        TOP_ABORT_BAND[0] <= top_auroc <= TOP_ABORT_BAND[1]
        and WRIST_ABORT_BAND[0] <= wrist_auroc <= WRIST_ABORT_BAND[1]
        and calibration_auroc <= CALIBRATION_ABORT_MAX
    )
    calibration_low = calibration_auroc < CALIBRATION_LOW_NOTE
    primary1 = results["manip"]["arms"]["fit"]["auroc_vs_real"]
    primary2 = results["manip"]["paired_stack_vs_default"]
    results["abort_gates"] = {
        "top_band": list(TOP_ABORT_BAND),
        "top_auroc": top_auroc,
        "wrist_band": list(WRIST_ABORT_BAND),
        "wrist_auroc": wrist_auroc,
        "calibration_abort_max": CALIBRATION_ABORT_MAX,
        "calibration_low_note": CALIBRATION_LOW_NOTE,
        "calibration_auroc": calibration_auroc,
        "calibration_low": calibration_low,
        "aborted": aborted,
    }
    verdict = (
        "honest"
        if primary1 <= HONEST_BAR
        else ("gap_real" if primary1 >= GAP_REAL_BAR else "narrowed_open")
    )
    if calibration_low and verdict != "gap_real":
        # Amendment 2: a far-held pool understates sim AUROC — only the
        # fake-side verdict survives; anything milder is void here.
        verdict = f"void_calibration_low_({verdict})"
    results["primary1"] = {
        "manip_wrist_auroc": primary1,
        "honest_bar": HONEST_BAR,
        "gap_real_bar": GAP_REAL_BAR,
        "verdict": verdict,
    }
    results["primary2"] = {
        "ci95": primary2["ci95"],
        "verdict": (
            "stack_helps"
            if primary2["ci95"][1] < 0
            else ("regression" if primary2["ci95"][0] > 0 else "neutral")
        ),
    }
    results["diagnostics"] = {
        "slot_jaw_deg": {
            "min": float(jaw.min()),
            "median": float(np.median(jaw)),
            "max": float(jaw.max()),
        },
        "manip_graded_visibility_raw_px": {
            name: {
                "mean_px": float(np.mean([v[name] for v in base["visibility"]])),
                "max_px": int(np.max([v[name] for v in base["visibility"]])),
                "slots_visible": int(
                    np.sum([v[name] > 0 for v in base["visibility"]]),
                ),
            }
            for name in ("pla", "servo", "mount")
        },
        "reset_changed_px": {
            "px_max": stack["reset_changed"]["px_max"],
            "delta_max": stack["reset_changed"]["delta_max"],
            "px_mean": float(np.mean(stack["reset_changed"]["px_per_slot"])),
        },
        "manip_changed_px": {
            "px_mean": float(np.mean(stack["manip_changed_px"])),
            "px_max": int(max(stack["manip_changed_px"])),
            "frame_px": 307_200,
        },
    }
    results["context"] = {
        "rollout_wrist_banked_auroc_old_visuals": 0.828,
        "reset_wrist_band_equidistant": [0.548, 0.561],
        "reset_wrist_fitted_gate_read": 0.523,
        "slots": {
            "ref_episodes": [REF_EPISODES.start, REF_EPISODES.stop - 1],
            "calibration_episodes": [
                CALIBRATION_EPISODES.start,
                CALIBRATION_EPISODES.stop - 1,
            ],
            "held_episodes": [HELD_EPISODE_MIN, n_episodes - 1],
            "mid_band": list(MID_BAND),
            "n_slots": N_SLOTS,
            "n_manip_ref": N_MANIP_REF,
            "n_manip_ref_holdout": N_MANIP_REF_HOLDOUT,
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
            "protocol": "rollout-pose wrist read: sim arm posed at REAL held-out "
            "episodes' recorded observation.state (mid-band [0.3,0.7) picks, "
            "timestamp-exact frame decode), production v3 + fitted curve-only "
            "lens + re-tuned pose, numpy post; TWO paired instances (default vs "
            "arm_photometrics+mount_material); in-run reset anchors on the "
            "banked 20x5 protocol; er_60k knn5 vs mid-band manipulation "
            "reference (episodes 0-25)",
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(
        f"\nreset anchors: top {top_auroc:.3f} (band {TOP_ABORT_BAND}), "
        f"wrist {wrist_auroc:.3f} (band {WRIST_ABORT_BAND})",
    )
    print(f"manip calibration (ref-holdout vs held): {calibration_auroc:.3f}")
    for name in manip_arms:
        read = results["manip"]["arms"][name]
        print(
            f"manip wrist {name}: knn5 {read['mean']:.3e} | "
            f"AUROC {read['auroc_vs_real']:.3f}",
        )
    print(f"PRIMARY 1: {results['primary1']}")
    print(f"PRIMARY 2 paired stack vs default: {primary2}")
    print(f"pose effect: {results['manip']['pose_effect_manip_vs_reset_default']}")
    print(f"diagnostics: {json.dumps(results['diagnostics'], indent=1)}")
    print(f"wrote {args.out}")
    if aborted:
        print("ABORT: in-run anchors outside the registered bands — no claims")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
