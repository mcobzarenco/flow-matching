"""Wrist-view read of the arm material fixes — the registered 20x5
probe gate read (queue `sim-wrist-view-material-read`; pre-reg posted
before this runs).

The two banked opt-in material fixes (`arm_photometrics='v1'`,
`mount_material='v1'`) are MODEL-LEVEL material writes at init — they
change every camera, and the wrist camera sees arm surfaces up close.
The pending promotion asks would flip both flags together; this read
supplies the wrist-side fact instead of assuming it.

Two production v3 instances over the SAME (seed, draw) schedule — the
default materials and the two-flag stack — NO composite hook needed:
the wrist frame is a raw render (v3 wrist is bit-identical to v2 by the
registered wrist guard), so the production `reset()` observations pair
1:1 across instances directly:

  baseline instance: v3          (top + wrist)
  stack instance:    v3_stack    (arm_photometrics='v1' + mount_material='v1')

In-run oracles: the two instances' settled qpos bit-equal per slot (the
grades consume no RNG draws); per-slot wrist changed-pixel fraction
<= 5% (an RNG-stream divergence flips ~100% of pixels through the
sensor noise — the feasibility probe measured ~0.5% at |delta| 1-2).

Registered reads (pre-reg 2026-08-14): ABORT unless in-run v3 TOP knn5
AUROC in 0.713 +/- 0.005 (the established 20x5 protocol gate) AND
in-run v3 WRIST knn5 AUROC in [0.50, 0.60] (the banked reset-render
wrist baseline: 0.5442 / 0.5476 on two 100x1 reads; wider band — no
20x5 wrist anchor exists). PRIMARY: paired wrist dknn5 CI95 (10k
resamples, rng 0) of v3_stack vs v3 entirely below 0. Record-only:
top paired stack delta (the mount read's rider replication), wrist
clean anchor, graded-surface wrist visibility diagnostic.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_wrist_material_read.py \
      --out reports/analysis__sim_wrist_material_read.json
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
REAL_KEYS = {"top": "observation.images.front", "wrist": "observation.images.wrist"}
TOP_ABORT_BAND = (0.708, 0.718)  # registered: banked 20x5 anchor 0.713 +/- 0.005
WRIST_ABORT_BAND = (0.50, 0.60)  # registered: banked 100x1 resets 0.5442 / 0.5476
CHANGED_PX_MAX = 15_360  # 5% of 307,200 — RNG-divergence tripwire (measured ~0.5%)


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
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_clean").expanduser(),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump-frames", type=Path, default=None)
    return parser.parse_args()


def graded_material_geoms(sim) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Geom-id sets whose materials the two flags touch, on the DEFAULT
    model (pre-split — the detached gripper geom is byte-identical and
    excluded): pla links, sts3215 servos, camera-mount holders of the
    shared wrist-roll material."""
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
    """Per-class visible pixel counts in the RAW wrist segmentation
    (pre-remap render grid) — the surfaces the flags can move."""
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


def run_pass(
    sim,  # noqa: ANN001
    base_qpos: list[np.ndarray] | None,
    base_wrist: list[np.ndarray] | None,
    classes: dict[str, np.ndarray] | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list, dict]:
    tops: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    qpos_log: list[np.ndarray] = []
    visibility: list[dict[str, int]] = []
    changed = {"px_max": 0, "delta_max": 0, "px_per_slot": []}
    for index in range(N_SLOTS):
        seed, draw = index // N_DRAWS, index % N_DRAWS
        obs = sim.reset(seed, appearance_seed=1000 * draw + seed)
        tops.append(obs.top)
        wrists.append(obs.wrist)
        qpos_log.append(sim.data.qpos.copy())
        if classes is not None:
            visibility.append(wrist_graded_px(sim, classes))
        if base_qpos is not None and not np.array_equal(
            sim.data.qpos,
            base_qpos[index],
        ):
            raise SystemExit(
                f"seed {seed} draw {draw}: instances diverge in qpos — "
                "the grades must consume no RNG draws",
            )
        if base_wrist is not None:
            delta = np.abs(
                base_wrist[index].astype(np.int16) - obs.wrist.astype(np.int16),
            )
            n_diff = int((delta.max(axis=-1) > 0).sum())
            changed["px_per_slot"].append(n_diff)
            changed["px_max"] = max(changed["px_max"], n_diff)
            changed["delta_max"] = max(changed["delta_max"], int(delta.max()))
            if n_diff > CHANGED_PX_MAX:
                raise SystemExit(
                    f"seed {seed} draw {draw}: {n_diff} wrist px differ "
                    "(> 5% of frame) — RNG-divergence tripwire",
                )
    sim.renderer.close()
    return tops, wrists, qpos_log, visibility, changed


def main() -> int:
    args = parse_args()

    from sim.so101_sim import SO101Sim

    print("baseline pass (v3 default): 100 slots ...")
    base = SO101Sim(render_style="v3", post_backend="numpy")
    classes = graded_material_geoms(base)
    print({name: len(ids) for name, ids in classes.items()}, "graded geoms")
    base_tops, base_wrists, base_qpos, visibility, _ = run_pass(
        base,
        None,
        None,
        classes,
    )
    del base

    print("stack pass (arm_photometrics='v1' + mount_material='v1'): 100 slots ...")
    stack = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        arm_photometrics="v1",
        mount_material="v1",
    )
    stack_tops, stack_wrists, _, _, changed = run_pass(
        stack,
        base_qpos,
        base_wrists,
        None,
    )
    del stack
    print(
        f"wrist changed-px stats vs baseline: {changed['px_max']} px max, "
        f"|delta| max {changed['delta_max']}",
    )

    real: dict[str, list[np.ndarray]] = {}
    for camera in ("top", "wrist"):
        for group, root, count in (
            ("real_v2", args.v2_root, probe.N_REAL_V2),
            ("real_clean", args.clean_root, probe.N_REAL_CLEAN),
        ):
            files = sorted(
                (root / "videos" / REAL_KEYS[camera] / "chunk-000").glob("*.mp4"),
            )
            total = probe.total_frames(files)
            real[f"{group}_{camera}"] = probe.decode_strided(
                files,
                total // count,
                count,
            )
            print(f"{group}/{camera}: {count} frames (stride {total // count})")

    groups: dict[str, list[np.ndarray]] = {
        "v3_top": base_tops,
        "v3_stack_top": stack_tops,
        "v3_wrist": base_wrists,
        "v3_stack_wrist": stack_wrists,
        **real,
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
    scores: dict[str, np.ndarray] = {}
    for camera in ("top", "wrist"):
        ref = emb[f"real_v2_{camera}"][:half]
        held = knn5(emb[f"real_v2_{camera}"][half:], ref)
        arms = {
            f"v3_{camera}": knn5(emb[f"v3_{camera}"], ref),
            f"v3_stack_{camera}": knn5(emb[f"v3_stack_{camera}"], ref),
        }
        clean = knn5(emb[f"real_clean_{camera}"], ref)
        scores.update(arms)
        results[camera] = {
            "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
            "clean_anchor": arm_read(clean, held),
            "arms": {name: arm_read(s, held) for name, s in arms.items()},
            "paired_stack_vs_v3": paired_read(
                arms[f"v3_stack_{camera}"],
                arms[f"v3_{camera}"],
            ),
        }

    top_auroc = results["top"]["arms"]["v3_top"]["auroc_vs_real"]
    wrist_auroc = results["wrist"]["arms"]["v3_wrist"]["auroc_vs_real"]
    aborted = not (
        TOP_ABORT_BAND[0] <= top_auroc <= TOP_ABORT_BAND[1]
        and WRIST_ABORT_BAND[0] <= wrist_auroc <= WRIST_ABORT_BAND[1]
    )
    primary = results["wrist"]["paired_stack_vs_v3"]

    vis = {
        name: {
            "mean_px": float(np.mean([v[name] for v in visibility])),
            "max_px": int(np.max([v[name] for v in visibility])),
            "slots_visible": int(np.sum([v[name] > 0 for v in visibility])),
        }
        for name in ("pla", "servo", "mount")
    }
    results["primary_pass"] = bool(primary["ci95"][1] < 0)
    results["wrist_regression"] = bool(primary["ci95"][0] > 0)
    results["abort_gates"] = {
        "top_band": list(TOP_ABORT_BAND),
        "top_auroc": top_auroc,
        "wrist_band": list(WRIST_ABORT_BAND),
        "wrist_auroc": wrist_auroc,
        "aborted": aborted,
    }
    results["wrist_graded_visibility_raw_px"] = vis
    results["wrist_changed_px"] = {
        "px_max": changed["px_max"],
        "delta_max": changed["delta_max"],
        "px_mean": float(np.mean(changed["px_per_slot"])),
        "frame_px": 307_200,
    }
    results["context"] = {
        "rollout_wrist_baseline_knn5_auroc": 0.828,
        "reset_wrist_baselines_100x1": [0.5442, 0.5476],
        "mount_read_top_rider_ci95": [-2.45e-07, -0.57e-07],
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
            "protocol": "sim_encoder_ood_probe A/B split, top + wrist cameras; "
            "20 seeds x 5 appearance draws, settled resets, TWO paired "
            "production v3 instances (numpy post backend: default, "
            "arm_photometrics='v1' + mount_material='v1' stack); production "
            "reset() observations (no composite hook — wrist is a raw "
            "render); per-slot qpos bit-equality + wrist changed-px "
            "tripwire oracles",
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    for camera in ("top", "wrist"):
        for name in (f"v3_{camera}", f"v3_stack_{camera}"):
            read = results[camera]["arms"][name]
            print(
                f"{name}: knn5 {read['mean']:.3e} | AUROC {read['auroc_vs_real']:.3f}",
            )
        print(
            f"{camera} clean anchor AUROC "
            f"{results[camera]['clean_anchor']['auroc_vs_real']:.3f}",
        )
        print(f"{camera} paired stack vs v3: {results[camera]['paired_stack_vs_v3']}")
    print(f"graded visibility (raw wrist px): {vis}")
    print(f"wrote {args.out}")
    if aborted:
        print(
            f"ABORT: in-run anchors outside the registered bands — top "
            f"{top_auroc:.3f} vs {TOP_ABORT_BAND}, wrist {wrist_auroc:.3f} "
            f"vs {WRIST_ABORT_BAND} — no claims from this run",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
