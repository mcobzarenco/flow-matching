"""Arm photometric fix — the registered 20x5 probe gate read (queue
`sim-arm-photometric-links`; pre-reg posted before this runs).

Two production v3 instances over the SAME (seed, draw) schedule — the
default materials and the arm_photometrics='v1' fitted grade — hooked at
``_composite`` exactly like the arm-split leg, so every arm shares
physics/plate/lighting/noise per slot and the paired delta IS the
material change's visible-pixel effect:

  baseline instance: v3, plate_only, only_links
  patched instance:  v3_photo, only_links_photo

In-run oracles: hooked v3/v3_photo bit-exact vs their own production
observation; the two instances' settled qpos bit-equal per slot (the
grade consumes no RNG draws); frames bit-equal outside the dilated
arm-class mask (the material change is arm-local).

Registered reads (pre-reg 2026-08-14): ABORT unless in-run v3 AUROC in
0.713 +/- 0.005. PRIMARY: paired dknn5 CI95 (10k resamples, rng 0) of
v3_photo vs v3 entirely below 0 (toward real). MECHANISM: same for
only_links_photo vs only_links. Anchors: plate_only, arm-split
only_links 0.705, no_mount removal best 0.654, real_fg 0.328.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_arm_photometric_read.py \
      --out reports/analysis__sim_arm_photometric_read.json
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
from sim_arm_split import arm_subclasses
from sim_fg_content_split import class_geoms
from sim_top_gap_decomposition import arm_read, dilate, knn5, paired_read

N_SEEDS = 20
N_DRAWS = 5
N_SLOTS = N_SEEDS * N_DRAWS
TOP_KEY = "observation.images.front"
V3_ABORT_BAND = (0.708, 0.718)  # registered: banked anchor 0.713 +/- 0.005
ARM_HALO_DILATE = 16  # PSF + fractional remap-edge margin for the locality oracle


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


def hook(sim, subsets: dict[str, np.ndarray]):  # noqa: ANN001, ANN201
    """The arm-split composite hook verbatim: per top composite, re-run
    the composite for each subset with the noise RNG state restored;
    returns (arms, arm_mask_store)."""
    import mujoco

    arms: dict[str, list[np.ndarray]] = {
        name: [] for name in subsets if name != "__armclass__"
    }
    stash: dict[str, np.ndarray] = {}
    arm_masks: list[np.ndarray] = []
    armclass = subsets["__armclass__"]

    def mask_hook(camera: str) -> np.ndarray:
        renderer = sim.renderer
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera=camera)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        if camera == "top_cam":
            stash["seg"] = seg
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        return (is_geom & np.isin(seg[..., 0], sim._dynamic_geoms)).astype(np.float64)

    orig_composite = sim._composite

    def composite_hook(frame, mask, camera, shadow=None) -> np.ndarray:  # noqa: ANN001
        if camera != "top":
            return orig_composite(frame, mask, camera, shadow=shadow)
        seg = stash.pop("seg")
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        arm_masks.append(
            sim._remap(
                (is_geom & np.isin(seg[..., 0], armclass)).astype(np.float64)[
                    ...,
                    None,
                ],
            )[..., 0]
            > 0.02,
        )
        state = sim._noise_rng.bit_generator.state
        for name, subset in subsets.items():
            if name == "__armclass__":
                continue
            sim._noise_rng.bit_generator.state = state
            sub_mask = (is_geom & np.isin(seg[..., 0], subset)).astype(np.float64)
            arms[name].append(orig_composite(frame, sub_mask, camera, shadow=shadow))
        sim._noise_rng.bit_generator.state = state
        return orig_composite(frame, mask, camera, shadow=shadow)

    sim._render_mask = mask_hook
    sim._composite = composite_hook
    return arms, arm_masks


def main() -> int:
    args = parse_args()

    from sim.so101_sim import SO101Sim

    scout = SO101Sim(render_style="v3", post_backend="numpy")
    classes = class_geoms(scout)
    sub = arm_subclasses(scout, classes["arm"])
    links = sub["links"]
    armclass = classes["arm"]
    full = scout._dynamic_geoms
    empty = np.array([], dtype=full.dtype)
    del scout

    # Sequential passes — one live GL context at a time (two concurrent
    # mujoco EGL renderers crash the process). Pairing survives: same
    # (seed, appearance) schedule is bit-deterministic, and the baseline
    # pass banks qpos/frames/halos for the cross-instance oracles.
    print("baseline pass: 3 arms x 100 slots ...")
    base = SO101Sim(render_style="v3", post_backend="numpy")
    base_arms, base_masks = hook(
        base,
        {
            "v3": full,
            "plate_only": empty,
            "only_links": links,
            "__armclass__": armclass,
        },
    )
    slots = []
    base_qpos: list[np.ndarray] = []
    outside_masks: list[np.ndarray] = []
    for seed in range(N_SEEDS):
        for draw in range(N_DRAWS):
            appearance = 1000 * draw + seed
            obs_base = base.reset(seed, appearance_seed=appearance)
            if not np.array_equal(obs_base.top, base_arms["v3"][-1]):
                raise SystemExit(
                    f"seed {seed} draw {draw}: baseline hook not bit-exact",
                )
            base_qpos.append(base.data.qpos.copy())
            halo = base_masks[-1].astype(np.float64)
            for _ in range(ARM_HALO_DILATE):
                halo = dilate(halo)
            outside_masks.append(halo < 0.5)
            slots.append({"seed": seed, "draw": draw})
    base.renderer.close()
    del base

    print("patched pass: 2 arms x 100 slots ...")
    patched = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        arm_photometrics="v1",
    )
    patched_arms, _ = hook(
        patched,
        {"v3_photo": full, "only_links_photo": links, "__armclass__": armclass},
    )
    for index, slot in enumerate(slots):
        appearance = 1000 * slot["draw"] + slot["seed"]
        obs_patch = patched.reset(slot["seed"], appearance_seed=appearance)
        if not np.array_equal(obs_patch.top, patched_arms["v3_photo"][-1]):
            raise SystemExit(f"slot {slot}: patched hook not bit-exact")
        if not np.array_equal(patched.data.qpos, base_qpos[index]):
            raise SystemExit(
                f"slot {slot}: instances diverge in qpos — "
                "the grade must consume no RNG draws",
            )
        outside = outside_masks[index]
        if not np.array_equal(
            base_arms["v3"][index][outside],
            obs_patch.top[outside],
        ):
            raise SystemExit(
                f"slot {slot}: frames differ outside the dilated arm "
                "mask — the material change must be arm-local",
            )
    patched.renderer.close()
    del patched

    real: dict[str, list[np.ndarray]] = {}
    for group, root, count in (
        ("real_v2", args.v2_root, probe.N_REAL_V2),
        ("real_clean", args.clean_root, probe.N_REAL_CLEAN),
    ):
        files = sorted((root / "videos" / TOP_KEY / "chunk-000").glob("*.mp4"))
        total = probe.total_frames(files)
        real[group] = probe.decode_strided(files, total // count, count)
        print(f"{group}: {count} frames (stride {total // count})")

    groups: dict[str, list[np.ndarray]] = {**base_arms, **patched_arms, **real}
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
    ref = emb["real_v2"][:half]
    held = knn5(emb["real_v2"][half:], ref)
    sim_arms = {**base_arms, **patched_arms}
    scores = {name: knn5(emb[name], ref) for name in sim_arms}
    clean = knn5(emb["real_clean"], ref)
    arms_read = {name: arm_read(scores[name], held) for name in sim_arms}
    clean_read = arm_read(clean, held)

    v3_auroc = arms_read["v3"]["auroc_vs_real"]
    aborted = not V3_ABORT_BAND[0] <= v3_auroc <= V3_ABORT_BAND[1]

    primary = paired_read(scores["v3_photo"], scores["v3"])
    mechanism = paired_read(scores["only_links_photo"], scores["only_links"])
    results = {
        "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
        "clean_anchor": clean_read,
        "arms": arms_read,
        "primary_v3_photo_vs_v3": primary,
        "mechanism_only_links_photo_vs_only_links": mechanism,
        "primary_pass": bool(primary["ci95"][1] < 0),
        "mechanism_pass": bool(mechanism["ci95"][1] < 0),
        "context": {
            "only_links_photo_vs_plate_only": paired_read(
                scores["only_links_photo"],
                scores["plate_only"],
            ),
            "arm_split_only_links_auroc": 0.705,
            "arm_split_no_mount_removal_best": 0.654,
            "decomposition_real_fg": 0.328,
        },
        "v3_abort_gate": {
            "band": list(V3_ABORT_BAND),
            "v3_auroc": v3_auroc,
            "aborted": aborted,
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
            "protocol": "sim_encoder_ood_probe A/B split, top camera; 20 seeds "
            "x 5 appearance draws, settled resets, TWO paired production v3 "
            "instances (numpy post backend, default vs arm_photometrics='v1') "
            "hooked at _composite; per-slot qpos/outside-arm bit-equality "
            "oracles; noise-RNG state restored per arm",
            "grade": SO101Sim.ARM_PHOTOMETRICS_V1,
            "slots": slots,
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    for name in sim_arms:
        print(
            f"{name}: knn5 {scores[name].mean():.3e} | "
            f"AUROC {arms_read[name]['auroc_vs_real']:.3f}",
        )
    print(f"clean anchor AUROC {clean_read['auroc_vs_real']:.3f}")
    print(f"PRIMARY v3_photo vs v3: {primary}")
    print(f"MECHANISM only_links_photo vs only_links: {mechanism}")
    print(f"wrote {args.out}")
    if aborted:
        print(
            f"ABORT: in-run v3 AUROC {v3_auroc:.3f} outside the registered "
            f"band {V3_ABORT_BAND} — no claims from this run",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
