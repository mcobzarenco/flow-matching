"""Arm TRUE surface texture — the registered 20x5 probe gate read
(queue `sim-arm-surface-texture-mjspec`; pre-reg posted before this
runs).

Two production v3 instances over the SAME (seed, draw) schedule — the
arm_photometrics='v1' grade alone and the grade + arm_texture='v2'
mjSpec surface texture — hooked at ``_composite`` exactly like the
micro-texture read, so every arm shares physics/plate/lighting/noise
per slot and the paired delta IS the texture's visible-pixel effect:

  baseline instance: v3_photo, plate_only, only_links_photo
  patched instance:  v3_surf, only_links_surf

In-run oracles: hooked v3_photo/v3_surf bit-exact vs their own
production observation; the two instances' settled qpos bit-equal per
slot (the recompiled model is physics-identical, oracle-pinned in
tests/test_arm_texture.py); frames bit-equal outside the dilated
arm-class mask EXCEPT the registered REFLECTION RIDER — a true surface
texture rides the tabletop's planar reflection of the arm
(mat_reflectance 0.02, mechanism confirmed 2026-08-14 by zeroing it),
so diffs outside the arm halo are allowed ONLY on the dilated
reflective-geom mask, magnitude <= 24 counts, < 1% of the frame; any
diff off that mask is a locality leak and aborts.

Registered reads (pre-reg 2026-08-14): ABORT unless in-run v3_photo
AUROC in 0.698 +/- 0.005 (the banked photometric-read anchor). PRIMARY:
paired dknn5 CI95 (10k resamples, rng 0) of v3_surf vs v3_photo
entirely below 0 (toward real). MECHANISM: same for only_links_surf vs
only_links_photo. Anchors: plate_only, banked v3 0.713, banked
only_links_photo 0.652, real_fg 0.328, and the REFUTED micro-texture
read (v3_tex +9.33e-07 [+8.27,+10.42]e-07, 0.698->0.751 — the delta
this escalation must NOT reproduce).

Usage:
  MUJOCO_GL=egl uv run python \
      fontaine/scripts/sim_arm_surface_texture_read.py \
      --out reports/analysis__sim_arm_surface_texture_read.json
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
from sim_arm_photometric_fit import render_masks
from sim_arm_photometric_read import hook
from sim_arm_split import arm_subclasses
from sim_fg_content_split import class_geoms
from sim_top_gap_decomposition import arm_read, dilate, knn5, paired_read

N_SEEDS = 20
N_DRAWS = 5
TOP_KEY = "observation.images.front"
V3_PHOTO_ABORT_BAND = (0.693, 0.703)  # registered: banked 0.698 +/- 0.005
ARM_HALO_DILATE = 16  # PSF + fractional remap-edge margin for the locality oracle
RIDER_MAX_DELTA = 24  # reflectance-scale bound, far below the on-arm signal
RIDER_MAX_FRACTION = 0.01  # of the frame, per slot


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
    reflective = np.array(
        sorted(
            g
            for g in range(scout.model.ngeom)
            if scout.model.geom_matid[g] >= 0
            and float(scout.model.mat_reflectance[scout.model.geom_matid[g]]) > 0
        ),
    )
    if reflective.size == 0:
        raise SystemExit("no reflective geoms — the rider region is undefined")
    del scout

    # Sequential passes — one live GL context at a time; pairing survives
    # on the bit-deterministic (seed, appearance) schedule, the baseline
    # pass banks qpos/frames/masks for the cross-instance oracles.
    print("baseline pass (grade only): 3 arms x 100 slots ...")
    base = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        arm_photometrics="v1",
    )
    base_arms, base_masks = hook(
        base,
        {
            "v3_photo": full,
            "plate_only": empty,
            "only_links_photo": links,
            "__armclass__": armclass,
        },
    )
    slots = []
    base_qpos: list[np.ndarray] = []
    strict_masks: list[np.ndarray] = []
    rider_masks: list[np.ndarray] = []
    for seed in range(N_SEEDS):
        for draw in range(N_DRAWS):
            appearance = 1000 * draw + seed
            obs_base = base.reset(seed, appearance_seed=appearance)
            if not np.array_equal(obs_base.top, base_arms["v3_photo"][-1]):
                raise SystemExit(
                    f"seed {seed} draw {draw}: baseline hook not bit-exact",
                )
            base_qpos.append(base.data.qpos.copy())
            halo = base_masks[-1].astype(np.float64)
            for _ in range(ARM_HALO_DILATE):
                halo = dilate(halo)
            outside = halo < 0.5
            refl = render_masks(base, {"reflective": reflective})["reflective"].astype(
                np.float64,
            )
            for _ in range(ARM_HALO_DILATE):
                refl = dilate(refl)
            allowed = refl >= 0.5
            strict_masks.append(outside & ~allowed)
            rider_masks.append(outside & allowed)
            slots.append({"seed": seed, "draw": draw})
    base.renderer.close()
    del base

    print("patched pass (grade + surface texture): 2 arms x 100 slots ...")
    patched = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        arm_photometrics="v1",
        arm_texture="v2",
    )
    patched_arms, _ = hook(
        patched,
        {"v3_surf": full, "only_links_surf": links, "__armclass__": armclass},
    )
    rider_stats = []
    for index, slot in enumerate(slots):
        appearance = 1000 * slot["draw"] + slot["seed"]
        obs_patch = patched.reset(slot["seed"], appearance_seed=appearance)
        if not np.array_equal(obs_patch.top, patched_arms["v3_surf"][-1]):
            raise SystemExit(f"slot {slot}: patched hook not bit-exact")
        if not np.array_equal(patched.data.qpos, base_qpos[index]):
            raise SystemExit(
                f"slot {slot}: instances diverge in qpos — the recompiled "
                "model must be physics-identical",
            )
        base_frame = base_arms["v3_photo"][index].astype(int)
        patch_frame = obs_patch.top.astype(int)
        strict = strict_masks[index]
        if (base_frame != patch_frame)[strict].any():
            raise SystemExit(
                f"slot {slot}: frames differ outside the dilated arm mask "
                "off the reflective rider region — locality leak",
            )
        rider = rider_masks[index]
        delta = np.abs(base_frame - patch_frame).max(axis=-1)
        changed = (delta > 0) & rider
        fraction = float(changed.mean())
        peak = int(delta[rider].max()) if rider.any() else 0
        if peak > RIDER_MAX_DELTA or fraction > RIDER_MAX_FRACTION:
            raise SystemExit(
                f"slot {slot}: reflection rider out of bounds — "
                f"|d| {peak} (<= {RIDER_MAX_DELTA}), "
                f"fraction {fraction:.4f} (<= {RIDER_MAX_FRACTION})",
            )
        rider_stats.append({"changed_fraction": fraction, "max_delta": peak})
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

    v3_photo_auroc = arms_read["v3_photo"]["auroc_vs_real"]
    aborted = not V3_PHOTO_ABORT_BAND[0] <= v3_photo_auroc <= V3_PHOTO_ABORT_BAND[1]

    primary = paired_read(scores["v3_surf"], scores["v3_photo"])
    mechanism = paired_read(scores["only_links_surf"], scores["only_links_photo"])
    results = {
        "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
        "clean_anchor": clean_read,
        "arms": arms_read,
        "primary_v3_surf_vs_v3_photo": primary,
        "mechanism_only_links_surf_vs_only_links_photo": mechanism,
        "primary_pass": bool(primary["ci95"][1] < 0),
        "mechanism_pass": bool(mechanism["ci95"][1] < 0),
        "reflection_rider": {
            "max_delta": max(s["max_delta"] for s in rider_stats),
            "max_changed_fraction": max(s["changed_fraction"] for s in rider_stats),
            "bounds": {
                "max_delta": RIDER_MAX_DELTA,
                "max_fraction": RIDER_MAX_FRACTION,
            },
        },
        "context": {
            "only_links_surf_vs_plate_only": paired_read(
                scores["only_links_surf"],
                scores["plate_only"],
            ),
            "photometric_read_v3": 0.713,
            "photometric_read_v3_photo": 0.698,
            "photometric_read_only_links_photo": 0.652,
            "decomposition_real_fg": 0.328,
            "refuted_micro_texture_primary": {
                "delta_knn5": 9.33e-07,
                "ci95": [8.27e-07, 1.04e-06],
                "auroc": 0.751,
            },
        },
        "v3_photo_abort_gate": {
            "band": list(V3_PHOTO_ABORT_BAND),
            "v3_photo_auroc": v3_photo_auroc,
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
            "instances (numpy post backend, arm_photometrics='v1' vs "
            "arm_photometrics='v1' + arm_texture='v2' mjSpec recompile) "
            "hooked at _composite; per-slot qpos bit-equality, strict "
            "outside-arm bit-equality off the reflective rider region, "
            "bounded tabletop-reflection rider; noise-RNG state restored "
            "per arm",
            "texture": SO101Sim.ARM_SURFACE_TEXTURE_V2,
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
    print(f"PRIMARY v3_surf vs v3_photo: {primary}")
    print(f"MECHANISM only_links_surf vs only_links_photo: {mechanism}")
    print(f"reflection rider: {results['reflection_rider']}")
    print(f"wrote {args.out}")
    if aborted:
        print(
            f"ABORT: in-run v3_photo AUROC {v3_photo_auroc:.3f} outside the "
            f"registered band {V3_PHOTO_ABORT_BAND} — no claims from this run",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
