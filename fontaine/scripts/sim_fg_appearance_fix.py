"""Foreground appearance fix gate: do real-crop clutter patches close
the clutter share of the top-cam gap? (queue
`sim-foreground-appearance-pass` legs (b)+(c); pre-reg in-channel
05:23Z 2026-08-13; leg (a) anchor: no_clutter 0.576 vs v3 0.713.)

The leg (a) harness verbatim (sim_fg_content_split): ONE production v3
instance (numpy post backend) hooked at ``_composite``, 20 seeds x 5
appearance draws, every arm produced by the exact production
arithmetic with the noise RNG state restored per arm. Arms:

  v3          full mask, drawn plate    — bit-exact production oracle
  no_clutter  mask minus clutter geoms  — leg (a) replication anchor
  patched     no_clutter mask, plate with the mined real crops pasted
              at the drawn poses (clutter_patch inverse fisheye warp,
              active episode grading, zero RNG draws)

Registered (pre-reg 05:23Z): abort if in-run v3 outside 0.708–0.718 or
no_clutter off the leg (a) 0.576 by > 0.01. PASS = patched vs v3
paired dAUROC <= -0.05 AND paired dknn5 CI95 excluding 0. Secondary
(record): gap-closed fraction (0.713 - patched)/(0.713 - 0.576);
patched <= 0.596 reads as the full removable share recovered. Partial
(-0.02..-0.05): record, queue crop iteration. No improvement:
appearance hypothesis falsified at this implementation — inspect the
dumped paste frames before claiming.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_fg_appearance_fix.py \
      --out reports/analysis__sim_fg_appearance_fix.json \
      --dump-frames reports/assets/fg_fix_frames
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

from sim.clutter_patch import ClutterCrops

N_SEEDS = 20
N_DRAWS = 5
TOP_KEY = "observation.images.front"
CLUTTER_NAMES = ("mouse", "mug", "laptop", "pcb")
V3_ABORT_BAND = (0.708, 0.718)  # registered: banked anchor 0.713 +/- 0.005
NO_CLUTTER_BAND = (0.566, 0.586)  # registered: leg (a) 0.576 +/- 0.01
LEG_A = {"v3": 0.713, "no_clutter": 0.576, "real_fg": 0.328}


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
    parser.add_argument(
        "--crops",
        type=Path,
        default=Path("assets/real_plates/bank/crops"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump-frames", type=Path, default=None)
    return parser.parse_args()


def render_arms(
    crops: ClutterCrops,
) -> tuple[dict[str, list[np.ndarray]], list[dict]]:
    """v3 / no_clutter / patched off ONE hooked production instance —
    same physics, plate, drawn clutter and sensor noise per slot."""
    import mujoco

    from sim.so101_sim import SO101Sim

    sim = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        clutter_appearance="standins",  # the gate's registered substrate
    )
    clutter_ids = np.array(
        sorted(sim.model.geom(name).id for name in CLUTTER_NAMES),
    )
    full = sim._dynamic_geoms
    no_clutter = np.setdiff1d(full, clutter_ids)
    arms: dict[str, list[np.ndarray]] = {
        name: [] for name in ("v3", "no_clutter", "patched")
    }
    slots: list[dict] = []
    stash: dict[str, np.ndarray] = {}

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
        sub = {
            "v3": (is_geom & np.isin(seg[..., 0], full)).astype(np.float64),
            "no_clutter": (is_geom & np.isin(seg[..., 0], no_clutter)).astype(
                np.float64,
            ),
        }
        patched_plate = crops.paste(
            sim._active_top_plate,
            sim._clutter_drawn,
            sim._clutter_base,
            sim._active_gain,
            sim._active_bias,
        )
        state = sim._noise_rng.bit_generator.state
        for name in ("v3", "no_clutter"):
            sim._noise_rng.bit_generator.state = state
            arms[name].append(orig_composite(frame, sub[name], camera, shadow=shadow))
        sim._noise_rng.bit_generator.state = state
        saved = sim._active_top_plate
        sim._active_top_plate = patched_plate
        arms["patched"].append(
            orig_composite(frame, sub["no_clutter"], camera, shadow=shadow),
        )
        sim._active_top_plate = saved
        sim._noise_rng.bit_generator.state = state
        return orig_composite(frame, sub["v3"], camera, shadow=shadow)

    sim._render_mask = mask_hook
    sim._composite = composite_hook

    for seed in range(N_SEEDS):
        for draw in range(N_DRAWS):
            appearance = 1000 * draw + seed
            obs = sim.reset(seed, appearance_seed=appearance)
            if not np.array_equal(obs.top, arms["v3"][-1]):
                raise SystemExit(
                    f"seed {seed} draw {draw}: hooked v3 arm is not "
                    "bit-exact with the production observation",
                )
            slots.append(
                {
                    "seed": seed,
                    "draw": draw,
                    "clutter_absent": sorted(
                        name
                        for name, (pos, _) in sim._clutter_drawn.items()
                        if tuple(pos) == sim.V3_ABSENT_POS
                    ),
                },
            )
    return arms, slots


def main() -> int:
    args = parse_args()
    crops = ClutterCrops(args.crops)

    print("rendering 3 arms (one production instance, 20 seeds x 5 draws) ...")
    arms, slots = render_arms(crops)

    real: dict[str, list[np.ndarray]] = {}
    for group, root, count in (
        ("real_v2", args.v2_root, probe.N_REAL_V2),
        ("real_clean", args.clean_root, probe.N_REAL_CLEAN),
    ):
        files = sorted((root / "videos" / TOP_KEY / "chunk-000").glob("*.mp4"))
        total = probe.total_frames(files)
        real[group] = probe.decode_strided(files, total // count, count)
        print(f"{group}: {count} frames (stride {total // count})")

    groups: dict[str, list[np.ndarray]] = {**arms, **real}
    if args.dump_frames is not None:
        from PIL import Image

        for name, frames in groups.items():
            out_dir = args.dump_frames / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in (0, 1, 2, 3):
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
    scores = {name: knn5(emb[name], ref) for name in arms}
    clean = knn5(emb["real_clean"], ref)
    arms_read = {name: arm_read(scores[name], held) for name in arms}
    clean_read = arm_read(clean, held)

    v3_auroc = arms_read["v3"]["auroc_vs_real"]
    nc_auroc = arms_read["no_clutter"]["auroc_vs_real"]
    patched_auroc = arms_read["patched"]["auroc_vs_real"]
    aborted_v3 = not V3_ABORT_BAND[0] <= v3_auroc <= V3_ABORT_BAND[1]
    aborted_nc = not NO_CLUTTER_BAND[0] <= nc_auroc <= NO_CLUTTER_BAND[1]

    paired_patched = paired_read(scores["patched"], scores["v3"])
    d_auroc = patched_auroc - v3_auroc
    ci = paired_patched["ci95"]
    ci_excludes_zero = ci[0] > 0 or ci[1] < 0
    passed = d_auroc <= -0.05 and ci_excludes_zero and not (aborted_v3 or aborted_nc)
    gap_closed = (v3_auroc - patched_auroc) / (v3_auroc - nc_auroc)

    results = {
        "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
        "clean_anchor": clean_read,
        "arms": arms_read,
        "paired_vs_v3": {
            name: paired_read(scores[name], scores["v3"])
            for name in ("no_clutter", "patched")
        },
        "paired_patched_vs_no_clutter": paired_read(
            scores["patched"],
            scores["no_clutter"],
        ),
        "registered_gate": {
            "v3_abort_band": list(V3_ABORT_BAND),
            "no_clutter_band": list(NO_CLUTTER_BAND),
            "aborted_v3": aborted_v3,
            "aborted_no_clutter": aborted_nc,
            "d_auroc_patched_vs_v3": d_auroc,
            "dknn5_ci95_excludes_zero": ci_excludes_zero,
            "pass": passed,
            "gap_closed_fraction": gap_closed,
            "full_recovery_read": patched_auroc <= 0.596,
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
            "protocol": "sim_encoder_ood_probe A/B split, top camera; "
            "20 seeds x 5 appearance draws, settled resets, ONE production "
            "v3 instance (numpy post backend) hooked at _composite; "
            "patched arm = no_clutter mask over the drawn plate with the "
            "mined real crops pasted at the drawn poses",
            "crops_manifest": crops.manifest,
            "leg_a_anchors": LEG_A,
            "slots": slots,
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    for name in arms:
        print(
            f"{name}: knn5 {scores[name].mean():.3e} | "
            f"AUROC {arms_read[name]['auroc_vs_real']:.3f}",
        )
    print(f"clean anchor AUROC {clean_read['auroc_vs_real']:.3f}")
    print(
        f"gate: dAUROC {d_auroc:+.3f} | CI-excl-0 {ci_excludes_zero} | "
        f"gap closed {gap_closed:.0%} | PASS {passed}",
    )
    print(f"wrote {args.out}")
    if aborted_v3 or aborted_nc:
        print(
            f"ABORT: v3 {v3_auroc:.3f} (band {V3_ABORT_BAND}) "
            f"no_clutter {nc_auroc:.3f} (band {NO_CLUTTER_BAND}) — "
            "no claims from this run",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
