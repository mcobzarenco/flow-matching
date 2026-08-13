"""Foreground content split: WHICH rendered class carries the top-cam
knn5 AUROC 0.713 (queue `sim-foreground-appearance-pass` leg (a);
anchor: top-gap decomposition 08-13 — real-fg 0.328 vs v3 0.713, the
whole residual gap is the rendered foreground pixels)?

Ablation arms on the pinned encoder probe (20 seeds x 5 appearance
draws, settled resets, er_60k trunk, top camera only). ONE production
sim instance (v3, numpy post backend), hooked at ``_composite``: per
slot every arm is produced by the exact production arithmetic with a
segmentation-restricted dynamic mask — same physics, same drawn plate,
same sensor noise (the noise RNG state is restored before each arm's
composite), so frames pair 1:1 across arms and the paired delta IS the
class's visible-pixel effect.

  content classes (partition of the 442 dynamic geoms)
  - arm       both arms + camera mounts (96 geoms)
  - benchy    the boat (341 geoms)
  - clutter   the contype-0 stand-ins: mouse, mug, laptop, pcb
  - disk      split out of "clutter": always rendered at canonical,
              named visual suspect (untextured white cylinder)

  arms (masks; the rendered frame and everything else is shared)
  - v3          full dynamic mask — must be bit-exact == the
                production observation (in-run oracle)
  - plate_only  empty mask: out = plate + noise via production code
  - no_<C>      class C's visible pixels -> plate, rest rendered
  - only_<C>    class C rendered on the bare plate

Reads per arm: knn5 AUROC vs the held-out real B half (same A/B
protocol as sim_encoder_ood_probe) + the clean-repo anchor; paired
per-frame dknn5 bootstrap CI95 (10k resamples, rng 0): no_<C> and
plate_only vs v3, only_<C> vs plate_only. Per-class visible-pixel
fraction and per-slot clutter presence are recorded (context for
cross-class magnitude comparisons; record-only).

Registered anchors/aborts (pre-reg in-channel 04:41Z 08-13): in-run v3
must read 0.713 +/- 0.005 else abort; plate_only expected 0.865 +/-
0.02 (fresh noise realization vs the decomposition's own draws).

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_fg_content_split.py \
      --out reports/analysis__sim_fg_content_split.json
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
TOP_KEY = "observation.images.front"
CLUTTER_NAMES = ("mouse", "mug", "laptop", "pcb")
V3_ABORT_BAND = (0.708, 0.718)  # registered: banked anchor 0.713 +/- 0.005


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
    parser.add_argument(
        "--dump-frames",
        type=Path,
        default=None,
        help="write sample frames per arm (report/chart fuel)",
    )
    return parser.parse_args()


def class_geoms(sim) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Partition of the dynamic geom-id set into the four content
    classes; raises if the partition is not exact (in-run oracle)."""
    model = sim.model
    benchy = model.body("benchy").id
    benchy_bodies = set()
    for body in range(model.nbody):
        chain = body
        while chain != 0:
            if chain == benchy:
                benchy_bodies.add(body)
                break
            chain = model.body_parentid[chain]
    dynamic = {int(g) for g in sim._dynamic_geoms}
    classes = {
        "benchy": {
            g for g in range(model.ngeom) if model.geom_bodyid[g] in benchy_bodies
        },
        "clutter": {model.geom(name).id for name in CLUTTER_NAMES},
        "disk": {model.geom("disk").id},
    }
    classes["arm"] = dynamic - classes["benchy"] - classes["clutter"] - classes["disk"]
    union = set().union(*classes.values())
    if union != dynamic or sum(len(v) for v in classes.values()) != len(dynamic):
        raise SystemExit("class partition does not tile the dynamic geom set")
    return {name: np.array(sorted(ids)) for name, ids in classes.items()}


def arm_subsets(
    classes: dict[str, np.ndarray],
    full: np.ndarray,
) -> dict[str, np.ndarray]:
    empty = np.array([], dtype=full.dtype)
    subsets = {"v3": full, "plate_only": empty}
    for name, ids in classes.items():
        subsets[f"no_{name}"] = np.setdiff1d(full, ids)
        subsets[f"only_{name}"] = ids
    return subsets


def render_arms(
    subsets: dict[str, np.ndarray],
    classes: dict[str, np.ndarray],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[float]], list[dict]]:
    """All arms from ONE production instance: hook ``_composite`` so the
    top composite runs once per subset with the noise RNG state restored
    — bit-exact production arithmetic per arm, perfect slot pairing."""
    import mujoco

    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v3", post_backend="numpy")
    arms: dict[str, list[np.ndarray]] = {name: [] for name in subsets}
    fractions: dict[str, list[float]] = {name: [] for name in classes}
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
        state = sim._noise_rng.bit_generator.state
        for name, subset in subsets.items():
            sim._noise_rng.bit_generator.state = state
            sub_mask = (is_geom & np.isin(seg[..., 0], subset)).astype(np.float64)
            arms[name].append(orig_composite(frame, sub_mask, camera, shadow=shadow))
        for name, ids in classes.items():
            fractions[name].append(
                float((is_geom & np.isin(seg[..., 0], ids)).mean()),
            )
        sim._noise_rng.bit_generator.state = state
        return orig_composite(frame, mask, camera, shadow=shadow)

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
            full_fraction = fractions["arm"][-1] + fractions["benchy"][-1]
            full_fraction += fractions["clutter"][-1] + fractions["disk"][-1]
            if not 0.005 < full_fraction < 0.6:
                raise SystemExit(
                    f"seed {seed} draw {draw}: dynamic mask fraction "
                    f"{full_fraction:.3f} outside sanity range",
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
    return arms, fractions, slots


def main() -> int:
    args = parse_args()

    from sim.so101_sim import SO101Sim

    scout = SO101Sim(render_style="v3", post_backend="numpy")
    classes = class_geoms(scout)
    subsets = arm_subsets(classes, scout._dynamic_geoms)
    del scout
    print({name: len(ids) for name, ids in classes.items()})

    print("rendering 10 arms (one production instance, 20 seeds x 5 draws) ...")
    arms, fractions, slots = render_arms(subsets, classes)

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
    scores = {name: knn5(emb[name], ref) for name in arms}
    clean = knn5(emb["real_clean"], ref)
    arms_read = {name: arm_read(scores[name], held) for name in arms}
    clean_read = arm_read(clean, held)

    v3_auroc = arms_read["v3"]["auroc_vs_real"]
    aborted = not V3_ABORT_BAND[0] <= v3_auroc <= V3_ABORT_BAND[1]

    results = {
        "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
        "clean_anchor": clean_read,
        "arms": arms_read,
        "paired_vs_v3": {
            name: paired_read(scores[name], scores["v3"])
            for name in scores
            if name.startswith("no_") or name == "plate_only"
        },
        "paired_vs_plate_only": {
            name: paired_read(scores[name], scores["plate_only"])
            for name in scores
            if name.startswith("only_")
        },
        "class_pixel_fraction_mean": {
            name: float(np.mean(values)) for name, values in fractions.items()
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
            "protocol": "sim_encoder_ood_probe A/B split, top camera; "
            "20 seeds x 5 appearance draws, settled resets, ONE production "
            "v3 instance (numpy post backend) hooked at _composite — all "
            "arms share physics/plate/noise per slot (rng-state restore)",
            "classes": {name: [int(g) for g in ids] for name, ids in classes.items()},
            "anchors": {
                "banked_v3_20x5": 0.713,
                "decomposition_plate_only": 0.865,
                "decomposition_real_fg": 0.328,
                "decomposition_clean_anchor": 0.283,
            },
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
