"""Arm sub-part split: WHICH arm sub-part carries the rendered-arm
ceiling (queue `sim-arm-appearance-leg`; pre-reg in-channel 06:18Z
08-13)? Leg (a) read only_arm 0.654 vs plate_only 0.866 with the arm
class at ~7.1% of pixels — the biggest rendered class left after the
clutter patches (patched 0.556 >> real-fg 0.328).

Same hooked harness as sim_fg_content_split (leg (a)): ONE production
v3 instance (numpy post backend), ``_composite`` hook, segmentation-
restricted masks, 20 seeds x 5 appearance draws, noise-RNG state
restored per arm — frames pair 1:1 and the paired delta IS the
sub-part's visible-pixel effect.

Two exact partitions of the 96 arm-class geoms (in-run tiling oracle):

  part axis (pooled over both arm instances)
  - gripper   gripper + moving_jaw bodies        (23 x 2 = 46 geoms)
  - links     base/shoulder/upper/lower/wrist    (22 x 2 = 44 geoms)
  - mount     camera_mount                       ( 3 x 2 =  6 geoms)

  instance axis
  - follower  the working arm subtree            (48 geoms)
  - leader    the leader-arm subtree             (48 geoms)

Arms (14): v3, plate_only, leg-(a) bridges no_arm/only_arm, and
no_/only_ per sub-class. no_<X> = full dynamic mask minus X (all other
content incl. benchy/clutter/disk stays); only_<X> = X alone on the
bare plate.

Registered anchors/aborts (pre-reg 06:18Z): in-run v3 must read
0.713 +/- 0.005 else ABORT (no claims). Bridge bands (fresh noise
realization; non-abort, flagged): plate_only 0.865 +/- 0.02, only_arm
0.654 +/- 0.02, no_arm 0.825 +/- 0.02.

Decision rule (registered): rank parts by paired dknn5 CI95 (10k
resamples, rng 0) of only_<part> vs plate_only; a part is NAMED the
photometric-fix target iff its CI excludes 0 AND it carries >= 60% of
the only_arm - plate_only paired delta; if two parts each carry
>= 35%, both are named. Instance axis is context (does a fix need to
treat both instances). Per-part pixel fractions recorded per slot.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_arm_split.py \
      --out reports/analysis__sim_arm_split.json
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
from sim_fg_content_split import class_geoms
from sim_top_gap_decomposition import arm_read, knn5, paired_read

N_SEEDS = 20
N_DRAWS = 5
N_SLOTS = N_SEEDS * N_DRAWS
TOP_KEY = "observation.images.front"
V3_ABORT_BAND = (0.708, 0.718)  # registered: banked anchor 0.713 +/- 0.005
BRIDGE_BANDS = {  # non-abort sanity, flagged if missed (pre-reg 06:18Z)
    "plate_only": (0.845, 0.885),
    "only_arm": (0.634, 0.674),
    "no_arm": (0.805, 0.845),
}
PART_BODIES = {
    "gripper": ("gripper", "moving_jaw_so101_v1"),
    "links": ("base", "shoulder", "upper_arm", "lower_arm", "wrist"),
    "mount": ("camera_mount",),
}
EXPECTED_SIZES = {"gripper": 46, "links": 44, "mount": 6, "follower": 48, "leader": 48}


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


def arm_subclasses(sim, armclass: np.ndarray) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Both partitions of the arm content class; raises unless each
    axis tiles the class exactly at the expected sizes (in-run oracle)."""
    model = sim.model
    arm_set = {int(g) for g in armclass}
    sub: dict[str, set[int]] = {
        name: set() for name in (*PART_BODIES, "follower", "leader")
    }
    for g in sorted(arm_set):
        body = model.geom_bodyid[g]
        name = model.body(body).name
        stem = name.removeprefix("leader-")
        part = next((p for p, bodies in PART_BODIES.items() if stem in bodies), None)
        if part is None:
            raise SystemExit(f"geom {g} (body {name}) matches no part class")
        sub[part].add(g)
        sub["leader" if name.startswith("leader-") else "follower"].add(g)
    for axis in (("gripper", "links", "mount"), ("follower", "leader")):
        union = set().union(*(sub[name] for name in axis))
        if union != arm_set or sum(len(sub[name]) for name in axis) != len(arm_set):
            raise SystemExit(f"axis {axis} does not tile the arm class")
    for name, ids in sub.items():
        if len(ids) != EXPECTED_SIZES[name]:
            raise SystemExit(
                f"subclass {name}: {len(ids)} geoms != expected {EXPECTED_SIZES[name]}",
            )
    return {name: np.array(sorted(ids)) for name, ids in sub.items()}


def build_subsets(
    full: np.ndarray,
    armclass: np.ndarray,
    sub: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    empty = np.array([], dtype=full.dtype)
    subsets = {
        "v3": full,
        "plate_only": empty,
        "no_arm": np.setdiff1d(full, armclass),
        "only_arm": armclass,
    }
    for name, ids in sub.items():
        subsets[f"no_{name}"] = np.setdiff1d(full, ids)
        subsets[f"only_{name}"] = ids
    return subsets


def render_arms(
    subsets: dict[str, np.ndarray],
    sub: dict[str, np.ndarray],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[float]], list[dict]]:
    """All arms from ONE production instance — the leg-(a) hook verbatim:
    top composite re-run per subset with the noise RNG state restored."""
    import mujoco

    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v3", post_backend="numpy")
    arms: dict[str, list[np.ndarray]] = {name: [] for name in subsets}
    fractions: dict[str, list[float]] = {name: [] for name in sub}
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
        for name, ids in sub.items():
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
            arm_fraction = fractions["follower"][-1] + fractions["leader"][-1]
            if not 0.005 < arm_fraction < 0.4:
                raise SystemExit(
                    f"seed {seed} draw {draw}: arm-class mask fraction "
                    f"{arm_fraction:.3f} outside sanity range",
                )
            slots.append({"seed": seed, "draw": draw})
    return arms, fractions, slots


def main() -> int:
    args = parse_args()

    from sim.so101_sim import SO101Sim

    scout = SO101Sim(render_style="v3", post_backend="numpy")
    classes = class_geoms(scout)
    sub = arm_subclasses(scout, classes["arm"])
    subsets = build_subsets(scout._dynamic_geoms, classes["arm"], sub)
    del scout
    print({name: len(ids) for name, ids in sub.items()})

    print("rendering 14 arms (one production instance, 20 seeds x 5 draws) ...")
    arms, fractions, slots = render_arms(subsets, sub)

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
    bridge_flags = {
        name: {
            "band": list(band),
            "auroc": arms_read[name]["auroc_vs_real"],
            "in_band": band[0] <= arms_read[name]["auroc_vs_real"] <= band[1],
        }
        for name, band in BRIDGE_BANDS.items()
    }

    # registered decision rule: only_<part> vs plate_only paired deltas,
    # shares of the only_arm - plate_only paired delta
    arm_delta = paired_read(scores["only_arm"], scores["plate_only"])
    decision = {}
    for part in PART_BODIES:
        read = paired_read(scores[f"only_{part}"], scores["plate_only"])
        share = read["mean_delta"] / arm_delta["mean_delta"]
        ci_excl = read["ci95"][0] > 0 or read["ci95"][1] < 0
        decision[part] = {
            "paired_vs_plate_only": read,
            "share_of_only_arm_delta": float(share),
            "ci_excludes_zero": ci_excl,
            "named_60pct": bool(ci_excl and share >= 0.60),
            "named_split_35pct": bool(ci_excl and share >= 0.35),
        }
    named = [p for p, d in decision.items() if d["named_60pct"]]
    if not named:
        split = [p for p, d in decision.items() if d["named_split_35pct"]]
        named = split if len(split) >= 2 else []

    results = {
        "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
        "clean_anchor": clean_read,
        "arms": arms_read,
        "paired_vs_v3": {
            name: paired_read(scores[name], scores["v3"])
            for name in scores
            if name.startswith("no_")
        },
        "paired_vs_plate_only": {
            name: paired_read(scores[name], scores["plate_only"])
            for name in scores
            if name.startswith("only_")
        },
        "only_arm_vs_plate_only": arm_delta,
        "decision": {"per_part": decision, "named_targets": named},
        "subclass_pixel_fraction_mean": {
            name: float(np.mean(values)) for name, values in fractions.items()
        },
        "v3_abort_gate": {
            "band": list(V3_ABORT_BAND),
            "v3_auroc": v3_auroc,
            "aborted": aborted,
        },
        "bridge_bands": bridge_flags,
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
            "arms share physics/plate/noise per slot (rng-state restore); "
            "arm-class partitions per pre-reg 06:18Z 08-13",
            "subclasses": {name: [int(g) for g in ids] for name, ids in sub.items()},
            "anchors": {
                "banked_v3_20x5": 0.713,
                "leg_a_plate_only": 0.866,
                "leg_a_only_arm": 0.654,
                "leg_a_no_arm": 0.825,
                "decomposition_real_fg": 0.328,
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
    for part, d in decision.items():
        print(
            f"decision {part}: share {d['share_of_only_arm_delta']:.2f} | "
            f"CI {d['paired_vs_plate_only']['ci95']} | "
            f"named60 {d['named_60pct']}",
        )
    print(f"named targets: {named or 'NONE (no part meets the registered rule)'}")
    print(f"wrote {args.out}")
    if aborted:
        print(
            f"ABORT: in-run v3 AUROC {v3_auroc:.3f} outside the registered "
            f"band {V3_ABORT_BAND} — no claims from this run",
        )
        return 1
    for name, flag in bridge_flags.items():
        if not flag["in_band"]:
            print(f"FLAG: bridge {name} {flag['auroc']:.3f} outside {flag['band']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
