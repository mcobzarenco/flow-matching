"""Full opt-in stack read — prices the COMBINED promotion (queue
`sim-full-optin-stack-read`; pre-reg posted before this runs).

Three appearance promotions are banked SEPARATELY against the same
20x5 protocol: clutter real-crop patches (0.713 -> 0.556, the big
one), arm_photometrics='v1' (0.713 -> 0.698), mount_material='v1'
(alone n.s.; rides the material stack to 0.702). If the owner flips
them together the interactions are unmeasured — this read measures
the full stack in one paired harness.

Composition of the two landed mechanisms:

  materials  MODEL-level grades at init (arm_photometrics +
             mount_material) — need a SECOND instance
  clutter    composite-level: no_clutter mask over the drawn plate
             with the mined real crops pasted at the drawn poses
             (clutter_patch inverse fisheye warp, episode grading)

Two production v3 instances (numpy post backend) over the SAME
(seed, appearance_seed) schedule, both hooked at ``_composite`` with
the noise RNG state restored per arm (the fg-fix harness):

  baseline instance: v3       production output (bit-exact oracle)
                     patched  no_clutter mask + pasted plate — the
                              in-run best-single replication anchor
  stack instance:    (production output = materials only, full mask —
                     bit-exact oracle, not embedded)
                     stack_full  no_clutter mask + pasted plate ON the
                                 graded-materials render — THE ARM

In-run oracles: per-slot qpos bit-equality across instances (the
grades consume no RNG); clutter draws + episode affine bit-equal
across instances; top changed-px fraction between the two production
frames <= 30% (an RNG-stream divergence flips ~100% through the
sensor noise; the material grades touch only arm/servo/mount pixels).

Registered (pre-reg 2026-08-14, posted before run): ABORT unless
in-run v3 AUROC in 0.713 +/- 0.005 AND in-run patched AUROC in 0.556
+/- 0.010 (the banked best-single replication). PRIMARY PASS = paired
dknn5 CI95 (10k resamples, rng 0) of stack_full vs v3 entirely below
0 AND stack_full AUROC <= 0.5511 (banked best single 0.5561 - eps,
eps = 0.005 registered). Record-only: additivity — measured stack
AUROC vs the additive prediction v3_inrun - 0.1566 - 0.0103 (banked
clutter delta + banked material-stack delta), the deviation is the
interaction term; paired stack_full vs patched (the materials'
marginal contribution on top of clutter, banked material-stack CI
[-2.45e-07, -5.70e-08] as the no-interaction reference).

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_full_optin_stack_read.py \
      --out reports/analysis__sim_full_optin_stack_read.json \
      --dump-frames reports/assets/full_optin_stack_frames
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
N_SLOTS = N_SEEDS * N_DRAWS
TOP_KEY = "observation.images.front"
CLUTTER_NAMES = ("mouse", "mug", "laptop", "pcb")
V3_ABORT_BAND = (0.708, 0.718)  # registered: banked 20x5 anchor 0.713 +/- 0.005
PATCHED_ABORT_BAND = (0.546, 0.566)  # registered: banked fg-fix 0.5561 +/- 0.010
EPSILON = 0.005  # registered: stack must beat the banked best single by this
BEST_SINGLE = 0.5561  # banked fg-fix patched arm
BANKED_DELTAS = {"clutter_patched": -0.1566, "material_stack": -0.0103}
CHANGED_PX_MAX = 92_160  # 30% of 307,200 — RNG-divergence tripwire
MATERIAL_STACK_REF_CI = (-2.45e-07, -5.70e-08)  # banked v3_full_fix vs v3


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


def run_instance(
    crops: ClutterCrops,
    *,
    material_stack: bool,
    base_state: list[dict] | None,
) -> tuple[dict[str, list[np.ndarray]], list[dict]]:
    """One hooked production v3 instance over the 20x5 schedule.

    Returns (arms, slot_state). ``arms`` holds the production frame
    per slot plus one side-arm: ``patched`` (baseline) or
    ``stack_full`` (material_stack=True) — both are no_clutter mask
    over the pasted plate, differing only in the model materials.
    ``slot_state`` carries qpos / clutter draws / affine for the
    cross-instance pairing oracles; pass the baseline's as
    ``base_state`` to check the stack instance against it.
    """
    import mujoco

    from sim.so101_sim import SO101Sim

    kwargs = (
        {"arm_photometrics": "v1", "mount_material": "v1"} if material_stack else {}
    )
    sim = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        clutter_appearance="standins",  # the read's registered substrate
        **kwargs,
    )
    clutter_ids = np.array(
        sorted(sim.model.geom(name).id for name in CLUTTER_NAMES),
    )
    no_clutter = np.setdiff1d(sim._dynamic_geoms, clutter_ids)
    side_name = "stack_full" if material_stack else "patched"
    arms: dict[str, list[np.ndarray]] = {"production": [], side_name: []}
    slot_state: list[dict] = []

    orig_composite = sim._composite

    def composite_hook(frame, mask, camera, shadow=None) -> np.ndarray:  # noqa: ANN001
        if camera != "top":
            return orig_composite(frame, mask, camera, shadow=shadow)
        patched_plate = crops.paste(
            sim._active_top_plate,
            sim._clutter_drawn,
            sim._clutter_base,
            sim._active_gain,
            sim._active_bias,
        )
        renderer = sim.renderer
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="top_cam")
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        nc_mask = (is_geom & np.isin(seg[..., 0], no_clutter)).astype(np.float64)
        state = sim._noise_rng.bit_generator.state
        saved = sim._active_top_plate
        sim._active_top_plate = patched_plate
        arms[side_name].append(
            orig_composite(frame, nc_mask, camera, shadow=shadow),
        )
        sim._active_top_plate = saved
        sim._noise_rng.bit_generator.state = state
        return orig_composite(frame, mask, camera, shadow=shadow)

    sim._composite = composite_hook

    for index in range(N_SLOTS):
        seed, draw = index // N_DRAWS, index % N_DRAWS
        obs = sim.reset(seed, appearance_seed=1000 * draw + seed)
        arms["production"].append(obs.top)
        slot_state.append(
            {
                "qpos": sim.data.qpos.copy(),
                "drawn": {
                    name: (pos.copy(), float(yaw))
                    for name, (pos, yaw) in sim._clutter_drawn.items()
                },
                "gain": np.array(sim._active_gain, dtype=np.float64).copy(),
                "bias": np.array(sim._active_bias, dtype=np.float64).copy(),
            },
        )
        if base_state is not None:
            ref = base_state[index]
            if not np.array_equal(sim.data.qpos, ref["qpos"]):
                raise SystemExit(
                    f"seed {seed} draw {draw}: instances diverge in qpos — "
                    "the grades must consume no RNG draws",
                )
            drawn = slot_state[-1]["drawn"]
            same_draws = set(drawn) == set(ref["drawn"]) and all(
                np.array_equal(drawn[k][0], ref["drawn"][k][0])
                and drawn[k][1] == ref["drawn"][k][1]
                for k in drawn
            )
            if not (
                same_draws
                and np.array_equal(slot_state[-1]["gain"], ref["gain"])
                and np.array_equal(slot_state[-1]["bias"], ref["bias"])
            ):
                raise SystemExit(
                    f"seed {seed} draw {draw}: clutter draws / episode "
                    "affine diverge across instances",
                )
    sim.renderer.close()
    return arms, slot_state


def main() -> int:
    args = parse_args()
    crops = ClutterCrops(args.crops)

    print("baseline pass (v3 default, patched side-arm): 100 slots ...")
    base_arms, base_state = run_instance(crops, material_stack=False, base_state=None)
    print("stack pass (arm_photometrics + mount_material, stack_full): 100 slots ...")
    stack_arms, _ = run_instance(crops, material_stack=True, base_state=base_state)

    changed_px = []
    for a, b in zip(base_arms["production"], stack_arms["production"], strict=True):
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        changed_px.append(int((delta.max(axis=-1) > 0).sum()))
    if max(changed_px) > CHANGED_PX_MAX:
        raise SystemExit(
            f"{max(changed_px)} top px differ between production frames "
            "(> 30% of frame) — RNG-divergence tripwire",
        )
    print(
        f"top changed-px between production frames: max {max(changed_px)} "
        f"({max(changed_px) / 307_200:.1%}), mean {np.mean(changed_px):.0f}",
    )

    real: dict[str, list[np.ndarray]] = {}
    for group, root, count in (
        ("real_v2", args.v2_root, probe.N_REAL_V2),
        ("real_clean", args.clean_root, probe.N_REAL_CLEAN),
    ):
        files = sorted((root / "videos" / TOP_KEY / "chunk-000").glob("*.mp4"))
        total = probe.total_frames(files)
        real[group] = probe.decode_strided(files, total // count, count)
        print(f"{group}: {count} frames (stride {total // count})")

    groups: dict[str, list[np.ndarray]] = {
        "v3": base_arms["production"],
        "patched": base_arms["patched"],
        "stack_full": stack_arms["stack_full"],
        **real,
    }
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
    scores = {name: knn5(emb[name], ref) for name in ("v3", "patched", "stack_full")}
    clean = knn5(emb["real_clean"], ref)
    arms_read = {name: arm_read(s, held) for name, s in scores.items()}
    clean_read = arm_read(clean, held)

    v3_auroc = arms_read["v3"]["auroc_vs_real"]
    patched_auroc = arms_read["patched"]["auroc_vs_real"]
    stack_auroc = arms_read["stack_full"]["auroc_vs_real"]
    aborted_v3 = not V3_ABORT_BAND[0] <= v3_auroc <= V3_ABORT_BAND[1]
    aborted_patched = not (
        PATCHED_ABORT_BAND[0] <= patched_auroc <= PATCHED_ABORT_BAND[1]
    )
    aborted = aborted_v3 or aborted_patched

    primary = paired_read(scores["stack_full"], scores["v3"])
    ci_below_zero = primary["ci95"][1] < 0
    beats_best_single = stack_auroc <= BEST_SINGLE - EPSILON
    passed = ci_below_zero and beats_best_single and not aborted

    additive_prediction = (
        v3_auroc + BANKED_DELTAS["clutter_patched"] + BANKED_DELTAS["material_stack"]
    )
    results = {
        "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
        "clean_anchor": clean_read,
        "arms": arms_read,
        "paired_stack_vs_v3": primary,
        "paired_patched_vs_v3": paired_read(scores["patched"], scores["v3"]),
        "paired_stack_vs_patched": paired_read(
            scores["stack_full"],
            scores["patched"],
        ),
        "registered_gate": {
            "v3_abort_band": list(V3_ABORT_BAND),
            "patched_abort_band": list(PATCHED_ABORT_BAND),
            "aborted_v3": aborted_v3,
            "aborted_patched": aborted_patched,
            "primary_ci95_below_zero": ci_below_zero,
            "stack_auroc": stack_auroc,
            "best_single_bar": BEST_SINGLE - EPSILON,
            "beats_best_single": beats_best_single,
            "pass": passed,
        },
        "additivity": {
            "banked_deltas": BANKED_DELTAS,
            "additive_prediction": additive_prediction,
            "measured": stack_auroc,
            "interaction_auroc": stack_auroc - additive_prediction,
            "material_marginal_ref_ci95_knn5": list(MATERIAL_STACK_REF_CI),
        },
        "top_changed_px_between_instances": {
            "max": max(changed_px),
            "mean": float(np.mean(changed_px)),
            "frame_px": 307_200,
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
            "20 seeds x 5 appearance draws, settled resets, TWO paired "
            "production v3 instances (numpy post backend: default vs "
            "arm_photometrics='v1' + mount_material='v1'), both hooked at "
            "_composite; patched/stack_full = no_clutter mask over the "
            "drawn plate with the mined real crops pasted at the drawn "
            "poses; per-slot qpos + clutter-draw + affine bit-equality "
            "across instances, changed-px tripwire",
            "crops_manifest": crops.manifest,
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    for name in ("v3", "patched", "stack_full"):
        print(
            f"{name}: knn5 {scores[name].mean():.3e} | "
            f"AUROC {arms_read[name]['auroc_vs_real']:.3f}",
        )
    print(f"clean anchor AUROC {clean_read['auroc_vs_real']:.3f}")
    print(f"paired stack vs v3: {primary}")
    print(
        f"gate: CI-below-0 {ci_below_zero} | stack {stack_auroc:.4f} vs bar "
        f"{BEST_SINGLE - EPSILON:.4f} -> beats-best-single {beats_best_single} "
        f"| PASS {passed}",
    )
    print(
        f"additivity: predicted {additive_prediction:.4f} | measured "
        f"{stack_auroc:.4f} | interaction {stack_auroc - additive_prediction:+.4f}",
    )
    print(f"wrote {args.out}")
    if aborted:
        print(
            f"ABORT: in-run anchors outside the registered bands — v3 "
            f"{v3_auroc:.3f} vs {V3_ABORT_BAND}, patched {patched_auroc:.3f} "
            f"vs {PATCHED_ABORT_BAND} — no claims from this run",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
