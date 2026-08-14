"""Wrist content split — how much of the banked 0.877 manipulation-pose
wrist AUROC is the benchy, not the arm? (queue
`sim-manip-wrist-content-split`; pre-reg posted before this runs:
posts/2026-08-14-prereg-sim-wrist-content-split.md).

The rollout-pose read (banked 12:2xZ 08-14) left scene content
unmatched: the sim benchy sits at its seeded spawn while real
mid-grasp frames hold the boat elsewhere. This read prices the benchy
term: the banked harness verbatim (same slots, reference, calibration,
20x5 schedule), TWO default-material instances differing by ZERO
flags — PRESENT (the banked arm, in-run anchor) vs ABSENT (manip pass
only: benchy free joint relocated to (0, 0, -10) before mj_forward;
reset passes bit-identical by construction, so appearance/content/
noise RNG streams stay aligned and each paired slot differs only in
benchy presence). Feasibility (registered): benchy visible in 61/100
slots, relocation zeroes it in all 100.

Registered gates (frozen in the pre-reg): reset TOP AUROC in
0.708-0.718 AND reset WRIST AUROC in [0.49, 0.57]; calibration <= 0.65
directional (low-note < 0.35 caveats AUROC-vs-real readings only —
the PRIMARY paired delta is sim-sim vs a common reference and is not
biased by calibration direction); reset changed-px == 0 (zero-flag
instances — any divergence is RNG drift); arm-qpos bit-equality x100;
benchy px == 0 in every ABSENT wrist segmentation; PRESENT manip AUROC
in [0.86, 0.89] (banked 0.877, sim-side spread 0.874-0.877).

PRIMARY: paired dknn5 ABSENT - PRESENT, CI95 (10k, rng 0). CI < 0:
content term real, share = point-delta / banked pose-effect
(+8.71e-06), >= 50% means 0.877 materially overstates the arm term.
Straddle: benchy term NIL, renderer-class keeps its full wrist price.
CI > 0: deletion is anti-matching; the arm term stands.

Two stages so the ~0.02 GPU-h embed can wait out the owner's GPU
reserve while renders land:

  MUJOCO_GL=egl uv run python fontaine/scripts/sim_wrist_content_split_read.py \
      --stage render --cache /tmp/content_split_frames.npz
  uv run python fontaine/scripts/sim_wrist_content_split_read.py \
      --stage score --cache /tmp/content_split_frames.npz \
      --out reports/analysis__sim_wrist_content_split.json
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
import sim_rollout_pose_wrist_read as banked
from sim_top_gap_decomposition import arm_read, knn5, paired_read

BENCHY_REMOVED_POS = (0.0, 0.0, -10.0)  # registered: 0 benchy px x100 slots
PRESENT_MANIP_BAND = (0.86, 0.89)  # registered replication anchor (banked 0.877)
POSE_EFFECT_BANKED = 8.71e-06  # banked pose-effect point delta, the share unit
CONTENT_SHARE_BAR = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("render", "score"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
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
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def benchy_px(sim) -> int:  # noqa: ANN001
    """Benchy pixel count in the RAW wrist segmentation."""
    import mujoco

    renderer = sim.renderer
    renderer.enable_segmentation_rendering()
    renderer.update_scene(sim.data, camera="wrist_cam")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
    return int((is_geom & (seg[..., 0] == sim.model.geom("benchy_visual").id)).sum())


def run_instance(
    states_deg: np.ndarray,
    base: dict | None,
) -> dict:
    """Reset pass + manip pass on a default-material production v3
    instance. `base` is None for PRESENT; for ABSENT it carries the
    PRESENT logs and switches on the benchy relocation + the
    cross-instance oracles (reset frames bit-identical, arm qpos
    bit-equal, benchy px == 0)."""
    import mujoco

    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v3", post_backend="numpy", lens_model="fitted")
    adr = sim._benchy_qpos
    out = {
        "reset_top": [],
        "reset_wrist": [],
        "manip_wrist": [],
        "reset_qpos": [],
        "manip_arm_qpos": [],
        "benchy_px": [],
        "manip_changed_px": [],
    }
    for index in range(banked.N_SLOTS):
        seed, draw = index // banked.N_DRAWS, index % banked.N_DRAWS
        obs = sim.reset(seed, appearance_seed=1000 * draw + seed)
        out["reset_top"].append(obs.top)
        out["reset_wrist"].append(obs.wrist)
        out["reset_qpos"].append(sim.data.qpos.copy())
        if base is not None:
            if not np.array_equal(sim.data.qpos, base["reset_qpos"][index]):
                raise SystemExit(f"reset slot {index}: qpos diverges — RNG drift")
            for camera in ("top", "wrist"):
                if not np.array_equal(
                    obs.top if camera == "top" else obs.wrist,
                    base[f"reset_{camera}"][index],
                ):
                    raise SystemExit(
                        f"reset slot {index}: {camera} frame differs — the "
                        "instances share zero flag changes, this is RNG drift",
                    )
    for index in range(banked.N_SLOTS):
        seed, draw = index // banked.N_DRAWS, index % banked.N_DRAWS
        sim.reset(seed, appearance_seed=1000 * draw + seed)
        sim.data.qpos[sim._joint_qpos] = np.clip(
            np.deg2rad(states_deg[index]),
            sim._ctrl_low,
            sim._ctrl_high,
        )
        if base is not None:
            sim.data.qpos[adr : adr + 3] = BENCHY_REMOVED_POS
        mujoco.mj_forward(sim.model, sim.data)
        out["benchy_px"].append(benchy_px(sim))
        obs = sim.observe()
        out["manip_wrist"].append(obs.wrist)
        out["manip_arm_qpos"].append(sim.data.qpos[sim._joint_qpos].copy())
        if base is not None:
            if out["benchy_px"][index] != 0:
                raise SystemExit(
                    f"manip slot {index}: {out['benchy_px'][index]} benchy px "
                    "after relocation — removal oracle failed",
                )
            if not np.array_equal(
                out["manip_arm_qpos"][index],
                base["manip_arm_qpos"][index],
            ):
                raise SystemExit(f"manip slot {index}: arm qpos diverges")
            delta = np.abs(
                base["manip_wrist"][index].astype(np.int16)
                - obs.wrist.astype(np.int16),
            )
            out["manip_changed_px"].append(int((delta.max(axis=-1) > 0).sum()))
    sim.renderer.close()
    return out


def stage_render(args: argparse.Namespace) -> int:
    data, eps = banked.load_tables(args.v2_root)
    n_episodes = int(data.episode_index.max()) + 1
    ref_pool = banked.mid_band_pool(data, list(banked.REF_EPISODES))
    calibration_pool = banked.mid_band_pool(data, list(banked.CALIBRATION_EPISODES))
    held_pool = banked.mid_band_pool(
        data,
        list(range(banked.HELD_EPISODE_MIN, n_episodes)),
    )
    ref_rows = banked.pick_evenly(ref_pool, banked.N_MANIP_REF)
    ref_holdout_rows = banked.pick_evenly(
        calibration_pool,
        banked.N_MANIP_REF_HOLDOUT,
    )
    held_rows = banked.pick_evenly(held_pool, banked.N_SLOTS)
    states_deg = np.stack(held_rows["observation.state"].to_list()).astype(np.float64)

    print("decoding exact real frames (manip ref/holdout/held) ...")
    real = {
        "real_manip_ref": banked.decode_exact(args.v2_root, "wrist", eps, ref_rows),
        "real_manip_ref_holdout": banked.decode_exact(
            args.v2_root,
            "wrist",
            eps,
            ref_holdout_rows,
        ),
        "real_manip_held": banked.decode_exact(args.v2_root, "wrist", eps, held_rows),
    }
    print("decoding standard strided real_v2 sets (reset-anchor protocol) ...")
    for camera in ("top", "wrist"):
        files = sorted(
            (args.v2_root / "videos" / banked.REAL_KEYS[camera] / "chunk-000").glob(
                "*.mp4",
            ),
        )
        real[f"real_{camera}_strided"] = probe.decode_strided(
            files,
            probe.total_frames(files) // probe.N_REAL_V2,
            probe.N_REAL_V2,
        )

    print("PRESENT instance (banked default arm): reset + manip passes ...")
    present = run_instance(states_deg, None)
    print("ABSENT instance (benchy relocated at manip): both passes ...")
    absent = run_instance(states_deg, present)
    print(
        f"oracles green: reset frames bit-identical x{banked.N_SLOTS}, arm qpos "
        f"bit-equal x{banked.N_SLOTS}, benchy px 0 x{banked.N_SLOTS}; manip "
        f"changed-px mean {np.mean(absent['manip_changed_px']):.0f} "
        f"max {max(absent['manip_changed_px'])}",
    )

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        reset_top=np.stack(present["reset_top"]),
        reset_wrist=np.stack(present["reset_wrist"]),
        manip_wrist_present=np.stack(present["manip_wrist"]),
        manip_wrist_absent=np.stack(absent["manip_wrist"]),
        benchy_px_present=np.array(present["benchy_px"]),
        manip_changed_px=np.array(absent["manip_changed_px"]),
        **{name: np.stack(frames) for name, frames in real.items()},
    )
    print(f"cached frames -> {args.cache}")
    return 0


def stage_score(args: argparse.Namespace) -> int:
    if args.out is None:
        raise SystemExit("--out is required for --stage score")
    cache = np.load(args.cache)
    model, info = probe.from_checkpoint(args.checkpoint, device="cuda")
    vision = model.backbone.vision
    del model.decoder
    emb = {}
    for name in cache.files:
        if name.startswith(("benchy_px", "manip_changed_px")):
            continue
        emb[name] = probe.embed(vision, list(cache[name]))
        print(f"embedded {name}: {tuple(emb[name].shape)}")

    half = probe.N_REAL_V2 // 2
    results: dict[str, object] = {}

    # In-run reset anchors — one instance's frames (bit-identical across
    # instances by the render-stage oracle).
    reset = {}
    for camera in ("top", "wrist"):
        ref = emb[f"real_{camera}_strided"][:half]
        held = knn5(emb[f"real_{camera}_strided"][half:], ref)
        reset[camera] = arm_read(knn5(emb[f"reset_{camera}"], ref), held)
    results["reset_anchor"] = reset

    ref = emb["real_manip_ref"]
    held_scores = knn5(emb["real_manip_held"], ref)
    calibration_auroc = probe.auroc(
        knn5(emb["real_manip_ref_holdout"], ref),
        held_scores,
    )
    arms = {
        "present": knn5(emb["manip_wrist_present"], ref),
        "absent": knn5(emb["manip_wrist_absent"], ref),
    }
    paired = paired_read(arms["absent"], arms["present"])
    results["manip"] = {
        "calibration_auroc_refholdout_vs_held": calibration_auroc,
        "arms": {name: arm_read(s, held_scores) for name, s in arms.items()},
        "paired_absent_vs_present": paired,
    }

    top_auroc = reset["top"]["auroc_vs_real"]
    wrist_auroc = reset["wrist"]["auroc_vs_real"]
    present_auroc = results["manip"]["arms"]["present"]["auroc_vs_real"]
    aborted = not (
        banked.TOP_ABORT_BAND[0] <= top_auroc <= banked.TOP_ABORT_BAND[1]
        and banked.WRIST_ABORT_BAND[0] <= wrist_auroc <= banked.WRIST_ABORT_BAND[1]
        and calibration_auroc <= banked.CALIBRATION_ABORT_MAX
        and PRESENT_MANIP_BAND[0] <= present_auroc <= PRESENT_MANIP_BAND[1]
    )
    calibration_low = calibration_auroc < banked.CALIBRATION_LOW_NOTE
    results["abort_gates"] = {
        "top_band": list(banked.TOP_ABORT_BAND),
        "top_auroc": top_auroc,
        "wrist_band": list(banked.WRIST_ABORT_BAND),
        "wrist_auroc": wrist_auroc,
        "calibration_abort_max": banked.CALIBRATION_ABORT_MAX,
        "calibration_auroc": calibration_auroc,
        "calibration_low": calibration_low,
        "present_manip_band": list(PRESENT_MANIP_BAND),
        "present_manip_auroc": present_auroc,
        "aborted": aborted,
    }

    point, ci = paired["mean_delta"], paired["ci95"]
    share = point / POSE_EFFECT_BANKED
    if ci[1] < 0:
        verdict = (
            "content_large" if -share >= CONTENT_SHARE_BAR else "content_real_minor"
        )
    elif ci[0] > 0:
        verdict = "removal_anti_matching"
    else:
        verdict = "content_nil"
    results["primary"] = {
        "paired_delta_absent_minus_present": point,
        "ci95": ci,
        "content_share_of_banked_pose_effect": -share,
        "content_share_bar": CONTENT_SHARE_BAR,
        "verdict": verdict,
    }
    results["secondary"] = {
        "absent_manip_auroc": results["manip"]["arms"]["absent"]["auroc_vs_real"],
        "calibration_low_caveat_active": calibration_low,
    }

    visible = cache["benchy_px_present"] > 0
    scores_delta = arms["absent"] - arms["present"]
    results["riders"] = {
        "benchy_visible_slots": int(visible.sum()),
        "paired_delta_visible_slots": paired_read(
            arms["absent"][visible],
            arms["present"][visible],
        ),
        "paired_delta_blind_slots": paired_read(
            arms["absent"][~visible],
            arms["present"][~visible],
        ),
        "corr_benchy_px_vs_abs_delta": float(
            np.corrcoef(
                cache["benchy_px_present"][visible],
                np.abs(scores_delta[visible]),
            )[0, 1],
        ),
        "manip_changed_px": {
            "mean": float(cache["manip_changed_px"].mean()),
            "max": int(cache["manip_changed_px"].max()),
        },
    }
    results["context"] = {
        "banked_present_auroc": 0.877,
        "banked_pose_effect_delta": POSE_EFFECT_BANKED,
        "benchy_removed_pos": list(BENCHY_REMOVED_POS),
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
            "protocol": "wrist content split: banked rollout-pose harness run-3 "
            "verbatim (100 pose-matched slots, 150-frame manip reference "
            "episodes 0-19, episode-disjoint calibration 20-25), two zero-flag "
            "default instances, PRESENT (benchy at spawn) vs ABSENT (benchy "
            "relocated to (0,0,-10) at manip only), er_60k knn5",
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(
        f"\nreset anchors: top {top_auroc:.3f} (band {banked.TOP_ABORT_BAND}), "
        f"wrist {wrist_auroc:.3f} (band {banked.WRIST_ABORT_BAND})",
    )
    print(
        f"calibration: {calibration_auroc:.3f}; PRESENT manip AUROC "
        f"{present_auroc:.3f} (band {PRESENT_MANIP_BAND})",
    )
    for name in arms:
        read = results["manip"]["arms"][name]
        print(
            f"manip wrist {name}: knn5 {read['mean']:.3e} | "
            f"AUROC {read['auroc_vs_real']:.3f}",
        )
    print(f"PRIMARY: {json.dumps(results['primary'], indent=1)}")
    print(f"riders: {json.dumps(results['riders'], indent=1)}")
    print(f"wrote {args.out}")
    if aborted:
        print("ABORT: gates outside the registered bands — no claims")
        return 3
    return 0


def main() -> int:
    args = parse_args()
    if args.stage == "render":
        return stage_render(args)
    return stage_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
