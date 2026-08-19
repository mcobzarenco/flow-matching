"""Clutter-patch promotion re-gate: does the PRODUCTION patched path
reproduce the gate arm's read? (queue `clutter-patch-promotion-regate`;
pre-reg posts/2026-08-18-prereg-clutter-patch-promotion-regate.md; the
re-gate itself was registered with the 08-13 in-channel gate pre-reg —
"re-gate on the pinned 20x5 probe before any behavioral eval moves".)

The gate harness's slot grid verbatim (20 seeds x 5 appearance draws,
settled resets, numpy post backend, top camera) but PRODUCTION arms,
no hooks:

  patched   SO101Sim(render_style='v3') — the promotion default
  standins  same with clutter_appearance='standins' — in-run anchor

Registered bands (frozen in the pre-reg): standins AUROC in
0.708-0.718 (the gate's v3 abort band; outside -> abort, no claims);
PASS = patched within +/-0.010 of the gate's 0.556; +0.010..+0.030 ->
record + inspect frames before any behavioral move; worse -> the
promotion is suspect, revert the default pending diagnosis.

The model loads on CPU and only the vision trunk moves to CUDA — the
re-gate runs beside a live training run and must stay inside the free
VRAM margin.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_clutter_promotion_regate.py \
      --out reports/analysis__clutter_patch_promotion_regate.json \
      --dump-frames reports/assets/clutter_regate_frames
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
TOP_KEY = "observation.images.front"
STANDINS_BAND = (0.708, 0.718)  # the gate's registered v3 abort band
GATE_PATCHED = 0.556  # the gate arm's banked read
PASS_TOL = 0.010
RECORD_TOL = 0.030


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            # the VLA-format conversion of er_60k step_060000 (the raw
            # train dir is legacy-format; same weights)
            "~/checkpoints/converted/er_60k_step_060000_vla_v2",
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


def render_arms() -> dict[str, list[np.ndarray]]:
    """The two production arms over the pinned slot grid — separate
    instances, same seeds: physics, plate draw, drawn clutter and the
    noise stream are bit-identical across modes (oracle-pinned)."""
    from sim.so101_sim import SO101Sim

    sims = {
        "patched": SO101Sim(render_style="v3", post_backend="numpy"),
        "standins": SO101Sim(
            render_style="v3",
            post_backend="numpy",
            clutter_appearance="standins",
        ),
    }
    arms: dict[str, list[np.ndarray]] = {name: [] for name in sims}
    for seed in range(N_SEEDS):
        for draw in range(N_DRAWS):
            appearance = 1000 * draw + seed
            for name, sim in sims.items():
                obs = sim.reset(seed, appearance_seed=appearance)
                arms[name].append(obs.top)
    return arms


def main() -> int:
    args = parse_args()

    print("rendering 2 production arms (20 seeds x 5 draws each) ...")
    arms = render_arms()

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

    # CPU load, vision-only to CUDA: the full-model GPU mount would
    # ride too close to the live run's free-VRAM margin.
    import torch

    from bijou.loading import load_vla

    model = load_vla(args.checkpoint, device="cpu", dtype=torch.bfloat16)
    vision = model.backbone.vision.cuda()
    emb = {}
    for name, frames in groups.items():
        emb[name] = probe.embed(vision, frames)
        print(f"embedded {name}: {tuple(emb[name].shape)}")

    half = probe.N_REAL_V2 // 2
    ref = emb["real_v2"][:half]
    held = knn5(emb["real_v2"][half:], ref)
    scores = {name: knn5(emb[name], ref) for name in arms}
    clean_read = arm_read(knn5(emb["real_clean"], ref), held)
    arms_read = {name: arm_read(scores[name], held) for name in arms}

    standins_auroc = arms_read["standins"]["auroc_vs_real"]
    patched_auroc = arms_read["patched"]["auroc_vs_real"]
    aborted = not STANDINS_BAND[0] <= standins_auroc <= STANDINS_BAND[1]
    deviation = patched_auroc - GATE_PATCHED
    passed = abs(deviation) <= PASS_TOL and not aborted
    record_band = PASS_TOL < deviation <= RECORD_TOL

    results = {
        "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
        "clean_anchor": clean_read,
        "arms": arms_read,
        "paired_patched_vs_standins": paired_read(
            scores["patched"],
            scores["standins"],
        ),
        "registered_gate": {
            "standins_band": list(STANDINS_BAND),
            "aborted_standins": aborted,
            "gate_patched_anchor": GATE_PATCHED,
            "deviation_from_gate": deviation,
            "pass": passed,
            "record_band": record_band,
            "revert_indicated": deviation > RECORD_TOL and not aborted,
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
            "protocol": "sim_encoder_ood_probe A/B split, top camera; "
            "20 seeds x 5 appearance draws, settled resets, TWO production "
            "instances (numpy post backend), no hooks — patched = the "
            "promotion default, standins = the pre-promotion substrate",
            "prereg": "posts/2026-08-18-prereg-clutter-patch-promotion-regate.md",
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
        f"re-gate: patched {patched_auroc:.3f} vs gate {GATE_PATCHED} "
        f"(dev {deviation:+.3f}) | standins {standins_auroc:.3f} "
        f"(band {STANDINS_BAND}) | PASS {passed}",
    )
    print(f"wrote {args.out}")
    if aborted:
        print("ABORT: standins anchor outside the registered band — no claims")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
