"""Encoder OOD probe: the sim-vs-real visual gap measured at the
policy's own eyes (owner ask 01:11Z 08-12; rides the sim100 pre-reg,
posts/2026-08-11-prereg-sim-policy-eval-100seeds.md).

Embeds three frame groups per camera through the frozen er_60k vision
trunk (the exact eval mount: ``from_checkpoint`` bf16 backbone,
``max_crops=1``, features = the Molmo2VisionBackbone pooled+projected
tokens the LM consumes; per-frame embedding = fp32 mean over the
image's valid vision tokens, L2-normalized):

  - SIM: the banked sim100 er60k-arm rollout videos
    (outputs/sim/eval100/er60k/rollout_seed*.mp4, top|wrist halves),
    ticks {0, 300, 600} per seed;
  - REAL v2: mcobzarenco/so101_pick_place_v2 camera videos (front is
    the rig's top view per the judge kind stamp), every
    ``total//300``-th frame of the concatenated v3.0 timeline, first
    300; split at the timeline midpoint into A (first 150, the
    reference set) and B (last 150, the held-out real control —
    disjoint episodes from A by construction);
  - REAL clean: mcobzarenco/so101_pick_place_clean, every
    ``total//100``-th frame, 100 — the same-rig-different-repo anchor.

Read, per camera: c_A = L2-normalized mean of the A embeddings;
d(x) = 1 - cos(emb(x), c_A). Reported: mean/std of d over B, clean and
sim; gap ratio mean d(sim) / mean d(B); AUROC of d separating sim from
B (0.5 = indistinguishable, 1.0 = perfectly separable); sim per-tick
breakdown. Record-only; the number is the baseline the
sim-visual-matching lever must move.

SECONDARY read (labeled, added after the first run of the registered
primary): the raw embeddings share a constant direction so dominant
that every centroid-cosine distance is a ~1e-5 residual — a weak 1-D
lens. knn5(x) = mean cosine distance to the 5 nearest A embeddings
(local-support coverage, the standard OOD score) with the same
B/clean/sim + ratio + AUROC reads on it.

Usage:
  uv run python fontaine/scripts/sim_encoder_ood_probe.py \
      --out reports/analysis__sim_encoder_ood_probe.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

from bijou.loading import from_checkpoint
from bijou.molmo2.processor import process_image

SIM_TICKS = (0, 300, 600)
N_REAL_V2 = 300
N_REAL_CLEAN = 100
BATCH = 32

# Sim rollout videos are top|wrist side-by-side; the rig repos record
# the top view under "front" (data-side mislabel, judge-stamped kind
# top) — pair them under the semantic camera name.
CAMERAS = ("top", "wrist")
REAL_KEYS = {"top": "observation.images.front", "wrist": "observation.images.wrist"}


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
        "--sim-dir",
        type=Path,
        default=Path("outputs/sim/eval100/er60k"),
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
        help="also write the selected frames as PNGs (chart/gallery fuel)",
    )
    parser.add_argument(
        "--render-resets",
        type=int,
        default=None,
        metavar="N",
        help="sim source = live settled reset frames of seeds 0..N-1 "
        "(the visual-matching iteration read; needs MUJOCO_GL=egl) "
        "instead of the banked rollout videos",
    )
    parser.add_argument(
        "--appearance-draws",
        type=int,
        default=1,
        metavar="K",
        help="with --render-resets: render each seed K times with "
        "appearance seed 1000*draw+seed (texture-sensitivity read)",
    )
    parser.add_argument(
        "--render-style",
        default=None,
        help="with --render-resets: SO101Sim render_style override "
        "(default: the class default)",
    )
    parser.add_argument(
        "--lens-model",
        default=None,
        help="with --render-resets: SO101Sim lens_model override "
        "('equidistant' | 'fitted'; default: the class default)",
    )
    return parser.parse_args()


def decode_strided(files: list[Path], stride: int, count: int) -> list[np.ndarray]:
    """Every ``stride``-th frame of the files' concatenated timeline
    (sorted order), first ``count`` — HWC uint8."""
    frames: list[np.ndarray] = []
    index = 0
    for path in files:
        container = av.open(str(path))
        for frame in container.decode(video=0):
            if len(frames) == count:
                break
            if index % stride == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
            index += 1
        container.close()
        if len(frames) == count:
            break
    if len(frames) != count:
        raise SystemExit(
            f"selected {len(frames)}/{count} frames from {files[0].parent}",
        )
    return frames


def total_frames(files: list[Path]) -> int:
    total = 0
    for path in files:
        container = av.open(str(path))
        n = container.streams.video[0].frames
        container.close()
        if not n:
            raise SystemExit(f"{path}: container reports no frame count")
        total += n
    return total


def sim_frames(sim_dir: Path) -> dict[str, list[np.ndarray]]:
    """Ticks SIM_TICKS of every rollout video, split into camera halves."""
    videos = sorted(sim_dir.glob("rollout_seed*.mp4"))
    if not videos:
        raise SystemExit(f"no rollout videos under {sim_dir}")
    per_camera: dict[str, list[np.ndarray]] = {name: [] for name in CAMERAS}
    for path in videos:
        container = av.open(str(path))
        wanted = set(SIM_TICKS)
        for index, frame in enumerate(container.decode(video=0)):
            if index > max(SIM_TICKS):
                break
            if index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                half = rgb.shape[1] // 2
                per_camera["top"].append(rgb[:, :half])
                per_camera["wrist"].append(rgb[:, half:])
        container.close()
    expected = len(videos) * len(SIM_TICKS)
    for name, frames in per_camera.items():
        if len(frames) != expected:
            raise SystemExit(f"sim {name}: {len(frames)} frames, expected {expected}")
    return per_camera


@torch.inference_mode()
def embed(vision: torch.nn.Module, frames: list[np.ndarray]) -> torch.Tensor:
    """[N, text_hidden] fp32 L2-normalized mean-over-tokens embeddings."""
    processed = [
        process_image(
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0,
            max_crops=1,
        )
        for frame in frames
    ]
    grids = {p.grid for p in processed}
    if len(grids) != 1:
        raise SystemExit(f"mixed image grids {grids} — same-shape batching only")
    out: list[torch.Tensor] = []
    for start in range(0, len(processed), BATCH):
        chunk = processed[start : start + BATCH]
        crops = torch.stack([p.crops for p in chunk])
        pooled_idx = torch.stack([p.pooled_idx for p in chunk])
        tokens = vision(crops.cuda(), pooled_idx.cuda())  # [B*P_valid, D]
        per_image = (pooled_idx >= 0).any(-1).sum(-1)
        if len(set(per_image.tolist())) != 1:
            raise SystemExit("uneven valid-token counts in a same-grid batch")
        tokens = tokens.view(len(chunk), int(per_image[0]), -1).float()
        out.append(torch.nn.functional.normalize(tokens.mean(dim=1), dim=-1).cpu())
    return torch.cat(out)


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUROC of score d separating positive (sim) from negative (real) —
    the Mann-Whitney U statistic, ties at half credit."""
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties.
    for value in np.unique(scores):
        mask = scores == value
        ranks[mask] = ranks[mask].mean()
    u = ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2
    return float(u / (len(positive) * len(negative)))


def rendered_reset_frames(
    n_seeds: int,
    appearance_draws: int,
    render_style: str | None = None,
    lens_model: str | None = None,
) -> dict[str, list[np.ndarray]]:
    """Live settled reset frames, seeds 0..n_seeds-1 (x appearance
    draws), through the exact SO101Sim.observe() path the policy sees.
    Frame order: seed-major, draw-minor."""
    from sim.so101_sim import SO101Sim

    kwargs: dict[str, Any] = {}
    if render_style is not None:
        kwargs["render_style"] = render_style
    if lens_model is not None:
        kwargs["lens_model"] = lens_model
    sim = SO101Sim(**kwargs)
    per_camera: dict[str, list[np.ndarray]] = {name: [] for name in CAMERAS}
    for seed in range(n_seeds):
        for draw in range(appearance_draws):
            kwargs = (
                {}
                if appearance_draws == 1
                else {
                    "appearance_seed": 1000 * draw + seed,
                }
            )
            obs = sim.reset(seed, **kwargs)
            per_camera["top"].append(obs.top)
            per_camera["wrist"].append(obs.wrist)
    return per_camera


def main() -> int:
    args = parse_args()
    model, info = from_checkpoint(args.checkpoint, device="cuda")
    vision = model.backbone.vision
    del model.decoder  # inference on the vision trunk only

    groups: dict[str, dict[str, list[np.ndarray]]] = {}
    if args.render_resets is not None:
        sim = rendered_reset_frames(
            args.render_resets,
            args.appearance_draws,
            args.render_style,
            args.lens_model,
        )
    else:
        sim = sim_frames(args.sim_dir)
    for name in CAMERAS:
        groups.setdefault("sim", {})[name] = sim[name]
    for group, root, count in (
        ("real_v2", args.v2_root, N_REAL_V2),
        ("real_clean", args.clean_root, N_REAL_CLEAN),
    ):
        for name in CAMERAS:
            files = sorted(
                (root / "videos" / REAL_KEYS[name] / "chunk-000").glob("*.mp4"),
            )
            total = total_frames(files)
            frames = decode_strided(files, total // count, count)
            groups.setdefault(group, {})[name] = frames
            print(
                f"{group}/{name}: {len(frames)} frames of {total} (stride {total // count})",
            )

    if args.dump_frames is not None:
        from PIL import Image

        for group, cameras in groups.items():
            for name, frames in cameras.items():
                out_dir = args.dump_frames / group / name
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(frames):
                    Image.fromarray(frame).save(out_dir / f"{i:04d}.png")

    results: dict[str, dict[str, object]] = {}
    for name in CAMERAS:
        emb = {group: embed(vision, cameras[name]) for group, cameras in groups.items()}
        print(f"{name}: embedded { ({g: tuple(e.shape) for g, e in emb.items()}) }")
        v2 = emb["real_v2"]
        ref, held = v2[: N_REAL_V2 // 2], v2[N_REAL_V2 // 2 :]
        centroid = torch.nn.functional.normalize(ref.mean(dim=0), dim=-1)

        def dist(
            embeddings: torch.Tensor,
            centroid: torch.Tensor = centroid,
        ) -> np.ndarray:
            return (1.0 - embeddings @ centroid).numpy()

        def knn5(embeddings: torch.Tensor, ref: torch.Tensor = ref) -> np.ndarray:
            pairwise = 1.0 - embeddings @ ref.T  # [N, |A|]
            return pairwise.topk(5, dim=1, largest=False).values.mean(dim=1).numpy()

        d_held, d_clean, d_sim = dist(held), dist(emb["real_clean"]), dist(emb["sim"])
        k_held, k_clean, k_sim = knn5(held), knn5(emb["real_clean"]), knn5(emb["sim"])
        sim_ticks = (0,) if args.render_resets is not None else SIM_TICKS
        ticks = np.array([sim_ticks[i % len(sim_ticks)] for i in range(len(d_sim))])
        knn = {
            "real_heldout": {"mean": float(k_held.mean()), "std": float(k_held.std())},
            "real_clean": {"mean": float(k_clean.mean()), "std": float(k_clean.std())},
            "sim": {"mean": float(k_sim.mean()), "std": float(k_sim.std())},
            "gap_ratio_sim_vs_real": float(k_sim.mean() / k_held.mean()),
            "gap_ratio_clean_vs_real": float(k_clean.mean() / k_held.mean()),
            "auroc_sim_vs_real": auroc(k_sim, k_held),
            "auroc_clean_vs_real": auroc(k_clean, k_held),
            "distances": {
                "real_heldout": [float(f"{v:.3e}") for v in k_held],
                "real_clean": [float(f"{v:.3e}") for v in k_clean],
                "sim": [float(f"{v:.3e}") for v in k_sim],
            },
        }
        results[name] = {
            "knn5_secondary": knn,
            "real_heldout": {
                "mean": float(d_held.mean()),
                "std": float(d_held.std()),
                "n": len(d_held),
            },
            "real_clean": {
                "mean": float(d_clean.mean()),
                "std": float(d_clean.std()),
                "n": len(d_clean),
            },
            "sim": {
                "mean": float(d_sim.mean()),
                "std": float(d_sim.std()),
                "n": len(d_sim),
            },
            "gap_ratio_sim_vs_real": float(d_sim.mean() / d_held.mean()),
            "gap_ratio_clean_vs_real": float(d_clean.mean() / d_held.mean()),
            "auroc_sim_vs_real": auroc(d_sim, d_held),
            "auroc_clean_vs_real": auroc(d_clean, d_held),
            "sim_per_tick": {
                str(tick): {
                    "mean": float(d_sim[ticks == tick].mean()),
                    "std": float(d_sim[ticks == tick].std()),
                }
                for tick in sim_ticks
            },
            "distances": {
                "real_heldout": [float(f"{v:.3e}") for v in d_held],
                "real_clean": [float(f"{v:.3e}") for v in d_clean],
                "sim": [float(f"{v:.3e}") for v in d_sim],
            },
        }
        print(
            f"[{name}] centroid: d(realB) {d_held.mean():.3e} | "
            f"d(clean) {d_clean.mean():.3e} | d(sim) {d_sim.mean():.3e} | "
            f"ratio {d_sim.mean() / d_held.mean():.2f}x | "
            f"AUROC {results[name]['auroc_sim_vs_real']:.3f}",
        )
        print(
            f"[{name}] knn5:     k(realB) {k_held.mean():.3e} | "
            f"k(clean) {k_clean.mean():.3e} | k(sim) {k_sim.mean():.3e} | "
            f"ratio {knn['gap_ratio_sim_vs_real']:.2f}x | "
            f"AUROC {knn['auroc_sim_vs_real']:.3f}",
        )

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
            "feature": "Molmo2VisionBackbone pooled+projected tokens, "
            "fp32 mean over valid tokens, L2-normalized; max_crops=1; "
            "bf16 eval mount",
            "distance": "1 - cosine to the L2-normalized real_v2 "
            "reference-half centroid (first 150 strided frames)",
            "sim_source": (
                f"live reset renders, seeds 0..{args.render_resets - 1}"
                f" x {args.appearance_draws} appearance draws, "
                f"render_style={args.render_style or 'default'}, "
                f"lens_model={args.lens_model or 'default'}"
                if args.render_resets is not None
                else str(args.sim_dir)
            ),
            "sim_ticks": list(
                (0,) if args.render_resets is not None else SIM_TICKS,
            ),
            "real_v2": {
                "root": str(args.v2_root),
                "n": N_REAL_V2,
                "split": "timeline halves",
            },
            "real_clean": {"root": str(args.clean_root), "n": N_REAL_CLEAN},
            "commit": commit,
        },
        "cameras": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
