"""Top-cam gap decomposition: WHERE does the remaining knn5 AUROC
0.713 live (queue `sim-top-gap-foreground-decomposition`; anchors:
lens-gate equidistant arm 08-13, contact-shadow gate 08-13)?

Ablation arms on the pinned encoder probe (20 seeds x 5 appearance
draws, settled resets, er_60k trunk, top camera only — the wrist
reads 0.523 under the curve-only fitted lens and is no longer the
frontier). All sim arms render on the numpy post backend so the
constructed arms share the exact composite arithmetic:

  full-frame arms
  - v3            baseline composite (banked anchor 0.713, torch
                  backend; this run re-reads it fresh on numpy).
  - v4            v3 + fitted contact shadow (same seeds; no extra
                  RNG draws, so frames pair 1:1 with v3).
  - fg_to_plate   the composite arithmetic with the rendered
                  foreground CONTENT replaced by the drawn plate:
                  out = w*blur(plate) + (1-w)*plate + noise, with
                  w = clip(blur(remap(mask))) the production edge
                  weight. What remains is the arithmetic residue
                  (edge weighting + foreground PSF on real content).
  - plate_only    out = plate + noise — the drawn plate's own read,
                  the floor every composite inherits.
  - real_fg       real dynamic pixels composited on a drawn plate by
                  the same arithmetic: source frames from bank (A-half)
                  episodes at global timeline indices == stride//2
                  (mod stride) — never coincident with the reference
                  half's == 0 (mod stride) picks; mask = channel-mean
                  |frame - own episode plate| > inlier_delta (erode 1,
                  dilate 2); foreground re-lit source->target episode
                  via the bank affines; pasted on a different drawn
                  plate with the same feathered edge + noise. The
                  upper bound of what the compositing pipeline can
                  reach with perfect foreground appearance.

  shadow-band crop arms — the fixed box where the v4 fitted shadow
  lands (bbox of the mean remapped shadow map >= 0.25 max, padded
  12 px), cropped identically from sim and real frames, embedded as
  their own groups (knn5 against the A crops):
  - crop_v3 / crop_v4   does the band still separate sim from real,
                        and does the fitted shadow close it locally?

Reads per arm: knn5 AUROC vs the held-out real B half (same A/B
protocol as sim_encoder_ood_probe), the clean-repo anchor, paired
per-frame dknn5 vs the v3 baseline with bootstrap CI95 (10k
resamples, rng 0) where frames pair (v4, fg_to_plate, plate_only —
same (seed, draw) slot; real_fg is unpaired by construction), and
the fg_to_plate minus plate_only paired residue.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_top_gap_decomposition.py \
      --out reports/analysis__sim_top_gap_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import av
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import sim_encoder_ood_probe as probe

N_SEEDS = 20
N_DRAWS = 5
N_SLOTS = N_SEEDS * N_DRAWS
TOP_KEY = "observation.images.front"
CROP_THRESHOLD = 0.25
CROP_PAD_PX = 12
BOOTSTRAP_N = 10_000


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
        "--bank-dir",
        type=Path,
        default=Path("assets/real_plates/bank"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dump-frames",
        type=Path,
        default=None,
        help="write sample frames per arm (report/chart fuel)",
    )
    return parser.parse_args()


def render_sim_arms() -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
    """The four sim-side full-frame arms + the accumulated output-space
    shadow map (crop-box fuel). numpy post backend throughout so the
    constructed arms share the production arithmetic bit-for-bit."""
    from sim.so101_sim import SO101Sim

    # standins: the decomposition's registered anchors (v3 0.713) are
    # pre-promotion; the pasted-clutter production read lives elsewhere.
    sim3 = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        clutter_appearance="standins",
    )
    sim4 = SO101Sim(
        render_style="v4",
        post_backend="numpy",
        clutter_appearance="standins",
    )
    noise_rng = np.random.default_rng(0)
    arms: dict[str, list[np.ndarray]] = {
        name: [] for name in ("v3", "v4", "fg_to_plate", "plate_only")
    }
    shadow_sum = np.zeros(sim3._render_size, dtype=np.float64)
    for seed in range(N_SEEDS):
        for draw in range(N_DRAWS):
            appearance = 1000 * draw + seed
            obs3 = sim3.reset(seed, appearance_seed=appearance)
            arms["v3"].append(obs3.top)
            obs4 = sim4.reset(seed, appearance_seed=appearance)
            arms["v4"].append(obs4.top)
            plate = sim3._active_top_plate
            if not np.array_equal(plate, sim4._active_top_plate):
                raise SystemExit(
                    f"seed {seed} draw {draw}: v3/v4 drew different plates "
                    "— the paired-frame invariant is broken",
                )
            # The production mask/shadow re-rendered at the settled
            # state (deterministic; observe() leaves the drawn clutter
            # poses in place).
            mask = sim3._render_mask("top_cam")
            if not 0.005 < mask.mean() < 0.6:
                raise SystemExit(
                    f"seed {seed} draw {draw}: dynamic mask fraction "
                    f"{mask.mean():.3f} outside sanity range",
                )
            shadow = sim4._render_shadow("top_cam", sim4._render_mask("top_cam"))
            shadow_sum += sim4._remap(shadow[..., None])[..., 0]
            weight = np.clip(sim3._blur(sim3._remap(mask[..., None])), 0.0, 1.0)
            noise = (
                noise_rng.standard_normal((*plate.shape[:2], 3), dtype=np.float32)
                * sim3.V1_NOISE_SIGMA
            )
            fg_to_plate = weight * sim3._blur(plate) + (1.0 - weight) * plate + noise
            arms["fg_to_plate"].append(
                np.clip(fg_to_plate, 0, 255).astype(np.uint8),
            )
            noise = (
                noise_rng.standard_normal((*plate.shape[:2], 3), dtype=np.float32)
                * sim3.V1_NOISE_SIGMA
            )
            arms["plate_only"].append(
                np.clip(plate + noise, 0, 255).astype(np.uint8),
            )
    return arms, shadow_sum / N_SLOTS


def erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for axis in (0, 1):
        for shift in (1, -1):
            out = np.minimum(out, np.roll(mask, shift, axis=axis))
    return out


def dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for axis in (0, 1):
        for shift in (1, -1):
            out = np.maximum(out, np.roll(mask, shift, axis=axis))
    return out


def load_bank(
    bank_dir: Path,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]], float]:
    from PIL import Image

    manifest = json.loads((bank_dir / "bank_manifest.json").read_text())
    plates = {
        int(path.stem.removeprefix("top_ep")): np.asarray(
            Image.open(path),
            dtype=np.float64,
        )
        for path in sorted(bank_dir.glob("top_ep*.png"))
    }
    affines = {
        int(index): (np.array(entry["gain"]), np.array(entry["bias"]))
        for index, entry in manifest["episodes"].items()
    }
    if set(plates) != set(affines):
        raise SystemExit("bank plates and manifest episodes disagree")
    return plates, affines, float(manifest["inlier_delta"])


def real_foreground_frames(
    v2_root: Path,
    bank_dir: Path,
    blur: Callable[[np.ndarray], np.ndarray],
    noise_sigma: float,
) -> tuple[list[np.ndarray], list[dict[str, int]]]:
    """N_SLOTS composites of real dynamic pixels on a drawn bank plate
    (arm real_fg). Source frames sit at global timeline indices
    == stride//2 (mod stride) — disjoint from the probe's reference
    picks (== 0 mod stride) by construction — and only frames whose
    episode is a bank (A-half) episode are eligible."""
    import pandas as pd

    plates, affines, delta = load_bank(bank_dir)
    episodes = pd.read_parquet(
        v2_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    spans: dict[int, list[tuple[float, float, int]]] = {}
    for _, row in episodes.iterrows():
        spans.setdefault(int(row[f"videos/{TOP_KEY}/file_index"]), []).append(
            (
                float(row[f"videos/{TOP_KEY}/from_timestamp"]),
                float(row[f"videos/{TOP_KEY}/to_timestamp"]),
                int(row["episode_index"]),
            ),
        )
    files = sorted((v2_root / "videos" / TOP_KEY / "chunk-000").glob("*.mp4"))
    stride = probe.total_frames(files) // probe.N_REAL_V2
    offset = stride // 2

    rng = np.random.default_rng(0)
    noise_rng = np.random.default_rng(1)
    frames: list[np.ndarray] = []
    slots: list[dict[str, int]] = []
    index = 0
    for path in files:
        file_spans = sorted(spans[int(path.stem.removeprefix("file-"))])
        container = av.open(str(path))
        for frame in container.decode(video=0):
            take = index % stride == offset
            index += 1
            if not take or len(frames) == N_SLOTS:
                continue
            t = float(frame.time)
            episode = next(
                (ep for start, end, ep in file_spans if start <= t < end),
                None,
            )
            if episode is None or episode not in plates:
                continue
            rgb = frame.to_ndarray(format="rgb24").astype(np.float64)
            source_plate = plates[episode]
            mask = (np.abs(rgb - source_plate).mean(axis=-1) > delta).astype(np.float64)
            mask = dilate(dilate(erode(mask)))
            if not 0.01 < mask.mean() < 0.6:
                raise SystemExit(
                    f"global frame {index - 1} (episode {episode}): real "
                    f"foreground mask fraction {mask.mean():.3f} outside "
                    "sanity range",
                )
            others = [ep for ep in plates if ep != episode]
            target = int(others[rng.integers(len(others))])
            gain_s, bias_s = affines[episode]
            gain_t, bias_t = affines[target]
            foreground = (rgb - bias_s) / gain_s * gain_t + bias_t
            weight = np.clip(blur(mask[..., None]), 0.0, 1.0)
            noise = (
                noise_rng.standard_normal((*mask.shape, 3), dtype=np.float32)
                * noise_sigma
            )
            out = weight * foreground + (1.0 - weight) * plates[target] + noise
            frames.append(np.clip(out, 0, 255).astype(np.uint8))
            slots.append(
                {
                    "global_index": index - 1,
                    "source_episode": episode,
                    "target_episode": target,
                },
            )
        container.close()
        if len(frames) == N_SLOTS:
            break
    if len(frames) != N_SLOTS:
        raise SystemExit(f"real_fg: {len(frames)}/{N_SLOTS} eligible frames")
    return frames, slots


def crop_box(mean_shadow: np.ndarray) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) — bbox of the mean shadow map's live region,
    padded, clamped to the frame."""
    live = mean_shadow >= CROP_THRESHOLD * mean_shadow.max()
    ys, xs = np.nonzero(live)
    height, width = mean_shadow.shape
    y0 = max(0, int(ys.min()) - CROP_PAD_PX)
    y1 = min(height, int(ys.max()) + 1 + CROP_PAD_PX)
    x0 = max(0, int(xs.min()) - CROP_PAD_PX)
    x1 = min(width, int(xs.max()) + 1 + CROP_PAD_PX)
    if y1 - y0 < 48 or x1 - x0 < 48:
        raise SystemExit(f"shadow crop box degenerate: {(y0, y1, x0, x1)}")
    return y0, y1, x0, x1


def knn5(embeddings: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
    pairwise = 1.0 - embeddings @ ref.T
    return pairwise.topk(5, dim=1, largest=False).values.mean(dim=1).numpy()


def paired_read(arm: np.ndarray, base: np.ndarray) -> dict[str, object]:
    delta = arm - base
    rng = np.random.default_rng(0)
    boots = np.array(
        [float(np.mean(rng.choice(delta, len(delta)))) for _ in range(BOOTSTRAP_N)],
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "n_closer": int((delta < 0).sum()),
        "n": len(delta),
    }


def arm_read(scores: np.ndarray, held: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "auroc_vs_real": probe.auroc(scores, held),
        "distances": [float(f"{v:.3e}") for v in scores],
    }


def main() -> int:
    args = parse_args()

    print("rendering sim arms (numpy backend, 20 seeds x 5 draws) ...")
    sim_arms, mean_shadow = render_sim_arms()
    box = crop_box(mean_shadow)
    print(f"shadow-band crop box (y0, y1, x0, x1): {box}")

    from sim.so101_sim import SO101Sim

    blur_sim = SO101Sim(render_style="v0")  # blur/noise constants only
    print("building real_fg arm ...")
    real_fg, slots = real_foreground_frames(
        args.v2_root,
        args.bank_dir,
        blur_sim._blur,
        blur_sim.V1_NOISE_SIGMA,
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

    groups: dict[str, list[np.ndarray]] = {**sim_arms, "real_fg": real_fg, **real}
    y0, y1, x0, x1 = box
    crops = {
        f"crop_{name}": [f[y0:y1, x0:x1] for f in groups[name]]
        for name in ("v3", "v4", "real_v2", "real_clean")
    }

    if args.dump_frames is not None:
        from PIL import Image

        for name, frames in {**groups, **crops}.items():
            out_dir = args.dump_frames / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in (0, 1, 2):
                Image.fromarray(frames[i]).save(out_dir / f"{i:04d}.png")

    model, info = probe.from_checkpoint(args.checkpoint, device="cuda")
    vision = model.backbone.vision
    del model.decoder
    emb = {}
    for name, frames in {**groups, **crops}.items():
        emb[name] = probe.embed(vision, frames)
        print(f"embedded {name}: {tuple(emb[name].shape)}")

    half = probe.N_REAL_V2 // 2
    results: dict[str, dict[str, object]] = {}
    for space, ref_name in (("full_frame", "real_v2"), ("crop", "crop_real_v2")):
        ref = emb[ref_name][:half]
        held = knn5(emb[ref_name][half:], ref)
        prefix = "crop_" if space == "crop" else ""
        arm_names = (
            ("v3", "v4", "fg_to_plate", "plate_only", "real_fg")
            if space == "full_frame"
            else ("v3", "v4")
        )
        scores = {name: knn5(emb[prefix + name], ref) for name in arm_names}
        clean = knn5(emb[prefix + "real_clean"], ref)
        arms_read = {name: arm_read(scores[name], held) for name in arm_names}
        clean_read = arm_read(clean, held)
        block: dict[str, object] = {
            "real_heldout": {"mean": float(held.mean()), "std": float(held.std())},
            "clean_anchor": clean_read,
            "arms": arms_read,
        }
        paired = {
            name: paired_read(scores[name], scores["v3"])
            for name in arm_names
            if name not in ("v3", "real_fg")  # real_fg is unpaired
        }
        block["paired_vs_v3"] = paired
        if space == "full_frame":
            block["arithmetic_residue_fg_to_plate_minus_plate_only"] = paired_read(
                scores["fg_to_plate"],
                scores["plate_only"],
            )
        results[space] = block
        for name in arm_names:
            print(
                f"[{space}] {name}: knn5 {scores[name].mean():.3e} | "
                f"AUROC {arms_read[name]['auroc_vs_real']:.3f}",
            )
        print(
            f"[{space}] clean anchor AUROC {clean_read['auroc_vs_real']:.3f}",
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
            "protocol": "sim_encoder_ood_probe A/B split, top camera; "
            "20 seeds x 5 appearance draws, settled resets, numpy post "
            "backend for every sim arm",
            "anchors": {
                "banked_v3_20x5_torch": 0.713,
                "v4_gate_100x1": {"v3": 0.721, "v4": 0.715},
            },
            "crop_box_y0_y1_x0_x1": list(box),
            "crop_rule": f"mean v4 shadow map >= {CROP_THRESHOLD} max, "
            f"padded {CROP_PAD_PX} px",
            "real_fg_slots": slots,
            "commit": commit,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
