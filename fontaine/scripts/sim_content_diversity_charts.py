"""House dark-mode charts for the sim-content-diversity close:

1. kdist strip — top-cam 5-NN distance distributions, four named
   groups (real held-out / real clean / sim v2 / sim v3), spread
   (k std/mean) + AUROC annotated per sim group. Reads only banked
   analysis jsons.
2. gallery — REAL(A) | v2 | v3 top rows x 4 columns (v3 columns are
   probe-dumped seeds; v2 rendered fresh at the same seeds).

Usage:
  PYTHONPATH=. MUJOCO_GL=egl uv run python \
      fontaine/scripts/sim_content_diversity_charts.py \
      --v2-analysis reports/analysis__sim_encoder_ood_probe_v2_inpaint.json \
      --v3-analysis reports/analysis__sim_encoder_ood_probe_v3_content.json \
      --v3-frames outputs/sim/probe_frames_v3 \
      --out-dir reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# House eval-report scheme (sim100_charts.py / sim_encoder_ood_chart
# .py): dark page, IBM CVD-safe categorical hues; identity is never
# color-alone — groups are named on the y axis.
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
SCALE = 1e5
GALLERY_SEEDS = (2, 6, 8, 13)
REAL_PICKS = (10, 55, 100, 145)  # spread across the 150 A-half dump


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def kdist_chart(v2: dict, v3: dict, out: Path) -> None:
    k2 = v2["cameras"]["top"]["knn5_secondary"]
    k3 = v3["cameras"]["top"]["knn5_secondary"]
    groups = (  # fixed order, bottom to top
        ("sim v3 (plate bank + clutter draws)", k3["distances"]["sim"], "#dc267f", k3),
        ("sim v2 (fixed plate)", k2["distances"]["sim"], "#785ef0", k2),
        ("real clean (anchor)", k3["distances"]["real_clean"], "#ffb000", None),
        ("real v2 held-out", k3["distances"]["real_heldout"], "#648fff", None),
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=150)
    fig.patch.set_facecolor(PAGE)
    style_axis(ax)
    rng = np.random.default_rng(0)
    notes = []
    for row, (_label, values, color, block) in enumerate(groups):
        values = np.asarray(values) * SCALE
        jitter = rng.uniform(-0.24, 0.24, len(values))
        ax.scatter(
            values,
            np.full(len(values), row) + jitter,
            s=14,
            color=color,
            alpha=0.5,
            linewidths=0,
            zorder=2,
        )
        ax.plot(
            [values.mean(), values.mean()],
            [row - 0.34, row + 0.34],
            color=color,
            linewidth=2.5,
            zorder=3,
        )
        note = f"k std/mean {values.std() / values.mean():.3f}"
        if block is not None:
            note += f" · AUROC {block['auroc_sim_vs_real']:.3f}"
        notes.append((row, note))
    right = ax.get_xlim()[1]
    for row, note in notes:
        ax.text(
            right - 0.03,
            row + 0.4,
            note,
            ha="right",
            va="bottom",
            color=META,
            fontsize=8.5,
        )
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups], color=TEXT, fontsize=9)
    ax.set_ylim(-0.6, len(groups) - 0.25)
    ax.set_xlabel("5-NN cosine distance to the real A reference  (x1e-5)", fontsize=9)
    ax.set_title(
        "Content diversity at the policy's eyes: per-reset plate + clutter draws "
        "widen the sim spread (top camera, 100 resets)",
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {out}")


def gallery(v3_frames: Path, out: Path) -> None:
    from PIL import Image, ImageDraw

    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v2")
    rows: list[list[np.ndarray]] = [[], [], []]
    for pick, seed in zip(REAL_PICKS, GALLERY_SEEDS, strict=True):
        real = np.asarray(Image.open(v3_frames / "real_v2" / "top" / f"{pick:04d}.png"))
        rows[0].append(real)
        rows[1].append(sim.reset(seed).top)
        v3 = np.asarray(Image.open(v3_frames / "sim" / "top" / f"{seed:04d}.png"))
        rows[2].append(v3)
    sheet = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
    image = Image.fromarray(sheet).resize((1280, 720), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    for i, label in enumerate(("REAL (A half)", "v2 composite", "v3 draws")):
        y = i * 240 + 8
        draw.rectangle((8, y, 158, y + 22), fill="#121417")
        draw.text((14, y + 4), label, fill="#d8dade")
    image.save(out)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-analysis", type=Path, required=True)
    parser.add_argument("--v3-analysis", type=Path, required=True)
    parser.add_argument("--v3-frames", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    v2 = json.loads(args.v2_analysis.read_text())
    v3 = json.loads(args.v3_analysis.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kdist_chart(v2, v3, args.out_dir / "chart__sim_content_diversity_kdist.png")
    gallery(
        args.v3_frames,
        args.out_dir / "chart__sim_content_diversity_top_gallery.png",
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
