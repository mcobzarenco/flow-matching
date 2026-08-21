"""Probe-curve contrast chart for the democlean endpoint close.

Renders the registered record-only read (pre-reg
posts/2026-08-20-prereg-demos-plus-clean.md, "Record-only" (2)): the
demos+clean cell's eval-250 probe curve against the convicted
three-way cell's curve (the 2250-2750 elevation signature this cell
hunts) and the exonerated onerig cell's curve, all read live from the
three runs' banked train_log.jsonl eval rows. The convicted elevation
window is banded so the read is visual: did clean-alone reproduce it?

House eval-report dark scheme (pdnorm_panel_ladder_chart constants);
identity is never color-alone — every series is direct-labeled at its
endpoint.

Usage:
  uv run python fontaine/scripts/democlean_probe_curve_chart.py \
      [--out-png fontaine/blog/src/img/democlean/probe_curve_contrast.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_PNG = ROOT / "fontaine/blog/src/img/democlean/probe_curve_contrast.png"

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
HEADING = "#eceef1"
BLUE = "#648fff"  # subject: demos + clean
MAGENTA = "#dc267f"  # convicted three-way cell
PURPLE = "#785ef0"  # exonerated onerig cell

CKPT_ROOT = Path.home() / "checkpoints/finetune"
RUNS = [
    ("demos + clean (this cell)", "grasp_sft_v2_joint_1gpu_pdnorm_democlean", BLUE, 9),
    ("convicted three-way (1/100)", "grasp_sft_v2_joint_1gpu_pdnorm", MAGENTA, 0),
    ("onerig demos+v2 (28/100)", "grasp_sft_v2_joint_1gpu_pdnorm_onerig", PURPLE, -9),
]


def eval_curve(run: str) -> list[tuple[int, float]]:
    rows = []
    for ln in (CKPT_ROOT / run / "train_log.jsonl").read_text().splitlines():
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "eval_chunk_mae" in row:
            rows.append((int(row["step"]), float(row["eval_chunk_mae"])))
    assert len(rows) == 12, f"{run}: expected 12 eval rows, got {len(rows)}"
    return sorted(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=160)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)

    # The convicted elevation window: the signature this cell hunts.
    ax.axvspan(2250, 2750, color=MAGENTA, alpha=0.08, zorder=0)
    ax.text(
        2500,
        12.6,
        "convicted elevation\nwindow (2250–2750)",
        ha="center",
        va="top",
        fontsize=8,
        color=META,
    )

    for label, run, color, dy in RUNS:
        curve = eval_curve(run)
        steps = [s for s, _ in curve]
        vals = [v for _, v in curve]
        ax.plot(
            steps,
            vals,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            zorder=3,
        )
        ax.annotate(
            f"{label}  {vals[-1]:.2f}",
            (steps[-1], vals[-1]),
            xytext=(8, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            va="center",
        )

    ax.set_xlim(200, 4150)
    ax.set_xticks([250, 500, 1000, 1500, 2000, 2500, 3000])
    ax.set_xlabel("training step", color=TEXT, fontsize=9)
    ax.set_ylabel("eval_chunk_mae (probe, deg)", color=TEXT, fontsize=9)
    ax.set_title(
        "Did clean-alone reproduce the poison signature? — eval probe curves,"
        " demos+clean vs the convicted and exonerated cells",
        color=HEADING,
        fontsize=10.5,
        pad=12,
    )
    ax.tick_params(colors=META, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5, zorder=1)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    return_code = main()
    raise SystemExit(return_code)
