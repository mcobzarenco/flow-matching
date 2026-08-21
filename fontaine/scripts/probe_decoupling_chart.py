"""Probe-decoupling columns chart for the methods note.

Renders the consolidation read behind the probe-decoupling note
(posts/2026-08-21-probe-decoupling-note.md): four cells from the
pdnorm mix lineage, one row each, three columns — the sim100 verdict
(grasps/100 on unseen 0-99), the k4l2 panel truth-fit chunk-MAE, and
the in-train eval probe endpoint. The sim100 column spans 1..28 (a
28x spread); the two offline columns sit in bands of ~1.2 deg and
~1.6 deg with the collapsed cells interleaved among the healthy ones
— the decoupling the note documents.

Every number is read live from the banked artifacts: sim100 counts
from the paired-read jsons, panel rows from the truthfit-wear /
rewear audits, probe endpoints from the runs' train_log.jsonl.

House eval-report dark scheme (pdnorm_panel_ladder_chart constants);
identity is never color-alone — every row is direct-labeled.

Usage:
  uv run python fontaine/scripts/probe_decoupling_chart.py \
      [--out-png fontaine/blog/src/img/probe-decoupling/decoupling_columns.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CKPT_ROOT = Path.home() / "checkpoints/finetune"
OUT_PNG = ROOT / "fontaine/blog/src/img/probe-decoupling/decoupling_columns.png"

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
HEADING = "#eceef1"
BLUE = "#648fff"  # demos + clean (democlean)
MAGENTA = "#dc267f"  # convicted three-way cell
PURPLE = "#785ef0"  # exonerated onerig cell
AMBER = "#ffb000"  # demosonly control (disc-1000)


def paired_success(name: str) -> tuple[int, int]:
    read = json.loads((REPORTS / name).read_text())["read"]["success"]
    return int(read["count_a"]), int(read["count_b"])


def truthfit(name: str) -> float:
    return float(json.loads((REPORTS / name).read_text())["truthfit_wear_chunk_mae"])


def probe_endpoint(run: str) -> tuple[int, float]:
    rows = []
    for ln in (CKPT_ROOT / run / "train_log.jsonl").read_text().splitlines():
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "eval_chunk_mae" in row:
            rows.append((int(row["step"]), float(row["eval_chunk_mae"])))
    assert rows, f"{run}: no eval rows"
    return max(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = parser.parse_args()

    demo_a, onerig = paired_success(
        "analysis__sim100_paired_democlean3000_vs_onerig3000.json",
    )
    demo_b, control = paired_success(
        "analysis__sim100_paired_democlean3000_vs_disc1000.json",
    )
    demo_c, convicted = paired_success(
        "analysis__sim100_paired_democlean3000_vs_pdnorm3000.json",
    )
    assert demo_a == demo_b == demo_c, "democlean count must agree across paired reads"

    panel = {
        "onerig": truthfit("analysis__onerig_endpoint_truthfit_wear.json"),
        "democlean": truthfit("analysis__democlean_endpoint_truthfit_wear.json"),
        "convicted": truthfit("analysis__pdnorm_endpoint_truthfit_wear.json"),
        "control": float(
            json.loads(
                (REPORTS / "analysis__disc1000_panel_row_audit.json").read_text(),
            )["rewear"]["repo_rows_chunk_mae"],
        ),
    }
    released = float(
        json.loads((REPORTS / "analysis__released_row_honest_wear.json").read_text())[
            "honest_wear_chunk_mae"
        ],
    )
    null_mid = float(
        json.loads((REPORTS / "analysis__released_row_honest_wear.json").read_text())[
            "null_repo_midpoint_chunk_mae"
        ],
    )

    probe = {
        "onerig": probe_endpoint("grasp_sft_v2_joint_1gpu_pdnorm_onerig"),
        "democlean": probe_endpoint("grasp_sft_v2_joint_1gpu_pdnorm_democlean"),
        "convicted": probe_endpoint("grasp_sft_v2_joint_1gpu_pdnorm"),
        "control": probe_endpoint("grasp_sft_v2_demosonly_1gpu_disc"),
    }

    # Rows ordered by sim100, best grasper on top.
    rows = [
        ("demos + v2 (onerig)", "onerig", onerig, PURPLE),
        ("demos only (control)", "control", control, AMBER),
        ("demos + clean (democlean)", "democlean", demo_a, BLUE),
        ("three-way mix (convicted)", "convicted", convicted, MAGENTA),
    ]
    ys = list(range(len(rows) - 1, -1, -1))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.9),
        dpi=160,
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )
    fig.patch.set_facecolor(PAGE)

    ax = axes[0]
    for y, (label, _key, sim, color) in zip(ys, rows, strict=True):
        ax.barh(y, sim, height=0.52, color=color, zorder=3)
        ax.text(sim + 0.8, y, f"{sim}", va="center", ha="left", fontsize=9, color=color)
        ax.text(
            -1.6,
            y,
            label,
            va="center",
            ha="right",
            fontsize=9,
            color=TEXT,
            clip_on=False,
        )
    ax.set_xlim(0, 34)
    ax.set_title(
        "sim100 grasps / 100\n(the verdict instrument)",
        color=HEADING,
        fontsize=9.5,
    )

    ax = axes[1]
    ax.axvline(null_mid, color=GRID, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(
        null_mid,
        -0.62,
        f"null {null_mid:.1f}",
        ha="center",
        fontsize=7.5,
        color=META,
    )
    ax.axvline(released, color=META, linewidth=1.0, linestyle=":", zorder=1)
    ax.text(
        released,
        3.45,
        f"released {released:.1f}",
        ha="center",
        fontsize=7.5,
        color=META,
    )
    for y, (_label, key, _sim, color) in zip(ys, rows, strict=True):
        ax.plot([panel[key]], [y], "o", markersize=8, color=color, zorder=3)
        ax.text(
            panel[key],
            y + 0.28,
            f"{panel[key]:.2f}",
            ha="center",
            fontsize=8.5,
            color=color,
        )
    ax.set_xlim(24.3, 29.7)
    ax.set_title(
        "k4l2 panel truth-fit chunk-MAE\n(all four within 1.2 deg)",
        color=HEADING,
        fontsize=9.5,
    )

    ax = axes[2]
    for y, (_label, key, _sim, color) in zip(ys, rows, strict=True):
        step, val = probe[key]
        hollow = step != 3000
        ax.plot(
            [val],
            [y],
            "o",
            markersize=8,
            color=color,
            markerfacecolor=PAGE if hollow else color,
            markeredgewidth=1.8,
            zorder=3,
        )
        note = f"{val:.2f}" + (f" @{step}" if hollow else "")
        ax.text(val, y + 0.28, note, ha="center", fontsize=8.5, color=color)
    ax.set_xlim(4.0, 7.0)
    ax.set_title(
        "eval probe chunk-MAE @3000\n(hollow: run ends @1000)",
        color=HEADING,
        fontsize=9.5,
    )

    for ax in axes:
        ax.set_facecolor(PAGE)
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.set_yticks([])
        ax.tick_params(colors=META, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)

    fig.suptitle(
        "The offline instruments do not rank grasping: 28x sim100 spread,"
        " flat panel and probe columns",
        color=HEADING,
        fontsize=11,
        y=1.04,
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    return_code = main()
    raise SystemExit(return_code)
