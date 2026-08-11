"""House dark-mode charts for the 100-seed sim policy eval
(posts/2026-08-11-prereg-sim-policy-eval-100seeds.md).

Reads ONLY the banked per-arm JSONs (regenerable, no live hosts) and
renders three figures into fontaine/blog/src/img/sim100/:

  1. distance_over_time.png — per-arm mean boat→disk distance vs time
     (episodes that ended early persist their final state, which is what
     the physical boat would do);
  2. progress_strip.png — per-seed progress_final strip + mean per arm;
  3. ordering_vs_panel.png — sim mean progress (bootstrap CI95) vs the
     banked panel MAE per rung — the validation ordering read as a
     picture.

Tolerant of partially-landed arms: plots whatever JSONs exist (the
preliminary in-session render), full set on rc=0.

Usage:
  uv run python fontaine/scripts/sim100_charts.py \
      --in-dir outputs/sim/eval100 --out-dir fontaine/blog/src/img/sim100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fontaine.scripts.sim100_reads import PANEL_MAE, bootstrap_ci

# House eval-report scheme (er60k_screen_close_charts.py): dark page,
# rungs on a single-hue magenta ramp (er60k = the lineage's banked
# magenta, max emphasis), hold floor in neutral gray. Adjacent-pair
# OKLab deltaE 12.9-29.6 on this surface (validated 2026-08-11);
# identity is never color-alone — every series is direct-labeled.
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
ARM_COLORS = {
    "er15k": "#ffd6e8",
    "er35k": "#f593bd",
    "er55k": "#a3125d",
    "er60k": "#dc267f",
    "hold": "#9aa0a8",
}
ARM_ORDER = ("er15k", "er35k", "er55k", "er60k", "hold")
CONTROL_HZ = 30


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def new_fig(width: float = 8.0, height: float = 4.5) -> tuple[Any, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(PAGE)
    style_axis(ax)
    return fig, ax


def load_arms(in_dir: Path) -> dict[str, dict[str, Any]]:
    arms = {}
    for name in ARM_ORDER:
        path = in_dir / f"{name}.json"
        if path.exists():
            arms[name] = json.loads(path.read_text())
    return arms


def padded_mean_series(episodes: list[dict[str, Any]]) -> np.ndarray:
    """Mean distance per tick across seeds; an episode that ended early
    (success latch) holds its final value — the boat stays where it
    ended."""
    longest = max(len(e["distance_cm"]) for e in episodes)
    rows = np.array(
        [
            np.pad(e["distance_cm"], (0, longest - len(e["distance_cm"])), mode="edge")
            for e in episodes
        ],
    )
    return rows.mean(axis=0)


def chart_distance_over_time(arms: dict[str, dict[str, Any]], out: Path) -> None:
    fig, ax = new_fig()
    for name in ARM_ORDER:
        if name not in arms:
            continue
        series = padded_mean_series(arms[name]["episodes"])
        t = np.arange(len(series)) / CONTROL_HZ
        hero = name == "er60k"
        ax.plot(
            t,
            series,
            color=ARM_COLORS[name],
            linewidth=2.6 if hero else 2.0,
            linestyle="--" if name == "hold" else "-",
            zorder=3 if hero else 2,
        )
        ax.annotate(
            name,
            (t[-1], series[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=ARM_COLORS[name],
            fontsize=9,
            va="center",
        )
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean boat→disk distance (cm)")
    ax.set_title("100-seed sim eval: mean distance to goal over the episode")
    ax.set_xlim(left=0)
    fig.savefig(out, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def chart_progress_strip(arms: dict[str, dict[str, Any]], out: Path) -> None:
    fig, ax = new_fig(7.0, 4.2)
    rng = np.random.default_rng(0)
    present = [n for n in ARM_ORDER if n in arms]
    for x, name in enumerate(present):
        values = np.array(
            [e["progress_final_cm"] for e in arms[name]["episodes"]],
        )
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(
            x + jitter,
            values,
            s=14,
            color=ARM_COLORS[name],
            alpha=0.65,
            edgecolors="none",
            zorder=2,
        )
        ax.hlines(
            values.mean(),
            x - 0.28,
            x + 0.28,
            color=TEXT,
            linewidth=2.2,
            zorder=3,
        )
        ax.annotate(
            f"{values.mean():+.2f}",
            (x, values.mean()),
            xytext=(0, 8),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            ha="center",
        )
    ax.axhline(0, color=GRID, linewidth=1.0, zorder=1)
    ax.set_xticks(range(len(present)), present)
    ax.tick_params(axis="x", colors=TEXT)
    ax.set_ylabel("progress: initial − final distance (cm)")
    ax.set_title("Per-seed progress by arm (bar = arm mean)")
    fig.savefig(out, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def chart_ordering_vs_panel(arms: dict[str, dict[str, Any]], out: Path) -> None:
    rungs = [n for n in PANEL_MAE if n in arms]
    if len(rungs) < 2:
        return
    fig, ax = new_fig(6.4, 4.4)
    for name in rungs:
        values = np.array(
            [e["progress_final_cm"] for e in arms[name]["episodes"]],
        )
        low, high = bootstrap_ci(values)
        ax.errorbar(
            PANEL_MAE[name],
            values.mean(),
            yerr=[[values.mean() - low], [high - values.mean()]],
            fmt="o",
            color=ARM_COLORS[name],
            markersize=9,
            capsize=4,
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            name,
            (PANEL_MAE[name], values.mean()),
            xytext=(8, 6),
            textcoords="offset points",
            color=ARM_COLORS[name],
            fontsize=10,
        )
    ax.invert_xaxis()  # panel-better (lower MAE) to the right
    ax.set_xlabel("banked panel MAE (k4l2 core, ← worse | better →)")
    ax.set_ylabel("sim mean progress (cm, CI95)")
    ax.set_title("The validation read: sim progress vs offline panel")
    fig.savefig(out, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    arms = load_arms(args.in_dir)
    if not arms:
        raise SystemExit(f"no arm JSONs in {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chart_distance_over_time(arms, args.out_dir / "distance_over_time.png")
    chart_progress_strip(arms, args.out_dir / "progress_strip.png")
    chart_ordering_vs_panel(arms, args.out_dir / "ordering_vs_panel.png")
    print(f"charts for arms {sorted(arms)} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
