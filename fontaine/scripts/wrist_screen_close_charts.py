"""Wrist-transfer screen close-out charts (results post, house
eval-report scheme, dark-mode) — consumes the stage-1 cell JSONs the
boundary reads consumed (`outputs/sim/wrist_screen/stage1_*.json`) and
recomputes the paired deltas with the exact wrist_stage1_reads.py
recipe (per-seed treatment − W0, bootstrap CI95 seed 0, 10k), so every
number drawn matches reports/analysis__wrist_screen_stage1.json.

Outputs (fontaine/blog/src/img/wrist_screen/):
  delta_strips.svg     per-seed Δprogress strips per arm, shared axis —
                       the n=25 control's CI width vs n=100 is the point
  engagement_split.svg engagement (moved ≥0.5 cm) rate per arm, with
                       the W3 flip CI and the T1 sample-size caveat

Usage: uv run python fontaine/scripts/wrist_screen_close_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/wrist_screen"
CELLS = ROOT / "outputs/sim/wrist_screen"

PAGE = "#121417"
BLUE = "#648fff"  # wrist treatments (W1, W3)
MAGENTA = "#dc267f"  # the T1 top-blackout positive control
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

UNTOUCHED_CM = 0.5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def load_arm(name: str) -> dict[str, np.ndarray]:
    episodes = json.loads((CELLS / f"stage1_{name}.json").read_text())["episodes"]
    return {
        "progress": np.array([e["progress_final_cm"] for e in episodes]),
        "moved": np.array(
            [abs(e["progress_final_cm"]) >= UNTOUCHED_CM for e in episodes],
            dtype=float,
        ),
    }


def bootstrap_ci(deltas: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas)), replace=True)
    low, high = np.percentile(draws.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.svg")


def delta_strips(arms: dict[str, dict[str, np.ndarray]]) -> None:
    panels = (
        ("W1 blackout − W0", "w1", BLUE, 100),
        ("W3 arm-blur − W0", "w3", BLUE, 100),
        ("T1 top-blackout − W0\n(the positive control)", "t1", MAGENTA, 25),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
    fig.patch.set_facecolor(PAGE)
    rng = np.random.default_rng(7)  # jitter only — display, not stats
    for ax, (title, name, color, n) in zip(axes, panels, strict=True):
        deltas = arms[name]["progress"][:n] - arms["w0"]["progress"][:n]
        low, high = bootstrap_ci(deltas)
        x = rng.uniform(-0.16, 0.16, size=len(deltas))
        ax.axhline(0, color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(
            x,
            deltas,
            s=14,
            color=color,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
        ax.errorbar(
            0.42,
            float(deltas.mean()),
            yerr=[[float(deltas.mean()) - low], [high - float(deltas.mean())]],
            fmt="o",
            color=TEXT,
            ecolor=TEXT,
            elinewidth=2,
            capsize=5,
            markersize=6,
            zorder=4,
        )
        ax.annotate(
            f"{deltas.mean():+.2f}\n[{low:+.2f}, {high:+.2f}]",
            (0.42, high),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=TEXT,
            fontsize=8.5,
        )
        ax.set_title(f"{title}  (n={n})", color=META, fontsize=9)
        ax.set_xlim(-0.35, 0.7)
        ax.set_xticks([])
        style_axis(ax)
    axes[0].set_ylabel("Δ final progress (cm, per paired seed)", color=META, fontsize=9)
    fig.suptitle(
        "Every arm's paired CI straddles zero — the control included",
        color=TEXT,
        fontsize=11,
        y=1.02,
    )
    save(fig, "delta_strips")


def engagement_split(arms: dict[str, dict[str, np.ndarray]]) -> None:
    rows = (
        ("W0\nclassic", "w0", BLUE, 100),
        ("W1\nblackout", "w1", BLUE, 100),
        ("W3\narm-blur", "w3", BLUE, 100),
        ("T1 top-blackout\n(n=25 control)", "t1", MAGENTA, 25),
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor(PAGE)
    for i, (_label, name, color, n) in enumerate(rows):
        moved = arms[name]["moved"][:n]
        rate = float(moved.mean())
        ax.bar(i, rate * 100, width=0.62, color=color, zorder=3)
        ax.annotate(
            f"{int(moved.sum())}/{n}",
            (i, rate * 100),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color=TEXT,
            fontsize=9.5,
        )
    flips = arms["w3"]["moved"][:100] - arms["w0"]["moved"][:100]
    low, high = bootstrap_ci(flips)
    ax.annotate(
        f"W3−W0 engagement flips {flips.mean():+.2f}\n"
        f"CI95 [{low:+.2f}, {high:+.2f}] — excludes zero",
        (2, arms["w3"]["moved"][:100].mean() * 100),
        textcoords="offset points",
        xytext=(14, 26),
        color=TEXT,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.8},
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], color=META, fontsize=9)
    ax.set_ylabel("episodes moved ≥ 0.5 cm (%)", color=META, fontsize=9)
    ax.set_ylim(0, 80)
    style_axis(ax)
    ax.set_title(
        "Engagement by arm: blurring the ARM out of the wrist view raises it",
        color=TEXT,
        fontsize=11,
    )
    save(fig, "engagement_split")


def main() -> int:
    arms = {name: load_arm(name) for name in ("w0", "w1", "w3", "t1")}
    banked = json.loads(
        (ROOT / "reports/analysis__wrist_screen_stage1.json").read_text(),
    )
    w3 = banked["paired_reads"]["w3_minus_w0"]["moved_flips"]
    flips = arms["w3"]["moved"][:100] - arms["w0"]["moved"][:100]
    if round(float(flips.mean()), 4) != w3["mean"]:
        sys.exit(
            f"ABORT: recomputed W3 flips {flips.mean():.4f} != banked {w3['mean']}",
        )
    delta_strips(arms)
    engagement_split(arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
