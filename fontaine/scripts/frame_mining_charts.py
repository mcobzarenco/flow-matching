"""Charts for the frame-mining stage of the field/subgoal meta-report
(owner steering 2026-08-08 13:21Z). Rendered from the BANKED mining
artifacts only — no re-computation of claimed numbers
(`analysis__framemining_ar100k_k4l2.json`,
`framemining__ar100k_k4l2__flagged.npz`).

Output: fontaine/blog/src/img/framemining/*.svg (committed).

Palette: the dataviz reference palette's pre-validated categorical
slots + neutral text tokens, matching the goldenticket charts (no node
on the box, so the palette validator can't run; the reference instance
unchanged is the sanctioned fallback).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/framemining"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#9b9a95"
DARKGRAY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=DARKGRAY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(width: float = 8.2, height: float = 4.2) -> tuple:
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def chart_concentration(analysis: dict) -> None:
    """Decile curve: alias score (x, deciles) vs mean per-frame Δ_oracle
    (y). Single series → no legend; the flagged decile is the orange
    point; zero and pooled-mean reference lines."""
    curve = analysis["decile_curve"]
    deltas = [c["delta_mean"] for c in curve]
    pooled = float(np.average(deltas, weights=[c["n"] for c in curve]))
    read = analysis["delta_oracle"]

    fig, ax = new_fig(8.6, 4.6)
    ax.axhline(0, color=DARKGRAY, linewidth=1, linestyle=":")
    ax.axhline(pooled, color=GRAY, linewidth=1, linestyle="--")
    ax.annotate(
        f"pooled mean {pooled:+.3f}",
        xy=(0.7, pooled),
        fontsize=8.5,
        color=DARKGRAY,
        va="bottom",
    )
    x = np.arange(1, 11)
    ax.plot(x, deltas, color=BLUE, linewidth=2, zorder=3)
    ax.scatter(x[:-1], deltas[:-1], s=42, color=BLUE, zorder=4)
    ax.scatter([x[-1]], [deltas[-1]], s=64, color=ORANGE, zorder=5)
    ax.annotate(
        f"flagged decile {deltas[-1]:+.3f}",
        xy=(x[-1], deltas[-1]),
        xytext=(-8, -16),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=DARKGRAY,
    )
    ax.annotate(
        f"flagged − rest {read['difference']:+.3f} "
        f"[CI95 {read['ci95'][0]:+.3f}, {read['ci95'][1]:+.3f}], "
        f"Spearman rho {read['spearman_rho']:+.2f}",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=9,
        color=DARKGRAY,
    )
    ax.set_xticks(x)
    ax.set_xlabel(
        "alias-score decile (within-dataset NN divergence, low → high)",
        fontsize=9.5,
        color=DARKGRAY,
    )
    ax.set_ylabel(
        "mean per-frame Δ_oracle (subgoal-conditioned − baseline)",
        fontsize=9.5,
        color=DARKGRAY,
    )
    ax.set_title(
        "Does the subgoal-conditioning gain concentrate on aliased frames?",
        fontsize=11,
        color="#0b0b0b",
        loc="left",
        pad=12,
    )
    save(fig, "concentration_deciles.svg")


def chart_score_distribution() -> None:
    """Alias-score distribution with the flag bar — how much of the
    panel is ambiguous at all (the census view)."""
    z = np.load(
        ROOT / "reports/framemining__ar100k_k4l2__flagged.npz",
        allow_pickle=True,
    )
    analysis = json.loads(
        (ROOT / "reports/analysis__framemining_ar100k_k4l2.json").read_text(),
    )
    scores = z["alias_score"][z["qualifying"]]
    bar = analysis["pool"]["alias_score_bar"]

    fig, ax = new_fig(8.6, 3.6)
    bins = np.linspace(0, np.quantile(scores, 0.995), 60)
    ax.hist(scores, bins=bins, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(bar, color=ORANGE, linewidth=1.6)
    ax.annotate(
        f"flag bar (top decile) = {bar:.2f}",
        xy=(bar, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=DARKGRAY,
    )
    ax.set_xlabel(
        "alias score (mean std-normalized chunk divergence to top-5 visual neighbors)",
        fontsize=9.5,
        color=DARKGRAY,
    )
    ax.set_ylabel("frames", fontsize=9.5, color=DARKGRAY)
    ax.set_title(
        "Aliasing census over the panel's qualifying core frames",
        fontsize=11,
        color="#0b0b0b",
        loc="left",
        pad=12,
    )
    save(fig, "alias_score_distribution.svg")


def main() -> None:
    analysis = json.loads(
        (ROOT / "reports/analysis__framemining_ar100k_k4l2.json").read_text(),
    )
    chart_concentration(analysis)
    chart_score_distribution()


if __name__ == "__main__":
    main()
