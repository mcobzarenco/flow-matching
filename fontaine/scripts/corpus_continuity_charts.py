"""Charts for the corpus kinematic-continuity screen results post.

Reads reports/analysis__corpus_continuity_screen.json (banked by
corpus_continuity_screen.py) and writes two dark-mode SVG+PNG pairs to
fontaine/blog/src/img/continuity/:

  continuity_ratio_hist:  log-x histogram of per-episode worst-tick
      displacement ratios, with the two scoring knees (r=1 full-marks
      bar, r=9 exponential knee) and the rig anchors' own maximum.
  continuity_repo_bars:   EXP-tail episode fraction for the repos with
      any tail hits (>= 3 scored episodes).

Usage:
  uv run python fontaine/scripts/corpus_continuity_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/analysis__corpus_continuity_screen.json"
OUT = ROOT / "fontaine/blog/src/img/continuity"

PAGE = "#121417"
BLUE = "#648fff"
AMBER = "#ffb000"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=META, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}/{name}.{{svg,png}}")


def ratio_hist(report: dict) -> None:
    ratios = np.asarray(report["_all_max_ratios"])
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=100)
    fig.patch.set_facecolor(PAGE)
    style(ax)
    bins = np.logspace(np.log10(0.01), np.log10(80), 70)
    ax.hist(ratios, bins=bins, color=BLUE, edgecolor=PAGE, linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    rig_max = report["R1_distribution"]["rig_anchors"]["max_ratio_max"]
    for x, label, color in (
        (1.0, "r = 1 (rig p99.9 bar)", META),
        (rig_max, f"rig anchors max ({rig_max:.2f})", TEXT),
        (9.0, "r = 9 (exponential knee)", AMBER),
    ):
        ax.axvline(x, color=color, linewidth=1.4, linestyle="--", zorder=4)
        ax.text(
            x * 1.06,
            ax.get_ylim()[1] * 0.55,
            label,
            color=color,
            fontsize=8.5,
            rotation=90,
            va="top",
        )
    ax.set_xlabel(
        "per-episode worst tick: max joint displacement / rig p99.9 bar (log)",
        color=META,
        fontsize=9,
    )
    ax.set_ylabel("episodes (log)", color=META, fontsize=9)
    n_tail = report["R2_exp_tail"]["n_tail_episodes"]
    n = report["R1_distribution"]["corpus"]["n_episodes"]
    ax.set_title(
        f"Worst per-tick action jump per episode — {n_tail} of {n:,} community "
        "episodes past the teleport knee; rig data never exceeds 1.7",
        color=TEXT,
        fontsize=10.5,
        loc="left",
        pad=12,
    )
    save(fig, "continuity_ratio_hist")


def repo_bars(report: dict) -> None:
    rows = [r for r in report["R3_repo_ranking"]["top15"] if r["n_exp"] > 0]
    rows = sorted(rows, key=lambda r: r["exp_fraction"])
    labels = [f"{r['repo']}  ({r['n_exp']}/{r['n_episodes']})" for r in rows]
    fracs = [100 * r["exp_fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.6), dpi=100)
    fig.patch.set_facecolor(PAGE)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    y = np.arange(len(rows))
    ax.barh(y, fracs, height=0.62, color=BLUE, zorder=3)
    for yi, frac in zip(y, fracs, strict=True):
        ax.text(
            min(frac + 1.5, 97),
            yi,
            f"{frac:.0f}%",
            color=TEXT,
            fontsize=8.5,
            va="center",
        )
    ax.set_yticks(y, labels, fontsize=8.5, color=TEXT)
    ax.set_xlim(0, 104)
    ax.set_xlabel("episodes past the teleport knee (%)", color=META, fontsize=9)
    ax.set_title(
        "Teleport-class episodes concentrate in two structurally non-conforming repos",
        color=TEXT,
        fontsize=10.5,
        loc="left",
        pad=12,
    )
    save(fig, "continuity_repo_bars")


def main() -> None:
    report = json.loads(REPORT.read_text())
    # The banked json keeps only the tail's per-episode rows; the histogram
    # needs every episode's max ratio, so recover them from the quantile
    # summary if a raw dump is absent — or better, from the sidecar npz.
    sidecar = REPORT.with_suffix(".max_ratios.npy")
    if not sidecar.exists():
        raise SystemExit(
            f"{sidecar} missing — re-run corpus_continuity_screen.py "
            "(it writes the sidecar alongside the json)",
        )
    report["_all_max_ratios"] = np.load(sidecar).tolist()
    ratio_hist(report)
    repo_bars(report)


if __name__ == "__main__":
    main()
