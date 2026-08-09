"""Charts for the #6 rung-(c) post-mortem addendum (record-only).

Reads the banked post-mortem json + raw sidecar npz and renders two
dark-mode SVGs in the eval-report scheme (page #121417, the
colorblind-safe #648fff/#ffb000 pair — pair re-validated on this
surface: normal dE 36.9, deutan 39.8, protan 35.6, contrast 6.1/10.1):

  1. oracle-best rank histogram on both scorer axes vs the uniform
     null (the "where do the good candidates sit" read);
  2. per-row Spearman(score, error) distributions for both axes.

Color follows the axis identity everywhere: KL = blue, SC = amber.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
JSON_IN = REPO / "reports/analysis__subgoal_mcselect_postmortem_q4_ar100k_k4l2.json"
RAW_IN = REPO / "reports/analysis__subgoal_mcselect_postmortem_q4_ar100k_k4l2_raw.npz"
ASSETS = REPO / "fontaine/blog/src/assets"

PAGE = "#121417"
BLUE = "#648fff"  # the KL (informativeness) axis
AMBER = "#ffb000"  # the SC (self-certainty) axis
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=META, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(width: float = 8.6, height: float = 4.6) -> tuple:
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(PAGE)
    style_axes(ax)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(
            ASSETS / f"{name}.{ext}",
            format=ext,
            bbox_inches="tight",
            facecolor=PAGE,
        )
    plt.close(fig)
    print(f"wrote {ASSETS / name}.svg (+.png)")


def rank_hist(res: dict, raw: dict) -> None:
    kl = np.array(res["oracle_best_on_kl_axis"]["histogram_rank0_is_axis_top"])
    sc = np.array(res["oracle_best_on_sc_axis"]["histogram_rank0_is_axis_top"])
    ne = raw["n_eligs"]
    width = len(kl)
    # uniform-pick null: a rank r bin only exists for rows with more
    # than r eligible candidates, each contributing 1/n_elig
    null = np.array([(1.0 / ne[ne > r]).sum() for r in range(width)])
    x = np.arange(width)
    fig, ax = new_fig()
    ax.bar(x - 0.2, kl, width=0.36, color=BLUE, label="KL axis (MC)", zorder=3)
    ax.bar(x + 0.2, sc, width=0.36, color=AMBER, label="SC axis", zorder=3)
    ax.plot(
        x,
        null,
        color=META,
        linestyle="--",
        linewidth=1.4,
        marker="o",
        markersize=4,
        label="uniform-pick null",
        zorder=4,
    )
    ax.annotate(
        f"SC rank-0 excess: {sc[0]} rows vs {null[0]:.0f} expected\n"
        "(a real but weak signal)",
        xy=(0, sc[0]),
        xytext=(1.0, sc[0] * 0.97),
        color=TEXT,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.8},
    )
    ax.annotate(
        f"KL: flat — mean normalized rank "
        f"{res['oracle_best_on_kl_axis']['mean_normalized_rank']:.3f} "
        "(uniform = 0.5)",
        xy=(3.4, max(kl[2:]) * 1.25),
        color=TEXT,
        fontsize=9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in x])
    ax.set_xlabel(
        "oracle-best candidate's rank on the scorer axis "
        "(0 = the candidate the argmax picks)",
        color=META,
        fontsize=10,
    )
    ax.set_ylabel("rows (of 4,301)", color=META, fontsize=10)
    ax.set_title(
        "Where the genuinely-best candidate sits on each closed scorer axis",
        color=TEXT,
        fontsize=12,
        loc="left",
        pad=12,
    )
    leg = ax.legend(
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
        loc="upper right",
    )
    leg.set_zorder(5)
    save(fig, "mcselect-postmortem-rank-hist")


def spearman_dist(res: dict, raw: dict) -> None:
    bins = np.linspace(-1, 1, 21)
    fig, ax = new_fig()
    for arr, color, label in (
        (raw["rho_kl_err"], BLUE, "Spearman(KL, error)"),
        (raw["rho_sc_err"], AMBER, "Spearman(SC, error)"),
    ):
        ax.hist(
            arr,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=label,
            zorder=3,
        )
        ax.hist(arr, bins=bins, color=color, alpha=0.18, zorder=2)
    ax.axvline(0, color=META, linewidth=1.2, linestyle=":")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
    kve = res["kl_vs_error"]["per_row_spearman"]
    sve = res["sc_vs_error"]["per_row_spearman"]
    backing = {"facecolor": PAGE, "edgecolor": "none", "alpha": 0.9, "pad": 1.5}
    ax.annotate(
        f"KL mean {kve['mean']:+.3f}  CI [{kve['ci95'][0]:+.3f}, "
        f"{kve['ci95'][1]:+.3f}] — indistinguishable from noise",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        color=BLUE,
        fontsize=9.5,
        bbox=backing,
    )
    ax.annotate(
        f"SC mean {sve['mean']:+.3f}  CI [{sve['ci95'][0]:+.3f}, "
        f"{sve['ci95'][1]:+.3f}] — real, right-signed, tiny",
        xy=(0.02, 0.89),
        xycoords="axes fraction",
        color=AMBER,
        fontsize=9.5,
        bbox=backing,
    )
    ax.annotate(
        "negative = scorer prefers better candidates",
        xy=(0.02, 0.83),
        xycoords="axes fraction",
        color=META,
        fontsize=9,
        bbox=backing,
    )
    ax.set_xlabel(
        "per-row Spearman rank correlation, scorer value vs frame MAE "
        "(eligible candidates only)",
        color=META,
        fontsize=10,
    )
    ax.set_ylabel("rows", color=META, fontsize=10)
    ax.set_title(
        "Within-row rank signal of the two closed zero-training axes",
        color=TEXT,
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.legend(
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
        loc="upper right",
    )
    save(fig, "mcselect-postmortem-spearman-dist")


def main() -> None:
    res = json.loads(JSON_IN.read_text())
    raw = dict(np.load(RAW_IN))
    rank_hist(res, raw)
    spearman_dist(res, raw)


if __name__ == "__main__":
    main()
