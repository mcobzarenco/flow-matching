"""Chart for the #6 rung-(b') clean-list subgoal-draws results post —
rendered from the BANKED adjudication json
(`analysis__subgoal_draws_cleanlist_q4_ar100k_k4l2.json`), no
re-computation of any claimed number.

Output: fontaine/blog/src/img/fieldcond/subgoal_cleandraws_deltas.svg.

Palette: the eval reports' DARK theme (dark_background + the IBM
colorblind-safe pair #648fff/#ffb000 on page #121417; standing owner
rule 2026-08-08 16:32Z). No node on this host, so the palette validator
can't run; the standing pre-validated pair unchanged is the sanctioned
fallback. Polarity: blue = arm beats the baseline (delta below 0),
amber = arm loses; neutral zero line; arm identity is on the axis, so
identity is never color-alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/fieldcond"
ANALYSIS = ROOT / "reports/analysis__subgoal_draws_cleanlist_q4_ar100k_k4l2.json"

PAGE = "#121417"
BLUE = "#648fff"  # arm beats the baseline
AMBER = "#ffb000"  # arm loses to the baseline
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


def main() -> None:
    d = json.loads(ANALYSIS.read_text())
    assert d["candidate_filter"] == "clean" and d["subset_mode"] is True

    # The registered reads, top-to-bottom: oracle ceiling (labeled
    # subset, read 2), banked self (re-pooled comparator, no CI stored
    # in this json — drawn without a whisker), narrated pass 1, and the
    # primary SC-pick arm (core, read 1).
    rows = [
        ("oracle ceiling pick\n(Δ_ceil, labeled)", d["arms"]["ceil"]["labeled_subset"]),
        (
            "greedy self subgoal\n(banked rung-a arm)",
            {
                "delta_pooled": d["banked_self"]["delta_pooled"],
                "ci95": None,
            },
        ),
        ("narrated pass 1\n(free channel)", d["arms"]["narr"]["core"]),
        ("self-certainty pick\n(Δ_bon, primary)", d["arms"]["bon"]["core"]),
    ]
    floor = d["noise_floor_per_frame_ci"]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=110)
        fig.patch.set_facecolor(PAGE)
        ax.set_facecolor(PAGE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=META, labelsize=9)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

        ax.axvspan(-floor, floor, color=GRID, alpha=0.45, linewidth=0)
        ax.axvline(0, color=META, linewidth=1.0)

        for y, (_label, read) in enumerate(rows):
            delta = read["delta_pooled"]
            color = BLUE if delta < 0 else AMBER
            if read["ci95"] is not None:
                lo, hi = read["ci95"]
                ax.plot([lo, hi], [y, y], color=color, linewidth=2, zorder=3)
                for x in (lo, hi):
                    ax.plot(
                        [x, x],
                        [y - 0.09, y + 0.09],
                        color=color,
                        linewidth=2,
                        zorder=3,
                    )
            ax.plot(
                [delta],
                [y],
                "o",
                color=color,
                markersize=9,
                zorder=4,
                markeredgecolor=PAGE,
                markeredgewidth=2,
            )
            ci_txt = (
                f"  [{read['ci95'][0]:+.3f}, {read['ci95'][1]:+.3f}]"
                if read["ci95"] is not None
                else "  (banked, re-pooled)"
            )
            ax.text(
                delta,
                y + 0.22,
                f"{delta:+.3f}{ci_txt}",
                color=TEXT,
                fontsize=9.5,
                ha="center",
            )

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], color=TEXT, fontsize=9.5)
        ax.set_ylim(-0.6, len(rows) - 0.2)
        ax.set_xlabel(
            "Δ chunk MAE vs bare baseline, q4 subset (negative = arm wins; "
            f"grey band = ±{floor} decode-noise floor)",
            color=META,
            fontsize=9.5,
        )
        ax.set_title(
            "Subgoal-draws rung (b'): the width holds better subgoals — "
            "the scorer can't find them",
            color=TEXT,
            fontsize=11.5,
            pad=12,
            loc="left",
        )
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUT / "subgoal_cleandraws_deltas.svg",
            facecolor=PAGE,
            bbox_inches="tight",
        )
        # PNG proof for the eyeball pass (not committed).
        fig.savefig(
            "/tmp/subgoal_cleandraws_deltas.png",
            facecolor=PAGE,
            bbox_inches="tight",
        )
    print("wrote", OUT / "subgoal_cleandraws_deltas.svg")


if __name__ == "__main__":
    main()
