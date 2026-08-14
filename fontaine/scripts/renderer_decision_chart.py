"""Lead chart for the renderer-class decision brief (queue
`renderer-class-decision-brief`; eval-report dark scheme). Banked
numbers only, read from the frozen analysis JSONs in reports/. Two
panels: where each camera stands after the appearance screen (the
shipped/pending flips vs the anchors, with the upgrade-addressable
span marked per camera); and the three paired reads that price the
decision — the pose-switched arm term against the two ~20x-smaller
terms (content nil, material-stack regression) on one shared axis.

Anchor-mark gray is #6f747c (not the text gray #9aa0a8): the
teal/gray pair fails the OKLab CVD floor on the dark surface at the
text gray; #6f747c passes all pairs (checked 2026-08-14).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
FIX, REFUTED, ANCHOR = "#08bdba", "#ffb000", "#6f747c"


def _results(reports: Path, stem: str) -> dict:
    return json.loads((reports / f"analysis__{stem}.json").read_text())["results"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/sim/renderer_decision/chart__renderer_class_decision.png",
        ),
    )
    args = ap.parse_args()

    stack = _results(args.reports, "sim_full_optin_stack_read")
    decomp = _results(args.reports, "sim_top_gap_decomposition")["full_frame"]
    pose = _results(args.reports, "sim_rollout_pose_wrist_read")
    split = _results(args.reports, "sim_wrist_content_split")

    v3 = stack["arms"]["v3"]["auroc_vs_real"]
    stacked = stack["arms"]["stack_full"]["auroc_vs_real"]
    floor = decomp["arms"]["real_fg"]["auroc_vs_real"]
    clean = stack["clean_anchor"]["auroc_vs_real"]
    w_reset = pose["abort_gates"]["wrist_auroc"]
    w_manip = pose["primary1"]["manip_wrist_auroc"]
    w_absent = split["secondary"]["absent_manip_auroc"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(13.6, 5.4), width_ratios=[1.15, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
        panel.set_axisbelow(True)

    # left: where each camera stands; the addressable span per camera
    rows = [
        ("TOP (composited)", None, None),
        ("v3 default", v3, ANCHOR),
        ("full opt-in stack (banked flips)", stacked, FIX),
        ("real-fg composite (pipeline floor)", floor, ANCHOR),
        ("WRIST (raw render)", None, None),
        ("reset poses (fitted lens)", w_reset, ANCHOR),
        ("manipulation poses (pose-matched)", w_manip, FIX),
        ("benchy deleted (content control)", w_absent, ANCHOR),
    ]
    n = len(rows)
    for i, (label, auroc, color) in enumerate(rows):
        y = n - 1 - i
        if auroc is None:
            ax.text(
                0.245,
                y,
                label,
                color=TEXT,
                fontsize=10,
                va="center",
                ha="right",
                fontweight="bold",
            )
            continue
        ax.barh(y, auroc - 0.25, left=0.25, height=0.5, color=color, alpha=0.85)
        ax.text(auroc + 0.006, y, f"{auroc:.3f}", color=TEXT, fontsize=9, va="center")
        ax.text(0.245, y, label, color=TEXT, fontsize=9, ha="right", va="center")
    # addressable spans: what a renderer-class upgrade could reach
    for y0, hi, lo, note, side in (
        (
            4.5,
            stacked,
            floor,
            f"addressable −{stacked - floor:.3f} (floor measured)",
            "right",
        ),
        (
            1.5,
            w_manip,
            w_reset,
            f"addressable −{w_manip - w_reset:.3f} (ceiling unmeasured)",
            "left",
        ),
    ):
        ax.plot([lo, hi], [y0, y0], color=FIX, lw=1.4, ls=(0, (3, 2)))
        for x in (lo, hi):
            ax.plot([x, x], [y0 - 0.12, y0 + 0.12], color=FIX, lw=1.4)
        if side == "right":
            ax.text(hi + 0.012, y0, note, color=FIX, fontsize=8, ha="left", va="center")
        else:
            ax.text(
                lo - 0.012,
                y0,
                note,
                color=FIX,
                fontsize=8,
                ha="right",
                va="center",
            )
    ax.axvline(clean, color=META, linewidth=0.8, alpha=0.7)
    ax.text(
        clean,
        n - 0.4,
        f"clean real {clean:.3f}",
        color=META,
        fontsize=8,
        ha="center",
    )
    ax.axvline(0.5, color=TEXT, linewidth=1.0, ls=(0, (4, 2)), alpha=0.8)
    ax.text(0.5, n - 0.4, "0.5 null", color=TEXT, fontsize=8, ha="center")
    ax.set_xlim(0.25, 0.97)
    ax.set_ylim(-1.5, n + 0.1)
    ax.set_yticks([])
    ax.set_xlabel("knn5 AUROC vs held-out real (lower = reads more real)", color=META)
    ax.set_title(
        "Where each camera stands — and what a renderer-class fix addresses",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    # right: the three paired reads that price the decision, one axis
    reads = [
        (
            "pose effect: manip − reset\n(the arm term the upgrade owns)",
            pose["manip"]["pose_effect_manip_vs_reset_default"],
            FIX,
        ),
        (
            "material stack at manip poses\n(classic-renderer regression)",
            pose["manip"]["paired_stack_vs_default"],
            REFUTED,
        ),
        (
            "benchy removal at manip poses\n(content term: nil)",
            split["manip"]["paired_absent_vs_present"],
            ANCHOR,
        ),
    ]
    for y, (label, read, color) in enumerate(reversed(reads)):
        lo, hi = (v * 1e7 for v in read["ci95"])
        dax.plot([lo, hi], [y, y], color=color, lw=2.4)
        dax.plot([read["mean_delta"] * 1e7], [y], "o", color=color, ms=8)
        dax.text(
            hi + 2.2,
            y,
            f"{read['mean_delta'] * 1e7:+.1f}  ({read['n_closer']}/100 closer)",
            color=META,
            fontsize=8.5,
            va="center",
        )
        dax.text(-7.5, y + 0.3, label, color=TEXT, fontsize=9.2, ha="left")
    dax.axvline(0.0, color=TEXT, linewidth=1.0)
    dax.text(0.0, len(reads) - 0.28, "zero", color=TEXT, fontsize=8, ha="center")
    dax.set_xlim(-8.0, 132.0)
    dax.set_ylim(-0.55, len(reads))
    dax.set_yticks([])
    dax.set_xlabel(
        "paired Δknn5 at manipulation poses (×1e-7), CI95 10k resamples — "
        "right of zero = reads more fake",
        color=META,
    )
    dax.set_title(
        "The three facts that price it — one shared axis",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
