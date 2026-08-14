"""Results chart for the wrist content split read (queue
`sim-manip-wrist-content-split`; eval-report dark scheme — the shipped
Carbon pair). Reads the frozen analysis JSON only. Two panels: the
manipulation-pose wrist AUROC with the benchy present vs removed (the
content term the caveat asked about), next to the read's anchors; and
the paired ABSENT-PRESENT Δknn5 CI95 against the zero rule — all
slots, the 61 benchy-visible slots, and the 39 blind slots (the
shadow/bounce control) — with the banked pose-effect bar for scale.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
FIX, REFUTED, ANCHOR = "#08bdba", "#ffb000", "#9aa0a8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/wrist_content_split/chart__wrist_content_split.png"),
    )
    args = ap.parse_args()

    results = json.loads(
        (args.reports / "analysis__sim_wrist_content_split.json").read_text(),
    )["results"]
    manip = results["manip"]
    riders = results["riders"]
    calibration = results["abort_gates"]["calibration_auroc"]
    pose_effect = results["context"]["banked_pose_effect_delta"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(13.6, 5.0), width_ratios=[1.1, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.set_axisbelow(True)

    # left: manipulation-pose wrist AUROC, benchy present vs removed
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
    rows = [
        ("real-real calibration (eps 20-25, drift floor)", calibration, ANCHOR),
        (
            "RESET pose anchor (in-run, replicated)",
            results["reset_anchor"]["wrist"]["auroc_vs_real"],
            FIX,
        ),
        (
            "MANIP pose, benchy AT SPAWN (the banked arm)",
            manip["arms"]["present"]["auroc_vs_real"],
            REFUTED,
        ),
        (
            "MANIP pose, benchy REMOVED (clean table)",
            manip["arms"]["absent"]["auroc_vs_real"],
            REFUTED,
        ),
    ]
    for y, (label, auroc, color) in enumerate(reversed(rows)):
        ax.barh(y, auroc, height=0.5, color=color, alpha=0.85)
        ax.text(auroc + 0.008, y, f"{auroc:.3f}", color=TEXT, fontsize=9, va="center")
        ax.text(0.02, y + 0.38, label, color=TEXT, fontsize=8.5, va="center")
    ax.axvline(0.5, color=META, linewidth=0.8, alpha=0.5)
    ax.text(0.5, -0.62, "chance", color=META, fontsize=8, ha="center")
    ax.set_xlim(0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("er_60k knn5 AUROC vs held-out real (higher = more fake)", color=META)
    ax.set_title(
        "Deleting the benchy does NOT make the wrist view honest —\n"
        "the rendered arm carries the manipulation-pose gap",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    # right: paired ABSENT-PRESENT deltas vs the zero rule
    dax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    reads = [
        (
            "ALL 100 slots\n(PRIMARY:\nCI straddles 0)",
            results["manip"]["paired_absent_vs_present"],
        ),
        ("61 benchy-\nVISIBLE slots", riders["paired_delta_visible_slots"]),
        (
            "39 blind slots\n(shadow/bounce\ncontrol)",
            riders["paired_delta_blind_slots"],
        ),
    ]
    for x, (_label, read) in enumerate(reads):
        lo, hi = (v * 1e7 for v in read["ci95"])
        mean = read["mean_delta"] * 1e7
        dax.vlines(x, lo, hi, color=FIX, linewidth=2.5)
        for cap in (lo, hi):
            dax.hlines(cap, x - 0.06, x + 0.06, color=FIX, linewidth=2.5)
        dax.plot(
            [x],
            [mean],
            marker="o",
            markersize=8,
            color=FIX,
            markeredgecolor=PAGE,
            markeredgewidth=2,
        )
        dax.text(
            x + 0.12,
            mean,
            f"{read['mean_delta']:+.2e}\n{read['n_closer']}/{read['n']} closer",
            color=TEXT,
            fontsize=8.5,
            va="center",
        )
    dax.axhline(0.0, color=META, linewidth=0.9, alpha=0.8)
    scale = pose_effect * 1e7
    dax.axhline(scale, color=REFUTED, linewidth=0.9, linestyle="--", alpha=0.8)
    dax.text(
        len(reads) - 0.25,
        scale - 4.5,
        f"banked POSE effect +{pose_effect:.2e}\n(the term that carries the gap)",
        color=REFUTED,
        fontsize=8,
        ha="right",
        va="top",
    )
    dax.set_xticks(range(len(reads)))
    dax.set_xticklabels([r[0] for r in reads], color=TEXT, fontsize=8.5)
    dax.set_xlim(-0.5, len(reads) - 0.2)
    dax.set_ylabel("paired Δknn5, benchy ABSENT − PRESENT (×1e-7)", color=META)
    dax.set_title(
        "Content term ≈ NIL: removal moves nothing, at any visibility —\n"
        "and sits at 4% of the pose effect that does carry the gap",
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
