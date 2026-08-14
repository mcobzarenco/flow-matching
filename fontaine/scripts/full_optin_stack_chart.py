"""Results chart for the full opt-in stack read (queue
`sim-full-optin-stack-read`; eval-report dark scheme — the shipped
Carbon pair). Two panels: the paired dknn5 reads (PRIMARY stack vs
v3, the in-run patched replication, and the materials' marginal on
top of clutter with its banked no-interaction reference) with CI95
whiskers against the zero rule; and the AUROC ladder v3 -> patched ->
stack_full against the registered best-single bar and the additive
prediction.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
STACK, PATCHED, ANCHOR = "#08bdba", "#ffb000", "#9aa0a8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/analysis__sim_full_optin_stack_read.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/full_optin_stack/chart__full_optin_stack_read.png"),
    )
    args = ap.parse_args()
    results = json.loads(args.analysis.read_text())["results"]
    gate = results["registered_gate"]
    add = results["additivity"]

    fig, (dax, ax) = plt.subplots(1, 2, figsize=(13.2, 4.9), width_ratios=[1, 1.25])
    fig.patch.set_facecolor(PAGE)
    for panel in (dax, ax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
        panel.set_axisbelow(True)

    # left: paired dknn5 reads, CI95 whiskers, zero rule
    reads = [
        ("PRIMARY (registered)\nstack_full - v3", results["paired_stack_vs_v3"], STACK),
        (
            "in-run replication\npatched - v3",
            results["paired_patched_vs_v3"],
            PATCHED,
        ),
        (
            "materials marginal\nstack_full - patched",
            results["paired_stack_vs_patched"],
            STACK,
        ),
    ]
    spans = []
    for y, (label, read, color) in enumerate(reads):
        lo, hi = (v * 1e7 for v in read["ci95"])
        spans.extend((lo, hi))
        dax.plot([lo, hi], [y, y], color=color, lw=2.4)
        dax.plot([read["mean_delta"] * 1e7], [y], "o", color=color, ms=8)
        dax.text(
            (lo + hi) / 2,
            y + 0.16,
            f"{read['n_closer']}/100 slots closer",
            color=META,
            fontsize=8.5,
            ha="center",
        )
        dax.text(
            (lo + hi) / 2,
            y - 0.30,
            label,
            color=TEXT,
            fontsize=9.5,
            ha="center",
            va="center",
        )
    ref_lo, ref_hi = (v * 1e7 for v in add["material_marginal_ref_ci95_knn5"])
    dax.plot([ref_lo, ref_hi], [2.34, 2.34], color=ANCHOR, lw=1.4, ls=(0, (3, 2)))
    dax.text(
        (ref_lo + ref_hi) / 2,
        2.46,
        "banked material-stack CI (no-interaction ref)",
        color=META,
        fontsize=8,
        ha="center",
    )
    dax.axvline(0.0, color=TEXT, linewidth=1.0)
    lo, hi = min(spans), max(spans)
    pad = 0.12 * max(hi - lo, 1.0)
    dax.set_xlim(lo - pad, max(hi + pad, 1.2))
    dax.set_ylim(-0.55, 2.75)
    dax.set_yticks([])
    dax.set_xlabel(
        "paired Δknn5 vs real reference (×1e-7), CI95 10k resamples",
        color=META,
    )
    dax.set_title(
        "Paired reads — the full stack vs baseline, and its decomposition",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    # right: AUROC ladder vs the registered bar + additive prediction
    rows = [
        ("v3 default", results["arms"]["v3"]["auroc_vs_real"], ANCHOR),
        (
            "+ clutter patches\n(best single, in-run)",
            results["arms"]["patched"]["auroc_vs_real"],
            PATCHED,
        ),
        (
            "full opt-in stack\n(+ materials)",
            results["arms"]["stack_full"]["auroc_vs_real"],
            STACK,
        ),
    ]
    for y, (label, auroc, color) in enumerate(rows):
        ax.barh(y, auroc - 0.25, left=0.25, height=0.44, color=color, alpha=0.85)
        ax.text(
            auroc + 0.004,
            y,
            f"{auroc:.3f}",
            color=TEXT,
            fontsize=9.5,
            va="center",
        )
        ax.text(0.245, y, label, color=TEXT, fontsize=9.5, ha="right", va="center")
    ax.axvline(
        gate["best_single_bar"],
        color=TEXT,
        linewidth=1.0,
        ls=(0, (4, 2)),
    )
    ax.text(
        gate["best_single_bar"],
        2.62,
        f"registered bar {gate['best_single_bar']:.4f}",
        color=TEXT,
        fontsize=8.5,
        ha="center",
    )
    ax.axvline(add["additive_prediction"], color=PATCHED, linewidth=1.0, alpha=0.7)
    ax.text(
        add["additive_prediction"],
        -0.58,
        f"additive prediction {add['additive_prediction']:.3f}",
        color=PATCHED,
        fontsize=8.5,
        ha="center",
    )
    clean = results["clean_anchor"]["auroc_vs_real"]
    ax.axvline(clean, color=META, linewidth=0.8, alpha=0.7)
    ax.text(clean, 2.62, f"clean real {clean:.3f}", color=META, fontsize=8, ha="center")
    ax.set_xlim(0.25, 0.78)
    ax.set_ylim(-0.75, 2.85)
    ax.set_yticks([])
    ax.set_xlabel("knn5 AUROC vs held-out real (lower = reads more real)", color=META)
    ax.set_title(
        f"AUROC ladder — interaction term {add['interaction_auroc']:+.4f}",
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
