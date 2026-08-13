"""Results chart for the foreground appearance fix gate
(sim-foreground-appearance-pass legs (b)+(c); eval-report dark scheme,
hues CVD-checked on the dark surface, adjacent pairs dE 21/27). Two
panels: the AUROC ladder (v3 -> no_clutter -> patched against the
banked anchors and the registered bars) and the paired per-frame
dknn5 with bootstrap CI95 whiskers — real-crop clutter patches beating
the removal ceiling (100/100 slots closer than v3, 75/100 closer than
no_clutter) is the story. A separate v3-vs-patched frame strip carries
the qualitative read.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
ARM_COLOR = {"v3": META, "no_clutter": "#ffb000", "patched": "#08bdba"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/analysis__sim_fg_appearance_fix.json"),
    )
    ap.add_argument(
        "--frames",
        type=Path,
        default=Path("reports/assets/fg_fix_frames"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/fg_fix/chart__sim_fg_appearance_fix.png"),
    )
    ap.add_argument(
        "--strip-out",
        type=Path,
        default=Path("outputs/sim/fg_fix/fg_fix_v3_vs_patched_strip.png"),
    )
    args = ap.parse_args()
    payload = json.loads(args.analysis.read_text())
    results = payload["results"]
    arms = results["arms"]
    gate = results["registered_gate"]
    anchors = payload["config"]["leg_a_anchors"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(11.4, 4.6), width_ratios=[1.15, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)

    order = ("v3", "no_clutter", "patched")
    labels = ("v3\n(baseline)", "no_clutter\n(leg-a ceiling)", "patched\n(real crops)")
    x = np.arange(3, dtype=float)
    values = [arms[key]["auroc_vs_real"] for key in order]
    ax.bar(x, values, 0.56, color=[ARM_COLOR[k] for k in order], zorder=3)
    for xi, val in zip(x, values, strict=True):
        ax.text(xi, val + 0.012, f"{val:.3f}", color=TEXT, fontsize=9.5, ha="center")
    for level, style, label, dy in (
        (0.5, (GRID, "-"), "0.5 null", -0.012),
        (values[0] - 0.05, (META, "--"), "registered PASS bar (v3 −0.05)", 0.008),
        (0.596, ("#08bdba", ":"), None, -0.012),
        (
            anchors["real_fg"],
            (META, "-."),
            f"real-fg anchor {anchors['real_fg']}",
            0.008,
        ),
    ):
        color, ls = style
        ax.axhline(level, color=color, lw=1.0, ls=ls, zorder=2)
        if label is not None:
            ax.text(
                2.62,
                level + dy,
                label,
                color=color if color != GRID else META,
                fontsize=7.3,
                ha="right",
                va="bottom" if dy > 0 else "top",
                zorder=4,
            )
    ax.text(
        -0.48,
        0.596 - 0.012,
        "full-recovery read\n(no_clutter +0.02)",
        color="#08bdba",
        fontsize=7.3,
        ha="left",
        va="top",
        zorder=4,
    )
    delta = arms["patched"]["auroc_vs_real"] - arms["v3"]["auroc_vs_real"]
    ax.annotate(
        f"ΔAUROC {delta:+.3f}\ngap closed {gate['gap_closed_fraction']:.0%}",
        xy=(2, values[2]),
        xytext=(1.35, 0.86),
        color=ARM_COLOR["patched"],
        fontsize=9,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": ARM_COLOR["patched"], "lw": 0.8},
    )
    ax.set_xticks(x, labels, color=TEXT, fontsize=8.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("top 5-NN AUROC (arm vs held-out real)", color=TEXT, fontsize=9.5)
    ax.set_title(
        "AUROC ladder (20 seeds x 5 draws, shared physics/plate/noise)",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    pairs = (
        ("no_clutter\nvs v3", results["paired_vs_v3"]["no_clutter"], "no_clutter"),
        ("patched\nvs v3", results["paired_vs_v3"]["patched"], "patched"),
        (
            "patched vs\nno_clutter",
            results["paired_patched_vs_no_clutter"],
            "patched",
        ),
    )
    for xi, (_label, block, hue) in enumerate(pairs):
        mean, (lo, hi) = block["mean_delta"], block["ci95"]
        dax.errorbar(
            xi,
            mean * 1e6,
            yerr=[[(mean - lo) * 1e6], [(hi - mean) * 1e6]],
            fmt="o",
            color=ARM_COLOR[hue],
            ecolor=ARM_COLOR[hue],
            capsize=5,
            markersize=7,
            zorder=3,
        )
        dax.text(
            xi,
            mean * 1e6 - 0.22,
            f"{block['n_closer']}/{block['n']} closer",
            color=TEXT,
            fontsize=8.5,
            ha="center",
            va="top",
        )
    dax.axhline(0.0, color=GRID, lw=1.0, zorder=2)
    dax.set_xticks(range(3), [p[0] for p in pairs], color=TEXT, fontsize=8.6)
    dax.set_xlim(-0.6, 2.6)
    dax.set_ylabel("paired dknn5 (x1e-6), bootstrap CI95", color=TEXT, fontsize=9.5)
    dax.set_title(
        "Paired per-frame dknn5 (negative = closer to real)",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    fig.suptitle(
        "Foreground appearance fix: real-crop clutter patches beat the "
        "removal ceiling — registered gate PASS",
        color=TEXT,
        fontsize=11.5,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")

    # v3 | patched frame strip, slots with clutter drawn present
    from PIL import Image

    rows = []
    for slot in (1, 3):
        pair = [
            np.asarray(Image.open(args.frames / arm / f"{slot:04d}.png"))
            for arm in ("v3", "patched")
        ]
        rows.append(np.concatenate(pair, axis=1))
    strip = np.concatenate(
        [np.pad(r, ((0, 6), (0, 0), (0, 0))) for r in rows],
        axis=0,
    )
    args.strip_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(args.strip_out)
    print(f"wrote {args.strip_out}")


if __name__ == "__main__":
    main()
