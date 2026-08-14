"""Results chart for the camera-mount material split (queue
`sim-mount-material-split`; eval-report dark scheme, hues follow the
arm-split entities — the mount keeps Carbon purple-40, whole-frame v3
the teal). Two panels: the AUROC ladder as before->after dumbbells
against the pinned anchors (incl. the no_mount amputation best — the
registered rider: does painting it right beat cutting it off?), and the
registered paired dknn5 reads with bootstrap CI95 whiskers plus the
record-only two-flag stack."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
MOUNT, FRAME, STACK, ANCHOR = "#a56eff", "#08bdba", "#ffb000", "#9aa0a8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/analysis__sim_mount_material_read.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/mount_material/chart__mount_material_read.png"),
    )
    args = ap.parse_args()
    results = json.loads(args.analysis.read_text())["results"]
    arms = {name: read["auroc_vs_real"] for name, read in results["arms"].items()}
    clean = results["clean_anchor"]["auroc_vs_real"]
    context = results["context"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(13.2, 5.0), width_ratios=[1.5, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
        panel.set_axisbelow(True)

    # left: before -> after dumbbells on the AUROC axis, anchors as rules
    rows = [
        ("whole frame\nv3 -> v3_mount", arms["v3"], arms["v3_mount"], FRAME),
        (
            "mount on plate\nonly_mount -> only_mount_v1",
            arms["only_mount"],
            arms["only_mount_v1"],
            MOUNT,
        ),
        (
            "two-flag stack\nv3 -> v3_full_fix",
            arms["v3"],
            arms["v3_full_fix"],
            STACK,
        ),
    ]
    for y, (label, before, after, color) in enumerate(rows):
        ax.annotate(
            "",
            xy=(after, y),
            xytext=(before, y),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2},
        )
        ax.plot([before], [y], "o", color=META, ms=7, zorder=3)
        ax.plot([after], [y], "o", color=color, ms=8, zorder=3)
        ax.text(
            before + 0.004,
            y + 0.16,
            f"{before:.3f}",
            color=META,
            fontsize=9,
            ha="left",
        )
        ax.text(
            after - 0.004,
            y + 0.16,
            f"{after:.3f}",
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="right",
        )
        ax.text(0.24, y - 0.02, label, color=TEXT, fontsize=9.5, va="center")
    for x, name, height in (
        (arms["plate_only"], "plate_only 0.866", 2.62),
        (context["arm_split_no_mount_removal_best"], "no_mount removal 0.654", 2.62),
        (context["decomposition_real_fg"], "real_fg 0.328", 2.62),
        (clean, f"clean {clean:.3f}", 2.40),
    ):
        ax.axvline(x, color=ANCHOR, linewidth=0.8, linestyle=":", alpha=0.8)
        ax.text(x, height, name, color=META, fontsize=8, ha="center", va="bottom")
    ax.set_xlim(0.22, 0.92)
    ax.set_ylim(-0.5, 2.9)
    ax.set_yticks([])
    ax.set_xlabel("knn5 AUROC vs held-out real (lower = closer to real)", color=META)
    ax.set_title(
        "The measured white-bracket material, on the AUROC ladder",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    # right: the registered paired reads + the record-only stack,
    # CI95 whiskers, zero rule
    reads = [
        ("PRIMARY\nv3_mount - v3", results["primary_v3_mount_vs_v3"], FRAME),
        (
            "MECHANISM\nonly_mount_v1 - only_mount",
            results["mechanism_only_mount_v1_vs_only_mount"],
            MOUNT,
        ),
        (
            "record-only\nv3_full_fix - v3",
            results["rider_v3_full_fix"]["vs_v3"],
            STACK,
        ),
    ]
    lows = []
    for y, (label, read, color) in enumerate(reads):
        lo, hi = read["ci95"]
        lows.append(lo)
        dax.plot([lo * 1e7, hi * 1e7], [y, y], color=color, lw=2.4)
        dax.plot([read["mean_delta"] * 1e7], [y], "o", color=color, ms=8)
        dax.text(
            (lo + hi) / 2 * 1e7,
            y + 0.18,
            f"{read['n_closer']}/100 slots closer",
            color=META,
            fontsize=8.5,
            ha="center",
        )
        dax.text(
            (lo + hi) / 2 * 1e7,
            y - 0.34,
            label,
            color=TEXT,
            fontsize=9.5,
            ha="center",
            va="center",
        )
    dax.axvline(0.0, color=TEXT, linewidth=1.0)
    dax.text(-0.25, 2.55, "0 = no change", color=META, fontsize=8.5, ha="right")
    dax.set_xlim(min(lows) * 1e7 * 1.25, 1.2)
    dax.set_ylim(-0.5, 2.9)
    dax.set_yticks([])
    dax.set_xlabel(
        "paired Δknn5 vs real reference (×1e-7), CI95 10k resamples",
        color=META,
    )
    dax.set_title(
        "Registered paired reads",
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
