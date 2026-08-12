"""Per-seed progress_final chart for the official-map rerun (pre-reg
amendment 1): the parent fitted-map read's flat zero vs the two
official-map arms' scatter, same 20 seeds, fixed post-flip sim
(eval-report dark scheme, IBM CVD-safe hues + per-arm marker shapes as
secondary encoding)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("outputs/sim/ftrig_eval20_flip_parallel")
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
ARMS = [
    ("release_convmap", "fitted map, lift +180 (parent: inert)", "#648fff", "o"),
    ("release_officialmap_a", "official map, arm A (wrist identity)", "#ffb000", "D"),
    ("release_officialmap_b", "official map, arm B (wrist −90)", "#dc267f", "^"),
]


def main() -> None:
    rows = {
        arm: {
            e["seed"]: e["progress_final_cm"]
            for e in json.loads((BASE / arm / "rows.json").read_text())["episodes"]
        }
        for arm, *_ in ARMS
    }
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    for side in ax.spines.values():
        side.set_color(GRID)
    ax.tick_params(colors=META, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=META, linewidth=1, linestyle=":")

    offsets = {
        "release_convmap": -0.22,
        "release_officialmap_a": 0.0,
        "release_officialmap_b": 0.22,
    }
    for arm, label, color, marker in ARMS:
        seeds = sorted(rows[arm])
        ax.scatter(
            [s + offsets[arm] for s in seeds],
            [rows[arm][s] for s in seeds],
            s=46,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
            edgecolors=PAGE,
            linewidths=1.2,
        )
    # Selective direct labels: the engagement the lift sign unlocked.
    a = rows["release_officialmap_a"]
    ax.annotate(
        f"arm A s6 {a[6]:+.2f} (reached 1.4 cm from the disk)",
        (6, a[6]),
        textcoords="offset points",
        xytext=(8, 2),
        color=META,
        fontsize=8,
    )
    ax.annotate(
        f"arm A s16 {a[16]:+.2f} (knock-away — the boat was TOUCHED)",
        (16, a[16]),
        textcoords="offset points",
        xytext=(-150, -12),
        color=META,
        fontsize=8,
    )
    ax.set_title(
        "Released MolmoAct2 in sim, official v3.0→v2.1 map: "
        "engages the scene on a few seeds, still no pickups",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax.set_xlabel(
        "seed (sim100 list, same spawns across arms)",
        color=META,
        fontsize=10,
    )
    ax.set_ylabel("progress_final (cm)", color=META, fontsize=10)
    ax.set_xticks(range(0, 20, 2))
    legend = ax.legend(
        loc="lower left",
        fontsize=8,
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
    )
    legend.set_zorder(4)
    out = BASE / "release_officialmap_a"
    for suffix in ("png", "svg"):
        fig.savefig(
            out / f"chart__release_officialmap_per_seed.{suffix}",
            bbox_inches="tight",
            facecolor=PAGE,
            dpi=160,
        )
    print(f"wrote {out}/chart__release_officialmap_per_seed.{{png,svg}}")


if __name__ == "__main__":
    main()
