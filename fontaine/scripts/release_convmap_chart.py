"""Per-seed progress_final chart for the release-in-sim convmap read:
the released checkpoint's flat zero line vs the two ftrig arms' scatter,
same 20 seeds, fixed post-flip sim (eval-report dark scheme, IBM
CVD-safe hues + per-arm marker shapes as secondary encoding)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("outputs/sim/ftrig_eval20_flip_parallel")
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
ARMS = [
    ("release_convmap", "release _convmap (off-contract)", "#648fff", "o"),
    ("postflip_v2", "ftrig step-2000", "#ffb000", "D"),
    ("step500", "ftrig step-500", "#dc267f", "^"),
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

    offsets = {"release_convmap": -0.22, "postflip_v2": 0.0, "step500": 0.22}
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
    # Selective direct labels: the day's extremes, not every point.
    worst_seed = min(rows["postflip_v2"], key=rows["postflip_v2"].get)
    best_seed = max(rows["step500"], key=rows["step500"].get)
    ax.annotate(
        f"step-2000 s{worst_seed} {rows['postflip_v2'][worst_seed]:+.2f} (knock-away)",
        (worst_seed, rows["postflip_v2"][worst_seed]),
        textcoords="offset points",
        xytext=(-115, -12),
        color=META,
        fontsize=8,
    )
    ax.annotate(
        f"step-500 s{best_seed} {rows['step500'][best_seed]:+.2f} (best approach)",
        (best_seed + offsets["step500"], rows["step500"][best_seed]),
        textcoords="offset points",
        xytext=(8, 2),
        color=META,
        fontsize=8,
    )
    ax.set_title(
        "Released MolmoAct2 in sim (unit shim): inert zero on all 20 seeds",
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
        loc="upper right",
        fontsize=8,
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
    )
    legend.set_zorder(4)
    out = BASE / "release_convmap"
    for suffix in ("png", "svg"):
        fig.savefig(
            out / f"chart__release_convmap_per_seed.{suffix}",
            bbox_inches="tight",
            facecolor=PAGE,
            dpi=160,
        )
    print(f"wrote {out}/chart__release_convmap_per_seed.{{png,svg}}")


if __name__ == "__main__":
    main()
