"""Per-seed chart for the 100-episode arm-A eval at the 30 s budget
(pre-reg amendment 3): progress_final per seed colored by outcome, plus
the success-tick strip showing every success landed PAST tick 450 — the
old 15 s budget's cutoff (eval-report dark scheme, IBM CVD-safe hues)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("outputs/sim/ftrig_eval20_flip_parallel/release_officialmap_a_100ep_30s")
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
QUIET, KNOCK, WIN = "#648fff", "#dc267f", "#ffb000"


def main() -> None:
    episodes = json.loads((BASE / "rows.json").read_text())["episodes"]
    fig, (ax, ticks_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 5.6),
        height_ratios=[3.2, 1.0],
        sharex=False,
    )
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, ticks_ax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.set_axisbelow(True)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.axhline(0, color=META, linewidth=1, linestyle=":")
    for row in episodes:
        seed = row["seed"]
        pf = row["initial_cm"] - row["final_cm"]
        if row["success_tick"] is not None:
            color, marker, size = WIN, "*", 130
        elif pf <= -1.0:
            color, marker, size = KNOCK, "v", 34
        else:
            color, marker, size = QUIET, "o", 22
        ax.scatter(
            seed,
            pf,
            s=size,
            color=color,
            marker=marker,
            zorder=3,
            edgecolors=PAGE,
            linewidths=0.8,
        )
    ax.set_title(
        "Released MolmoAct2, official map, 30 s budget — 9/100 SUCCESSES "
        "(first sim successes on this task; every prior eval read 0)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax.set_ylabel("progress_final (cm)", color=META, fontsize=10)
    ax.set_xlabel("seed", color=META, fontsize=10)
    handles = [
        Line2D(
            [],
            [],
            color=WIN,
            marker="*",
            linestyle="",
            markersize=12,
            label="success (boat resting on the disk): 9",
        ),
        Line2D(
            [],
            [],
            color=KNOCK,
            marker="v",
            linestyle="",
            markersize=7,
            label="knock-away (≤ −1 cm): 27",
        ),
        Line2D(
            [],
            [],
            color=QUIET,
            marker="o",
            linestyle="",
            markersize=6,
            label="no pickup: 64",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=8,
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
    )
    legend.set_zorder(4)

    wins = [r for r in episodes if r["success_tick"] is not None]
    ticks_ax.scatter(
        [r["success_tick"] for r in wins],
        [0] * len(wins),
        s=120,
        color=WIN,
        marker="*",
        zorder=3,
        edgecolors=PAGE,
        linewidths=0.8,
    )
    ticks_ax.axvline(450, color=KNOCK, linewidth=1.4, linestyle="--")
    ticks_ax.annotate(
        "tick 450 — where the old 15 s budget ended:\nevery success lands past it",
        (450, 0),
        textcoords="offset points",
        xytext=(-232, 6),
        color=META,
        fontsize=8,
    )
    ticks_ax.set_xlim(0, 900)
    ticks_ax.set_yticks([])
    ticks_ax.set_xlabel(
        "success tick (30 Hz; episode = 900 ticks = 30 s)",
        color=META,
        fontsize=10,
    )

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            BASE / f"chart__release_100ep_30s.{suffix}",
            bbox_inches="tight",
            facecolor=PAGE,
            dpi=160,
        )
    print(f"wrote {BASE}/chart__release_100ep_30s.{{png,svg}}")


if __name__ == "__main__":
    main()
