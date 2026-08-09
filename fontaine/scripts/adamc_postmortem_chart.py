"""Chart for the AdamC-100k post-mortem post (owner kill 22:40Z 08-09).

Two panels, shared y: the in-run 256-frame probe (eval_chunk_mae) of
the AdamC run vs the 40k AR baseline, (left) at matched training
steps and (right) at matched samples seen — the runs differ in
effective batch (32 vs 48), so the samples view is the fairer one.
Values transcribed from each run's train_log.jsonl (box); eval-report
dark theme (page #121417), amber = AdamC (killed), blue = 40k
baseline. Identity carried by direct labels, never color alone.

Usage: uv run python fontaine/scripts/adamc_postmortem_chart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/adamc_postmortem"

PAGE = "#121417"
BLUE = "#648fff"  # 40k AR baseline (AdamW, vision frozen, eff-48)
AMBER = "#ffb000"  # AdamC 100k (vision unfrozen, eff-32; owner-killed)
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

# eval_chunk_mae at each 500-step eval, from train_log.jsonl.
ADAMC = [
    (500, 31.2959),
    (1000, 24.4834),
    (1500, 16.8716),
    (2000, 15.3931),
    (2500, 14.0294),
    (3000, 13.5823),
    (3500, 12.0657),
    (4000, 11.402),
    (4500, 11.3236),
    (5000, 12.646),
    (5500, 12.1188),
    (6000, 12.591),
    (6500, 12.6027),
    (7000, 11.6945),
    (7500, 11.7238),
    (8000, 11.0237),
    (8500, 11.4422),
    (9000, 11.5313),
    (9500, 10.6338),
    (10000, 10.7968),
    (10500, 11.0576),
    (11000, 11.4129),
    (11500, 10.2954),
]
AR40K = [
    (500, 30.844),
    (1000, 25.7188),
    (1500, 15.2531),
    (2000, 13.206),
    (2500, 12.0944),
    (3000, 12.5951),
    (3500, 10.4878),
    (4000, 10.4737),
    (4500, 9.462),
    (5000, 9.6394),
    (5500, 9.2401),
    (6000, 8.5413),
    (6500, 8.9431),
    (7000, 8.7838),
    (7500, 8.6356),
    (8000, 8.6371),
    (8500, 7.6695),
    (9000, 8.2554),
    (9500, 7.3889),
    (10000, 7.1652),
    (10500, 7.1514),
    (11000, 7.9665),
    (11500, 7.2014),
    (12000, 7.5549),
    (12500, 7.8968),
    (13000, 7.092),
]
ADAMC_EFF_BATCH = 32
AR40K_EFF_BATCH = 48
KILL_STEP = 11840  # owner kill 22:40Z 08-09 ("not looking great")


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.6),
        dpi=160,
        sharey=True,
    )
    fig.patch.set_facecolor(PAGE)

    ax, ay = zip(*ADAMC, strict=True)
    bx, by = zip(*AR40K, strict=True)

    # Left: matched training steps.
    ax1.plot(bx, by, color=BLUE, linewidth=2, zorder=4)
    ax1.plot(ax, ay, color=AMBER, linewidth=2, zorder=4)
    ax1.axvline(
        KILL_STEP,
        color=AMBER,
        linestyle=(0, (2, 3)),
        linewidth=1.2,
        zorder=2,
    )
    ax1.text(
        KILL_STEP - 200,
        26.5,
        "killed ~11,840\n(owner cost call)",
        color=AMBER,
        fontsize=8.5,
        ha="right",
        va="top",
    )
    ax1.text(
        7300,
        12.6,
        "AdamC 100k",
        color=AMBER,
        fontsize=9.5,
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    ax1.text(
        8600,
        6.0,
        "40k AR baseline",
        color=BLUE,
        fontsize=9.5,
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    ax1.annotate(
        "10.80 vs 7.17 @10k",
        xy=(10000, 10.7968),
        xytext=(4600, 17.6),
        color=TEXT,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.8},
    )
    ax1.set_xlabel("training step", color=META, fontsize=9)
    ax1.set_ylabel(
        "in-run probe chunk MAE (256 held-out frames)",
        color=META,
        fontsize=9,
    )
    ax1.set_xlim(300, 13300)
    ax1.set_title("matched steps", color=META, fontsize=9.5, loc="left")

    # Right: matched samples seen (the fairer axis: eff-32 vs eff-48).
    ax2.plot(
        [s * AR40K_EFF_BATCH / 1000 for s in bx],
        by,
        color=BLUE,
        linewidth=2,
        zorder=4,
    )
    ax2.plot(
        [s * ADAMC_EFF_BATCH / 1000 for s in ax],
        ay,
        color=AMBER,
        linewidth=2,
        zorder=4,
    )
    ax2.axvline(
        KILL_STEP * ADAMC_EFF_BATCH / 1000,
        color=AMBER,
        linestyle=(0, (2, 3)),
        linewidth=1.2,
        zorder=2,
    )
    ax2.annotate(
        "run-best 10.30 vs ~8.6\nat the same 368k samples",
        xy=(11500 * ADAMC_EFF_BATCH / 1000, 10.2954),
        xytext=(232, 15.6),
        color=TEXT,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.8},
    )
    ax2.set_xlabel(
        "samples seen (thousands; eff-batch 32 vs 48)",
        color=META,
        fontsize=9,
    )
    ax2.set_xlim(10, 640)
    ax2.set_title("matched samples", color=META, fontsize=9.5, loc="left")

    for axis in (ax1, ax2):
        axis.set_facecolor(PAGE)
        axis.tick_params(colors=META, labelsize=9)
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax1.set_ylim(5.4, 32.5)

    fig.suptitle(
        "AdamC-100k post-mortem: the probe plateaued ~10.3–11.4 from step "
        "4k while the 40k baseline kept descending — behind under both "
        "matched views",
        color=TEXT,
        fontsize=10.5,
        x=0.123,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "adamc_probe_views.svg", facecolor=PAGE, bbox_inches="tight")
    fig.savefig(OUT / "adamc_probe_views.png", facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {OUT}/adamc_probe_views.{{svg,png}}")


if __name__ == "__main__":
    main()
