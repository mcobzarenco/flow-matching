"""Chart for the stage-2 attachment decision memo (idea #4 seam screen).

One panel: the in-run 256-frame probe curves of both screen arms —
F (frozen trunk, ran its full 10k) and K (KI-joint, owner-killed at
step ~4160 on cost). Values transcribed from the box train_log.jsonl
of each run (fontaine_molmo2_flow_{frozen,kijoint}_10k_ddp4);
eval-report dark theme (page #121417), blue = F, amber = K.
Identity carried by direct labels, never color alone.

Usage: uv run python fontaine/scripts/attach_screen_probe_chart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/attach_screen"

PAGE = "#121417"
BLUE = "#648fff"  # F — frozen trunk (the adopted default)
AMBER = "#ffb000"  # K — KI-joint (killed at ~4160)
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

# eval_chunk_mae at each 500-step eval, from train_log.jsonl (box).
F_CURVE = [
    (500, 16.7479),
    (1000, 12.7446),
    (1500, 12.4489),
    (2000, 11.2685),
    (2500, 11.4606),
    (3000, 11.6389),
    (3500, 10.7844),
    (4000, 11.1257),
    (4500, 11.0008),
    (5000, 10.2595),
    (5500, 10.6745),
    (6000, 10.0847),
    (6500, 9.9192),
    (7000, 9.6171),
    (7500, 9.9391),
    (8000, 9.3736),
    (8500, 9.6348),
    (9000, 9.1823),
    (9500, 9.6911),
    (10000, 9.3798),
]
K_CURVE = [
    (500, 15.9236),
    (1000, 13.0752),
    (1500, 13.0096),
    (2000, 11.6668),
    (2500, 12.4241),
    (3000, 11.6124),
    (3500, 11.2033),
    (4000, 10.9664),
]
K_KILL_STEP = 4160
STATE_COPY = 11.7639  # panel_v2 execution oracle (core 15,056 frames)
F_PANEL_10K = 9.4157  # F endpoint panel_v2 heun30/draws1/stable chunk MAE


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=160)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)

    ax.axhline(STATE_COPY, color=META, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.text(
        10000,
        STATE_COPY + 0.12,
        "state-copy execution oracle 11.76 (panel)",
        color=META,
        fontsize=8.5,
        ha="right",
        va="bottom",
    )

    fx, fy = zip(*F_CURVE, strict=True)
    kx, ky = zip(*K_CURVE, strict=True)
    ax.plot(fx, fy, color=BLUE, linewidth=2, zorder=4)
    ax.plot(kx, ky, color=AMBER, linewidth=2, zorder=3)

    ax.axvline(K_KILL_STEP, color=AMBER, linestyle=(0, (2, 3)), linewidth=1.2, zorder=2)
    ax.text(
        K_KILL_STEP + 120,
        15.6,
        "K killed ~4160\n(owner cost call:\n3.78 vs 0.92 s/step)",
        color=AMBER,
        fontsize=8.5,
        ha="left",
        va="top",
    )

    ax.text(
        6600,
        10.55,
        "F — frozen trunk",
        color=BLUE,
        fontsize=9.5,
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    ax.text(
        2450,
        12.75,
        "K — KI-joint",
        color=AMBER,
        fontsize=9.5,
        ha="left",
        va="bottom",
        fontweight="bold",
    )

    ax.plot([10000], [F_PANEL_10K], marker="o", markersize=8, color=BLUE, zorder=5)
    ax.text(
        9800,
        F_PANEL_10K + 0.35,
        "F panel @10k: 9.42",
        color=TEXT,
        fontsize=8.5,
        ha="right",
        va="bottom",
    )
    ax.set_ylim(8.6, 17.3)

    ax.set_xlabel("training step", color=META, fontsize=9)
    ax.set_ylabel(
        "in-run probe chunk MAE (256 held-out frames)",
        color=META,
        fontsize=9,
    )
    ax.tick_params(colors=META, labelsize=9)
    ax.set_xlim(300, 10300)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(
        "Attachment screen probes: K never opened a gap before the cost kill "
        "(K−F mean +0.21 over 8 matched evals)",
        color=TEXT,
        fontsize=10.5,
        loc="left",
        pad=12,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "attach_probe_curves.svg", facecolor=PAGE, bbox_inches="tight")
    fig.savefig(OUT / "attach_probe_curves.png", facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {OUT}/attach_probe_curves.{{svg,png}}")


if __name__ == "__main__":
    main()
