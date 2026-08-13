"""Results chart for the fitted-lens wrist gate probe
(sim-fit-real-lens-model leg (c); eval-report dark scheme). Two
panels: wrist 5-NN AUROC per lens arm against the 0.548 registered
gate (the 08-12 periphery-retune anchor), and the paired per-frame
dknn5 vs the equidistant control with bootstrap CI95 whiskers — the
decomposition that shows the fit's CENTER component reproducing the
full-fit regression while the curve-only refit passes the gate.
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
BLUE, ACCENT, RED = "#4589ff", "#ffb000", "#fa4d56"
GATE = 0.548

ARMS = (
    ("equidistant\n(deployed)", "equidistant"),
    ("full fit\n(center+curve)", "fitted"),
    ("center only", "centeronly"),
    ("curve only\n(refit)", "curveonly"),
)


def knn5(report_dir: Path, arm: str) -> tuple[float, np.ndarray]:
    path = report_dir / f"analysis__sim_encoder_ood_probe_lensgate_{arm}_arm.json"
    data = json.loads(path.read_text())
    block = data["cameras"]["wrist"]["knn5_secondary"]
    return block["auroc_sim_vs_real"], np.array(block["distances"]["sim"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/lens_fit/chart__lens_gate.png"),
    )
    args = ap.parse_args()

    aurocs, sims = zip(*(knn5(args.reports, arm) for _, arm in ARMS), strict=True)
    control = sims[0]
    rng = np.random.default_rng(0)
    deltas, cis = [], []
    for sim in sims[1:]:
        delta = sim - control
        boots = [float(np.mean(rng.choice(delta, len(delta)))) for _ in range(10000)]
        deltas.append(delta.mean())
        cis.append(np.percentile(boots, [2.5, 97.5]))

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(11.0, 4.6), width_ratios=[1, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)

    colors = [META, RED, RED, ACCENT]
    x = np.arange(len(ARMS))
    ax.bar(x, aurocs, 0.62, color=colors, zorder=3)
    ax.axhline(GATE, color=BLUE, lw=1.2, ls="--", zorder=2)
    ax.text(
        2.98,
        GATE - 0.004,
        "gate 0.548",
        color=BLUE,
        fontsize=8.5,
        ha="right",
        va="top",
    )
    ax.axhline(0.5, color=GRID, lw=1.0, zorder=2)
    ax.text(-0.55, 0.502, "0.5 = indistinguishable", color=META, fontsize=8, ha="left")
    for xi, val in zip(x, aurocs, strict=True):
        ax.text(xi, val + 0.004, f"{val:.3f}", color=TEXT, fontsize=9.5, ha="center")
    ax.set_xticks(x, [label for label, _ in ARMS], color=TEXT, fontsize=9)
    ax.set_ylim(0.45, 0.72)
    ax.set_ylabel("wrist 5-NN AUROC (sim vs held-out real)", color=TEXT, fontsize=9.5)
    ax.set_title(
        "Lens arms on the pinned encoder probe (20 seeds x 5 draws)",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    dx = np.arange(3)
    dcolors = [RED, RED, ACCENT]
    for xi, (mean, (lo, hi), color) in enumerate(
        zip(deltas, cis, dcolors, strict=True),
    ):
        dax.errorbar(
            xi,
            mean * 1e6,
            yerr=[[(mean - lo) * 1e6], [(hi - mean) * 1e6]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=5,
            markersize=8,
            zorder=3,
        )
    dax.axhline(0.0, color=GRID, lw=1.0, zorder=2)
    labels = [label for label, _ in ARMS[1:]]
    closers = [(sim - control < 0).sum() for sim in sims[1:]]
    for xi, (mean, n) in enumerate(zip(deltas, closers, strict=True)):
        dax.text(
            xi,
            mean * 1e6 + (0.12 if mean > 0 else -0.16),
            f"{n}/100 closer",
            color=TEXT,
            fontsize=8.5,
            ha="center",
        )
    dax.set_xticks(dx, labels, color=TEXT, fontsize=9)
    dax.set_xlim(-0.6, 2.6)
    dax.set_ylabel("paired dknn5 vs deployed control (x1e-6)", color=TEXT, fontsize=9.5)
    dax.set_title(
        "Per-frame paired delta, bootstrap CI95",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    fig.suptitle(
        "Fitted wrist lens gate: the fit's center term double-counts the pose fit; "
        "the curve-only refit passes",
        color=TEXT,
        fontsize=11.5,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
