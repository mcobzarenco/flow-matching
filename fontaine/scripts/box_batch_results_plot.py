"""Figure for the box-batch 40k results post (pre-reg 2026-08-05).

Two panels:
  (left)  per-arm pooled panel chunk_mae bars (A-s0/s1/s2 replicates vs B
          aux-off) with the state-copy baseline and the mainline AR-100k
          anchor as reference lines (magnitude job; replicates share one
          categorical slot, B takes the second).
  (right) per-frame chunk_mae delta histogram (B - A-s0, core frames),
          diverging blue/red around zero with the bootstrap mean + CI
          annotated; clipped tails reported in the caption text, never
          silently.

Reads the analysis JSON + the two npzs; writes SVG to blog assets.
Palette: dataviz reference instance (light mode), same constants as
sign_stage2_plot.py.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"  # slot 1: A (aux-on) replicates
ORANGE = "#eb6834"  # slot 2: B (aux-off)
RED = "#e34948"  # diverging warm pole (B worse)

A_NPZ = "reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000__panel_curated_v0_k4l2.npz"
B_NPZ = (
    "/home/ubuntu/boxsync/reports/"
    "eval__fontaine_arb_rcond_auxoff_40k_1xh100__step_040000__panel_curated_v0_k4l2.npz"
)
ANALYSIS = "reports/analysis__box_batch_40k_k4l2.json"
OUT = "fontaine/blog/src/assets/2026-08-06-box-batch-results.svg"

STATE_COPY = 11.7848
AR_ANCHOR = 5.8026
CLIP = 12.0


def per_frame(d):
    truth, valid = d["truth"], d["valid"]
    pred = d["pred:bijou@40000"]
    err = np.abs(pred - truth)
    m = valid[:, :, None] & np.isfinite(truth).all(-1, keepdims=True)
    w = np.broadcast_to(m, err.shape)
    pf = np.where(w, err, 0).sum((1, 2)) / np.maximum(w.sum((1, 2)), 1)
    return pf, d["core"]


def main():
    an = json.load(open(ANALYSIS))
    pooled = an["pooled"]
    prim = an["primary_B_minus_As0"]

    pa, core = per_frame(np.load(A_NPZ, allow_pickle=True))
    pb, _ = per_frame(np.load(B_NPZ, allow_pickle=True))
    dlt = (pb - pa)[core]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "svg.fonttype": "none",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- left: per-arm pooled chunk_mae -------------------------------
    arms = ["A-s0", "A-s1", "A-s2", "B"]
    vals = [pooled[a]["chunk_mae"] for a in arms]
    colors = [BLUE, BLUE, BLUE, ORANGE]
    x = np.arange(len(arms))
    ax1.bar(x, vals, width=0.55, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.15, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax1.axhline(STATE_COPY, color=BASELINE, lw=1.4, ls="--", zorder=2)
    ax1.text(3.55, STATE_COPY + 0.12, f"state-copy {STATE_COPY:.2f}", ha="right", fontsize=8.5, color=MUTED)
    ax1.axhline(AR_ANCHOR, color=MUTED, lw=1.2, ls=":", zorder=2)
    ax1.text(3.55, AR_ANCHOR - 0.55, f"mainline AR-100k 4×H100 {AR_ANCHOR:.2f}", ha="right", fontsize=8.5, color=MUTED)
    ax1.set_xticks(x, [f"{a}\n(aux-on)" if a.startswith("A") else f"{a}\n(aux-off)" for a in arms], fontsize=9)
    ax1.set_ylabel("panel chunk MAE (deg)", fontsize=9.5, color=INK_2)
    ax1.set_ylim(0, 13.2)
    ax1.set_title("40k @ eff-10, pooled panel chunk MAE", fontsize=10.5, color=INK, pad=10)

    # ---- right: per-frame delta histogram -----------------------------
    clipped = np.clip(dlt, -CLIP, CLIP)
    n_clip = int((np.abs(dlt) > CLIP).sum())
    bins = np.linspace(-CLIP, CLIP, 61)
    neg = clipped[clipped < 0]
    pos = clipped[clipped >= 0]
    ax2.hist(neg, bins=bins, color=BLUE, zorder=3)
    ax2.hist(pos, bins=bins, color=RED, zorder=3)
    ax2.axvline(0, color=BASELINE, lw=1.2, zorder=4)
    lo, hi = prim["ci95"]
    ax2.axvspan(lo, hi, color=RED, alpha=0.18, zorder=2)
    ax2.axvline(prim["mean"], color=INK, lw=1.4, zorder=5)
    ax2.text(
        prim["mean"] + 0.35,
        ax2.get_ylim()[1] * 0.93,
        f"mean +{prim['mean']:.3f}\nCI95 [{lo:.3f}, {hi:.3f}]",
        fontsize=8.5,
        color=INK,
        va="top",
    )
    ax2.text(-CLIP + 0.4, ax2.get_ylim()[1] * 0.93, "← B better", fontsize=9, color=BLUE, va="top")
    ax2.text(CLIP - 0.4, ax2.get_ylim()[1] * 0.80, "B worse →", fontsize=9, color=RED, va="top", ha="right")
    ax2.set_xlabel("per-frame Δ chunk MAE, B − A-s0 (deg)", fontsize=9.5, color=INK_2)
    ax2.set_ylabel("core frames", fontsize=9.5, color=INK_2)
    ax2.set_title(
        f"aux-off effect per frame (n={prim['n_frames']:,}; {n_clip} beyond ±{CLIP:.0f} clipped)",
        fontsize=10.5,
        color=INK,
        pad=10,
    )

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT, format="svg", facecolor=SURFACE)
    print(f"wrote {OUT} (clipped {n_clip} frames beyond +/-{CLIP})")


if __name__ == "__main__":
    main()
