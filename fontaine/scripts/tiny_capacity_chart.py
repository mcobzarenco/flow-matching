"""Chart for the T1 tiny-expert capacity rung readout (Δ_capacity@10k).

Three panels, eval-report dark theme (page #121417), identity by
direct labels + distinct markers, never color alone:
  A) in-run 256-frame probe curves, tiny (h256/d12) vs F (h1024/d12),
     fully step-matched (both eff-48, 10k, same LR schedule); tiny's
     host-RAM OOM + resume-from-8750 window annotated.
  B) per-step-in-horizon panel-v2 MAE curves for both arms (read 5,
     record-only), from the analysis json's step_curve_{F,K}.
  C) the primary read: paired per-frame Δ_capacity@10k (tiny − F) with
     bootstrap CI95 on a band-shaded axis (|Δ| ≤ 0.3 prior confirmed;
     Δ ≥ +1.0 capacity binds — bands pinned in the pre-reg).

F curve transcribed from the box train_log.jsonl (same values as
attach_screen_probe_chart.py); tiny curve read from the local jsonl
(resume replay rungs overwrite pre-kill duplicates, matching the
lineage that produced step_010000). Δ numbers from the analysis json
written by attach_seam_results.py pointed at explicit tiny/F paths.

Usage: uv run python fontaine/scripts/tiny_capacity_chart.py \
    [--analysis reports/analysis__tiny10k_delta_capacity.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/tiny10k"
TINY_JSONL = (
    ROOT / "outputs/train/fontaine_molmo2_flow_tiny_h256_10k_1xh100/train_log.jsonl"
)

PAGE = "#121417"
BLUE = "#648fff"  # F — h1024 expert (entity color carried from the attach chart)
MAGENTA = "#dc267f"  # tiny — h256 expert
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

# eval_chunk_mae at each 500-step eval, F arm, from the box train_log.jsonl
# (identical transcription to attach_screen_probe_chart.py).
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

CONFIRM_BAND = 0.3  # |Δ| ≤ 0.3 → capacity prior confirmed (pre-reg)
BINDS_AT = 1.0  # Δ ≥ +1.0 → capacity binds (pre-reg)
RESUME_FROM, OOM_AT = 8750, 9060


def tiny_curve() -> list:
    rungs = {}
    for line in TINY_JSONL.read_text().splitlines():
        d = json.loads(line)
        if "eval_chunk_mae" in d:
            rungs[d["step"]] = d["eval_chunk_mae"]
    return sorted(rungs.items())


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=META, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis",
        default=str(ROOT / "reports/analysis__tiny10k_delta_capacity.json"),
    )
    a = p.parse_args()
    res = json.loads(Path(a.analysis).read_text())
    r1 = res["read1_delta_seam"]
    pooled_f = res["arms_pooled"]["F"]["chunk_mae"]
    pooled_t = res["arms_pooled"]["K"]["chunk_mae"]
    mean, (lo, hi) = r1["mean"], r1["ci95"]
    curve_f = res["read5_record_only"]["step_curve_F"]
    curve_t = res["read5_record_only"]["step_curve_K"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(12.6, 4.4),
        facecolor=PAGE,
        width_ratios=[2.1, 1.3, 1.0],
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.14, wspace=0.3)

    # --- A: probe ladder, step-matched -----------------------------------
    style_axes(ax_a)
    fx, fy = zip(*F_CURVE, strict=True)
    tx, ty = zip(*tiny_curve(), strict=True)
    ax_a.plot(fx, fy, color=BLUE, lw=2, marker="o", ms=4, mfc=PAGE, mew=1.4)
    ax_a.plot(tx, ty, color=MAGENTA, lw=2, marker="s", ms=4, mfc=PAGE, mew=1.4)
    ax_a.axvspan(RESUME_FROM, OOM_AT, color=MAGENTA, alpha=0.08, lw=0)
    ax_a.text(
        RESUME_FROM + 40,
        16.4,
        "OOM →\nresume\nfrom 8750",
        fontsize=6.5,
        color=META,
        ha="left",
        va="top",
    )
    ax_a.annotate(
        f"F  {fy[-1]:.4f}",
        (fx[-1], fy[-1]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        color=TEXT,
        fontweight="bold",
    )
    ax_a.annotate(
        f"tiny  {ty[-1]:.4f}",
        (tx[-1], ty[-1]),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=8,
        color=TEXT,
        fontweight="bold",
    )
    ax_a.text(
        2600,
        16.6,
        "F (h1024/d12)",
        fontsize=8,
        color=BLUE,
        fontweight="bold",
    )
    ax_a.text(
        2600,
        15.6,
        "tiny (h256/d12)",
        fontsize=8,
        color=MAGENTA,
        fontweight="bold",
    )
    ax_a.set_xlim(0, 11600)
    ax_a.set_xlabel("training step (both arms eff-batch 48)", fontsize=8, color=META)
    ax_a.set_ylabel("probe eval_chunk_mae (256 frames)", fontsize=8, color=META)
    ax_a.set_title(
        "In-run probe, step-matched",
        fontsize=9.5,
        color=TEXT,
        loc="left",
        pad=6,
    )

    # --- B: per-step-in-horizon MAE curves (record-only) ------------------
    style_axes(ax_b)
    steps = range(1, len(curve_f) + 1)
    ax_b.plot(steps, curve_f, color=BLUE, lw=2)
    ax_b.plot(steps, curve_t, color=MAGENTA, lw=2, dashes=(4, 1.6))
    ax_b.text(
        len(curve_f) * 0.45,
        curve_f[int(len(curve_f) * 0.55)],
        "F",
        fontsize=8.5,
        color=BLUE,
        fontweight="bold",
        va="bottom",
    )
    ax_b.text(
        len(curve_t) * 0.7,
        curve_t[int(len(curve_t) * 0.62)],
        "tiny",
        fontsize=8.5,
        color=MAGENTA,
        fontweight="bold",
        va="top",
    )
    ax_b.set_xlabel("step in 50-step horizon", fontsize=8, color=META)
    ax_b.set_ylabel("panel-v2 MAE (core frames)", fontsize=8, color=META)
    ax_b.set_title(
        "Error over the horizon @10k",
        fontsize=9.5,
        color=TEXT,
        loc="left",
        pad=6,
    )

    # --- C: the primary read — Δ_capacity with CI95 on the pinned bands ---
    style_axes(ax_c)
    ax_c.axvspan(-CONFIRM_BAND, CONFIRM_BAND, color="#2ecc71", alpha=0.10, lw=0)
    ax_c.axvspan(BINDS_AT, 1.6, color="#e74c3c", alpha=0.10, lw=0)
    ax_c.axvline(0, color=META, lw=0.8, dashes=(3, 2))
    ax_c.errorbar(
        [mean],
        [0.5],
        xerr=[[mean - lo], [hi - mean]],
        fmt="D",
        ms=7,
        color=MAGENTA,
        ecolor=MAGENTA,
        elinewidth=2.2,
        capsize=5,
        capthick=2.2,
    )
    ax_c.text(
        mean,
        0.62,
        f"Δ = {mean:+.4f}",
        fontsize=10,
        color=TEXT,
        ha="center",
        fontweight="bold",
    )
    ax_c.text(
        mean,
        0.38,
        f"CI95 [{lo:+.3f}, {hi:+.3f}]",
        fontsize=7.5,
        color=META,
        ha="center",
    )
    ax_c.text(
        0,
        0.06,
        "prior confirmed\n|Δ| ≤ 0.3",
        fontsize=7,
        color="#2ecc71",
        ha="center",
    )
    ax_c.text(
        1.3,
        0.06,
        "capacity binds\nΔ ≥ +1.0",
        fontsize=7,
        color="#e74c3c",
        ha="center",
    )
    ax_c.set_xlim(-0.8, 1.6)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([])
    ax_c.set_xlabel(
        "Δ_capacity@10k = tiny − F, paired per-frame",
        fontsize=8,
        color=META,
    )
    ax_c.set_title(
        "Primary read (bands pre-registered)",
        fontsize=9.5,
        color=TEXT,
        loc="left",
        pad=6,
    )

    fig.suptitle(
        "T1 tiny-expert capacity rung — h256 vs h1024 on the frozen 60k trunk, "
        f"fully matched @10k  ·  panel_v2 pooled: tiny {pooled_t:.4f} vs F {pooled_f:.4f}",
        fontsize=11,
        color=TEXT,
        x=0.055,
        ha="left",
        y=0.97,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "delta_capacity_10k.png"
    fig.savefig(out, dpi=160, facecolor=PAGE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
