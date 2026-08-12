"""Screen-close charts for the ER-init screen (er_60k results post).

Three figures for the consolidated results post, all data read from
banked artifacts (no live hosts — the box is gone; train logs come
from the ~/box_archive salvage):

1. er60k_probe_overlay_close — the full-run in-run probe
   (eval_chunk_mae, 256 held-out frames) of er_60k vs the 40k AR
   baseline (shared shuffle seed 0) and the 40k->60k continuation.
2. er60k_rung_trajectory — panel core chunk_mae at the four eval
   rungs (15k/35k/55k/60k) against the two banked anchor lines.
3. er60k_decision_cis — the two paired per-frame decision reads with
   CI95 (from analysis__er60k_endpoint_vs_banked_k4l2.json).

Eval-report dark theme; blue = Molmo2-init lineage (40k baseline
solid, continuation dashed — same entity, same hue), magenta =
er_60k (CVD-checked house pair on this surface).

Usage: uv run python fontaine/scripts/er60k_screen_close_charts.py
       [--archive PATH]   # train-log salvage root (default ~/box_archive)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/er60k"
REPORTS = ROOT / "reports"

PAGE = "#121417"
BLUE = "#648fff"  # Molmo2-init lineage (40k baseline + continuation)
MAGENTA = "#dc267f"  # er_60k (Molmo2-ER init)
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

RUNS = {
    "er": "fontaine_molmo2_er_60k_ddp4",
    "ar40k": "fontaine_molmo2_ar_40k_ddp4",
    "cont": "fontaine_molmo2_ar_60k_ddp4",
}
RUNG_STEPS = [15000, 35000, 55000, 60000]


def probe_series(archive: Path, run: str) -> list[tuple[int, float]]:
    log = archive / "outputs_train_logs" / run / "train_log.jsonl"
    series = []
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval_chunk_mae" in rec and "step" in rec:
            series.append((int(rec["step"]), float(rec["eval_chunk_mae"])))
    series.sort()
    if len(series) < 2:
        sys.exit(f"ABORT: {len(series)} probe point(s) in {log}")
    return series


def panel_core(path: Path, step: int) -> float:
    d = json.loads(path.read_text())
    for s in d["summaries"]:
        if s["policy"] == f"bijou@{step}":
            return float(s["chunk_mae"])
    sys.exit(f"ABORT: no bijou@{step} arm in {path.name}")


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {OUT}/{name}.{{svg,png}}")


def fig_probe_overlay(archive: Path) -> None:
    er = probe_series(archive, RUNS["er"])
    ar = probe_series(archive, RUNS["ar40k"])
    cont = probe_series(archive, RUNS["cont"])

    ar_d = dict(ar)
    matched = [(s, m, ar_d[s]) for s, m in er if s in ar_d]
    late = [(s, e - b) for s, e, b in matched if s >= 20000]
    mean_late = sum(d for _, d in late) / len(late)
    er_best = min(er, key=lambda p: p[1])
    print(f"probe: {len(matched)} matched steps, mean delta 20k+: {mean_late:+.3f}")
    print(f"probe: er run-best {er_best[1]:.3f}@{er_best[0]}")

    fig, (a0, a1) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.4),
        dpi=160,
        width_ratios=[1, 1.25],
    )
    fig.patch.set_facecolor(PAGE)

    for ax, lo in ((a0, 500), (a1, 15000)):
        for pts, color, ls in (
            (ar, BLUE, "-"),
            (cont, BLUE, "--"),
            (er, MAGENTA, "-"),
        ):
            sub = [(s, v) for s, v in pts if s >= lo]
            if sub:
                xs, ys = zip(*sub, strict=True)
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, zorder=4)
        style_axis(ax)
        ax.set_xlabel("training step", color=META, fontsize=9)

    a0.set_ylabel("probe chunk MAE (256 held-out frames)", color=META, fontsize=9)
    a0.set_title("full run", color=META, fontsize=9)
    a1.set_title("steps 15k+ (the decided stretch)", color=META, fontsize=9)

    a1.set_ylim(4.8, 7.65)
    a1.text(
        27000,
        7.42,
        "40k baseline (Molmo2 init)",
        color=BLUE,
        fontsize=9,
        ha="left",
        va="top",
        fontweight="bold",
    )
    a1.text(
        59500,
        6.95,
        "40k→60k continuation",
        color=BLUE,
        fontsize=9,
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    a1.text(
        59500,
        dict(er)[60000] - 0.28,
        "er_60k (Molmo2-ER init)",
        color=MAGENTA,
        fontsize=9,
        ha="right",
        va="top",
        fontweight="bold",
    )
    a1.annotate(
        f"run-best {er_best[1]:.2f}@{er_best[0]}",
        xy=er_best,
        xytext=(27500, 5.02),
        color=TEXT,
        fontsize=8.5,
        va="center",
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.8},
    )

    fig.suptitle(
        "er_60k screen close: in-run probe, ER init vs Molmo2 init at shared "
        f"shuffle seed — mean matched-step delta {mean_late:+.2f} from 20k on "
        "(record-only; the panel makes the claims)",
        color=TEXT,
        fontsize=10,
        x=0.09,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "er60k_probe_overlay_close")
    plt.close(fig)


def fig_rung_trajectory() -> None:
    er_name = RUNS["er"]
    rungs = [
        (
            s,
            panel_core(
                REPORTS / f"eval__{er_name}__step_{s:06d}__panel_curated_v0_k4l2.json",
                s,
            ),
        )
        for s in RUNG_STEPS
    ]
    ar40k = panel_core(
        REPORTS
        / "eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2.json",
        40000,
    )
    cont = panel_core(
        REPORTS
        / "eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2.json",
        60000,
    )
    print("panel rungs:", ", ".join(f"{v:.4f}@{s}" for s, v in rungs))
    print(f"anchors: 40k endpoint {ar40k:.4f}, 60k-cont {cont:.4f}")

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    fig.patch.set_facecolor(PAGE)

    ax.axhline(ar40k, color=BLUE, linewidth=1.6, zorder=3)
    ax.axhline(cont, color=BLUE, linewidth=1.6, linestyle="--", zorder=3)
    xs, ys = zip(*rungs, strict=True)
    ax.plot(xs, ys, color=MAGENTA, linewidth=2, marker="o", markersize=6, zorder=4)

    for s, v in rungs:
        d = v - ar40k
        ax.annotate(
            f"{d:+.2f}",
            xy=(s, v),
            xytext=(0, -17 if d > 0 else -19),
            textcoords="offset points",
            ha="center",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
    ax.text(
        14200,
        ar40k + 0.035,
        f"40k endpoint {ar40k:.4f} (Molmo2 init)",
        color=BLUE,
        fontsize=9,
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    ax.text(
        14200,
        cont - 0.035,
        f"60k continuation {cont:.4f}",
        color=BLUE,
        fontsize=9,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax.text(
        43000,
        6.45,
        "er_60k rungs (Molmo2-ER init)",
        color=MAGENTA,
        fontsize=9,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    style_axis(ax)
    ax.set_xlabel("training step", color=META, fontsize=9)
    ax.set_ylabel("panel core chunk MAE (17,204 frames)", color=META, fontsize=9)
    ax.set_xticks(xs)
    ax.set_ylim(min(ys) - 0.22, max(ys) + 0.15)

    fig.suptitle(
        "er_60k panel rung trajectory: deltas vs the banked 40k endpoint — "
        "crosses below both anchors between 35k and 55k",
        color=TEXT,
        fontsize=10,
        x=0.125,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "er60k_rung_trajectory")
    plt.close(fig)


def fig_decision_cis() -> None:
    d = json.loads(
        (REPORTS / "analysis__er60k_endpoint_vs_banked_k4l2.json").read_text(),
    )
    reads = d["paired_reads"]
    rows = [
        ("vs 40k endpoint\n(the init read)", reads["ar_40k endpoint"]),
        ("vs 60k continuation\n(steps-matched)", reads["ar_60k continuation"]),
    ]
    for label, r in rows:
        print(
            f"decision: {label.splitlines()[0]} delta {r['delta_pooled']:+.4f} "
            f"CI95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] {r['classification']}",
        )

    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=160)
    fig.patch.set_facecolor(PAGE)

    ax.axvline(0, color=META, linewidth=1.2, zorder=3)
    for i, (_label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = r["ci95"]
        ax.plot(
            [lo, hi],
            [y, y],
            color=MAGENTA,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            [r["delta_pooled"]],
            [y],
            marker="o",
            markersize=8,
            color=MAGENTA,
            markeredgecolor=PAGE,
            markeredgewidth=2,
            zorder=5,
        )
        ax.annotate(
            f"{r['delta_pooled']:+.4f}  [{lo:+.3f}, {hi:+.3f}]",
            xy=(r["delta_pooled"], y),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)], color=TEXT, fontsize=9)
    ax.set_xlim(-0.32, 0.09)
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.text(0.004, -0.45, "0 = no effect", color=META, fontsize=8.5, ha="left")
    ax.text(-0.315, -0.45, "← er_60k better", color=META, fontsize=8.5, ha="left")

    style_axis(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel(
        "paired per-frame Δ chunk MAE at the endpoint (n = 17,204, CI95)",
        color=META,
        fontsize=9,
    )

    fig.suptitle(
        "The decision read: BELOW-BASELINE on both legs, CI excludes zero "
        "— the ER init wins",
        color=TEXT,
        fontsize=10,
        x=0.125,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "er60k_decision_cis")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive",
        default=str(Path.home() / "box_archive"),
        help="train-log salvage root",
    )
    args = ap.parse_args()
    archive = Path(args.archive)

    fig_probe_overlay(archive)
    fig_rung_trajectory()
    fig_decision_cis()


if __name__ == "__main__":
    main()
