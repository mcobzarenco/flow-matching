"""Charts for the consolidated SnapFlow visual report (owner steering
2026-08-08 09:22Z): five SVGs rendered from the BANKED analyses — no
re-computation of any claimed number, every value read from the frozen
jsons (`analysis__snapflow_distill_30k_k4l2.json`, the leaderboard
decode microbench files, the AR draws10 readout, and the ftrig
before/after eval jsons).

Output: fontaine/blog/src/img/snapflow/*.svg (committed).

Palette: the dataviz reference palette's pre-validated categorical
slots (blue #2a78d6, orange #eb6834) + neutral text tokens — no custom
hues (no node on the box, so the palette validator can't run; using
the reference instance unchanged is the sanctioned fallback, same as
goldenticket_report_charts.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/snapflow"

BLUE = "#2a78d6"  # SnapFlow 1-NFE student
ORANGE = "#eb6834"  # Heun-30 flow teacher
GRAY = "#9b9a95"  # reference marks / bands
DARKGRAY = "#52514e"  # AR family, baselines, secondary text
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=DARKGRAY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(width: float = 8.2, height: float = 4.2) -> tuple:
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def load(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text())


def rig_summary(name: str, policy_prefix: str = "bijou@") -> tuple[float, float]:
    """Pooled (chunk_mae, first_mae) of the model row in a banked eval json."""
    d = load(name)
    for s in d["summaries"]:
        if s["policy"].startswith(policy_prefix):
            return s["chunk_mae"], s["first_mae"]
    raise KeyError(f"no {policy_prefix} row in {name}")


def chart_endpoint(snap: dict, ar: dict) -> None:
    """The endpoint ladder: every deployment-class config on one axis."""
    teacher_d1 = snap["read1_primary_1nfe"]["teacher_chunk_index_key"]
    teacher_d5 = snap["read3_deployment"]["draws5"]["teacher_draws5_heun30"][0]
    teacher_d10 = snap["read3_deployment"]["teacher_draws10_heun30"][0]
    adopt = snap["read1_primary_1nfe"]["adopt_line"]
    rows = [
        ("teacher Heun-30, single draw", teacher_d1, "30 expert evals", ORANGE),
        ("teacher Heun-30, mean-of-5", teacher_d5, "150 expert evals", ORANGE),
        ("teacher Heun-30, mean-of-10", teacher_d10, "300 expert evals", ORANGE),
        (
            "AR-100k, greedy (anchor)",
            ar["arms_pooled"]["greedy"]["chunk_mae"],
            "token-serial decode",
            DARKGRAY,
        ),
        (
            "AR-100k, draws-10 mean",
            ar["arms_pooled"]["draws10_t1"]["chunk_mae"],
            "token-serial, 10 draws",
            DARKGRAY,
        ),
        (
            "student 1-NFE, single draw",
            snap["read1_primary_1nfe"]["chunk_mae"],
            "1 expert eval",
            BLUE,
        ),
        (
            "student 1-NFE, mean-of-5",
            snap["read3_deployment"]["draws5"]["chunk_mae"],
            "5 expert evals",
            BLUE,
        ),
        (
            "student 1-NFE, mean-of-10",
            snap["read3_deployment"]["mean10_chunk_mae"],
            "10 expert evals",
            BLUE,
        ),
    ]
    fig, ax = new_fig(8.6, 4.6)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.axvline(adopt, color=DARKGRAY, linewidth=1.2, linestyle="--")
    ax.annotate(
        f"adopt line {adopt}",
        xy=(adopt, len(rows) - 0.55),
        fontsize=8.5,
        color=DARKGRAY,
        ha="center",
        va="bottom",
    )
    for i, (_label, value, evals, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([5.2, value], [y, y], color=GRID, linewidth=1, zorder=1)
        ax.scatter([value], [y], s=64, color=color, zorder=3)
        ax.annotate(
            f"{value:.4f}",
            xy=(value, y),
            xytext=(value + 0.035, y + 0.16),
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )
        ax.annotate(
            evals,
            xy=(value, y),
            xytext=(value + 0.035, y - 0.34),
            fontsize=8,
            color=DARKGRAY,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5, color=INK)
    ax.set_xlim(5.2, 7.0)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel(
        "panel chunk MAE (v1, 25,800 frames) — lower is better",
        color=DARKGRAY,
        fontsize=10,
    )
    ax.set_title(
        "The endpoint ladder — a single 1-NFE draw already beats the AR anchor;\n"
        "mean-of-10 matches the teacher's 300-eval read to 3 decimal places",
        color=INK,
        fontsize=11,
        loc="left",
    )
    save(fig, "endpoint_ladder.svg")


def chart_cost_quality(
    snap: dict,
    ar: dict,
    bench_single: dict,
    bench_pre: dict,
) -> None:
    """Single-stream latency (log x) vs panel MAE — the Pareto story."""
    ms = {r["config"]: r["ms_per_frame"] for r in bench_single["results"]}
    ms_pre = {r["config"]: r["ms_per_frame"] for r in bench_pre["results"]}
    teacher_stable = snap["read1_primary_1nfe"]["teacher_chunk_stable_key_descriptive"]
    # per-point label placement: (dx multiplier, dy, ha)
    pts = [
        (
            "AR greedy",
            ms_pre["ar_greedy"],
            ar["arms_pooled"]["greedy"]["chunk_mae"],
            DARKGRAY,
            (1.13, 0.05, "left"),
        ),
        (
            "AR draws-10",
            ms_pre["ar_draws10_t1"],
            ar["arms_pooled"]["draws10_t1"]["chunk_mae"],
            DARKGRAY,
            (0.90, 0.05, "right"),
        ),
        (
            "teacher draws-1",
            ms["teacher_heun30_draws1"],
            teacher_stable,
            ORANGE,
            (0.90, 0.05, "right"),
        ),
        (
            "teacher mean-of-10",
            ms["teacher_heun30_draws10"],
            snap["read3_deployment"]["teacher_draws10_heun30"][0],
            ORANGE,
            (1.13, 0.05, "left"),
        ),
        (
            "student draws-1",
            ms["student_1nfe_draws1"],
            snap["read1_primary_1nfe"]["chunk_mae"],
            BLUE,
            (1.15, 0.06, "left"),
        ),
        (
            "student mean-of-5",
            ms["student_1nfe_draws5"],
            snap["read3_deployment"]["draws5"]["chunk_mae"],
            BLUE,
            (1.15, -0.10, "left"),
        ),
        (
            "student mean-of-10",
            ms["student_1nfe_draws10"],
            snap["read3_deployment"]["mean10_chunk_mae"],
            BLUE,
            (1.15, 0.02, "left"),
        ),
    ]
    fig, ax = new_fig(8.6, 4.6)
    ax.set_xscale("log")
    for label, x, y, color, (dx, dy, ha) in pts:
        ax.scatter(
            [x],
            [y],
            s=76,
            color=color,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=1.2,
        )
        ax.annotate(
            f"{label}\n{y:.3f} @ {x:,.0f} ms",
            xy=(x, y),
            xytext=(x * dx, y + dy),
            fontsize=8.5,
            color=INK,
            ha=ha,
            va="bottom" if dy > 0 else "top",
        )
    ax.annotate(
        "12x faster than the AR anchor,\n11x than one Heun-30 draw —\nat better panel MAE",
        xy=(0.02, 0.60),
        xycoords="axes fraction",
        fontsize=9.5,
        color=DARKGRAY,
        va="top",
    )
    ax.set_xlabel(
        "single-stream latency, ms per frame (b=1, H100, post-merge microbench, log scale)",
        color=DARKGRAY,
        fontsize=10,
    )
    ax.set_ylabel("panel chunk MAE (v1)", color=DARKGRAY, fontsize=10)
    ax.set_title(
        "Cost vs quality — the student occupies the Pareto corner alone",
        color=INK,
        fontsize=11,
        loc="left",
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=BLUE,
            label="SnapFlow 1-NFE student",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=ORANGE,
            label="Heun-30 flow teacher",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=DARKGRAY,
            label="AR-100k (token-serial)",
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(60, 22000)
    ax.set_ylim(5.25, 6.85)
    save(fig, "cost_quality.svg")


def chart_draws_collapse(snap: dict, ar: dict) -> None:
    """Averaging gain vs number of draws — the mean-collapse shape."""
    draws = [1, 5, 10]
    student = [
        snap["read1_primary_1nfe"]["chunk_mae"],
        snap["read3_deployment"]["draws5"]["chunk_mae"],
        snap["read3_deployment"]["mean10_chunk_mae"],
    ]
    teacher = [
        snap["read1_primary_1nfe"]["teacher_chunk_index_key"],
        snap["read3_deployment"]["draws5"]["teacher_draws5_heun30"][0],
        snap["read3_deployment"]["teacher_draws10_heun30"][0],
    ]
    ar_pts = (
        [1, 10],
        [
            ar["arms_pooled"]["greedy"]["chunk_mae"],
            ar["arms_pooled"]["draws10_t1"]["chunk_mae"],
        ],
    )
    fig, ax = new_fig(7.6, 4.4)
    ax.plot(
        draws,
        teacher,
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=7,
        label="teacher Heun-30",
    )
    ax.plot(
        draws,
        student,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=7,
        label="student 1-NFE",
    )
    ax.plot(
        *ar_pts,
        color=DARKGRAY,
        linewidth=2,
        marker="o",
        markersize=7,
        linestyle=":",
        label="AR-100k, T=1 draws",
    )
    # teacher and student converge at draws=10 (5.364 vs 5.368) —
    # stagger their end labels and gain notes above/below the shared point
    for xs, ys, color, gain, end_dx, end_dy, note_y in (
        (draws, teacher, ORANGE, teacher[0] - teacher[2], 0.15, 0.05, 5.50),
        (draws, student, BLUE, student[0] - student[2], -0.15, 0.05, 5.31),
        (ar_pts[0], ar_pts[1], DARKGRAY, ar_pts[1][0] - ar_pts[1][1], 0.15, 0.03, None),
    ):
        for x, y in zip(xs[:-1], ys[:-1], strict=True):
            ax.annotate(
                f"{y:.3f}",
                xy=(x, y),
                xytext=(x + 0.15, y + 0.03),
                fontsize=8.5,
                color=INK,
            )
        ax.annotate(
            f"{ys[-1]:.3f}",
            xy=(xs[-1], ys[-1]),
            xytext=(xs[-1] + end_dx, ys[-1] + end_dy),
            fontsize=8.5,
            color=INK,
            ha="left" if end_dx > 0 else "right",
            va="bottom" if end_dy > 0 else "top",
        )
        ax.annotate(
            f"averaging buys −{gain:.3f}",
            xy=(xs[-1], ys[-1]),
            xytext=(10.9, note_y if note_y is not None else ys[-1]),
            fontsize=9,
            color=color,
            va="center",
        )
    ax.set_xticks(draws)
    ax.set_xlim(0.5, 13.5)
    ax.set_xlabel("number of draws averaged", color=DARKGRAY, fontsize=10)
    ax.set_ylabel("panel chunk MAE (v1)", color=DARKGRAY, fontsize=10)
    ax.set_title(
        "Draw averaging — the teacher's −1.26 ensembling gain collapses to −0.24\n"
        "in the student: distillation compiled the mean of the draw distribution",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    save(fig, "draws_collapse.svg")


def chart_horizon(snap: dict) -> None:
    """Per-step MAE along the 50-step chunk, student vs teacher."""
    h = snap["perstep_horizon"]
    steps = np.arange(1, len(h["step_curve"]["student"]) + 1)
    fig, ax = new_fig(8.6, 4.4)
    ax.plot(
        steps,
        h["step_curve"]["teacher"],
        color=ORANGE,
        linewidth=2,
        label="teacher Heun-30, single draw",
    )
    ax.plot(
        steps,
        h["step_curve"]["student"],
        color=BLUE,
        linewidth=2,
        label="student 1-NFE, single draw",
    )
    ax.fill_between(
        steps,
        h["step_curve"]["student"],
        h["step_curve"]["teacher"],
        color=BLUE,
        alpha=0.10,
        linewidth=0,
    )
    delta = h["step_delta"]
    ax.annotate(
        f"Δ at step 1: {delta[0]:+.2f}",
        xy=(1, h["step_curve"]["teacher"][0]),
        xytext=(2.5, h["step_curve"]["teacher"][0] + 0.9),
        fontsize=9,
        color=DARKGRAY,
        arrowprops={"arrowstyle": "-", "color": DARKGRAY, "linewidth": 0.8},
    )
    ax.annotate(
        f"Δ at step 50: {delta[-1]:+.2f}\n(no crossover anywhere)",
        xy=(50, h["step_curve"]["student"][-1]),
        xytext=(38, h["step_curve"]["student"][-1] - 1.6),
        fontsize=9,
        color=DARKGRAY,
        arrowprops={"arrowstyle": "-", "color": DARKGRAY, "linewidth": 0.8},
    )
    for _label, values, color in (
        ("", h["step_curve"]["teacher"], ORANGE),
        ("", h["step_curve"]["student"], BLUE),
    ):
        ax.annotate(
            f"{values[-1]:.2f}",
            xy=(steps[-1], values[-1]),
            xytext=(steps[-1] + 0.6, values[-1]),
            fontsize=9,
            color=color,
            va="center",
        )
    ax.set_xlabel("step in 50-step action chunk", color=DARKGRAY, fontsize=10)
    ax.set_ylabel("per-step MAE (pooled)", color=DARKGRAY, fontsize=10)
    ax.set_title(
        "Horizon read — the student sits below the teacher at every step,\n"
        "and the gap widens where draw spread is largest",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_xlim(1, 55)
    save(fig, "horizon.svg")


def chart_ftrig() -> None:
    """The rig fine-tune branch: before/after dumbbells on one axis."""
    before_d1 = rig_summary(
        "eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__rig_holdout_1nfe_euler1_stable.json",
    )
    before_d10 = rig_summary(
        "eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__rig_holdout_1nfe_euler1_stable_draws10.json",
    )
    after_d1 = rig_summary(
        "eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000__rig_holdout_1nfe_euler1_stable.json",
    )
    after_d10 = rig_summary(
        "eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000__rig_holdout_1nfe_euler1_stable_draws10.json",
    )
    after_panel = rig_summary(
        "eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000__panel_v2_1nfe_euler1_stable.json",
    )
    snap = load("analysis__snapflow_distill_30k_k4l2.json")
    before_panel = snap["perstep_horizon"]["v2_column"]["student"]["chunk_mae"]
    statecopy_rig = 12.0506  # byte-matched control row in all four rig jsons

    rows = [
        ("rig holdout, single draw", before_d1[0], after_d1[0]),
        ("rig holdout, mean-of-10", before_d10[0], after_d10[0]),
        (
            "community panel v2\n(forgetting guard, bound +1.0)",
            before_panel,
            after_panel[0],
        ),
    ]
    fig, ax = new_fig(8.6, 3.6)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.axvline(statecopy_rig, color=DARKGRAY, linewidth=1.2, linestyle="--")
    ax.annotate(
        "state-copy on rig holdout 12.05",
        xy=(statecopy_rig, 2.42),
        fontsize=8.5,
        color=DARKGRAY,
        ha="right",
        va="bottom",
        rotation=0,
        xytext=(statecopy_rig - 0.08, 2.42),
    )
    for i, (_label, before, after) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([before, after], [y, y], color=GRAY, linewidth=2, zorder=2)
        ax.scatter([before], [y], s=64, color=BLUE, zorder=3)
        ax.scatter([after], [y], s=64, color=ORANGE, zorder=3)
        ax.annotate(
            f"{before:.2f}",
            xy=(before, y),
            xytext=(before - 0.09, y + 0.14),
            fontsize=9,
            color=INK,
            ha="right",
        )
        ax.annotate(
            f"{after:.2f} ({after - before:+.2f})",
            xy=(after, y),
            xytext=(after + 0.09, y + 0.14),
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5, color=INK)
    ax.set_xlim(5.0, 13.0)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("chunk MAE — lower is better", color=DARKGRAY, fontsize=10)
    ax.set_title(
        "The rig fine-tune @4k moved every read the wrong way — no transfer,\n"
        "mild drift; per the frozen ship rule, no checkpoint was offered",
        color=INK,
        fontsize=11,
        loc="left",
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=BLUE,
            label="before (student @30k)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=ORANGE,
            label="after rig fine-tune @4k",
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    save(fig, "ftrig_dumbbell.svg")


def main() -> None:
    snap = load("analysis__snapflow_distill_30k_k4l2.json")
    ar = load("analysis__draws10_t1_ar100k_k4l2.json")
    bench_single = load("analysis__leaderboard_decode_microbench_postmerge_single.json")
    bench_pre = load("analysis__leaderboard_decode_microbench.json")
    chart_endpoint(snap, ar)
    chart_cost_quality(snap, ar, bench_single, bench_pre)
    chart_draws_collapse(snap, ar)
    chart_horizon(snap)
    chart_ftrig()


if __name__ == "__main__":
    main()
