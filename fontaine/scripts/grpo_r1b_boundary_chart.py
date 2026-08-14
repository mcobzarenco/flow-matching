"""Boundary chart for token-GRPO phase 2 R1-B (queue
`grpo-r1b-boundary-reads`; eval-report dark scheme, IBM CVD-safe hues).
Banked numbers only, read from the two runs' frozen train.jsonl files.
Three panels: the knock-away wire story across R1-A and R1-B; the v2
reward's earned/shoved decomposition (the patch's own prediction); and
the held-out paired-delta probe (v1 metric) that stayed flat through
both runs.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
R1A, R1B, SHOVE, WIRE = "#648fff", "#08bdba", "#dc267f", "#ffb000"
BASELINE, WIRE_LINE = 0.0833, 0.1667
PROBE_BASE = 1.868


def _rows(path: Path) -> tuple[list[dict], list[dict]]:
    train, evals = [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "loss" in row:
            train.append(row)
        elif row.get("eval_reward_mean") is not None:
            evals.append(row)
    return train, evals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/grpo_phase2_b/chart__grpo_r1b_boundary.png"),
    )
    args = ap.parse_args()

    a_train, a_evals = _rows(Path("outputs/sim/grpo_phase2_a/train.jsonl"))
    b_train, b_evals = _rows(Path("outputs/sim/grpo_phase2_b/train.jsonl"))

    fig, (ax, bx, cx) = plt.subplots(
        1,
        3,
        figsize=(14.6, 4.9),
        width_ratios=[1.15, 1, 0.85],
    )
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, bx, cx):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
        panel.set_axisbelow(True)
        panel.set_xlabel("trainer step (fresh wave)", color=META, fontsize=9)

    # A: the knock-away wire fired under BOTH rewards
    for rows, color, marker, name in (
        (a_train, R1A, "o", "R1-A (v1 reward, lr 1e-6)"),
        (b_train, R1B, "s", "R1-B (v2 reward, lr 3e-7)"),
    ):
        xs = [r["step"] for r in rows]
        ys = [r["knockaway_frac"] for r in rows]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=6,
            markerfacecolor=color,
            markeredgecolor=PAGE,
            markeredgewidth=1,
        )
        dy = -18 if name.startswith("R1-A") else 10
        ax.annotate(
            name,
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(0, dy),
            color=color,
            fontsize=8.5,
            ha="right",
        )
    for rows in (a_train, b_train):
        tip = rows[-1]
        ax.plot(
            tip["step"],
            tip["knockaway_frac"],
            marker="o",
            markersize=13,
            markerfacecolor="none",
            markeredgecolor=WIRE,
            markeredgewidth=1.6,
        )
    ax.axhline(WIRE_LINE, color=WIRE, linewidth=1.2, linestyle="--", alpha=0.9)
    ax.axhline(BASELINE, color=META, linewidth=0.9, linestyle=":", alpha=0.8)
    ax.text(
        0.62,
        WIRE_LINE + 0.008,
        "tripwire 0.167 (2× probe baseline)",
        color=WIRE,
        fontsize=8.5,
    )
    ax.text(0.62, BASELINE + 0.008, "probe baseline 0.083", color=META, fontsize=8.5)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("knockaway_frac (endpoint ≤ −1 cm)", color=META, fontsize=9)
    ax.set_title(
        "the wire fired under both rewards\n(rings = 3rd straight step above, run self-stops)",
        color=TEXT,
        fontsize=10,
        loc="left",
    )

    # B: R1-B's v2 decomposition vs the patch's prediction
    xs = [r["step"] for r in b_train]
    earned = [r["earned_progress_mean"] for r in b_train]
    shoved = [r["ungrasped_disp_mean"] for r in b_train]
    bx.plot(
        xs,
        earned,
        color=R1B,
        linewidth=2,
        marker="s",
        markersize=6,
        markeredgecolor=PAGE,
        markeredgewidth=1,
    )
    bx.plot(
        xs,
        shoved,
        color=SHOVE,
        linewidth=2,
        marker="^",
        markersize=7,
        markeredgecolor=PAGE,
        markeredgewidth=1,
    )
    bx.annotate(
        "shoved (ungrasped |Δd|, charged)",
        (xs[0], shoved[0]),
        textcoords="offset points",
        xytext=(6, 12),
        color=SHOVE,
        fontsize=8.5,
        ha="left",
    )
    bx.annotate(
        "earned (pinched progress, paid)",
        (xs[-1], earned[-1]),
        textcoords="offset points",
        xytext=(0, -16),
        color=R1B,
        fontsize=8.5,
        ha="right",
    )
    bx.set_ylim(0, 5.6)
    bx.set_xticks(xs)
    bx.set_ylabel("cm per episode (wave mean)", color=META, fontsize=9)
    bx.set_title(
        "v2 charged the shove — it shrank a little,\nbut earned progress collapsed instead",
        color=TEXT,
        fontsize=10,
        loc="left",
    )

    # C: the held-out probe never moved (paired Δ, v1 metric)
    for evals, color, marker in ((a_evals, R1A, "o"), (b_evals, R1B, "s")):
        pts = [r for r in evals if r.get("eval_delta_mean") is not None]
        xs = [r["step"] for r in pts]
        ys = [r["eval_delta_mean"] for r in pts]
        lo = [r["eval_delta_mean"] - r["eval_delta_ci_lo"] for r in pts]
        hi = [r["eval_delta_ci_hi"] - r["eval_delta_mean"] for r in pts]
        cx.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=color,
            fmt=marker,
            markersize=6,
            markeredgecolor=PAGE,
            markeredgewidth=1,
            capsize=3,
            linewidth=0,
            elinewidth=1.4,
        )
    cx.axhline(0.0, color=META, linewidth=0.9, linestyle=":", alpha=0.8)
    cx.set_ylabel(f"paired Δ vs step-0 policy ({PROBE_BASE})", color=META, fontsize=9)
    cx.set_title(
        "held-out probe: flat through both runs\n(CI95 straddles 0 at every step)",
        color=TEXT,
        fontsize=10,
        loc="left",
    )

    handles = [
        Line2D(
            [],
            [],
            color=R1A,
            marker="o",
            linewidth=2,
            markeredgecolor=PAGE,
            label="R1-A (v1 reward)",
        ),
        Line2D(
            [],
            [],
            color=R1B,
            marker="s",
            linewidth=2,
            markeredgecolor=PAGE,
            label="R1-B (v2 reward, resumed from R1-A step 4)",
        ),
        Line2D(
            [],
            [],
            color=SHOVE,
            marker="^",
            linewidth=2,
            markeredgecolor=PAGE,
            label="ungrasped displacement (v2-charged)",
        ),
        Line2D(
            [],
            [],
            color=WIRE,
            linewidth=1.2,
            linestyle="--",
            label="knock-away tripwire",
        ),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    for text in leg.get_texts():
        text.set_color(TEXT)
    fig.suptitle(
        "token-GRPO phase 2, R1-B boundary — shoving is not reward-driven at this surface",
        color=TEXT,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
