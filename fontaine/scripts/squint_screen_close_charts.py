"""Squint qualification-screen close-out charts (results post, house
eval-report scheme, dark-mode) — consumes the Gate-1 cell JSONs the
verdict consumed (`outputs/squint_screen/eval/adapt_onerig_*.json`)
and recounts milestone attainment from the per-rollout predicate
rows, asserting the success counts match `gate1.log` before drawing.

Outputs (fontaine/blog/src/img/squint_screen/):
  gate1_milestones.svg  per-task milestone ladder, ever-true /100 —
                        partial competence up the ladder, 0 at success
  adapt_twins.svg       probe curves of both adapted arms — the
                        positive control trained fine; rollouts zeroed

Usage: uv run python fontaine/scripts/squint_screen_close_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/squint_screen"
EVAL = ROOT / "outputs/squint_screen/eval"
CKPT = Path.home() / "checkpoints/finetune"

PAGE = "#121417"
BLUE = "#648fff"  # onerig (the stronger arm / primary)
MAGENTA = "#dc267f"  # democlean
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

GATE1_BAR = 20  # >=20/100 best task, frozen in the pre-reg

MILESTONES = {
    "lift": ["reached_object", "is_item_grasped", "item_lifted", "success"],
    "place": ["is_item_grasped", "item_lifted", "is_item_above_bin", "success"],
}
SHORT = {
    "reached_object": "reached\nobject",
    "is_item_grasped": "grasped",
    "item_lifted": "lifted",
    "is_item_above_bin": "above\nbin",
    "success": "success",
}


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.svg")


def load_task(task: str) -> tuple[list[dict], int]:
    d = json.loads((EVAL / f"adapt_onerig_{task}.json").read_text())
    return d["rows"], int(d["successes"])


def gate1_milestones() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    fig.patch.set_facecolor(PAGE)
    for ax, task in zip(axes, ("lift", "place"), strict=True):
        rows, _ = load_task(task)
        counts = [
            sum(1 for r in rows if any(r["predicates"][m])) for m in MILESTONES[task]
        ]
        xs = range(len(counts))
        ax.bar(xs, counts, width=0.62, color=BLUE, zorder=3)
        for x, c in zip(xs, counts, strict=True):
            ax.annotate(
                f"{c}/100",
                (x, c),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                color=TEXT,
                fontsize=9.5,
            )
        ax.axhline(GATE1_BAR, color=MAGENTA, linewidth=1.4, zorder=2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([SHORT[m] for m in MILESTONES[task]], color=META, fontsize=9)
        ax.set_title(f"{task}  (adapted onerig, n=100)", color=META, fontsize=9)
        style_axis(ax)
    axes[0].set_ylabel("rollouts ever attaining (/100)", color=META, fontsize=9)
    axes[0].set_ylim(0, 30)
    axes[1].annotate(
        f"Gate-1 bar: ≥{GATE1_BAR}/100 success, best task",
        (0.02, GATE1_BAR),
        textcoords="offset points",
        xytext=(0, 5),
        color=MAGENTA,
        fontsize=8.5,
    )
    fig.suptitle(
        "Gate-1 positive control: partial competence up the ladder, zero at success",
        color=TEXT,
        fontsize=11,
        y=1.03,
    )
    save(fig, "gate1_milestones")


def adapt_twins() -> None:
    arms = (
        ("adapt onerig", "grasp_sft_v2_squint_adapt_onerig", BLUE),
        ("adapt democlean", "grasp_sft_v2_squint_adapt_democlean", MAGENTA),
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor(PAGE)
    for (label, run, color), dy in zip(arms, (-9, 9), strict=True):
        rows = [
            json.loads(line)
            for line in (CKPT / run / "train_log.jsonl").read_text().splitlines()
        ]
        pts = [(r["step"], r["eval_chunk_mae"]) for r in rows if "eval_chunk_mae" in r]
        steps, mae = zip(*pts, strict=True)
        ax.plot(steps, mae, color=color, linewidth=2, marker="o", markersize=5)
        ax.annotate(
            f"{label}  {mae[-1]:.2f}",
            (steps[-1], mae[-1]),
            textcoords="offset points",
            xytext=(8, dy),
            va="center",
            color=TEXT,
            fontsize=9,
        )
    ax.set_xlim(80, 660)
    ax.set_xlabel(
        "adaptation step (frozen recipe, identical both arms)",
        color=META,
        fontsize=9,
    )
    ax.set_ylabel("holdout eval_chunk_mae (twin demos)", color=META, fontsize=9)
    style_axis(ax)
    ax.set_title(
        "Adaptation itself worked — twin curves, monotone down; rollouts still 0/100",
        color=TEXT,
        fontsize=11,
    )
    save(fig, "adapt_twins")


def main() -> int:
    gate1 = json.loads((EVAL / "gate1.log").read_text().splitlines()[0])
    for task in ("lift", "place"):
        rows, successes = load_task(task)
        recounted = sum(1 for r in rows if any(r["predicates"]["success"]))
        if not (len(rows) == 100 and successes == recounted == gate1["cells"][task]):
            sys.exit(
                f"ABORT: {task} n={len(rows)} successes={successes} "
                f"recount={recounted} vs gate1 {gate1['cells'][task]}",
            )
    gate1_milestones()
    adapt_twins()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
