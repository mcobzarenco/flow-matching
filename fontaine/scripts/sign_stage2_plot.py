"""Render the stage-2 sign-convention population figure (static SVG).

Reads the probe's report JSON (`~/sign_stage2_results.json`) and draws,
per target dim, the 15-repo reference population's Spearman rho values as
a de-emphasized strip, with the candidate / control / record-only cell
reads overlaid as emphasized marks. Emphasis form (one accent + gray
context); categorical slots 1-3 only (validated all-pairs in light
mode); text in ink tokens, never series colors.

Run: uv run python fontaine/scripts/sign_stage2_plot.py
Writes: fontaine/blog/src/assets/sign_convention_stage2_populations.svg
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPORT = Path.home() / "sign_stage2_results.json"
OUT = (
    Path(__file__).resolve().parents[1]
    / "blog/src/assets/sign_convention_stage2_populations.svg"
)

# Reference palette (light mode): ink + first three categorical slots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"  # slot 1: candidate cells
ORANGE = "#eb6834"  # slot 2: control cells
AQUA = "#1baf7a"  # slot 3: record-only cells

DIM_ORDER = [
    "main_wrist_flex",
    "main_wrist_roll",
    "main_shoulder_lift",
    "main_shoulder_pan",
]
ROLE_STYLE = {
    "candidate": (BLUE, "D", 9),
    "control": (ORANGE, "^", 10),
    "record-only": (AQUA, "s", 7),
}


def main() -> None:
    report = json.loads(REPORT.read_text())
    populations = report["populations"]
    gates = report["population_gates"]
    # Candidate cells never opened when the hard gate fails (the
    # escalation branch): the figure then shows populations only.
    cells = [c for c in report.get("cells", {}).get("cells", []) if "rho" in c]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for row, dim in enumerate(DIM_ORDER):
        y = len(DIM_ORDER) - 1 - row
        rhos = [r["rho"] for r in populations[dim]]
        ax.scatter(
            rhos,
            [y] * len(rhos),
            s=34,
            color=MUTED,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        gate = gates[dim]
        ax.plot(
            [gate["median_rho"]] * 2,
            [y - 0.22, y + 0.22],
            color=INK_2,
            lw=1.6,
            zorder=3,
        )
        for cell in cells:
            if cell["dim"] != dim:
                continue
            color, marker, size = ROLE_STYLE[cell["role"]]
            ax.scatter(
                [cell["rho"]],
                [y],
                s=size**2,
                color=color,
                marker=marker,
                edgecolors=SURFACE,
                linewidths=1.2,
                zorder=4,
            )
            short = cell["repo"].split("/")[1]
            label = f"{short[:22]} · {cell['verdict']}"
            ax.annotate(
                label,
                (cell["rho"], y),
                textcoords="offset points",
                xytext=(0, 11 if cell["role"] != "control" else -17),
                ha="center",
                fontsize=7.2,
                color=INK,
            )
        gate_note = "" if gate["valid"] else "  (population INVALID)"
        ax.annotate(
            f"n={gate['n']}, {gate['agree']}/{gate['n']} agree, "
            f"median {gate['median_rho']:+.2f}{gate_note}",
            (1.02, y),
            xycoords=("axes fraction", "data"),
            fontsize=7.2,
            color=INK_2,
            va="center",
        )

    ax.axvline(0.0, color=BASELINE, lw=1.0, zorder=1)
    ax.set_yticks(range(len(DIM_ORDER)))
    ax.set_yticklabels(
        [d.removeprefix("main_") for d in reversed(DIM_ORDER)],
        fontsize=9,
        color=INK,
    )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.6, len(DIM_ORDER) - 0.4 + 0.35)
    ax.set_xlabel(
        "Spearman rho (state velocity vs flow statistic, ego cam)",
        fontsize=9,
        color=INK_2,
    )
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for t in ax.get_yticklabels():
        t.set_color(INK)

    ax.set_title(
        "Stage-2 sign probe: reference populations vs candidate cells",
        fontsize=11,
        color=INK,
        loc="left",
        pad=14,
    )
    plotted_roles = {c["role"] for c in cells}
    handles = [
        Line2D(
            [],
            [],
            marker=m,
            color=c,
            linestyle="",
            markersize=7,
            label=role,
            markeredgecolor=SURFACE,
        )
        for role, (c, m, _) in ROLE_STYLE.items()
        if role in plotted_roles
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            color=MUTED,
            alpha=0.55,
            linestyle="",
            markersize=6,
            label="reference repo",
        ),
        Line2D([], [], color=INK_2, lw=1.6, label="population median"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.2,
        frameon=False,
        ncol=2,
        labelcolor=INK_2,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg", facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
