"""Panel anchor-ladder chart for the pdnorm endpoint read (PRE-GO prep).

Renders the wear-audit panel ladder (pre-reg
posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md, audit banked
06:4xZ 08-18) as a labeled horizontal-bar rung figure: raw disc-1000
58.14 (worn demos global table), re-worn 27.40 (same-model
wear-corrected reference), released pre-SFT re-worn 27.14 (same
honest rows, re-expressed 09:xxZ 08-18 from the banked own-table
25.89 — released_row_rewear.py), repo-midpoint null 25.15
(carries-any-signal bar), worn-box clamp floor 14.40, state-copy
8.37 (the real bar) —
plus a FILL-AT-ENDPOINT slot for the pdnorm endpoint row, stamped via
--endpoint on GO. House eval-report dark scheme (page #121417; blue =
measured rows, amber = the real bar, magenta = the stamped endpoint,
gray = nulls/floors).

Usage:
  uv run python fontaine/scripts/pdnorm_panel_ladder_chart.py \
      [--endpoint 23.5] \
      [--out-png fontaine/blog/src/img/pdnorm/panel_ladder.png] \
      [--out-b64 reports/pdnorm_panel_ladder.b64]
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_PNG = ROOT / "fontaine/blog/src/img/pdnorm/panel_ladder.png"
OUT_B64 = ROOT / "reports/pdnorm_panel_ladder.b64"

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
HEADING = "#eceef1"
BLUE = "#648fff"  # measured model rows
AMBER = "#ffb000"  # state-copy: the real bar
MAGENTA = "#dc267f"  # the pdnorm endpoint, once stamped
GRAY = "#9aa0a8"  # nulls / floors

# The frozen ladder (pre-reg calibration note, wear-corrected class).
# (label, value, color, note) — order is the render order, top rung
# first. The endpoint slot is prepended by build_rungs.
LADDER: list[tuple[str, float, str, str]] = [
    ("disc-1000 raw (worn demos global table)", 58.14, BLUE, "wear-mismatched row"),
    (
        "disc-1000 re-worn (honest per-repo rows)",
        27.40,
        BLUE,
        "same-model wear-corrected reference",
    ),
    (
        "released pre-SFT re-worn (same honest rows)",
        27.14,
        BLUE,
        "own-table 25.89; same-wear: SFT ended where it started",
    ),
    ("repo-midpoint null (constant)", 25.15, GRAY, "carries-any-signal bar"),
    ("worn-box clamp floor", 14.40, GRAY, "clamp floor of the 58.14"),
    ("state-copy", 8.37, AMBER, "the real bar for usable-on-real-data"),
]

ENDPOINT_LABEL = "pdnorm endpoint step3000"
PENDING_NOTE = "FILL-AT-ENDPOINT — stamped on GO"


def build_rungs(
    endpoint: float | None,
) -> list[tuple[str, float | None, str, str]]:
    """The render rows: the endpoint slot (pending or stamped) + LADDER."""
    slot: tuple[str, float | None, str, str] = (
        (ENDPOINT_LABEL, None, META, PENDING_NOTE)
        if endpoint is None
        else (ENDPOINT_LABEL, endpoint, MAGENTA, "this run — read vs the class above")
    )
    return [slot, *LADDER]


def render(endpoint: float | None, out_png: Path, out_b64: Path) -> None:
    rungs = build_rungs(endpoint)
    xmax = 64.0
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=140)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    ys = list(range(len(rungs), 0, -1))
    for y, (_, value, color, note) in zip(ys, rungs, strict=True):
        if value is None:
            # Pending slot: a dashed full-width outline, deliberately NOT
            # a bar — it must not read as a value.
            ax.add_patch(
                FancyBboxPatch(
                    (0.4, y - 0.3),
                    xmax - 1.4,
                    0.6,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    linewidth=1.1,
                    linestyle=(0, (5, 3)),
                    edgecolor=META,
                    facecolor="none",
                    zorder=3,
                ),
            )
            ax.annotate(
                note,
                xy=(xmax / 2, y),
                color=META,
                fontsize=9,
                ha="center",
                va="center",
                zorder=4,
            )
            continue
        ax.barh(y, value, height=0.6, color=color, zorder=3)
        ax.annotate(
            f"{value:.2f}",
            xy=(value + 0.7, y + 0.12),
            color=TEXT,
            fontsize=9.5,
            fontweight="bold",
            va="center",
            zorder=4,
        )
        ax.annotate(
            note,
            xy=(value + 0.7, y - 0.21),
            color=META,
            fontsize=8,
            va="center",
            zorder=4,
        )
    # The two decision rungs as faint guides across the ladder.
    for guide, color in ((25.15, GRAY), (8.37, AMBER)):
        ax.axvline(
            guide,
            color=color,
            linestyle=(0, (3, 4)),
            linewidth=0.9,
            alpha=0.45,
            zorder=2,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rungs], color=TEXT, fontsize=9)
    ax.set_ylim(0.3, len(rungs) + 0.7)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("panel pooled core-chunk MAE (lower is better)", fontsize=9)
    ax.xaxis.label.set_color(META)
    ax.tick_params(colors=META, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title(
        "pdnorm endpoint — panel anchor ladder, wear-corrected class",
        color=HEADING,
        fontsize=11,
        pad=10,
    )
    fig.text(
        0.99,
        0.01,
        "panel v2 k4l2 · euler-10 · wear audit 08-18 — the endpoint reads"
        " against the wear-corrected class, never the raw 58.14",
        color=META,
        fontsize=7.5,
        ha="right",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    out_b64.parent.mkdir(parents=True, exist_ok=True)
    out_b64.write_text(base64.b64encode(out_png.read_bytes()).decode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=float, default=None)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--out-b64", type=Path, default=OUT_B64)
    args = parser.parse_args()
    render(args.endpoint, args.out_png, args.out_b64)
    print(f"wrote {args.out_png} + {args.out_b64}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
