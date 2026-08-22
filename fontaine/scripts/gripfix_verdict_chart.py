"""Gripfix verdict chart — sim100 successes across the isolation-cell family.

One chart for the gripper-carrier pre-reg's results append: unseen-100
success counts for the five protocol-matched cells, drawn against the
frozen verdict bands, with the paired read vs democlean (THE read)
annotated on the subject bar. Counts are re-read from the banked
flow_unseen JSONs at draw time and abort on mismatch with the paired
JSONs (chart-recomputes-the-recipe house rule).

Dark-mode, eval-report scheme (house constants from
democlean_probe_curve_chart.py); entity colors follow the lineage
roles across posts (subject blue, onerig purple, convicted magenta),
identity carried by the axis labels, values direct-labeled.

Usage:
  uv run python -m fontaine.scripts.gripfix_verdict_chart \
      --out-png fontaine/blog/src/img/gripfix/verdict_columns.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
HEADING = "#eceef1"
BLUE = "#648fff"  # subject: demos + clean_gripfix (ch5 remapped)
PURPLE = "#785ef0"  # onerig recovery yardstick
MAGENTA = "#dc267f"  # convicted three-way cell
AMBER = "#ffb000"  # democlean: the paired comparator (THE read)

SIM = Path("outputs/sim/grasp_sft")
CELLS = [
    # (label, json path, color)
    ("demos + v2\n(onerig)", SIM / "onerig_endpoint/flow_unseen.json", PURPLE),
    ("demos only\n(control)", SIM / "disc1000_baseline/flow_unseen.json", META),
    ("demos + clean\n(democlean)", SIM / "democlean_endpoint/flow_unseen.json", AMBER),
    (
        "demos + clean_gripfix\n(THIS CELL)",
        SIM / "gripfix_endpoint/flow_unseen.json",
        BLUE,
    ),
    (
        "demos + v2 + clean\n(convicted)",
        SIM / "pdnorm_endpoint/flow_unseen.json",
        MAGENTA,
    ),
]
EXPECTED = {"onerig": 28, "control": 11, "democlean": 8, "convicted": 1}
PAIRED_VS_DEMOCLEAN = Path("reports/analysis__sim100_paired_gripfix_vs_democlean.json")


def successes(path: Path) -> int:
    d = json.loads(path.read_text())
    return sum(1 for e in d["episodes"] if e["success_tick"] is not None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-png",
        default="fontaine/blog/src/img/gripfix/verdict_columns.png",
    )
    args = ap.parse_args()

    counts = [successes(p) for _, p, _ in CELLS]
    # Oracle: anchors must match their banked values.
    banked = dict(
        zip(
            ("onerig", "control", "democlean", "gripfix", "convicted"),
            counts,
            strict=True,
        ),
    )
    for k, v in EXPECTED.items():
        if banked[k] != v:
            msg = f"anchor mismatch: {k} re-read {banked[k]} vs banked {v}"
            raise SystemExit(msg)
    paired = json.loads(PAIRED_VS_DEMOCLEAN.read_text())["read"]
    if paired["success"]["count_a"] != banked["gripfix"]:
        raise SystemExit("paired JSON disagrees with flow_unseen re-read")

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)

    # Frozen verdict bands on the count axis.
    ax.axvspan(0, 10.5, color=MAGENTA, alpha=0.06, zorder=0)
    ax.axvspan(10.5, 19.5, color=META, alpha=0.06, zorder=0)
    ax.axvspan(19.5, 32, color=PURPLE, alpha=0.06, zorder=0)
    for x, s in (
        (5.2, "≤10 NOT sole carrier"),
        (15.0, "11–19 ambiguous"),
        (25.8, "≥20 carrier confirmed"),
    ):
        ax.text(x, -0.42, s, ha="center", va="center", color=META, fontsize=7)

    ys = range(len(CELLS))
    for y, ((_label, _, color), n) in enumerate(zip(CELLS, counts, strict=True)):
        ax.barh(y, n, height=0.52, color=color, zorder=3)
        ax.text(n + 0.4, y, f"{n}", va="center", color=TEXT, fontsize=9)
    ax.text(
        counts[3] + 2.2,
        3.02,
        "Δ vs democlean −3 [−8, +1], McNemar p = 0.375\npaired Δprogress −2.07 cm [−3.03, −1.12]",
        va="center",
        color=META,
        fontsize=7,
    )

    ax.set_yticks(list(ys), [c[0] for c in CELLS])
    ax.invert_yaxis()
    ax.set_xlim(0, 32)
    ax.set_ylim(4.7, -0.55)
    ax.set_xlabel(
        "sim100 successes / 100 (unseen seeds 0–99, step 3000)",
        color=TEXT,
        fontsize=9,
    )
    ax.set_title(
        "The gripper remap does not de-poison: 5/100 in the ≤10 band",
        color=HEADING,
        fontsize=10.5,
        pad=10,
    )
    ax.tick_params(colors=META, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=1)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
