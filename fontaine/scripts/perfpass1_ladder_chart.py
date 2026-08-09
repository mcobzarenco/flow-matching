"""Perf pass-1 box ladder chart — median step time A/B/C vs the frozen
>=5% decision bar (analysis__perfpass1_box_ladder.json, banked 02:3xZ
08-09).

Output: fontaine/blog/src/img/perfpass1/box_ladder.svg. House
eval-report dark theme (page #121417, blue = baseline HEAD, amber =
regresses vs baseline; standing owner rule: dark-mode friendly).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/perfpass1"
SRC = ROOT / "reports/analysis__perfpass1_box_ladder.json"

PAGE = "#121417"
BLUE = "#648fff"  # baseline HEAD
AMBER = "#ffb000"  # regresses vs baseline
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


def main() -> None:
    data = json.loads(SRC.read_text())
    a = data["bench"]["perfpass1_box_bench_A"]["median_s_per_step_tail"]
    b = data["bench"]["perfpass1_box_bench_B"]["median_s_per_step_tail"]
    c = data["bench"]["perfpass1_box_bench_C"]["median_s_per_step_tail"]
    rows = [
        ("A — HEAD (baseline)", a, BLUE, ""),
        ("B — +P1 suffix cuDNN", b, AMBER, f"{(a - b) / a * 100:+.1f}%"),
        ("C — full bundle P1–P4", c, AMBER, f"{(a - c) / a * 100:+.1f}%"),
    ]
    bar = a * 0.95  # the frozen landing bar: >=5% FASTER than A

    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=110)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    y = range(len(rows))
    # Dot plot, not bars: the axis is deliberately non-zero (the whole
    # story lives in a ~0.25s band), and truncated-axis BARS would
    # exaggerate — dots stay honest.
    for index, (_, value, color, _) in enumerate(rows):
        ax.plot(
            [value],
            [index],
            marker="o",
            markersize=10,
            color=color,
            zorder=4,
        )
    ax.axvline(bar, color=META, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.annotate(
        f"landing bar: A −5% = {bar:.3f}s — C had to be left of this line",
        xy=(bar - 0.008, 2.55),
        color=META,
        fontsize=8.5,
        ha="right",
        va="center",
        annotation_clip=False,
    )
    for index, (_, value, _, delta) in enumerate(rows):
        label = f"{value:.3f}s" + (f"   {delta} vs A" if delta else "")
        ax.annotate(
            label,
            xy=(value + 0.018, index),
            color=TEXT,
            fontsize=9.5,
            va="center",
            zorder=4,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], color=TEXT, fontsize=9.5)
    ax.set_ylim(-0.7, 2.8)
    ax.invert_yaxis()
    ax.set_xlim(2.05, 2.66)
    ax.set_xlabel(
        "median s/step (320-step rung, tail past step 80; box 4xDDP true recipe)",
        color=META,
        fontsize=9.5,
    )
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Perf pass-1 on the real recipe: the bundle is slower, not faster",
        color=TEXT,
        fontsize=11.5,
        loc="left",
        pad=12,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "box_ladder.svg",
        facecolor=PAGE,
        bbox_inches="tight",
    )
    print(f"wrote {OUT / 'box_ladder.svg'}")


if __name__ == "__main__":
    main()
