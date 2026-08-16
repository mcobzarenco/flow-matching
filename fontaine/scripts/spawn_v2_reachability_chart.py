"""Chart the spawn-v2 reachability probe fields (instrument §3 of the
spawn-v2 pre-reg DRAFT): panel A the IK-residual field binned at the
candidate mask bars, panel B the static shoulder-moment fraction — the
visual evidence that the posture-pulled solves keep static torque far
from the 3.478 wall (max ~0.25), leaving residual the binding
constraint. Overlays: the v1 spawn band, the v1 disk, the pan axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Rectangle

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
# Sequential ramps (one hue each, monotone lightness on the dark
# surface; brightest = the pole the eye should find first).
TEAL_BINS = ["#7deee2", "#2fb3a8", "#1d6f68", "#16393c"]  # residual: best->worst
BLUE_BINS = ["#a6c8ff", "#648fff", "#3155b8", "#1e2f57"]  # moment: low->high
V1_BAND = "#ffb000"

RESIDUAL_EDGES = [0.0, 1e-3, 3e-3, 1e-2, np.inf]
RESIDUAL_LABELS = ["< 1 mm (mask bar)", "1–3 mm", "3–10 mm", "≥ 10 mm"]
MOMENT_EDGES = [0.0, 0.10, 0.15, 0.20, 1.0]
MOMENT_LABELS = ["< 0.10", "0.10–0.15", "0.15–0.20", "≥ 0.20"]


def field(cells: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array(sorted({c["x"] for c in cells}))
    ys = np.array(sorted({c["y"] for c in cells}))
    grid = np.full((len(ys), len(xs)), np.nan)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    for c in cells:
        grid[yi[c["y"]], xi[c["x"]]] = c[key]
    return xs, ys, grid


def panel(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    grid: np.ndarray,
    bins: list[str],
    edges: list[float],
    labels: list[str],
    title: str,
) -> None:
    cmap = ListedColormap(bins)
    norm = BoundaryNorm(edges, cmap.N)
    ax.pcolormesh(xs, ys, grid, cmap=cmap, norm=norm, shading="nearest")
    ax.set_facecolor(PAGE)
    ax.set_aspect("equal")
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("x (m)", color=META, fontsize=9)
    ax.set_ylabel("y (m)", color=META, fontsize=9)
    ax.tick_params(colors=META, labelsize=8)
    for side in ax.spines.values():
        side.set_color(GRID)
    # v1 overlays: spawn band, disk (radius 0.04), pan axis
    ax.add_patch(
        Rectangle(
            (0.195, -0.005),
            0.075,
            0.045,
            fill=False,
            edgecolor=V1_BAND,
            linewidth=1.6,
        ),
    )
    ax.add_patch(
        Circle((0.22, 0.11), 0.04, fill=False, edgecolor=TEXT, linewidth=1.4),
    )
    ax.plot([0.0388], [0.0], marker="+", color=TEXT, markersize=9)
    ax.annotate(
        "v1 spawn band",
        (0.27, -0.005),
        (0.30, -0.06),
        color=V1_BAND,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": V1_BAND, "lw": 0.8},
    )
    ax.annotate(
        "v1 disk",
        (0.22, 0.15),
        (0.13, 0.21),
        color=TEXT,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": META, "lw": 0.8},
    )
    ax.annotate("pan axis", (0.045, 0.005), color=TEXT, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in bins]
    leg = ax.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=7.5,
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(0.8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--probe",
        type=Path,
        default=Path("reports/analysis__spawn_v2_reachability_v0.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/spawn_v2/chart__spawn_v2_reachability_v0.png"),
    )
    args = ap.parse_args()
    data = json.loads(args.probe.read_text())
    cells = data["cells"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.patch.set_facecolor(PAGE)
    xs, ys, res = field(cells, "residual")
    panel(
        ax,
        xs,
        ys,
        res,
        TEAL_BINS,
        RESIDUAL_EDGES,
        RESIDUAL_LABELS,
        "Grasp-pose IK residual — the reachable field",
    )
    _, _, mom = field(cells, "moment_frac_shoulder")
    panel(
        bx,
        xs,
        ys,
        mom,
        BLUE_BINS,
        MOMENT_EDGES,
        MOMENT_LABELS,
        "Static shoulder moment (fraction of the 3.478 servo limit)",
    )
    n_ok = sum(c["residual"] < 1e-3 for c in cells)
    fig.suptitle(
        "Spawn-v2 reachability probe v0 — stage-A IK + sysid statics as the "
        "workspace instrument",
        color=TEXT,
        fontsize=12.5,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.015,
        f"{n_ok} of {len(cells)} cells inside the 1 mm bar "
        f"(~{n_ok} cm² vs the v1 band's 34 cm²); "
        "max shoulder moment over the reachable field 0.25 of the limit — "
        "residual, not torque, binds. Grid 1 cm, grasp z 0.014 m, radial-hull "
        "yaw; head " + data["head"],
        color=META,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"[chart] -> {args.out}")


if __name__ == "__main__":
    main()
