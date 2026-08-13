"""Results chart for the contact-shadow light fit
(fit_contact_shadow.py output; eval-report dark scheme). Two panels:
the direction landscape (shadow-vs-ring darkening contrast across
azimuth, one lightness step per zenith — ordered series, sequential
single-hue ramp, direct-labeled) and the softness correlation curve,
with the fitted constants and their CIs in the header block.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
# Sequential blue steps, light -> dark with zenith; amber = the winner.
ZENITH_RAMP = {10: "#a6c8ff", 20: "#78a9ff", 30: "#4589ff", 40: "#2b5fd9"}
ACCENT = "#ffb000"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fit",
        type=Path,
        default=Path("reports/analysis__contact_shadow_fit.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/shadow_fit/chart__contact_shadow_fit.png"),
    )
    args = ap.parse_args()
    r = json.loads(args.fit.read_text())

    # The coarse grid only (the refine pass appends off-grid directions
    # that would draw as partial series).
    landscape: dict[float, list[tuple[float, float]]] = {}
    for key, value in r["contrast"]["per_direction_coarse"].items():
        zen, az = (float(g) for g in re.match(r"zen(.+)_az(.+)", key).groups())
        if zen == 0.0 or (zen in (10.0, 20.0, 30.0, 40.0) and az % 30 == 0):
            landscape.setdefault(zen, []).append((az, value))

    fig, (ax, sig_ax) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.6),
        width_ratios=[2.4, 1.0],
    )
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, sig_ax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.set_axisbelow(True)
        panel.yaxis.grid(True, color=GRID, linewidth=0.8)

    zen0 = landscape.pop(0.0, None)
    if zen0:
        ax.axhline(
            zen0[0][1],
            color=META,
            linewidth=1.2,
            linestyle="--",
        )
        ax.annotate(
            f"straight down: {zen0[0][1]:+.3f}",
            xy=(357, zen0[0][1]),
            ha="right",
            va="bottom",
            fontsize=8,
            color=META,
        )
    ax.axhline(0.0, color=GRID, linewidth=1.0)
    for zen in sorted(landscape):
        pts = sorted(landscape[zen])
        az = [a for a, _ in pts]
        ax.plot(
            az,
            [c for _, c in pts],
            color=ZENITH_RAMP.get(int(zen), TEXT),
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )
        ax.annotate(
            f"zenith {zen:g}°",
            xy=(az[-1] + 4, pts[-1][1]),
            fontsize=8,
            color=ZENITH_RAMP.get(int(zen), TEXT),
            va="center",
        )
    d = r["direction"]
    best = (d["zenith_deg"], d["azimuth_deg"])
    ax.plot(
        best[1],
        r["contrast"]["mean"],
        marker="*",
        markersize=15,
        color=ACCENT,
        markeredgecolor=PAGE,
        zorder=5,
    )
    ci = r["contrast"]["ci95"]
    ax.annotate(
        f"fit: zenith {best[0]:g}° azimuth {best[1]:g}°\n"
        f"contrast {r['contrast']['mean']:+.3f} CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
        xy=(best[1], r["contrast"]["mean"]),
        xytext=(best[1] + 55, r["contrast"]["mean"] - 0.004),
        ha="left",
        va="top",
        fontsize=8.5,
        color=ACCENT,
    )
    ax.set_xlabel(
        "light azimuth (deg, 0 = +x world, toward +y)",
        color=META,
        fontsize=9,
    )
    ax.set_ylabel("darkening contrast, shadow − ring", color=META, fontsize=9)
    ax.set_xlim(-8, 400)
    ax.set_title(
        "Where is the light? predicted-shadow vs ring-control darkening",
        color=TEXT,
        fontsize=10,
        loc="left",
    )

    sig = r["softness"]["correlation_by_sigma"]
    pairs = sorted((float(k), v) for k, v in sig.items())
    sig_ax.plot(
        [k for k, _ in pairs],
        [v for _, v in pairs],
        color=ZENITH_RAMP[30],
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    best_sigma = r["softness"]["sigma_px"]
    sig_ax.plot(
        best_sigma,
        max(sig.values()),
        marker="*",
        markersize=13,
        color=ACCENT,
        markeredgecolor=PAGE,
        zorder=5,
    )
    sig_ax.set_xlabel("shadow softness σ (px)", color=META, fontsize=9)
    sig_ax.set_ylabel("corr(map, darkening)", color=META, fontsize=9)
    sig_ax.set_title("How soft?", color=TEXT, fontsize=10, loc="left")

    s = r["strength"]
    fig.suptitle(
        "The real arm's contact shadow, measured from the episodes themselves — "
        f"{r['config']['frames']} frames, {r['config']['episodes']} episodes   |   "
        f"strength fit {s['fit']:.3f} CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}], "
        f"σ {best_sigma:g} px",
        color=TEXT,
        fontsize=10.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
