"""Results chart for the plumb-line wrist-lens fit
(fit_lens_plumbline.py output; eval-report dark scheme, IBM CVD-safe
hues). Two panels: the radial displacement of the fitted lens vs the
deployed ideal-equidistant assumption across the field (with the
bootstrap 95% band), and the plank-straightness residual under each
model variant.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
FITTED, DEPLOYED = "#648fff", "#ffb000"


def disp_curve(k2: float, k4: float, f_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """(theta_deg, r_fitted - r_equidistant) on a fixed theta grid."""
    r_grid = np.linspace(0.0, 420.0, 8401)
    rho = r_grid / f_dist
    theta_of_r = rho * (1 + k2 * rho**2 + k4 * rho**4)
    theta_s = np.linspace(0.0, np.deg2rad(50.0), 300)
    r_fit = np.interp(theta_s, theta_of_r, r_grid)
    return np.rad2deg(theta_s), r_fit - f_dist * theta_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fit",
        type=Path,
        default=Path("outputs/sim/lens_fit/wrist_lens_fit.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/lens_fit/chart__wrist_lens_fit.png"),
    )
    args = ap.parse_args()
    r = json.loads(args.fit.read_text())
    f_dist = r["f_dist_px"]
    p = r["params"]

    fig, (ax, res_ax) = plt.subplots(2, 1, figsize=(9.5, 6.4), height_ratios=[2.2, 1.4])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, res_ax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.set_axisbelow(True)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    th, d = disp_curve(p["k2"], p["k4"], f_dist)
    boots = np.array(r["bootstrap"]["params"])
    if len(boots):
        bands = np.array([disp_curve(b[2], b[3], f_dist)[1] for b in boots])
        lo, hi = np.percentile(bands, [2.5, 97.5], axis=0)
        ax.fill_between(th, lo, hi, color=FITTED, alpha=0.18, linewidth=0)
    ax.plot(th, d, color=FITTED, linewidth=2)
    ax.axhline(0, color=DEPLOYED, linewidth=1.4, linestyle="--")
    ax.annotate(
        "deployed assumption: ideal equidistant (r = f·θ)",
        (1.0, 0.6),
        color=DEPLOYED,
        fontsize=9,
    )
    ax.annotate(
        "fitted real lens (bootstrap 95% band)",
        (th[170], d[170] - 1.2),
        color=FITTED,
        fontsize=9,
        va="top",
    )
    # where the frame edge and corner sit in ray angle under the fit
    r_grid = np.linspace(0.0, 420.0, 8401)
    rho = r_grid / f_dist
    theta_of_r = np.rad2deg(rho * (1 + p["k2"] * rho**2 + p["k4"] * rho**4))
    for r_px, label, ha in (
        (240.0, "frame edge r=240", "right"),
        (400.0, "corner r=400", "right"),
    ):
        t_mark = float(np.interp(r_px, r_grid, theta_of_r))
        ax.axvline(t_mark, color=GRID, linewidth=1, linestyle=":")
        ax.annotate(
            label,
            (t_mark - 0.5, ax.get_ylim()[0] * 0.35),
            color=META,
            fontsize=8.5,
            ha=ha,
        )
    dd = r["displacement_px"]
    ax.annotate(
        f"Δr at corner: {dd['at_r400_corner']:+.1f} px "
        f"(CI95 [{dd['bootstrap_ci95']['at_r400'][0]:.1f}, "
        f"{dd['bootstrap_ci95']['at_r400'][1]:.1f}])",
        (th[-1] - 1, d[-1] + 1.0),
        color=TEXT,
        fontsize=9,
        ha="right",
    )
    ax.set_xlabel("ray angle θ (deg)", color=TEXT, fontsize=9.5)
    ax.set_ylabel("radial placement, fitted − assumed (px)", color=TEXT, fontsize=9.5)
    ax.set_title(
        "Real wrist lens vs the deployed equidistant assumption — "
        "plumb-line fit on 150 pinned real frames "
        f"(center offset {p['cx'] - 319.5:+.0f}, {p['cy'] - 239.5:+.0f} px)",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    variants = [
        ("raw frames\n(no undistort)", r["residual_px"]["raw_bow"], GRID),
        ("deployed\nequidistant", r["residual_px"]["equidistant_deployed"], DEPLOYED),
        ("center-only\nrefit", r["residual_px"]["center_only"], META),
        ("curve-only\nrefit", r["residual_px"]["curve_only"], META),
        ("fitted\n(center + curve)", r["residual_px"]["fitted"], FITTED),
    ]
    xs = np.arange(len(variants))
    res_ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    res_ax.bar(
        xs,
        [v[1] for v in variants],
        width=0.62,
        color=[v[2] for v in variants],
        edgecolor=PAGE,
        linewidth=1.5,
    )
    for x, (_, val, _) in zip(xs, variants, strict=True):
        res_ax.annotate(
            f"{val:.2f}",
            (x, val),
            color=TEXT,
            fontsize=9,
            ha="center",
            va="bottom",
            xytext=(0, 2),
            textcoords="offset points",
        )
    res_ax.set_xticks(xs)
    res_ax.set_xticklabels([v[0] for v in variants], color=TEXT, fontsize=8.5)
    res_ax.set_ylabel("plank straightness\nRMS residual (px)", color=TEXT, fontsize=9.5)
    res_ax.set_ylim(0, r["residual_px"]["raw_bow"] * 1.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
