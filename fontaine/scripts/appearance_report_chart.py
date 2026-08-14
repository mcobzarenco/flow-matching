"""Lead chart for the appearance-programme consolidated report (queue
`sim-appearance-consolidated-report`; eval-report dark scheme — the
shipped Carbon pair). Banked numbers only, read from the frozen
analysis JSONs in reports/. Two panels: the top-cam knn5 AUROC ladder
across every measured arm of the screen (shipped fixes, refuted
texture arms, diagnostic references, the real-fg floor); and the
promotion-decision paired reads — what each opt-in flag is worth
alone vs stacked, with the wrist-side read — against the zero rule.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
FIX, REFUTED, ANCHOR = "#08bdba", "#ffb000", "#9aa0a8"


def _results(reports: Path, stem: str) -> dict:
    return json.loads((reports / f"analysis__{stem}.json").read_text())["results"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/sim/appearance_report/chart__appearance_screen_ladder.png",
        ),
    )
    args = ap.parse_args()

    fgfix = _results(args.reports, "sim_fg_appearance_fix")
    decomp = _results(args.reports, "sim_top_gap_decomposition")["full_frame"]
    photo = _results(args.reports, "sim_arm_photometric_read")
    mount = _results(args.reports, "sim_mount_material_read")
    stack = _results(args.reports, "sim_full_optin_stack_read")
    tex = _results(args.reports, "sim_arm_texture_read")
    surf = _results(args.reports, "sim_arm_surface_texture_read")
    wrist = _results(args.reports, "sim_wrist_material_read")["wrist"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(13.6, 5.6), width_ratios=[1.2, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
        panel.set_axisbelow(True)

    # left: the ladder — every measured top-cam arm of the screen
    rows = [
        ("plate only (armless is itself OOD)", decomp["arms"]["plate_only"], ANCHOR),
        ("+ micro-texture (REFUTED)", tex["arms"]["v3_tex"], REFUTED),
        ("+ surface texture (REFUTED)", surf["arms"]["v3_surf"], REFUTED),
        ("v3 default (banked anchor)", stack["arms"]["v3"], ANCHOR),
        ("+ photometrics + mount", mount["arms"]["v3_full_fix"], FIX),
        ("+ arm photometrics", photo["arms"]["v3_photo"], FIX),
        ("no_clutter (removal ceiling)", fgfix["arms"]["no_clutter"], ANCHOR),
        ("+ clutter patches", fgfix["arms"]["patched"], FIX),
        ("full opt-in stack", stack["arms"]["stack_full"], FIX),
        ("real-fg composite (pipeline floor)", decomp["arms"]["real_fg"], ANCHOR),
    ]
    for y, (label, info, color) in enumerate(reversed(rows)):
        auroc = info["auroc_vs_real"]
        ax.barh(y, auroc - 0.25, left=0.25, height=0.5, color=color, alpha=0.85)
        ax.text(auroc + 0.005, y, f"{auroc:.3f}", color=TEXT, fontsize=9, va="center")
        ax.text(0.245, y, label, color=TEXT, fontsize=9, ha="right", va="center")
    clean = stack["clean_anchor"]["auroc_vs_real"]
    ax.axvline(clean, color=META, linewidth=0.8, alpha=0.7)
    ax.text(
        clean,
        len(rows) - 0.25,
        f"clean real {clean:.3f}",
        color=META,
        fontsize=8,
        ha="center",
    )
    ax.axvline(0.5, color=TEXT, linewidth=1.0, ls=(0, (4, 2)), alpha=0.8)
    ax.text(0.5, len(rows) - 0.25, "0.5 null", color=TEXT, fontsize=8, ha="center")
    ax.set_xlim(0.25, 0.93)
    ax.set_ylim(-0.6, len(rows) + 0.05)
    ax.set_yticks([])
    ax.set_xlabel("knn5 AUROC vs held-out real (lower = reads more real)", color=META)
    ax.set_title(
        "The top-cam appearance ladder — pinned 20×5 harness, er_60k probe",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    # right: the promotion-decision paired reads vs the zero rule
    reads = [
        ("full stack − v3", stack["paired_stack_vs_v3"], FIX),
        ("clutter patches − v3", fgfix["paired_vs_v3"]["patched"], FIX),
        ("materials − v3 (solo)", mount["rider_v3_full_fix"]["vs_v3"], FIX),
        ("materials on top of clutter", stack["paired_stack_vs_patched"], ANCHOR),
        ("wrist cam: stack − v3", wrist["paired_stack_vs_v3"], ANCHOR),
    ]
    for y, (label, read, color) in enumerate(reversed(reads)):
        lo, hi = (v * 1e7 for v in read["ci95"])
        dax.plot([lo, hi], [y, y], color=color, lw=2.4)
        dax.plot([read["mean_delta"] * 1e7], [y], "o", color=color, ms=8)
        dax.text(
            max(hi, 0) + 0.6,
            y,
            f"{read['mean_delta'] * 1e7:+.2f}  ({read['n_closer']}/100)",
            color=META,
            fontsize=8.5,
            va="center",
        )
        dax.text(-24.0, y + 0.28, label, color=TEXT, fontsize=9.5, ha="left")
    dax.axvline(0.0, color=TEXT, linewidth=1.0)
    dax.text(0.0, len(reads) - 0.32, "zero", color=TEXT, fontsize=8, ha="center")
    dax.set_xlim(-24.5, 9.5)
    dax.set_ylim(-0.55, len(reads))
    dax.set_yticks([])
    dax.set_xlabel(
        "paired Δknn5 vs real reference (×1e-7), CI95 10k resamples — "
        "left of zero = toward real",
        color=META,
    )
    dax.set_title(
        "What each flip is worth — alone, stacked, and on the wrist cam",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
