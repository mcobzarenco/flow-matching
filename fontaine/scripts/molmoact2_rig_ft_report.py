"""Rig fine-tune rung report — anchors + rung curve, house dark theme.

Renders the `rig_ft_r1` (fontaine_so101_rig_ae_r1) anchor-rung story
from the banked 240-row analysis files into one browsable HTML page:

  * rung MAE curve (zero-shot -> step 2000) vs the state-copy anchor
  * matched-window MAE by chunk timestep, every rung + both anchors
  * per-joint motion-correlation small multiples across rungs
  * strided trajectory gallery (truth vs final rung vs zero-shot vs
    state-copy)
  * rung + per-joint tables

Inputs are the frozen preflight/rung reads written by
``molmoact2_rig_preflight.py`` (json + npz per rung, identical 240
rows). Every chart number is recomputed from the npz and cross-checked
against the banked json (loud fail on mismatch) — the page can't drift
from the frozen reads.

Usage:
    uv run python fontaine/scripts/molmoact2_rig_ft_report.py \
        [--out reports/eval__fontaine_so101_rig_ae_r1__anchor_rungs.html]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from html import escape
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RUNGS = [
    ("zero-shot", 0, "analysis__molmoact2_rig_preflight"),
    ("step 500", 500, "analysis__molmoact2_rig_ft_step500"),
    ("step 1000", 1000, "analysis__molmoact2_rig_ft_step1000"),
    ("step 1500", 1500, "analysis__molmoact2_rig_ft_step1500"),
    ("step 2000", 2000, "analysis__molmoact2_rig_ft_step2000"),
]
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# house 10-color palette (bijou.eval.report dark series + panel-report
# extension), assigned in fixed order: rungs light->final, anchors last
STEP_COLORS = [
    "#9aa0a8",  # zero-shot (recessive gray-blue)
    "#785ef0",
    "#dc267f",
    "#ffb000",
    "#648fff",  # final rung = primary blue
    "#4ec9b0",  # state-copy anchor
]
N_GALLERY = 8


def load_rungs() -> list[dict]:
    out = []
    for label, step, stem in RUNGS:
        d = json.loads((REPO_ROOT / f"reports/{stem}.json").read_text())
        z = np.load(REPO_ROOT / f"reports/{stem}.npz")
        mae = float(np.abs(z["preds"] - z["truths"]).mean())
        if abs(mae - d["zero_shot_mae_matched"]) > 5e-4:
            raise SystemExit(
                f"ORACLE FAIL {stem}: npz MAE {mae:.4f} != json "
                f"{d['zero_shot_mae_matched']:.4f}",
            )
        copy_mae = float(
            np.abs(z["states"][:, None, :] - z["truths"]).mean(),
        )
        if abs(copy_mae - d["state_copy_mae_matched"]) > 5e-4:
            raise SystemExit(f"ORACLE FAIL {stem}: state-copy mismatch")
        out.append(
            {
                "label": label,
                "step": step,
                "json": d,
                "preds": z["preds"],
                "truths": z["truths"],
                "states": z["states"],
                "rows": z["rows"],
                "mae": mae,
                "copy_mae": copy_mae,
            },
        )
    return out


def _fig_uri(fig, theme) -> str:  # noqa: ANN001
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=110,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def rung_curve_chart(rungs, rt) -> str:  # noqa: ANN001
    import matplotlib.pyplot as plt

    theme = rt.THEMES["dark"]
    with plt.style.context(theme.mpl_style):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        fig.patch.set_facecolor(theme.page_bg)
        ax.set_facecolor(theme.page_bg)
        xs = [r["step"] for r in rungs]
        ys = [r["mae"] for r in rungs]
        ax.plot(
            xs,
            ys,
            color=STEP_COLORS[4],
            linewidth=2,
            marker="o",
            markersize=7,
            zorder=3,
        )
        for x, y in zip(xs, ys, strict=False):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color=theme.text,
            )
        ax.axhline(
            rungs[0]["copy_mae"],
            color=STEP_COLORS[5],
            linestyle="--",
            linewidth=1.6,
        )
        ax.annotate(
            f"state-copy anchor {rungs[0]['copy_mae']:.2f}",
            (xs[-1], rungs[0]["copy_mae"]),
            ha="right",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=9,
            color=STEP_COLORS[5],
        )
        ax.set_xlabel("fine-tune step (0 = released checkpoint, zero-shot)")
        ax.set_ylabel("matched-window MAE (240 rig frames)")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    return _fig_uri(fig, theme)


def per_step_chart(rungs, rt) -> str:  # noqa: ANN001
    import matplotlib.pyplot as plt

    theme = rt.THEMES["dark"]
    with plt.style.context(theme.mpl_style):
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        fig.patch.set_facecolor(theme.page_bg)
        ax.set_facecolor(theme.page_bg)
        steps = np.arange(rungs[0]["truths"].shape[1])
        for i, r in enumerate(rungs):
            mae_t = np.abs(r["preds"] - r["truths"]).mean(axis=(0, 2))
            ax.plot(steps, mae_t, color=STEP_COLORS[i], linewidth=1.8, label=r["label"])
        copy_t = np.abs(
            rungs[0]["states"][:, None, :] - rungs[0]["truths"],
        ).mean(axis=(0, 2))
        ax.plot(
            steps,
            copy_t,
            color=STEP_COLORS[5],
            linewidth=1.8,
            linestyle="--",
            label="state-copy",
        )
        ax.set_xlabel("chunk timestep (30 steps = 1.0 s at 30 fps)")
        ax.set_ylabel("MAE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.6, which="both")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=6,
            fontsize=9,
            frameon=False,
        )
    return _fig_uri(fig, theme)


def corr_smallmultiples(rungs, rt) -> str:  # noqa: ANN001
    import matplotlib.pyplot as plt

    theme = rt.THEMES["dark"]
    with plt.style.context(theme.mpl_style):
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey=True)
        fig.patch.set_facecolor(theme.page_bg)
        xs = [r["step"] for r in rungs]
        for j, ax in enumerate(axes.flat):
            ax.set_facecolor(theme.page_bg)
            ys = [r["json"]["joints"][j]["motion_corr"] for r in rungs]
            ax.plot(
                xs,
                ys,
                color=STEP_COLORS[4],
                linewidth=1.8,
                marker="o",
                markersize=5,
            )
            ax.axhline(0.0, color=theme.truth_color, linewidth=0.8, alpha=0.4)
            ax.set_title(f"joint {j} — {JOINT_NAMES[j]}", fontsize=9)
            ax.set_ylim(-0.3, 1.05)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.annotate(
                f"{ys[-1]:+.2f}",
                (xs[-1], ys[-1]),
                fontsize=8,
                xytext=(-4, -12),
                textcoords="offset points",
                ha="right",
                color=theme.text,
            )
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.supxlabel("fine-tune step (0 = zero-shot)", fontsize=10)
        fig.supylabel("motion correlation (pooled 30-step window)", fontsize=10)
        fig.tight_layout()
    return _fig_uri(fig, theme)


def gallery_block(rungs, rt) -> str:  # noqa: ANN001
    import matplotlib.pyplot as plt

    theme = rt.THEMES["dark"]
    final = rungs[-1]
    zero = rungs[0]
    n = final["truths"].shape[0]
    picks = np.linspace(0, n - 1, N_GALLERY).round().astype(int)
    blocks = []
    with plt.style.context(theme.mpl_style):
        for p in picks:
            fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
            fig.patch.set_facecolor(theme.page_bg)
            steps = np.arange(final["truths"].shape[1])
            for j, ax in enumerate(axes.flat):
                ax.set_facecolor(theme.page_bg)
                ax.plot(
                    steps,
                    final["truths"][p, :, j],
                    color=theme.truth_color,
                    linewidth=2.4,
                    zorder=2,
                )
                ax.plot(
                    steps,
                    final["preds"][p, :, j],
                    color=STEP_COLORS[4],
                    linewidth=1.7,
                    zorder=3,
                )
                ax.plot(
                    steps,
                    zero["preds"][p, :, j],
                    color=STEP_COLORS[0],
                    linewidth=1.4,
                    zorder=1,
                )
                ax.axhline(
                    final["states"][p, j],
                    color=STEP_COLORS[5],
                    linestyle="--",
                    linewidth=1.2,
                    zorder=1,
                )
                ax.set_title(JOINT_NAMES[j], fontsize=8)
                ax.grid(True, alpha=0.2, linewidth=0.5)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
            fig.suptitle(
                f"anchor row {int(final['rows'][p])} — frame {p + 1} of {n} (strided)",
                fontsize=10,
            )
            fig.tight_layout()
            blocks.append(f'<img class="chart" src="{_fig_uri(fig, theme)}">')
    return "".join(blocks)


def gallery_legend(rt) -> str:  # noqa: ANN001
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    theme = rt.THEMES["dark"]
    with plt.style.context(theme.mpl_style):
        fig = plt.figure(figsize=(10, 0.6))
        fig.patch.set_facecolor(theme.page_bg)
        handles = [
            Line2D([0], [0], color=theme.truth_color, linewidth=2.4),
            Line2D([0], [0], color=STEP_COLORS[4], linewidth=1.7),
            Line2D([0], [0], color=STEP_COLORS[0], linewidth=1.4),
            Line2D([0], [0], color=STEP_COLORS[5], linewidth=1.2, linestyle="--"),
        ]
        fig.legend(
            handles,
            [
                "truth",
                "fine-tuned step 2000",
                "zero-shot",
                "state-copy (current state held)",
            ],
            loc="center",
            ncol=4,
            fontsize=9,
            frameon=False,
        )
    return _fig_uri(fig, theme)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from bijou.eval import report as rt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "reports/eval__fontaine_so101_rig_ae_r1__anchor_rungs.html",
        ),
    )
    args = parser.parse_args()

    rungs = load_rungs()
    theme = rt.THEMES["dark"]

    rung_rows = []
    for r in rungs:
        joints = r["json"]["joints"]
        corrs = [j["motion_corr"] for j in joints]
        offs = [abs(j["step0_signed_offset"]) for j in joints]
        rung_rows.append(
            [
                r["label"],
                f"{r['mae']:.4f}",
                f"{min(corrs):+.3f} … {max(corrs):+.3f}",
                f"{max(offs):.2f}",
            ],
        )
    rung_table = rt._table(
        [
            "checkpoint",
            "matched-window MAE",
            "motion corr (min … max)",
            "max |step-0 offset|",
        ],
        rung_rows,
    )
    joint_rows = []
    for j, d in enumerate(rungs[-1]["json"]["joints"]):
        joint_rows.append(
            [
                f"{j} — {JOINT_NAMES[j]}",
                f"{d['motion_corr']:+.4f}",
                f"{d['step0_signed_offset']:+.3f}",
                f"{d['step0_err_std']:.2f}",
                f"{d['rig_span_q01_q99']:.1f}",
            ],
        )
    joint_table = rt._table(
        ["joint", "motion corr", "step-0 offset", "step-0 err std", "rig q01–q99 span"],
        joint_rows,
    )

    config_lines = [
        (
            "run rig_ft_r1 (fontaine_so101_rig_ae_r1) — AE-only fine-tune of "
            "allenai/MolmoAct2-SO100_101 on the 2 SO-101 rig repos"
        ),
        (
            "trainer: their train_lerobot.py (branch fontaine-so101-rig), 2000 "
            "steps, global batch 64, AE lr 5e-5, --ft_vlm=false "
            "--ft_embedding=none (577M trainable / 5.5B)"
        ),
        (
            "data: mcobzarenco/so101_pick_place_clean (7 ep) + _v2 (50 ep), "
            "LeRobot v3.0 end-to-end, rig-only q01/q99 norm stats"
        ),
        (
            "launched 2026-08-10 17:48:18Z, rc=0 20:27:44Z, ~2.7 GPU-h (gate "
            "12); pre-reg posts/2026-08-10-prereg-molmoact2-rig-finetune.md "
            "(+ Amendment 1)"
        ),
        (
            "reads: 240 evenly strided rig frames, matched 30-step / 1.0 s "
            "window, identical rows every rung (frozen preflight instrument)"
        ),
        (
            "CAVEAT (pre-registered): train-frame sanity reads, contaminated "
            "by construction — the real eval is on-rig rollouts (runbook "
            "sections 3-4)"
        ),
    ]

    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>rig_ft_r1 anchor rungs</title>"
        f"<style>{theme.css()}</style></head><body>"
        "<h1>MolmoAct2 SO-101 rig fine-tune — anchor rungs "
        "(rig_ft_r1, pre-reg PASS)</h1>"
        f"<pre>{escape(chr(10).join(config_lines))}</pre>"
        "<h2>Rung curve — matched-window MAE vs fine-tune step</h2>"
        f'<img class="chart" src="{rung_curve_chart(rungs, rt)}">'
        f"<h2>Rung summary</h2>{rung_table}"
        "<h2>MAE by chunk timestep (log scale) — every rung + anchors</h2>"
        f'<img class="chart" src="{per_step_chart(rungs, rt)}">'
        "<h2>Per-joint motion correlation across rungs (zero line marked; "
        "at zero-shot joint 1 sat at +0.22 with a +79-unit step-0 offset "
        "— the posture-collapse finding of pre-reg Amendment 1)</h2>"
        f'<img class="chart" src="{corr_smallmultiples(rungs, rt)}">'
        f"<h2>Final rung (step 2000) per joint</h2>{joint_table}"
        f"<h2>Sample trajectories — {N_GALLERY} of 240 anchor frames, "
        "evenly strided. Legend below applies to every panel.</h2>"
        f'<img class="chart" src="{gallery_legend(rt)}">'
        f"{gallery_block(rungs, rt)}"
        "</body></html>"
    )
    Path(args.out).write_text(document)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
