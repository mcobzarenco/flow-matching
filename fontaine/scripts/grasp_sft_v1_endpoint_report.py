"""House dark-mode charts + table for the grasp_sft_v1_joint endpoint
(queue item grasp-sft-v1-endpoint-boundary; run grasp_sft_v1_joint_8xa100,
8xA100, 3000 steps eff-96, launched 18:21:14Z 2026-08-16).

Reads ONLY banked artifacts (regenerable, no live hosts):
  reports/curve__grasp_sft_v1_train.jsonl     (rsynced train_log.jsonl)
  reports/curve__grasp_sft_v1_wandb.json      (written by --extract:
      the per-dataset eval/train MAE series wandb keeps —
      eval/chunk_mae_dataset/<repo> et al.)
  outputs/sim/grasp_sft/v1_endpoint/flow_unseen.json   (merged shards)
  outputs/sim/grasp_sft/v1_endpoint/token_unseen.json  (merged shards)
Sim legs are tolerant-absent (mid-run preview renders anchors-only).

Figures -> fontaine/blog/src/img/grasp_sft_v1/:
  1. loss_curves.png — loss_action_flow / loss_action_ar vs step, log y
  2. eval_mae.png    — eval_chunk_mae + train_mae, with the per-dataset
     eval series (the --eval-dataset-breakdown read as a picture)
  3. sim_strip.png   — per-seed progress strips, flow + token legs, vs
     the route-C joint-probe anchors (flow 44/100; token R2 bar 20)

Also prints the per-dataset MAE table (markdown) + headline facts and
writes reports/analysis__grasp_sft_v1_endpoint.json for the post.

Usage:
  uv run python fontaine/scripts/grasp_sft_v1_endpoint_report.py --extract
  uv run python fontaine/scripts/grasp_sft_v1_endpoint_report.py
  # v2 boundary (paths re-pointed; wandb id fetched at boundary):
  uv run python ... --run v2 --extract --wandb-run aristotle1337/fontaine/<id>
  uv run python ... --run v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/ubuntu/flow-matching")
TRAIN_JSONL = REPO / "reports/curve__grasp_sft_v1_train.jsonl"
WANDB_JSON = REPO / "reports/curve__grasp_sft_v1_wandb.json"
SIM_DIR = REPO / "outputs/sim/grasp_sft/v1_endpoint"
IMG_DIR = REPO / "fontaine/blog/src/img/grasp_sft_v1"
OUT_JSON = REPO / "reports/analysis__grasp_sft_v1_endpoint.json"

# Run 2 (--recompute-stats restart, 21:14:48Z 08-16). Run 1b (killed
# ~1900, remap-only table) was 8t78ipnl; absolute MAE not comparable
# across the two (different normalization scale).
WANDB_RUN = "aristotle1337/fontaine/cgo3by9j"  # grasp_sft_v1_joint_8xa100

# House eval-report scheme (sim100_charts.py / grasp_sft_chain_charts.py):
# dark page, IBM CVD-safe categorical hues (adjacent-pair OKLab deltaE
# validated on this surface 2026-08-11); identity never color-alone —
# every series direct-labeled.
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
MAGENTA, BLUE, GOLD = "#dc267f", "#648fff", "#ffb000"

# Route-C joint-probe anchors (posts/2026-08-16-amendment-...-joint.md,
# probe results banked in-channel 08-16): the step-2000 probe checkpoint
# scored 44/100 unseen (flow) and the B SS3 R2 bar is >=20/100 (token).
ANCHOR_FLOW_PROBE = 44
ANCHOR_TOKEN_R2 = 20

DATASET_LABELS = {
    "grasp_demos_v1/merged": "grasp_demos_v1 (91%)",
    "mcobzarenco/so101_pick_place_v2": "pick_place_v2 (8%)",
    "mcobzarenco/so101_pick_place_clean": "pick_place_clean (1%)",
}
DATASET_COLORS = {
    "grasp_demos_v1/merged": MAGENTA,
    "mcobzarenco/so101_pick_place_v2": BLUE,
    "mcobzarenco/so101_pick_place_clean": GOLD,
}

# --run v2 re-points every path at the v2 boundary artifacts
# (grasp_sft_v2_joint_8xa100, launched 09:57:39Z 08-17 on the v2 regen
# corpus). Its wandb id is unknown until the run registers — pass
# --wandb-run at --extract time (babysit/queue item carries the lookup).
RUNS = {
    "v1": {
        "run_name": "grasp_sft_v1_joint",
        "wandb_run": "aristotle1337/fontaine/cgo3by9j",
        "labels": DATASET_LABELS,
        "extra_sim_anchors": {},
    },
    "v2": {
        "run_name": "grasp_sft_v2_joint",
        "wandb_run": None,
        "labels": {
            "grasp_demos_v2/merged": "grasp_demos_v2 (93%)",
            "mcobzarenco/so101_pick_place_v2": "pick_place_v2 (6%)",
            "mcobzarenco/so101_pick_place_clean": "pick_place_clean (1%)",
        },
        "extra_sim_anchors": {"v1_endpoint_flow": 5, "v1_step500_flow": 2},
    },
}
RUN_NAME = "grasp_sft_v1_joint"
EXTRA_SIM_ANCHORS: dict[str, int] = {}


def configure(run_key: str, wandb_run: str | None) -> None:
    """Re-point the module-level path/label constants at one run's
    artifacts (module attributes, since every chart fn reads them)."""
    mod = sys.modules[__name__]
    cfg = RUNS[run_key]
    short = run_key  # v1 / v2
    colors = list(DATASET_COLORS.values())
    mod.RUN_NAME = cfg["run_name"]
    mod.TRAIN_JSONL = REPO / f"reports/curve__grasp_sft_{short}_train.jsonl"
    mod.WANDB_JSON = REPO / f"reports/curve__grasp_sft_{short}_wandb.json"
    mod.SIM_DIR = REPO / f"outputs/sim/grasp_sft/{short}_endpoint"
    mod.IMG_DIR = REPO / f"fontaine/blog/src/img/grasp_sft_{short}"
    mod.OUT_JSON = REPO / f"reports/analysis__grasp_sft_{short}_endpoint.json"
    mod.WANDB_RUN = wandb_run or cfg["wandb_run"]
    mod.DATASET_LABELS = cfg["labels"]
    mod.DATASET_COLORS = dict(zip(cfg["labels"], colors, strict=True))
    mod.EXTRA_SIM_ANCHORS = cfg["extra_sim_anchors"]


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=META, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.5)


def new_fig(width: float = 8.0, height: float = 4.5) -> tuple[Any, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=PAGE)
    style_axis(ax)
    return fig, ax


def load_train_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # The relaunch (18:21:14Z, no save existed) appended to launch 1's
    # train_log.jsonl in the same save-dir, so steps 10-250 appear twice.
    # Keep the LAST monotone segment: rows after the final step decrease.
    rows = [json.loads(line) for line in TRAIN_JSONL.read_text().splitlines()]
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["step"] < rows[i - 1]["step"]:
            start = i
    rows = rows[start:]
    steps = [r for r in rows if "eval_chunk_mae" not in r]
    evals = [r for r in rows if "eval_chunk_mae" in r]
    return steps, evals


def extract_wandb() -> None:
    """Bank the wandb history rows that carry eval metrics."""
    import wandb

    api = wandb.Api()
    run = api.run(WANDB_RUN)
    rows = [
        row
        for row in run.scan_history(page_size=2000)
        if any("chunk_mae" in k or k == "eval/chunk_mae" for k in row)
    ]
    WANDB_JSON.write_text(json.dumps({"run": WANDB_RUN, "rows": rows}, indent=1))
    print(f"banked {len(rows)} wandb eval rows -> {WANDB_JSON}")


def chart_loss(steps: list[dict[str, Any]], out: Path) -> None:
    fig, ax = new_fig()
    xs = [r["step"] for r in steps]
    for key, color, label in (
        ("loss_action_ar", BLUE, "action-token CE"),
        ("loss_action_flow", MAGENTA, "flow MSE"),
    ):
        ys = [r[key] for r in steps]
        ax.plot(xs, ys, color=color, linewidth=2)
        ax.annotate(
            f"{label}  {ys[-1]:.3f}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color=color,
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title(f"{RUN_NAME} — component losses")
    ax.set_xlim(0, max(xs) * 1.18)
    fig.savefig(out, bbox_inches="tight", facecolor=PAGE, dpi=150)
    plt.close(fig)


def chart_eval_mae(
    evals: list[dict[str, Any]],
    wandb_rows: list[dict[str, Any]],
    out: Path,
) -> None:
    fig, ax = new_fig()
    xs = [r["step"] for r in evals]
    ys = [r["eval_chunk_mae"] for r in evals]
    ax.plot(xs, ys, color=TEXT, linewidth=2.5)
    ax.annotate(
        f"pooled eval  {ys[-1]:.2f}",
        (xs[-1], ys[-1]),
        textcoords="offset points",
        xytext=(6, 0),
        color=TEXT,
        fontsize=9,
    )
    for repo, color in DATASET_COLORS.items():
        key = f"eval/chunk_mae_dataset/{repo}"
        pts = [(r["_step"], r[key]) for r in wandb_rows if key in r]
        if not pts:
            continue
        px, py = zip(*sorted(pts), strict=True)
        ax.plot(px, py, color=color, linewidth=2, alpha=0.9)
        ax.annotate(
            f"{DATASET_LABELS[repo]}  {py[-1]:.2f}",
            (px[-1], py[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color=color,
            fontsize=9,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("held-out chunk MAE")
    ax.set_title("eval MAE — pooled + per-dataset (--eval-dataset-breakdown)")
    ax.set_xlim(0, max(xs) * 1.30)
    fig.savefig(out, bbox_inches="tight", facecolor=PAGE, dpi=150)
    plt.close(fig)


def load_leg(name: str) -> dict[str, Any] | None:
    path = SIM_DIR / f"{name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    episodes = payload["episodes"]
    succ = sorted(e["seed"] for e in episodes if e.get("success_tick") is not None)
    return {
        "n": len(episodes),
        "successes": len(succ),
        "success_seeds": succ,
        "moved_gt_half_cm": sum(
            1 for e in episodes if e.get("progress_final_cm", 0.0) > 0.5
        ),
        "mean_progress_cm": float(
            np.mean([e.get("progress_final_cm", 0.0) for e in episodes]),
        ),
        "serve_head": payload.get("config", {}).get("serve_head"),
        "episodes": episodes,
    }


def chart_sim_strip(
    flow: dict[str, Any] | None,
    token: dict[str, Any] | None,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.2), facecolor=PAGE, sharex=True)
    legs = (
        ("flow (euler-10)", flow, MAGENTA, ANCHOR_FLOW_PROBE, "probe ckpt 44/100"),
        ("token (_arhead greedy)", token, BLUE, ANCHOR_TOKEN_R2, "R2 bar 20/100"),
    )
    for ax, (label, leg, color, anchor, anchor_label) in zip(axes, legs, strict=True):
        style_axis(ax)
        if leg is None:
            ax.text(
                0.5,
                0.5,
                f"{label}: PENDING",
                transform=ax.transAxes,
                ha="center",
                color=META,
                fontsize=11,
            )
        else:
            for e in leg["episodes"]:
                seed = e["seed"]
                hit = e.get("success_tick") is not None
                ax.bar(
                    seed,
                    max(e.get("progress_final_cm", 0.0), 0.0),
                    width=0.85,
                    color=color if hit else GRID,
                    alpha=1.0 if hit else 0.9,
                )
            ax.text(
                0.01,
                0.86,
                f"{label} — {leg['successes']}/{leg['n']} success "
                f"({anchor_label}), moved>0.5cm {leg['moved_gt_half_cm']}",
                transform=ax.transAxes,
                color=TEXT,
                fontsize=10,
            )
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_ylabel("progress cm")
        _ = anchor
    axes[-1].set_xlabel("seed (0–99, colored = success)")
    axes[0].set_title("sim100 endpoint rollouts — step-3000 checkpoint")
    fig.savefig(out, bbox_inches="tight", facecolor=PAGE, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--run", choices=sorted(RUNS), default="v1")
    parser.add_argument(
        "--wandb-run",
        default=None,
        help="wandb run path override (REQUIRED for --run v2 --extract: "
        "the id only exists once the run registers)",
    )
    args = parser.parse_args()
    configure(args.run, args.wandb_run)
    if args.extract:
        if WANDB_RUN is None:
            parser.error(f"--run {args.run} has no banked wandb id; pass --wandb-run")
        extract_wandb()
        return 0

    steps, evals = load_train_rows()
    wandb_rows: list[dict[str, Any]] = []
    if WANDB_JSON.exists():
        wandb_rows = json.loads(WANDB_JSON.read_text())["rows"]
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    chart_loss(steps, IMG_DIR / "loss_curves.png")
    chart_eval_mae(evals, wandb_rows, IMG_DIR / "eval_mae.png")
    flow, token = load_leg("flow_unseen"), load_leg("token_unseen")
    chart_sim_strip(flow, token, IMG_DIR / "sim_strip.png")

    last_eval = evals[-1]
    table_rows = []
    if wandb_rows:
        final = max(
            (r for r in wandb_rows if any("chunk_mae_dataset" in k for k in r)),
            key=lambda r: r["_step"],
        )
        for repo in DATASET_LABELS:
            ev = final.get(f"eval/chunk_mae_dataset/{repo}")
            tr = final.get(f"train/chunk_mae_dataset/{repo}")
            table_rows.append(
                {
                    "dataset": repo,
                    "eval_chunk_mae": ev,
                    "train_chunk_mae": tr,
                },
            )
        print(f"\nper-dataset MAE @ step {final['_step']}:\n")
        print("| dataset | eval MAE | train MAE |")
        print("|---|---|---|")
        for row in table_rows:
            ev = (
                "—" if row["eval_chunk_mae"] is None else f"{row['eval_chunk_mae']:.2f}"
            )
            tr = (
                "—"
                if row["train_chunk_mae"] is None
                else f"{row['train_chunk_mae']:.2f}"
            )
            print(f"| {row['dataset']} | {ev} | {tr} |")

    summary = {
        "final_step": last_eval["step"],
        "eval_chunk_mae": last_eval["eval_chunk_mae"],
        "train_mae": last_eval["train_mae"],
        "eval_curve": [
            {"step": r["step"], "eval_chunk_mae": r["eval_chunk_mae"]} for r in evals
        ],
        "per_dataset_final": table_rows,
        "sim100": {
            "flow_unseen": {k: v for k, v in (flow or {}).items() if k != "episodes"}
            if flow
            else None,
            "token_unseen": {k: v for k, v in (token or {}).items() if k != "episodes"}
            if token
            else None,
            "anchors": {
                "flow_probe_step2000": ANCHOR_FLOW_PROBE,
                "token_r2_bar": ANCHOR_TOKEN_R2,
                **EXTRA_SIM_ANCHORS,
            },
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1))
    print(f"\nheadline: eval {last_eval['eval_chunk_mae']:.2f} @ {last_eval['step']}")
    print(f"wrote {OUT_JSON} + 3 figures -> {IMG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
