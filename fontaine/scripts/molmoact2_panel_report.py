"""MolmoAct2 out-of-band 3-policy side-by-side HTML report (owner spec 11:59Z).

Owner GO 2026-08-10 11:59:33Z: "generate an html eval report similar
to the one we normally do, but with both our best policy (snapflow
80k) predictions vs. molmo2act and state copy on the same frames. as
well as summary statistics obviously." Pre-reg
posts/2026-08-10-prereg-molmoact2-oob-panel.md.

Builds one self-contained HTML page in the house eval-report style
(bijou.eval.report theme + table + chart machinery), from banked npzs
only — no model runs:

  * summary block on top: MATCHED-WINDOW (steps 0..29 = their native
    1.0 s horizon) chunk/first MAE per policy, pooled + clean/
    contaminated splits, and the paired Δ + bootstrap CI95 rows —
    all read from the frozen-reads analysis json
    (molmoact2_panel_reads.py output), never recomputed here;
  * per-motor matched-window MAE table;
  * full-50 numbers for our arms as a collapsed secondary table;
  * per-frame sample gallery (32 evenly strided core frames, house
    convention): camera thumbnails + task + state + per-joint chart
    overlaying truth vs snapflow-80k (both decode configs) vs
    MolmoAct2 vs state-copy — the MolmoAct2 line stops at step 30 by
    construction (NaN tail is never drawn), which is the honest
    visual of the horizon difference. Per-sample MAE lines are
    matched-window for every policy.

Images/state/task are re-fetched from the LeRobot datasets for the
gallery rows, re-verified by the same alignment oracle as the
predictor. Pure CPU. Record-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from html import escape
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_HERE = Path(__file__).resolve().parent


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mpr = _sibling("molmoact2_panel_reads")
mpp = _sibling("molmoact2_panel_predict")

WINDOW = 30
SAMPLES = 32
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# Display order: our best first (owner's ask), then theirs, then floor.
# Trunk in every displayed name (owner 2026-08-10 14:49Z).
POLICIES = [
    (
        "snapflow 80k top10tickets [gemma4-E2B]",
        "top10",
        "pred:bijou@80000_draws10_ticket",
    ),
    ("snapflow 80k stablekey [gemma4-E2B]", "stable", "pred:bijou@80000"),
    ("MolmoAct2 SO100_101 [molmo2-ER]", "cand", mpr.CAND_KEY),
    ("state-copy [no model]", "cand", "pred:state-copy"),
]
# analysis-json keys (stable) -> displayed names with trunk
DISPLAY = {
    "snapflow80k top10tickets": (
        "snapflow 80k top-10-tickets — Gemma-4-E2B trunk (frozen, AR-pretrained) + flow expert h1024"
    ),
    "snapflow80k stablekey": (
        "snapflow 80k stable-key — Gemma-4-E2B trunk (frozen, AR-pretrained) + flow expert h1024"
    ),
    "molmoact2": "MolmoAct2 SO100_101 — Molmo2-ER trunk + their flow action expert",
    "state-copy": "state-copy — no model (repeat current state)",
    "ar_40k endpoint": "ar 40k endpoint — Molmo2-4B trunk, AR decoder",
    "ar_60k continuation": "ar 60k continuation — Molmo2-4B trunk, AR decoder",
    "er_60k@15000": "er 60k @15000 — Molmo2-ER trunk, AR decoder (mid-training)",
}


def display(label: str) -> str:
    return DISPLAY.get(label, label)


OUT_DEFAULT = "reports/eval__molmoact2_oob_3policy__panel_curated_v0_k4l2.html"


def load_arms() -> tuple[dict, dict, dict]:
    cand = mpr._load_npz(f"{mpr.CAND_STEM}.npz")
    top10 = mpr._load_npz(f"{mpr.BASELINES[0]['stem']}.npz")
    stable = mpr._load_npz(f"{mpr.BASELINES[1]['stem']}.npz")
    for other in (top10, stable):
        for key in mpr.bbr.PAIR_KEYS:
            if not np.array_equal(cand[key], other[key]):
                sys.exit(f"pairing broken on {key} — stop")
    return cand, top10, stable


def summary_tables(analysis: dict, rt: ModuleType) -> list:
    """The headline blocks, straight from the frozen-reads json."""
    mw = analysis["matched_window"]
    order = [
        "snapflow80k top10tickets",
        "snapflow80k stablekey",
        "molmoact2",
        "state-copy",
        "ar_40k endpoint",
        "ar_60k continuation",
        "er_60k@15000",
    ]
    rows = [
        [
            display(label),
            *(
                f"{mw[label][split]['chunk_mae']:.4f}"
                for split in ("pooled", "clean", "contaminated")
            ),
            f"{mw[label]['pooled']['first_mae']:.4f}",
        ]
        for label in order
        if label in mw
    ]
    ns = next(iter(analysis["paired_reads"].values()))
    t1 = rt.ReportTable(
        title=(
            "Matched-window chunk MAE (steps 0-29 = 1.0 s, core frames; "
            "clean = repos absent from their SO-100/101 fine-tune mix)"
        ),
        header=[
            "policy",
            *(
                f"{split} ({ns[split]['n_frames']})"
                for split in ("pooled", "clean", "contaminated")
            ),
            "first",
        ],
        rows=rows,
    )
    rows2 = []
    for label, splits in analysis["paired_reads"].items():
        for split in ("pooled", "clean", "contaminated"):
            r = splits[split]
            lo, hi = r["ci95"]
            rows2.append(
                [
                    f"MolmoAct2 − {display(label)}",
                    split,
                    f"{r['delta_frame_mean']:+.4f}",
                    f"[{lo:+.4f}, {hi:+.4f}]",
                    str(r["n_frames"]),
                    r["classification"],
                ],
            )
    t2 = rt.ReportTable(
        title="Paired per-frame Δ chunk MAE, seeded bootstrap CI95 (matched window)",
        header=["read", "split", "Δ mean", "CI95", "n", "classification"],
        rows=rows2,
    )
    rows3 = [
        [display(label), f"{v['chunk_mae']:.4f}", f"{v['first_mae']:.4f}"]
        for label, v in analysis["secondary_full50"].items()
    ]
    t3 = rt.ReportTable(
        title="Secondary: our arms on the full 50-step chunk (never vs theirs)",
        header=["policy", "chunk_mae (50)", "first_mae"],
        rows=rows3,
    )
    return [t1, t2, t3]


def per_motor_table(arms: dict, truth: np.ndarray, m2: np.ndarray, rt: ModuleType):  # noqa: ANN201
    rows = []
    for label, pred in arms.items():
        err = np.abs(pred[:, :WINDOW] - truth[:, :WINDOW])
        per = [float(err[:, :, d][m2].mean()) for d in range(err.shape[2])]
        rows.append([label, *(f"{v:.2f}" for v in per)])
    return rt.ReportTable(
        title="Per-motor matched-window chunk MAE (core frames)",
        header=["policy", *MOTOR_NAMES],
        rows=rows,
    )


def gallery(cand: dict, preds: dict, indices: np.ndarray, rt: ModuleType) -> list[str]:
    """Re-fetch images/state/task for the sampled rows; render blocks."""
    theme = rt.THEMES["dark"]
    dataset = mpp.build_dataset()
    blocks = []
    for row in indices.tolist():
        item = dataset[int(cand["index"][row])]
        mpp.check_alignment(item, cand, row)
        cameras = {
            key.removeprefix("observation.images."): item[key]
            for key in item
            if key.startswith("observation.images.")
        }
        truth_t = torch.as_tensor(cand["truth"][row])
        valid_t = torch.as_tensor(cand["valid"][row])
        predictions = {
            label: torch.as_tensor(pred[row]) for label, pred in preds.items()
        }
        sample = rt.ReportSample(
            index=int(cand["index"][row]),
            episode=int(cand["episode_index"][row]),
            frame_in_episode=int(cand["frame_index"][row]),
            repo_id=str(cand["repo_id"][row]),
            task=str(item["task"]),
            state=torch.as_tensor(
                np.asarray(item["observation.state"], dtype=np.float32),
            ),
            cameras=cameras,
            truth=truth_t,
            valid=valid_t,
            predictions=predictions,
            aux_generated=None,
            aux_label=None,
        )
        # Matched-window MAE line (the stock block's full-valid MAE would
        # be NaN on the molmoact2 tail).
        vw = cand["valid"][row][:WINDOW]
        mae_line = "  |  ".join(
            f"{label}: "
            f"{np.abs(pred[row][:WINDOW] - cand['truth'][row][:WINDOW])[vw].mean():.2f}"
            for label, pred in preds.items()
        )
        state_line = ", ".join(f"{x:.1f}" for x in sample.state.tolist())
        cams = "".join(
            f'<img src="{rt._image_data_uri(image)}" alt="{escape(name)}" '
            f'title="{escape(name)}">'
            for name, image in sorted(sample.cameras.items())
        )
        chart = rt._chart_data_uri(sample, MOTOR_NAMES, theme)
        blocks.append(
            f'<div class="sample">'
            f"<h3>{escape(sample.repo_id)} &mdash; episode {sample.episode}, "
            f"frame {sample.frame_in_episode} (global index {sample.index})</h3>"
            f'<p class="meta">task: {escape(sample.task)}<br>'
            f"state: [{state_line}]<br>"
            f"matched-window (1.0 s) chunk MAE &mdash; {escape(mae_line)}</p>"
            f'<div class="cams">{cams}</div>'
            f'<img class="chart" src="{chart}">'
            f"</div>",
        )
        print(f"gallery: rendered row {row} ({sample.repo_id})", flush=True)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--analysis", default=mpr.OUT_DEFAULT)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    args = parser.parse_args()

    from bijou.eval import report as rt

    analysis = json.loads(Path(args.analysis).read_text())
    cand, top10, stable = load_arms()
    src = {"top10": top10, "stable": stable, "cand": cand}
    preds = {label: src[which][key] for label, which, key in POLICIES}

    truth, valid = cand["truth"], cand["valid"]
    # owner amendment 13:14Z: excluded repos out of every surface here too
    core = cand["core"] & ~np.isin(cand["repo_id"], list(mpr.EXCLUDED_REPOS))
    m2 = (valid[:, :WINDOW] & np.isfinite(truth[:, :WINDOW]).all(-1))[core]
    arms_core = {label: pred[core] for label, pred in preds.items()}
    motor = per_motor_table(arms_core, truth[core], m2, rt)

    core_rows = np.flatnonzero(core)
    stride = max(len(core_rows) // args.samples, 1)
    sampled = core_rows[::stride][: args.samples]
    blocks = gallery(cand, preds, sampled, rt)

    theme = rt.THEMES["dark"]
    tables = summary_tables(analysis, rt)
    contam = analysis["contamination"]
    config_lines = [
        "MolmoAct2 out-of-band 3-policy comparison — record-only",
        f"panel: {mpr.CAND_STEM}.npz (25,800 rows, 17,204 core)",
        "molmoact2: allenai/MolmoAct2-SO100_101 — Molmo2-ER trunk + their flow",
        "  action expert; their predict_action end-to-end, continuous mode,",
        "  10-step Euler, bf16, seed = global concat index",
        "snapflow 80k: banked bijou_flow_artrunk_h1024@80000 panel npzs —",
        "  Gemma-4-E2B trunk (frozen, AR-pretrained) + flow expert h1024",
        "  (top-10-tickets = our best decode config; stable-key single draw)",
        "ar 40k / ar 60k-cont: Molmo2-4B trunk, AR decoder;",
        "  er 60k@15000: Molmo2-ER trunk, AR decoder (mid-training)",
        "PRIMARY WINDOW: chunk steps 0-29 (their native 30-step / 1.0 s horizon);",
        "  our full-50 numbers are secondary and never quoted against theirs",
        f"contamination: {contam['repos']} of 878 panel repos are in their",
        f"  SO-100/101 fine-tune mixture = {contam['core_frames']} core frames",
        "CAVEAT: our arms trained on this panel's repo distribution (holdout",
        "  episodes, same repos) — 'clean' is repo-level OOD for MolmoAct2 only;",
        "  the contaminated split is the closest to a fair fight",
        (
            f"excluded (owner amendment 13:14Z): {analysis['excluded']['repos']}"
            f" — {analysis['excluded']['frames']} frames"
            f" ({analysis['excluded']['core_frames']} core), wraparound-unit truth"
        ),
        "prereg: posts/2026-08-10-prereg-molmoact2-oob-panel.md",
    ]
    extra = "".join(
        f"<h2>{escape(t.title)}</h2>{rt._table(t.header, t.rows)}"
        for t in [*tables, motor]
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MolmoAct2 out-of-band 3-policy report</title>"
        f"<style>{theme.css()}</style></head><body>"
        "<h1>MolmoAct2 vs snapflow-80k vs state-copy — same frames</h1>"
        f"<pre>{escape(chr(10).join(config_lines))}</pre>"
        f"{extra}"
        f"<h2>Sample predictions ({len(blocks)} of {len(core_rows)} core "
        "frames, evenly strided; MolmoAct2's line stops at step 30 — its "
        f"native horizon)</h2>{''.join(blocks)}"
        "</body></html>"
    )
    Path(args.out).write_text(document)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
