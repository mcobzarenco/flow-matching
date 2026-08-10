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
        "flow teacher 80k top10tickets [gemma4-E2B]",
        "top10",
        "pred:bijou@80000_draws10_ticket",
    ),
    ("flow teacher 80k stablekey [gemma4-E2B]", "stable", "pred:bijou@80000"),
    ("MolmoAct2 SO100_101 [molmo2-ER]", "cand", mpr.CAND_KEY),
    ("state-copy [no model]", "cand", "pred:state-copy"),
]
# analysis-json keys (stable) -> displayed names with trunk
DISPLAY = {
    "snapflow80k top10tickets": (
        "flow teacher 80k top-10-tickets — Gemma-4-E2B trunk (frozen, AR-pretrained) + flow expert h1024 (Heun-30)"
    ),
    "snapflow80k stablekey": (
        "flow teacher 80k stable-key — Gemma-4-E2B trunk (frozen, AR-pretrained) + flow expert h1024 (Heun-30)"
    ),
    "molmoact2": "MolmoAct2 SO100_101 — Molmo2-ER trunk + their flow action expert",
    "snapflow80k heun30": (
        "flow teacher 80k heun-30 original single-draw — Gemma-4-E2B trunk + flow expert h1024"
    ),
    "flow80k draws10 seating": (
        "flow teacher 80k mean-of-10 draws (seating keying) — Gemma-4-E2B trunk + flow expert h1024 (Heun-30)"
    ),
    "snapflow student 30k 1nfe": (
        "snapflow student 30k 1-NFE single draw — Gemma-4-E2B trunk + distilled flow expert (1 Euler step)"
    ),
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
        "flow80k draws10 seating",
        "snapflow80k stablekey",
        "snapflow80k heun30",
        "snapflow student 30k 1nfe",
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
        [
            display(label),
            f"{v['chunk_mae']:.4f}",
            f"{v.get('chunk_mae_excl', float('nan')):.4f}",
            f"{v['chunk_mae'] - v.get('chunk_mae_excl', float('nan')):+.4f}",
            f"{v['first_mae']:.4f}",
        ]
        for label, v in analysis["secondary_full50"].items()
    ]
    t3 = rt.ReportTable(
        title=(
            "Secondary: our arms on the full 50-step chunk (never vs theirs) "
            "— as-banked vs willnorris/bbox-2 excluded (the exclusion effect)"
        ),
        header=[
            "policy",
            "chunk_mae (50, banked)",
            "chunk_mae (50, excl bbox-2)",
            "bbox-2 effect",
            "first_mae (banked)",
        ],
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


# 10 mutually-legible series colors on the dark page (state-copy last,
# gray + dashed by convention below).
STEP_COLORS = [
    "#648fff",
    "#ffb000",
    "#dc267f",
    "#785ef0",
    "#fe6100",
    "#4ec9b0",
    "#e8e857",
    "#38d430",
    "#ff7eb6",
    "#9aa0a8",
]


def legend_strip(labels: list, rt: ModuleType) -> str:
    """Standalone legend image (owner 15:22Z: the in-chart legend was
    covering series) — one strip reused above the charts."""
    import base64
    import io

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.use("Agg", force=True)
    theme = rt.THEMES["dark"]
    with plt.style.context(theme.mpl_style):
        fig = plt.figure(figsize=(13, 1.4))
        fig.patch.set_facecolor(theme.page_bg)
        handles = [
            Line2D(
                [0],
                [0],
                color=STEP_COLORS[i % len(STEP_COLORS)],
                linewidth=2.2,
                linestyle="--" if label.startswith("state-copy") else "-",
            )
            for i, label in enumerate(labels)
        ]
        handles.append(Line2D([0], [0], color=theme.truth_color, linewidth=2.2))
        fig.legend(
            handles,
            [*labels, "truth"],
            loc="center",
            ncol=3,
            fontsize=9,
            frameon=False,
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def traj_chart(sample, motor_names: list, order: list, rt: ModuleType) -> str:  # noqa: ANN001
    """Per-joint truth-vs-all-policies chart — the house chart with the
    10-color palette and NO in-axes legend (the strip above carries it)."""
    import base64
    import io

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg", force=True)
    theme = rt.THEMES["dark"]
    dims = sample.truth.shape[-1]
    ncols = 3
    nrows = (dims + ncols - 1) // ncols
    n_valid = int(sample.valid.sum())
    steps = range(n_valid)
    with plt.style.context(theme.mpl_style):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 2.6 * nrows),
            squeeze=False,
        )
        fig.patch.set_facecolor(theme.page_bg)
        for dim in range(dims):
            ax = axes[dim // ncols][dim % ncols]
            ax.set_facecolor(theme.page_bg)
            ax.plot(
                steps,
                sample.truth[:n_valid, dim].tolist(),
                color=theme.truth_color,
                linewidth=1.9,
            )
            for series, name in enumerate(order):
                predicted = sample.predictions[name]
                ax.plot(
                    steps,
                    predicted[:n_valid, dim].tolist(),
                    color=STEP_COLORS[series % len(STEP_COLORS)],
                    linestyle="--" if name.startswith("state-copy") else "-",
                    linewidth=1.1,
                )
            name = motor_names[dim] if dim < len(motor_names) else f"dim {dim}"
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=8)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=90, facecolor=fig.get_facecolor())
        plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def per_step_chart(
    all_preds: dict,
    truth: np.ndarray,
    valid: np.ndarray,
    splits: dict,
    rt: ModuleType,
) -> str:
    """MAE by chunk timestep, one panel per split, all models (owner
    14:35Z/14:38Z). MolmoAct2's curve ends at step 30 (NaN tail)."""
    import base64
    import io

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg", force=True)
    theme = rt.THEMES["dark"]
    n_steps = truth.shape[1]
    with plt.style.context(theme.mpl_style):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        fig.patch.set_facecolor(theme.page_bg)
        handles = []
        for ax, (split_name, sel) in zip(axes, splits.items(), strict=True):
            ax.set_facecolor(theme.page_bg)
            for i, (label, pred) in enumerate(all_preds.items()):
                m = (
                    valid[sel]
                    & np.isfinite(truth[sel]).all(-1)
                    & np.isfinite(pred[sel]).all(-1)
                )
                err = np.abs(np.nan_to_num(pred[sel]) - truth[sel]).mean(-1)
                num = (err * m).sum(0)
                den = np.maximum(m.sum(0), 1)
                curve = np.where(m.sum(0) > 0, num / den, np.nan)
                (line,) = ax.plot(
                    range(n_steps),
                    curve,
                    label=label,
                    color=STEP_COLORS[i % len(STEP_COLORS)],
                    linewidth=1.5,
                    linestyle="--" if label.startswith("state-copy") else "-",
                )
                if ax is axes[0]:
                    handles.append(line)
            ax.set_title(f"{split_name} (n={int(sel.sum())})", fontsize=10)
            ax.set_xlabel("chunk step (30 fps)")
            ax.tick_params(labelsize=8)
            ax.axvline(29.5, color=theme.meta, linewidth=0.8, linestyle=":")
        axes[0].set_ylabel("MAE (raw action units)")
        # legend below the axes, never over the series (owner 15:22Z)
        fig.legend(
            handles,
            list(all_preds),
            loc="lower center",
            ncol=4,
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def gallery(cand: dict, preds: dict, indices: np.ndarray, rt: ModuleType) -> list[str]:
    """Re-fetch images/state/task for the sampled rows; render blocks."""
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
            label: torch.as_tensor(np.ascontiguousarray(pred[row]))
            for label, pred in preds.items()
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
        chart = traj_chart(sample, MOTOR_NAMES, list(preds), rt)
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
    cand, _top10, _stable = load_arms()

    truth, valid = cand["truth"], cand["valid"]
    # owner amendment 13:14Z: excluded repos out of every surface here too
    core = cand["core"] & ~np.isin(cand["repo_id"], list(mpr.EXCLUDED_REPOS))
    m2 = (valid[:, :WINDOW] & np.isfinite(truth[:, :WINDOW]).all(-1))[core]

    # every arm, loaded once; pairing was oracle-verified by the reads
    # run that produced the analysis json
    all_preds = {}
    for spec in mpr.BASELINES:
        npz = mpr._load_npz(f"{spec['stem']}.npz")
        short = spec["label"].replace(" endpoint", "").replace(" continuation", "-cont")
        all_preds[short] = npz[spec["key"]]
    all_preds["MolmoAct2 [molmo2-ER]"] = cand[mpr.CAND_KEY]
    all_preds["state-copy"] = cand["pred:state-copy"]

    arms_core = {label: pred[core] for label, pred in all_preds.items()}
    motor = per_motor_table(arms_core, truth[core], m2, rt)

    # MAE-by-timestep charts, all models x {pooled, clean, contaminated}
    # (owner 14:35Z/14:38Z)
    contam_mask, _stats = mpr.contamination_masks(
        cand["repo_id"],
        cand["core"],
        mpr.mixture_repo_set(),
        None,
    )
    step_splits = {
        "pooled": core,
        "clean": core & ~contam_mask,
        "contaminated": core & contam_mask,
    }
    step_chart = per_step_chart(all_preds, truth, valid, step_splits, rt)
    legend = legend_strip(list(all_preds), rt)
    print("per-step charts rendered", flush=True)

    core_rows = np.flatnonzero(core)
    stride = max(len(core_rows) // args.samples, 1)
    sampled = core_rows[::stride][: args.samples]
    # all models on the trajectory charts (owner 15:22Z)
    blocks = gallery(cand, all_preds, sampled, rt)

    theme = rt.THEMES["dark"]
    tables = summary_tables(analysis, rt)
    contam = analysis["contamination"]
    config_lines = [
        "MolmoAct2 out-of-band 3-policy comparison — record-only",
        f"panel: {mpr.CAND_STEM}.npz (25,800 rows, 17,204 core)",
        "molmoact2: allenai/MolmoAct2-SO100_101 — Molmo2-ER trunk + their flow",
        "  action expert; their predict_action end-to-end, continuous mode,",
        "  10-step Euler, bf16, seed = global concat index",
        "flow teacher 80k: banked bijou_flow_artrunk_h1024@80000 panel npzs —",
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
        "<h1>MolmoAct2 vs flow-teacher-80k vs state-copy — same frames</h1>"
        f"<pre>{escape(chr(10).join(config_lines))}</pre>"
        f"{extra}"
        "<h2>MAE by chunk timestep — all models (dotted line = their "
        "30-step horizon; MolmoAct2 has no prediction past it)</h2>"
        f'<img class="chart" src="{step_chart}">'
        f"<h2>Sample predictions ({len(blocks)} of {len(core_rows)} core "
        "frames, evenly strided; MolmoAct2's line stops at step 30 — its "
        "native horizon). Series legend (applies to every chart below):"
        f'</h2><img class="chart" src="{legend}">{"".join(blocks)}'
        "</body></html>"
    )
    Path(args.out).write_text(document)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
