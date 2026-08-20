"""Standalone HTML eval report for the route-C joint checkpoint's
flow-unseen probe leg (owner request 08:25Z 2026-08-16: "eval report
on unseen ones").

Consumes ONLY the banked leg json (regenerable): per-seed outcome
strip, progress distribution, full per-seed table, and a small clip
gallery copied beside the html so relative links stay live on the
reports Space (sim100_report pattern). House eval-report dark scheme
(sim100_charts constants).

Usage:
  uv run python fontaine/scripts/grasp_sft_joint_unseen_report.py \
      [--leg-json outputs/sim/grasp_sft/joint_probes/flow_unseen.json] \
      [--video-dir outputs/sim/grasp_sft/joint_probes/flow_unseen] \
      [--out-html reports/eval__grasp_sft_joint_step2000__flow_unseen100.html] \
      [--gallery-dir reports/joint_unseen_gallery] \
      [--paired-json reports/analysis__sim100_paired_probe_vs_disc1000.json] \
      [--ladder-b64 reports/pdnorm_panel_ladder.b64] \
      [--truthfit-json reports/analysis__pdnorm_endpoint_truthfit_wear.json]

--paired-json takes a frozen sim100_paired_read.py output and renders
it as a "Paired read" section (delta tiles + discordant-seed chart)
alongside the frozen absolute anchors; the read stays recorded,
non-gating (registered consumer: the pdnorm endpoint vs the disc-1000
demosonly baseline, whose 11/100 sits inside the pre-reg's own 11-19
ambiguous band).

--ladder-b64 embeds a pdnorm_panel_ladder_chart.py --out-b64 sidecar
as a "Panel anchor ladder" figure directly below the meta line's
textual ladder. The pdnormendpoint preset defaults to the chart
script's sidecar path and skips the section quietly when the file is
absent (reports/ is gitignored); an explicit flag is loud on a
missing file.

--truthfit-json renders a pdnorm_endpoint_truthfit_rewear.py output's
ladder_read block as an "estimator seam" line directly under the
ladder figure: the endpoint's native row vs its truth-fit
re-expression, the seam delta, and the truth-fit ladder anchors read
like for like. Same behavior split as --ladder-b64: the
pdnormendpoint preset defaults to the cross-check script's output
path and skips quietly when it is absent (the json exists only once
the ON-GO endpoint npz does); an explicit flag is loud on a missing
file.

The token preset builds the token (AR) head page of the same probe
family: subject leg token_unseen, the token_base leg rendered as a
second per-seed section on the same page, gallery picks pinned to the
decode-diagnosis diagnostics (not the fastest-success default), and a
"Decode diagnosis" section rendering the frozen
token_decode_diagnosis.py output (funnel / envelope / carry-speed
numbers + the committed 4-panel chart). --diagnosis-json follows the
--ladder-b64 behavior split: the preset default (reports/ is
gitignored, the json is regenerable) is skipped quietly when absent,
an explicit flag is loud.

The flowtrain preset builds the flow_train leg's page (seeds
1000-1099, the stage-B collection band) with the memorization split
as the headline: kept vs collector-rejected arms read live from the
banked stage-B collect curve's kept_seeds (the split IS the page, so
a missing collect json fails loudly), Wilson-CI rate chart against
the unseen sibling's 44/100, and gallery picks drawn from both split
arms deterministically (fastest success + nearest miss per arm,
median kept success).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PAGE = "#121417"
CARD = "#1a1e24"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
HEADING = "#eceef1"
SUCCESS = "#dc267f"  # the lineage's banked magenta — the hero series
FAIL = "#648fff"
ANCHOR = "#9aa0a8"

BASE_ANCHOR = 9
CORRUPT_ANCHOR = 28
PROBE_ANCHOR = 44  # route-C joint step2000, flow-unseen (banked 08-16)
V1_ENDPOINT_ANCHOR = 5  # v1 run-2 step3000, flow-unseen (banked 08-17)
DISC1000_ANCHOR = 11  # disc step1000 demosonly baseline (banked 04:19Z 08-18)
PDNORM_ENDPOINT_ANCHOR = 1  # pdnorm convicted mixed cell (banked 03:17Z 08-19)
TOKEN_BASE_ANCHOR = 0  # base (no SFT), token head (leg 4, banked 09:01Z 08-19)

# Sentinel gallery pick: the farthest spawn whose jaws never made
# contact (grip is contact-coded in rollout_sim, so no-touch reads
# directly off the trace) — the reach-envelope failure clip.
FAR_SPAWN_NO_TOUCH = "far_spawn_no_touch"

DEFAULT_LEG_JSON = Path("outputs/sim/grasp_sft/joint_probes/flow_unseen.json")
DEFAULT_VIDEO_DIR = Path("outputs/sim/grasp_sft/joint_probes/flow_unseen")
DEFAULT_OUT_HTML = Path(
    "reports/eval__grasp_sft_joint_step2000__flow_unseen100.html",
)
DEFAULT_GALLERY_DIR = Path("reports/joint_unseen_gallery")

# Page variants: step2000 (the original probe report, byte-identical
# output) and v1endpoint (grasp_sft_v1_joint step 3000 on the 5,000-demo
# corpus, anchored additionally on the probe checkpoint's 44/100).
PRESETS: dict[str, dict] = {
    "step2000": {
        "anchor_rows": [
            ("base (no SFT)", BASE_ANCHOR, ANCHOR),
            ("stage-C AE, corrupt table", CORRUPT_ANCHOR, "#f593bd"),
        ],
        "subject_label": "joint step2000, corrected",
        "anchors_tile_label": "anchors: base / corrupt-table",
        "title": "joint step2000 — flow head, unseen 100",
        "h1": "Grasp-SFT route C — joint step 2000, flow head on unseen seeds",
        "meta_html": (
            "Checkpoint <code>fontaine_grasp_sft_joint_corrected/step_002000</code>"
            " (corrected norm table, insulated joint objective, registered amendment"
            " 2026-08-16) · euler-10, execute-horizon 30, seeds 0–99, 30 s episodes ·"
            " run 2026-08-16 06:58–08:21Z, ~1.4 GPU-h · verdict surface A §5:"
            ' <b style="color:{success}">TABLE_FIX_POSITIVE</b> (44 &gt; 28+3)'
        ),
    },
    "v2endpoint": {
        "anchor_rows": [
            ("base (no SFT)", BASE_ANCHOR, ANCHOR),
            ("joint probe step2000 (313 demos)", PROBE_ANCHOR, "#f593bd"),
            ("v1 step3000 (broken table)", V1_ENDPOINT_ANCHOR, "#f593bd"),
        ],
        "subject_label": "v2 demosonly step3000",
        "anchors_tile_label": "anchors: base / probe / v1 endpoint",
        "title": "grasp_sft_v2_demosonly step3000 — flow head, unseen 100",
        "h1": "Grasp-SFT v2 demos-only — joint step 3000, flow head on unseen seeds",
        "meta_html": (
            "Checkpoint <code>grasp_sft_v2_demosonly_8xa100/step_003000</code>"
            " (v2 regen corpus ONLY — expert v1.3, 1.75M frames, rig datasets"
            " out per the owner's 11:27Z 08-17 order, so the recomputed table"
            " is demos-native; eff-96, 8×A100, launched 11:38:30Z; the mixed"
            " v2 run was killed at ~1150 on rising train MAE) · euler-10,"
            " execute-horizon 30, seeds 0–99, 30 s episodes, sharded 4×25"
            " (exact: triple-keyed noise) · isolation grid: ≥ probe band 44 ⇒"
            " mix/table was the poison; ~5 ⇒ suspicion moves past the mix"
        ),
    },
    "v1endpoint": {
        "anchor_rows": [
            ("base (no SFT)", BASE_ANCHOR, ANCHOR),
            ("stage-C AE, corrupt table", CORRUPT_ANCHOR, "#f593bd"),
            ("joint probe step2000 (313 demos)", PROBE_ANCHOR, "#f593bd"),
        ],
        "subject_label": "v1 step3000 (5,000 demos)",
        "anchors_tile_label": "anchors: base / corrupt-table / probe",
        "title": "grasp_sft_v1_joint step3000 — flow head, unseen 100",
        "h1": "Grasp-SFT v1 — joint step 3000, flow head on unseen seeds",
        "meta_html": (
            "Checkpoint <code>grasp_sft_v1_joint_8xa100/step_003000</code>"
            " (5,000-demo corpus + pick_place ×4, eff-96, 8×A100; run 2 —"
            " <code>--recompute-stats</code> restart launched 21:14:48Z"
            " 2026-08-16 on the owner's order, run 1b killed at ~1900) ·"
            " euler-10, execute-horizon 30, seeds 0–99,"
            " 30 s episodes, sharded 4×25 (exact: triple-keyed noise) ·"
            " primary anchor: the probe checkpoint's <b>44/100</b>"
        ),
    },
    "disc1000": {
        "anchor_rows": [
            ("base (no SFT)", BASE_ANCHOR, ANCHOR),
            ("joint probe step2000 (313 demos)", PROBE_ANCHOR, "#f593bd"),
            ("v1 step3000 (5k demos + rig mix)", V1_ENDPOINT_ANCHOR, "#f593bd"),
        ],
        "subject_label": "disc step1000 (demosonly v2, 1×H100)",
        "anchors_tile_label": "anchors: base / probe / v1-endpoint",
        "title": "disc step1000 — flow head, unseen 100 (demosonly-v2 cell)",
        "h1": "Drift discriminator — step 1000, flow head on unseen seeds",
        "meta_html": (
            "Checkpoint <code>grasp_sft_v2_demosonly_1gpu_disc/step_001000</code>"
            " — the first non-drifting v2-corpus checkpoint (verdict HEALTHY"
            " 00:42Z 08-18; demos-only corpus, 1×H100, honest recomputed"
            " stats) · euler-10, execute-horizon 30, seeds 0–99, 30 s"
            " episodes, default worn-row lookup (no rig key ⇒ merged"
            " demos-native table) · fills the demosonly-v2 grasp cell of the"
            " isolation grid; baseline arm of the"
            " <code>--per-dataset-flow-norm</code> pre-reg"
        ),
        "paired_band_note": (
            " The demosonly baseline's 11/100 sits inside the pdnorm"
            " draft's own 11&ndash;19 ambiguous band, so the paired read"
            " rides alongside the frozen absolute bands &mdash; it is"
            " recorded, never gating."
        ),
    },
    # ON-GO endpoint page for the pdnorm pre-reg (posts/2026-08-xx-prereg-
    # grasp-sft-v2-joint-pdnorm.md). Checkpoint/run fields are
    # FILL-AT-ENDPOINT placeholders the endpoint session stamps; the
    # bands and panel anchors below are the pre-reg's frozen values.
    "pdnormendpoint": {
        "anchor_rows": [
            ("base (no SFT)", BASE_ANCHOR, ANCHOR),
            ("joint probe step2000 (313 demos)", PROBE_ANCHOR, "#f593bd"),
            ("disc1000 demosonly baseline (paired arm)", DISC1000_ANCHOR, "#f593bd"),
        ],
        "subject_label": "pdnorm endpoint step3000",
        "anchors_tile_label": "anchors: base / probe / disc1000 baseline",
        "title": "pdnorm endpoint step3000 — flow head, unseen 100",
        "h1": "Grasp-SFT v2 joint pdnorm — endpoint, flow head on unseen seeds",
        "meta_html": (
            "Checkpoint <code>grasp_sft_v2_joint_1gpu_pdnorm/step_003000</code>"
            " (per-dataset flow norm, mixed v2 corpus; launched"
            " 2026-08-18 11:02Z, run complete 00:1xZ 08-19, ~12.9"
            " GPU-h) · euler-10, execute-horizon 30, seeds 0–99, 30 s"
            " episodes; sim leg wears the sim demos' row (frozen"
            " serving-row rule) · frozen decision grid: &le;10"
            " broken-class band / 11&ndash;19 ambiguous band /"
            ' <b style="color:{success}">&ge;20 exonerates the mix</b> ·'
            " panel anchors, wear-corrected class: 27.40 re-worn"
            " disc-1000 / 25.89 released pre-SFT (measured 08-18, at"
            " the null) / 25.15 repo-midpoint null / 8.37 state-copy"
            " (real bar) · verdict: <b>1/100 &le; 10 — CONVICT: the mix"
            " is the prime suspect</b> (read taken 03:17:39Z 08-19;"
            " panel endpoint 29.18 native / 27.44 truth-fit)"
        ),
        "paired_band_note": (
            " The demosonly baseline's 11/100 sits inside the pdnorm"
            " draft's own 11&ndash;19 ambiguous band, so the paired read"
            " rides alongside the frozen absolute bands &mdash; it is"
            " recorded, never gating."
        ),
        "ladder_b64": Path("reports/pdnorm_panel_ladder.b64"),
        "truthfit_json": Path(
            "reports/analysis__pdnorm_endpoint_truthfit_wear.json",
        ),
    },
    # Endpoint page for the demos+one-rig pre-reg (posts/2026-08-19-
    # prereg-demos-plus-one-rig.md) — the two-dataset mix cell that
    # exonerated mixing (verdict 10:5xZ 08-20). Same battery shape as
    # pdnormendpoint; anchors add the convicted three-way cell.
    "onerigendpoint": {
        "leg_json": Path("outputs/sim/grasp_sft/onerig_endpoint/flow_unseen.json"),
        "video_dir": Path("outputs/sim/grasp_sft/onerig_endpoint/flow_unseen"),
        "out_html": Path(
            "reports/eval__grasp_sft_v2_joint_1gpu_pdnorm_onerig"
            "__step_003000__flow_unseen100.html",
        ),
        "gallery_dir": Path("reports/onerig_unseen_gallery"),
        "anchor_rows": [
            ("base (no SFT)", BASE_ANCHOR, ANCHOR),
            ("disc1000 demosonly control (paired arm)", DISC1000_ANCHOR, "#f593bd"),
            (
                "pdnorm convicted mixed cell (paired arm)",
                PDNORM_ENDPOINT_ANCHOR,
                "#f593bd",
            ),
            ("joint probe step2000 (313 demos)", PROBE_ANCHOR, "#f593bd"),
        ],
        "subject_label": "onerig endpoint step3000",
        "anchors_tile_label": "anchors: base / control / convicted mix / probe",
        "title": "onerig endpoint step3000 — flow head, unseen 100",
        "h1": "Grasp-SFT demos+one-rig — endpoint, flow head on unseen seeds",
        "meta_html": (
            "Checkpoint <code>grasp_sft_v2_joint_1gpu_pdnorm_onerig/"
            "step_003000</code> (demos + so101_pick_place_v2 ×4 ONLY —"
            " clean dropped, ~6% rig share; per-dataset flow norm;"
            " launched 18:47Z 08-19, train complete 08:2xZ 08-20, ~13.4"
            " GPU-h) · euler-10, execute-horizon 30, seeds 0–99, 30 s"
            " episodes; sim leg wears the sim demos' row + stand-ins"
            " substrate pin (frozen serving rules) · frozen decision"
            " grid: &le;10 interference-reproduced / 11&ndash;19"
            ' ambiguous / <b style="color:{success}">&ge;20 the mix'
            " grasps</b> · verdict: <b>28/100 &ge; 20 —"
            " MIX-EXONERATED</b> (read 10:5xZ 08-20); paired +17 vs the"
            " 11/100 control (CI95 [8, 26], McNemar p = 0.0009), +27 vs"
            " the convicted cell's 1/100 (CI95 [19, 36], p = 1.5e-8) ·"
            " panel endpoint 28.81 native / 27.26 truth-fit vs 27.40"
            " re-worn disc-1000 / 27.14 released / 25.15 null / 8.37"
            " state-copy"
        ),
        "paired_band_note": (
            " The demosonly control's 11/100 sits inside this pre-reg's"
            " own 11&ndash;19 ambiguous band, so the paired read rides"
            " alongside the frozen absolute bands &mdash; it is"
            " recorded, never gating."
        ),
        "ladder_b64": Path("reports/onerig_panel_ladder.b64"),
        "truthfit_json": Path(
            "reports/analysis__onerig_endpoint_truthfit_wear.json",
        ),
    },
    # Token (AR) head page of the same route-C probe family: subject =
    # token_unseen (leg 3, re-run finished 06:17Z 08-19 after the 08-16
    # owner pause at seed 24), second leg = token_base (leg 4, finished
    # 09:01:18Z 08-19). Gallery picks pinned to the decode-diagnosis
    # diagnostics; the diagnosis json + committed 4-panel chart render
    # as their own section.
    "token": {
        "leg_json": Path("outputs/sim/grasp_sft/joint_probes/token_unseen.json"),
        "video_dir": Path("outputs/sim/grasp_sft/joint_probes/token_unseen"),
        "out_html": Path(
            "reports/eval__grasp_sft_joint_step2000__token_unseen100.html",
        ),
        "gallery_dir": Path("reports/joint_token_gallery"),
        "anchor_rows": [
            ("base (no SFT), token head", TOKEN_BASE_ANCHOR, ANCHOR),
            ("same checkpoint, flow head", PROBE_ANCHOR, "#f593bd"),
        ],
        "subject_label": "joint step2000, token head",
        "anchors_tile_label": "anchors: token base / flow sibling",
        "title": "joint step2000 — token head, unseen 100",
        "h1": "Grasp-SFT route C — joint step 2000, token (AR) head on unseen seeds",
        "meta_html": (
            "Checkpoint <code>fontaine_grasp_sft_joint_corrected/step_002000</code>"
            " (the flow-leg page's checkpoint, served through the token head:"
            " grammar-constrained greedy decode, no temperature) ·"
            " execute-horizon 30, seeds 0–99, 30 s episodes · leg 3 re-run"
            " finished 06:17Z 08-19 (first attempt owner-paused at seed 24 on"
            " 08-16), leg 4 token-base finished 09:01:18Z 08-19, ~2.2–2.4"
            " GPU-h each · verdict surface B §3: <b>7/100 — OWNER_DECISION"
            " band (5–19)</b>; SFT delta over the token base +7 (0 → 7), real"
            " but ~5× below the flow sibling's +35 (9 → 44) · decode"
            ' diagnosis (08-19, CPU): <b style="color:{success}">greedy'
            " magnitude attenuation, not calibration</b> — recommendation"
            " posted: activate R2 from the 7% checkpoint (R2 samples at"
            " T=1.0)"
        ),
        "gallery_picks": [
            ("flow-overlap success", 35),
            ("flow-overlap success", 96),
            ("timeout-holding carry (clock runs out, boat in hand)", 29),
            ("timeout-holding carry (clock runs out, boat in hand)", 41),
            ("far-spawn no-touch", FAR_SPAWN_NO_TOUCH),
        ],
        "diagnosis_json": Path("reports/analysis__token_decode_diagnosis.json"),
        "diagnosis_chart": Path(
            "fontaine/blog/src/img/grasp_sft/token_decode_diagnosis.png",
        ),
        "second_leg": (
            "Token base (no SFT) — leg 4 anchor",
            Path("outputs/sim/grasp_sft/joint_probes/token_base.json"),
        ),
    },
    # Flow head on the TRAINING band (leg 2, seeds 1000-1099 — the
    # stage-B collection block the 313-demo corpus was drawn from).
    # Headline = the memorization split: the kept arm (a demo from the
    # seed's scene entered the SFT corpus) vs the collector-rejected
    # arm, membership read live from the stage-B collect curve's
    # kept_seeds. The split is the page, so the collect json is a loud
    # requirement, not a quiet default.
    "flowtrain": {
        "leg_json": Path("outputs/sim/grasp_sft/joint_probes/flow_train.json"),
        "video_dir": Path("outputs/sim/grasp_sft/joint_probes/flow_train"),
        "out_html": Path(
            "reports/eval__grasp_sft_joint_step2000__flow_train100.html",
        ),
        "gallery_dir": Path("reports/joint_train_gallery"),
        "seed_start": 1000,
        "strip_xlabel": "training seed (1000–1099, stage-B collection band)",
        "bar_xlabel": "successes per 100 seeds",
        "anchor_rows": [
            ("base (no SFT), unseen", BASE_ANCHOR, ANCHOR),
            ("same checkpoint, unseen 100", PROBE_ANCHOR, "#f593bd"),
        ],
        "subject_label": "joint step2000, train seeds",
        "anchors_tile_label": "anchors: base / unseen sibling",
        "title": "joint step2000 — flow head, train 100",
        "h1": "Grasp-SFT route C — joint step 2000, flow head on training seeds",
        "meta_html": (
            "Checkpoint <code>fontaine_grasp_sft_joint_corrected/step_002000</code>"
            " (the unseen-leg page's checkpoint, flow head served on the"
            " training band) · euler-10, execute-horizon 30, seeds 1000–1099"
            " — the stage-B collection band the 313-demo SFT corpus was drawn"
            " from (64 kept / 36 collector-rejected among these 100 scenes) ·"
            " leg 2 of the probe chain, finished 09:47Z 08-16, ~1.4 GPU-h"
            " (registered A §4: unseen 0–99, then train band 1000–1099) ·"
            ' headline: <b style="color:{success}">42/100 ≈ the unseen'
            " sibling's 44/100 — no memorization signature</b>; kept 29/64"
            " vs rejected 13/36, CIs overlap (split section below)"
        ),
        "split_json": Path("reports/curve__grasp_sft_stageb_collect.json"),
    },
}


def fig_to_b64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style_ax(ax: Axes) -> None:
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=META, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(HEADING)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.5)


def outcome_strip(eps: list[dict], xlabel: str = "unseen seed (0–99)") -> str:
    # Seed positions are plotted verbatim, so the axis window follows the
    # leg's band (0-99 unseen, 1000-1099 train) — lo = 0 reproduces the
    # original page byte-for-byte.
    lo = min(e["seed"] for e in eps)
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=PAGE)
    for e in sorted(eps, key=lambda e: e["seed"]):
        ok = e.get("success_tick") is not None
        color = SUCCESS if ok else FAIL
        final = min(e["final_cm"], 14.0)
        ax.plot(
            [e["seed"], e["seed"]],
            [e["initial_cm"], final],
            color=color,
            linewidth=1.4,
            alpha=0.9 if ok else 0.55,
        )
        ax.plot([e["seed"]], [final], "o", color=color, markersize=2.6)
    ax.axhline(3.0, color=ANCHOR, linewidth=0.8, linestyle="--")
    ax.text(
        lo + 100.5,
        3.0,
        "success radius 3 cm",
        color=META,
        fontsize=8,
        va="center",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("boat → disk center (cm)")
    ax.set_title(
        "Per-seed outcome: spawn distance → final distance "
        "(magenta = success, blue = miss)",
    )
    style_ax(ax)
    ax.set_xlim(lo - 1.5, lo + 113)
    return fig_to_b64(fig)


def success_bar(
    rows: list[tuple[str, int, str]],
    xlabel: str = "successes on unseen seeds 0–99",
) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 2.6), facecolor=PAGE)
    for i, (label, val, color) in enumerate(rows):
        ax.barh(i, val, color=color, height=0.55)
        ax.text(val + 1, i, f"{val}/100", color=TEXT, fontsize=10, va="center")
        ax.text(-1.5, i, label, color=TEXT, fontsize=9, va="center", ha="right")
    ax.set_yticks([])
    ax.set_xlim(0, max(60, int(max(v for _, v, _ in rows) * 1.2)))
    ax.set_xlabel(xlabel)
    ax.set_title("Against the frozen anchors")
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.34)
    return fig_to_b64(fig)


def discordant_bar(table: dict[str, int], label_a: str, label_b: str) -> str:
    # Concordant cells share the muted anchor gray on purpose: only the
    # discordant seeds decide a McNemar read.
    rows = [
        (f"{label_a} only", table["a_only"], SUCCESS, 1.0),
        (f"{label_b} only", table["b_only"], FAIL, 1.0),
        ("both succeed", table["both_succeed"], ANCHOR, 0.55),
        ("both fail", table["both_fail"], ANCHOR, 0.55),
    ]
    xmax = max(10, int(max(v for _, v, _, _ in rows) * 1.25))
    fig, ax = plt.subplots(figsize=(6.4, 2.6), facecolor=PAGE)
    for i, (label, val, color, alpha) in enumerate(rows):
        ax.barh(i, val, color=color, height=0.55, alpha=alpha)
        ax.text(val + xmax * 0.015, i, str(val), color=TEXT, fontsize=10, va="center")
        ax.text(
            -xmax * 0.025,
            i,
            label,
            color=TEXT,
            fontsize=9,
            va="center",
            ha="right",
        )
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_xlabel("seeds (paired on the shared unseen set)")
    ax.set_title("Discordant seeds decide the paired read")
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.34)
    return fig_to_b64(fig)


def paired_section(payload: dict, band_note: str) -> str:
    """Render a frozen sim100_paired_read.py output as an html section:
    meta line (role + pre-reg + preset band note), delta tiles, and the
    McNemar discordant-seed chart."""
    label_a = payload["arms"]["a"]["label"]
    label_b = payload["arms"]["b"]["label"]
    read = payload["read"]
    succ, disc, prog = read["success"], read["discordant"], read["progress"]

    def ci_text(ci: list[float], *, excludes: bool) -> str:
        return f"CI95 [{ci[0]:g}, {ci[1]:g}], {'excludes' if excludes else 'spans'} 0"

    count_ci = ci_text(
        succ["count_delta_ci95"],
        excludes=succ["ci_excludes_zero"],
    )
    tiles = [
        (
            f"{succ['count_delta']:+d}",
            (
                f"success-count delta, {succ['count_a']} vs {succ['count_b']}"
                f" ({count_ci})"
            ),
        ),
        (
            f"p = {disc['mcnemar_exact_p_two_sided']:.1e}",
            f"McNemar exact, {disc['a_only']} vs {disc['b_only']} discordant seeds",
        ),
        (
            f"{prog['mean_delta_cm']:+.2f} cm",
            (
                "mean progress delta"
                f" ({ci_text(prog['ci95'], excludes=prog['ci_excludes_zero'])})"
            ),
        ),
        (
            f"{prog['win_rate']:.0%}",
            f"per-seed progress win rate ({prog['tie_rate']:.0%} ties)",
        ),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="big">{v}</div><div class="lab">{html.escape(k)}</div></div>'
        for v, k in tiles
    )
    strikes = read["reset_strikes"]
    return (
        f"<h2>Paired read: {html.escape(label_a)} vs {html.escape(label_b)}</h2>\n"
        f'<p class="meta">{html.escape(payload["role"])} · pre-reg'
        f" <code>{html.escape(payload['prereg'])}</code> · paired on the"
        f" {read['n_seeds']} shared unseen seeds · reset strikes"
        f" {strikes['a']}/{strikes['b']}.{band_note}</p>\n"
        f'<div class="tiles">{tiles_html}</div>\n'
        f'<img src="data:image/png;base64,'
        f'{discordant_bar(disc, label_a, label_b)}">\n'
    )


def ladder_section(b64_path: Path) -> str:
    """Render a pdnorm_panel_ladder_chart.py --out-b64 sidecar as an
    embedded-figure section (sits below the meta line's textual
    ladder). The payload must be a base64 PNG — the sidecar is written
    from the rendered chart's bytes, so anything else is a stale or
    corrupt file."""
    payload = b64_path.read_text().strip()
    assert base64.b64decode(payload)[:8] == b"\x89PNG\r\n\x1a\n", (
        f"{b64_path} is not a base64-PNG ladder sidecar"
    )
    return (
        "<h2>Panel anchor ladder</h2>\n"
        '<p class="meta">Wear-corrected class (wear audit 08-18) &mdash;'
        " the endpoint reads against this class, never the raw 58.14."
        " The endpoint session re-runs"
        " <code>pdnorm_panel_ladder_chart.py --endpoint &lt;row&gt;</code>"
        " on GO, so the embedded sidecar"
        f" <code>{html.escape(b64_path.name)}</code> carries the stamped"
        " magenta rung.</p>\n"
        f'<img src="data:image/png;base64,{payload}" '
        'alt="pdnorm panel anchor ladder">\n'
    )


def estimator_seam_line(payload: dict) -> str:
    """Render a pdnorm_endpoint_truthfit_rewear.py output's ladder_read
    block as an "estimator seam" line under the ladder figure: the
    endpoint's native row vs its truth-fit re-expression, the seam
    delta, and the truth-fit ladder anchors read like for like."""
    read = payload["ladder_read"]
    assert {
        "endpoint_native",
        "endpoint_truthfit",
        "estimator_seam_delta",
        "sft_disc1000_truthfit",
        "null_repo_midpoint",
    } <= read.keys(), "ladder_read is missing seam keys — stale or foreign json"
    released = read.get("released_truthfit")
    released_txt = "" if released is None else f" / released {released:.2f}"
    return (
        '<p class="meta"><b>Estimator seam</b>'
        " (<code>pdnorm_endpoint_truthfit_rewear.py</code>, output-side"
        " re-expression): endpoint native"
        f" <b>{read['endpoint_native']:.2f}</b> &rarr; truth-fit"
        f" {read['endpoint_truthfit']:.2f} (seam"
        f" {read['estimator_seam_delta']:+.2f}); truth-fit ladder:"
        f" disc-1000 {read['sft_disc1000_truthfit']:.2f}{released_txt}"
        f" / repo-midpoint null {read['null_repo_midpoint']:.2f}."
        " The native row stays the headline (a served rig wears recorded"
        " tables); the truth-fit row reads the ladder like for"
        " like.</p>\n"
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def split_rate_bar(rows: list[tuple[str, int, int, str]]) -> str:
    """Horizontal success-RATE bars (the arms' denominators differ, so
    counts don't compare) with Wilson-CI whiskers; every bar carries its
    own count label, so identity never rides on color alone."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6), facecolor=PAGE)
    for i, (label, k, n, color) in enumerate(rows):
        rate = 100 * k / n
        lo, hi = (100 * v for v in wilson_ci(k, n))
        ax.barh(i, rate, color=color, height=0.55)
        ax.plot([lo, hi], [i, i], color=META, linewidth=1.2)
        ax.plot([lo, lo], [i - 0.12, i + 0.12], color=META, linewidth=1.2)
        ax.plot([hi, hi], [i - 0.12, i + 0.12], color=META, linewidth=1.2)
        ax.text(
            hi + 1.5,
            i,
            f"{k}/{n} = {rate:.0f}%",
            color=TEXT,
            fontsize=10,
            va="center",
        )
        ax.text(-2, i, label, color=TEXT, fontsize=9, va="center", ha="right")
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("success rate, % (whiskers: Wilson CI95)")
    ax.set_title("Memorization split — rates, not counts")
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.30)
    return fig_to_b64(fig)


def memorization_section(eps: list[dict], kept_seeds: set[int]) -> str:
    """The flowtrain page's headline: split the training-band episodes
    by stage-B corpus membership and read kept vs collector-rejected vs
    the unseen sibling. Every number is computed live from the banked
    leg json + kept_seeds — nothing is retyped."""
    kept = [e for e in eps if e["seed"] in kept_seeds]
    rej = [e for e in eps if e["seed"] not in kept_seeds]
    assert kept and rej, "memorization split needs both arms non-empty"

    def stats(arm: list[dict]) -> tuple[int, int, float]:
        succ = sum(1 for e in arm if e.get("success_tick") is not None)
        prog = [e["initial_cm"] - min(e["min_cm"], e["final_cm"]) for e in arm]
        return succ, len(arm), sum(prog) / len(prog)

    ks, kn, kprog = stats(kept)
    rs, rn, rprog = stats(rej)
    klo, khi = wilson_ci(ks, kn)
    rlo, rhi = wilson_ci(rs, rn)
    gap_pp = 100 * (ks / kn - rs / rn)
    tiles = [
        (
            f"{ks}/{kn}",
            f"kept arm ({100 * ks / kn:.0f}%, CI {100 * klo:.0f}–{100 * khi:.0f}%)",
        ),
        (
            f"{rs}/{rn}",
            f"rejected arm ({100 * rs / rn:.0f}%, CI {100 * rlo:.0f}–{100 * rhi:.0f}%)",
        ),
        (f"{gap_pp:+.0f} pp", "kept − rejected rate gap (CIs overlap)"),
        (
            f"{PROBE_ANCHOR}/100",
            "unseen sibling — the anchor the kept arm must beat to claim memorization",
        ),
        (f"{kprog:.2f} vs {rprog:.2f} cm", "mean progress, kept vs rejected"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="big">{v}</div><div class="lab">{html.escape(k)}</div></div>'
        for v, k in tiles
    )
    chart = split_rate_bar(
        [
            ("kept (demo in corpus)", ks, kn, SUCCESS),
            ("collector-rejected", rs, rn, FAIL),
            ("unseen sibling (anchor)", PROBE_ANCHOR, 100, ANCHOR),
        ],
    )
    return (
        "<h2>Memorization split: kept vs collector-rejected scenes</h2>\n"
        '<p class="meta">Membership from the banked stage-B collect'
        " curve's <code>kept_seeds</code>: a <b>kept</b> training seed"
        " contributed a demo to the 313-demo SFT corpus; a"
        " <b>rejected</b> seed's scene was attempted but the scripted"
        " collector failed there, so the model never saw it (and those"
        " scenes skew harder — the rejection reason confounds the arms'"
        f" gap). The decisive read is kept vs unseen: <b>{ks}/{kn}"
        f" ({100 * ks / kn:.0f}%) on scenes the model trained on vs"
        f" {PROBE_ANCHOR}/100 on scenes it never saw</b> — no"
        " memorization signature; the checkpoint generalizes rather than"
        " replays. The kept−rejected gap"
        f" ({gap_pp:+.0f} pp) sits well inside the overlapping Wilson"
        " CIs.</p>\n"
        f'<div class="tiles">{tiles_html}</div>\n'
        f'<img src="data:image/png;base64,{chart}">\n'
    )


def touched(e: dict) -> bool:
    """token_decode_diagnosis.py's contact predicate: grip is
    contact-coded in rollout_sim, so any grip > 0 means the jaws made
    contact with the benchy."""
    return any(g > 0 for g in e.get("grip", []))


def leg_table(eps: list[dict]) -> str:
    rows_html = ""
    for e in sorted(eps, key=lambda e: e["seed"]):
        ok = e.get("success_tick") is not None
        rows_html += (
            f"<tr{' class=ok' if ok else ''}><td>{e['seed']}</td>"
            f"<td>{e['initial_cm']:.1f}</td><td>{e['min_cm']:.1f}</td>"
            f"<td>{e['final_cm']:.1f}</td>"
            f"<td>{e['initial_cm'] - min(e['min_cm'], e['final_cm']):.1f}</td>"
            f"<td>{'✓ @ ' + str(e['success_tick']) if ok else '—'}</td></tr>\n"
        )
    return (
        "<table><tr><th>seed</th><th>spawn cm</th><th>min cm</th>"
        "<th>final cm</th><th>progress cm</th><th>success</th></tr>\n"
        f"{rows_html}</table>\n"
    )


def diagnosis_section(diag: dict, chart_png: Path) -> str:
    """Render the frozen token_decode_diagnosis.py output as a section:
    funnel / envelope / carry-speed tiles, the committed 4-panel chart,
    and the taxonomy + verdict paragraph. Every number is read from the
    json — nothing on the page is retyped."""
    fu, tu, tb = (
        diag["funnel"][a] for a in ("flow_unseen", "token_unseen", "token_base")
    )
    far = {
        a: sum(b["success"] for b in diag["reach_envelope"][a] if b["bin"][0] >= 10.0)
        for a in ("flow_unseen", "token_unseen")
    }
    cs = diag["carry_speed"]
    tax = diag["pinch_fail_taxonomy"]["token_unseen"]
    tax_fu = diag["pinch_fail_taxonomy"]["flow_unseen"]
    ov = diag["success_overlap_token_vs_flow"]
    ratio = diag["headline"]["carry_speed_ratio_token_over_flow"]

    def seeds(xs: list[int]) -> str:
        return ", ".join(str(s) for s in xs)

    tiles = [
        (
            f"{tu['touch']} → {tu['pinch']} → {tu['success']}",
            (
                "token funnel: touch → pinch → success"
                f" (flow {fu['touch']} → {fu['pinch']} → {fu['success']};"
                f" base {tb['touch']} → {tb['pinch']} → {tb['success']})"
            ),
        ),
        (
            f"{far['token_unseen']} vs {far['flow_unseen']}",
            "successes past 10 cm spawn, token vs flow",
        ),
        (
            (
                f"{cs['token_unseen']['median_cm_per_s']:.2f} vs"
                f" {cs['flow_unseen']['median_cm_per_s']:.2f}"
            ),
            f"median carry speed cm/s, token vs flow (ratio {ratio})",
        ),
        (
            f"{tu['knock_aways_gt1cm']} vs {fu['knock_aways_gt1cm']}",
            "knock-aways >1 cm, token vs flow",
        ),
        (
            f"{len(ov['overlap_seeds'])} + {len(ov['token_only'])}",
            (
                f"successes shared with flow ({seeds(ov['overlap_seeds'])})"
                f" + token-only ({seeds(ov['token_only'])})"
            ),
        ),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="big">{v}</div><div class="lab">{html.escape(k)}</div></div>'
        for v, k in tiles
    )
    chart_b64 = base64.b64encode(chart_png.read_bytes()).decode()
    tc, fc = tax["classes"], tax_fu["classes"]
    return (
        "<h2>Decode diagnosis: greedy magnitude attenuation</h2>\n"
        '<p class="meta">Frozen <code>token_decode_diagnosis.py</code> read'
        " (08-19, CPU-only, computed from the banked probe episodes +"
        " videos). SFT bought <i>reach</i> (touch"
        f" {tb['touch']} → {tu['touch']}) but greedy decode under-commands"
        " amplitude everywhere past contact: touch collapses with spawn"
        " distance, grasped carries move at"
        f" {ratio:.0%} of flow's speed, and {tax['n']} pinch-failures split"
        f" {tc['wrong_way']} wrong-way / {tc['stalled_carry']} stalled-carry /"
        f" {tc['timeout_holding']} timeout-holding (flow, n={tax_fu['n']}:"
        f" {fc['wrong_way']}/{fc['stalled_carry']}/{fc['timeout_holding']})."
        " Zero frozen episodes in any arm (motion instrument over all 300"
        " videos) — the 08-13 no-op-chunk class is retired; this is the"
        " ar-draws mean-collapse shape in closed loop.</p>\n"
        f'<div class="tiles">{tiles_html}</div>\n'
        f'<img src="data:image/png;base64,{chart_b64}"'
        ' alt="token decode diagnosis: funnel, reach envelope, carry speed,'
        ' carry traces">\n'
    )


def second_leg_section(label: str, eps: list[dict]) -> str:
    """Render a second probe leg on the same page: headline meta line,
    outcome strip, and the per-seed table collapsed behind a details
    element (an all-fail anchor leg earns the record, not the scroll)."""
    succ = sum(1 for e in eps if e.get("success_tick") is not None)
    contact = sum(1 for e in eps if touched(e))
    moved = sum(
        1 for e in eps if e["initial_cm"] - min(e["min_cm"], e["final_cm"]) > 0.5
    )
    strikes = sum(e.get("reset_strikes", 0) for e in eps)
    return (
        f"<h2>{html.escape(label)}</h2>\n"
        f'<p class="meta"><b>{succ}/100</b> successes · {contact}/100 made'
        f" contact · {moved}/100 moved the boat &gt;0.5 cm · {strikes} reset"
        " strikes — the no-SFT token head barely reaches the boat; the +7"
        " delta above is real signal.</p>\n"
        f'<img src="data:image/png;base64,{outcome_strip(eps)}">\n'
        "<details><summary>per-seed table (token base)</summary>\n"
        f"{leg_table(eps)}</details>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    # Path defaults are per-preset (resolved after parse); an explicit
    # flag always wins. Presets without their own paths keep the
    # original flow-leg defaults byte-for-byte.
    parser.add_argument("--leg-json", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--gallery-dir", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="step2000",
    )
    parser.add_argument(
        "--paired-json",
        type=Path,
        default=None,
        help="frozen sim100_paired_read.py output to render as a paired-read"
        " section (recorded, non-gating)",
    )
    parser.add_argument(
        "--ladder-b64",
        type=Path,
        default=None,
        help="panel anchor-ladder b64 sidecar (pdnorm_panel_ladder_chart.py"
        " --out-b64) to embed below the meta line; defaults to the preset's"
        " sidecar path, skipped quietly when that default is absent",
    )
    parser.add_argument(
        "--truthfit-json",
        type=Path,
        default=None,
        help="pdnorm_endpoint_truthfit_rewear.py output to render as an"
        " estimator-seam line under the ladder figure; defaults to the"
        " preset's path, skipped quietly when that default is absent",
    )
    parser.add_argument(
        "--diagnosis-json",
        type=Path,
        default=None,
        help="frozen token_decode_diagnosis.py output to render as a"
        " decode-diagnosis section; defaults to the preset's path, skipped"
        " quietly when that default is absent",
    )
    args = parser.parse_args()
    preset = PRESETS[args.preset]

    leg_json = args.leg_json or preset.get("leg_json", DEFAULT_LEG_JSON)
    video_dir = args.video_dir or preset.get("video_dir", DEFAULT_VIDEO_DIR)
    out_html = args.out_html or preset.get("out_html", DEFAULT_OUT_HTML)
    gallery_dir = args.gallery_dir or preset.get("gallery_dir", DEFAULT_GALLERY_DIR)

    data = json.loads(leg_json.read_text())
    eps = data["episodes"]
    seed_start = preset.get("seed_start", 0)
    assert sorted(e["seed"] for e in eps) == list(
        range(seed_start, seed_start + 100),
    ), f"seeds must be {seed_start}-{seed_start + 99}"
    succ = [e for e in eps if e.get("success_tick") is not None]
    strikes = sum(e.get("reset_strikes", 0) for e in eps)
    moved = [e for e in eps if e["initial_cm"] - min(e["min_cm"], e["final_cm"]) > 0.5]
    prog = [e["initial_cm"] - min(e["min_cm"], e["final_cm"]) for e in eps]
    mean_prog = sum(prog) / len(prog)

    # Memorization split (flowtrain): the collect json is the page's
    # headline input, so it is asserted present — never skipped quietly.
    kept_seeds: set[int] = set()
    if "split_json" in preset:
        split_json = preset["split_json"]
        assert split_json.exists(), (
            f"{split_json} is required for the memorization split"
            " (regenerate from the stage-B collect run's banked curve)"
        )
        kept_seeds = set(json.loads(split_json.read_text())["kept_seeds"])

    picks: list[tuple[str, dict]] = []
    if kept_seeds:
        # Split-arm gallery: fastest success + nearest miss per arm,
        # plus the kept arm's median success — deterministic from the
        # banked json, both arms represented.
        for arm_label, arm in (
            ("kept", [e for e in eps if e["seed"] in kept_seeds]),
            ("rejected", [e for e in eps if e["seed"] not in kept_seeds]),
        ):
            arm_succ = sorted(
                (e for e in arm if e.get("success_tick") is not None),
                key=lambda e: e["success_tick"],
            )
            arm_miss = sorted(
                (e for e in arm if e.get("success_tick") is None),
                key=lambda e: e["final_cm"],
            )
            if arm_succ:
                picks.append((f"fastest success ({arm_label} arm)", arm_succ[0]))
            if arm_label == "kept" and len(arm_succ) >= 3:
                picks.append(
                    ("median success (kept arm)", arm_succ[len(arm_succ) // 2]),
                )
            if arm_miss:
                picks.append((f"nearest miss ({arm_label} arm)", arm_miss[0]))
    elif "gallery_picks" in preset:
        # Pinned diagnostic picks (seed or the far-spawn-no-touch
        # sentinel) — the token page shows the decode-diagnosis clips,
        # not the fastest successes.
        by_seed = {e["seed"]: e for e in eps}
        for label, sel in preset["gallery_picks"]:
            if sel == FAR_SPAWN_NO_TOUCH:
                no_touch = [e for e in eps if not touched(e)]
                pick = max(no_touch, key=lambda e: e["initial_cm"])
            else:
                pick = by_seed[sel]
            picks.append((label, pick))
    else:
        # Deterministic gallery: best 3 successes by success_tick
        # (fastest), plus the median success and the nearest miss by
        # final_cm.
        by_tick = sorted(succ, key=lambda e: e["success_tick"])
        if by_tick:
            picks.append(("fastest success", by_tick[0]))
        if len(by_tick) >= 2:
            picks.append(("2nd fastest success", by_tick[1]))
        if len(by_tick) >= 3:
            picks.append(("median success", by_tick[len(by_tick) // 2]))
        misses = sorted(
            (e for e in eps if e.get("success_tick") is None),
            key=lambda e: e["final_cm"],
        )
        # Low-success legs: show the near-miss tail instead of a success
        # gallery (nearest misses are the diagnostic clips there).
        for i, miss in enumerate(misses[: max(1, 4 - len(picks))]):
            picks.append(("nearest miss" if i == 0 else f"miss #{i + 1}", miss))

    gallery_dir.mkdir(parents=True, exist_ok=True)
    gallery_html = ""
    copied = 0
    for label, e in picks:
        name = f"rollout_seed{e['seed']:03d}.mp4"
        src = video_dir / name
        if not src.exists():
            continue
        copied += 1
        shutil.copy2(src, gallery_dir / name)
        gallery_html += (
            f'<figure><video src="{gallery_dir.name}/{name}" controls '
            f'muted loop width="320"></video>'
            f"<figcaption>{html.escape(label)} — seed {e['seed']}, "
            f"final {e['final_cm']:.1f} cm"
            + (
                f", success at tick {e['success_tick']}"
                if e.get("success_tick") is not None
                else ""
            )
            + "</figcaption></figure>\n"
        )
    if not gallery_html:
        gallery_html = (
            '<p class="meta">No clips: the rollout videos were lost with the'
            " box <code>outputs/</code> wipe of 2026-08-17 before they were"
            " synced off; per-seed numbers on this page are reconstructed"
            " from the surviving shard logs (0.1 cm print precision).</p>\n"
        )

    tiles = [
        (f"{len(succ)}/100", "successes (≤3 cm, held)"),
        (
            " / ".join(str(v) for _, v, _ in preset["anchor_rows"]),
            preset["anchors_tile_label"],
        ),
        (f"{len(moved)}/100", "moved the boat >0.5 cm"),
        (f"{mean_prog:.2f} cm", "mean progress toward disk"),
        (str(strikes), "reset strikes"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="big">{v}</div><div class="lab">{k}</div></div>'
        for v, k in tiles
    )

    paired_html = ""
    if args.paired_json is not None:
        paired_html = paired_section(
            json.loads(args.paired_json.read_text()),
            preset.get("paired_band_note", ""),
        )

    # Explicit --ladder-b64 is loud on a missing file; the preset
    # default (reports/ is gitignored, the sidecar is regenerable) is
    # skipped quietly so PRE-GO builds work from any checkout.
    ladder_html = ""
    ladder_b64 = args.ladder_b64 or preset.get("ladder_b64")
    if ladder_b64 is not None and (args.ladder_b64 is not None or ladder_b64.exists()):
        ladder_html = ladder_section(ladder_b64)

    # Same split for the estimator-seam line: the preset default (the
    # cross-check json exists only once the ON-GO endpoint npz does) is
    # quiet when absent, an explicit flag is loud.
    seam_html = ""
    truthfit_json = args.truthfit_json or preset.get("truthfit_json")
    if truthfit_json is not None and (
        args.truthfit_json is not None or truthfit_json.exists()
    ):
        seam_html = estimator_seam_line(json.loads(truthfit_json.read_text()))

    # Decode-diagnosis section (token preset): same quiet/loud split —
    # the preset default (reports/ is gitignored, the json regenerable
    # by token_decode_diagnosis.py) skips quietly, an explicit flag is
    # loud on a missing file.
    diagnosis_html = ""
    diagnosis_json = args.diagnosis_json or preset.get("diagnosis_json")
    if diagnosis_json is not None and (
        args.diagnosis_json is not None or diagnosis_json.exists()
    ):
        diagnosis_html = diagnosis_section(
            json.loads(diagnosis_json.read_text()),
            preset["diagnosis_chart"],
        )

    split_html = ""
    if kept_seeds:
        split_html = memorization_section(eps, kept_seeds)

    second_html = ""
    if "second_leg" in preset:
        second_label, second_json = preset["second_leg"]
        second_eps = json.loads(second_json.read_text())["episodes"]
        assert sorted(e["seed"] for e in second_eps) == list(range(100)), (
            f"{second_json}: seeds must be 0-99"
        )
        second_html = second_leg_section(second_label, second_eps)

    meta_html = preset["meta_html"].format(success=SUCCESS)
    bar_rows = [
        *preset["anchor_rows"],
        (preset["subject_label"], len(succ), SUCCESS),
    ]
    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{preset["title"]}</title>
<style>
 body {{ background:{PAGE}; color:{TEXT}; font-family:system-ui,sans-serif;
        max-width:1000px; margin:2rem auto; padding:0 1rem; }}
 h1,h2 {{ color:{HEADING}; }}
 .meta {{ color:{META}; font-size:0.9rem; }}
 .tiles {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1.2rem 0; }}
 .tile {{ background:{CARD}; border:1px solid {GRID}; border-radius:8px;
         padding:0.8rem 1.2rem; text-align:center; }}
 .tile .big {{ font-size:1.6rem; font-weight:700; color:{HEADING}; }}
 .tile .lab {{ color:{META}; font-size:0.8rem; }}
 img {{ max-width:100%; border:1px solid {GRID}; border-radius:6px;
       margin:0.6rem 0; }}
 table {{ border-collapse:collapse; font-size:0.85rem; margin:1rem 0; }}
 th,td {{ border:1px solid {GRID}; padding:0.25rem 0.6rem; text-align:right; }}
 th {{ background:{CARD}; color:{HEADING}; }}
 tr.ok td {{ color:{SUCCESS}; }}
 figure {{ display:inline-block; margin:0.5rem; }}
 figcaption {{ color:{META}; font-size:0.8rem; max-width:320px; }}
 a {{ color:{FAIL}; }}
</style></head><body>
<h1>{preset["h1"]}</h1>
<p class="meta">{meta_html}</p>
{ladder_html}{seam_html}<div class="tiles">{tiles_html}</div>
<h2>Against the anchors</h2>
<img src="data:image/png;base64,{success_bar(bar_rows, preset.get("bar_xlabel", "successes on unseen seeds 0–99"))}">
{split_html}{paired_html}{diagnosis_html}<h2>Per-seed outcomes</h2>
<img src="data:image/png;base64,{outcome_strip(eps, preset.get("strip_xlabel", "unseen seed (0–99)"))}">
<h2>Clips</h2>
{gallery_html}
<h2>Per-seed table</h2>
{leg_table(eps)}{second_html}<p class="meta">Regenerate:
 <code>fontaine/scripts/grasp_sft_joint_unseen_report.py</code>
 from the banked <code>{leg_json.name}</code> only.</p>
</body></html>"""
    out_html.write_text(page)
    print(f"wrote {out_html} ({out_html.stat().st_size / 1024:.0f} KiB)")
    print(f"gallery: {gallery_dir} ({copied} of {len(picks)} picks copied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
