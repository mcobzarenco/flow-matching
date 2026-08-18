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
      [--paired-json reports/analysis__sim100_paired_probe_vs_disc1000.json]

--paired-json takes a frozen sim100_paired_read.py output and renders
it as a "Paired read" section (delta tiles + discordant-seed chart)
alongside the frozen absolute anchors; the read stays recorded,
non-gating (registered consumer: the pdnorm endpoint vs the disc-1000
demosonly baseline, whose 11/100 sits inside the pre-reg's own 11-19
ambiguous band).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
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
            "Checkpoint <code>FILL-AT-ENDPOINT/step_003000</code>"
            " (per-dataset flow norm, mixed v2 corpus; launched"
            " FILL-AT-ENDPOINT, run FILL-AT-ENDPOINT, ~FILL-AT-ENDPOINT"
            " GPU-h) · euler-10, execute-horizon 30, seeds 0–99, 30 s"
            " episodes; sim leg wears the sim demos' row (frozen"
            " serving-row rule) · frozen decision grid: &le;10"
            " broken-class band / 11&ndash;19 ambiguous band /"
            ' <b style="color:{success}">&ge;20 exonerates the mix</b> ·'
            " panel anchors, wear-corrected class: 27.40 re-worn"
            " disc-1000 / 25.15 repo-midpoint null / 8.37 state-copy"
            " (real bar) · verdict: <b>FILL-AT-ENDPOINT</b>"
        ),
        "paired_band_note": (
            " The demosonly baseline's 11/100 sits inside the pdnorm"
            " draft's own 11&ndash;19 ambiguous band, so the paired read"
            " rides alongside the frozen absolute bands &mdash; it is"
            " recorded, never gating."
        ),
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


def outcome_strip(eps: list[dict]) -> str:
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
    ax.text(100.5, 3.0, "success radius 3 cm", color=META, fontsize=8, va="center")
    ax.set_xlabel("unseen seed (0–99)")
    ax.set_ylabel("boat → disk center (cm)")
    ax.set_title(
        "Per-seed outcome: spawn distance → final distance "
        "(magenta = success, blue = miss)",
    )
    style_ax(ax)
    ax.set_xlim(-1.5, 113)
    return fig_to_b64(fig)


def success_bar(rows: list[tuple[str, int, str]]) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 2.6), facecolor=PAGE)
    for i, (label, val, color) in enumerate(rows):
        ax.barh(i, val, color=color, height=0.55)
        ax.text(val + 1, i, f"{val}/100", color=TEXT, fontsize=10, va="center")
        ax.text(-1.5, i, label, color=TEXT, fontsize=9, va="center", ha="right")
    ax.set_yticks([])
    ax.set_xlim(0, max(60, int(max(v for _, v, _ in rows) * 1.2)))
    ax.set_xlabel("successes on unseen seeds 0–99")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leg-json",
        type=Path,
        default=Path("outputs/sim/grasp_sft/joint_probes/flow_unseen.json"),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("outputs/sim/grasp_sft/joint_probes/flow_unseen"),
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("reports/eval__grasp_sft_joint_step2000__flow_unseen100.html"),
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=Path("reports/joint_unseen_gallery"),
    )
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
    args = parser.parse_args()
    preset = PRESETS[args.preset]

    data = json.loads(args.leg_json.read_text())
    eps = data["episodes"]
    assert sorted(e["seed"] for e in eps) == list(range(100)), "seeds must be 0-99"
    succ = [e for e in eps if e.get("success_tick") is not None]
    strikes = sum(e.get("reset_strikes", 0) for e in eps)
    moved = [e for e in eps if e["initial_cm"] - min(e["min_cm"], e["final_cm"]) > 0.5]
    prog = [e["initial_cm"] - min(e["min_cm"], e["final_cm"]) for e in eps]
    mean_prog = sum(prog) / len(prog)

    # Deterministic gallery: best 3 successes by success_tick (fastest),
    # plus the median success and the nearest miss by final_cm.
    by_tick = sorted(succ, key=lambda e: e["success_tick"])
    picks: list[tuple[str, dict]] = []
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

    args.gallery_dir.mkdir(parents=True, exist_ok=True)
    gallery_html = ""
    copied = 0
    for label, e in picks:
        name = f"rollout_seed{e['seed']:03d}.mp4"
        src = args.video_dir / name
        if not src.exists():
            continue
        copied += 1
        shutil.copy2(src, args.gallery_dir / name)
        gallery_html += (
            f'<figure><video src="{args.gallery_dir.name}/{name}" controls '
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
<div class="tiles">{tiles_html}</div>
<h2>Against the anchors</h2>
<img src="data:image/png;base64,{success_bar(bar_rows)}">
{paired_html}<h2>Per-seed outcomes</h2>
<img src="data:image/png;base64,{outcome_strip(eps)}">
<h2>Clips</h2>
{gallery_html}
<h2>Per-seed table</h2>
<table><tr><th>seed</th><th>spawn cm</th><th>min cm</th><th>final cm</th>
<th>progress cm</th><th>success</th></tr>
{rows_html}</table>
<p class="meta">Regenerate: <code>fontaine/scripts/grasp_sft_joint_unseen_report.py</code>
 from the banked <code>flow_unseen.json</code> only.</p>
</body></html>"""
    args.out_html.write_text(page)
    print(f"wrote {args.out_html} ({args.out_html.stat().st_size / 1024:.0f} KiB)")
    print(f"gallery: {args.gallery_dir} ({copied} of {len(picks)} picks copied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
