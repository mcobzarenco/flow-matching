"""Standalone HTML report + video gallery for the 100-seed sim eval
(posts/2026-08-11-prereg-sim-policy-eval-100seeds.md).

Consumes ONLY banked artifacts (regenerable, no live hosts): the frozen
analysis JSON from sim100_reads.py, the per-arm episode JSONs, the
house charts from sim100_charts.py, and the per-seed rollout videos.
Produces:

  reports/report__sim100_seed_eval.html   — dark standalone report
  reports/sim100_gallery/                 — selected clips + chart pngs

Gallery selection is deterministic: per arm best / median / worst seed
by progress_final, plus two er60k "reach-but-miss" clips (boat
untouched, |progress| < 0.5 cm — the phase-1 fingerprint: confident
reaching over the table, never at the boat). The HTML references
gallery files relatively, so uploading the html at the Space root and
the gallery under sim100_gallery/ keeps every link live.

Usage:
  uv run python fontaine/scripts/sim100_report.py \
      --analysis reports/analysis__sim100_seed_eval.json \
      --in-dir outputs/sim/eval100 \
      --charts-dir fontaine/blog/src/img/sim100 \
      --out-html reports/report__sim100_seed_eval.html \
      --gallery-dir reports/sim100_gallery
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

from fontaine.scripts.sim100_charts import ARM_ORDER

DARK = {
    "bg": "#121417",
    "text": "#d8dade",
    "heading": "#eceef1",
    "border": "#3a3f46",
    "th": "#1f242b",
    "pre": "#1a1e24",
    "meta": "#9aa0a8",
    "accent": "#648fff",
    "good": "#42be65",
    "bad": "#dc267f",
    "warn": "#ffb000",
}
ARM_LABELS = {
    "er60k": "er_60k trunk (euler-1 expert, heun-10 flow)",
    "hold": "hold control (zero action)",
    "er15k": "er_15k rung",
    "er35k": "er_35k rung",
    "er55k": "er_55k rung",
    "ftrig4k": "rig-ft snapflow student (euler-1)",
    "snap30k": "snapflow student (euler-1)",
    "teacher80k": "artrunk 80k teacher (heun-30)",
}
CHARTS = ("distance_over_time.png", "progress_strip.png", "ordering_vs_panel.png")
UNTOUCHED_CM = 0.5
REACH_MISS_CLIPS = 2


def pick_gallery(episodes: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    """Best / median / worst by progress_final; er60k adds two
    reach-but-miss clips (boat untouched)."""
    by_progress = sorted(episodes, key=lambda e: e["progress_final_cm"])
    picks = [
        {"role": "best", "ep": by_progress[-1]},
        {"role": "median", "ep": by_progress[len(by_progress) // 2]},
        {"role": "worst", "ep": by_progress[0]},
    ]
    if arm == "er60k":
        chosen = {p["ep"]["seed"] for p in picks}
        untouched = [
            e
            for e in episodes
            if abs(e["progress_final_cm"]) < UNTOUCHED_CM and e["seed"] not in chosen
        ]
        picks.extend(
            {"role": "reach-but-miss", "ep": ep} for ep in untouched[:REACH_MISS_CLIPS]
        )
    return picks


def video_card(arm: str, pick: dict[str, Any], gallery: Path) -> str:
    ep = pick["ep"]
    src = f"{gallery.name}/{arm}_seed{ep['seed']:03d}.mp4"
    success = (
        f"success @ tick {ep['success_tick']}"
        if ep["success_tick"] is not None
        else "no success"
    )
    return (
        '<div class="card"><video controls preload="metadata" '
        f'src="{src}"></video><div class="lines">'
        f'<span><span class="tag model">{pick["role"]}</span> seed '
        f"{ep['seed']}</span>"
        f"<span>progress {ep['progress_final_cm']:+.2f} cm &middot; "
        f"{success}</span>"
        f'<span class="meta2">initial {ep["initial_cm"]:.1f} &rarr; final '
        f"{ep['final_cm']:.1f} cm (min {ep['min_cm']:.1f})</span>"
        "</div></div>"
    )


def fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "&mdash;"
    return format(value, spec) if spec else html.escape(str(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--charts-dir", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    summary = analysis["summary"]
    arms_present = [a for a in ARM_ORDER if a in summary]
    episodes = {
        arm: json.loads((args.in_dir / f"{arm}.json").read_text())["episodes"]
        for arm in arms_present
    }

    args.gallery_dir.mkdir(parents=True, exist_ok=True)
    galleries: dict[str, list[dict[str, Any]]] = {}
    for arm in arms_present:
        if arm == "hold":
            continue  # zero-action control: nothing to watch
        picks = pick_gallery(episodes[arm], arm)
        for pick in picks:
            seed = pick["ep"]["seed"]
            src = args.in_dir / arm / f"rollout_seed{seed:03d}.mp4"
            if not src.exists():
                continue
            shutil.copy2(src, args.gallery_dir / f"{arm}_seed{seed:03d}.mp4")
        galleries[arm] = picks
    for chart in CHARTS:
        src = args.charts_dir / chart
        if src.exists():
            shutil.copy2(src, args.gallery_dir / chart)

    gates = analysis["gates"]
    gate_ok = gates["reset_strikes_gate"] and gates["hold_floor_gate"]
    gate_color = DARK["good"] if gate_ok else DARK["bad"]

    tiles = []
    for arm in arms_present:
        s = summary[arm]
        tiles.append(
            f'<div class="tile">{arm}<br><span class="big">'
            f"{s['mean_progress_final_cm']:+.2f} cm</span><br>"
            f"success {s['success_rate']:.0%} &middot; best-point "
            f"{s['mean_progress_min_cm']:+.2f}</div>",
        )

    summary_rows = []
    for arm in arms_present:
        s = summary[arm]
        moved = sum(
            1 for e in episodes[arm] if abs(e["progress_final_cm"]) >= UNTOUCHED_CM
        )
        summary_rows.append(
            f"<tr><td>{arm}</td><td>{html.escape(ARM_LABELS.get(arm, arm))}</td>"
            f"<td>{s['mean_progress_final_cm']:+.4f}</td>"
            f"<td>{s['median_progress_final_cm']:+.4f}</td>"
            f"<td>{s['mean_progress_min_cm']:+.4f}</td>"
            f"<td>{moved}/100</td>"
            f"<td>{s['success_rate']:.0%}</td>"
            f"<td>{fmt(s['median_success_tick'], '.0f')}</td>"
            f"<td>{fmt(s['median_latency_ms'], '.0f')}</td>"
            f"<td>{fmt(s['panel_mae'])}</td></tr>",
        )

    paired_rows = []
    for key, p in analysis["paired"].items():
        a, b = key.split("_minus_")
        excl = (
            f'<span style="color:{DARK["accent"]}">yes</span>'
            if p["ci_excludes_zero"]
            else "no"
        )
        paired_rows.append(
            f"<tr><td>{a} &minus; {b}</td><td>{p['mean_delta_cm']:+.4f}</td>"
            f"<td>[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]</td>"
            f"<td>{excl}</td><td>{p['win_rate']:.0%}</td></tr>",
        )

    ordering = analysis["ordering"]
    ordering_html = (
        f'<p class="meta">Ordering read: {html.escape(ordering["skipped"])} '
        "(the phase-2 owner amendment killed the rung arms).</p>"
        if "skipped" in ordering
        else (
            f"<p>Gated pairs correct: {ordering['gated_pairs_correct']}/"
            f"{ordering['gated_pairs_total']} &middot; expectation "
            f"{'MET' if ordering['expectation_met'] else 'VIOLATED'} &middot; "
            f"Spearman &rho; {ordering['spearman_rho_progress_vs_neg_panel']:.3f}</p>"
        )
    )

    charts_html = "".join(
        f'<img src="{args.gallery_dir.name}/{c}" alt="{c}">'
        for c in CHARTS
        if (args.gallery_dir / c).exists()
    )

    gallery_html = []
    for arm, picks in galleries.items():
        cards = "".join(
            video_card(arm, p, args.gallery_dir)
            for p in picks
            if (args.gallery_dir / f"{arm}_seed{p['ep']['seed']:03d}.mp4").exists()
        )
        note = (
            " &mdash; the money shot: confident reaching over the table, "
            "never at the boat (the visual-gap fingerprint)"
            if arm == "er60k"
            else ""
        )
        gallery_html.append(
            f'<h3>{arm} <span class="meta">'
            f"{html.escape(ARM_LABELS.get(arm, arm))}{note}</span></h3>"
            f'<div class="cards">{cards}</div>',
        )

    n_seeds = analysis["n_seeds"]
    configs = analysis["configs"]
    commit = next(iter(configs.values())).get("commit", "?")
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>100-seed sim policy eval</title><style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto;
  max-width: 1200px; color: {DARK["text"]}; background: {DARK["bg"]};
  padding: 0 1em; }}
h1, h2, h3 {{ color: {DARK["heading"]}; }}
table {{ border-collapse: collapse; margin: 0.8em 0; font-size: 13px;
  font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid {DARK["border"]}; padding: 3px 8px;
  text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
td:nth-child(2) {{ text-align: left; }}
th {{ background: {DARK["th"]}; }}
.meta {{ color: {DARK["meta"]}; font-size: 12.5px; font-weight: 400; }}
.meta2 {{ color: {DARK["meta"]}; font-size: 11.5px; }}
.tiles {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 1em 0; }}
.tile {{ background: {DARK["th"]}; border: 1px solid {DARK["border"]};
  border-radius: 8px; padding: 10px 16px; font-size: 13px;
  color: {DARK["meta"]}; }}
.tile .big {{ font-size: 22px; color: {DARK["heading"]};
  font-variant-numeric: tabular-nums; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill,
  minmax(300px, 1fr)); gap: 12px; margin-bottom: 1.4em; }}
.card {{ background: {DARK["pre"]}; border: 1px solid {DARK["border"]};
  border-radius: 8px; overflow: hidden; font-size: 13px; }}
.card video {{ width: 100%; display: block; background: #000; }}
.card .lines {{ padding: 8px 10px; display: flex; flex-direction: column;
  gap: 4px; }}
.tag {{ display: inline-block; border-radius: 4px; padding: 0 6px;
  font-size: 11px; color: #121417; font-weight: 600;
  background: {DARK["accent"]}; }}
img {{ max-width: 100%; display: block; margin: 1em 0;
  border: 1px solid {DARK["border"]}; border-radius: 8px; }}
.gate {{ color: {gate_color}; font-weight: 600; }}
</style></head><body>
<h1>100-seed sim policy eval &mdash; can any policy family engage the
boat?</h1>
<p class="meta">Pre-reg posts/2026-08-11-prereg-sim-policy-eval-100seeds.md
&middot; v0 MuJoCo sim, seeds 0&ndash;{n_seeds - 1}, 30 s horizon, 30-step
replan &middot; primary metric = progress (initial &minus; final
boat&rarr;disk distance, cm), success secondary &middot; phase 2 arms per
owner amendment 22:58Z 08-11 (rung arms killed) &middot; commit {commit}
&middot; record-only.</p>

<div class="tiles">{"".join(tiles)}</div>

<p>Gates: <span class="gate">reset strikes
{gates["reset_strikes_total"]} (gate {"PASS" if gates["reset_strikes_gate"] else "FAIL"})
&middot; hold floor {gates["hold_floor_cm"]:+.5f} cm
(gate {"PASS" if gates["hold_floor_gate"] else "FAIL"})</span></p>

<h2>Per-arm summary <span class="meta">(100 paired seeds; "moved" =
|progress| &ge; {UNTOUCHED_CM} cm)</span></h2>
<table><tr><th>arm</th><th>policy</th><th>mean progress</th>
<th>median</th><th>mean best-point</th><th>moved</th><th>success</th>
<th>med. success tick</th><th>med. latency ms</th><th>panel MAE</th></tr>
{"".join(summary_rows)}</table>

<h2>Paired per-seed reads <span class="meta">(bootstrap CI95, 10k
resamples, seed 0)</span></h2>
<table><tr><th>pair</th><th>&Delta; mean (cm)</th><th>CI95</th>
<th>CI excl. 0</th><th>win rate</th></tr>{"".join(paired_rows)}</table>
{ordering_html}

<h2>Charts</h2>
{charts_html}

<h2>Video gallery <span class="meta">(deterministic picks: best / median
/ worst seed by progress)</span></h2>
{"".join(gallery_html)}

<p class="meta">Generated by fontaine/scripts/sim100_report.py from the
banked analysis JSON + per-arm episode JSONs; every artifact
regenerable offline.</p>
</body></html>"""
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(doc)
    clips = sorted(p.name for p in args.gallery_dir.glob("*.mp4"))
    print(f"report -> {args.out_html}")
    print(f"gallery ({len(clips)} clips) -> {args.gallery_dir}: {clips}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
