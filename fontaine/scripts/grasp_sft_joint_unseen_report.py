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
      [--gallery-dir reports/joint_unseen_gallery]
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
    picks: list[tuple[str, dict]] = [
        ("fastest success", by_tick[0]),
        ("2nd fastest success", by_tick[1]),
        ("median success", by_tick[len(by_tick) // 2]),
    ]
    misses = sorted(
        (e for e in eps if e.get("success_tick") is None),
        key=lambda e: e["final_cm"],
    )
    picks.append(("nearest miss", misses[0]))

    args.gallery_dir.mkdir(parents=True, exist_ok=True)
    gallery_html = ""
    for label, e in picks:
        name = f"rollout_seed{e['seed']:03d}.mp4"
        src = args.video_dir / name
        if not src.exists():
            continue
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
<h2>Per-seed outcomes</h2>
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
    print(f"gallery: {args.gallery_dir} ({len(picks)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
