"""Standalone HTML panel for the 3-leg sft-v1 sim100 eval chain
(queue item sft-v1-eval-chain-html-panel; chain ALL DONE 14:17:56Z
2026-08-17, ~6.2/12 GPU-h on the local H100, verdict post
1538917693032243293).

The chain dated the v1 collapse and separated the heads:
  leg 1  step500  flow  (euler-10)        4/100  — collapsed from the start
  leg 2  step500  token (_arhead greedy) 16/100  — above the R2 bar at 500
  leg 3  step3000 token, serving fix     14/100  — ~flat across training

Consumes ONLY the banked leg JSONs (regenerable, no live hosts) + the
local rollout mp4s; charts b64-embedded, clip gallery copied beside the
html so relative links stay live on the reports Space (sim100_report
pattern). House eval-report dark scheme.

Usage:
  uv run python fontaine/scripts/sft_v1_chain_report.py \
      [--out-html reports/eval__grasp_sft_v1__sim100_chain.html] \
      [--gallery-dir reports/v1_chain_gallery] \
      [--out-json reports/analysis__sft_v1_chain.json]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import shutil
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PAGE = "#121417"
CARD = "#1a1e24"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
HEADING = "#eceef1"
SUCCESS = "#dc267f"  # house magenta — the hero series
FAIL = "#648fff"
GOLD = "#ffb000"
ANCHOR = "#9aa0a8"

# Frozen anchors (banked): route-C probe flow 44/100, v1 endpoint flow
# 5/100 (log-reconstructed), R2 token competence bar >=20/100, base 9.
ANCHOR_FLOW_PROBE = 44
ANCHOR_FLOW_ENDPOINT = 5
ANCHOR_TOKEN_R2 = 20
ANCHOR_BASE = 9

SIM = Path("outputs/sim/grasp_sft")
LEGS: list[dict[str, Any]] = [
    {
        "key": "flow500",
        "label": "leg 1 — step 500, flow head (euler-10)",
        "short": "step500 flow",
        "json": SIM / "step500_sim100/flow_s0.json",
        "videos": SIM / "step500_sim100/flow",
        "color": SUCCESS,
    },
    {
        "key": "token500",
        "label": "leg 2 — step 500, token head (_arhead greedy)",
        "short": "step500 token",
        "json": SIM / "step500_sim100/token_s0.json",
        "videos": SIM / "step500_sim100/token",
        "color": FAIL,
    },
    {
        "key": "token_endpoint_fixed",
        "label": "leg 3 — step 3000, token head + serving fix (b779ba4)",
        "short": "endpoint token-fixed",
        "json": SIM / "endpoint_token_fixed_sim100/token_s0.json",
        "videos": SIM / "endpoint_token_fixed_sim100",
        "color": GOLD,
    },
]


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


def progress(e: dict[str, Any]) -> float:
    """Best-point progress: spawn distance minus the closest the boat
    ever got (the chain-verdict definition)."""
    return e["initial_cm"] - min(e["min_cm"], e["final_cm"])


def summarize(leg: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(leg["json"].read_text())
    eps = sorted(data["episodes"], key=lambda e: e["seed"])
    assert [e["seed"] for e in eps] == list(range(100)), "seeds must be 0-99"
    succ = sorted(e["seed"] for e in eps if e.get("success_tick") is not None)
    prog = [progress(e) for e in eps]
    return {
        **leg,
        "episodes": eps,
        "config": data["config"],
        "successes": len(succ),
        "success_seeds": succ,
        "moved_gt_half_cm": sum(1 for p in prog if p > 0.5),
        "median_progress_cm": median(prog),
        "reset_strikes": sum(e.get("reset_strikes", 0) for e in eps),
    }


def anchors_bar(legs: list[dict[str, Any]]) -> str:
    rows = [
        ("base, flow (no SFT)", ANCHOR_BASE, ANCHOR),
        ("probe step2000 flow (313 demos)", ANCHOR_FLOW_PROBE, "#f593bd"),
        ("v1 endpoint flow (reconstructed)", ANCHOR_FLOW_ENDPOINT, "#f593bd"),
        *((leg["short"], leg["successes"], leg["color"]) for leg in legs),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.0), facecolor=PAGE)
    for i, (label, val, color) in enumerate(rows):
        ax.barh(i, val, color=color, height=0.55)
        ax.text(val + 1, i, f"{val}/100", color=TEXT, fontsize=10, va="center")
        ax.text(-1.5, i, label, color=TEXT, fontsize=9, va="center", ha="right")
    ax.axvline(ANCHOR_TOKEN_R2, color=META, linewidth=0.8, linestyle="--")
    ax.text(
        ANCHOR_TOKEN_R2,
        len(rows) - 0.25,
        "R2 token bar 20",
        color=META,
        fontsize=8,
        ha="center",
    )
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("successes on unseen seeds 0–99")
    ax.set_title("Chain legs against the frozen anchors")
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.36)
    return fig_to_b64(fig)


def head_asymmetry() -> str:
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor=PAGE)
    for label, vals, color in (
        ("token (greedy, fixed serving)", (16, 14), FAIL),
        ("flow (euler-10)", (4, 5), SUCCESS),
    ):
        ax.plot([500, 3000], vals, "o-", color=color, linewidth=2, markersize=5)
        ax.annotate(
            f"{label}  {vals[0]}→{vals[1]}",
            (3000, vals[1]),
            textcoords="offset points",
            xytext=(8, 0),
            color=color,
            fontsize=9,
        )
    ax.axhline(ANCHOR_TOKEN_R2, color=META, linewidth=0.8, linestyle="--")
    ax.text(560, ANCHOR_TOKEN_R2 + 0.6, "R2 token bar", color=META, fontsize=8)
    ax.axhline(ANCHOR_FLOW_PROBE, color="#f593bd", linewidth=0.8, linestyle="--")
    ax.text(560, ANCHOR_FLOW_PROBE + 0.6, "probe flow 44", color="#f593bd", fontsize=8)
    ax.set_xticks([500, 3000])
    ax.set_xlim(300, 4600)
    ax.set_ylim(0, 50)
    ax.set_xlabel("training step")
    ax.set_ylabel("successes /100")
    ax.set_title("Head asymmetry across training")
    style_ax(ax)
    return fig_to_b64(fig)


def outcome_strip(leg: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(10, 2.9), facecolor=PAGE)
    for e in leg["episodes"]:
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
    ax.set_ylabel("boat → disk (cm)")
    ax.set_title(
        f"{leg['label']} — spawn → final distance (magenta = success, blue = miss)",
    )
    style_ax(ax)
    ax.set_xlim(-1.5, 113)
    return fig_to_b64(fig)


def gallery(legs: list[dict[str, Any]], gallery_dir: Path) -> str:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for leg in legs:
        succ = sorted(
            (e for e in leg["episodes"] if e.get("success_tick") is not None),
            key=lambda e: e["success_tick"],
        )
        misses = sorted(
            (e for e in leg["episodes"] if e.get("success_tick") is None),
            key=lambda e: e["final_cm"],
        )
        picks = [
            ("fastest success", succ[0]),
            ("median success", succ[len(succ) // 2]),
            ("nearest miss", misses[0]),
        ]
        cards = ""
        for label, e in picks:
            name = f"rollout_seed{e['seed']:03d}.mp4"
            src = leg["videos"] / name
            if not src.exists():
                continue
            dst = f"{leg['key']}_seed{e['seed']:03d}.mp4"
            shutil.copy2(src, gallery_dir / dst)
            cards += (
                f'<figure><video src="{gallery_dir.name}/{dst}" controls '
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
        blocks.append(f"<h3>{html.escape(leg['label'])}</h3>\n{cards}")
    return "".join(blocks)


def per_seed_table(legs: list[dict[str, Any]]) -> str:
    rows = ""
    for seed in range(100):
        eps = [leg["episodes"][seed] for leg in legs]
        cells = f"<td>{seed}</td><td>{eps[0]['initial_cm']:.1f}</td>"
        for e in eps:
            ok = e.get("success_tick") is not None
            mark = f"✓ @ {e['success_tick']}" if ok else "—"
            cells += (
                f"<td>{e['final_cm']:.1f}</td><td>{progress(e):.1f}</td>"
                f"<td{' class=ok' if ok else ''}>{mark}</td>"
            )
        rows += f"<tr>{cells}</tr>\n"
    leg_heads = "".join(
        f'<th colspan="3">{html.escape(leg["short"])}</th>' for leg in legs
    )
    sub = "<th>final</th><th>prog</th><th>succ</th>" * len(legs)
    return (
        f'<table><tr><th rowspan="2">seed</th><th rowspan="2">spawn cm</th>'
        f"{leg_heads}</tr><tr>{sub}</tr>\n{rows}</table>"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("reports/eval__grasp_sft_v1__sim100_chain.html"),
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=Path("reports/v1_chain_gallery"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/analysis__sft_v1_chain.json"),
    )
    args = parser.parse_args()

    legs = [summarize(leg) for leg in LEGS]
    spawn0 = [e["initial_cm"] for e in legs[0]["episodes"]]
    for leg in legs[1:]:
        assert all(
            abs(a - e["initial_cm"]) < 1e-6
            for a, e in zip(spawn0, leg["episodes"], strict=True)
        ), "spawns must pair across legs (same seeds, same sim)"

    tiles = [
        *((f"{leg['successes']}/100", leg["short"]) for leg in legs),
        (f"{ANCHOR_FLOW_ENDPOINT}/100", "endpoint flow (anchor)"),
        (f"{legs[2]['median_progress_cm']:.2f} cm", "leg-3 median progress"),
        (f"{legs[2]['moved_gt_half_cm']}/100", "leg-3 moved >0.5 cm"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="big">{v}</div><div class="lab">{k}</div></div>'
        for v, k in tiles
    )
    strips = "".join(
        f'<img src="data:image/png;base64,{outcome_strip(leg)}">' for leg in legs
    )

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>grasp_sft_v1 — sim100 eval chain, 3 legs</title>
<style>
 body {{ background:{PAGE}; color:{TEXT}; font-family:system-ui,sans-serif;
        max-width:1100px; margin:2rem auto; padding:0 1rem; }}
 h1,h2,h3 {{ color:{HEADING}; }}
 .meta {{ color:{META}; font-size:0.9rem; }}
 .tiles {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1.2rem 0; }}
 .tile {{ background:{CARD}; border:1px solid {GRID}; border-radius:8px;
         padding:0.8rem 1.2rem; text-align:center; }}
 .tile .big {{ font-size:1.6rem; font-weight:700; color:{HEADING}; }}
 .tile .lab {{ color:{META}; font-size:0.8rem; }}
 img {{ max-width:100%; border:1px solid {GRID}; border-radius:6px;
       margin:0.6rem 0; }}
 table {{ border-collapse:collapse; font-size:0.8rem; margin:1rem 0; }}
 th,td {{ border:1px solid {GRID}; padding:0.2rem 0.5rem; text-align:right; }}
 th {{ background:{CARD}; color:{HEADING}; }}
 td.ok {{ color:{SUCCESS}; }}
 figure {{ display:inline-block; margin:0.5rem; }}
 figcaption {{ color:{META}; font-size:0.8rem; max-width:320px; }}
 a {{ color:{FAIL}; }}
</style></head><body>
<h1>Grasp-SFT v1 — the 3-leg sim100 eval chain</h1>
<p class="meta">One protocol, three legs, seeds 0–99 each (paired
spawns, triple-keyed noise): flow + token heads of
<code>grasp_sft_v1_joint_8xa100</code> at step 500
(<code>run2_step500</code>, box download) and the step-3000 endpoint
token head under the serving fix <code>b779ba4</code> (the 0/100
endpoint token read was an inference-collator bug — per-item dataset
quantiles instead of the merged training table). euler-10 /
<code>--serve-head ar</code> greedy, execute-horizon 30, 30 s episodes
· run 2026-08-17 06:0x–14:17:56Z on the local H100, ~6.2/12 GPU-h ·
verdict: <b style="color:{SUCCESS}">the collapse is flow-specific and
present from step 500</b> — the token head sits near its 500-step
level at both ends (16→14) while flow never leaves the floor (4→5);
the v1 normalization table mis-fit poisons the flow targets, not the
shared trunk.</p>
<div class="tiles">{tiles_html}</div>
<h2>Against the anchors</h2>
<img src="data:image/png;base64,{anchors_bar(legs)}">
<img src="data:image/png;base64,{head_asymmetry()}">
<h2>Per-seed outcomes</h2>
{strips}
<h2>Clips</h2>
{gallery(legs, args.gallery_dir)}
<h2>Per-seed table <span class="meta">(prog = spawn − best-point
distance, cm)</span></h2>
{per_seed_table(legs)}
<p class="meta">Regenerate:
<code>fontaine/scripts/sft_v1_chain_report.py</code> from the banked
leg JSONs only. Frozen summary:
<code>analysis__sft_v1_chain.json</code>.</p>
</body></html>"""
    args.out_html.write_text(page)

    args.out_json.write_text(
        json.dumps(
            {
                "legs": [
                    {
                        k: leg[k]
                        for k in (
                            "key",
                            "successes",
                            "success_seeds",
                            "moved_gt_half_cm",
                            "median_progress_cm",
                            "reset_strikes",
                        )
                    }
                    | {
                        "checkpoint": leg["config"]["checkpoint"],
                        "serve_head": leg["config"].get("serve_head"),
                        "commit": leg["config"].get("commit"),
                    }
                    for leg in legs
                ],
                "anchors": {
                    "base_flow": ANCHOR_BASE,
                    "probe_step2000_flow": ANCHOR_FLOW_PROBE,
                    "v1_endpoint_flow_reconstructed": ANCHOR_FLOW_ENDPOINT,
                    "token_r2_bar": ANCHOR_TOKEN_R2,
                },
            },
            indent=1,
        ),
    )
    clips = sorted(p.name for p in args.gallery_dir.glob("*.mp4"))
    print(f"wrote {args.out_html} ({args.out_html.stat().st_size / 1024:.0f} KiB)")
    print(f"wrote {args.out_json}")
    print(f"gallery ({len(clips)} clips) -> {args.gallery_dir}: {clips}")
    for leg in legs:
        print(
            f"  {leg['key']}: {leg['successes']}/100, moved "
            f"{leg['moved_gt_half_cm']}, median prog "
            f"{leg['median_progress_cm']:.2f}, strikes {leg['reset_strikes']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
