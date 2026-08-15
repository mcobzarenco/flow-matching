"""House dark-mode charts for the grasp-SFT chain results page
(posts/2026-08-14-prereg-grasp-sft-bootstrap.md; results page
posts/2026-08-15-grasp-sft-chain-results.md).

Reads ONLY banked artifacts (regenerable, no live hosts):
  reports/analysis__grasp_sft_stageA_gate.json / ..._a1.json
  ~/datasets/fontaine/grasp_sft_demos_v0/collect_state.json  (banked
      copy: reports/curve__grasp_sft_stageb_collect.json — written by
      --extract so the page stays regenerable off-repo)
  reports/curve__grasp_sft_stagec_ar_loss.json (written by --extract
      from the train logs; rig-ft r1 reference curve included)
  outputs/sim/grasp_sft/stageD/ar.json (tolerant: absent -> the strip
      renders anchors-only with a PENDING note)
  outputs/sim/eval100/ftrig4k.json (banked context anchor)

Figures -> fontaine/blog/src/img/grasp_sft/:
  1. stagea_gate_arc.png   — per-seed gate tiles, FAIL 11/20 -> A1 ->
     PASS 15/20, with the n=200 true-rate context
  2. stageb_keep_rate.png  — per-band keep rate over the 486 attempted
     seeds vs the measured 62.5% true rate + n=20 gate-read optimism
  3. stagec_loss.png       — action_flow_loss vs step (log y) vs the
     rig-ft r1 reference curve
  4. staged_progress_strip.png — per-seed progress strip vs the banked
     ftrig4k / W0 context anchors (record-only, not gates)

Usage:
  uv run python fontaine/scripts/grasp_sft_chain_charts.py --extract
  uv run python fontaine/scripts/grasp_sft_chain_charts.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# House eval-report scheme (er60k_screen_close_charts.py /
# sim100_charts.py): dark page, banked lineage hues, identity never
# color-alone (every series direct-labeled).
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
MAGENTA = "#dc267f"  # grasp-SFT chain (house emphasis hue)
BLUE = "#648fff"  # rig-ft r1 reference (banked snap30k-family blue)
GOLD = "#ffb000"  # ftrig4k (its banked color across eval pages)
GRAY = "#9aa0a8"

REPO = Path("/home/ubuntu/flow-matching")
REPORTS = REPO / "reports"
IMG_OUT = REPO / "fontaine/blog/src/img/grasp_sft"

STAGEC_LOG = Path("/home/ubuntu/logs/molmoact2_grasp_sft_stagec_ar.log")
RIGFT_LOG = Path("/home/ubuntu/logs/molmoact2_rig_ft.log")
COLLECT_STATE = Path(
    "/home/ubuntu/datasets/fontaine/grasp_sft_demos_v0/collect_state.json",
)
CURVE_C = REPORTS / "curve__grasp_sft_stagec_ar_loss.json"
CURVE_B = REPORTS / "curve__grasp_sft_stageb_collect.json"
STAGED_AR = REPO / "outputs/sim/grasp_sft/stageD/ar.json"
FTRIG4K_100 = REPO / "outputs/sim/eval100/ftrig4k.json"
PROBE_ANALYSIS = REPORTS / "analysis__grasp_sft_step2000_probe.json"

STEP_RE = re.compile(r"\[step=(\d+)/\d+")
LOSS_RE = re.compile(r"train/action_flow_loss=([0-9.eE+-]+|nan|inf)")


def parse_loss_curve(log_path: Path) -> list[list[float]]:
    """(step, action_flow_loss) pairs; loss line follows its step line."""
    pairs: list[list[float]] = []
    step = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            continue
        m = LOSS_RE.search(line)
        if m and step is not None:
            pairs.append([step, float(m.group(1))])
            step = None
    return pairs


def extract() -> None:
    """Bank the log-derived curves so rendering never needs live hosts."""
    out = {
        "read": "grasp_sft_stagec_ar_loss_curve",
        "source_log": str(STAGEC_LOG),
        "stagec_ar": parse_loss_curve(STAGEC_LOG),
        "rig_ft_r1_reference": parse_loss_curve(RIGFT_LOG),
    }
    CURVE_C.write_text(json.dumps(out))
    n = len(out["stagec_ar"])
    print(
        f"[extract] {CURVE_C}: stage-C {n} pts "
        f"(last step {out['stagec_ar'][-1][0] if n else '-'}), "
        f"rig-ft r1 {len(out['rig_ft_r1_reference'])} pts",
    )
    state = json.loads(COLLECT_STATE.read_text())
    CURVE_B.write_text(
        json.dumps(
            {
                "read": "grasp_sft_stageb_collect_state",
                "kept_seeds": state["kept_seeds"],
                "attempted": state["attempted"],
                "next_seed": state["next_seed"],
            },
        ),
    )
    print(
        f"[extract] {CURVE_B}: kept {len(state['kept_seeds'])} "
        f"of {state['attempted']} attempted",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def new_fig(w: float, h: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=PAGE)
    style_axis(ax)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    path = IMG_OUT / name
    fig.savefig(path, dpi=150, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {path}")


def fig_stagea() -> None:
    reads = []
    for path, label in [
        (
            REPORTS / "analysis__grasp_sft_stageA_gate.json",
            "gate read (seeds 1020–1039, burned by the A1 pass)",
        ),
        (
            REPORTS / "analysis__grasp_sft_stageA_gate_a1.json",
            "A1 fresh gate (held seeds 1040–1059)",
        ),
    ]:
        d = json.loads(path.read_text())
        reads.append((label, d))

    fig, ax = new_fig(9.0, 2.9)
    ax.yaxis.grid(False)
    for row, (label, d) in enumerate(reversed(reads)):
        succ = sum(r["success"] for r in d["rows"])
        y0 = row * 1.55
        for i, r in enumerate(d["rows"]):
            color = MAGENTA if r["success"] else PAGE
            ax.add_patch(
                plt.Rectangle(
                    (i, y0),
                    0.9,
                    0.62,
                    facecolor=color,
                    edgecolor=MAGENTA if r["success"] else GRID,
                    linewidth=1.0,
                    zorder=3,
                ),
            )
        verdict = "PASS" if d["gate_pass"] else "FAIL"
        ax.text(
            -0.4,
            y0 + 0.31,
            f"{label}\n{succ}/20 {verdict} (gate ≥14)",
            ha="right",
            va="center",
            color=TEXT,
            fontsize=9,
        )
    ax.text(
        10,
        1.08,
        "amendment A1 (registered 02:33Z): robustness pass on the burned "
        "set — lower place-droop · re-grasp recovery · jam-flip budget 3",
        ha="center",
        va="center",
        color=META,
        fontsize=8,
        zorder=5,
    )
    ax.set_xlim(-9.5, 20.5)
    ax.set_ylim(-0.6, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Stage A — scripted-expert gate: FAIL 11/20 → A1 → "
        "PASS 15/20  (n=200 true rate later measured 62.5%)",
        fontsize=11,
        loc="left",
        color=TEXT,
    )
    ax.text(
        0,
        -0.42,
        "each tile = one held seed · filled = "
        "success (grasp–lift–place inside the disk)",
        color=META,
        fontsize=8,
    )
    save(fig, "stagea_gate_arc.png")


def fig_stageb() -> None:
    d = json.loads(CURVE_B.read_text())
    kept = set(d["kept_seeds"])
    attempted = list(range(1000, 1000 + d["attempted"]))
    band = 50
    edges = list(range(1000, attempted[-1] + 1, band))
    rates, labels, centers = [], [], []
    for e in edges:
        seeds = [s for s in attempted if e <= s < e + band]
        if not seeds:
            continue
        rates.append(100.0 * sum(s in kept for s in seeds) / len(seeds))
        labels.append(f"{e}–{min(e + band - 1, attempted[-1])}")
        centers.append(len(rates) - 1)

    fig, ax = new_fig(9.0, 3.4)
    ax.bar(centers, rates, width=0.82, color=MAGENTA, zorder=3)
    overall = 100.0 * len(kept) / len(attempted)
    ax.axhline(overall, color=TEXT, linewidth=1.2, zorder=4)
    ax.text(
        4.5,
        68.5,
        f"stage-B overall {overall:.0f}%  (313 kept / 486 attempted)",
        ha="center",
        color=TEXT,
        fontsize=9,
        zorder=6,
    )
    ax.axhline(62.5, color=GRAY, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(
        -0.4,
        53.5,
        "measured true expert rate 62.5% (n=200, §8) ↑",
        color=TEXT,
        fontsize=8.5,
        zorder=6,
    )
    ax.axhspan(75, 80, color=GRID, alpha=0.45, zorder=1)
    ax.text(
        -0.4,
        82.0,
        "n=20 gate-read band 75–80% (the small-sample optimism the §8 record priced)",
        color=META,
        fontsize=8.5,
        zorder=6,
    )
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("keep rate (%)", fontsize=10)
    ax.set_xlabel("demo seed band (50 seeds each)", fontsize=10)
    ax.set_title(
        "Stage B — demo-collection keep rate by seed band "
        "(4 h wall, 313 kept ≥ 300 gate GREEN)",
        fontsize=11,
        loc="left",
        color=TEXT,
    )
    save(fig, "stageb_keep_rate.png")


def fig_stagec() -> None:
    d = json.loads(CURVE_C.read_text())
    ours = np.array(d["stagec_ar"], dtype=float)
    ref = np.array(d["rig_ft_r1_reference"], dtype=float)

    fig, ax = new_fig(9.0, 4.0)
    ax.plot(ref[:, 0], ref[:, 1], color=BLUE, linewidth=2.0, zorder=3)
    ax.plot(ours[:, 0], ours[:, 1], color=MAGENTA, linewidth=2.0, zorder=4)
    ax.set_yscale("log")
    mid = len(ours) // 2
    ax.text(
        ours[mid, 0],
        ours[mid, 1] * 3.2,
        f"grasp-SFT stage-C AR · latest step {int(ours[-1, 0])} loss {ours[-1, 1]:.3f}",
        color=MAGENTA,
        fontsize=9,
        ha="center",
        va="bottom",
        zorder=5,
    )
    ax.text(
        360,
        0.013,
        "rig-ft r1 (2000 steps, the recipe class this run copies)",
        color=BLUE,
        fontsize=9,
        ha="left",
        va="bottom",
        zorder=5,
    )
    ax.axhline(ours[0, 1], color=GRAY, linestyle=":", linewidth=1.0, zorder=2)
    ax.text(
        ref[-1, 0],
        ours[0, 1] * 0.85,
        f"warm-start loss {ours[0, 1]:.3f} (step {int(ours[0, 0])})",
        color=META,
        fontsize=8.5,
        ha="right",
        va="top",
        zorder=5,
    )
    ax.set_xlabel("training step", fontsize=10)
    ax.set_ylabel("train/action_flow_loss (log)", fontsize=10)
    ax.set_title(
        "Stage C — AR-primary SFT on the 313 collected demos "
        "(rig-ft r1 recipe verbatim-class, AE-only 5e-5, gb64)",
        fontsize=11,
        loc="left",
        color=TEXT,
    )
    save(fig, "stagec_loss.png")


def fig_staged() -> None:
    fig, ax = new_fig(9.0, 4.0)
    anchors_note = (
        "context anchors (record-only, NOT gates): "
        "ftrig4k +0.08 cm / 47 moved / ~1 success · "
        "stage-1 W0 +0.054 / 44 / 2 successes"
    )

    if FTRIG4K_100.exists():
        eps = json.loads(FTRIG4K_100.read_text())["episodes"]
        prog = [e["progress_final_cm"] for e in eps]
        seeds = [e["seed"] for e in eps]
        ax.scatter(seeds, prog, s=14, color=GOLD, alpha=0.75, zorder=3)
        ax.text(
            99,
            float(np.mean(prog)) + 0.35,
            f"ftrig4k anchor · mean {np.mean(prog):+.2f} cm",
            color=GOLD,
            fontsize=9,
            ha="right",
        )
        ax.axhline(np.mean(prog), color=GOLD, linewidth=1.0, linestyle="--", zorder=2)

    if STAGED_AR.exists():
        eps = json.loads(STAGED_AR.read_text())["episodes"]
        prog = [e["progress_final_cm"] for e in eps]
        seeds = [e["seed"] for e in eps]
        succ = [e["success_tick"] is not None for e in eps]
        ax.scatter(
            [s for s, ok in zip(seeds, succ, strict=True) if not ok],
            [p for p, ok in zip(prog, succ, strict=True) if not ok],
            s=18,
            color=MAGENTA,
            alpha=0.8,
            zorder=4,
        )
        ax.scatter(
            [s for s, ok in zip(seeds, succ, strict=True) if ok],
            [p for p, ok in zip(prog, succ, strict=True) if ok],
            s=52,
            color=MAGENTA,
            edgecolor=TEXT,
            linewidth=1.2,
            marker="o",
            zorder=5,
        )
        n_succ = sum(succ)
        ax.axhline(np.mean(prog), color=MAGENTA, linewidth=1.2, zorder=2)
        ax.text(
            0,
            float(np.mean(prog)) + 0.35,
            f"stage-D AR · mean {np.mean(prog):+.2f} cm · "
            f"{n_succ}/100 successes (ringed)",
            color=MAGENTA,
            fontsize=9,
        )
        title_tail = f"{n_succ}/100 successes"
    else:
        ax.text(
            50,
            2.6,
            "stage-D sim100 PENDING — fills at the eval boundary",
            ha="center",
            va="center",
            color=TEXT,
            fontsize=12,
            zorder=6,
        )
        title_tail = "PENDING"

    ax.axhline(0, color=GRID, linewidth=1.0, zorder=1)
    ax.set_xlabel("eval seed (frozen 0–99)", fontsize=10)
    ax.set_ylabel("progress toward disk, final (cm)", fontsize=10)
    ax.set_title(
        f"Stage D — sim100 per-seed progress vs banked anchors ({title_tail})",
        fontsize=11,
        loc="left",
        color=TEXT,
    )
    ax.text(0, ax.get_ylim()[0] + 0.15, anchors_note, color=META, fontsize=8)
    save(fig, "staged_progress_strip.png")


def fig_probe() -> None:
    """step2000 two-arm probe: success rate by seed band vs the anchor
    lineages (bar form — one measure, six identities; identity lives on
    the category axis + direct labels, hue follows the banked entity
    colors)."""
    if not PROBE_ANALYSIS.exists():
        return
    probe = json.loads(PROBE_ANALYSIS.read_text())
    rows: list[tuple[str, int, int, str]] = []  # label, successes, n, color
    band_specs = (
        ("train_band_kept", "trained spawns (kept demos)"),
        ("train_band_nonkept_expert_failed", "expert-failed spawns\n(never trained)"),
        ("unseen_0_99", "unseen seeds 0–99"),
    )
    for key, label in band_specs:
        arm = probe.get(key)
        if arm:
            rows.append((label, arm["successes"], arm["n"], MAGENTA))
    rows.append(("released base (init)\nPRIMARY ANCHOR", 9, 100, BLUE))
    rows.append(("ftrig4k (our flow stack)", 1, 100, GOLD))
    rows.append(("stage-1 W0", 2, 100, GRAY))

    fig, ax = new_fig(9.0, 3.4)
    y = np.arange(len(rows))[::-1]
    rates = [s / n * 100 for _, s, n, _ in rows]
    for yi, rate, (_label, s, n, color) in zip(y, rates, rows, strict=True):
        ax.barh(yi, rate, height=0.62, color=color, zorder=3)
        ax.text(
            rate + 1.2,
            yi,
            f"{s}/{n}  ({rate:.0f}%)",
            va="center",
            color=TEXT,
            fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, max(rates) * 1.25)
    ax.set_xlabel("success rate (%)", fontsize=10)
    ax.set_title(
        "step2000 probe — success by seed band vs anchors "
        "(base 9/100 = the causal comparator; corrupt-table checkpoint)",
        fontsize=11,
        loc="left",
        color=TEXT,
    )
    save(fig, "probe_bands.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extract",
        action="store_true",
        help="re-bank the log/state-derived curve JSONs",
    )
    args = ap.parse_args()
    if args.extract:
        extract()
        return
    fig_stagea()
    fig_stageb()
    fig_stagec()
    fig_staged()
    fig_probe()


if __name__ == "__main__":
    main()
