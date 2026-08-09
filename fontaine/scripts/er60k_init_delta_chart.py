"""Mid-run ER-init-delta chart for the er_60k run (record-only read).

One panel: the in-run 256-frame probe (eval_chunk_mae) of the live
fontaine_molmo2_er_60k_ddp4 run vs the 40k AR baseline at matched
steps. Seed 0 is shared (owner seed policy 22:51Z 08-09), so
shuffle-order variance is removed and the curve delta IS the init
effect (Molmo2-ER vs Molmo2) + the 0.19% natural-share rig data.
Never a kill line (pre-reg: record-only).

The er_60k series is pulled live from the box train log over ssh
(same host/log as the babysit registry entry); the 40k curve is the
adamc_postmortem_chart transcription (steps 500-13000) plus the
banked late anchors. Eval-report dark theme; blue = 40k baseline
(same entity color as the post-mortem chart), magenta = er_60k
(CVD-checked pair on this surface).

Usage: uv run python fontaine/scripts/er60k_init_delta_chart.py
       [--log-file PATH]   # local jsonl instead of ssh (testing)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adamc_postmortem_chart import AR40K

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/er60k"

BOX_HOST = "ubuntu@192.222.55.210"
BOX_LOG = "/home/ubuntu/train_fontaine_molmo2_er_60k_ddp4.log"
SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]

PAGE = "#121417"
BLUE = "#648fff"  # 40k AR baseline (Molmo2 init, seed 0)
MAGENTA = "#dc267f"  # er_60k (Molmo2-ER init, seed 0, +rig 0.19%)
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

# Banked 40k anchors past the postmortem transcription window.
AR40K_LATE = [(26500, 5.91), (40000, 6.2075)]


def fetch_er_series(log_file: str | None) -> list[tuple[int, float]]:
    if log_file:
        text = Path(log_file).read_text()
    else:
        proc = subprocess.run(
            ["ssh", *SSH_OPTS, BOX_HOST, f"grep -h eval_chunk_mae {BOX_LOG}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            sys.exit(
                f"ABORT: ssh fetch failed rc={proc.returncode}: {proc.stderr[:200]}",
            )
        text = proc.stdout
    series = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval_chunk_mae" in rec and "step" in rec:
            series.append((int(rec["step"]), float(rec["eval_chunk_mae"])))
    series.sort()
    if len(series) < 2:
        sys.exit(
            f"ABORT: only {len(series)} probe point(s) in the log — too early to chart",
        )
    return series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default=None, help="local jsonl copy (skip ssh)")
    args = ap.parse_args()

    er = fetch_er_series(args.log_file)
    er_max = er[-1][0]
    if er_max < 5000:
        print(
            f"NOTE: latest er_60k probe is step {er_max} < 5000 — the item opens at 5000",
        )

    ar40k = dict(AR40K + AR40K_LATE)
    matched = [(s, m, ar40k[s]) for s, m in er if s in ar40k]

    print("matched-step deltas (er_60k - 40k, negative = ER init ahead):")
    for s, e, b in matched:
        print(f"  step {s:>6}: er {e:8.4f}  40k {b:8.4f}  delta {e - b:+8.4f}")

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    fig.patch.set_facecolor(PAGE)

    xlim_hi = er_max + 800
    b_pts = [(s, v) for s, v in AR40K if s <= xlim_hi]
    bx, by = zip(*b_pts, strict=True)
    ex, ey = zip(*er, strict=True)

    ax.plot(bx, by, color=BLUE, linewidth=2, zorder=4)
    ax.plot(ex, ey, color=MAGENTA, linewidth=2, zorder=4)

    ax.text(
        bx[-1],
        by[-1] - 0.9,
        "40k baseline (Molmo2 init)",
        color=BLUE,
        fontsize=9.5,
        ha="right",
        va="top",
        fontweight="bold",
    )
    ax.text(
        ex[-1],
        ey[-1] + 0.9,
        "er_60k (Molmo2-ER init)",
        color=MAGENTA,
        fontsize=9.5,
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    if matched:
        s, e, b = matched[-1]
        ax.annotate(
            f"{e:.2f} vs {b:.2f} @{s}\n(delta {e - b:+.2f})",
            xy=(s, e),
            xytext=(s * 0.45, max(ey) * 0.75),
            color=TEXT,
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.8},
        )

    ax.set_xlabel("training step", color=META, fontsize=9)
    ax.set_ylabel(
        "in-run probe chunk MAE (256 held-out frames)",
        color=META,
        fontsize=9,
    )
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)

    delta_str = (
        f"{matched[-1][1] - matched[-1][2]:+.2f} @{matched[-1][0]}"
        if matched
        else "n/a"
    )
    fig.suptitle(
        "er_60k mid-run: ER init vs Molmo2 init at matched steps, shared "
        f"shuffle seed — latest matched delta {delta_str} (record-only read)",
        color=TEXT,
        fontsize=10,
        x=0.125,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "er60k_init_delta_midrun.svg",
        facecolor=PAGE,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "er60k_init_delta_midrun.png",
        facecolor=PAGE,
        bbox_inches="tight",
    )
    print(f"wrote {OUT}/er60k_init_delta_midrun.{{svg,png}}")


if __name__ == "__main__":
    main()
