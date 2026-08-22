"""ch0 shift-constant read (CPU, record-only).

Input to the `ch0-shift-isolation-prereg` draft (the gripfix
verdict's <=10-branch follow-up, carrier-hunt ladder rung 2). Spec
frozen in-channel 06:45Z 08-22 (post 1540612893836574780) BEFORE
compute:

- Location stats: ch0 (shoulder pan) mean / median / q05 / q25 /
  q75 / q95 / std, action AND state, for demos / v2 / clean.
- Candidate constants: delta_mean = mean_demos - mean_clean and
  delta_median = median_demos - median_clean (demos = the shared
  convention target, the same choice gripfix made with 41.69).
- Post-shift check (record-only, descriptive): re-KS of shifted
  clean ch0 vs demos and vs v2 under each candidate - does a pure
  one-scalar location shift land ch0 inside the demos<->v2 reference
  band, or is the mismatch shape-carried?
- Oracles: recomputed unshifted KS must reproduce the banked
  manifold-probe values to 1e-9; shift-invariance sanity (equal
  shift of both samples leaves KS unchanged).

Output: reports/analysis__ch0_shift_constant_read.json + one
dark-mode distribution chart under fontaine/blog/src/assets/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from clean_content_manifold_probe import (
    DATASETS,
    EXPECTED_FRAMES,
    ks_distance,
    load_dataset,
)

CH = 0  # shoulder pan
REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "fontaine/blog/src/assets"
REPORT = REPO / "reports/analysis__ch0_shift_constant_read.json"

# Banked manifold-probe ch0 KS values (reports/analysis__clean_content_
# manifold_probe.json) - the recompute oracle reproduces these.
BANKED_KS = {
    "action": {
        "clean_vs_demos": 0.29459608348900745,
        "clean_vs_v2": 0.22828810035255076,
        "demos_vs_v2": 0.16098741148123732,
    },
    "state": {
        "clean_vs_demos": 0.2952079865254146,
        "clean_vs_v2": 0.22885552306156443,
        "demos_vs_v2": 0.15886174930311647,
    },
}

# House eval-report dark scheme (pdnorm_panel_ladder_chart.py lineage).
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
BLUE = "#648fff"  # demos
AMBER = "#ffb000"  # v2
MAGENTA = "#dc267f"  # clean
GREEN = "#2e9e73"  # shifted clean


def location_stats(x: np.ndarray) -> dict:
    q05, q25, q75, q95 = np.percentile(x, [5, 25, 75, 95])
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
        "std": float(np.std(x)),
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for name, path in DATASETS.items():
        d = load_dataset(path)
        n = d["action"].shape[0]
        assert n == EXPECTED_FRAMES[name], (name, n)
        data[name] = d
        print(f"loaded {name}: {n} frames")

    ch0 = {
        kind: {name: data[name][kind][:, CH].astype(np.float64) for name in DATASETS}
        for kind in ("action", "state")
    }

    # Oracle 1: recomputed unshifted KS reproduces the banked values.
    recomputed = {}
    for kind in ("action", "state"):
        recomputed[kind] = {
            "clean_vs_demos": ks_distance(ch0[kind]["clean"], ch0[kind]["demos"]),
            "clean_vs_v2": ks_distance(ch0[kind]["clean"], ch0[kind]["v2"]),
            "demos_vs_v2": ks_distance(ch0[kind]["demos"], ch0[kind]["v2"]),
        }
        for pair, banked in BANKED_KS[kind].items():
            got = recomputed[kind][pair]
            assert abs(got - banked) < 1e-9, (kind, pair, got, banked)
    print("oracle 1 green: banked ch0 KS reproduced to 1e-9")

    # Oracle 2: shift invariance - shifting BOTH samples by the same
    # constant leaves KS unchanged.
    a, b = ch0["action"]["clean"], ch0["action"]["demos"]
    base = ks_distance(a, b)
    shifted_both = ks_distance(a + 7.3, b + 7.3)
    assert abs(base - shifted_both) < 1e-12, (base, shifted_both)
    print("oracle 2 green: KS shift-invariant under equal shift")

    stats = {
        kind: {name: location_stats(ch0[kind][name]) for name in DATASETS}
        for kind in ("action", "state")
    }

    # Candidate constants: demos is the shared-convention target
    # (the same choice gripfix made with the 41.69 open convention).
    # One scalar for both columns, taken from the action stats
    # (the commanded convention; state-vs-action gap reported below).
    candidates = {
        "delta_mean": stats["action"]["demos"]["mean"]
        - stats["action"]["clean"]["mean"],
        "delta_median": stats["action"]["demos"]["median"]
        - stats["action"]["clean"]["median"],
    }
    state_deltas = {
        "delta_mean": stats["state"]["demos"]["mean"] - stats["state"]["clean"]["mean"],
        "delta_median": stats["state"]["demos"]["median"]
        - stats["state"]["clean"]["median"],
    }

    # Post-shift check (record-only): does a pure location shift land
    # clean's ch0 inside the demos<->v2 reference band?
    post_shift = {}
    for cand, delta in candidates.items():
        post_shift[cand] = {}
        for kind in ("action", "state"):
            shifted = ch0[kind]["clean"] + delta
            post_shift[cand][kind] = {
                "vs_demos": ks_distance(shifted, ch0[kind]["demos"]),
                "vs_v2": ks_distance(shifted, ch0[kind]["v2"]),
                "reference": recomputed[kind]["demos_vs_v2"],
            }

    report = {
        "spec_post": "1540612893836574780",
        "channel": CH,
        "frames": EXPECTED_FRAMES,
        "banked_ks_reproduced": recomputed,
        "location_stats": stats,
        "candidates_from_action": candidates,
        "state_deltas_for_reference": state_deltas,
        "post_shift_ks": post_shift,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"report -> {REPORT}")

    # Chart: ch0 action ECDFs, unshifted + the winning candidate.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=PAGE)
    series = [("demos", BLUE), ("v2", AMBER), ("clean", MAGENTA)]
    for ax, cand in zip(axes, ("delta_mean", "delta_median"), strict=True):
        ax.set_facecolor(PAGE)
        for name, color in series:
            x = np.sort(ch0["action"][name])
            sub = max(1, x.size // 4000)
            ax.plot(
                x[::sub],
                np.linspace(0, 1, x.size)[::sub],
                color=color,
                lw=1.6,
                label=name,
            )
        xs = np.sort(ch0["action"]["clean"] + candidates[cand])
        ax.plot(
            xs,
            np.linspace(0, 1, xs.size),
            color=GREEN,
            lw=1.6,
            ls="--",
            label=f"clean + {cand} ({candidates[cand]:+.2f})",
        )
        ks_post = post_shift[cand]["action"]["vs_demos"]
        ax.set_title(
            f"{cand}: post-shift KS vs demos {ks_post:.3f} (ref 0.161)",
            color=TEXT,
            fontsize=10,
        )
        ax.set_xlabel("ch0 shoulder pan (action, raw units)", color=META, fontsize=9)
        ax.tick_params(colors=META, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.6)
        ax.legend(facecolor=PAGE, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    axes[0].set_ylabel("ECDF", color=META, fontsize=9)
    fig.tight_layout()
    out = ASSETS / "ch0_shift_constant_ecdf.png"
    fig.savefig(out, dpi=140, facecolor=PAGE, bbox_inches="tight")
    print(f"chart -> {out}")

    # Console table for the pre-reg draft.
    print("\nch0 location stats (action / state):")
    for name in DATASETS:
        a, s = stats["action"][name], stats["state"][name]
        print(
            f"  {name:6s} action mean {a['mean']:+8.3f} median {a['median']:+8.3f} "
            f"[q05 {a['q05']:+8.3f} q95 {a['q95']:+8.3f}] std {a['std']:7.3f}",
        )
        print(
            f"  {name:6s} state  mean {s['mean']:+8.3f} median {s['median']:+8.3f} "
            f"[q05 {s['q05']:+8.3f} q95 {s['q95']:+8.3f}] std {s['std']:7.3f}",
        )
    print(f"\ncandidates (action-derived): {json.dumps(candidates, indent=1)}")
    print(f"state deltas (reference): {json.dumps(state_deltas, indent=1)}")
    print("\npost-shift KS:")
    for cand, kinds in post_shift.items():
        for kind, v in kinds.items():
            print(
                f"  {cand:13s} {kind:6s} vs_demos {v['vs_demos']:.4f} "
                f"vs_v2 {v['vs_v2']:.4f} (ref {v['reference']:.4f})",
            )


if __name__ == "__main__":
    main()
