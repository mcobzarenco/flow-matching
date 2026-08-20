"""Mechanism-(a) clean-content manifold probe (CPU, record-only).

Queue item `clean-content-manifold-probe`, riding the demos+clean
pre-reg (posts/2026-08-20-prereg-demos-plus-clean.md). Spec frozen
in-channel 14:46Z 08-20 (post 1540009095803703316) BEFORE compute:

- Primary: per-channel two-sample KS distance D (action ch0-5 +
  state ch0-5), clean<->demos and clean<->v2, each judged against the
  demos<->v2 D on the same channel as the reference pair. Clean
  counts as off-manifold on a channel only where its D exceeds that
  baseline. Record-only, no gate.
- Secondary: overlap coefficient (shared 256-bin histogram over the
  pooled 0.1-99.9% range).
- Pacing: episode length (frames) + within-episode per-step
  |d action| median/p95 per channel (velocity proxy).
- Gripper-cycle shape: action ch5 per-episode trajectories over
  normalized time + open/close transition count (hysteresis at
  25%/75% of the range pooled across all three datasets).

Outputs: reports/analysis__clean_content_manifold_probe.json,
two dark-mode charts under fontaine/blog/src/assets/, table printed
for the pre-reg post's results section.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATASETS = {
    "demos": "~/datasets/fontaine/grasp_demos_v2/merged",
    "v2": "~/datasets/mcobzarenco/so101_pick_place_v2",
    "clean": "~/datasets/mcobzarenco/so101_pick_place_clean",
}
EXPECTED_FRAMES = {"demos": 1_942_375, "v2": 32_679, "clean": 3_399}
N_CH = 6
GRIPPER_CH = 5
OVL_BINS = 256
HYST_LO_FRAC, HYST_HI_FRAC = 0.25, 0.75

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "fontaine/blog/src/assets"
REPORT = REPO / "reports/analysis__clean_content_manifold_probe.json"

# House eval-report dark scheme (see pdnorm_panel_ladder_chart.py);
# trio re-validated for dark-surface CVD separation this session
# (min pairwise OKLab dE 19.8 deutan, 29.5 normal).
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
BLUE = "#648fff"  # demos (and the clean-vs-demos pair)
AMBER = "#ffb000"  # v2 (and the clean-vs-v2 pair)
MAGENTA = "#dc267f"  # clean
GRAY = "#9aa0a8"  # demos-vs-v2 reference baseline


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D (no p-value)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    pooled = np.concatenate([a, b])
    fa = np.searchsorted(a, pooled, side="right") / a.size
    fb = np.searchsorted(b, pooled, side="right") / b.size
    return float(np.abs(fa - fb).max())


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = OVL_BINS) -> float:
    """Histogram-intersection OVL on shared bins over the pooled 0.1-99.9% range."""
    pooled = np.concatenate([a, b]).astype(np.float64)
    lo, hi = np.percentile(pooled, [0.1, 99.9])
    if hi <= lo:
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return float(np.minimum(pa, pb).sum())


def gripper_transitions(traj: np.ndarray, lo: float, hi: float) -> int:
    """Open/close transition count with hysteresis: count state flips
    between <lo (closed) and >hi (open); mid-band never flips state."""
    state = 0  # -1 closed, +1 open, 0 unknown
    flips = 0
    for v in traj:
        s = -1 if v < lo else (1 if v > hi else 0)
        if s == 0:
            continue
        if state != 0 and s != state:
            flips += 1
        state = s
    return flips


def load_dataset(path: str) -> dict:
    import pyarrow.parquet as pq

    files = sorted(Path(path).expanduser().glob("data/*/*.parquet"))
    assert files, f"no parquet under {path}"
    acts, states, eps = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["action", "observation.state", "episode_index"])
        acts.append(np.stack(t["action"].to_numpy(zero_copy_only=False)))
        states.append(np.stack(t["observation.state"].to_numpy(zero_copy_only=False)))
        eps.append(np.asarray(t["episode_index"]))
    return {
        "action": np.concatenate(acts),
        "state": np.concatenate(states),
        "episode_index": np.concatenate(eps),
    }


def episode_slices(ep_idx: np.ndarray) -> list[slice]:
    bounds = np.flatnonzero(np.diff(ep_idx) != 0) + 1
    starts = np.concatenate([[0], bounds])
    stops = np.concatenate([bounds, [ep_idx.size]])
    return [slice(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def within_episode_deltas(arr: np.ndarray, ep_idx: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(arr, axis=0))
    same_ep = np.diff(ep_idx) == 0
    return d[same_ep]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for name, path in DATASETS.items():
        d = load_dataset(path)
        n = d["action"].shape[0]
        assert n == EXPECTED_FRAMES[name], (name, n)
        d["slices"] = episode_slices(d["episode_index"])
        d["deltas"] = within_episode_deltas(d["action"], d["episode_index"])
        data[name] = d
        print(f"loaded {name}: {n} frames, {len(d['slices'])} episodes")

    pairs = [("clean", "demos"), ("clean", "v2"), ("demos", "v2")]
    ks = {"action": {}, "state": {}, "action_delta": {}}
    ovl = {"action": {}, "state": {}}
    for kind in ("action", "state"):
        for x, y in pairs:
            key = f"{x}_vs_{y}"
            ks[kind][key] = [
                ks_distance(data[x][kind][:, c], data[y][kind][:, c])
                for c in range(N_CH)
            ]
            ovl[kind][key] = [
                overlap_coefficient(data[x][kind][:, c], data[y][kind][:, c])
                for c in range(N_CH)
            ]
    for x, y in pairs:
        ks["action_delta"][f"{x}_vs_{y}"] = [
            ks_distance(data[x]["deltas"][:, c], data[y]["deltas"][:, c])
            for c in range(N_CH)
        ]

    pacing = {}
    for name, d in data.items():
        lengths = np.array([s.stop - s.start for s in d["slices"]])
        pacing[name] = {
            "episodes": int(lengths.size),
            "ep_len_mean": float(lengths.mean()),
            "ep_len_median": float(np.median(lengths)),
            "ep_len_min": int(lengths.min()),
            "ep_len_max": int(lengths.max()),
            "delta_median": [float(v) for v in np.median(d["deltas"], axis=0)],
            "delta_p95": [float(v) for v in np.percentile(d["deltas"], 95, axis=0)],
        }

    pooled_grip = np.concatenate([d["action"][:, GRIPPER_CH] for d in data.values()])
    gmin, gmax = float(pooled_grip.min()), float(pooled_grip.max())
    lo = gmin + HYST_LO_FRAC * (gmax - gmin)
    hi = gmin + HYST_HI_FRAC * (gmax - gmin)
    gripper = {"pooled_min": gmin, "pooled_max": gmax, "hyst_lo": lo, "hyst_hi": hi}
    for name, d in data.items():
        g = d["action"][:, GRIPPER_CH]
        counts = np.array([gripper_transitions(g[s], lo, hi) for s in d["slices"]])
        # POST-HOC robustness read (labeled; the frozen spec's pooled-range
        # hysteresis is primary): thresholds from the dataset's OWN range,
        # so amplitude compression doesn't mask cycling behavior.
        own_lo = g.min() + HYST_LO_FRAC * (g.max() - g.min())
        own_hi = g.min() + HYST_HI_FRAC * (g.max() - g.min())
        own_counts = np.array(
            [gripper_transitions(g[s], own_lo, own_hi) for s in d["slices"]],
        )
        gripper[name] = {
            "transitions_mean": float(counts.mean()),
            "transitions_median": float(np.median(counts)),
            "transitions_min": int(counts.min()),
            "transitions_max": int(counts.max()),
            "post_hoc_own_range_transitions_mean": float(own_counts.mean()),
            "post_hoc_own_range_transitions_median": float(np.median(own_counts)),
            "open_plateau_q99": float(np.percentile(g, 99)),
            "open_max": float(g.max()),
        }

    # Off-manifold verdict per the frozen framing: clean's D exceeds
    # the demos<->v2 baseline on the same channel.
    exceed = {}
    for kind in ("action", "state", "action_delta"):
        base = np.array(ks[kind]["demos_vs_v2"])
        exceed[kind] = {
            "clean_vs_demos": [
                bool(v) for v in np.array(ks[kind]["clean_vs_demos"]) > base
            ],
            "clean_vs_v2": [bool(v) for v in np.array(ks[kind]["clean_vs_v2"]) > base],
        }

    payload = {
        "spec_post": "1540009095803703316",
        "datasets": {k: str(Path(v).expanduser()) for k, v in DATASETS.items()},
        "frames": EXPECTED_FRAMES,
        "ks": ks,
        "ovl": ovl,
        "ks_exceeds_reference_pair": exceed,
        "pacing": pacing,
        "gripper": gripper,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {REPORT}")

    # ---- Chart A: per-channel KS distance, three panels ----
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.6), sharex=True)
    fig.patch.set_facecolor(PAGE)
    panel_kinds = [
        ("action", "Action (raw units)"),
        ("state", "State"),
        ("action_delta", "Per-step |Δaction| (velocity proxy)"),
    ]
    pair_style = [
        ("clean_vs_demos", "clean ↔ demos", BLUE),
        ("clean_vs_v2", "clean ↔ v2", AMBER),
        ("demos_vs_v2", "demos ↔ v2 (reference pair)", GRAY),
    ]
    width = 0.26
    xs = np.arange(N_CH)
    for ax, (kind, title) in zip(axes, panel_kinds, strict=True):
        ax.set_facecolor(PAGE)
        for i, (key, label, color) in enumerate(pair_style):
            vals = ks[kind][key]
            ax.bar(
                xs + (i - 1) * width,
                vals,
                width - 0.03,
                color=color,
                zorder=3,
                label=label if kind == "action" else None,
            )
        ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=6)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("KS distance D", color=META, fontsize=8)
        ax.tick_params(colors=META, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5, zorder=1)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(
        [
            "ch0\nshoulder pan",
            "ch1\nshoulder lift",
            "ch2\nelbow",
            "ch3\nwrist flex",
            "ch4\nwrist roll",
            "ch5\ngripper",
        ],
        color=META,
    )
    axes[0].legend(
        loc="upper left",
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
    )
    fig.suptitle(
        "How far off the manifold are the 7 clean episodes?  KS distance per channel\n"
        "(off-manifold evidence only where a clean bar exceeds the gray reference pair)",
        color=TEXT,
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_a = ASSETS / "clean-manifold-ks.png"
    fig.savefig(out_a, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"wrote {out_a}")

    # ---- Chart B: gripper trajectories, small multiples ----
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    fig.patch.set_facecolor(PAGE)
    panels = [
        ("demos", BLUE, "demos — 50 of 5000 eps (strided)"),
        ("v2", AMBER, "v2 — all 50 eps"),
        ("clean", MAGENTA, "clean — all 7 eps"),
    ]
    for ax, (name, color, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(PAGE)
        slices = data[name]["slices"]
        if len(slices) > 50:
            slices = slices[:: len(slices) // 50][:50]
        for s in slices:
            traj = data[name]["action"][s, GRIPPER_CH]
            t = np.linspace(0, 1, traj.size)
            ax.plot(
                t,
                traj,
                color=color,
                linewidth=0.8,
                alpha=0.9 if name == "clean" else 0.25,
                zorder=3,
            )
        for guide, ls in ((lo, ":"), (hi, ":")):
            ax.axhline(guide, color=META, linewidth=0.7, linestyle=ls, zorder=2)
        ax.set_title(title, color=TEXT, fontsize=9, loc="left")
        ax.set_xlabel("normalized episode time", color=META, fontsize=8)
        ax.tick_params(colors=META, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.4, zorder=1)
    axes[0].set_ylabel("gripper (action ch5, raw)", color=META, fontsize=8)
    fig.suptitle(
        "Gripper cycle shape per dataset (dotted = hysteresis band)",
        color=TEXT,
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_b = ASSETS / "clean-manifold-gripper.png"
    fig.savefig(out_b, dpi=150, facecolor=PAGE)
    plt.close(fig)
    print(f"wrote {out_b}")

    # ---- Table for the post ----
    print(
        "\n| channel | KS clean↔demos | KS clean↔v2 | KS demos↔v2 (ref) | OVL clean↔demos | OVL clean↔v2 |",
    )
    print("|---|---|---|---|---|---|")
    names = [
        "shoulder pan",
        "shoulder lift",
        "elbow",
        "wrist flex",
        "wrist roll",
        "gripper",
    ]
    for c in range(N_CH):
        print(
            f"| ch{c} {names[c]} | "
            f"{ks['action']['clean_vs_demos'][c]:.3f} | "
            f"{ks['action']['clean_vs_v2'][c]:.3f} | "
            f"{ks['action']['demos_vs_v2'][c]:.3f} | "
            f"{ovl['action']['clean_vs_demos'][c]:.3f} | "
            f"{ovl['action']['clean_vs_v2'][c]:.3f} |",
        )


if __name__ == "__main__":
    main()
