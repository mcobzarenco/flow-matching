"""Stage-1 boundary reads for the wrist-transfer screen
(posts/2026-08-14-prereg-wrist-transfer-screen.md §2-§3, frozen).

Consumes the stage-1 cell JSONs (`launch_wrist_screen_stage1.sh`:
w0/w1/w3 100 paired seeds, t1 25, hold 25) and enforces the registered
gates before any stage-2 spend:

- W0 sanity band: mean progress in [-0.3, +0.5] cm AND moved
  (|progress| >= 0.5 cm) in [25, 70]/100 (banked +0.08 / 47 is
  context, not a bit-anchor — registered config drift);
- hold floor: |mean progress| <= 0.01 cm, reset strikes 0;
- T1 moves: top-blackout paired Delta(engagement or |progress|) CI95
  excluding zero, else F-instrument;
- cross-arm pairing: per-seed spawn_xy bit-equal across arms.

Primary read: paired per-seed Delta progress (treatment - W0), mean +
bootstrap CI95 (seed 0, 10k resamples, the established recipe).
Secondary: Delta best-point, engagement flips, successes. Latency
medians are recorded as tripwires (not gated). The W0 determinism gate
ran at unit entry (launcher abort) and the honesty placement gate is a
stage-0 receipt (c5be36f) — neither is recomputed here.

Usage:
  uv run python fontaine/scripts/wrist_stage1_reads.py \
      [--in-dir outputs/sim/wrist_screen] \
      [--out reports/analysis__wrist_screen_stage1.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SANITY_PROGRESS_CM = (-0.3, 0.5)
SANITY_MOVED = (25, 70)
HOLD_FLOOR_CM = 0.01
UNTOUCHED_CM = 0.5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def load_arm(path: Path) -> dict[str, np.ndarray]:
    episodes = json.loads(path.read_text())["episodes"]
    return {
        "seeds": np.array([e["seed"] for e in episodes]),
        "progress": np.array([e["progress_final_cm"] for e in episodes]),
        "best_point": np.array([e["initial_cm"] - e["min_cm"] for e in episodes]),
        "moved": np.array(
            [abs(e["progress_final_cm"]) >= UNTOUCHED_CM for e in episodes],
            dtype=float,
        ),
        "success": np.array(
            [e["success_tick"] is not None for e in episodes],
            dtype=float,
        ),
        "strikes": np.array([e["reset_strikes"] for e in episodes]),
        "spawn_xy": [tuple(e["spawn_xy"]) for e in episodes],
        # First replan carries model warmup — dropped, per the sim100
        # recipe. The hold arm has no policy: no latency channel.
        "median_latency_ms": (
            float(np.median(lat))
            if (lat := [v for e in episodes for v in e["latency_ms"][1:]])
            else None
        ),
    }


def bootstrap_ci(deltas: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas)), replace=True)
    low, high = np.percentile(draws.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def paired_delta(treat: dict, base: dict, channel: str) -> dict[str, Any]:
    n = len(treat[channel])
    deltas = treat[channel][:n] - base[channel][:n]
    low, high = bootstrap_ci(deltas)
    return {
        "n": n,
        "mean": round(float(deltas.mean()), 4),
        "ci95": [round(low, 4), round(high, 4)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("outputs/sim/wrist_screen"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/analysis__wrist_screen_stage1.json"),
    )
    args = parser.parse_args()

    arms = {
        name: load_arm(args.in_dir / f"stage1_{name}.json")
        for name in ("w0", "w1", "w3", "t1", "hold")
        if (args.in_dir / f"stage1_{name}.json").exists()
    }
    missing = {"w0", "w1", "w3", "t1", "hold"} - set(arms)
    if missing:
        print(f"MISSING CELLS: {sorted(missing)} — boundary reads need all five")
        return 1

    w0 = arms["w0"]
    gates: dict[str, Any] = {}

    mean_w0 = float(w0["progress"].mean())
    moved_w0 = int(w0["moved"].sum())
    gates["w0_sanity_band"] = {
        "mean_progress_cm": round(mean_w0, 4),
        "band_cm": SANITY_PROGRESS_CM,
        "moved": moved_w0,
        "moved_band": SANITY_MOVED,
        "strikes": int(w0["strikes"].sum()),
        "pass": bool(
            SANITY_PROGRESS_CM[0] <= mean_w0 <= SANITY_PROGRESS_CM[1]
            and SANITY_MOVED[0] <= moved_w0 <= SANITY_MOVED[1],
        ),
    }

    hold_mean = float(arms["hold"]["progress"].mean())
    hold_strikes = int(arms["hold"]["strikes"].sum())
    gates["hold_floor"] = {
        "mean_progress_cm": round(hold_mean, 4),
        "strikes": hold_strikes,
        "pass": bool(abs(hold_mean) <= HOLD_FLOOR_CM and hold_strikes == 0),
    }

    n_t1 = len(arms["t1"]["seeds"])
    t1_abs = {"abs_progress": np.abs(arms["t1"]["progress"])}
    w0_abs = {"abs_progress": np.abs(w0["progress"][:n_t1])}
    t1_engagement = paired_delta(arms["t1"], w0, "moved")
    t1_absprog = paired_delta(t1_abs, w0_abs, "abs_progress")
    gates["t1_moves"] = {
        "delta_engagement": t1_engagement,
        "delta_abs_progress": t1_absprog,
        "pass": bool(
            t1_engagement["ci_excludes_zero"] or t1_absprog["ci_excludes_zero"],
        ),
    }

    pairing_ok = True
    for name in ("w1", "w3", "t1"):
        n = len(arms[name]["spawn_xy"])
        if arms[name]["spawn_xy"] != w0["spawn_xy"][:n]:
            pairing_ok = False
    gates["spawn_xy_pairing"] = {"pass": pairing_ok}

    gates["all_pass"] = all(g["pass"] for g in gates.values() if isinstance(g, dict))

    reads = {
        f"{name}_minus_w0": {
            "progress": paired_delta(arms[name], w0, "progress"),
            "best_point": paired_delta(arms[name], w0, "best_point"),
            "moved_flips": paired_delta(arms[name], w0, "moved"),
        }
        for name in ("w1", "w3", "t1")
    }
    summary = {
        name: {
            "mean_progress_cm": round(float(a["progress"].mean()), 4),
            "moved": int(a["moved"].sum()),
            "successes": int(a["success"].sum()),
            "median_latency_ms": (
                round(a["median_latency_ms"], 1)
                if a["median_latency_ms"] is not None
                else None
            ),
        }
        for name, a in arms.items()
    }

    payload = {"gates": gates, "paired_reads": reads, "per_arm": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    for name, gate in gates.items():
        if isinstance(gate, dict):
            print(f"GATE {name}: {'PASS' if gate['pass'] else 'FAIL'} {gate}")
    print(f"ALL GATES: {'PASS' if gates['all_pass'] else 'FAIL'}")
    for pair, r in reads.items():
        print(
            f"READ {pair}: dProgress {r['progress']['mean']:+.4f} cm "
            f"CI95 {r['progress']['ci95']} "
            f"(excl0={r['progress']['ci_excludes_zero']}), "
            f"moved flips {r['moved_flips']['mean']:+.3f}",
        )
    for name, s in summary.items():
        print(f"ARM {name}: {s}")
    print(f"banked -> {args.out}")
    return 0


if __name__ == "__main__":
    return_code = main()
    raise SystemExit(return_code)
