"""Arm TRUE surface texture — fit the mjSpec-path texture parameters
(queue `sim-arm-surface-texture-mjspec`; the escalation registered by
the composite micro-texture REFUTATION 2026-08-14: statistically-matched
screen-space grain read MORE fake — the encoder wants coherent
surface-tracking structure, not matched marginals).

arm_texture='v2' bakes a quasi-periodic layer-line texture into the PLA
link materials at model compile time (mjSpec recompile,
physics-preservation oracles in tests/test_arm_texture.py). This script
solves its two free parameters through the PRODUCTION v3 composite
against the same mined real-PLA statistics the photometric fit used
(reports/analysis__arm_photometric_mine.json):

  1. period_px — small probe over candidate texture rows/line at a
     fixed amplitude; pick the period with the strongest local-contrast
     response through the blur chain (the most lc-efficient structure;
     an inefficient period would need a larger amplitude for the same
     read, spending clip headroom on nothing);
  2. amplitude — two-probe quadrature solve on the real PLA
     local-contrast median (texture and the existing surface variance
     add roughly in quadrature through the blur chain), capped at the
     tanh-bound clip headroom, then a final confirm pass.

Each configuration is a fresh SO101Sim (the texture is baked at compile
time). Output JSON feeds the frozen ARM_SURFACE_TEXTURE_V2 constants in
sim/so101_sim.py — gated by the pinned 20x5 probe
(sim_arm_surface_texture_read.py) before any promotion talk.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_arm_surface_texture_fit.py \
      --out reports/analysis__arm_surface_texture_fit.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
from sim_arm_photometric_fit import (
    EPISODES,
    FIT_LOOKS,
    FIT_POSES,
    load_states,
    population_geoms,
    sample_sim_stats,
    stats_loss,
)

PERIOD_GRID = (6.0, 10.0, 16.0, 24.0, 32.0)
PERIOD_PROBE_AMP = 0.25
AMP_PROBES = (0.15, 0.35)
AMP_MAX = 0.42  # tanh-bound clip headroom at center 0.7 is (1-0.7)/0.7
LC_TOLERANCE = 0.75  # counts of local-contrast median, same bar as the v1 fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    )
    parser.add_argument(
        "--mined",
        type=Path,
        default=Path("reports/analysis__arm_photometric_mine.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def make_sim(period_px: float | None, amplitude: float | None):  # noqa: ANN201
    """A fresh production-composite sim; texture params override the
    frozen class constants for the probe (None, None -> grade only)."""
    from sim.so101_sim import SO101Sim

    if period_px is None:
        return SO101Sim(
            render_style="v3",
            post_backend="numpy",
            arm_photometrics="v1",
        )
    frozen = SO101Sim.ARM_SURFACE_TEXTURE_V2
    SO101Sim.ARM_SURFACE_TEXTURE_V2 = {
        **frozen,
        "period_px": float(period_px),
        "amplitude": float(amplitude),
    }
    try:
        return SO101Sim(
            render_style="v3",
            post_backend="numpy",
            arm_photometrics="v1",
            arm_texture="v2",
        )
    finally:
        SO101Sim.ARM_SURFACE_TEXTURE_V2 = frozen


def solve_amplitude(
    lc0: float,
    probes: list[tuple[float, float]],
    target: float,
) -> float:
    """lc(a)^2 ~= lc0^2 + c * a^2 through the blur chain; c averaged
    over the probes, a* solved for the target (clipped to [0, AMP_MAX])."""
    cs = [max(lc**2 - lc0**2, 1e-6) / a**2 for a, lc in probes]
    c = float(np.mean(cs))
    if target <= lc0:
        return 0.0
    return float(np.clip(np.sqrt((target**2 - lc0**2) / c), 0.0, AMP_MAX))


def main() -> int:
    args = parse_args()

    mined = json.loads(args.mined.read_text())
    real = mined["populations"]
    target = real["pla"]["local_contrast_median"]

    states = load_states(args.v2_root)
    rng = np.random.default_rng(0)
    pose_eps = rng.choice(EPISODES, size=FIT_POSES, replace=False)
    combos = []
    for k, episode in enumerate(pose_eps):
        traj = states[int(episode)]
        t = int(rng.integers(int(len(traj) * 0.15), int(len(traj) * 0.9)))
        combos.extend((traj[t], k * FIT_LOOKS + look) for look in range(FIT_LOOKS))

    def measure(period_px, amplitude):  # noqa: ANN001, ANN202
        sim = make_sim(period_px, amplitude)
        sim.reset(0)
        stats = sample_sim_stats(sim, combos, population_geoms(sim))
        label = (
            "grade only"
            if period_px is None
            else f"period {period_px:.0f} amp {amplitude:.3f}"
        )
        print(
            f"{label}: pla lc {stats['pla']['local_contrast_median']:.2f} "
            f"(target {target:.2f})",
        )
        return stats

    print("baseline (grade, no texture) ...")
    baseline = measure(None, None)
    lc0 = baseline["pla"]["local_contrast_median"]

    print("period probe ...")
    period_probe = {}
    for period in PERIOD_GRID:
        stats = measure(period, PERIOD_PROBE_AMP)
        period_probe[period] = stats["pla"]["local_contrast_median"]
    period = max(period_probe, key=lambda p: period_probe[p])
    print(f"period {period:.0f} wins (lc response {period_probe[period]:.2f})")

    print("amplitude solve ...")
    probes = []
    for amp in AMP_PROBES:
        stats = measure(period, amp)
        probes.append((amp, stats["pla"]["local_contrast_median"]))
    amplitude = solve_amplitude(lc0, probes, target)
    print(f"amplitude solve: {amplitude:.4f} (cap {AMP_MAX})")

    print("final confirm ...")
    final_stats = measure(period, amplitude)
    lc_final = final_stats["pla"]["local_contrast_median"]
    lc_gap = lc_final - target
    print(
        f"pla: lc {lc_final:.2f} (real {target:.2f}, gap {lc_gap:+.2f}, "
        f"tolerance {LC_TOLERANCE})",
    )
    if abs(lc_gap) > LC_TOLERANCE and amplitude < AMP_MAX:
        # one linear correction inside the quadrature model, then stop —
        # the gate read prices the visible effect either way
        floor = max(lc0**2, 1e-6)
        c = max(lc_final**2 - floor, 1e-6) / amplitude**2
        amplitude = float(
            np.clip(np.sqrt(max(target**2 - floor, 0.0) / c), 0.0, AMP_MAX),
        )
        print(f"re-solve on the confirm point: {amplitude:.4f}")
        final_stats = measure(period, amplitude)
        lc_final = final_stats["pla"]["local_contrast_median"]
        lc_gap = lc_final - target

    payload = {
        "fitted": {"period_px": float(period), "amplitude": float(amplitude)},
        "final_sim_stats": {"pla": final_stats["pla"]},
        "final_texture_loss": {
            "pla": {
                "lc_gap": lc_gap,
                "photometric_guard_loss": stats_loss(final_stats["pla"], real["pla"]),
            },
        },
        "baseline_sim_stats": {"pla": baseline["pla"]},
        "baseline_texture_loss": {
            "pla": {
                "lc_gap": lc0 - target,
                "photometric_guard_loss": stats_loss(baseline["pla"], real["pla"]),
            },
        },
        "real_stats": {"pla": real["pla"]},
        "period_probe": {
            "grid": list(PERIOD_GRID),
            "probe_amplitude": PERIOD_PROBE_AMP,
            "lc_response": {str(p): lc for p, lc in period_probe.items()},
            "chosen": float(period),
        },
        "amplitude_solve": {
            "lc0": lc0,
            "probes": probes,
            "solved": float(amplitude),
            "amp_max": AMP_MAX,
        },
        "combos": [{"look": look} for _, look in combos],
    }
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    payload["config"] = {
        "v2_root": str(args.v2_root),
        "mined": str(args.mined),
        "episodes": list(EPISODES),
        "commit": commit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
