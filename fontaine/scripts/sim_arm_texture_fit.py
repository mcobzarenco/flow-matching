"""Arm micro-texture — fit the composite-stage texture parameters
(queue `sim-arm-texture-followup`; the photometric grade's registered
residuals: PLA print-layer local contrast real 8.36 vs graded 4.66,
STS3215 glint tail real p97 205.6 / p99 250.0 vs 125.2 / 127.2).

Solve-based fit through the PRODUCTION v3 composite (numpy post backend,
arm_photometrics='v1' + arm_texture='v1'), measuring the same pooled
statistics the photometric mine took from real
(reports/analysis__arm_photometric_mine.json):

  1. amplitude per population — two-probe quadrature solve on the
     local-contrast target (texture and the existing surface variance
     add roughly in quadrature through the blur chain);
  2. servo speckle (density, gain) — small grid on the glint-tail loss
     (p97/p99), then a linear gain refine at the winning density;
  3. servo amplitude re-solve with the chosen speckle on (the speckle
     itself adds local contrast), then a final confirm pass.

Output JSON feeds the frozen constants in sim/so101_sim.py
(ARM_TEXTURE_V1) — gated by the pinned 20x5 probe
(sim_arm_texture_read.py) before any promotion talk.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_arm_texture_fit.py \
      --mined reports/analysis__arm_photometric_mine.json \
      --out reports/analysis__arm_texture_fit.json
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

AMP_PROBES = (0.15, 0.35)
AMP_MAX = 0.6
DENSITY_GRID = (0.03, 0.08)
GAIN_GRID = (0.6, 1.0)
LC_RESOLVE_TOLERANCE = 0.75  # counts of local-contrast median


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


def texture_params(
    pla_amp: float,
    servo_amp: float,
    servo_density: float,
    servo_gain: float,
) -> dict:
    from sim.so101_sim import SO101Sim

    base = SO101Sim.ARM_TEXTURE_V1
    return {
        **{k: v for k, v in base.items() if k not in ("pla", "servo")},
        "pla": {
            "amplitude": float(pla_amp),
            "speckle_density": 0.0,
            "speckle_gain": 0.0,
        },
        "servo": {
            "amplitude": float(servo_amp),
            "speckle_density": float(servo_density),
            "speckle_gain": float(servo_gain),
        },
    }


def set_texture(sim, params: dict) -> None:  # noqa: ANN001
    sim.ARM_TEXTURE_V1 = params
    sim._init_arm_texture()


def tail_loss(sim_stats: dict, real_stats: dict) -> float:
    """The glint-tail objective: p97 + p99 squared error."""
    return float(
        sum(
            (sim_stats["luma_percentiles"][p] - real_stats["luma_percentiles"][p]) ** 2
            for p in ("97", "99")
        ),
    )


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

    from sim.so101_sim import SO101Sim

    mined = json.loads(args.mined.read_text())
    real = mined["populations"]
    targets = {
        "pla": real["pla"]["local_contrast_median"],
        "servo": real["servo"]["local_contrast_median"],
    }

    sim = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        arm_photometrics="v1",
        arm_texture="v1",
    )
    sim.reset(0)
    pops = population_geoms(sim)

    states = load_states(args.v2_root)
    rng = np.random.default_rng(0)
    pose_eps = rng.choice(EPISODES, size=FIT_POSES, replace=False)
    combos = []
    for k, episode in enumerate(pose_eps):
        traj = states[int(episode)]
        t = int(rng.integers(int(len(traj) * 0.15), int(len(traj) * 0.9)))
        combos.extend((traj[t], k * FIT_LOOKS + look) for look in range(FIT_LOOKS))

    def measure(pla_amp, servo_amp, density, gain):  # noqa: ANN001, ANN202
        set_texture(sim, texture_params(pla_amp, servo_amp, density, gain))
        stats = sample_sim_stats(sim, combos, pops)
        print(
            f"amp ({pla_amp:.3f}, {servo_amp:.3f}) speckle "
            f"({density:.3f}, {gain:.2f}): "
            + " | ".join(
                f"{name} lc {stats[name]['local_contrast_median']:.2f} "
                f"p97 {stats[name]['luma_percentiles']['97']:.1f} "
                f"p99 {stats[name]['luma_percentiles']['99']:.1f}"
                for name in ("pla", "servo")
            ),
        )
        return stats

    print("baseline (grade, zero texture) ...")
    baseline = measure(0.0, 0.0, 0.0, 0.0)
    lc0 = {name: baseline[name]["local_contrast_median"] for name in targets}

    probes = {name: [] for name in targets}
    for amp in AMP_PROBES:
        stats = measure(amp, amp, 0.0, 0.0)
        for name in targets:
            probes[name].append((amp, stats[name]["local_contrast_median"]))
    amps = {
        name: solve_amplitude(lc0[name], probes[name], targets[name])
        for name in targets
    }
    print("amplitude solve:", {k: round(v, 4) for k, v in amps.items()})

    print("servo speckle grid ...")
    grid = []
    best = (None, np.inf)
    for density in DENSITY_GRID:
        for gain in GAIN_GRID:
            stats = measure(amps["pla"], amps["servo"], density, gain)
            loss = tail_loss(stats["servo"], real["servo"])
            grid.append(
                {
                    "density": density,
                    "gain": gain,
                    "tail_loss": loss,
                    "servo": stats["servo"],
                },
            )
            if loss < best[1]:
                best = ((density, gain, stats), loss)
    density, gain, best_stats = best[0]

    # linear gain refine at the winning density: p97 responds ~affinely
    # to gain between the two grid gains
    others = [
        cell for cell in grid if cell["density"] == density and cell["gain"] != gain
    ]
    refine = None
    if others:
        g0, p0 = others[0]["gain"], others[0]["servo"]["luma_percentiles"]["97"]
        g1 = gain
        p1 = best_stats["servo"]["luma_percentiles"]["97"]
        if abs(p1 - p0) > 1e-6:
            solved = g0 + (real["servo"]["luma_percentiles"]["97"] - p0) * (g1 - g0) / (
                p1 - p0
            )
            solved = float(np.clip(solved, 0.0, 1.0))
            if abs(solved - gain) > 0.02:
                stats = measure(amps["pla"], amps["servo"], density, solved)
                if tail_loss(stats["servo"], real["servo"]) < best[1]:
                    refine = {"gain": solved, "servo": stats["servo"]}
                    gain = solved
                    best_stats = stats

    # the speckle adds servo local contrast on top of the modulation:
    # re-solve the servo amplitude with the chosen speckle live
    lc_now = best_stats["servo"]["local_contrast_median"]
    resolved = None
    if abs(lc_now - targets["servo"]) > LC_RESOLVE_TOLERANCE:
        # quadrature model with the speckle-on floor at the current amp
        floor = max(
            lc_now**2
            - (probes["servo"][-1][1] ** 2 - lc0["servo"] ** 2)
            * (amps["servo"] / AMP_PROBES[-1]) ** 2,
            1e-6,
        )
        resolved = solve_amplitude(
            float(np.sqrt(floor)),
            probes["servo"],
            targets["servo"],
        )
        print(f"servo amplitude re-solve with speckle on: {resolved:.4f}")
        amps["servo"] = resolved

    final_params = texture_params(amps["pla"], amps["servo"], density, gain)
    print("final confirm ...")
    set_texture(sim, final_params)
    final_stats = sample_sim_stats(sim, combos, pops)
    for name, target in targets.items():
        print(
            f"{name}: lc {final_stats[name]['local_contrast_median']:.2f} "
            f"(real {target:.2f}) "
            f"p97 {final_stats[name]['luma_percentiles']['97']:.1f} "
            f"(real {real[name]['luma_percentiles']['97']:.1f})",
        )

    payload = {
        "fitted": final_params,
        "final_sim_stats": final_stats,
        "final_texture_loss": {
            name: {
                "lc_gap": final_stats[name]["local_contrast_median"] - targets[name],
                "tail_loss": tail_loss(final_stats[name], real[name]),
                "photometric_guard_loss": stats_loss(final_stats[name], real[name]),
            }
            for name in targets
        },
        "baseline_sim_stats": baseline,
        "baseline_texture_loss": {
            name: {
                "lc_gap": baseline[name]["local_contrast_median"] - targets[name],
                "tail_loss": tail_loss(baseline[name], real[name]),
                "photometric_guard_loss": stats_loss(baseline[name], real[name]),
            }
            for name in targets
        },
        "real_stats": {name: real[name] for name in targets},
        "amplitude_solve": {
            "lc0": lc0,
            "probes": probes,
            "solved": amps,
            "servo_resolved_with_speckle": resolved,
        },
        "speckle_grid": grid,
        "gain_refine": refine,
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
