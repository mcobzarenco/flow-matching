"""Servo/controller sysid for the SO-101 sim (queue: sim-servo-sysid).

Resolves the 56x kp discrepancy between our vendored menagerie model
(kp 998.22, kv 2.731, forcerange +-2.94) and TheRobotStudio's upstream
MJCF (kp 17.8, kv 0, +-3.35) for the same STS3215 servo, by SIMPLER's
recipe: open-loop replay of real rig episodes' recorded action streams
through the sim arm, scoring the sim joint trajectory against the
recorded observation.state, then fitting the actuator/joint params that
minimize that replay error.

Design notes:
  - Replay is arm-only (benchy parked down-table): the real episodes
    grasp a boat the sim replay does not carry, so the GRIPPER joint is
    contact-coupled in real and excluded from the fit metric (reported
    record-only); the ~40 g payload on the arm joints is a stated
    limitation, not modeled.
  - Fit episodes are TRAIN-side; the reported before/after MAE is on the
    er-60k deterministic HOLDOUT episodes (fraction 0.1, split-seed 0:
    clean ep 2; v2 eps 1, 4, 10, 36, 44) so the validation set matches
    the policy-eval holdout convention.
  - The sim control tick is 7 x 5 ms = 35 ms vs the rig's 33.3 ms; the
    replay compares tick-for-tick, so the ~5% timebase skew is absorbed
    into the fitted gains - correct for eval use, stated in the note.
  - Params are shared across the 6 follower actuators (same servo
    class), fitted in log10 space: kp, kv, forcerange, joint damping,
    frictionloss, armature.

The winning fit is pinned in so101_sim.SERVO_SYSID and applied at model
load; scoring here stays self-contained because every candidate is
written onto the model via set_params before replay.

Usage:
  MUJOCO_GL=egl uv run python -m sim.sysid_servo            # references + fit
  MUJOCO_GL=egl uv run python -m sim.sysid_servo --no-fit   # references only
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from . import OUTPUT_DIR
from .so101_sim import JOINTS, LEADER_DEGREES, SERVO_SYSID, SO101Sim

DATASETS_ROOT = Path.home() / "datasets" / "mcobzarenco"
# Train-side fit episodes vs the er-60k deterministic holdout (see
# bijou.data.holdout_episodes(fraction=0.1, split_seed=0)).
FIT_EPISODES = (
    ("so101_pick_place_clean", 0),
    *(("so101_pick_place_v2", e) for e in (0, 7, 20, 30, 47)),
)
VAL_EPISODES = (
    ("so101_pick_place_clean", 2),
    *(("so101_pick_place_v2", e) for e in (1, 4, 10, 36, 44)),
)
ARM = slice(0, 5)  # fit metric: 5 arm joints; gripper record-only

PARAM_NAMES = ("kp", "kv", "forcerange", "damping", "frictionloss", "armature")
MENAGERIE = {
    "kp": 998.22,
    "kv": 2.731,
    "forcerange": 2.94,
    "damping": 0.6,
    "frictionloss": 0.052,
    "armature": 0.028,
}
# Upstream publishes only the actuator triple; joint params kept at the
# menagerie values so the reference isolates the gain question.
UPSTREAM = dict(MENAGERIE, kp=17.8, kv=0.0, forcerange=3.35)
# Log10 bounds for the fit (kv floored at 1e-3 so kv=0 is representable).
LOG_BOUNDS = {
    "kp": (0.5, 3.5),
    "kv": (-3.0, 1.5),
    "forcerange": (-0.5, 1.0),
    "damping": (-3.0, 1.0),
    "frictionloss": (-4.0, 0.5),
    "armature": (-4.0, 0.0),
}


def load_episode(repo: str, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """(actions, states) [T, 6] float64 degrees for one recorded episode."""
    frames = []
    for file in sorted((DATASETS_ROOT / repo / "data").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(
            file,
            columns=["episode_index", "frame_index", "action", "observation.state"],
        )
        part = df[df.episode_index == episode]
        if len(part):
            frames.append(part)
    if not frames:
        raise FileNotFoundError(
            f"episode {episode} not found under {DATASETS_ROOT / repo}",
        )
    df = pd.concat(frames).sort_values("frame_index")
    return (
        np.stack(df["action"].to_list()).astype(np.float64),
        np.stack(df["observation.state"].to_list()).astype(np.float64),
    )


def set_params(sim: SO101Sim, params: dict[str, float]) -> None:
    """Write the shared servo params onto the 6 follower actuators/dofs."""
    model = sim.model
    for name in JOINTS:
        actuator = model.actuator(name)
        actuator.gainprm[0] = params["kp"]
        actuator.biasprm[1] = -params["kp"]
        actuator.biasprm[2] = -params["kv"]
        actuator.forcerange[:] = (-params["forcerange"], params["forcerange"])
        dof = model.joint(name).dofadr[0]
        model.dof_damping[dof] = params["damping"]
        model.dof_frictionloss[dof] = params["frictionloss"]
        model.dof_armature[dof] = params["armature"]


def replay_trajectory(
    sim: SO101Sim,
    actions: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Open-loop replay; returns the sim joint trajectory [T-1, 6]
    (degrees) aligned so row t compares against states[t+1]."""
    model, data = sim.model, sim.data
    mujoco.mj_resetData(model, data)
    adr = sim._benchy_qpos
    data.qpos[adr : adr + 3] = (0.9, 0.1, 0.001)  # benchy parked down-table
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    start = np.clip(np.deg2rad(states[0]), sim._ctrl_low, sim._ctrl_high)
    data.qpos[sim._joint_qpos] = start
    data.ctrl[sim._actuator_ids] = start
    data.ctrl[sim._leader_actuators] = np.deg2rad(LEADER_DEGREES)
    mujoco.mj_step(
        model,
        data,
        nstep=105,
    )  # 0.5 s settle to servo equilibrium at state[0]

    sim_states = np.empty((len(actions) - 1, len(JOINTS)))
    for t in range(len(actions) - 1):
        data.ctrl[sim._actuator_ids] = np.clip(
            np.deg2rad(actions[t]),
            sim._ctrl_low,
            sim._ctrl_high,
        )
        mujoco.mj_step(model, data, nstep=7)
        sim_states[t] = np.rad2deg(data.qpos[sim._joint_qpos])
    return sim_states


def replay_episode(
    sim: SO101Sim,
    actions: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Per-frame per-joint abs replay error [T-1, 6] (degrees)."""
    return np.abs(replay_trajectory(sim, actions, states) - states[1:])


def score(sim: SO101Sim, episodes: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Pooled replay metrics over episodes for the CURRENT model params."""
    errors = np.concatenate([replay_episode(sim, a, s) for a, s in episodes])
    per_joint = errors.mean(axis=0)
    return {
        "arm_mae_deg": float(per_joint[ARM].mean()),
        "arm_p95_deg": float(np.quantile(errors[:, ARM], 0.95)),
        "gripper_mae_deg": float(per_joint[5]),
        "per_joint_mae_deg": {
            name: float(v) for name, v in zip(JOINTS, per_joint, strict=True)
        },
        "frames": len(errors),
    }


def _golden_section(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float,
) -> tuple[float, float]:
    """Deterministic 1-D golden-section minimize on [lo, hi]."""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    x = (a + b) / 2.0
    return x, min(fc, fd)


def fit(
    sim: SO101Sim,
    episodes: list[tuple[np.ndarray, np.ndarray]],
    start: dict[str, float],
    sweeps: int = 4,
) -> tuple[dict, int]:
    """Coordinate descent (golden-section per param) in clipped log10
    space from `start`; dependency-free and deterministic. Returns
    (fitted params, objective evaluations)."""
    nfev = 0

    def decode(x: np.ndarray) -> dict[str, float]:
        params: dict[str, float] = dict(zip(PARAM_NAMES, 10.0**x, strict=True))
        if params["kv"] <= 2e-3:
            params["kv"] = 0.0
        return params

    def objective(x: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        set_params(sim, decode(x))
        return float(
            np.concatenate([replay_episode(sim, a, s) for a, s in episodes])[
                :,
                ARM,
            ].mean(),
        )

    x = np.array(
        [np.clip(np.log10(max(start[n], 1e-3)), *LOG_BOUNDS[n]) for n in PARAM_NAMES],
    )
    best = objective(x)
    span = 1.0  # +-1 decade first sweep, halved each sweep
    for _ in range(sweeps):
        for i, name in enumerate(PARAM_NAMES):
            lo = max(LOG_BOUNDS[name][0], x[i] - span)
            hi = min(LOG_BOUNDS[name][1], x[i] + span)

            def f1(v: float, i: int = i) -> float:
                trial = x.copy()
                trial[i] = v
                return objective(trial)

            xi, fx = _golden_section(f1, lo, hi, tol=0.02)
            if fx < best:
                x[i], best = xi, fx
        span /= 2.0
    return decode(x), nfev


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fit", action="store_true", help="score references only")
    parser.add_argument("--maxfev", type=int, default=300)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = SO101Sim(width=64, height=48)  # renderer unused on the replay path
    fit_eps = [load_episode(r, e) for r, e in FIT_EPISODES]
    val_eps = [load_episode(r, e) for r, e in VAL_EPISODES]
    # Context floor: how far the REAL servo trails its own commands
    # (|action[t] - state[t+1]|). A sim replay cannot be expected to track
    # the recorded states better than the real tracking dynamics are wide;
    # this is the scale replay MAEs should be read against, not zero.
    real_lag = {
        split: float(
            np.concatenate([np.abs(a[:-1, ARM] - s[1:, ARM]) for a, s in eps]).mean(),
        )
        for split, eps in (("fit", fit_eps), ("val", val_eps))
    }
    print(
        f"real servo lag |action[t]-state[t+1]| arm mean: fit {real_lag['fit']:.3f} deg, val {real_lag['val']:.3f} deg",
    )
    # Merge into an existing report so a references-only rerun never
    # discards previously fitted candidates.
    out = OUTPUT_DIR / "sysid_servo.json"
    report: dict = json.loads(out.read_text()) if out.exists() else {}
    report.update(
        {
            "fit_episodes": [list(e) for e in FIT_EPISODES],
            "val_episodes": [list(e) for e in VAL_EPISODES],
            "real_lag_arm_mae_deg": real_lag,
        },
    )
    report.setdefault("candidates", {})

    for label, params in (
        ("menagerie", MENAGERIE),
        ("upstream", UPSTREAM),
        ("pinned_so101_sim", SERVO_SYSID),
    ):
        set_params(sim, params)
        entry = {
            "params": params,
            "fit": score(sim, fit_eps),
            "val": score(sim, val_eps),
        }
        report["candidates"][label] = entry
        print(f"[{label}] {params}")
        print(
            f"  fit arm MAE {entry['fit']['arm_mae_deg']:.3f} deg | val arm MAE {entry['val']['arm_mae_deg']:.3f} deg",
        )
        print(f"  val per-joint {entry['val']['per_joint_mae_deg']}")

    if not args.no_fit:
        best_label, best = None, None
        for label, start in (
            ("from_menagerie", MENAGERIE),
            ("from_upstream", dict(UPSTREAM, kv=0.5)),
        ):
            t0 = time.time()
            fitted, nfev = fit(sim, fit_eps, start)
            set_params(sim, fitted)
            entry: dict = {
                "params": fitted,
                "fit": score(sim, fit_eps),
                "val": score(sim, val_eps),
                "nfev": nfev,
            }
            report["candidates"][f"fitted_{label}"] = entry
            print(
                f"[fitted_{label}] {nfev} evals in {time.time() - t0:.0f}s -> {fitted}",
            )
            print(
                f"  fit arm MAE {entry['fit']['arm_mae_deg']:.3f} deg | val arm MAE {entry['val']['arm_mae_deg']:.3f} deg",
            )
            print(f"  val per-joint {entry['val']['per_joint_mae_deg']}")
            if best is None or entry["val"]["arm_mae_deg"] < best["val"]["arm_mae_deg"]:
                best_label, best = f"fitted_{label}", entry
        report["best"] = best_label

    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
