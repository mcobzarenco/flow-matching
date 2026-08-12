"""Replay control-loss probe (queue: sim-sysid-replay-control-loss).

SIMPLER's offline sysid validator, run against our servo model for the
first time: replay the 26 reference episodes' recorded action streams
through the sim physics-only (no GL, no GPU) and score the sim
end-effector trajectory against the EE trajectory implied by the
recorded ``observation.state``, with SIMPLER's loss

    L = mean ||dx||  +  mean arcsin(||dR||_F / (2 sqrt 2))

(translation in meters, rotation in radians; the arcsin term is HALF
the geodesic angle between the frames — ||R1-R2||_F = 2 sqrt2
sin(theta/2) — kept exactly as SIMPLER specifies so our number reads
against their Table II anchors: control loss 0.131/0.267/0.432 mapped
monotonically to MMRV 0.031/0.070/0.100).

Both trajectories go through the SAME kinematic chain (the sim model's
``gripperframe`` site, which rides the 5 arm joints; the jaw joint
moves a child body), so the loss isolates servo DYNAMICS: recorded
states vs what the sim servo model produces from the same commands.
Scale caveat for the anchor comparison: SIMPLER's arms are lab-scale
(~1 m reach) while the SO-101 reaches ~0.35 m, so our translation term
is intrinsically smaller at equal relative fidelity — the per-term
breakdown and the real-lag floor below are the honest context.

Reported per candidate param set (vendored menagerie kp 998 / upstream
TheRobotStudio kp 17.8 / pinned SERVO_SYSID fit):
  - pooled L, translation and rotation terms, per-episode rows;
  - arm joint-space MAE (continuity with sim.sysid_servo's numbers);
  - split "all 26" vs "held-out 23" (v2 episodes 0, 7, 20 were
    sysid FIT episodes — the held-out split is the leakage-free read);
  - the real-lag FLOOR: EE distance between the clipped commanded
    targets and the next recorded state — a replay cannot be expected
    to track the recorded EE more tightly than the real servo tracks
    its own commands.

Usage:
  uv run python -m sim.replay_control_loss            # all candidates
  uv run python -m sim.replay_control_loss --episodes 4   # smoke
"""

import argparse
import json
import time

import mujoco
import numpy as np

from . import OUTPUT_DIR
from .so101_sim import JOINTS, SERVO_SYSID, SO101Sim
from .sysid_servo import (
    ARM,
    FIT_EPISODES,
    MENAGERIE,
    UPSTREAM,
    load_episode,
    replay_trajectory,
    set_params,
)

V2_REPO = "so101_pick_place_v2"
# The 26 reference episodes: every so101_pick_place_v2 episode lying
# wholly inside the encoder probe's reference half A
# (assets/real_plates/manifest.json "episodes"; boundary frame 16200).
REFERENCE_EPISODES = tuple(range(26))
# v2 members of sim.sysid_servo.FIT_EPISODES that fall in the
# reference set — excluded from the held-out split.
SYSID_FIT_OVERLAP = tuple(
    sorted(e for r, e in FIT_EPISODES if r == V2_REPO and e in REFERENCE_EPISODES),
)
_FRO_TO_HALF_ANGLE = 2.0 * np.sqrt(2.0)

CANDIDATES = (
    ("menagerie", MENAGERIE),
    ("upstream", UPSTREAM),
    ("pinned_so101_sim", SERVO_SYSID),
)


def ee_trajectory(
    sim: SO101Sim,
    data: mujoco.MjData,
    qpos_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kinematics-only EE pass: joint trajectory [T, 6] (degrees) ->
    (site positions [T, 3] m, site rotations [T, 3, 3]) of the follower
    ``gripperframe`` site. `data` is a scratch MjData for `sim.model`;
    everything outside the follower joints stays at qpos0."""
    site = sim.model.site("gripperframe").id
    pos = np.empty((len(qpos_deg), 3))
    mat = np.empty((len(qpos_deg), 3, 3))
    for t, q in enumerate(qpos_deg):
        data.qpos[sim._joint_qpos] = np.deg2rad(q)
        mujoco.mj_kinematics(sim.model, data)
        pos[t] = data.site_xpos[site]
        mat[t] = data.site_xmat[site].reshape(3, 3)
    return pos, mat


def control_loss_terms(
    pos_a: np.ndarray,
    mat_a: np.ndarray,
    pos_b: np.ndarray,
    mat_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame SIMPLER terms: (||dx|| [T] m, arcsin(||dR||_F / 2sqrt2)
    [T] rad). The Frobenius gap is rotation-invariant on the right, so
    the site's fixed local orientation cancels."""
    trans = np.linalg.norm(pos_a - pos_b, axis=1)
    fro = np.linalg.norm((mat_a - mat_b).reshape(len(mat_a), 9), axis=1)
    rot = np.arcsin(np.clip(fro / _FRO_TO_HALF_ANGLE, 0.0, 1.0))
    return trans, rot


def ee_sensitivity_mm_per_deg(
    sim: SO101Sim,
    data: mujoco.MjData,
    pose_deg: np.ndarray,
) -> dict[str, float]:
    """First-order EE lever arm per joint at `pose_deg`: mm of site
    displacement per degree of joint motion (central difference). Turns
    the per-joint MAEs into EE-space contributions — joint-space MAE
    weights every joint equally, the EE loss does not."""
    out: dict[str, float] = {}
    for j, name in enumerate(JOINTS):
        lo, hi = pose_deg.copy(), pose_deg.copy()
        lo[j] -= 0.5
        hi[j] += 0.5
        pos, _ = ee_trajectory(sim, data, np.stack([lo, hi]))
        out[name] = float(np.linalg.norm(pos[1] - pos[0]) * 1000.0)
    return out


def _pooled(per_episode: list[dict], episodes: tuple[int, ...]) -> dict:
    """Frame-weighted pooled metrics over the selected episode ids."""
    rows = [r for r in per_episode if r["episode"] in episodes]
    frames = np.array([r["frames"] for r in rows], dtype=np.float64)

    def pool(key: str) -> float:
        return float(np.average([r[key] for r in rows], weights=frames))

    trans, rot = pool("trans_m"), pool("rot_rad")
    return {
        "episodes": len(rows),
        "frames": int(frames.sum()),
        "trans_m": trans,
        "rot_rad": rot,
        "control_loss": trans + rot,
        "arm_mae_deg": pool("arm_mae_deg"),
        "per_joint_mae_deg": {
            name: float(
                np.average([r["per_joint_mae_deg"][i] for r in rows], weights=frames),
            )
            for i, name in enumerate(JOINTS)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=len(REFERENCE_EPISODES),
        help="first N reference episodes (smoke runs)",
    )
    args = parser.parse_args()
    episode_ids = REFERENCE_EPISODES[: args.episodes]

    sim = SO101Sim(width=64, height=48, render_style="v0")
    fk_data = mujoco.MjData(sim.model)
    episodes = {e: load_episode(V2_REPO, e) for e in episode_ids}

    # Real EE trajectories and the command floor are param-independent:
    # FK only reads qpos, and set_params touches actuators/dofs.
    real_ee: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    floor_rows: list[dict] = []
    for e, (actions, states) in episodes.items():
        real_ee[e] = ee_trajectory(sim, fk_data, states[1:])
        commanded = np.rad2deg(
            np.clip(np.deg2rad(actions[:-1]), sim._ctrl_low, sim._ctrl_high),
        )
        trans, rot = control_loss_terms(
            *ee_trajectory(sim, fk_data, commanded),
            *real_ee[e],
        )
        floor_rows.append(
            {
                "episode": e,
                "frames": len(trans),
                "trans_m": float(trans.mean()),
                "rot_rad": float(rot.mean()),
                "control_loss": float(trans.mean() + rot.mean()),
                "arm_mae_deg": float(
                    np.abs(commanded[:, ARM] - states[1:, ARM]).mean(),
                ),
                "per_joint_mae_deg": np.abs(commanded - states[1:])
                .mean(axis=0)
                .tolist(),
            },
        )

    heldout = tuple(e for e in episode_ids if e not in SYSID_FIT_OVERLAP)
    median_pose = np.median(
        np.concatenate([s for _, s in episodes.values()]),
        axis=0,
    )
    sensitivity = ee_sensitivity_mm_per_deg(sim, fk_data, median_pose)
    report: dict = {
        "reference_episodes": list(episode_ids),
        "sysid_fit_overlap": list(SYSID_FIT_OVERLAP),
        "loss": "L = mean ||dx|| (m) + mean arcsin(||dR||_F / 2sqrt2) (rad)",
        "simpler_table2_anchors": {"0.131": 0.031, "0.267": 0.070, "0.432": 0.100},
        "median_pose_deg": median_pose.tolist(),
        "ee_sensitivity_mm_per_deg": sensitivity,
        "real_command_floor": {
            "per_episode": floor_rows,
            "pooled_all": _pooled(floor_rows, episode_ids),
            "pooled_heldout": _pooled(floor_rows, heldout),
        },
        "candidates": {},
    }
    print(
        "EE sensitivity at median pose (mm/deg): "
        + ", ".join(f"{k} {v:.2f}" for k, v in sensitivity.items()),
    )
    floor = report["real_command_floor"]["pooled_all"]
    print(
        f"real command->state EE floor: L {floor['control_loss']:.4f} "
        f"(trans {floor['trans_m'] * 1000:.1f} mm, rot {np.rad2deg(floor['rot_rad']):.2f} deg, "
        f"arm MAE {floor['arm_mae_deg']:.2f} deg)",
    )

    for label, params in CANDIDATES:
        t0 = time.time()
        set_params(sim, params)
        rows: list[dict] = []
        for e, (actions, states) in episodes.items():
            sim_states = replay_trajectory(sim, actions, states)
            trans, rot = control_loss_terms(
                *ee_trajectory(sim, fk_data, sim_states),
                *real_ee[e],
            )
            rows.append(
                {
                    "episode": e,
                    "frames": len(trans),
                    "trans_m": float(trans.mean()),
                    "rot_rad": float(rot.mean()),
                    "control_loss": float(trans.mean() + rot.mean()),
                    "arm_mae_deg": float(
                        np.abs(sim_states[:, ARM] - states[1:, ARM]).mean(),
                    ),
                    "per_joint_mae_deg": np.abs(sim_states - states[1:])
                    .mean(axis=0)
                    .tolist(),
                },
            )
        a, h = _pooled(rows, episode_ids), _pooled(rows, heldout)
        entry: dict = {
            "params": params,
            "per_episode": rows,
            "pooled_all": a,
            "pooled_heldout": h,
            "seconds": round(time.time() - t0, 1),
        }
        report["candidates"][label] = entry
        print(
            f"[{label}] L {a['control_loss']:.4f} all / {h['control_loss']:.4f} held-out "
            f"(trans {a['trans_m'] * 1000:.1f} mm, rot {np.rad2deg(a['rot_rad']):.2f} deg, "
            f"arm MAE {a['arm_mae_deg']:.2f} deg, {entry['seconds']}s)",
        )

    # Leave the model as SO101Sim ships it.
    set_params(sim, SERVO_SYSID)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "replay_control_loss.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
