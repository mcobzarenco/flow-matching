"""Contact-physics findings probe for the benchy pick-place scene
(sim-review, owner-flagged "boat contact physics is poor").

Findings-first instrument: quantifies the reset/settle behavior, rest
drift, the jaw pinch seam (penetration, batting, hold, torsional slip)
and determinism, and times physics+render for the 100-seed eval budget.
Prints one labeled block per probe; saves pinch-test frames to
outputs/sim/probe_pinch_*.png.

Usage: MUJOCO_GL=egl uv run python -m sim.probe_benchy_contact
"""

import time

import mujoco
import numpy as np
from PIL import Image

from . import OUTPUT_DIR
from .so101_sim import HOME_DEGREES, SO101Sim

# Arm pose of menagerie's scene_box.xml "pickup" keyframe (radians,
# gripper open); the grasp point sits ~0.22 m forward of the base.
PICKUP_QPOS = np.array([0.0, 0.000382, 0.4735, 1.17717, 1.58437, 0.727663])


def benchy_xy_yaw(sim: SO101Sim) -> tuple[np.ndarray, float]:
    adr = sim._benchy_qpos
    xy = sim.data.qpos[adr : adr + 2].copy()
    w, x, y, z = sim.data.qpos[adr + 3 : adr + 7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return xy, float(np.degrees(yaw))


def gripper_benchy_contacts(sim: SO101Sim) -> list[float]:
    """Contact distances (negative = penetration, meters) between any
    benchy collision piece and any gripper-class geom this step."""
    model, data = sim.model, sim.data
    dists = []
    for index in range(data.ncon):
        contact = data.contact[index]
        names = {
            model.geom(contact.geom1).name,
            model.geom(contact.geom2).name,
        }
        benchy = any(n.startswith("benchy_col_") for n in names)
        jaw = any("jaw" in n or "gripper" in n for n in names)
        if benchy and jaw:
            dists.append(float(contact.dist))
    return dists


def probe_settle() -> None:
    print("== P1 settle: reset() end-state vs HOME, +extra settle time ==")
    sim = SO101Sim()
    for seed in (0, 1, 2, 3):
        obs = sim.reset(seed)
        err0 = obs.state - HOME_DEGREES
        # Keep driving to home: does the error decay (unsettled) or
        # persist (steady-state offset)?
        for _ in range(90):  # +3 s
            obs = sim.step(HOME_DEGREES)
        err3 = obs.state - HOME_DEGREES
        print(
            f" seed {seed}: |err| at reset-end "
            f"{np.round(np.abs(err0), 1)} -> +3 s {np.round(np.abs(err3), 1)} deg",
        )


def probe_reset_strike() -> None:
    print("== P2 reset strike: does the arm hit the boat during reset()? ==")
    sim = SO101Sim()
    struck = 0
    moved = []
    seeds = range(100)  # the candidate 100-seed eval list
    for seed in seeds:
        sim.reset(seed)
        hit = sim.reset_strike_contacts > 0
        struck += hit
        xy, _ = benchy_xy_yaw(sim)
        x, y = sim.reset_spawn_xy
        moved.append(np.hypot(xy[0] - x, xy[1] - y) * 1000)
        if hit:
            print(
                f" seed {seed}: ARM-BOAT CONTACT during reset "
                f"({sim.reset_strike_contacts} contact-steps)",
            )
    moved_arr = np.array(moved)
    print(
        f" {struck}/{len(list(seeds))} seeds strike the boat; spawn->settled "
        f"displacement mm: median {np.median(moved_arr):.1f} "
        f"max {moved_arr.max():.1f}",
    )


def probe_rest_drift() -> None:
    print("== P3 rest drift: 10 s untouched after a full settle ==")
    sim = SO101Sim()
    sim.reset(0)
    for _ in range(90):  # settle a further 3 s first
        sim.step(HOME_DEGREES)
    xy0, yaw0 = benchy_xy_yaw(sim)
    for _ in range(300):  # 10 s
        sim.step(HOME_DEGREES)
    xy1, yaw1 = benchy_xy_yaw(sim)
    print(
        f" drift {np.hypot(*(xy1 - xy0)) * 1000:.3f} mm, "
        f"spin {abs(yaw1 - yaw0):.3f} deg",
    )


def probe_pinch() -> None:
    print("== P4 pinch: boat placed between the jaws, close, lift ==")
    sim = SO101Sim()
    sim.reset(0)
    model, data = sim.model, sim.data

    # Teleport the arm into the pickup pose (gripper open) and hold it.
    data.qpos[sim._joint_qpos] = PICKUP_QPOS
    data.qvel[:] = 0.0
    data.ctrl[sim._actuator_ids] = PICKUP_QPOS
    mujoco.mj_forward(model, data)

    # Grasp point: midpoint of the jaw pad geoms; jaw axis from their
    # separation. The boat's long axis goes perpendicular to it.
    fixed = data.geom("fixed_jaw_box1").xpos
    moving = data.geom("moving_jaw_box1").xpos
    grasp = (fixed + moving) / 2
    jaw_axis = np.arctan2(moving[1] - fixed[1], moving[0] - fixed[0])
    gap = np.linalg.norm(moving - fixed)
    print(
        f" grasp point {np.round(grasp, 3)}, pad gap {gap * 100:.1f} cm "
        f"(boat beam 3.1 cm)",
    )

    adr = sim._benchy_qpos
    yaw = jaw_axis + np.pi / 2
    data.qpos[adr : adr + 3] = (grasp[0], grasp[1], 0.001)
    data.qpos[adr + 3 : adr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def snap(tag: str) -> None:
        obs = sim.observe()
        Image.fromarray(np.concatenate([obs.top, obs.wrist], axis=1)).save(
            OUTPUT_DIR / f"probe_pinch_{tag}.png",
        )

    def run(ticks: int, target_degrees: np.ndarray) -> tuple[float, float]:
        """Step with a fixed target; return (max penetration mm, max
        boat speed m/s) over the window."""
        pen, speed = 0.0, 0.0
        vadr = model.joint("benchy_free").dofadr[0]
        for _ in range(ticks):
            sim.step(target_degrees)
            dists = gripper_benchy_contacts(sim)
            if dists:
                pen = max(pen, -min(dists))
            speed = max(speed, float(np.linalg.norm(data.qvel[vadr : vadr + 3])))
        return pen * 1000, speed

    hold_open = np.rad2deg(PICKUP_QPOS)
    pen, speed = run(15, hold_open)  # 0.5 s open-jaw settle
    print(f" open settle : max pen {pen:6.2f} mm, max boat speed {speed:.3f} m/s")
    snap("open")

    closed = hold_open.copy()
    closed[5] = 0.0  # close the gripper
    pos_before, _ = sim.benchy_pose()
    _, yaw_before = benchy_xy_yaw(sim)
    pen, speed = run(60, closed)  # 2 s closing + hold
    pos_after, upright = sim.benchy_pose()
    print(
        f" close+hold : max pen {pen:6.2f} mm, max boat speed {speed:.3f} m/s, "
        f"boat moved {np.linalg.norm(pos_after - pos_before) * 1000:.1f} mm, "
        f"upright {upright:.2f}",
    )
    snap("closed")

    lifted = closed.copy()
    lifted[1] -= 25.0  # raise the shoulder
    z_before = pos_after[2]
    pen, speed = run(60, lifted)  # 2 s lift + hold
    pos_lift, upright = sim.benchy_pose()
    _, yaw_after = benchy_xy_yaw(sim)
    dz = (pos_lift[2] - z_before) * 100
    spin = abs((yaw_after - yaw_before + 180) % 360 - 180)
    held = dz > 1.0
    print(
        f" lift+hold  : max pen {pen:6.2f} mm, max boat speed {speed:.3f} m/s, "
        f"boat rose {dz:.1f} cm ({'HELD' if held else 'DROPPED/SLIPPED'}), "
        f"in-grip spin {spin:.1f} deg, upright {upright:.2f}",
    )
    snap("lifted")


def probe_determinism() -> None:
    print("== P5 determinism: same seed, same actions, twice ==")
    actions = np.stack(
        [
            HOME_DEGREES + np.array([15, 20, -20, -10, 0, 30]) * np.sin(k / 10)
            for k in range(60)
        ],
    )

    def rollout() -> tuple[np.ndarray, np.ndarray]:
        sim = SO101Sim()
        obs = sim.reset(7)
        for action in actions:
            obs = sim.step(action)
        return sim.data.qpos.copy(), obs.top

    qpos_a, top_a = rollout()
    qpos_b, top_b = rollout()
    print(
        f" qpos bit-identical: {bool(np.array_equal(qpos_a, qpos_b))}; "
        f"render bit-identical: {bool(np.array_equal(top_a, top_b))}",
    )


def probe_perf() -> None:
    print("== P6 perf: physics + render throughput (100-seed budget) ==")
    sim = SO101Sim()
    sim.reset(0)
    start = time.perf_counter()
    for _ in range(300):
        sim.step(HOME_DEGREES)  # includes 2 camera renders per tick
    per_tick = (time.perf_counter() - start) / 300
    ticks = 15 * 30  # rollout_sim defaults: 15 replans x 30-tick horizon
    print(
        f" {per_tick * 1000:.1f} ms/control-tick (physics + 2 renders) -> "
        f"{per_tick * ticks:.0f} s/episode sim-side; 100 seeds ~ "
        f"{per_tick * ticks * 100 / 60:.0f} min + policy inference",
    )


def main() -> int:
    probe_settle()
    probe_reset_strike()
    probe_rest_drift()
    probe_pinch()
    probe_determinism()
    probe_perf()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
