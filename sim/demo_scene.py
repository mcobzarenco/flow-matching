"""Prototype demo: render seeded resets from both cameras, run a scripted
arm sweep, and sanity-check the success predicate by teleporting benchy
onto the disk. Writes PNGs to outputs/sim/.

Usage: MUJOCO_GL=egl uv run python -m sim.demo_scene
"""

import mujoco
import numpy as np
from PIL import Image

from . import OUTPUT_DIR
from .so101_sim import DISK_CENTER, SO101Sim

OUT = OUTPUT_DIR


def save(name: str, image: np.ndarray) -> None:
    Image.fromarray(image).save(OUT / f"{name}.png")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    sim = SO101Sim()

    # Three seeded resets: benchy pose + color should differ, arm homed.
    for seed in (0, 1, 2):
        obs = sim.reset(seed)
        save(f"reset{seed}_top", obs.top)
        save(f"reset{seed}_wrist", obs.wrist)
        print(f"reset {seed}: state (deg) = {np.round(obs.state, 1)}")

    # Scripted sweep: lean toward the workspace, close the gripper.
    obs = sim.reset(0)
    sweep = np.array([20.0, -25.0, 25.0, -40.0, 0.0, 20.0])
    for tick in range(45):  # 1.5 s
        obs = sim.step(sweep * min(1.0, tick / 30))
    save("sweep_top", obs.top)
    save("sweep_wrist", obs.wrist)
    print(f"after sweep: state (deg) = {np.round(obs.state, 1)}")
    print(f"success (should be False): {sim.success()}")

    # Teleport benchy onto the disk -> predicate must flip to True.
    adr = sim._benchy_qpos  # prototype demo pokes internals
    sim.data.qpos[adr : adr + 3] = (*DISK_CENTER, 0.005)
    sim.data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(30):
        sim.step(np.zeros(6))
    save("ondisk_top", sim.observe().top)
    pos, upright = sim.benchy_pose()
    print(f"teleported onto disk: pos={np.round(pos, 3)} upright={upright:.2f}")
    print(f"success (should be True): {sim.success()}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
