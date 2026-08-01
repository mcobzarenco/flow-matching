"""Interactive MuJoCo viewer for the pick-place scene.

Opens the standard MuJoCo GUI (drag to orbit, double-click to select,
ctrl+drag to perturb bodies; the Control tab drives the six actuators).
The env is reset first, so you see exactly what SO101Sim.reset produces:
homed arm, seeded benchy pose + color.

Usage: uv run python -m sim.view [--seed N]
"""

import argparse

import mujoco.viewer

from .so101_sim import SO101Sim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sim = SO101Sim()
    sim.reset(args.seed)
    print("launching viewer (close the window to exit)")
    mujoco.viewer.launch(sim.model, sim.data)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
