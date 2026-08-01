"""One-off: render candidate home poses (top + wrist) to pick one where
the wrist camera overlooks the workspace. Writes outputs/sim/pose<i>_*.png."""

import numpy as np
from PIL import Image

from . import OUTPUT_DIR
from .so101_sim import SO101Sim

OUT = OUTPUT_DIR

CANDIDATES = (
    np.array([0.0, -40.0, 60.0, 15.0, 0.0, 30.0]),
    np.array([0.0, -30.0, 50.0, 0.0, 0.0, 30.0]),
    np.array([0.0, -55.0, 90.0, -20.0, 0.0, 30.0]),
    np.array([0.0, -20.0, 70.0, -35.0, 0.0, 30.0]),
)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    sim = SO101Sim()
    for index, pose in enumerate(CANDIDATES):
        sim.reset(0)
        obs = sim.observe()
        for _ in range(45):
            obs = sim.step(pose)
        Image.fromarray(obs.top).save(OUT / f"pose{index}_top.png")
        Image.fromarray(obs.wrist).save(OUT / f"pose{index}_wrist.png")
        print(f"pose{index}: target={pose} reached={np.round(obs.state, 1)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
