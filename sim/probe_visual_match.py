"""Visual-matching iteration aid: real rig frame | sim render, side by
side, for the front(top) camera. Writes outputs/sim/match_front.png.

Usage: MUJOCO_GL=egl uv run python -m sim.probe_visual_match
"""

import numpy as np
from PIL import Image

from . import OUTPUT_DIR
from .so101_sim import SO101Sim

REAL_FRAME = OUTPUT_DIR / "real" / "front_00146.png"


def main() -> int:
    sim = SO101Sim()
    obs = sim.reset(0)
    real = np.asarray(Image.open(REAL_FRAME).convert("RGB"))
    side = np.concatenate([real, obs.top], axis=1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(side).save(OUTPUT_DIR / "match_front.png")
    print(f"wrote {OUTPUT_DIR / 'match_front.png'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
