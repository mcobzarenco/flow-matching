"""Build the seamless tabletop texture from a real rig frame.

Crops a clean-wood patch from a pinned real top frame and mirror-tiles
it 2x2 (patch | flip-x / flip-y | flip-xy): every tile edge then meets a
mirrored copy of itself, so tiling is seamless by construction — wood
grain mirroring reads naturally. The crop is rotated 90 deg before
tiling so plank boundary lines run along the texture s-axis, which the
table box maps to world +x — the direction the real planks run (they
read vertical in the top camera, whose image-up is +x). Writes
assets/table/table_texture.png (tracked).

Scale: the real planks measure ~65-70 px against the 8 cm disk's
~76 px in the same image rows => ~7 cm plank width; the 2x2 tile then
spans ~10 planks = 0.7 m along y and ~0.34 m of grain along x, which
sets the scene's texrepeat (2.4/0.34, 1.0/0.7) ~= (7.0, 1.43).

Usage: uv run python -m sim.make_table_texture
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Clean-wood window in the pinned OOD-probe real_v2 top frame 0140
# (empty mid-table, level daylight): right of the left-edge background,
# above the disk row, left of the laptop (re-measure if the source
# frame changes). ~5 planks across.
SOURCE = Path("outputs/sim/ood_probe_frames/real_v2/top/0140.png")
CROP = (130, 40, 450, 200)
TILE = 256


def main() -> int:
    patch = Image.open(SOURCE).convert("RGB").crop(CROP)
    base = np.asarray(patch.resize((TILE, TILE // 2))).astype(np.float64)
    # The crop's contrast reads ~30% hot once lit in-scene (the render
    # adds its own shading); compress toward the mean to match the real
    # frames' measured table std (~14 vs ~20 unscaled).
    base = np.clip(base.mean((0, 1)) + 0.72 * (base - base.mean((0, 1))), 0, 255)
    base = base.astype(np.uint8)
    base = np.rot90(base)  # planks: vertical in crop -> lines along s
    row_a = np.concatenate([base, base[:, ::-1]], axis=1)
    row_b = np.concatenate([base[::-1], base[::-1, ::-1]], axis=1)
    seamless = np.concatenate([row_a, row_b], axis=0)
    out = Path(__file__).parents[1] / "assets" / "table" / "table_texture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(seamless).save(out)
    print(f"wrote {out} ({seamless.shape[1]}x{seamless.shape[0]}, mirror-tiled)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
