"""Build the seamless tabletop texture from a real rig frame.

Crops a clean-wood patch from a downloaded front frame and mirror-tiles
it 2x2 (patch | flip-x / flip-y | flip-xy): every tile edge then meets a
mirrored copy of itself, so tiling is seamless by construction — wood
grain mirroring reads naturally. Writes assets/table/table_texture.png
(tracked; the source frame lives in outputs/sim/real/, fetched by the
snippet in probe_visual_match's docstring).

Usage: uv run python -m sim.make_table_texture
"""

from pathlib import Path

import numpy as np
from PIL import Image

from . import OUTPUT_DIR

# Clean-wood window in front_00146.png: right of the PCB, below the
# mouse, above the arms (re-measure if the source frame changes).
SOURCE = OUTPUT_DIR / "real" / "front_00146.png"
CROP = (262, 124, 384, 200)
TILE = 256


def main() -> int:
    patch = Image.open(SOURCE).convert("RGB").crop(CROP).resize((TILE, TILE))
    base = np.asarray(patch)
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
