# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trimesh>=4",
#     "rtree",
#     "numpy",
#     "scipy",
# ]
# ///
"""Phantom collision volume probe (sim-review): how far outside the
visible boat do the CoACD collision hulls extend? The gripper "touches"
at the collision surface, so the margin is exactly how early contacts
fire vs what the cameras show.

Reports hull-vs-visual volumes and the signed-distance distribution of
collision-surface sample points to the visual mesh (positive = outside
the boat = phantom).

Usage: uv run sim/probe_phantom_volume.py
"""

from pathlib import Path

import numpy as np
import trimesh

ASSETS = Path(__file__).parents[1] / "assets" / "benchy"
SAMPLES_PER_PIECE = 400


def main() -> int:
    visual = trimesh.load(ASSETS / "benchy_visual.obj", force="mesh")
    pieces = sorted(ASSETS.glob("benchy_col_*.obj"))
    hulls = [trimesh.load(p, force="mesh") for p in pieces]

    # The decimated visual mesh is not watertight, but its enclosed
    # volume still lands on the official benchy displacement (~15.6 cm3)
    # - usable as the denominator with that caveat.
    visual_volume = abs(visual.volume)
    print(
        f"visual mesh: {len(visual.faces)} faces, watertight "
        f"{visual.is_watertight}, volume {visual_volume * 1e6:.1f} cm3",
    )
    total = sum(h.volume for h in hulls)
    print(
        f"{len(hulls)} collision hulls: {total * 1e6:.1f} cm3 total = "
        f"{total / visual_volume:.2f}x the visual volume",
    )

    query = trimesh.proximity.ProximityQuery(visual)
    margins = []
    for hull in hulls:
        points, _ = trimesh.sample.sample_surface(hull, SAMPLES_PER_PIECE)
        # signed_distance: positive INSIDE the mesh -> negate for
        # "phantom margin outside the boat".
        margins.append(-query.signed_distance(points))
    margin = np.concatenate(margins) * 1000  # mm
    outside = margin > 0.05  # beyond mesh-tolerance noise
    print(
        f"collision-surface phantom margin (mm): median "
        f"{np.median(margin):.2f}, p90 {np.percentile(margin, 90):.2f}, "
        f"p99 {np.percentile(margin, 99):.2f}, max {margin.max():.2f}; "
        f"{outside.mean() * 100:.0f}% of the collision surface sits "
        f"outside the visible boat",
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
