# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trimesh>=4",
#     "xatlas>=0.0.9",
#     "fast-simplification>=0.1",
#     "numpy",
#     "scipy",  # trimesh.convex_hull backend
# ]
# ///
"""One-off: 3DBenchy STL -> MuJoCo-ready OBJs.

Input: the official CreativeTools STL (mm units, 225k faces, no UVs).
Outputs (sim/assets/benchy/):
  - benchy_visual.obj    decimated to ~24k faces, xatlas-generated UVs,
                         meters, origin at the hull-base center (z=0)
  - benchy_collision.obj convex hull (MuJoCo uses convex collision meshes)

Benchy is CC BY-ND 4.0: fine to download and convert for internal use;
don't redistribute the converted mesh.

Usage: uv run sim/convert_benchy.py /tmp/3DBenchy.stl
"""

import sys
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh
import xatlas

TARGET_FACES = 24_000


def main() -> int:
    stl_path = Path(sys.argv[1])
    out_dir = Path(__file__).parent / "assets" / "benchy"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(stl_path)
    print(f"loaded: {len(mesh.faces)} faces, extents {mesh.extents} (mm)")

    # mm -> m, origin at base center so z=0 sits on the table.
    mesh.apply_scale(0.001)
    minx, miny, minz = mesh.bounds[0]
    maxx, maxy, _ = mesh.bounds[1]
    mesh.apply_translation([-(minx + maxx) / 2, -(miny + maxy) / 2, -minz])

    points, faces = fast_simplification.simplify(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
        target_reduction=1.0 - TARGET_FACES / len(mesh.faces),
    )
    visual = trimesh.Trimesh(vertices=points, faces=faces)
    print(f"decimated: {len(visual.faces)} faces, extents {visual.extents} (m)")

    # xatlas remaps vertices (vmapping) and yields per-vertex UVs.
    vmapping, indices, uvs = xatlas.parametrize(visual.vertices, visual.faces)
    visual = trimesh.Trimesh(
        vertices=visual.vertices[vmapping],
        faces=indices,
        visual=trimesh.visual.TextureVisuals(uv=uvs),
        process=False,
    )
    visual.export(out_dir / "benchy_visual.obj")

    hull = mesh.convex_hull
    hull.export(out_dir / "benchy_collision.obj")
    print(
        f"wrote {out_dir}/benchy_visual.obj ({len(visual.faces)} faces, UV'd) "
        f"and benchy_collision.obj ({len(hull.faces)} faces)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
