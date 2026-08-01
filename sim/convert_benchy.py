# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trimesh>=4",
#     "xatlas>=0.0.9",
#     "fast-simplification>=0.1",
#     "coacd>=1.0",
#     "numpy",
#     "scipy",  # trimesh volume/hull backend
# ]
# ///
"""One-off: 3DBenchy STL -> MuJoCo-ready OBJs.

Input: the official CreativeTools STL (mm units, 225k faces, no UVs).
Outputs (assets/benchy/, repo root):
  - benchy_visual.obj      decimated to ~24k faces, xatlas UVs, meters,
                           origin at the hull-base center (z=0)
  - benchy_col_NN.obj      CoACD convex decomposition pieces (a single
                           convex hull measured 2.63x the boat's volume -
                           the gripper collided with invisible deck/bow
                           space; the decomposition tracks concavities)
  - benchy_col_assets.xml  generated <mujocoinclude> mesh declarations
  - benchy_col_geoms.xml   generated <mujocoinclude> collision geoms
                           (condim 6 + tors/roll friction + solref from
                           the drift fix; density sized so the pieces sum
                           to the 40 g print)

Benchy is CC BY-ND 4.0: fine to download and convert for internal use;
don't redistribute the converted meshes.

Usage: uv run sim/convert_benchy.py /tmp/3DBenchy.stl
"""

import sys
from pathlib import Path

import coacd
import fast_simplification
import numpy as np
import trimesh
import xatlas

TARGET_FACES = 24_000
BENCHY_MASS_KG = 0.04
# Piece budget: enough to open the deck/bow concavities, few enough to
# keep contact checks trivial.
COACD_THRESHOLD = 0.05
COACD_MAX_HULLS = 16


def main() -> int:
    stl_path = Path(sys.argv[1])
    out_dir = Path(__file__).parents[1] / "assets" / "benchy"
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
    textured = trimesh.Trimesh(
        vertices=visual.vertices[vmapping],
        faces=indices,
        visual=trimesh.visual.TextureVisuals(uv=uvs),
        process=False,
    )
    textured.export(out_dir / "benchy_visual.obj")

    # Convex decomposition for contacts.
    parts = coacd.run_coacd(
        coacd.Mesh(visual.vertices, visual.faces),
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_HULLS,
    )
    pieces = [
        trimesh.Trimesh(vertices=verts, faces=np.asarray(fcs)).convex_hull
        for verts, fcs in parts
    ]
    total_volume = sum(p.volume for p in pieces)
    density = BENCHY_MASS_KG / total_volume
    boat_volume = visual.convex_hull.volume
    print(
        f"coacd: {len(pieces)} pieces, {total_volume * 1e6:.1f} cm3 total "
        f"(single hull {boat_volume * 1e6:.1f} cm3, "
        f"{boat_volume / total_volume:.2f}x looser)",
    )

    asset_lines = ["<mujocoinclude>", "  <asset>"]
    geom_lines = ["<mujocoinclude>"]
    for index, piece in enumerate(pieces):
        name = f"benchy_col_{index:02d}"
        piece.export(out_dir / f"{name}.obj")
        asset_lines.append(f'    <mesh name="{name}" file="../../benchy/{name}.obj"/>')
        geom_lines.append(
            f'  <geom name="{name}" type="mesh" mesh="{name}" '
            f'density="{density:.1f}" condim="6" friction="1 0.05 0.01" '
            f'solref="0.005 1" group="3" rgba="0.5 0.5 0.5 0.3"/>',
        )
    asset_lines += ["  </asset>", "</mujocoinclude>"]
    geom_lines += ["</mujocoinclude>"]
    (out_dir / "benchy_col_assets.xml").write_text("\n".join(asset_lines) + "\n")
    (out_dir / "benchy_col_geoms.xml").write_text("\n".join(geom_lines) + "\n")
    print(
        f"wrote {out_dir}/benchy_visual.obj, {len(pieces)} collision pieces + includes",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
