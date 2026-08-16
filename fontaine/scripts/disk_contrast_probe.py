"""Top-cam disk visibility instrument (owner 17:07Z 2026-08-16: 'the
cylinder somehow does not render in the top camera').

Measures the disk/table luminance ratio in the FINAL top composite via
a segmentation-derived disk mask (median-center pruned — stray edge
pixels otherwise pollute the blob), plus the boundary gradient, at N
reset seeds. The real-footage anchor (measured once from
so101_pick_place_v2 file-000 t=1s, disk at (220,268) r=33):
ratio 1.78, boundary gradient 8.4 — the v1 sim material reads 0.95
(darker than its surround). ``--disk-appearance realcal`` measures the
recalibrated material.

Usage:
  uv run python fontaine/scripts/disk_contrast_probe.py \
      --disk-appearance v1|realcal [--seeds 10000,10002,10010]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from sim.so101_sim import SO101Sim

REAL_ANCHOR = {"ratio": 1.78, "gradient": 8.4, "px": 3405}


def measure(sim: SO101Sim, seed: int) -> dict:
    obs = sim.reset(seed)
    renderer = sim.renderer
    renderer.enable_segmentation_rendering()
    renderer.update_scene(sim.data, camera="top_cam")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    mask = seg[..., 0] == sim._disk_geom_id
    ys, xs = np.where(mask)
    if not len(ys):
        return {"seed": seed, "visible": False}
    cy0, cx0 = np.median(ys), np.median(xs)
    r_est = np.sqrt(mask.sum() / np.pi)
    keep = (ys - cy0) ** 2 + (xs - cx0) ** 2 < (2.5 * r_est) ** 2
    mask = np.zeros_like(mask)
    mask[ys[keep], xs[keep]] = True

    lum = obs.top.astype(float).mean(axis=2)
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = np.where(mask)[0].mean(), np.where(mask)[1].mean()
    rr = np.sqrt(mask.sum() / np.pi)
    ring = (
        ((yy - cy) ** 2 + (xx - cx) ** 2 < (rr + 22) ** 2)
        & ~((yy - cy) ** 2 + (xx - cx) ** 2 < (rr + 6) ** 2)
        & ~mask
    )
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    edge = ((yy - cy) ** 2 + (xx - cx) ** 2 < (rr + 4) ** 2) & ~(
        (yy - cy) ** 2 + (xx - cx) ** 2 < (rr - 4) ** 2
    )
    return {
        "seed": seed,
        "visible": True,
        "px": int(mask.sum()),
        "disk_lum": round(float(lum[mask].mean()), 1),
        "ring_lum": round(float(lum[ring].mean()), 1),
        "ratio": round(float(lum[mask].mean() / lum[ring].mean()), 2),
        "boundary_grad": round(float(grad[edge].mean()), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disk-appearance", default="v1")
    parser.add_argument("--seeds", default="10000,10002,10010")
    args = parser.parse_args()

    from sim.so101_sim import SO101Sim

    sim = SO101Sim(
        spawn_version="v2.1",
        tint_band="mix70",
        disk_appearance=args.disk_appearance,
    )
    rows = [measure(sim, int(x)) for x in args.seeds.split(",")]
    for row in rows:
        print(row)
    ratios = [r["ratio"] for r in rows if r.get("visible")]
    if ratios:
        print(
            f"mean ratio {np.mean(ratios):.2f} vs real anchor "
            f"{REAL_ANCHOR['ratio']} (v1 measured 0.95)",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
