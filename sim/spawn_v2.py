"""Spawn-v2 sampler — disk and boat both placed randomly (pre-reg
posts/2026-08-16-prereg-sim-spawn-v2.md §2; owner steering 2026-08-16
09:16Z).

FINALIZED 2026-08-16: the §5 table below is frozen at its proposed
(measured) values — owner approved the v1-dataset protocol built on it
12:21:03Z ("agree with v1 with just the boat upright in the annulus")
and the registered objection window ("flag it before the box lands")
closed with the A100 box landing 12:25:56Z, no objections. The
workspace mask W is committed as ``sim/spawn_v2_mask.json`` (977
cells, the §3.1 v1 instrument read); ``SO101Sim(spawn_version="v2")``
consumes this sampler, and ``spawn_version="v1"`` reproduces the v1
draw order bit-identically (oracle-guarded).

Draw order (one RNG, fixed): disk cell, disk jitter, then per-attempt
boat (r, theta), then yaw — the order is part of the protocol (seed ->
placements bit-stable, oracle-pinned).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# FROZEN constants (pre-reg §5, finalized 2026-08-16 — see module
# docstring). Sources: residual bar = the probe's 1 mm class
# (425/2196 cells); moment bound = backstop at 2x the measured max
# 0.25; r_min = disk radius 0.04 + hull half-length 0.03 + 0.01
# margin (the v1 band's measured clearance); r_max caps the traverse
# inside the scripted expert's phase clock (v1 mean start distance
# ~9.5 cm; drafted at 2x); jaw keep-out reproduces the v1 near-bound
# arithmetic (parked jaw tips x~0.155: hull 0.03 + 0.01 margin).
RESIDUAL_BAR = 1e-3
MOMENT_BOUND = 0.5
R_MIN = 0.08
R_MAX = 0.19
JAW_TIP_XY = (0.155, 0.0)
JAW_KEEPOUT = 0.04
# Refusal bar: zero accepts across this window means acceptance is
# below ~1/N with overwhelming odds — a degenerate mask/annulus
# combination, not bad luck (sane configs measure >50%). The exact
# floor semantics freeze at finalization with the other constants.
ACCEPT_PROBE_N = 200
MAX_DRAWS = 10_000

# v2.1 amendment (2026-08-16, registered in prereg §7): the v2 mask
# was cut with a broken torque instrument — the reachability probe
# reported static shoulder moment <= 0.25 of forcerange across W, but
# direct actuator-force measurement (spawn_v2_hold_probe) shows the
# sysid'd shoulder-lift servo SATURATED (fraction 1.00) holding
# extended poses, sagging up to ~2 cm at r_base 0.36; the live expert
# measures 48% success at boat r_base < 0.26 falling to ~1% beyond
# 0.34 (600-seed field, spawn_v2_expert_probe). v2.1 constrains both
# placements to the MEASURED competence bands (68.2% expert success on
# the joint band, n=110) while keeping the annulus geometry, full
# yaw, and uniform-in-W draws.
BOAT_R_BASE = (0.16, 0.27)
DISK_R_BASE = (0.18, 0.32)


MASK_PATH = Path(__file__).parent / "spawn_v2_mask.json"


@dataclass(frozen=True)
class WorkspaceMask:
    """The measured workspace W as an explicit cell set — inspectable
    data (the pre-reg's requirement), not a runtime IK call."""

    pitch: float
    cells: frozenset[tuple[int, int]]

    @classmethod
    def from_probe(
        cls,
        probe_json: Path,
        *,
        residual_bar: float = RESIDUAL_BAR,
        moment_bound: float = MOMENT_BOUND,
    ) -> WorkspaceMask:
        data = json.loads(Path(probe_json).read_text())
        pitch = float(data["params"]["pitch"])
        cells = frozenset(
            (round(c["x"] / pitch), round(c["y"] / pitch))
            for c in data["cells"]
            if c["residual"] < residual_bar
            and max(c["moment_frac_shoulder"], c["moment_frac_elbow"]) < moment_bound
        )
        if not cells:
            raise ValueError(f"empty workspace mask from {probe_json}")
        return cls(pitch=pitch, cells=cells)

    @classmethod
    def frozen(cls) -> WorkspaceMask:
        """The FINALIZED workspace W — the committed 977-cell asset,
        already cleaned (never re-clean it: the one-pass rule). The
        cell count is pinned so a drifted asset refuses instead of
        silently changing the protocol."""
        data = json.loads(MASK_PATH.read_text())
        mask = cls(
            pitch=float(data["pitch"]),
            cells=frozenset((int(i), int(j)) for i, j in data["cells"]),
        )
        if len(mask.cells) != int(data["n_cells"]) or len(mask.cells) != 977:
            raise ValueError(
                f"frozen spawn-v2 mask drifted: {len(mask.cells)} cells "
                f"(asset says {data['n_cells']}, protocol froze 977)",
            )
        return mask

    def contains(self, x: float, y: float) -> bool:
        return (round(x / self.pitch), round(y / self.pitch)) in self.cells

    def cleaned(self, *, min_neighbors: int = 5) -> WorkspaceMask:
        """Morphological clean (pre-reg §3.1's v0-caveat fix): drop
        cells with fewer than ``min_neighbors`` of their 8 neighbors in
        W (speckle and ragged edge — the marginal-convergence cells
        whose annulus is mostly rejected, the measured 194/200 tail),
        then keep only the largest 4-connected component so W is one
        solid region. ONE filter pass, deliberately — iterating to
        fixpoint erodes any finite region to nothing (every pass
        manufactures fresh sub-threshold corners)."""
        cells = {
            (i, j)
            for (i, j) in self.cells
            if sum(
                (i + di, j + dj) in self.cells
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)
            )
            >= min_neighbors
        }
        if not cells:
            raise ValueError("workspace mask empty after morphological clean")
        components: list[set[tuple[int, int]]] = []
        todo = set(cells)
        while todo:
            frontier = [todo.pop()]
            comp = set(frontier)
            while frontier:
                i, j = frontier.pop()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (i + di, j + dj)
                    if n in todo:
                        todo.remove(n)
                        comp.add(n)
                        frontier.append(n)
            components.append(comp)
        return WorkspaceMask(
            pitch=self.pitch,
            cells=frozenset(max(components, key=len)),
        )

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        """Uniform over W: a uniform cell, then uniform jitter inside
        it (jitter stays within the drawn cell, so the point is in W
        by construction)."""
        ordered = sorted(self.cells)
        i, j = ordered[int(rng.integers(len(ordered)))]
        half = self.pitch / 2
        return (
            i * self.pitch + float(rng.uniform(-half, half)),
            j * self.pitch + float(rng.uniform(-half, half)),
        )


@dataclass(frozen=True)
class SpawnV2:
    disk_xy: tuple[float, float]
    boat_xy: tuple[float, float]
    boat_yaw: float
    draws: int  # boat attempts spent — the acceptance telemetry


def draw_spawn_v2(
    mask: WorkspaceMask,
    rng: np.random.Generator,
    *,
    radial_bands: bool = False,
) -> SpawnV2:
    """One episode's placements: disk uniform over W, boat uniform
    over the [R_MIN, R_MAX] annulus around it (area-uniform radius),
    rejection-sampled against W membership and the parked-jaw
    keep-out. ``radial_bands=True`` is the v2.1 amendment: disk and
    boat additionally rejection-sampled into the measured competence
    bands DISK_R_BASE / BOAT_R_BASE (base-relative radius). REFUSES
    loudly (RuntimeError) on a degenerate configuration instead of
    stalling: zero acceptance over the first ACCEPT_PROBE_N attempts,
    or MAX_DRAWS exhausted."""
    disk: tuple[float, float] | None = None
    for _ in range(ACCEPT_PROBE_N):
        candidate = mask.sample(rng)
        if not radial_bands or (
            DISK_R_BASE[0] <= float(np.hypot(*candidate)) <= DISK_R_BASE[1]
        ):
            disk = candidate
            break
    if disk is None:
        raise RuntimeError(
            f"spawn-v2 disk draw degenerate: 0/{ACCEPT_PROBE_N} in the "
            f"radial band {DISK_R_BASE} — mask/band constants inconsistent",
        )
    accepted: tuple[float, float] | None = None
    attempts = 0
    while attempts < MAX_DRAWS:
        attempts += 1
        r = float(np.sqrt(rng.uniform(R_MIN**2, R_MAX**2)))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        boat = (disk[0] + r * np.cos(theta), disk[1] + r * np.sin(theta))
        jaw_clear = (
            np.hypot(boat[0] - JAW_TIP_XY[0], boat[1] - JAW_TIP_XY[1]) >= JAW_KEEPOUT
        )
        band_ok = not radial_bands or (
            BOAT_R_BASE[0] <= float(np.hypot(*boat)) <= BOAT_R_BASE[1]
        )
        if mask.contains(*boat) and jaw_clear and band_ok:
            accepted = boat
            break
        if attempts % ACCEPT_PROBE_N == 0 and accepted is None and radial_bands:
            # Disks near the outer disk band + tight boat band can
            # have thin acceptance; re-draw the DISK rather than
            # refusing outright (bounded by MAX_DRAWS overall).
            disk = None
            for _ in range(ACCEPT_PROBE_N):
                candidate = mask.sample(rng)
                if DISK_R_BASE[0] <= float(np.hypot(*candidate)) <= DISK_R_BASE[1]:
                    disk = candidate
                    break
            if disk is None:
                raise RuntimeError("spawn-v2.1 disk re-draw degenerate")
        elif attempts == ACCEPT_PROBE_N and accepted is None:
            raise RuntimeError(
                f"spawn-v2 acceptance degenerate: 0/{ACCEPT_PROBE_N} boat "
                f"draws accepted around disk {disk} — mask/annulus "
                "constants are inconsistent (probe the mask, do not retry)",
            )
    if accepted is None:
        raise RuntimeError(
            f"spawn-v2 exhausted {MAX_DRAWS} boat draws around disk {disk}",
        )
    yaw = float(rng.uniform(-np.pi, np.pi))
    return SpawnV2(disk_xy=disk, boat_xy=accepted, boat_yaw=yaw, draws=attempts)
