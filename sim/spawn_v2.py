"""Spawn-v2 sampler — disk and boat both placed randomly (pre-reg
DRAFT posts/2026-08-16-prereg-sim-spawn-v2.md §2; owner steering
2026-08-16 09:16Z).

STANDALONE until the pre-reg finalizes: nothing here is wired into
``SO101Sim.reset`` — the ``spawn_version="v2"`` integration (and the
registered v1 stream-compat guard) lands only once the DRAFT constants
below freeze. Until then this module is the sampler the oracles pin,
sampling from the measured workspace mask the reachability probe
emits (fontaine/scripts/spawn_v2_reachability_probe.py).

Draw order (one RNG, fixed): disk cell, disk jitter, then per-attempt
boat (r, theta), then yaw — the order is part of the protocol (seed ->
placements bit-stable, oracle-pinned).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# DRAFT constants — candidates recorded in the pre-reg, FROZEN at
# finalization (§5). Sources: residual bar = the probe's 1 mm class
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

    def contains(self, x: float, y: float) -> bool:
        return (round(x / self.pitch), round(y / self.pitch)) in self.cells

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


def draw_spawn_v2(mask: WorkspaceMask, rng: np.random.Generator) -> SpawnV2:
    """One episode's placements: disk uniform over W, boat uniform
    over the [R_MIN, R_MAX] annulus around it (area-uniform radius),
    rejection-sampled against W membership and the parked-jaw
    keep-out. REFUSES loudly (RuntimeError) on a degenerate
    configuration instead of stalling: acceptance below ACCEPT_FLOOR
    over the first ACCEPT_PROBE_N attempts, or MAX_DRAWS exhausted."""
    disk = mask.sample(rng)
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
        if mask.contains(*boat) and jaw_clear:
            accepted = boat
            break
        if attempts == ACCEPT_PROBE_N:
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
