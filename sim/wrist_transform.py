"""``--wrist-transform`` hook — the wrist-transfer screen's treatment
seam (pre-reg posts/2026-08-14-prereg-wrist-transfer-screen.md §1,
stage 0).

Pure ``obs.wrist`` transforms, applied to the observation handed to the
policy and nowhere else: each transform builds a NEW ``SimObservation``
around a new wrist array and never mutates its input, so the sim's own
observation stream (video recording included) stays raw and physics
cannot see the treatment by construction. One transform instance serves
ONE episode — ``freeze`` and the W3 coverage trace are per-episode
state, so drivers construct a fresh one per (seed, draw) unit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .so101_sim import SimObservation

WRIST_TRANSFORMS = ("none", "blackout", "freeze", "arm_blur")

# W3 "strong" blur (design memo §3): sigma in px at the 640x480 wrist
# frame — wide enough that shading structure and specular texture die
# inside the arm mask while silhouette and mean color survive. Pinned
# at stage 0; the honesty placement (GPU-gated in the run item) scores
# this exact constant on the knn5 axis, and any change before stage 1
# rides the pre-reg's amendment policy.
ARM_BLUR_SIGMA = 8.0

WristTransform = Callable[[SimObservation], SimObservation]


def make_wrist_transform(name: str, sim: Any) -> WristTransform | None:
    """A fresh per-episode transform, or None for ``none`` — the
    untouched pipeline must stay bit-identical to a driver without the
    hook, so ``none`` routes around the hook entirely rather than
    through an identity call."""
    if name == "none":
        return None
    if name == "blackout":
        return _Blackout()
    if name == "freeze":
        return _Freeze()
    if name == "arm_blur":
        return ArmBlurTransform(sim)
    raise ValueError(f"wrist transform {name!r} not in {WRIST_TRANSFORMS}")


class _Blackout:
    """W1: zeros — the does-it-listen bracket endpoint."""

    def __call__(self, obs: SimObservation) -> SimObservation:
        return SimObservation(
            top=obs.top,
            wrist=np.zeros_like(obs.wrist),
            state=obs.state,
        )


TOP_TRANSFORMS = ("none", "blackout")


def make_top_transform(name: str) -> WristTransform | None:
    """The T1 positive-control seam (pre-reg §1 arm grid): the same
    obs→obs contract on ``obs.top``. ``none`` routes around the hook
    like the wrist factory."""
    if name == "none":
        return None
    if name == "blackout":
        return _TopBlackout()
    raise ValueError(f"top transform {name!r} not in {TOP_TRANSFORMS}")


class _TopBlackout:
    """T1: zeros on the top view — the harness-sensitivity positive
    control (a policy that consumes the top view must move)."""

    def __call__(self, obs: SimObservation) -> SimObservation:
        return SimObservation(
            top=np.zeros_like(obs.top),
            wrist=obs.wrist,
            state=obs.state,
        )


def chain_transforms(
    *transforms: WristTransform | None,
) -> WristTransform | None:
    """Compose the per-view hooks into the loop's single transform
    slot, skipping Nones; all-None returns None so the untouched
    pipeline stays hook-free."""
    live = [t for t in transforms if t is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def chained(obs: SimObservation) -> SimObservation:
        for transform in live:
            obs = transform(obs)
        return obs

    return chained


class _Freeze:
    """W2: every policy tick sees the FIRST frame this instance saw —
    the reset frame, when the driver builds one instance per episode."""

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None

    def __call__(self, obs: SimObservation) -> SimObservation:
        if self._frame is None:
            self._frame = obs.wrist.copy()
        return SimObservation(top=obs.top, wrist=self._frame, state=obs.state)


class ArmBlurTransform:
    """W3: strong Gaussian blur inside the arm+gripper mask only.

    The mask is the sim's per-tick wrist segmentation
    (``sim.wrist_arm_mask()``, [H, W] float 0..1 in the final wrist
    frame). The blur is mask-normalized — ``blur(mask*frame) /
    blur(mask)`` — so corrupted pixels are rebuilt from ARM content
    only and background never bleeds into the arm; outside the mask the
    frame is untouched, so the silhouette survives from both sides.
    Per-tick mask coverage (mask mean) is recorded on ``.coverage`` —
    the run's logged W3 diagnostic."""

    def __init__(self, sim: Any, sigma: float = ARM_BLUR_SIGMA) -> None:
        if not hasattr(sim, "wrist_arm_mask"):
            raise TypeError(
                "arm_blur needs a sim exposing wrist_arm_mask() "
                f"(got {type(sim).__name__})",
            )
        self._sim = sim
        self._sigma = sigma
        self.coverage: list[float] = []

    def __call__(self, obs: SimObservation) -> SimObservation:
        mask = np.asarray(self._sim.wrist_arm_mask(), dtype=np.float64)
        self.coverage.append(float(mask.mean()))
        soft = mask[..., None]
        frame = obs.wrist.astype(np.float64)
        weight = gaussian_blur(soft, self._sigma)
        inside = gaussian_blur(frame * soft, self._sigma) / np.maximum(weight, 1e-9)
        out = soft * inside + (1.0 - soft) * frame
        return SimObservation(
            top=obs.top,
            wrist=np.clip(np.rint(out), 0, 255).astype(np.uint8),
            state=obs.state,
        )


def print_coverage(transform: object, seed: int, draw: int) -> None:
    """The run's logged W3 diagnostic (design memo §7): per-tick mask
    coverage for one finished episode. No-op for every other arm."""
    coverage = getattr(transform, "coverage", None)
    if coverage:
        per_tick = " ".join(f"{value:.3f}" for value in coverage)
        print(
            f"  seed {seed} draw {draw} arm_blur coverage: "
            f"mean {float(np.mean(coverage)):.3f} "
            f"min {min(coverage):.3f} | {per_tick}",
            flush=True,
        )


def gaussian_blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian, [H, W, C] float in/out — same kernel
    convention as SO101Sim's sensor PSF (2.5-sigma radius, edge pad)."""
    radius = max(1, int(np.ceil(2.5 * sigma)))
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    for axis in (0, 1):
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(frame, pad, mode="edge")
        out = np.zeros_like(frame)
        index: list[Any] = [slice(None)] * frame.ndim
        for offset, weight in enumerate(kernel):
            index[axis] = slice(offset, offset + frame.shape[axis])
            out += weight * padded[tuple(index)]
        frame = out
    return frame
