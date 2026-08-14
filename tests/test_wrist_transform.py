"""Stage-0 oracles for the ``--wrist-transform`` hook (wrist-transfer
screen pre-reg posts/2026-08-14-prereg-wrist-transfer-screen.md §1):
golden frame per transform, per-episode freeze semantics, and the
transform-purity contract — transforms touch pixels, never state.

CPU-tier by design (no GL): the pixel math runs on deterministic
synthetic frames and a synthetic soft mask; the loop-level purity
oracle rides the same FakeSim the parallel-harness oracle pins. The
two GPU-adjacent stage-0 oracles (a ``none`` rollout bit-replaying a
banked seed; the real-sim W3 mask on banked pose slots) live in the
run item, gated on the GPU release — the real-render mask path is
spot-checked by fontaine/scripts/wrist_transform_spotcheck.py.
"""

from dataclasses import asdict
from hashlib import md5

import numpy as np
import pytest
from test_sim_parallel_rollouts import HORIZON, REPLANS, FakeSim, fake_chunk

from sim.rollout_sim import EpisodeResult, run_episode_loop
from sim.so101_sim import SimObservation
from sim.wrist_transform import (
    ARM_BLUR_SIGMA,
    WRIST_TRANSFORMS,
    ArmBlurTransform,
    WristTransform,
    gaussian_blur,
    make_wrist_transform,
)


def _built(name: str, sim: object = None) -> WristTransform:
    """The factory for a treatment arm, narrowed non-None (only
    ``none`` returns None)."""
    transform = make_wrist_transform(name, sim)
    assert transform is not None
    return transform


def synthetic_obs(tick: int = 0) -> SimObservation:
    """Deterministic 480x640 observation with real-frame-like content
    (smooth gradients + seeded texture — blur visibly changes it)."""
    rng = np.random.default_rng(20260814 + tick)
    v, u = np.mgrid[0:480, 0:640].astype(np.float64)
    base = np.stack(
        [
            120 + 80 * np.sin(u / 40.0) + 30 * (v / 480.0),
            90 + 60 * np.cos(v / 25.0),
            60 + 40 * np.sin((u + v) / 60.0),
        ],
        axis=-1,
    )
    frame = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)
    return SimObservation(
        top=frame[::2, ::2].copy(),
        wrist=frame,
        state=np.linspace(-30.0, 30.0, 6) + tick,
    )


def synthetic_mask() -> np.ndarray:
    """Soft-edged blob covering ~1/4 of the frame — the shape class the
    remapped segmentation produces (interior 1.0, bilinear skirt)."""
    v, u = np.mgrid[0:480, 0:640].astype(np.float64)
    r = np.hypot((u - 400.0) / 220.0, (v - 300.0) / 160.0)
    return np.clip(2.0 - 2.0 * r, 0.0, 1.0)


class MaskSource:
    """Mask provider for the full-size golden test."""

    def __init__(self) -> None:
        self.mask_calls = 0

    def wrist_arm_mask(self) -> np.ndarray:
        self.mask_calls += 1
        return synthetic_mask()


class MaskedFakeSim(FakeSim):
    """FakeSim + a soft mask sized to its 4x4 frames, for the
    loop-level purity oracle."""

    def wrist_arm_mask(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0, 1.0],
                [0.5, 1.0, 1.0, 1.0],
                [0.0, 0.5, 1.0, 0.5],
            ],
        )


def test_none_routes_around_the_hook() -> None:
    # 'none' must stay bit-identical to a driver without the hook: the
    # factory returns None and the loop never calls into transform code.
    assert make_wrist_transform("none", sim=None) is None


def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="not in"):
        make_wrist_transform("sepia", sim=None)
    assert WRIST_TRANSFORMS == ("none", "blackout", "freeze", "arm_blur")


def test_blackout_golden() -> None:
    obs = synthetic_obs()
    out = _built("blackout")(obs)
    assert out.wrist.shape == obs.wrist.shape
    assert out.wrist.dtype == np.uint8
    assert not out.wrist.any()
    # top and state pass through untouched (same objects — zero copies)
    assert out.top is obs.top
    assert out.state is obs.state


def test_freeze_replays_the_first_frame() -> None:
    transform = _built("freeze")
    first = synthetic_obs(tick=0)
    later = synthetic_obs(tick=7)
    assert not np.array_equal(first.wrist, later.wrist)
    out0 = transform(first)
    out7 = transform(later)
    np.testing.assert_array_equal(out0.wrist, first.wrist)
    np.testing.assert_array_equal(out7.wrist, first.wrist)
    # frozen PIXELS, live everything else
    assert out7.state is later.state
    assert out7.top is later.top
    # a fresh instance (a fresh episode) freezes its own first frame
    fresh = _built("freeze")(later)
    np.testing.assert_array_equal(fresh.wrist, later.wrist)


def test_arm_blur_golden() -> None:
    sim = MaskSource()
    transform = make_wrist_transform("arm_blur", sim)
    assert isinstance(transform, ArmBlurTransform)
    obs = synthetic_obs()
    out = transform(obs)
    mask = synthetic_mask()
    hard = mask > 0.999
    outside = mask == 0.0
    # outside the mask: bit-untouched; inside: texture dies, mean
    # color survives (mask-normalized blur — no background bleed)
    np.testing.assert_array_equal(out.wrist[outside], obs.wrist[outside])
    assert (out.wrist[hard] != obs.wrist[hard]).mean() > 0.9
    before = obs.wrist[hard].astype(np.float64)
    after = out.wrist[hard].astype(np.float64)
    assert abs(after.mean() - before.mean()) < 2.0
    # "shading structure and specular texture die": high-frequency
    # energy inside the mask collapses (px-to-px differences), while
    # the smooth large-scale gradients (mean color, silhouette
    # shading) are what survives
    pair = hard[:, 1:] & hard[:, :-1]

    def hf(frame: np.ndarray) -> float:
        diff = np.diff(frame.astype(np.float64), axis=1)
        return float(diff[pair].std())

    assert hf(out.wrist) < 0.3 * hf(obs.wrist)
    assert transform.coverage == [pytest.approx(float(mask.mean()))]
    assert sim.mask_calls == 1
    # golden frame: the exact bytes are pinned — any change to the
    # sigma, the kernel, or the composite is a registered amendment
    assert ARM_BLUR_SIGMA == 8.0
    assert md5(out.wrist.tobytes()).hexdigest() == "6c3f07229e9041cb6b79a093198152f1"


def test_gaussian_blur_conserves_flat_fields() -> None:
    flat = np.full((40, 50, 3), 173.0)
    np.testing.assert_allclose(gaussian_blur(flat, ARM_BLUR_SIGMA), flat)


def test_arm_blur_requires_a_mask_source() -> None:
    with pytest.raises(TypeError, match="wrist_arm_mask"):
        ArmBlurTransform(object())


def _record_episode(
    seed: int,
    transform_name: str,
) -> tuple[EpisodeResult, list[SimObservation]]:
    """One FakeSim episode through the shared loop; returns (row, the
    observations the 'policy' actually saw)."""
    sim = FakeSim() if transform_name != "arm_blur" else MaskedFakeSim()
    seen: list[SimObservation] = []

    def next_chunk(obs: SimObservation, replan: int) -> np.ndarray:
        seen.append(obs)
        # key the chunk off the RAW frames for both runs, so the
        # commanded actions (and thus physics) agree by construction
        # unless the transform leaked into the loop's own obs stream
        return fake_chunk(seed, replan, sim._observe())

    row = run_episode_loop(
        sim,
        seed,
        next_chunk,
        replans=REPLANS,
        horizon=HORIZON,
        video_path=None,
        latencies=[],
        transform=make_wrist_transform(transform_name, sim),
    )
    return row, seen


@pytest.mark.parametrize("name", ["blackout", "freeze", "arm_blur"])
def test_purity_transforms_touch_pixels_never_state(name: str) -> None:
    for seed in (0, 3):
        base_row, base_seen = _record_episode(seed, "none")
        row, seen = _record_episode(seed, name)
        # identical physics: every row field, the full distance trace
        # and the grip trace bit-match the untransformed episode
        assert asdict(row) == asdict(base_row)
        assert len(seen) == len(base_seen)
        for raw, treated in zip(base_seen, seen, strict=True):
            # the treatment reached the policy...
            if name == "blackout":
                assert not treated.wrist.any()
            elif name == "freeze":
                np.testing.assert_array_equal(treated.wrist, base_seen[0].wrist)
            else:
                assert treated.wrist.shape == raw.wrist.shape
            # ...while top and state stayed raw
            np.testing.assert_array_equal(treated.top, raw.top)
            np.testing.assert_array_equal(treated.state, raw.state)


def test_purity_input_obs_never_mutated() -> None:
    sim = MaskSource()
    for name in ("blackout", "freeze", "arm_blur"):
        obs = synthetic_obs()
        pinned = (obs.top.copy(), obs.wrist.copy(), obs.state.copy())
        _built(name, sim)(obs)
        np.testing.assert_array_equal(obs.top, pinned[0])
        np.testing.assert_array_equal(obs.wrist, pinned[1])
        np.testing.assert_array_equal(obs.state, pinned[2])
