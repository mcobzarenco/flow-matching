"""Oracles for the rollout --draws keying (GRPO signal probe
instrument): draw 0 must leave the banked identity triple BIT-EXACTLY
untouched (every deterministic row ever banked stays comparable), and
draws >= 1 must re-key BOTH policy-side stochastic streams (stable-key
flow noise and the AR sample RNG) to values distinct per draw."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from bijou.data import DatasetStats
from bijou.eval.policies import (
    stable_noise,
    stable_sample_rng,
    stable_sde_step_noise,
)
from sim.rollout_sim import EpisodeResult, sim_item
from sim.so101_sim import SimObservation

_STATS = DatasetStats.from_state_dict(
    {
        "action": {"mean": [0.0] * 6, "std": [1.0] * 6},
        "observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6},
    },
)


def _obs() -> SimObservation:
    rng = np.random.default_rng(0)
    return SimObservation(
        top=rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
        wrist=rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
        state=rng.standard_normal(6).astype(np.float64),
    )


def _repo_id(draw: int) -> str:
    item = sim_item(_obs(), seed=3, replan=2, stats=_STATS, chunk_size=50, draw=draw)
    assert item["episode_index"] == 3
    assert item["frame_index"] == 2
    return str(item["repo_id"])


def test_draw0_is_the_banked_identity() -> None:
    default = sim_item(_obs(), seed=3, replan=2, stats=_STATS, chunk_size=50)
    assert default["repo_id"] == "sim/eval100"
    assert _repo_id(0) == "sim/eval100"


def test_draws_rekey_both_stochastic_streams() -> None:
    repos = [_repo_id(draw) for draw in range(4)]
    assert len(set(repos)) == 4
    noises = [stable_noise(0, repo, 3, 2, 0, shape=(50, 6)) for repo in repos]
    for a in range(len(noises)):
        for b in range(a + 1, len(noises)):
            assert not torch.equal(noises[a], noises[b])
    # Reproducible per draw (keyed, not ambient).
    assert torch.equal(
        noises[1],
        stable_noise(0, repos[1], 3, 2, 0, shape=(50, 6)),
    )
    # The AR sample RNG derives from the same triple: draw-distinct too.
    draws = [
        torch.from_numpy(stable_sample_rng(0, repo, 3, 2, 0).standard_normal(4))
        for repo in repos
    ]
    for a in range(len(draws)):
        for b in range(a + 1, len(draws)):
            assert not torch.equal(draws[a], draws[b])


def test_sde_step_noise_keyed_and_domain_separated() -> None:
    """The SDE per-step stream must re-key per draw through the repo_id
    suffix (like every other policy-side stream), reproduce exactly
    under the same key, and never replay the initial-noise stream at
    the same (seed, frame, draw) — different domain constants."""
    repos = [_repo_id(draw) for draw in range(3)]
    stacks = [
        stable_sde_step_noise(0, repo, 3, 2, 0, num_steps=10, shape=(50, 6))
        for repo in repos
    ]
    assert stacks[0].shape == (10, 50, 6)
    for a in range(len(stacks)):
        for b in range(a + 1, len(stacks)):
            assert not torch.equal(stacks[a], stacks[b])
    assert torch.equal(
        stacks[1],
        stable_sde_step_noise(0, repos[1], 3, 2, 0, num_steps=10, shape=(50, 6)),
    )
    # Domain separation from the initial-noise stream: same key, same
    # shape, different bits.
    initial = stable_noise(0, repos[0], 3, 2, 0, shape=(50, 6))
    assert not torch.equal(stacks[0][0], initial)
    # Steps within one stack are distinct (a stuck stream would make
    # every SDE step reuse one ε).
    assert not torch.equal(stacks[0][0], stacks[0][1])


def test_episode_result_draw_defaults_zero() -> None:
    row = EpisodeResult(
        seed=0,
        initial_cm=10.0,
        min_cm=8.0,
        final_cm=9.0,
        success_tick=None,
        spawn_xy=(0.2, 0.0),
        reset_strikes=0,
        final_z_mm=1.0,
        final_upright=1.0,
        ticks=450,
    )
    assert row.draw == 0
    assert row.progress_cm == pytest.approx(2.0)
