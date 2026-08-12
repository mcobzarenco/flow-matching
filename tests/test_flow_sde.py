"""Oracles for the Flow-GRPO SDE sampler (FlowDecoder.sample_actions_sde,
the GRPO signal probe's cell-5 sampler, owner-approved 13:16Z 08-12):

- noise_level=0 must reproduce the Euler ODE decode BIT-FOR-BIT (the
  memo's bit-identity-at-a=0 oracle) — the SDE is a strict superset of
  the sampler our banked numbers use, not a reimplementation drift;
- seeded reproducibility and genuine per-draw diversity at a>0;
- the returned logprob matches an independent Gaussian recomputation of
  the per-step transitions (the exactness GRPO ratios depend on).
"""

import math

import pytest
import torch
from test_flow_decoder import BATCH, fabricate, tiny_config

from bijou.decoders.flow import FlowDecoder, SamplingMethod, TimeConditioning

NUM_STEPS = 10


@pytest.fixture(scope="module")
def decoder() -> FlowDecoder:
    torch.manual_seed(0)
    decoder = FlowDecoder(
        tiny_config(TimeConditioning.ADARMS),
        device="cpu",
        dtype=torch.float32,
    )
    # Zero-init (adaLN-Zero) gives a zero field — perturb into a
    # non-degenerate one so the oracles cannot pass vacuously.
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    decoder.eval()
    return decoder


def test_a0_bit_identity_with_euler(decoder: FlowDecoder) -> None:
    memory, state, _actions, _time = fabricate()
    noise = torch.randn(BATCH, 4, 6, generator=torch.Generator().manual_seed(3))
    ode = decoder.sample_actions(
        memory,
        state,
        num_steps=NUM_STEPS,
        method=SamplingMethod.EULER,
        noise=noise.clone(),
    )
    sde = decoder.sample_actions_sde(
        memory,
        state,
        noise_level=0.0,
        num_steps=NUM_STEPS,
        noise=noise.clone(),
    )
    assert isinstance(sde, torch.Tensor)
    assert torch.equal(ode, sde)


def test_seeded_reproducible_and_diverse(decoder: FlowDecoder) -> None:
    memory, state, _actions, _time = fabricate()

    def draw(seed: int) -> torch.Tensor:
        out = decoder.sample_actions_sde(
            memory,
            state,
            noise_level=0.5,
            num_steps=NUM_STEPS,
            generator=torch.Generator().manual_seed(seed),
        )
        assert isinstance(out, torch.Tensor)
        return out

    first, again, other = draw(7), draw(7), draw(8)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert first.isfinite().all()


def test_logprob_matches_gaussian_recomputation(decoder: FlowDecoder) -> None:
    """Replay the trajectory under the same generator and recompute each
    transition's N(mean, std²) logprob independently of the sampler's
    internal accumulation."""
    memory, state, _actions, _time = fabricate()
    a, seed = 0.5, 11
    result = decoder.sample_actions_sde(
        memory,
        state,
        noise_level=a,
        num_steps=NUM_STEPS,
        generator=torch.Generator().manual_seed(seed),
        return_logprob=True,
    )
    assert isinstance(result, tuple)
    actions_out, logprob = result
    assert logprob.shape == (BATCH,)

    generator = torch.Generator().manual_seed(seed)
    actions = torch.randn(BATCH, 4, 6, generator=generator)
    expected = torch.zeros(BATCH)
    for k in range(NUM_STEPS):
        t = 1.0 - k / NUM_STEPS
        dt = (1.0 - (k + 1) / NUM_STEPS) - t
        time = torch.full((BATCH,), t)
        velocity = decoder(memory, state, actions, time, None)
        denominator = 1.0 - t if k > 0 else 1.0 - (1.0 - 1.0 / NUM_STEPS)
        sigma = a * math.sqrt(t / denominator)
        mean = actions + dt * (
            velocity + (sigma**2 / (2.0 * t)) * (actions + (1.0 - t) * velocity)
        )
        std = sigma * math.sqrt(-dt)
        step = torch.randn(BATCH, 4, 6, generator=generator)
        actions = mean + std * step
        expected += (
            torch.distributions.Normal(mean, std).log_prob(actions).sum(dim=(1, 2))
        )
    assert torch.equal(actions_out, actions)
    torch.testing.assert_close(logprob, expected, rtol=1e-5, atol=1e-5)


def test_supplied_step_noise_matches_generator_stream(decoder: FlowDecoder) -> None:
    """Explicit step_noise (the batch-composition-invariant path the
    keyed rollout decode uses) must reproduce the generator draw when
    fed the same ε sequence — the two paths are one sampler."""
    memory, state, _actions, _time = fabricate()
    seed = 13
    from_generator = decoder.sample_actions_sde(
        memory,
        state,
        noise_level=0.5,
        num_steps=NUM_STEPS,
        generator=torch.Generator().manual_seed(seed),
    )
    replay = torch.Generator().manual_seed(seed)
    initial = torch.randn(BATCH, 4, 6, generator=replay)
    steps = torch.stack(
        [torch.randn(BATCH, 4, 6, generator=replay) for _ in range(NUM_STEPS)],
    )
    supplied = decoder.sample_actions_sde(
        memory,
        state,
        noise_level=0.5,
        num_steps=NUM_STEPS,
        noise=initial,
        step_noise=steps,
    )
    assert isinstance(from_generator, torch.Tensor)
    assert isinstance(supplied, torch.Tensor)
    assert torch.equal(from_generator, supplied)


def test_step_noise_shape_refused(decoder: FlowDecoder) -> None:
    memory, state, _actions, _time = fabricate()
    with pytest.raises(ValueError, match="step_noise shaped"):
        decoder.sample_actions_sde(
            memory,
            state,
            noise_level=0.5,
            num_steps=NUM_STEPS,
            step_noise=torch.zeros(NUM_STEPS - 1, BATCH, 4, 6),
        )


def test_logprob_refused_at_a0(decoder: FlowDecoder) -> None:
    memory, state, _actions, _time = fabricate()
    with pytest.raises(ValueError, match="noise_level=0"):
        decoder.sample_actions_sde(
            memory,
            state,
            noise_level=0.0,
            return_logprob=True,
        )
