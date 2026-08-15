"""SnapFlow self-distillation oracles (2026-08-06 pre-registration).

Pure CPU/synthetic, mirroring test_flow_decoder's tiny fixture. The
load-bearing guarantees:

- φ_s extension identity: a teacher checkpoint loaded into the
  φ_s-extended decoder is BIT-IDENTICAL on s=t forwards (zero-initialized
  φ_s output), and the extension adds exactly the φ_s keys.
- The distill objective is the declared mix α·L_FM + (1−α)·λ·L_shortcut
  (the SnapflowObjective payload's knobs, exercised here at the
  historical run values α=0.5, λ=0.1) with stop-gradient two-step-Euler
  shortcut targets, and its sum form reconstructs the mean form exactly.
- 1-NFE sampling is exactly ``x̂ = ε − F(ε, s=0, t=1)``.
- The --init-from config guard sanctions the extension direction only.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from test_chunked_backward import FakeInputs
from test_flow_decoder import (
    ACTION_DIM,
    BATCH,
    CHUNK,
    build,
    fabricate,
    tiny_config,
)

from bijou.modelling.decoders.flow import (
    FlowDecoder,
    SamplingMethod,
    TimeConditioning,
    _snapflow_squared_errors,
    snapflow_distill_loss,
    snapflow_distill_loss_sums,
)
from bijou.modelling.interface import CollatedBatch, NormStats

# The historical run mix (the module constants the payload replaced) —
# these oracles pin the SAME numbers under the threaded knobs.
ALPHA = 0.5
SHORTCUT_WEIGHT = 0.1

PHI_S_KEYS = {
    "target_time_in_proj.weight",
    "target_time_in_proj.bias",
    "target_time_out_proj.weight",
    "target_time_out_proj.bias",
}


def build_extended() -> FlowDecoder:
    torch.manual_seed(0)
    config = dataclasses.replace(
        tiny_config(TimeConditioning.ADARMS),
        target_time_embed=True,
    )
    return FlowDecoder(config, device="cpu", dtype=torch.float32)


def randomize(decoder: FlowDecoder, seed: int = 7) -> None:
    """Simulate a trained checkpoint: every parameter non-trivial."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator) * 0.1,
            )


def flow_batch() -> CollatedBatch[Any]:
    generator = torch.Generator().manual_seed(3)
    stats = NormStats(
        mean=torch.zeros(BATCH, ACTION_DIM),
        std=torch.ones(BATCH, ACTION_DIM),
        q01=None,
        q99=None,
    )
    return CollatedBatch(
        encoder_inputs=FakeInputs(),
        state=torch.randn(BATCH, ACTION_DIM, generator=generator),
        actions=torch.randn(BATCH, CHUNK, ACTION_DIM, generator=generator),
        action_is_pad=torch.zeros(BATCH, CHUNK, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=None,
        suffix_tokens=None,
        suffix_is_aux=None,
    )


def test_extension_adds_exactly_phi_s_keys() -> None:
    teacher_keys = set(build(TimeConditioning.ADARMS).state_dict())
    extended_keys = set(build_extended().state_dict())
    assert extended_keys - teacher_keys == PHI_S_KEYS
    assert teacher_keys - extended_keys == set()


def test_zero_init_identity_is_bit_exact() -> None:
    """Teacher weights loaded into the φ_s-extended decoder: s=t forwards
    (implicit AND explicit) are bitwise the teacher's — the pre-launch
    validation gate (a)."""
    teacher = build(TimeConditioning.ADARMS)
    randomize(teacher)
    extended = build_extended()
    missing, unexpected = extended.load_state_dict(
        teacher.state_dict(),
        strict=False,
    )
    assert set(missing) == PHI_S_KEYS and not unexpected
    memory, state, actions, time = fabricate()
    reference = teacher(memory, state, actions, time)
    assert torch.equal(extended(memory, state, actions, time), reference)
    assert torch.equal(
        extended(memory, state, actions, time, time),  # explicit s=t
        reference,
    )
    # s=0 is ALSO identical at init (φ_s output is zero for every s) —
    # the shortcut field starts exactly at the teacher's field.
    assert torch.equal(
        extended(memory, state, actions, time, torch.zeros_like(time)),
        reference,
    )


def test_unextended_decoder_refuses_target_time() -> None:
    teacher = build(TimeConditioning.ADARMS)
    memory, state, actions, time = fabricate()
    with pytest.raises(ValueError, match="target_time"):
        teacher(memory, state, actions, time, torch.zeros_like(time))


def test_phi_s_conditioning_goes_live_once_trained() -> None:
    extended = build_extended()
    randomize(extended)  # includes φ_s out_proj => conditioning is live
    memory, state, actions, time = fabricate()
    standard = extended(memory, state, actions, time)
    one_step = extended(memory, state, actions, time, torch.zeros_like(time))
    assert not torch.allclose(standard, one_step, atol=1e-5)


def test_snapflow_loss_is_the_declared_mix() -> None:
    """Mean form == α·mean(fm) + (1−α)·λ·mean(shortcut) with the SAME RNG
    draws, and the sum form reconstructs it exactly."""
    extended = build_extended()
    randomize(extended)
    memory, _, _, _ = fabricate()
    sample = flow_batch()

    torch.manual_seed(11)
    total = snapflow_distill_loss(
        extended,
        memory,
        sample,
        alpha=ALPHA,
        shortcut_weight=SHORTCUT_WEIGHT,
    )
    torch.manual_seed(11)
    fm_squared, shortcut_squared = _snapflow_squared_errors(
        extended,
        memory,
        sample,
    )
    expected = (
        ALPHA * fm_squared.mean()
        + (1 - ALPHA) * SHORTCUT_WEIGHT * shortcut_squared.mean()
    )
    assert torch.allclose(total, expected, atol=0, rtol=0)

    torch.manual_seed(11)
    loss_sum, count = snapflow_distill_loss_sums(
        extended,
        memory,
        sample,
        alpha=ALPHA,
        shortcut_weight=SHORTCUT_WEIGHT,
    )
    assert int(count) == sample.actions.numel()
    assert torch.allclose(loss_sum / count, total, atol=1e-6)


def test_snapflow_loss_closed_form_at_zero_field() -> None:
    """Fresh extended decoder (zero velocity field): the shortcut term is
    exactly 0 (all forwards return 0 ⇒ v_target = v_one_step = 0) and the
    total reduces to α·mean((ε − actions)²) — a closed-form oracle for
    the mix wiring."""
    extended = build_extended()  # true init: action_out_proj zeroed
    memory, _, _, _ = fabricate()
    sample = flow_batch()

    torch.manual_seed(23)
    total = snapflow_distill_loss(
        extended,
        memory,
        sample,
        alpha=ALPHA,
        shortcut_weight=SHORTCUT_WEIGHT,
    )
    torch.manual_seed(23)
    noise = torch.randn_like(sample.actions)
    _tau = (
        torch.distributions.Beta(1.5, 1.0)
        .sample((sample.actions.shape[0],))
        .to(sample.actions.device)
    )
    expected = ALPHA * (noise - sample.actions).pow(2).mean()
    assert torch.allclose(total, expected, atol=1e-6)


def test_snapflow_gradients_reach_phi_s() -> None:
    """The s=0 grad forward trains φ_s: after one backward, the φ_s
    output head has non-zero gradient (its input head follows next step
    through the now-nonzero output weight)."""
    extended = build_extended()
    randomize(extended)
    memory, _, _, _ = fabricate()
    sample = flow_batch()
    torch.manual_seed(5)
    snapflow_distill_loss(
        extended,
        memory,
        sample,
        alpha=ALPHA,
        shortcut_weight=SHORTCUT_WEIGHT,
    ).backward()
    assert extended.target_time_out_proj is not None
    grad = extended.target_time_out_proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_one_nfe_sampling_is_single_shortcut_forward() -> None:
    """sample_actions(euler, 1 step, target_time=0) ≡ ε − F(ε, s=0, t=1)."""
    extended = build_extended()
    randomize(extended)
    memory, state, _, _ = fabricate()
    generator = torch.Generator().manual_seed(9)
    epsilon = torch.randn(BATCH, CHUNK, ACTION_DIM, generator=generator)
    sampled = extended.sample_actions(
        memory,
        state,
        num_steps=1,
        method=SamplingMethod.EULER,
        noise=epsilon,
        target_time=0.0,
    )
    ones = torch.ones(BATCH)
    velocity = extended(memory, state, epsilon, ones, torch.zeros(BATCH))
    assert torch.equal(sampled, epsilon - velocity)


def test_config_roundtrip_and_backcompat() -> None:
    from bijou.loading import FlowDecoderSection, flow_decoder_config_from_expert

    extended = flow_decoder_config_from_expert(build_extended().config)
    assert extended.target_time_embed
    assert FlowDecoderSection.from_dict(extended.to_dict()) == extended
    # Checkpoints predating the field: absent key parses as unextended.
    legacy = dict(extended.to_dict())
    del legacy["target_time_embed"]
    assert not FlowDecoderSection.from_dict(legacy).target_time_embed


def test_phi_s_schema_directions() -> None:
    """The φ_s extension's schema facts behind the sanctioned warm
    start: a pre-field section parses unextended, and extended vs
    unextended sections differ ONLY in the extension key — the property
    the train path's exactly-φ_s-keys load tolerance leans on (the CLI
    direction rules live in tests/test_train_args.py:
    test_snapflow_implies_phi_s_where_mutable /
    test_resume_refuses_distill; the strict-load key tolerance in
    tests/test_train_vla.py)."""
    from bijou.loading import decoder_schema_dict

    teacher_dict = decoder_schema_dict(build(TimeConditioning.ADARMS))
    extended_dict = decoder_schema_dict(build_extended())
    differing = {
        key
        for key in set(teacher_dict) | set(extended_dict)
        if teacher_dict.get(key) != extended_dict.get(key)
    }
    assert differing == {"target_time_embed"}
    saved = {k: v for k, v in teacher_dict.items() if k != "target_time_embed"}
    from bijou.loading import FlowDecoderSection

    assert not FlowDecoderSection.from_dict(saved).target_time_embed
