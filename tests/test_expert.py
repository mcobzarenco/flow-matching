"""Tests for the action expert's time-conditioning modes.

Pure CPU/synthetic: a tiny ActionExpert with a fabricated prefix, no
backbone or checkpoint. The load-bearing guarantee is adaRMS
identity-at-init (every layer is the identity, so the body passes the
residual stream through untouched); the additive path must NOT have that
property, and both must produce a zero velocity field at true init
(zero-initialized output projection).
"""

from __future__ import annotations

import torch

from bijou.expert import (
    ActionExpert,
    ExpertConfig,
    ExpertLayer,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.interface import EncodedPrefix, MemoryStream
from bijou.nn import RopeParameters, RopeType

BATCH, PREFIX_LEN, CHUNK, ACTION_DIM, STATE_DIM, HIDDEN = 2, 5, 4, 6, 6, 32


def tiny_config(time_conditioning: TimeConditioning) -> ExpertConfig:
    return ExpertConfig(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
        self_attention_rope_theta=10_000.0,
        cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_rope=RopeParameters(
            rope_type=RopeType.DEFAULT,
            rope_theta=10_000.0,
            factor=1.0,
            partial_rotary_factor=1.0,
        ),
        cross_attention_schedule=(0, 0),
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        chunk_size=CHUNK,
        time_embed_dim=8,
        time_conditioning=time_conditioning,
    )


def fabricate() -> tuple[EncodedPrefix, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    head_dim = 16
    key = torch.randn(BATCH, 1, PREFIX_LEN, head_dim, generator=generator)
    value = torch.randn(BATCH, 1, PREFIX_LEN, head_dim, generator=generator)
    prefix = EncodedPrefix(
        streams={"kv0": MemoryStream(key=key, value=value)},
        length=PREFIX_LEN,
        padding_mask=None,
    )
    state = torch.randn(BATCH, STATE_DIM, generator=generator)
    actions = torch.randn(BATCH, CHUNK, ACTION_DIM, generator=generator)
    time = torch.rand(BATCH, generator=generator)
    return prefix, state, actions, time


def build(time_conditioning: TimeConditioning) -> ActionExpert:
    torch.manual_seed(0)
    return ActionExpert(
        tiny_config(time_conditioning),
        device="cpu",
        dtype=torch.float32,
    )


def test_zero_velocity_at_init() -> None:
    prefix, state, actions, time = fabricate()
    for mode in (TimeConditioning.ADDITIVE, TimeConditioning.ADARMS):
        out = build(mode)(prefix, state, actions, time)
        assert out.shape == (BATCH, CHUNK, ACTION_DIM)
        # Zero-init action_out_proj => zero field, both modes.
        torch.testing.assert_close(out, torch.zeros_like(out))


def test_adarms_modulation_heads_zero_at_init() -> None:
    expert = build(TimeConditioning.ADARMS)
    for layer in expert.layers:
        assert isinstance(layer, ExpertLayer)
        assert layer.modulation is not None
        head = layer.modulation[1]
        assert isinstance(head, torch.nn.Linear)
        assert bool((head.weight == 0).all()) and bool((head.bias == 0).all())
    assert expert.final_modulation is not None


def test_adarms_body_is_identity_at_init() -> None:
    """With a NON-trivial output projection, adaRMS at init reduces to
    action_out_proj(norm(input_hidden)) — i.e. every layer is the exact
    identity and no time is folded into the tokens."""
    expert = build(TimeConditioning.ADARMS)
    with torch.no_grad():
        expert.action_out_proj.weight.normal_(std=0.5)
        expert.action_out_proj.bias.normal_(std=0.5)
    prefix, state, actions, time = fabricate()
    out = expert(prefix, state, actions, time)

    state_embeds = expert.state_proj(state)[:, None, :]
    action_embeds = expert.action_in_proj(actions)  # no time add in adaRMS
    hidden = torch.cat([state_embeds, action_embeds], dim=1)
    expected = expert.action_out_proj(expert.norm(hidden))[:, 1:, :]
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_additive_body_is_not_identity() -> None:
    """The additive path does real per-layer computation at init (only the
    output projection zeros the field) — distinct from adaRMS."""
    expert = build(TimeConditioning.ADDITIVE)
    for layer in expert.layers:
        assert isinstance(layer, ExpertLayer)
        assert layer.modulation is None
    with torch.no_grad():
        expert.action_out_proj.weight.normal_(std=0.5)
        expert.action_out_proj.bias.normal_(std=0.5)
    prefix, state, actions, time = fabricate()
    out = expert(prefix, state, actions, time)

    state_embeds = expert.state_proj(state)[:, None, :]
    action_embeds = expert.action_in_proj(actions)
    hidden = torch.cat([state_embeds, action_embeds], dim=1)
    identity_out = expert.action_out_proj(expert.norm(hidden))[:, 1:, :]
    assert not torch.allclose(out, identity_out, atol=1e-3)
