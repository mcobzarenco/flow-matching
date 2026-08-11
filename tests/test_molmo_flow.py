"""Oracles for the molmo_flow decoder (§8.13 step 3).

The decoder OWNS a copy of the MolmoAct2 expert architecture (decision
3); ``bijou/molmoact2/action_expert.py`` is the frozen parity reference
while both exist, and the byte-parity oracles here pin the two — same
weights in, byte-equal tensors out, for the forward AND the whole
solver loop. Convention facts (§8.13 decision 2) get their own pins:
the ascending direction test ties loss target to sampler direction,
and the t-law support test pins the endpoint asymmetry (pure data IN
support, pure noise OUT — the mirror of flow.py's law) so nobody
"fixes" either. The released-shape mirror is sync-tested against the
port's staticmethod (literals live in ONE place until the port
retires). GPU-scale parity on the released weights is a box script
(§8.13 step-3 gate), not a unit test.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from bijou.decoders.molmo_flow import (
    MolmoFlowConfig,
    MolmoFlowDecoder,
    SamplingMethod,
    TimeLaw,
    flow_matching_loss_sums,
    load_expert_state,
    normalize_targets,
    unnormalize_chunk,
)
from bijou.molmoact2.action_expert import ActionExpertConfig
from bijou.molmoact2.wiring import generate_actions

TINY = MolmoFlowConfig(
    max_horizon=6,
    max_action_dim=8,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    ffn_multiple_of=32,
    timestep_embed_dim=16,
    dropout=0.0,
    attn_dropout=0.0,
    context_layer_norm=True,
    qk_norm=True,
    qk_norm_eps=1e-6,
    rope=True,
    causal_attn=False,
    llm_kv_dim=40,
)

RELEASED_LAW = TimeLaw(offset=0.001, scale=0.999, beta_alpha=1.0, beta_beta=1.5)


def _port_twin() -> ActionExpertConfig:
    """The port's config with the same tiny geometry (llm_kv_dim rides
    separately there)."""
    fields = {
        f.name: getattr(TINY, f.name) for f in dataclasses.fields(ActionExpertConfig)
    }
    return ActionExpertConfig(**fields)


def _tiny_pair() -> tuple[MolmoFlowDecoder, object]:
    """Decoder + port reference wearing IDENTICAL weights (the shared
    tensor-name contract makes the copy a plain state_dict load)."""
    torch.manual_seed(0)
    decoder = TINY.build()
    port = _port_twin().build(llm_kv_dim=TINY.llm_kv_dim)
    port.load_state_dict(decoder.state_dict(), strict=True)
    decoder.eval()
    port.eval()
    return decoder, port


def _inputs(batch: int = 2, ctx_len: int = 5) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    actions = torch.randn(batch, TINY.max_horizon, TINY.max_action_dim)
    timesteps = torch.rand(batch)
    kv_states = [
        (
            torch.randn(batch, ctx_len, TINY.llm_kv_dim),
            torch.randn(batch, ctx_len, TINY.llm_kv_dim),
        )
        for _ in range(TINY.num_layers)
    ]
    enc_mask = torch.ones(batch, ctx_len, dtype=torch.bool)
    enc_mask[1, -2:] = False  # a padded row exercises the additive mask
    return actions, timesteps, kv_states, enc_mask  # type: ignore[return-value]


def test_released_shape_mirrors_the_port() -> None:
    """Literals live in ONE place (the port's staticmethod); this
    mirror is pinned field-for-field until the port retires."""
    ours = MolmoFlowConfig.released_so100_101()
    theirs = ActionExpertConfig.released_so100_101()
    for field in dataclasses.fields(ActionExpertConfig):
        assert getattr(ours, field.name) == getattr(theirs, field.name), field.name
    assert ours.llm_kv_dim == 1024


def test_forward_byte_parity_with_port() -> None:
    """Same weights, same inputs -> byte-equal velocity, masked rows
    and padded batch columns included."""
    decoder, port = _tiny_pair()
    actions, timesteps, kv_states, enc_mask = _inputs()
    ours = decoder(
        actions,
        timesteps,
        kv_states,
        encoder_attention_mask=enc_mask,
    )
    reference = port(  # type: ignore[operator] — nn.Module reference
        actions,
        timesteps,
        kv_states,
        encoder_attention_mask=enc_mask,
    )
    assert torch.equal(ours, reference)
    # State path parity too (dormant at inference, still copied math).
    states = torch.randn(2, TINY.hidden_size)
    ours_state = decoder(actions, timesteps, kv_states, state_embeddings=states)
    reference_state = port(  # type: ignore[operator]
        actions,
        timesteps,
        kv_states,
        state_embeddings=states,
    )
    assert torch.equal(ours_state, reference_state)


def test_euler_loop_byte_parity_with_wiring() -> None:
    """The whole serving loop: decoder.sample_actions(EULER) equals the
    port's generate_actions under a shared generator — noise draw,
    per-step dim masking, trajectory update, byte-for-byte."""
    decoder, port = _tiny_pair()
    _actions, _timesteps, kv_states, enc_mask = _inputs()
    pad = torch.zeros(TINY.max_action_dim, dtype=torch.bool)
    pad[-2:] = True
    ours = decoder.sample_actions(
        kv_states,
        encoder_attention_mask=enc_mask,
        action_dim_is_pad=pad,
        num_steps=4,
        method=SamplingMethod.EULER,
        generator=torch.Generator().manual_seed(7),
    )
    reference = generate_actions(
        port,  # type: ignore[arg-type] — the parity reference
        encoder_kv_states=kv_states,
        encoder_attention_mask=enc_mask,
        action_dim_is_pad=pad,
        num_steps=4,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(ours, reference)


def test_ascending_direction_ties_loss_to_sampler() -> None:
    """The convention pin: x_t = (1−t)·ε + t·x with target x − ε, so a
    perfect field gives zero loss AND one ascending Euler step from
    pure noise lands exactly on the data — loss and sampler can never
    disagree about which end is data."""

    class _Echo(torch.nn.Module):
        def __init__(self, velocity: torch.Tensor) -> None:
            super().__init__()
            self.velocity = velocity
            self.seen: dict[str, torch.Tensor] = {}

        def forward(
            self,
            xt: torch.Tensor,
            times: torch.Tensor,
            encoder_kv_states: object = None,
            encoder_attention_mask: object = None,
        ) -> torch.Tensor:
            del encoder_kv_states, encoder_attention_mask
            self.seen["xt"] = xt.detach().clone()
            self.seen["times"] = times.detach().clone()
            return self.velocity.expand_as(xt)

    torch.manual_seed(2)
    batch, horizon, dim = 2, 4, 8
    actions = torch.randn(batch, horizon, dim)
    noise = torch.randn(batch, horizon, dim)
    dim_is_pad = torch.zeros(batch, dim, dtype=torch.bool)
    times = torch.full((batch,), 0.25)
    expert = _Echo(actions - noise)
    loss_sum, count = flow_matching_loss_sums(
        expert,  # type: ignore[arg-type] — stand-in
        kv_states=[],
        enc_mask=torch.ones(batch, 3, dtype=torch.bool),
        actions_norm=actions,
        action_dim_is_pad=dim_is_pad,
        times=times,
        noise=noise,
    )
    assert float(loss_sum) == pytest.approx(0.0, abs=1e-10)
    assert int(count) == batch * horizon
    # Interpolant: t=0.25 sits 25% of the way from NOISE to data —
    # the mirror of flow.py's τ (there τ=0.25 is 75% toward data).
    expected_xt = 0.75 * noise + 0.25 * actions
    assert torch.allclose(expert.seen["xt"], expected_xt)
    # One ascending Euler step of the true velocity from noise = data.
    assert torch.allclose(noise + (actions - noise) * 1.0, actions)


def test_time_law_support_pins_endpoint_asymmetry() -> None:
    """t ∈ [offset, offset + scale] = [0.001, 1.0]: pure DATA is in
    support, pure NOISE is not — the mirror of flow.py's law (τ=1 noise
    in, τ=0 data out). Mass sits at the noise end (Beta(1, 1.5) mean
    0.4). Generator-threaded draws are deterministic and obey the law."""
    torch.manual_seed(0)
    ambient = RELEASED_LAW.sample(20_000, device="cpu")
    assert float(ambient.min()) >= 0.001  # pure noise excluded
    assert float(ambient.max()) <= 1.0  # pure data included
    assert float(ambient.mean()) == pytest.approx(0.001 + 0.999 * 0.4, abs=0.02)
    seeded_a = RELEASED_LAW.sample(
        4096,
        device="cpu",
        generator=torch.Generator().manual_seed(3),
    )
    seeded_b = RELEASED_LAW.sample(
        4096,
        device="cpu",
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(seeded_a, seeded_b)
    assert float(seeded_a.min()) >= 0.001
    assert float(seeded_a.max()) <= 1.0
    assert float(seeded_a.mean()) == pytest.approx(0.001 + 0.999 * 0.4, abs=0.05)


def test_heun_matches_euler_on_constant_field() -> None:
    """A constant velocity field integrates exactly under both solvers
    (sum of dt = 1): x_1 = x_0 + v, and Heun == Euler byte-for-byte
    there (the corrector adds nothing on a constant field)."""
    torch.manual_seed(0)
    decoder = TINY.build().eval()
    with torch.no_grad():
        decoder.final_layer.linear.bias.normal_(std=1.0)
    _actions, _timesteps, kv_states, _mask = _inputs()
    kwargs = {
        "num_steps": 5,
        "mask_action_dim_padding": False,
    }
    euler = decoder.sample_actions(
        kv_states,
        method=SamplingMethod.EULER,
        generator=torch.Generator().manual_seed(11),
        **kwargs,  # type: ignore[arg-type]
    )
    heun = decoder.sample_actions(
        kv_states,
        method=SamplingMethod.HEUN,
        generator=torch.Generator().manual_seed(11),
        **kwargs,  # type: ignore[arg-type]
    )
    x0 = torch.randn(
        (2, TINY.max_horizon, TINY.max_action_dim),
        generator=torch.Generator().manual_seed(11),
    )
    v = decoder.final_layer.linear.bias
    torch.testing.assert_close(euler, x0 + v, atol=1e-6, rtol=0)
    torch.testing.assert_close(heun, euler, atol=1e-6, rtol=0)


def test_zero_init_returns_masked_noise() -> None:
    """Zero-init decoder = zero velocity everywhere: the loop returns
    exactly the initial noise with padded dims zeroed (their
    zero-field identity, same as the port's)."""
    torch.manual_seed(0)
    decoder = TINY.build().eval()
    _actions, _timesteps, kv_states, _mask = _inputs()
    pad = torch.zeros(TINY.max_action_dim, dtype=torch.bool)
    pad[-2:] = True
    out = decoder.sample_actions(
        kv_states,
        action_dim_is_pad=pad,
        num_steps=3,
        generator=torch.Generator().manual_seed(7),
    )
    expected = torch.randn(
        (2, TINY.max_horizon, TINY.max_action_dim),
        generator=torch.Generator().manual_seed(7),
    )
    expected[..., -2:] = 0.0
    assert torch.equal(out, expected)


def test_loss_masks_padded_dims_and_sum_form() -> None:
    """Padded dims are erased on both ends and excluded from the mean;
    sum/count reproduces the hand-built valid-dim mean exactly."""
    batch, horizon, dim, valid_dims = 2, 3, 6, 2

    class _Zero(torch.nn.Module):
        def forward(
            self,
            xt: torch.Tensor,
            times: object,
            encoder_kv_states: object = None,
            encoder_attention_mask: object = None,
        ) -> torch.Tensor:
            del times, encoder_kv_states, encoder_attention_mask
            return torch.zeros_like(xt)

    torch.manual_seed(4)
    actions = torch.randn(batch, horizon, dim)
    noise = torch.randn(batch, horizon, dim)
    actions[:, :, valid_dims:] = 1e6  # poison the padded dims
    noise[:, :, valid_dims:] = -1e6
    dim_is_pad = torch.ones(batch, dim, dtype=torch.bool)
    dim_is_pad[:, :valid_dims] = False
    loss_sum, count = flow_matching_loss_sums(
        _Zero(),  # type: ignore[arg-type]
        kv_states=[],
        enc_mask=None,
        actions_norm=actions,
        action_dim_is_pad=dim_is_pad,
        times=torch.full((batch,), 0.5),
        noise=noise,
    )
    target = actions[:, :, :valid_dims] - noise[:, :, :valid_dims]
    expected = (target**2).mean(dim=-1)
    assert float(loss_sum) == pytest.approx(float(expected.sum()), rel=1e-6)
    assert float(loss_sum / count) == pytest.approx(float(expected.mean()), rel=1e-6)


def test_normalize_unnormalize_tail() -> None:
    """Their clamp order on both ends: targets clamp AFTER normalizing,
    chunks clamp BEFORE unnormalizing — predictions can never leave the
    quantile box."""
    q01 = torch.tensor([-2.0, 0.0])
    q99 = torch.tensor([2.0, 10.0])
    raw = torch.tensor([[0.0, 5.0], [-4.0, 20.0]])  # second row out of range
    norm = normalize_targets(raw, q01, q99)
    assert torch.allclose(norm[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(norm[1], torch.tensor([-1.0, 1.0]))  # clamped
    back = unnormalize_chunk(torch.tensor([[0.0, 0.0], [-3.0, 3.0]]), q01, q99)
    assert torch.allclose(back[0], torch.tensor([0.0, 5.0]))
    assert torch.allclose(back[1], torch.tensor([-2.0, 10.0]))  # clamped to box
    # Round trip inside the box is exact.
    assert torch.allclose(unnormalize_chunk(norm[0], q01, q99), raw[0])


def test_load_expert_state_injects_compat_and_freezes() -> None:
    """The converter's unprefixed names load strictly (compat tensors
    injected: identity state_encoder, zero kv_proj), and the frozen
    surface matches the reference trainable set."""
    torch.manual_seed(5)
    source = TINY.build()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(std=0.02)
    exported = {
        name: tensor.clone()
        for name, tensor in source.state_dict().items()
        if "kv_proj" not in name and not name.startswith("state_encoder")
    }
    fresh = TINY.build()
    load_expert_state(fresh, exported)
    assert torch.equal(
        fresh.state_encoder.weight,
        torch.eye(TINY.hidden_size),
    )
    for block in fresh.iter_blocks():
        assert not block.cross_attn.kv_proj.weight.requires_grad
        assert torch.equal(
            block.cross_attn.kv_proj.weight,
            torch.zeros_like(block.cross_attn.kv_proj.weight),
        )
    assert not fresh.state_encoder.weight.requires_grad
    trainable = {name for name, p in fresh.named_parameters() if p.requires_grad}
    assert trainable  # everything else trains
    assert all("kv_proj" not in name for name in trainable)
    assert all(not name.startswith("state_encoder") for name in trainable)
    with pytest.raises(ValueError, match="empty expert state dict"):
        load_expert_state(fresh, {})


def test_gradients_reach_trainable_set_through_loss() -> None:
    """End-to-end on the tiny real decoder: the ascending loss
    backpropagates into every trainable parameter (a dead graph would
    train nothing while logging a healthy loss)."""
    torch.manual_seed(6)
    decoder = TINY.build()
    batch, seq = 2, 5
    kv = [
        (
            torch.randn(batch, seq, TINY.llm_kv_dim),
            torch.randn(batch, seq, TINY.llm_kv_dim),
        )
        for _ in range(TINY.num_layers)
    ]
    loss_sum, count = flow_matching_loss_sums(
        decoder,
        kv_states=kv,
        enc_mask=torch.ones(batch, seq, dtype=torch.bool),
        actions_norm=torch.randn(batch, 4, TINY.max_action_dim).clamp(-1, 1),
        action_dim_is_pad=torch.zeros(batch, TINY.max_action_dim, dtype=torch.bool),
        times=torch.tensor([0.3, 0.8]),
        noise=torch.randn(batch, 4, TINY.max_action_dim),
    )
    (loss_sum / count).backward()
    missing = [
        name
        for name, p in decoder.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, missing
