"""Oracles for the MolmoAct2 backbone↔expert wiring (item 1 remainder).

CPU-scale checks of the inference glue in ``bijou/molmoact2/wiring.py``
— KV extraction off ``Molmo2KVCache``, encoder masking, the ascending-
Euler flow loop, and the config guards. Real-weights parity vs their
executing HF code is the script gate
(``fontaine/scripts/molmoact2_wiring_parity.py``), not a unit test.
"""

import pytest
import torch

from bijou.molmo2.cache import Molmo2KVCache
from bijou.molmoact2 import (
    ActionExpert,
    ActionExpertConfig,
    encoder_attention_mask,
    extract_kv_states,
    flow_timesteps,
    generate_actions,
    layer_kv_to_sequence,
    validate_inference_config,
)

# Same tiny config as test_molmoact2_action_expert.py (tests/ is not a
# package, so the fixture is restated rather than imported).
TINY = ActionExpertConfig(
    max_horizon=6,
    max_action_dim=8,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    ffn_multiple_of=32,
    timestep_embed_dim=16,
)
TINY_KV_DIM = 40


def tiny_expert() -> ActionExpert:
    torch.manual_seed(0)
    return ActionExpertConfig(**vars(TINY)).build(llm_kv_dim=TINY_KV_DIM)


SUPPORTED_CONFIG = {
    "add_action_expert": True,
    "action_expert_depth_gate": False,
    "action_mode": "continuous",
}


def filled_cache(
    num_layers: int = 2,
    batch: int = 2,
    kv_heads: int = 4,
    seq: int = 5,
    head_dim: int = 10,
) -> Molmo2KVCache:
    torch.manual_seed(2)
    cache = Molmo2KVCache(num_layers=num_layers)
    for idx in range(num_layers):
        cache.update(
            idx,
            torch.randn(batch, kv_heads, seq, head_dim),
            torch.randn(batch, kv_heads, seq, head_dim),
        )
    cache.advance(seq)
    return cache


def tiny_kv_states(batch: int = 2, ctx_len: int = 5):  # noqa: ANN201
    torch.manual_seed(3)
    return [
        (
            torch.randn(batch, ctx_len, TINY_KV_DIM),
            torch.randn(batch, ctx_len, TINY_KV_DIM),
        )
        for _ in range(TINY.num_layers)
    ]


def test_layer_kv_to_sequence_heads_first_layout() -> None:
    # The Molmo2KVCache layout: [B, kv_heads, S, head_dim].
    x = torch.randn(2, 4, 5, 10)
    out = layer_kv_to_sequence(x, num_attention_heads=8, num_key_value_heads=4)
    assert out.shape == (2, 5, 40)
    # Position s must hold all heads' vectors for that position.
    assert torch.equal(out[1, 3], x[1, :, 3, :].reshape(-1))


def test_layer_kv_to_sequence_seq_first_layout() -> None:
    x = torch.randn(2, 5, 4, 10)
    out = layer_kv_to_sequence(x, num_attention_heads=8, num_key_value_heads=4)
    assert out.shape == (2, 5, 40)
    assert torch.equal(out[0, 2], x[0, 2].reshape(-1))


def test_layer_kv_to_sequence_rejects_non_4d() -> None:
    with pytest.raises(ValueError, match="4-dim"):
        layer_kv_to_sequence(
            torch.randn(2, 5, 40),
            num_attention_heads=8,
            num_key_value_heads=4,
        )


def test_extract_kv_states_shapes_and_order() -> None:
    cache = filled_cache()
    kv = extract_kv_states(
        cache,
        num_expert_blocks=2,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    assert len(kv) == 2
    for layer_idx, (k, v) in enumerate(kv):
        assert k.shape == v.shape == (2, 5, 40)
        layer = cache.layers[layer_idx]
        assert layer.keys is not None and layer.values is not None
        assert torch.equal(k[0, 1], layer.keys[0, :, 1, :].reshape(-1))
        assert torch.equal(v[0, 1], layer.values[0, :, 1, :].reshape(-1))


def test_extract_kv_states_truncates_to_seq_len() -> None:
    cache = filled_cache(seq=7)
    kv = extract_kv_states(
        cache,
        num_expert_blocks=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        seq_len=4,
    )
    for k, v in kv:
        assert k.shape[1] == v.shape[1] == 4


def test_extract_kv_states_layer_count_mismatch_raises() -> None:
    cache = filled_cache(num_layers=3)
    with pytest.raises(ValueError, match="expected 2 KV layers"):
        extract_kv_states(
            cache,
            num_expert_blocks=2,
            num_attention_heads=8,
            num_key_value_heads=4,
        )


def test_encoder_attention_mask_prefers_attention_mask() -> None:
    attn = torch.tensor([[1, 1, 0], [1, 0, 0]])
    ids = torch.tensor([[5, 6, -1], [7, -1, -1]])
    mask = encoder_attention_mask(ids, attn)
    assert mask is not None
    assert mask.dtype == torch.bool
    assert torch.equal(mask, attn.bool())
    # A clone: mutating the result must not touch the input.
    mask[0, 0] = False
    assert attn[0, 0] == 1


def test_encoder_attention_mask_falls_back_to_input_ids() -> None:
    ids = torch.tensor([[5, 6, -1]])
    mask = encoder_attention_mask(ids, None)
    assert mask is not None
    assert torch.equal(mask, torch.tensor([[True, True, False]]))
    assert encoder_attention_mask(None, None) is None


def test_encoder_attention_mask_rejects_both_mode() -> None:
    with pytest.raises(NotImplementedError, match="action_mode"):
        encoder_attention_mask(None, torch.ones(1, 3), action_mode="both")


def test_validate_inference_config_accepts_released_shape() -> None:
    validate_inference_config(SUPPORTED_CONFIG)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"add_action_expert": False}, ValueError),
        ({"action_expert_depth_gate": True}, NotImplementedError),
        ({"action_mode": "both"}, NotImplementedError),
    ],
)
def test_validate_inference_config_guards(override: dict, error: type) -> None:
    with pytest.raises(error):
        validate_inference_config({**SUPPORTED_CONFIG, **override})


def test_flow_timesteps_ascending_fp32_grid() -> None:
    steps = flow_timesteps(10, 3, torch.device("cpu"))
    assert len(steps) == 10
    values = [t[0].item() for t in steps]
    assert values == pytest.approx([i / 10 for i in range(10)])
    assert all(t.dtype == torch.float32 and t.shape == (3,) for t in steps)
    with pytest.raises(ValueError, match="num_steps"):
        flow_timesteps(0, 1, torch.device("cpu"))


def test_generate_actions_zero_field_returns_masked_noise() -> None:
    # Zero-init expert = zero velocity everywhere: the loop must return
    # exactly the initial noise (with padded dims zeroed).
    expert = tiny_expert().eval()
    kv = tiny_kv_states()
    pad = torch.zeros(TINY.max_action_dim, dtype=torch.bool)
    pad[-2:] = True
    out = generate_actions(
        expert,
        encoder_kv_states=kv,
        action_dim_is_pad=pad,
        num_steps=4,
        generator=torch.Generator().manual_seed(7),
    )
    expected = torch.randn(
        (2, TINY.max_horizon, TINY.max_action_dim),
        generator=torch.Generator().manual_seed(7),
    )
    expected[..., -2:] = 0.0
    assert torch.equal(out, expected)


def test_generate_actions_integrates_constant_velocity() -> None:
    # Constant velocity field v (via the final-layer bias on a zero
    # expert): sum of dt over the grid is exactly 1, so x_T = x_0 + v.
    expert = tiny_expert().eval()
    with torch.no_grad():
        expert.final_layer.linear.bias.normal_(std=1.0)
    kv = tiny_kv_states()
    gen = torch.Generator().manual_seed(11)
    out = generate_actions(
        expert,
        encoder_kv_states=kv,
        num_steps=5,
        mask_action_dim_padding=False,
        generator=gen,
    )
    x0 = torch.randn(
        (2, TINY.max_horizon, TINY.max_action_dim),
        generator=torch.Generator().manual_seed(11),
    )
    v = expert.final_layer.linear.bias
    torch.testing.assert_close(out, x0 + v, atol=1e-6, rtol=0)


def test_generate_actions_padding_masked_every_step() -> None:
    # A non-trivial expert must still leave padded dims at exactly 0.
    expert = tiny_expert().eval()
    with torch.no_grad():
        for block in expert.iter_blocks():
            block.modulation.linear.bias.fill_(0.1)
        expert.final_layer.linear.weight.normal_(std=0.05)
        expert.final_layer.linear.bias.normal_(std=0.05)
    pad = torch.zeros(TINY.max_action_dim, dtype=torch.bool)
    pad[-3:] = True
    out = generate_actions(
        expert,
        encoder_kv_states=tiny_kv_states(),
        action_dim_is_pad=pad,
        num_steps=3,
        generator=torch.Generator().manual_seed(13),
    )
    assert torch.equal(out[..., -3:], torch.zeros_like(out[..., -3:]))
    assert out[..., :-3].abs().max() > 0


def test_generate_actions_deterministic_under_seed() -> None:
    expert = tiny_expert().eval()
    kv = tiny_kv_states()
    a = generate_actions(
        expert,
        encoder_kv_states=kv,
        num_steps=3,
        generator=torch.Generator().manual_seed(5),
    )
    b = generate_actions(
        expert,
        encoder_kv_states=kv,
        num_steps=3,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.equal(a, b)


def test_generate_actions_horizon_and_input_guards() -> None:
    expert = tiny_expert().eval()
    kv = tiny_kv_states()
    out = generate_actions(
        expert,
        encoder_kv_states=kv,
        action_horizon=3,
        num_steps=2,
        generator=torch.Generator().manual_seed(1),
    )
    assert out.shape == (2, 3, TINY.max_action_dim)
    with pytest.raises(ValueError, match="action_horizon"):
        generate_actions(
            expert,
            encoder_kv_states=kv,
            action_horizon=TINY.max_horizon + 1,
        )
    with pytest.raises(ValueError, match="at least one encoder KV state"):
        generate_actions(expert, encoder_kv_states=[])


def test_time_conditioning_fp32_timesteps_on_bf16_expert() -> None:
    # The flow loop feeds fp32 grids regardless of expert dtype; their
    # HF path runs the sinusoid in fp32 and casts the embedding. Must
    # not crash, and must match sinusoid(fp32) -> cast -> MLP exactly.
    expert = tiny_expert().to(torch.bfloat16).eval()
    t = torch.full((2,), 0.3, dtype=torch.float32)
    conditioning = expert._time_conditioning(t)
    assert conditioning.dtype == torch.bfloat16
    sinusoid = expert.time_embed[0](t).to(torch.bfloat16)
    expected = expert.time_embed[3](
        torch.nn.functional.silu(expert.time_embed[1](sinusoid)),
    )
    assert torch.equal(conditioning, expected)
