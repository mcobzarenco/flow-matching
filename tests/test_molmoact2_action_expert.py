"""Oracles for the MolmoAct2 action-expert port (item 1, G3 CPU scale).

Checkpoint-facing facts pinned here were measured on the real HF
exports (released ``allenai/MolmoAct2-SO100_101`` and the rig-ft rung
dirs): the ``model.action_expert.*`` key template set, the absence of
``state_encoder``/``state_norm``/``kv_proj``/QK-norm weights, and the
577.6M parameter count at the released config. GPU forward parity vs
their HF forward is the pre-registered G1 gate (a script, not a unit
test).
"""

import re

import pytest
import torch

from bijou.molmoact2 import ActionExpert, ActionExpertConfig, load_action_expert_state

TINY = ActionExpertConfig(
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
)
TINY_KV_DIM = 40

# Distinct parameter-name templates measured in the real HF exports
# (block index -> N). kv_proj/state_encoder are absent there and
# compat-injected at load.
HF_KEY_TEMPLATES = {
    "action_embed.bias",
    "action_embed.weight",
    "blocks.N.cross_attn.out_proj.bias",
    "blocks.N.cross_attn.out_proj.weight",
    "blocks.N.cross_attn.q_proj.bias",
    "blocks.N.cross_attn.q_proj.weight",
    "blocks.N.mlp.down_proj.bias",
    "blocks.N.mlp.down_proj.weight",
    "blocks.N.mlp.gate_proj.bias",
    "blocks.N.mlp.gate_proj.weight",
    "blocks.N.mlp.up_proj.bias",
    "blocks.N.mlp.up_proj.weight",
    "blocks.N.modulation.linear.bias",
    "blocks.N.modulation.linear.weight",
    "blocks.N.self_attn.out_proj.bias",
    "blocks.N.self_attn.out_proj.weight",
    "blocks.N.self_attn.qkv.bias",
    "blocks.N.self_attn.qkv.weight",
    "context_k_proj.weight",
    "context_v_proj.weight",
    "final_layer.linear.bias",
    "final_layer.linear.weight",
    "final_layer.modulation.linear.bias",
    "final_layer.modulation.linear.weight",
    "time_embed.N.bias",
    "time_embed.N.weight",
}
COMPAT_TEMPLATES = {
    "state_encoder.weight",
    "state_encoder.bias",
    "blocks.N.cross_attn.kv_proj.weight",
    "blocks.N.cross_attn.kv_proj.bias",
}


def tiny_expert() -> ActionExpert:
    torch.manual_seed(0)
    return TINY.build(llm_kv_dim=TINY_KV_DIM)


def tiny_inputs(batch: int = 2, ctx_len: int = 5):  # noqa: ANN201
    torch.manual_seed(1)
    actions = torch.randn(batch, TINY.max_horizon, TINY.max_action_dim)
    timesteps = torch.rand(batch)
    kv_states = [
        (
            torch.randn(batch, ctx_len, TINY_KV_DIM),
            torch.randn(batch, ctx_len, TINY_KV_DIM),
        )
        for _ in range(TINY.num_layers)
    ]
    states = torch.randn(batch, TINY.hidden_size)
    return actions, timesteps, kv_states, states


def test_state_dict_key_set_matches_hf_export_template() -> None:
    expert = tiny_expert()
    templates = {re.sub(r"\.\d+\.", ".N.", k) for k in expert.state_dict()}
    assert templates == HF_KEY_TEMPLATES | COMPAT_TEMPLATES


def test_released_config_parameter_count() -> None:
    # 577,564,448 measured exactly on the real exports (588 tensors);
    # instantiated, the module additionally carries the compat-only
    # state_encoder + kv_proj — 620,677,664 total, the paper's "621M"
    # (outputs/probe_molmoact2_param_count.py has the breakdown).
    expert = ActionExpertConfig.released_so100_101().build(llm_kv_dim=1024)
    hf_visible = sum(
        p.numel()
        for name, p in expert.named_parameters()
        if "kv_proj" not in name and not name.startswith("state_encoder")
    )
    assert hf_visible == 577_564_448
    assert sum(p.numel() for p in expert.parameters()) == 620_677_664
    n_tensors = sum(
        1
        for name in expert.state_dict()
        if "kv_proj" not in name and not name.startswith("state_encoder")
    )
    assert n_tensors == 588


def test_forward_shape_and_zero_init() -> None:
    expert = tiny_expert().eval()
    actions, timesteps, kv_states, states = tiny_inputs()
    out = expert(actions, timesteps, kv_states, state_embeddings=states)
    assert out.shape == actions.shape
    # AdaLN gates and the final projection are zero-initialized: the
    # expert is exactly the zero field at init.
    assert torch.equal(out, torch.zeros_like(out))


def test_forward_nonzero_after_perturbation() -> None:
    expert = tiny_expert().eval()
    with torch.no_grad():
        for block in expert.iter_blocks():
            block.modulation.linear.bias.fill_(0.1)
        expert.final_layer.linear.weight.normal_(std=0.02)
        expert.final_layer.modulation.linear.bias.fill_(0.1)
    actions, timesteps, kv_states, states = tiny_inputs()
    out = expert(actions, timesteps, kv_states, state_embeddings=states)
    assert out.abs().max() > 0


def test_load_state_dict_round_trip() -> None:
    expert = tiny_expert()
    with torch.no_grad():
        for p in expert.parameters():
            p.normal_(std=0.02)
    # Mimic a real HF export: prefixed keys, compat tensors absent.
    exported = {
        f"model.action_expert.{k}": v.clone()
        for k, v in expert.state_dict().items()
        if "kv_proj" not in k and not k.startswith("state_encoder")
    }
    fresh = tiny_expert()
    load_action_expert_state(fresh, exported)
    actions, timesteps, kv_states, _states = tiny_inputs()
    # state_encoder loads as identity: encoded state = RMSNorm(state).
    a = expert.eval()
    b = fresh.eval()
    with torch.no_grad():
        expert.state_encoder.weight.copy_(torch.eye(TINY.hidden_size))
        expert.state_encoder.bias.zero_()
        states = torch.randn(2, TINY.hidden_size)
        out_a = a(actions, timesteps, kv_states, state_embeddings=states)
        out_b = b(actions, timesteps, kv_states, state_embeddings=states)
    assert torch.equal(out_a, out_b)


def test_load_state_dict_rejects_missing_prefix() -> None:
    fresh = tiny_expert()
    with pytest.raises(ValueError, match="no keys under prefix"):
        load_action_expert_state(fresh, {"action_embed.weight": torch.zeros(1)})


def test_action_mask_zeroes_padded_steps() -> None:
    expert = tiny_expert().eval()
    with torch.no_grad():
        for block in expert.iter_blocks():
            block.modulation.linear.bias.fill_(0.1)
        expert.final_layer.linear.weight.normal_(std=0.02)
        expert.final_layer.linear.bias.normal_(std=0.02)
    actions, timesteps, kv_states, states = tiny_inputs()
    mask = torch.ones(2, TINY.max_horizon)
    mask[:, -2:] = 0.0
    out = expert(
        actions,
        timesteps,
        kv_states,
        action_attention_mask=mask,
        state_embeddings=states,
    )
    assert torch.equal(out[:, -2:], torch.zeros_like(out[:, -2:]))
    assert out[:, :-2].abs().max() > 0


def test_kv_state_count_mismatch_raises() -> None:
    expert = tiny_expert()
    actions, timesteps, kv_states, states = tiny_inputs()
    with pytest.raises(ValueError, match="one KV state per action expert block"):
        expert(actions, timesteps, kv_states[:-1], state_embeddings=states)


def test_horizon_overflow_raises() -> None:
    expert = tiny_expert()
    _actions, timesteps, kv_states, states = tiny_inputs()
    long_actions = torch.randn(2, TINY.max_horizon + 1, TINY.max_action_dim)
    with pytest.raises(ValueError, match="exceeds max_horizon"):
        expert(long_actions, timesteps, kv_states, state_embeddings=states)
