"""Molmo2 text decoder (WP1) — CPU oracles on the tiny structurally-faithful
fixture:

- config parsing refuses every feature the 4B SKU does not use (per-layer
  RoPE scaling, post-norm, non-qwen3 qk-norm, biases, dropout, tied
  embeddings) instead of half-implementing it;
- the extension-vocab embedding routes ids exactly like the reference
  (ids >= vocab_size hit ``new_embedding``);
- residual taps obey the gemma4 contract (post both residual adds, no
  final norm, both-or-neither with the sink);
- eager and SDPA attention agree (the parity baseline vs the fast path);
- the truncated-mount loader strict-loads the real key layout (``model.``
  prefix, top-level lm_head, vision keys skipped) and its blocks compute
  bitwise the same states as the full stack's prefix;
- padding orientation does not change real-token states (the property the
  stream-export protocol is gated on, proved here at the trunk level).

HF weights parity (full 36-layer activation + greedy-decode agreement) is
the separate GPU harness — port plan §4, run at an idle-GPU boundary.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from bijou.molmo2.config import Molmo2Config, Molmo2TextConfig
from bijou.molmo2.loading import load_text_model, truncated_config
from bijou.molmo2.testing import tiny_config_json, write_tiny_text_checkpoint
from bijou.molmo2.text import (
    DecoderLayer,
    Molmo2Embedding,
    Molmo2TextModel,
    Molmo2Transformer,
)
from bijou.nn import AttentionBackend


def tiny_text_config() -> Molmo2TextConfig:
    return Molmo2Config.from_dict(tiny_config_json()).text


def test_config_refuses_unimplemented_features() -> None:
    data = tiny_config_json()

    def parse(**overrides: object) -> Molmo2Config:
        mutated = copy.deepcopy(data)
        text = mutated["text_config"]
        assert isinstance(text, dict)
        for key, value in overrides.items():
            if key == "tie_word_embeddings":
                mutated[key] = value
            else:
                text[key] = value
        return Molmo2Config.from_dict(mutated)

    parse()  # the fixture itself parses
    with pytest.raises(NotImplementedError, match="rope_scaling_layers"):
        parse(rope_scaling_layers=[0, 1])
    with pytest.raises(NotImplementedError, match="rope_scaling"):
        parse(rope_scaling={"rope_type": "dynamic"})
    with pytest.raises(NotImplementedError, match="norm_after"):
        parse(norm_after=True)
    with pytest.raises(NotImplementedError, match="qk_norm"):
        parse(qk_norm_type="olmo")
    with pytest.raises(NotImplementedError, match="qk_norm"):
        parse(use_qk_norm=False)
    with pytest.raises(NotImplementedError, match="qkv_bias"):
        parse(qkv_bias=True)
    with pytest.raises(NotImplementedError, match="residual_dropout"):
        parse(residual_dropout=0.1)
    with pytest.raises(NotImplementedError, match="tied"):
        parse(tie_word_embeddings=True)


def test_extension_vocab_embedding_routing() -> None:
    torch.manual_seed(0)
    config = tiny_text_config()
    model = Molmo2Transformer(config)
    base_id, ext_id = 7, config.vocab_size + 3
    ids = torch.tensor([[base_id, ext_id]])
    embeds = model.wte(ids)
    assert torch.equal(embeds[0, 0], model.wte.embedding[base_id])
    assert torch.equal(embeds[0, 1], model.wte.new_embedding[3])
    # The concat lookup differentiates through both matrices.
    grads = torch.autograd.grad(
        embeds.sum(),
        [model.wte.embedding, model.wte.new_embedding],
    )
    assert all(grad.abs().sum() > 0 for grad in grads)


def test_forward_shapes_and_residual_tap_contract() -> None:
    torch.manual_seed(0)
    config = tiny_text_config()
    model = Molmo2Transformer(config)
    batch, seq_len = 2, 5
    ids = torch.randint(0, config.total_vocab_size, (batch, seq_len))

    sink: dict[int, torch.Tensor] = {}
    last = config.num_hidden_layers - 1
    out = model(ids, residual_taps=(0, 2, last), residual_sink=sink)
    assert out.shape == (batch, seq_len, config.hidden_size)
    assert set(sink) == {0, 2, last}
    for tap in sink.values():
        assert tap.shape == (batch, seq_len, config.hidden_size)
    # Tap = the raw residual stream, ln_f applied only to the return value.
    assert torch.equal(out, model.ln_f(sink[last]))
    assert not torch.equal(out, sink[last])

    with pytest.raises(ValueError, match="travel together"):
        model(ids, residual_taps=(0,))
    with pytest.raises(ValueError, match="travel together"):
        model(ids, residual_sink={})
    with pytest.raises(ValueError, match="outside the stack"):
        model(ids, residual_taps=(config.num_hidden_layers,), residual_sink={})
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.wte(ids))


def test_eager_and_sdpa_attention_agree() -> None:
    torch.manual_seed(0)
    config = tiny_text_config()
    model = Molmo2Transformer(config, attn_backend=AttentionBackend.EAGER)
    ids = torch.randint(0, config.vocab_size, (2, 7))
    out_eager = model(ids)
    for block in model.blocks:
        # ModuleList iteration erases the element type; narrow first.
        assert isinstance(block, DecoderLayer)
        block.self_attn.attn_backend = AttentionBackend.SDPA
    out_sdpa = model(ids)
    torch.testing.assert_close(out_eager, out_sdpa, rtol=1e-5, atol=1e-5)


def test_truncated_mount_matches_full_prefix(tmp_path: Path) -> None:
    checkpoint = write_tiny_text_checkpoint(tmp_path / "tiny-molmo2")
    full = load_text_model(checkpoint)
    mount = load_text_model(checkpoint, truncate_layers=3)

    assert full.lm_head is not None
    assert mount.lm_head is None
    assert len(mount.transformer.blocks) == 3

    torch.manual_seed(1)
    config = mount.transformer.config
    ids = torch.randint(0, config.total_vocab_size, (2, 6))
    full_sink: dict[int, torch.Tensor] = {}
    mount_sink: dict[int, torch.Tensor] = {}
    full(ids, residual_taps=range(3), residual_sink=full_sink)
    mount(ids, residual_taps=range(3), residual_sink=mount_sink)
    for tap in range(3):
        assert torch.equal(full_sink[tap], mount_sink[tap])


def test_truncated_config_bounds() -> None:
    config = tiny_text_config()
    assert truncated_config(config, 1).num_hidden_layers == 1
    assert (
        truncated_config(config, config.num_hidden_layers).num_hidden_layers
        == config.num_hidden_layers
    )
    with pytest.raises(ValueError, match="num_layers"):
        truncated_config(config, 0)
    with pytest.raises(ValueError, match="num_layers"):
        truncated_config(config, config.num_hidden_layers + 1)


def test_padding_orientation_invariance() -> None:
    torch.manual_seed(0)
    config = tiny_text_config()
    model = Molmo2Transformer(config, attn_backend=AttentionBackend.EAGER)
    real_len, pad = 4, 2
    total = real_len + pad
    ids = torch.randint(0, config.vocab_size, (1, real_len))
    pad_ids = torch.zeros((1, pad), dtype=ids.dtype)
    positions = torch.arange(real_len).unsqueeze(0)

    out_right = model(
        torch.cat([ids, pad_ids], dim=1),
        position_ids=torch.cat([positions, torch.zeros_like(pad_ids)], dim=1),
        padding_mask=torch.tensor([[True] * real_len + [False] * pad]),
    )
    out_left = model(
        torch.cat([pad_ids, ids], dim=1),
        position_ids=torch.cat([torch.zeros_like(pad_ids), positions], dim=1),
        padding_mask=torch.tensor([[False] * pad + [True] * real_len]),
    )
    torch.testing.assert_close(
        out_right[:, :real_len],
        out_left[:, pad:],
        rtol=1e-5,
        atol=1e-5,
    )
    assert out_right.shape == (1, total, config.hidden_size)


def test_lm_head_covers_base_vocab_only() -> None:
    torch.manual_seed(0)
    config = tiny_text_config()
    model = Molmo2TextModel(config, lm_head=True)
    ids = torch.randint(0, config.total_vocab_size, (1, 3))
    logits = model(ids)
    assert logits.shape == (1, 3, config.vocab_size)
    # Headless (mount) forward returns hidden states.
    headless = Molmo2TextModel(config, lm_head=False)
    assert headless(ids).shape == (1, 3, config.hidden_size)


def test_fused_projection_split_convention() -> None:
    """The fused QKV split order (q, k, v) and the fused MLP chunk order
    (up-multiplicand first, gate second) are the two silent-corruption
    hazards of the fused layout — pin them against a manual computation."""
    torch.manual_seed(0)
    config = tiny_text_config()
    model = Molmo2Transformer(config)
    block = model.blocks[0]
    assert isinstance(block, DecoderLayer)
    x = torch.randn(1, 3, config.hidden_size)

    mlp = block.mlp
    up, gate = mlp.ff_proj(x).chunk(2, dim=-1)
    expected = mlp.ff_out(F.silu(gate) * up)
    torch.testing.assert_close(mlp(x), expected)

    attn = block.self_attn
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    assert attn.fused_dims == (q_dim, kv_dim, kv_dim)
    assert attn.att_proj.weight.shape == (q_dim + 2 * kv_dim, config.hidden_size)
    assert attn.scaling == pytest.approx(config.head_dim**-0.5)


def test_embedding_row_select_matches_cat_lookup() -> None:
    """The perf rewrite (no full-table cat per call) must be bitwise the
    reference semantics: rows are selected, never computed."""
    torch.manual_seed(11)
    embedding = Molmo2Embedding(16, 4, 8)
    ids = torch.tensor([[0, 15, 16, 19, 3], [19, 0, 1, 17, 15]])
    reference = torch.nn.functional.embedding(
        ids,
        torch.cat([embedding.embedding, embedding.new_embedding], dim=0),
    )
    assert torch.equal(embedding(ids), reference)
