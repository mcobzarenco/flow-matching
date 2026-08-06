"""Residual-stream conditioning oracles (arch-batch-1 arm B pre-reg,
2026-08-06) — the five pre-launch gates, CPU-only on the tiny
structurally-E2B-faithful backbone:

(i)   the adapters' streams obey the SAME contract as backbone K/V
      exports — shapes [B, kv_heads, P, head_dim] and logical-position
      RoPE (asserted via padding-orientation invariance, the property
      the kv streams are gated on);
(ii)  the trunk stays bitwise-frozen through a real optimizer step;
(iii) gradients reach EVERY adapter (all taps, every parameter);
(iv)  the config round-trips through the checkpoint schema with no
      flags (and the weights strict-load into the rebuilt decoder);
(v)   the K/V code path is unaffected — no adapter keys in a K/V
      expert's state_dict, no raw taps in its memory (the banked loss
      oracles and the state-dict-keys fixture gate the rest).
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch
from torch import Tensor

from bijou.decoders.flow import (
    ExpertConfig,
    FlowDecoder,
    ResidualStreamAdapter,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.encoders.gemma4 import GemmaEncoder
from bijou.gemma4.config import (
    Gemma4Config,
    Gemma4TextConfig,
    LayerType,
    RopeParameters,
    RopeType,
)
from bijou.gemma4.model import Gemma4Model
from bijou.interface import ObservationMemory
from bijou.loading import (
    FlowDecoderConfig,
    GemmaPromptConfig,
    expert_config_from_architecture,
    flow_decoder_config_from_expert,
    parse_decoder_config,
    residual_expert_config,
)
from bijou.model import BijouModel
from bijou.nn import AttentionBackend, rope_cos_sin, rope_inv_freq_from_params

BATCH, CHUNK, ACTION_DIM, STATE_DIM = 2, 4, 6, 6
PROMPT_LENGTHS = (13, 7)


def tiny_text_config() -> Gemma4TextConfig:
    """Structurally E2B-faithful, minimally sized (the
    test_backbone_continuation fixture): 8 layers, KV sharing over the
    last 2 — prefix depth 6, global prefix layers {1, 3, 5}."""
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        pad_token_id=0,
        eos_token_ids=(1,),
        bos_token_id=2,
        tie_word_embeddings=True,
        attention_bias=False,
        sliding_window=8,
        layer_types=(LayerType.SLIDING, LayerType.FULL) * 4,
        final_logit_softcapping=30.0,
        use_bidirectional_attention=None,
        rope_parameters={
            LayerType.SLIDING: RopeParameters(
                rope_type=RopeType.DEFAULT,
                rope_theta=10_000.0,
                factor=1.0,
                partial_rotary_factor=1.0,
            ),
            LayerType.FULL: RopeParameters(
                rope_type=RopeType.PROPORTIONAL,
                rope_theta=1_000_000.0,
                factor=1.0,
                partial_rotary_factor=0.25,
            ),
        },
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=4,
        global_head_dim=16,
        num_global_key_value_heads=None,
        attention_k_eq_v=False,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
        enable_moe_block=False,
    )


def tiny_gemma4_config() -> Gemma4Config:
    return Gemma4Config(
        text=tiny_text_config(),
        vision=None,
        image_token_id=999,
        video_token_id=998,
        audio_token_id=997,
        boi_token_id=996,
        eoi_token_id=995,
        dtype=torch.float32,
    )


def residual_config() -> ExpertConfig:
    return residual_expert_config(
        tiny_gemma4_config(),
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        cross_attention_heads=2,
        chunk_size=CHUNK,
        time_embed_dim=8,
        time_conditioning=TimeConditioning.ADARMS,
    )


PREFIX_DEPTH = 6  # first_kv_shared_layer_idx of the tiny config


def build_residual_model() -> BijouModel:
    torch.manual_seed(0)
    config = tiny_gemma4_config()
    backbone = Gemma4Model(config, attn_backend=AttentionBackend.EAGER)
    backbone.eval()
    backbone.requires_grad_(False)
    expert_config = residual_config()
    encoder = GemmaEncoder(
        config,
        exports=(),
        residual_exports=expert_config.streams,
        processor_dir="unused",
        max_soft_tokens=1,
        state_dim=STATE_DIM,
    )
    decoder = FlowDecoder(expert_config, device="cpu", dtype=torch.float32)
    return BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)


def prompt_ids(length: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(3, 64, (length,), generator=generator)


def padded_batch(*, pad_left: bool) -> tuple[Tensor, Tensor]:
    prompts = [prompt_ids(length, seed=i) for i, length in enumerate(PROMPT_LENGTHS)]
    width = max(int(p.shape[0]) for p in prompts)
    ids = torch.zeros((len(prompts), width), dtype=torch.long)
    real = torch.zeros((len(prompts), width), dtype=torch.bool)
    for i, prompt in enumerate(prompts):
        length = int(prompt.shape[0])
        span = slice(width - length, width) if pad_left else slice(0, length)
        ids[i, span] = prompt
        real[i, span] = True
    return ids, real


# -- (i) stream contract -----------------------------------------------------


def test_residual_streams_match_kv_stream_geometry() -> None:
    """Attached residual streams carry the exact K/V-export contract:
    kv_heads/head_dim from the backbone's global layers, one column per
    prompt token, raw taps consumed and dropped."""
    model = build_residual_model()
    ids, real = padded_batch(pad_left=True)
    with torch.no_grad():
        memory = model.encode_observation(ids, padding_mask=real)
    assert memory.residuals is None  # attached and dropped
    assert sorted(memory.streams) == [f"res{i}" for i in range(PREFIX_DEPTH)]
    text = tiny_text_config()
    kv_heads = text.num_global_key_value_heads or text.num_key_value_heads
    for stream in memory.streams.values():
        assert stream.key.shape == (BATCH, kv_heads, ids.shape[1], text.global_head_dim)
        assert stream.value.shape == stream.key.shape


def test_residual_keys_are_padding_orientation_invariant() -> None:
    """The kv streams' load-bearing position property, applied to the
    adapters: K/V at REAL token columns must not depend on padding
    orientation — true only if keys are RoPE'd at per-sample LOGICAL
    positions exactly like the backbone's own K/V exports."""
    model = build_residual_model()

    def encode(*, pad_left: bool) -> list[Tensor]:
        ids, real = padded_batch(pad_left=pad_left)
        with torch.no_grad():
            memory = model.encode_observation(ids, padding_mask=real)
        return [
            torch.cat(
                [
                    stream.key[i][:, real[i], :],
                    stream.value[i][:, real[i], :],
                ],
            )
            for stream in memory.streams.values()
            for i in range(len(PROMPT_LENGTHS))
        ]

    left = encode(pad_left=True)
    right = encode(pad_left=False)
    for a, b in zip(left, right, strict=True):
        delta = float((a - b).abs().max())
        assert delta < 1e-4, f"orientation-dependent residual K/V, max|Δ|={delta}"


def test_adapter_stream_is_contract_shaped() -> None:
    adapter = ResidualStreamAdapter(
        stream_dim=32,
        kv_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
    )
    hidden = torch.randn(BATCH, 5, 32)
    positions = torch.arange(5)[None, :]
    inv_freq = rope_inv_freq_from_params(
        tiny_text_config().rope_parameters[LayerType.FULL],
        16,
    )
    stream = adapter(hidden, rope_cos_sin(inv_freq, positions, torch.float32))
    assert stream.key.shape == (BATCH, 1, 5, 16)
    assert stream.value.shape == (BATCH, 1, 5, 16)


# -- (ii) trunk bitwise-frozen + (iii) grads reach every adapter -------------


def test_train_step_freezes_trunk_and_reaches_every_adapter() -> None:
    model = build_residual_model()
    decoder = model.decoder
    assert isinstance(decoder, FlowDecoder)
    # At TRUE init the zero-initialized heads (action_out_proj, the adaRMS
    # gates) make every upstream gradient exactly zero — the identity-at-
    # init property, not a wiring fault. Perturb them to a mid-training
    # state so the gradient PATH to the adapters is what's tested.
    torch.manual_seed(7)
    torch.nn.init.normal_(decoder.action_out_proj.weight, std=0.02)
    for module in decoder.modules():
        if isinstance(module, torch.nn.Linear) and bool((module.weight == 0).all()):
            torch.nn.init.normal_(module.weight, std=0.02)

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.backbone.named_parameters()
    }
    optimizer = torch.optim.SGD(
        [p for p in decoder.parameters() if p.requires_grad],
        lr=1e-2,
    )
    ids, real = padded_batch(pad_left=True)
    memory = model.encode_observation(ids, padding_mask=real)
    generator = torch.Generator().manual_seed(3)
    state = torch.randn(BATCH, STATE_DIM, generator=generator)
    noisy = torch.randn(BATCH, CHUNK, ACTION_DIM, generator=generator)
    time = torch.rand(BATCH, generator=generator)
    loss = decoder(memory, state, noisy, time).square().mean()
    loss.backward()

    assert decoder.res_adapters is not None
    assert len(decoder.res_adapters) == PREFIX_DEPTH
    for name, adapter in decoder.res_adapters.items():
        for parameter_name, parameter in adapter.named_parameters():
            grad = parameter.grad
            assert grad is not None, f"{name}.{parameter_name}: no gradient"
            assert float(grad.abs().max()) > 0, (
                f"{name}.{parameter_name}: zero gradient — the adapter is "
                "not on the loss path"
            )

    optimizer.step()
    for name, parameter in model.backbone.named_parameters():
        assert not parameter.requires_grad, f"trunk parameter {name} trainable"
        assert parameter.grad is None, f"trunk parameter {name} has a gradient"
        assert torch.equal(parameter, before[name]), (
            f"trunk parameter {name} drifted through the step"
        )


# -- (iv) config round-trip --------------------------------------------------


def test_residual_config_roundtrips_through_checkpoint_schema() -> None:
    expert_config = residual_config()
    prompt = GemmaPromptConfig(
        exports=(),
        residual_exports=expert_config.streams,
        max_soft_tokens=1,
        format=3,
        state_dim=STATE_DIM,
        condition_fields=(),
        generate_bracket=False,
    )
    decoder_schema = flow_decoder_config_from_expert(expert_config)
    assert list(decoder_schema.schedule) == [f"res{i}" for i in range(PREFIX_DEPTH)]
    # Through JSON, as a checkpoint would store it.
    prompt_back = GemmaPromptConfig.from_dict(
        json.loads(json.dumps(prompt.to_dict())),
    )
    decoder_back = parse_decoder_config(
        json.loads(json.dumps(decoder_schema.to_dict())),
    )
    assert isinstance(decoder_back, FlowDecoderConfig)
    assert prompt_back == prompt
    rebuilt = expert_config_from_architecture(
        prompt_back,
        decoder_back,
        tiny_gemma4_config(),
    )
    assert rebuilt == expert_config
    # "Eval loads it with no flags": the rebuilt config constructs a
    # decoder whose state_dict strict-loads the trained one's weights.
    torch.manual_seed(1)
    trained = FlowDecoder(residual_config(), device="cpu", dtype=torch.float32)
    fresh = FlowDecoder(rebuilt, device="cpu", dtype=torch.float32)
    fresh.load_state_dict(trained.state_dict(), strict=True)


def test_prompt_config_without_residual_field_defaults_empty() -> None:
    """Checkpoints written before the field existed load unchanged."""
    prompt = GemmaPromptConfig(
        exports=(1, 3, 5),
        max_soft_tokens=1,
        format=3,
        state_dim=STATE_DIM,
        condition_fields=(),
        generate_bracket=False,
    )
    data = prompt.to_dict()
    del data["residual_exports"]
    assert GemmaPromptConfig.from_dict(data).residual_exports == ()


def test_mixed_kind_schedule_is_rejected() -> None:
    expert_config = residual_config()
    prompt = GemmaPromptConfig(
        exports=(5,),
        residual_exports=(0, 1, 2, 3, 4),
        max_soft_tokens=1,
        format=3,
        state_dim=STATE_DIM,
        condition_fields=(),
        generate_bracket=False,
    )
    decoder_schema = flow_decoder_config_from_expert(expert_config)
    mixed = type(decoder_schema).from_dict(
        {
            **decoder_schema.to_dict(),
            "schedule": ["kv5", "res0", "res1", "res2", "res3", "res4"],
        },
    )
    with pytest.raises(SystemExit, match="mixes"):
        expert_config_from_architecture(prompt, mixed, tiny_gemma4_config())


def test_residual_config_requires_adapter_geometry() -> None:
    base = residual_config()
    with pytest.raises(ValueError, match="residual_stream_dim"):
        dataclasses.replace(base, residual_stream_dim=None)


# -- (v) the K/V path is unaffected ------------------------------------------


def test_kv_expert_state_dict_carries_no_adapters() -> None:
    config = ExpertConfig(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
        self_attention_rope_theta=10_000.0,
        cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_rope=tiny_text_config().rope_parameters[LayerType.FULL],
        cross_attention_schedule=(1, 3, 5),
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        chunk_size=CHUNK,
        time_embed_dim=8,
        time_conditioning=TimeConditioning.ADARMS,
    )
    decoder = FlowDecoder(config, device="cpu", dtype=torch.float32)
    assert decoder.res_adapters is None
    assert not any(k.startswith("res_adapters") for k in decoder.state_dict())
    # And its attach is the identity on K/V memories.
    memory = ObservationMemory(streams={}, length=0, padding_mask=None)
    assert decoder.attach_residual_streams(memory) is memory


def test_kv_encoder_produces_no_raw_taps() -> None:
    config = tiny_gemma4_config()
    torch.manual_seed(0)
    backbone = Gemma4Model(config, attn_backend=AttentionBackend.EAGER)
    backbone.eval()
    backbone.requires_grad_(False)
    encoder = GemmaEncoder(
        config,
        exports=(1, 3, 5),
        processor_dir="unused",
        max_soft_tokens=1,
        state_dim=STATE_DIM,
    )
    ids, real = padded_batch(pad_left=True)
    with torch.no_grad():
        memory = encoder.encode_tensors(backbone, ids, padding_mask=real)
    assert memory.residuals is None
    assert sorted(memory.streams) == ["kv1", "kv3", "kv5"]


def test_attach_guards_fire() -> None:
    model = build_residual_model()
    decoder = model.decoder
    assert isinstance(decoder, FlowDecoder)
    ids, real = padded_batch(pad_left=True)
    with torch.no_grad():
        memory = model.encode_observation(ids, padding_mask=real)
        # Attached once inside encode_observation; a second attach must
        # refuse rather than silently re-project.
        with pytest.raises(ValueError, match="has none"):
            decoder.attach_residual_streams(memory)
