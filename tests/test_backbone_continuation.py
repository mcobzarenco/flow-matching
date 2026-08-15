"""Prefill-then-continue == monolithic forward: the ar_backbone keystone.

The decoder-only (ar_backbone) path prefill-encodes the prompt once
(layers 0..first_kv_shared−1 via ``kv_stop_layer``) and then runs suffix
tokens through ALL layers against the cache. These tests pin the facts
that path rests on, using a tiny in-memory text-only model (vocab 64,
hidden 32 — milliseconds, no files):

1. **Dead deep half**: the monolithic reference runs every layer over
   the prompt; the split path never runs layers ≥ first_kv_shared on
   prompt positions — the suffix hidden states must match anyway (no
   K/V weights above the sharing horizon + causality + no prompt loss).
2. **LEFT-padded batching is sample-independent**: with left-padded
   prompts and per-sample logical ``position_ids``, each sample's
   suffix hiddens equal its OWN unpadded monolithic forward — and are
   invariant to batch composition.
3. **RIGHT padding is structurally wrong for suffix continuation**:
   sliding-window masks work in PHYSICAL index space
   (``masks._build_mask``), so the pad gap between a short sample's
   prompt and the suffix sits inside the window and evicts real prompt
   tokens from view. Asserted to differ, as executable documentation of
   why the ar_backbone prompt must collate left-padded — unlike the
   shipped cross-attention paths, where right padding is correct (the
   suffix never enters the backbone there) and stays untouched.

Exactness: bitwise equality is NOT achievable across the split — the
split and monolithic paths run different GEMM shapes, so fp32
reduction order differs (the repo already records the same effect for
fused-SDPA vs additive-mask paths). Measured on this fixture
(2026-08-01, torch CPU, both backends): split-vs-monolithic max|Δ| ≤
5.8e-6 on hiddens of magnitude ~2.5; batch-composition max|Δ| ≤ 4e-6;
the right-padding failure mode measures max|Δ| = 1.216 — six orders
above the noise. ATOL below is set at 1e-4: two orders above measured
noise, four below the failure signal. A failure here is semantic, not
numeric.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from bijou.modelling.encoders.gemma4 import GemmaEncoder
from bijou.modelling.gemma4.cache import KVCache
from bijou.modelling.gemma4.config import (
    Gemma4Config,
    Gemma4TextConfig,
    LayerType,
    RopeParameters,
    RopeType,
)
from bijou.modelling.gemma4.model import Gemma4Model
from bijou.modelling.gemma4.text import TextModel
from bijou.modelling.nn import AttentionBackend

# Window 8 with prompt lengths straddling it: the long prompt exceeds the
# window (exercising the sliding cache trim + kv_offset path), the short
# one stays under it, and the pad gap (6) is wide enough that the
# right-padding failure mode is decisive, not marginal.
SLIDING_WINDOW = 8
PROMPT_LENGTHS = (13, 7)
SUFFIX_LENGTH = 5


def tiny_text_config() -> Gemma4TextConfig:
    """Structurally E2B-faithful, minimally sized: hybrid sliding/full
    layers, KV sharing over the last 2 of 8 (sources = layers 4/5),
    double-wide shared MLPs, p-RoPE globals, tiny 64-token vocab (no
    processor involved, so the vocab need not match any tokenizer)."""
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
        sliding_window=SLIDING_WINDOW,
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


def build_model(backend: AttentionBackend) -> TextModel:
    torch.manual_seed(0)
    model = TextModel(tiny_text_config(), attn_backend=backend)
    model.eval()
    model.requires_grad_(False)
    return model


def prompt_ids(length: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    # ids in [3, vocab): keep pad/eos/bos out of real content.
    return torch.randint(3, 64, (length,), generator=generator)


def suffix_ids(batch: int) -> Tensor:
    generator = torch.Generator().manual_seed(99)
    return torch.randint(3, 64, (batch, SUFFIX_LENGTH), generator=generator)


@torch.no_grad()
def monolithic_suffix_hiddens(
    model: TextModel,
    prompt: Tensor,
    suffix: Tensor,
) -> Tensor:
    """The gold reference: one unpadded [1, L+S] forward through ALL
    layers (deep half included, over the prompt too); final-normed
    hidden states of the suffix positions, [S, hidden]."""
    ids = torch.cat([prompt, suffix])[None, :]
    hidden = model(input_ids=ids)
    return hidden[0, prompt.shape[0] :, :]


@torch.no_grad()
def split_suffix_hiddens(
    model: TextModel,
    prompts: list[Tensor],
    suffixes: Tensor,
    *,
    pad_left: bool,
) -> Tensor:
    """The ar_backbone computation: batched padded prefill to the KV
    horizon (``kv_stop_layer``; deep half never runs on the prompt),
    then one suffix forward through all layers against the cache, with
    per-sample logical positions. Returns [B, S, hidden]."""
    config = model.config
    stop_layer = config.first_kv_shared_layer_idx - 1
    batch = len(prompts)
    width = max(int(p.shape[0]) for p in prompts)
    ids = torch.full((batch, width), config.pad_token_id, dtype=torch.long)
    real = torch.zeros((batch, width), dtype=torch.bool)
    positions = torch.zeros((batch, width), dtype=torch.long)
    for i, prompt in enumerate(prompts):
        length = int(prompt.shape[0])
        span = slice(width - length, width) if pad_left else slice(0, length)
        ids[i, span] = prompt
        real[i, span] = True
        positions[i, span] = torch.arange(length)
    cache = KVCache(config)
    model(
        input_ids=ids,
        position_ids=positions,
        padding_mask=real,
        cache=cache,
        kv_stop_layer=stop_layer,
    )
    lengths = torch.tensor([int(p.shape[0]) for p in prompts])
    suffix_positions = lengths[:, None] + torch.arange(suffixes.shape[1])[None, :]
    full_mask = torch.cat(
        [real, torch.ones((batch, suffixes.shape[1]), dtype=torch.bool)],
        dim=1,
    )
    return model(
        input_ids=suffixes,
        position_ids=suffix_positions,
        padding_mask=full_mask,
        cache=cache,
    )


BACKENDS = [AttentionBackend.EAGER, AttentionBackend.SDPA]


@pytest.mark.parametrize("backend", BACKENDS)
def test_split_matches_monolithic_unpadded_reference(
    backend: AttentionBackend,
) -> None:
    model = build_model(backend)
    prompts = [prompt_ids(length, seed=i) for i, length in enumerate(PROMPT_LENGTHS)]
    suffixes = suffix_ids(len(prompts))
    split = split_suffix_hiddens(model, prompts, suffixes, pad_left=True)
    for i, prompt in enumerate(prompts):
        reference = monolithic_suffix_hiddens(model, prompt, suffixes[i])
        delta = float((split[i] - reference).abs().max())
        assert delta < 1e-4, (
            f"sample {i} (L={prompt.shape[0]}, backend={backend}): "
            f"max|Δ|={delta} — semantic, not reduction-order noise"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_composition_invariance(backend: AttentionBackend) -> None:
    """A sample's suffix hiddens must not depend on its batch-mates —
    the class of bug the flow decoder's padding-position fix closed
    (measured max|Δ| 0.55 there)."""
    model = build_model(backend)
    prompts = [prompt_ids(length, seed=i) for i, length in enumerate(PROMPT_LENGTHS)]
    suffixes = suffix_ids(len(prompts))
    batched = split_suffix_hiddens(model, prompts, suffixes, pad_left=True)
    for i, prompt in enumerate(prompts):
        alone = split_suffix_hiddens(
            model,
            [prompt],
            suffixes[i : i + 1],
            pad_left=True,
        )
        delta = float((batched[i] - alone[0]).abs().max())
        assert delta < 1e-4, f"sample {i}: batch-dependent by {delta}"


def tiny_gemma4_config() -> Gemma4Config:
    """Text-only wrapper for the encoder-path test (vision None; the
    image token id is outside the tiny vocab so no prompt id ever
    matches it)."""
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


def test_encode_is_padding_orientation_invariant() -> None:
    """What every EXISTING checkpoint depends on: the exported-stream
    K/V at real-token columns are identical (≤ reduction-order noise)
    whether the prompt batch is right-padded (the historical collation),
    left-padded (the current one), or not padded at all — through the
    real GemmaEncoder.encode_tensors path with per-sample logical
    position_ids. This is the executable form of the padding-impact
    analysis: the 2026-08-01 left-padding switch changes nothing for
    stream consumers."""
    config = tiny_gemma4_config()
    torch.manual_seed(0)
    backbone = Gemma4Model(config, attn_backend=AttentionBackend.EAGER)
    backbone.eval()
    backbone.requires_grad_(False)
    stop = config.text.first_kv_shared_layer_idx - 1
    encoder = GemmaEncoder(
        config,
        exports=(stop,),
        processor_dir="unused",
        max_soft_tokens=1,
        state_dim=6,
    )
    prompts = [prompt_ids(length, seed=i) for i, length in enumerate(PROMPT_LENGTHS)]
    width = max(int(p.shape[0]) for p in prompts)

    def encode(*, pad_left: bool) -> list[Tensor]:
        ids = torch.full((len(prompts), width), 0, dtype=torch.long)
        real = torch.zeros((len(prompts), width), dtype=torch.bool)
        for i, prompt in enumerate(prompts):
            length = int(prompt.shape[0])
            span = slice(width - length, width) if pad_left else slice(0, length)
            ids[i, span] = prompt
            real[i, span] = True
        with torch.no_grad():
            memory = encoder.encode_tensors(backbone, ids, padding_mask=real)
        stream = memory.streams[f"kv{stop}"]
        # Per-sample real-token columns, in logical order ([2·kv_heads,
        # L_i, head_dim] each — ragged across the batch).
        return [
            torch.cat([stream.key[i][:, real[i], :], stream.value[i][:, real[i], :]])
            for i in range(len(prompts))
        ]

    left = encode(pad_left=True)
    right = encode(pad_left=False)
    for i, prompt in enumerate(prompts):
        delta = float((left[i] - right[i]).abs().max())
        assert delta < 1e-4, f"sample {i}: orientation-dependent K/V, max|Δ|={delta}"
        # And both equal the unpadded single-sample encode.
        with torch.no_grad():
            alone = encoder.encode_tensors(backbone, prompt[None, :])
        stream = alone.streams[f"kv{stop}"]
        reference = torch.cat([stream.key[0], stream.value[0]])
        delta = float((left[i] - reference).abs().max())
        assert delta < 1e-4, f"sample {i}: padded vs unpadded max|Δ|={delta}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_right_padding_breaks_sliding_semantics(
    backend: AttentionBackend,
) -> None:
    """Executable documentation: with RIGHT padding the short sample's
    pad gap sits inside the sliding window at suffix positions, evicting
    real prompt tokens — per-sample position_ids cannot fix physical
    window geometry. This is why the ar_backbone prompt collates
    left-padded."""
    model = build_model(backend)
    prompts = [prompt_ids(length, seed=i) for i, length in enumerate(PROMPT_LENGTHS)]
    suffixes = suffix_ids(len(prompts))
    split = split_suffix_hiddens(model, prompts, suffixes, pad_left=False)
    # The unpadded (long) sample is unaffected by the batch layout...
    reference_long = monolithic_suffix_hiddens(model, prompts[0], suffixes[0])
    assert float((split[0] - reference_long).abs().max()) < 1e-4
    # ...the padded (short) one sees a truncated prompt through the
    # window and diverges decisively (measured max|Δ| = 1.216).
    reference_short = monolithic_suffix_hiddens(model, prompts[1], suffixes[1])
    assert float((split[1] - reference_short).abs().max()) > 0.1
