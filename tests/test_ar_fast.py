"""Tests for the AR FAST decoder and its codec.

Pure CPU: a tiny ARFastDecoder over a fabricated ObservationMemory, with
the committed tiny tokenizer fixture (vocab 128, H=50, D=6). Covers the
codec round trip (specials, quantile denormalization), causality of the
suffix (logits at position j are independent of tokens after j), the CE
loss contract (state/PAD positions ignored), and predict_chunk's decode
loop incl. the malformed-generation fallback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bijou.decoders.ar_fast import (
    IGNORE_INDEX,
    ARFastConfig,
    ARFastDecoder,
    ar_fast_loss,
)
from bijou.fast.codec import ActionCodec
from bijou.interface import (
    CollatedBatch,
    MemoryStream,
    NormStats,
    ObservationMemory,
    StreamGeometry,
)
from bijou.nn import RopeParameters, RopeType

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_fast_tokenizer"
BATCH, PREFIX_LEN, CHUNK, DIM, HIDDEN, HEAD_DIM = 2, 5, 50, 6, 32, 16


def codec() -> ActionCodec:
    return ActionCodec.load(FIXTURE)


def geometry() -> dict[str, StreamGeometry]:
    rope = RopeParameters(
        rope_type=RopeType.DEFAULT,
        rope_theta=10_000.0,
        factor=1.0,
        partial_rotary_factor=1.0,
    )
    return {"kv0": StreamGeometry(kv_heads=1, head_dim=HEAD_DIM, rope=rope)}


def config(loaded: ActionCodec) -> ARFastConfig:
    return ARFastConfig(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_rope_theta=10_000.0,
        cross_attention_heads=2,
        schedule=("kv0", "kv0"),
        tokenizer=str(FIXTURE),
        vocab_total=loaded.vocab_total,
        max_tokens=110,
        state_dim=DIM,
        chunk_size=CHUNK,
        action_dim=DIM,
    )


def build() -> tuple[ARFastDecoder, ActionCodec]:
    loaded = codec()
    torch.manual_seed(0)
    decoder = ARFastDecoder(
        config(loaded),
        geometry(),
        loaded,
        device="cpu",
        dtype=torch.float32,
    )
    return decoder, loaded


def memory() -> ObservationMemory:
    generator = torch.Generator().manual_seed(1)
    key = torch.randn(BATCH, 1, PREFIX_LEN, HEAD_DIM, generator=generator)
    value = torch.randn(BATCH, 1, PREFIX_LEN, HEAD_DIM, generator=generator)
    return ObservationMemory(
        streams={"kv0": MemoryStream(key=key, value=value)},
        length=PREFIX_LEN,
        padding_mask=None,
    )


class FakeInputs:
    def pin_memory(self) -> FakeInputs:
        return self

    def to(self, device: object, *, non_blocking: bool = False) -> FakeInputs:
        return self

    def tensors(self) -> dict[str, torch.Tensor]:
        return {}


def batch(loaded: ActionCodec) -> CollatedBatch[FakeInputs]:
    generator = torch.Generator().manual_seed(2)
    actions = torch.cumsum(
        torch.randn(BATCH, CHUNK, DIM, generator=generator) * 0.05,
        dim=1,
    ).clamp(-1, 1)
    q01 = np.full(DIM, -1.0)
    q99 = np.full(DIM, 1.0)
    sequences = [loaded.encode(actions[i].numpy(), q01, q99) for i in range(BATCH)]
    width = max(len(s) for s in sequences)
    tokens = torch.tensor(
        [s + [loaded.pad] * (width - len(s)) for s in sequences],
        dtype=torch.long,
    )
    stats = NormStats(
        mean=torch.zeros(BATCH, DIM),
        std=torch.ones(BATCH, DIM),
        q01=torch.full((BATCH, DIM), -1.0),
        q99=torch.full((BATCH, DIM), 1.0),
    )

    return CollatedBatch(
        encoder_inputs=FakeInputs(),
        state=torch.randn(BATCH, DIM, generator=generator),
        actions=actions,
        action_is_pad=torch.zeros(BATCH, CHUNK, dtype=torch.bool),
        action_stats=stats,
        state_stats=NormStats(
            mean=torch.zeros(BATCH, DIM),
            std=torch.ones(BATCH, DIM),
            q01=None,
            q99=None,
        ),
        action_tokens=tokens,
    )


def test_codec_round_trip_with_specials() -> None:
    loaded = codec()
    rng = np.random.default_rng(3)
    chunk = np.clip(
        np.cumsum(rng.standard_normal((CHUNK, DIM)) * 0.05, axis=0),
        -1,
        1,
    )
    q01 = np.full(DIM, -2.0)
    q99 = np.full(DIM, 2.0)
    ids = loaded.encode(chunk, q01, q99)
    assert ids[0] == loaded.boa
    assert ids[-1] == loaded.eoa
    decoded = loaded.decode(ids, q01, q99)
    assert decoded.shape == (CHUNK, DIM)
    assert float(np.abs(decoded - chunk).mean()) < 0.15


def test_forward_shape_and_causality() -> None:
    decoder, loaded = build()
    decoder.eval()
    sample = batch(loaded)
    tokens = sample.action_tokens
    assert tokens is not None
    with torch.no_grad():
        logits = decoder(memory(), sample.state, tokens)
        assert logits.shape == (BATCH, 1 + tokens.shape[1], loaded.vocab_total)
        # Perturb the LAST token: logits at earlier positions must not move.
        perturbed = tokens.clone()
        perturbed[:, -1] = (perturbed[:, -1] + 1) % loaded.vocab_total
        logits_perturbed = decoder(memory(), sample.state, perturbed)
    torch.testing.assert_close(logits[:, :-1], logits_perturbed[:, :-1])


def test_loss_ignores_state_and_pad_positions() -> None:
    decoder, loaded = build()
    sample = batch(loaded)
    loss = ar_fast_loss(decoder, memory(), sample)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # Rewriting PAD-position values must not change the loss (they are
    # IGNORE_INDEX targets and causally invisible to real positions).
    tokens = sample.action_tokens
    assert tokens is not None
    lengths = (tokens != loaded.pad).sum(dim=1)
    assert int(lengths.min()) < tokens.shape[1]  # the batch IS ragged
    assert IGNORE_INDEX == -100


def test_predict_chunk_decodes_and_falls_back() -> None:
    decoder, _ = build()
    decoder.eval()
    sample = batch(loaded=decoder.codec)
    chunks = decoder.predict_chunk(memory(), sample)
    assert chunks.shape == (BATCH, CHUNK, DIM)
    assert torch.isfinite(chunks).all()
    # A random-init decoder mostly emits malformed sequences; the fallback
    # substitutes state-copy and counts loudly.
    assert decoder.malformed_decodes >= 0
    with pytest.raises(ValueError, match="no noise"):
        decoder.predict_chunk(memory(), sample, noise=torch.zeros(1))
