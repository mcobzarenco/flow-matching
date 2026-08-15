"""Token-GRPO training-rows instrument oracles (design memo 2026-08-13
§8 item 1), pure CPU: the TokenRow capture surface + TrainingRowWriter
behind ``--emit-training-rows``.

What these pin: (1) capture is observation, never intervention — a
greedy decode with the surface on returns bit-identical actions;
(2) the item-1 oracle — emitted greedy logprobs against a
teacher-forced re-forward under the recorded mask; (3) the recorded
grammar mask is a pure function of the id prefix (budget arithmetic
over symbol lengths), so the trainer can recompute it from ids alone
and must land bit-for-bit on the packbits surface — the row side of
item 2's "train-time mask == rollout mask"; (4) sampled rows record
exactly the stream the chunk decoded from, reproducible under their
keys; (5) the block-column reduction equals the sampler's full-vocab
distribution (exp(-inf) terms are exact zeros); (6) the NPZ writer
round-trips every array bit-exact; (7) the loud guards.

MEASURED AMENDMENT to the memo's draft bar: §8 drafted the greedy
oracle as "bit-for-bit against a teacher-forced re-forward". The
re-forward is a one-shot batched trunk forward while the decode fed
its cache one token at a time, so the trunk logits carry
reduction-shape noise (~1e-4 on this fixture —
test_teacher_forced_matches_incremental_decode_path's bound), which
the log-softmax squeezes to ~2.4e-6 on chosen logprobs. What IS
bit-for-bit is the masked softmax itself: reducing the captured
logits under the recorded mask reproduces the emitted logprobs
exactly. Both halves are asserted below; the re-forward bound is
1e-5. Fixture family: tests/test_ar_backbone."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ar_backbone import BATCH, batch, build, encode_memory

from bijou.eval.policies import (
    BijouPolicy,
    TokenRow,
    stable_sample_rng,
    token_rows_from_capture,
)
from bijou.modelling.decoders.ar_gemma import GemmaARDecoder
from bijou.modelling.encoders.gemma4 import GemmaMemory
from bijou.modelling.gemma4.model import Gemma4Model
from bijou.modelling.interface import ActionCaptureStep, ARSampling
from sim.rollout_sim_parallel import TrainingRowWriter
from sim.rollout_sim_parallel import parse_args as rollout_parse_args


def rngs(draw: int) -> tuple[np.random.Generator, ...]:
    """One keyed stream per fixture row, the way eval builds them."""
    return tuple(
        stable_sample_rng(0, "fixture/repo", 0, row, draw) for row in range(BATCH)
    )


def unpack(row: TokenRow) -> torch.Tensor:
    """[T, vocab_total] bool grammar masks off the packbits surface."""
    return torch.from_numpy(
        np.unpackbits(row.allowed_packed, axis=1, count=row.vocab_total).astype(bool),
    )


def greedy_rows() -> tuple[
    Gemma4Model,
    GemmaARDecoder,
    GemmaMemory,
    list[ActionCaptureStep],
    list[TokenRow],
]:
    backbone, decoder, loaded = build()
    memory = encode_memory(backbone)
    snapshot = decoder.cache_snapshot(memory)
    capture: list[ActionCaptureStep] = []
    decoder.predict_chunk(backbone, memory, batch(loaded), action_capture=capture)
    rows = token_rows_from_capture(
        capture,
        block_base=decoder.config.block_base,
        temperature=None,
    )
    decoder.cache_restore(memory, snapshot)
    return backbone, decoder, memory, capture, rows


def test_greedy_capture_is_pure_observation() -> None:
    """The surface records the decode; it must never steer it — actions
    with capture on are bit-identical to the plain greedy path."""
    backbone, decoder, loaded = build()
    plain, _ = decoder.predict_chunk(backbone, encode_memory(backbone), batch(loaded))
    capture: list[ActionCaptureStep] = []
    captured, _ = decoder.predict_chunk(
        backbone,
        encode_memory(backbone),
        batch(loaded),
        action_capture=capture,
    )
    assert capture, "action phase must capture at least one step"
    assert torch.equal(plain, captured)


def test_greedy_logprobs_match_teacher_forced_reforward() -> None:
    """The item-1 oracle, both halves: the masked-softmax reduction is
    bit-exact on the captured logits, and a teacher-forced re-forward
    over the emitted ids reproduces the logprobs to reduction-shape
    noise only (measured 2.4e-6 on this fixture; bound 1e-5)."""
    backbone, decoder, memory, capture, rows = greedy_rows()
    reference = decoder.teacher_forced_block_logits(
        backbone,
        memory,
        [[int(i) for i in row.ids] for row in rows],
    )
    for row_index, row in enumerate(rows):
        assert row.temperature == 1.0  # greedy records the plain masked softmax
        emitted = torch.from_numpy(row.logprobs)
        allowed = unpack(row)
        positions = torch.arange(len(row.ids))
        chosen = torch.from_numpy(row.ids)
        own = torch.stack(
            [
                step.block_logits[row_index]
                for step in capture
                if bool(step.active[row_index])
            ],
        )
        own_logprobs = own.masked_fill(~allowed, float("-inf")).log_softmax(-1)
        assert torch.equal(own_logprobs[positions, chosen], emitted)
        ref_row = reference[row_index]
        assert ref_row is not None
        ref_logprobs = ref_row.masked_fill(~allowed, float("-inf")).log_softmax(-1)
        delta = float((ref_logprobs[positions, chosen] - emitted).abs().max())
        assert delta < 1e-5, (
            f"row {row_index}: re-forwarded chosen logprobs drifted "
            f"{delta:.2e} from the emitted rows — beyond reduction-shape "
            "noise, the re-forward is not the same masked softmax"
        )


def test_recorded_mask_reconstructs_from_ids_alone() -> None:
    """The trainer's half of "train-time grammar mask == rollout mask":
    the mask at step t is budget arithmetic over the id prefix, so a
    recomputation from the recorded ids alone must land bit-for-bit on
    the packbits surface — and the ids consume the chunk exactly."""
    _, decoder, _, _, rows = greedy_rows()
    lengths = decoder.symbol_lengths
    pad = decoder.codec.pad
    total = decoder.config.chunk_size * decoder.config.action_dim
    for row in rows:
        assert sum(int(lengths[i]) for i in row.ids) == total
        recorded = unpack(row)
        remaining = total
        for step, codec_id in enumerate(row.ids):
            allowed = (lengths > 0) & (lengths <= remaining)
            allowed[pad] = remaining == 0
            assert torch.equal(recorded[step], allowed), (
                f"step {step}: recorded mask diverges from the id-prefix recomputation"
            )
            remaining -= int(lengths[codec_id])


def test_sampled_rows_record_the_sampled_stream() -> None:
    """Sampled decode with capture: the recorded ids ARE the stream the
    chunk decoded from (codec.decode over a row's ids reproduces the
    returned actions bit-for-bit), rows are exactly reproducible under
    their keys, and a different draw records a different stream."""
    backbone, decoder, loaded = build()

    def decode(draw: int) -> tuple[torch.Tensor, list[TokenRow]]:
        capture: list[ActionCaptureStep] = []
        actions, _ = decoder.predict_chunk(
            backbone,
            encode_memory(backbone),
            batch(loaded),
            sampling=ARSampling(temperature=2.0, rngs=rngs(draw)),
            action_capture=capture,
        )
        rows = token_rows_from_capture(
            capture,
            block_base=decoder.config.block_base,
            temperature=2.0,
        )
        return actions, rows

    actions, rows = decode(0)
    _, again = decode(0)
    _, other = decode(1)
    q01 = np.full(loaded.action_dim, -1.0)
    q99 = np.full(loaded.action_dim, 1.0)
    for row_index, row in enumerate(rows):
        assert row.temperature == 2.0
        decoded = torch.from_numpy(
            loaded.decode([int(i) for i in row.ids], q01, q99),
        ).float()
        assert torch.equal(actions[row_index].cpu(), decoded)
        assert np.array_equal(row.ids, again[row_index].ids)
        assert np.array_equal(row.logprobs, again[row_index].logprobs)
        assert bool(np.isfinite(row.logprobs).all()) and bool(
            (row.logprobs <= 0).all(),
        )
    assert any(
        not np.array_equal(a.ids, b.ids) for a, b in zip(rows, other, strict=True)
    )


def test_block_softmax_equals_full_vocab_softmax() -> None:
    """token_rows_from_capture reduces over BLOCK columns only; the
    sampler's distribution lives on the full backbone vocab with every
    non-block column masked to -inf. Padding the block back to full
    width changes nothing: exp(-inf) terms are exact zeros, so the two
    log-softmaxes agree bit-for-bit on every legal column."""
    generator = torch.Generator().manual_seed(11)
    base, vocab = 7, 40
    block = torch.randn(vocab, generator=generator)
    legal = torch.rand(vocab, generator=generator) > 0.4
    legal[3] = True  # at least one legal id
    narrow = block.masked_fill(~legal, float("-inf")).log_softmax(-1)
    full = torch.cat([torch.zeros(base), block]).masked_fill(
        ~torch.cat([torch.zeros(base, dtype=torch.bool), legal]),
        float("-inf"),
    )
    wide = full.log_softmax(-1)[base:]
    assert torch.equal(narrow[legal], wide[legal])


def test_empty_capture_is_loud() -> None:
    with pytest.raises(ValueError, match="empty capture"):
        token_rows_from_capture([], block_base=8, temperature=None)


def test_policy_capture_guards_are_loud() -> None:
    """The flag promises rows — a predict that cannot produce them must
    refuse before any compute: ensembled draws execute no single token
    stream, and non-AR families have no token stream at all."""
    _, decoder, _ = build()
    policy = object.__new__(BijouPolicy)
    policy.mask_state = False
    policy.subgoal_swap = None
    policy.condition_override = {}
    policy.capture_token_rows = True
    policy.ar_decoder = decoder
    policy.sample_draws = 2
    with pytest.raises(SystemExit, match="sample-draws"):
        policy.predict_with_text([], [])
    policy.sample_draws = 1
    policy.ar_decoder = None
    with pytest.raises(SystemExit, match="AR-suffix"):
        policy.predict_with_text([], [])


def test_emit_training_rows_refuses_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """--emit-training-rows records a policy's token stream — the
    parser refuses it with --hold before anything spawns."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["rollout_sim_parallel.py", "--hold", "--emit-training-rows", "rows"],
    )
    with pytest.raises(SystemExit):
        rollout_parse_args()


def test_training_row_writer_roundtrip(tmp_path: Path) -> None:
    """One write: meta.json carries the provenance dict, index.jsonl
    gains exactly the row's entry, and the NPZ round-trips every array
    bit-exact with frames decodable back to their shape."""
    generator = np.random.default_rng(0)
    top = generator.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    wrist = generator.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    row = TokenRow(
        ids=np.array([4, 1, 9], dtype=np.int64),
        logprobs=np.array([-0.5, -1.25, -0.03125], dtype=np.float32),
        allowed_packed=generator.integers(0, 255, (3, 5), dtype=np.uint8),
        vocab_total=40,
        temperature=1.0,
    )
    writer = TrainingRowWriter(tmp_path / "rows", {"run_seed": 7})
    writer.write(
        seed=3,
        replan=5,
        draw=2,
        top=top,
        wrist=wrist,
        state=np.arange(6, dtype=np.float64),
        row=row,
    )
    assert writer.rows_written == 1
    assert json.loads((tmp_path / "rows" / "meta.json").read_text()) == {"run_seed": 7}
    index = [
        json.loads(line)
        for line in (tmp_path / "rows" / "index.jsonl").read_text().splitlines()
    ]
    assert index == [
        {
            "path": "seed003_draw02_replan005.npz",
            "seed": 3,
            "draw": 2,
            "replan": 5,
            "tokens": 3,
            "vocab_total": 40,
            "temperature": 1.0,
        },
    ]
    data = np.load(tmp_path / "rows" / index[0]["path"])
    assert np.array_equal(data["ids"], row.ids)
    assert np.array_equal(data["logprobs"], row.logprobs)
    assert np.array_equal(data["allowed_packed"], row.allowed_packed)
    assert data["state"].dtype == np.float32
    from PIL import Image

    for key in ("top_jpeg", "wrist_jpeg"):
        image = Image.open(io.BytesIO(data[key].tobytes()))
        assert image.size == (64, 48)
        assert image.mode == "RGB"
