"""AR sampled-draws instrument oracles (ideas #19), pure CPU.

The instrument mirrors flow noise-draw ensembling for the AR family:
temperature-sample the action block N times per frame (Gumbel-max over
the grammar-masked softmax, per-row CPU RNG streams keyed by frame
identity + draw), mean the decoded chunks. These oracles pin the
contracts the pre-registration quotes: (1) sampling=None IS the
historical greedy path; (2) the T→0 limit recovers greedy exactly;
(3) sampled decodes are grammar-valid and deterministic under their
keys, and draws are distinct; (4) a row's ids depend only on its OWN
stream — batch-composition-independent at the sampler; (5) the
cache snapshot/restore that shares one prefill across draws is exact
(restored-cache decode ≡ fresh-encode decode); (6) the keying is
stable, component-sensitive, and domain-separated from flow noise;
(7) the loud guards (temperature ≤ 0, RNG-count mismatch, CLI flag
interactions). Fixture family: tests/test_ar_backbone."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch
from test_ar_backbone import BATCH, batch, build, encode_memory

from bijou.eval.cli import parse_args
from bijou.eval.policies import stable_noise, stable_sample_rng
from bijou.modelling.decoders.ar_suffix import _sample_action_ids
from bijou.modelling.interface import ARSampling


def rngs(draw: int, *, seed: int = 0) -> tuple[np.random.Generator, ...]:
    """One keyed stream per fixture row, the way eval builds them."""
    return tuple(
        stable_sample_rng(seed, "fixture/repo", 0, row, draw) for row in range(BATCH)
    )


def test_low_temperature_limit_recovers_greedy() -> None:
    """argmax(logits/T + G) at tiny T is argmax(logits) unless two
    logits tie to ~1e-4·Gumbel — the sampled path degenerates to the
    greedy decode, proving both share one decision point."""
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    greedy, _ = decoder.predict_chunk(backbone, encode_memory(backbone), sample)
    cold, _ = decoder.predict_chunk(
        backbone,
        encode_memory(backbone),
        sample,
        sampling=ARSampling(temperature=1e-4, rngs=rngs(0)),
    )
    assert torch.equal(cold, greedy)


def test_sampled_decode_valid_deterministic_and_draw_distinct() -> None:
    """Hot sampling stays inside the grammar (full valid chunks decode)
    and is exactly reproducible under its keys; a different draw index
    yields a different chunk (distinct streams actually sample)."""
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    hot = lambda draw: decoder.predict_chunk(
        backbone,
        encode_memory(backbone),
        sample,
        sampling=ARSampling(temperature=2.0, rngs=rngs(draw)),
    )[0]
    first, again, other = hot(0), hot(0), hot(1)
    assert first.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(first).all())
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_sampler_rows_are_batch_composition_independent() -> None:
    """Permuting the batch permutes the sampled ids — each row's draw
    comes from its own CPU stream, never from a shared batch stream."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 40, generator=generator)
    allowed = torch.rand(2, 40, generator=generator) > 0.3
    allowed[:, 0] = True  # at least one legal id per row
    sampling = ARSampling(temperature=1.0, rngs=rngs(0))
    flipped = ARSampling(temperature=1.0, rngs=tuple(reversed(rngs(0))))
    ids = _sample_action_ids(logits, allowed, sampling)
    swapped = _sample_action_ids(logits.flip(0), allowed.flip(0), flipped)
    assert torch.equal(ids, swapped.flip(0))


def test_sampler_never_leaves_the_mask() -> None:
    """Illegal ids sit at -inf before the Gumbel add — no temperature
    can make one win, even with a single legal id."""
    logits = torch.zeros(1, 40)
    allowed = torch.zeros(1, 40, dtype=torch.bool)
    allowed[0, 7] = True
    for draw in range(50):
        sampling = ARSampling(temperature=10.0, rngs=rngs(draw)[:1])
        assert int(_sample_action_ids(logits, allowed, sampling)) == 7


def test_cache_snapshot_restore_shares_one_prefill_exactly() -> None:
    """The instrument's prefill reuse: decode (consuming the cache),
    restore, decode again — both decodes must equal a fresh-encode
    decode bit-for-bit. Valid while the trunk caches stay append-only
    (cache_snapshot's documented contract)."""
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    memory = encode_memory(backbone)
    snapshot = decoder.cache_snapshot(memory)
    first, _ = decoder.predict_chunk(backbone, memory, sample)
    decoder.cache_restore(memory, snapshot)
    second, _ = decoder.predict_chunk(backbone, memory, sample)
    fresh, _ = decoder.predict_chunk(backbone, encode_memory(backbone), sample)
    assert torch.equal(first, second)
    assert torch.equal(first, fresh)


def test_stable_sample_rng_keying() -> None:
    """Same key → same stream; any component change → a different
    stream; and the flow-noise stream for the identical key is a
    DIFFERENT bitstream (domain separation — an AR draw must never
    replay a flow draw's noise)."""
    base = ("repo/x", 3, 17, 2)

    def head(seed: int, repo: str, episode: int, frame: int, draw: int) -> np.ndarray:
        return stable_sample_rng(seed, repo, episode, frame, draw).standard_normal(8)

    assert np.array_equal(head(0, *base), head(0, *base))
    perturbed = [
        head(1, *base),
        head(0, "repo/y", 3, 17, 2),
        head(0, "repo/x", 4, 17, 2),
        head(0, "repo/x", 3, 18, 2),
        head(0, "repo/x", 3, 17, 3),
    ]
    for other in perturbed:
        assert not np.array_equal(head(0, *base), other)
    flow = stable_noise(0, "repo/x", 3, 17, 2, (8,)).numpy()
    assert not np.array_equal(head(0, *base).astype(np.float32), flow)


def test_sampling_guards_are_loud() -> None:
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    for temperature in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            ARSampling(temperature=temperature, rngs=rngs(0))
    with pytest.raises(ValueError, match="one keyed stream per row"):
        decoder.predict_chunk(
            backbone,
            encode_memory(backbone),
            sample,
            sampling=ARSampling(temperature=1.0, rngs=rngs(0)[:1]),
        )


def _parse(monkeypatch: pytest.MonkeyPatch, *extra: str) -> argparse.Namespace:
    monkeypatch.setattr(
        "sys.argv",
        ["bijou.eval", "--data", "corpus", *extra],
    )
    return parse_args()


def test_ar_temperature_without_checkpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--ar-temperature", "1.0")


def test_ar_temperature_with_checkpoint_and_draws_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(
        monkeypatch,
        "--checkpoint",
        "ckpt",
        "--ar-temperature",
        "1.0",
        "--sample-draws",
        "10",
    )
    assert args.ar_temperature == 1.0
    assert args.sample_draws == 10
