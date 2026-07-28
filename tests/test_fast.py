"""Tests for bijou.fast (DCT + BPE action tokenizer).

Pure CPU/numpy, synthetic data, milliseconds — these run inside check.py.
Smooth sinusoid chunks emulate normalized action trajectories: the DCT
concentrates them into few coefficients, so round-trips must be tight and
compression must be real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bijou.fast import (
    FastDecodeError,
    FastTokenizer,
    QuantileEntry,
    dct_matrix,
    load_quantile_table,
    quantile_entry_for,
    save_quantile_table,
)

H, D = 50, 6


def smooth_chunks(n: int, seed: int) -> np.ndarray:
    """[n, H, D] sums of low-frequency sinusoids in roughly [-1, 1]."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, H)[None, :, None]
    freq = rng.uniform(0.3, 3.0, size=(n, 1, D))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1, D))
    amp = rng.uniform(0.2, 0.9, size=(n, 1, D))
    offset = rng.uniform(-0.3, 0.3, size=(n, 1, D))
    return amp * np.sin(2.0 * np.pi * freq * t + phase) + offset


@pytest.fixture(scope="module")
def tokenizer() -> FastTokenizer:
    return FastTokenizer.fit(smooth_chunks(512, seed=0), vocab_size=512)


def test_dct_matrix_is_orthonormal() -> None:
    matrix = dct_matrix(H)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(H), atol=1e-12)


def test_dct_matches_known_constant_signal() -> None:
    # DCT-II (ortho) of a constant signal is sqrt(n) * value in bin 0.
    signal = np.full(H, 0.7)
    coefficients = dct_matrix(H) @ signal
    np.testing.assert_allclose(coefficients[0], 0.7 * np.sqrt(H), atol=1e-12)
    np.testing.assert_allclose(coefficients[1:], 0.0, atol=1e-12)


def test_round_trip_reconstruction(tokenizer: FastTokenizer) -> None:
    chunks = smooth_chunks(32, seed=1)
    for chunk in chunks:
        decoded = tokenizer.decode(tokenizer.encode(chunk))
        # Quantization at scale 10 bounds per-COEFFICIENT error by 0.05
        # (~uniform, sigma 0.029); the orthonormal inverse preserves that
        # variance per step, so the max over 300 samples sits around
        # 3-4 sigma (~0.1) and the mean |error| around 0.023.
        assert np.abs(decoded - chunk).max() < 0.15
        assert np.abs(decoded - chunk).mean() < 0.03


def test_compression_is_real(tokenizer: FastTokenizer) -> None:
    chunks = smooth_chunks(64, seed=2)
    lengths = [len(t) for t in tokenizer.encode_batch(chunks)]
    naive = H * D  # one token per (step, dim), the binning baseline
    assert max(lengths) < naive / 3
    assert sum(lengths) / len(lengths) < naive / 5


def test_encode_deterministic(tokenizer: FastTokenizer) -> None:
    chunk = smooth_chunks(1, seed=3)[0]
    assert tokenizer.encode(chunk) == tokenizer.encode(chunk)


def test_encode_rejects_wrong_shape(tokenizer: FastTokenizer) -> None:
    with pytest.raises(ValueError, match="expected chunk of shape"):
        tokenizer.encode(np.zeros((H + 1, D)))


def test_decode_rejects_truncated_sequence(tokenizer: FastTokenizer) -> None:
    tokens = tokenizer.encode(smooth_chunks(1, seed=4)[0])
    with pytest.raises(FastDecodeError, match="expected"):
        tokenizer.decode(tokens[: len(tokens) // 2])


def test_decode_rejects_unknown_token(tokenizer: FastTokenizer) -> None:
    with pytest.raises(FastDecodeError, match="outside the vocabulary"):
        tokenizer.decode([tokenizer.vocab_size + 7])


def test_out_of_alphabet_coefficients_clip_and_count(
    tokenizer: FastTokenizer,
) -> None:
    wild = smooth_chunks(1, seed=5)[0] * 50.0  # far outside the fit range
    before = tokenizer.clipped_coefficients
    decoded = tokenizer.decode(tokenizer.encode(wild))
    assert tokenizer.clipped_coefficients > before
    assert np.isfinite(decoded).all()


def test_save_load_round_trip(
    tokenizer: FastTokenizer,
    tmp_path: Path,
) -> None:
    tokenizer.save(tmp_path)
    reloaded = FastTokenizer.load(tmp_path)
    chunk = smooth_chunks(1, seed=6)[0]
    assert reloaded.encode(chunk) == tokenizer.encode(chunk)
    np.testing.assert_allclose(
        reloaded.decode(reloaded.encode(chunk)),
        tokenizer.decode(tokenizer.encode(chunk)),
    )


def test_fit_rejects_too_small_vocab() -> None:
    spiky = smooth_chunks(8, seed=7) * 20.0
    with pytest.raises(ValueError, match="alphabet"):
        FastTokenizer.fit(spiky, vocab_size=64)


def test_vocab_size_respected(tokenizer: FastTokenizer) -> None:
    assert tokenizer.vocab_size <= 512


def test_fit_quantized_equivalent_to_fit() -> None:
    chunks = smooth_chunks(128, seed=8)
    direct = FastTokenizer.fit(chunks, vocab_size=512)
    staged = FastTokenizer.fit_quantized(
        FastTokenizer.quantize_chunks(chunks, 10.0),
        scale=10.0,
        time_horizon=H,
        action_dim=D,
        vocab_size=512,
    )
    probe = smooth_chunks(4, seed=9)
    for chunk in probe:
        assert direct.encode(chunk) == staged.encode(chunk)


def test_quantile_entry_round_trip() -> None:
    rng = np.random.default_rng(10)
    raw = rng.uniform(-90.0, 90.0, size=(H, D))
    entry = QuantileEntry(
        q01=tuple(np.quantile(raw, 0.01, axis=0).tolist()),
        q99=tuple(np.quantile(raw, 0.99, axis=0).tolist()),
    )
    np.testing.assert_allclose(
        entry.unnormalize(entry.normalize(raw)),
        raw,
        atol=1e-9,
    )


def test_quantile_table_save_load_and_loud_lookup(tmp_path: Path) -> None:
    table = {
        "marius/rig": QuantileEntry(q01=(-1.0,) * D, q99=(1.0,) * D),
    }
    save_quantile_table(table, tmp_path)
    reloaded = load_quantile_table(tmp_path)
    assert reloaded == table
    assert quantile_entry_for(reloaded, "marius/rig") == table["marius/rig"]
    with pytest.raises(ValueError, match="no quantile stats"):
        quantile_entry_for(reloaded, "unknown/dataset")
