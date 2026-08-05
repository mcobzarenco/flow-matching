"""Stable-triple noise keying (bijou.eval.policies.stable_noise) — the
corpus-composition-invariant scheme behind ``--noise-key stable``
(ideas #18.2 / bijou deep-dive finding 1)."""

from __future__ import annotations

import pytest
import torch

from bijou.eval.policies import (
    noise_for_item,
    sample_noise,
    stable_noise,
)

SHAPE = (50, 6)


def _item(repo: str, episode: int, frame: int) -> dict[str, object]:
    return {"repo_id": repo, "episode_index": episode, "frame_index": frame}


def test_index_keying_is_byte_identical_to_history() -> None:
    """The default path must reproduce the historical noise exactly —
    every banked flow anchor depends on it until the reseed amendment
    executes."""
    for index in (0, 7, 20_719_388):
        assert torch.equal(
            noise_for_item("index", 3, _item("any/repo", 0, 0), index, 0, SHAPE),
            sample_noise(3 + index, SHAPE),
        )


def test_stable_keying_ignores_the_corpus_index() -> None:
    """The point of the scheme: the same frame identity draws the same
    noise no matter where the corpus concat placed it (a dataset added
    or removed upstream shifts every index downstream of the edit)."""
    item = _item("owner/task-a", 4, 117)
    before = noise_for_item("stable", 3, item, 1_000, 0, SHAPE)
    after_shift = noise_for_item("stable", 3, item, 999_999, 0, SHAPE)
    assert torch.equal(before, after_shift)


def test_stable_noise_is_deterministic_and_identity_sensitive() -> None:
    base = stable_noise(0, "owner/task-a", 4, 117, 0, SHAPE)
    assert torch.equal(base, stable_noise(0, "owner/task-a", 4, 117, 0, SHAPE))
    assert not torch.equal(base, stable_noise(1, "owner/task-a", 4, 117, 0, SHAPE))
    assert not torch.equal(base, stable_noise(0, "owner/task-b", 4, 117, 0, SHAPE))
    assert not torch.equal(base, stable_noise(0, "owner/task-a", 5, 117, 0, SHAPE))
    assert not torch.equal(base, stable_noise(0, "owner/task-a", 4, 118, 0, SHAPE))


def test_stable_draws_are_distinct() -> None:
    """Draw collapse is the silent ensembling killer (the torch 32-bit
    manual_seed trap that forced DRAW_SEED_STRIDE) — the stable scheme
    keys the draw into the SeedSequence entropy instead, so all draws
    must be pairwise distinct."""
    draws = [stable_noise(0, "owner/task-a", 4, 117, d, SHAPE) for d in range(10)]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not torch.equal(draws[i], draws[j]), (i, j)


def test_identity_fields_are_not_ambiguous_across_slots() -> None:
    """The triple is hashed with a separator so (episode=1, frame=11)
    can never alias (episode=11, frame=1) or a repo_id that happens to
    end in the digits."""
    assert not torch.equal(
        stable_noise(0, "r", 1, 11, 0, SHAPE),
        stable_noise(0, "r", 11, 1, 0, SHAPE),
    )
    assert not torch.equal(
        stable_noise(0, "r/1", 1, 1, 0, SHAPE),
        stable_noise(0, "r/", 11, 1, 0, SHAPE),
    )


def test_stable_noise_is_standard_normal() -> None:
    """The keying must not distort the distribution the flow decoder
    integrates from: N(0,1) within loose bounds over a large draw."""
    values = torch.cat(
        [
            stable_noise(0, "owner/task-a", e, f, 0, SHAPE).flatten()
            for e in range(10)
            for f in range(20)
        ],
    )
    assert abs(float(values.mean())) < 0.02
    assert abs(float(values.std()) - 1.0) < 0.02


def test_unknown_noise_key_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown noise key"):
        noise_for_item("typo", 0, _item("r", 0, 0), 0, 0, SHAPE)
