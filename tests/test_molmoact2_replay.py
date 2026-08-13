"""MolmoAct2 token-GRPO rollout/replay oracles (phase-2 instrument
item 3, retargeted to the molmoact2 discrete surface), pure CPU over
the tiny-REAL-trunk fixture of tests/test_molmoact2_predictor with its
REAL (random-init, deterministic) lm_head — no scripting: the masked
decode, the capture surface and the teacher-forced replay all run the
same genuine forward path.

What these pin:

1. the grammar-masked SAMPLED decode is bit-reproducible under its RNG
   key, draw-distinct across keys, and every stream is legal by
   construction (budget consumed exactly, decodes, actions equal the
   hand-composed tail);
2. capture is observation, never intervention — greedy and sampled
   decodes with the surface on emit bit-identical streams/actions;
3. the TokenRow surface off ``token_rows_from_capture`` and the mask
   contract BOTH directions: the recorded packbits masks land
   bit-for-bit on the bins-only recomputation
   (``grammar_masks_from_bins``), and corrupt rows are loud;
4. the item-3 headline: the one-shot teacher-forced replay forward
   (``replay_logprobs``) reproduces the rollout's recorded chosen
   logprobs within the registered reduction-noise bound (1e-5, the §8
   amended bar — one-shot vs incremental cache feeding), greedy AND
   sampled;
5. glued into the decoder-generic GRPO surrogate, a fresh policy shows
   ratio ≈ 1, clip fraction 0, k3 ≈ 0, and the loss reduces to the
   advantage-weighted mean; the temperature-mismatch guard is loud;
6. the TrainingRowWriter → ``load_training_rows`` round trip is
   bit-exact on every token array and the replay forward runs end to
   end on loaded (JPEG-decoded) rows — frame equality is NOT asserted
   (JPEG is the registered lossy budget, memo §8);
7. the loud guards on the new rollout kwargs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_molmoact2_discrete import (
    ACTION_TOKEN_START,
    DIM,
    HORIZON,
    N_OBS,
    N_STEPS,
    codec,
    decodable_bins,
    discrete_predictor,
)
from test_molmoact2_predictor import _observation, build_predictor

from bijou.decoders.ar_backbone import ActionCaptureStep
from bijou.eval.policies import (
    TokenRow,
    stable_sample_rng,
    token_rows_from_capture,
)
from bijou.molmoact2 import MolmoAct2Predictor, unnormalize_action
from bijou.molmoact2.replay import (
    ReplayRow,
    grammar_masks_from_bins,
    load_training_rows,
    molmoact2_grpo_loss,
    replay_logprobs,
    verify_recorded_masks,
)
from bijou.train_grpo import GRPOConfig
from sim.rollout_sim_parallel import TrainingRowWriter

REFORWARD_BOUND = 1e-5  # the §8 amended reduction-noise bar (item 1's)


@pytest.fixture(scope="module")
def predictor() -> MolmoAct2Predictor:
    """The tiny trunk with its REAL lm_head widened over the
    ``<action_i>`` block (the base fixture's head stops at 151,936 —
    below the block — which the scripted-head discrete tests never
    notice; these oracles run genuine block logits)."""
    return build_predictor(vocab_size=156_032)


def rng(replan: int = 0, draw: int = 0) -> np.random.Generator:
    """The driver's keying: stable_sample_rng(run_seed, repo_id(draw),
    seed, replan, 0)."""
    repo = "sim/eval100" if draw == 0 else f"sim/eval100/draw{draw:02d}"
    return stable_sample_rng(0, repo, 3, replan, 0)


def masked_decode(
    subject: MolmoAct2Predictor,
    *,
    temperature: float | None = None,
    sample_rng: np.random.Generator | None = None,
    action_capture: list[ActionCaptureStep] | None = None,
) -> Any:
    obs = _observation()
    return subject.predict_action_discrete(
        images=obs["images"],
        task=obs["task"],
        state=obs["state"],
        grammar_masked=True,
        temperature=temperature,
        sample_rng=sample_rng,
        action_capture=action_capture,
    )


def replay_row(row: TokenRow, *, seed: int = 3, draw: int = 0) -> ReplayRow:
    """A collator row built in memory from the SAME frames the decode
    consumed — the oracle isolates the forward-path equivalence from
    the JPEG budget (test 6 covers the storage round trip)."""
    obs = _observation()
    return ReplayRow(
        top=obs["images"][0],
        wrist=obs["images"][1],
        state=obs["state"],
        ids=row.ids,
        logprobs=row.logprobs,
        allowed_packed=row.allowed_packed,
        vocab_total=row.vocab_total,
        temperature=row.temperature,
        seed=seed,
        draw=draw,
        replan=0,
    )


def unpack(row: TokenRow | ReplayRow) -> torch.Tensor:
    return torch.from_numpy(
        np.unpackbits(row.allowed_packed, axis=1, count=row.vocab_total).astype(bool),
    )


def test_sampled_masked_decode_reproducible_and_legal(
    predictor: MolmoAct2Predictor,
) -> None:
    loaded = codec()
    subject = discrete_predictor(predictor, loaded)
    first = masked_decode(subject, temperature=1.0, sample_rng=rng())
    again = masked_decode(subject, temperature=1.0, sample_rng=rng())
    other = masked_decode(subject, temperature=1.0, sample_rng=rng(draw=1))
    assert first.token_ids[0].tolist() == again.token_ids[0].tolist()
    assert torch.equal(first.actions, again.actions)
    assert first.bins != other.bins, "draw-distinct keys must sample distinct streams"
    lengths = loaded.symbol_lengths
    assert int(lengths[first.bins].sum()) == HORIZON * DIM
    normalized = torch.from_numpy(
        loaded.decode(first.bins, time_horizon=HORIZON, action_dim=DIM),
    ).to(torch.float32)[None]
    expected = unnormalize_action(
        normalized[:, N_OBS - 1 : N_OBS - 1 + N_STEPS],
        predictor.action_stats,
    ).to(torch.float32)
    assert torch.equal(first.actions, expected)


def test_capture_is_pure_observation(
    predictor: MolmoAct2Predictor,
) -> None:
    loaded = codec()
    subject = discrete_predictor(predictor, loaded)
    plain = masked_decode(subject)
    capture: list[ActionCaptureStep] = []
    captured = masked_decode(subject, action_capture=capture)
    assert plain.token_ids[0].tolist() == captured.token_ids[0].tolist()
    assert torch.equal(plain.actions, captured.actions)
    assert len(capture) == len(captured.bins)
    assert [int(s.chosen[0]) - ACTION_TOKEN_START for s in capture] == captured.bins
    sampled_capture: list[ActionCaptureStep] = []
    sampled_plain = masked_decode(subject, temperature=0.7, sample_rng=rng(replan=1))
    sampled = masked_decode(
        subject,
        temperature=0.7,
        sample_rng=rng(replan=1),
        action_capture=sampled_capture,
    )
    assert sampled_plain.token_ids[0].tolist() == sampled.token_ids[0].tolist()
    assert len(sampled_capture) == len(sampled.bins)


def test_token_rows_and_mask_contract_both_directions(
    predictor: MolmoAct2Predictor,
) -> None:
    loaded = codec()
    subject = discrete_predictor(predictor, loaded)
    capture: list[ActionCaptureStep] = []
    result = masked_decode(subject, action_capture=capture)
    (row,) = token_rows_from_capture(
        capture,
        block_base=ACTION_TOKEN_START,
        temperature=None,
    )
    assert row.ids.tolist() == result.bins
    assert row.temperature == 1.0  # greedy records the plain masked softmax
    assert row.vocab_total == loaded.block_vocab
    assert bool(np.isfinite(row.logprobs).all()) and bool((row.logprobs <= 0).all())
    # Both directions: recorded packbits == bins-only recomputation.
    recomputed = grammar_masks_from_bins(subject, result.bins)
    assert torch.equal(unpack(row), recomputed)
    verify_recorded_masks(subject, replay_row(row))  # and the guard agrees
    # Corrupt rows are loud: short stream, untrained bin, empty
    # (constructed streams — a real greedy stream's length is
    # head-dependent, so its truncation is not a stable fixture).
    known_good = decodable_bins(loaded)
    with pytest.raises(ValueError, match="consume"):
        grammar_masks_from_bins(subject, known_good[:-1])
    untrained = int(np.nonzero(loaded.symbol_lengths == 0)[0][0])
    with pytest.raises(ValueError, match="untrained"):
        grammar_masks_from_bins(subject, [untrained, *known_good[1:]])
    with pytest.raises(ValueError, match="empty"):
        grammar_masks_from_bins(subject, [])
    tampered = replay_row(row)
    bad_masks = tampered.allowed_packed.copy()
    bad_masks[0, 0] ^= 0xFF
    with pytest.raises(ValueError, match="diverge"):
        verify_recorded_masks(
            subject,
            dataclasses.replace(tampered, allowed_packed=bad_masks),
        )


def test_replay_logprobs_reproduce_the_rollout(
    predictor: MolmoAct2Predictor,
) -> None:
    """The item-3 headline oracle: the one-shot teacher-forced forward
    over ``prompt + [<action_start>] + bins`` reproduces the decode's
    recorded chosen logprobs to reduction-shape noise only — greedy
    and sampled, each under its own temperature."""
    loaded = codec()
    subject = discrete_predictor(predictor, loaded)
    for temperature, generator in ((None, None), (0.7, rng(replan=2))):
        capture: list[ActionCaptureStep] = []
        masked_decode(
            subject,
            temperature=temperature,
            sample_rng=generator,
            action_capture=capture,
        )
        (row,) = token_rows_from_capture(
            capture,
            block_base=ACTION_TOKEN_START,
            temperature=temperature,
        )
        with torch.no_grad():
            new_logprobs, decisions = replay_logprobs(
                subject,
                [replay_row(row)],
                task=_observation()["task"],
                temperature=row.temperature,
            )
        assert decisions.shape == new_logprobs.shape == (1, len(row.ids))
        assert bool(decisions.all())
        delta = float(
            (new_logprobs[0] - torch.from_numpy(row.logprobs)).abs().max(),
        )
        assert delta < REFORWARD_BOUND, (
            f"T={row.temperature}: replayed chosen logprobs drifted "
            f"{delta:.2e} from the rollout's records — beyond "
            "reduction-shape noise, the replay is not the same masked "
            "softmax"
        )


def test_fresh_policy_glues_into_grpo(
    predictor: MolmoAct2Predictor,
) -> None:
    loaded = codec()
    subject = discrete_predictor(predictor, loaded)
    rows: list[ReplayRow] = []
    for index, (temperature, generator) in enumerate(
        ((None, None), (1.0, rng(replan=3))),
    ):
        capture: list[ActionCaptureStep] = []
        masked_decode(
            subject,
            temperature=temperature,
            sample_rng=generator,
            action_capture=capture,
        )
        (row,) = token_rows_from_capture(
            capture,
            block_base=ACTION_TOKEN_START,
            temperature=temperature,
        )
        rows.append(replay_row(row, draw=index))
    advantages = torch.tensor([0.7, -0.3])
    config = GRPOConfig()  # temperature 1.0 — both rows recorded 1.0
    loss, stats = molmoact2_grpo_loss(
        subject,
        rows,
        task=_observation()["task"],
        advantages=advantages,
        config=config,
    )
    tokens = [len(row.ids) for row in rows]
    assert stats.tokens == sum(tokens)
    assert stats.clip_fraction == 0.0
    assert abs(stats.mean_ratio - 1.0) < 1e-4
    assert stats.approx_kl < 1e-8
    expected = -float(
        (advantages * torch.tensor(tokens, dtype=torch.float32)).sum() / sum(tokens),
    )
    assert abs(float(loss) - expected) < 1e-4
    with pytest.raises(ValueError, match=r"T=1\.0"):
        molmoact2_grpo_loss(
            subject,
            rows,
            task=_observation()["task"],
            advantages=advantages,
            config=GRPOConfig(temperature=0.7),
        )


def test_writer_loader_roundtrip_and_replay_runs(
    predictor: MolmoAct2Predictor,
    tmp_path: Path,
) -> None:
    loaded = codec()
    subject = discrete_predictor(predictor, loaded)
    capture: list[ActionCaptureStep] = []
    masked_decode(subject, temperature=1.0, sample_rng=rng(), action_capture=capture)
    (row,) = token_rows_from_capture(
        capture,
        block_base=ACTION_TOKEN_START,
        temperature=1.0,
    )
    obs = _observation()
    writer = TrainingRowWriter(tmp_path / "rows", {"run_seed": 0, "task": obs["task"]})
    writer.write(
        seed=3,
        replan=0,
        draw=0,
        top=obs["images"][0],
        wrist=obs["images"][1],
        state=obs["state"].numpy(),
        row=row,
    )
    meta, loaded_rows = load_training_rows(tmp_path / "rows")
    assert meta["task"] == obs["task"]
    (stored,) = loaded_rows
    assert np.array_equal(stored.ids, row.ids)
    assert np.array_equal(stored.logprobs, row.logprobs)
    assert np.array_equal(stored.allowed_packed, row.allowed_packed)
    assert (stored.temperature, stored.vocab_total) == (1.0, loaded.block_vocab)
    assert (stored.seed, stored.draw, stored.replan) == (3, 0, 0)
    assert stored.top.shape == obs["images"][0].shape
    assert torch.equal(stored.state, obs["state"].to(torch.float32))
    verify_recorded_masks(subject, stored)
    with torch.no_grad():
        new_logprobs, decisions = replay_logprobs(
            subject,
            [stored],
            task=meta["task"],
            temperature=stored.temperature,
        )
    assert bool(decisions.all()) and bool(torch.isfinite(new_logprobs).all())
    with pytest.raises(ValueError, match="empty"):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "meta.json").write_text("{}")
        (empty / "index.jsonl").write_text("")
        load_training_rows(empty)


def test_rollout_kwarg_guards(
    predictor: MolmoAct2Predictor,
) -> None:
    loaded = codec()
    subject = discrete_predictor(predictor, loaded)
    obs = _observation()
    call = {"images": obs["images"], "task": obs["task"], "state": obs["state"]}
    with pytest.raises(ValueError, match="grammar_masked"):
        subject.predict_action_discrete(
            **call,
            temperature=1.0,
            sample_rng=rng(),
        )
    with pytest.raises(ValueError, match="grammar_masked"):
        subject.predict_action_discrete(**call, action_capture=[])
    with pytest.raises(ValueError, match="together"):
        subject.predict_action_discrete(
            **call,
            grammar_masked=True,
            temperature=1.0,
        )
    with pytest.raises(ValueError, match="together"):
        subject.predict_action_discrete(
            **call,
            grammar_masked=True,
            sample_rng=rng(),
        )
    with pytest.raises(ValueError, match="positive"):
        subject.predict_action_discrete(
            **call,
            grammar_masked=True,
            temperature=0.0,
            sample_rng=rng(),
        )
    with pytest.raises(ValueError, match="empty"):
        replay_logprobs(subject, [], task=obs["task"], temperature=1.0)
