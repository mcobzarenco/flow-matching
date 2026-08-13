"""MolmoAct2 discrete (AR) pathway oracles (molmoact2-ar-head-port
item (b)), pure CPU: ``predict_action_discrete`` over the tiny-trunk
fixture family of tests/test_molmoact2_predictor.py with a SCRIPTED
lm_head — the real packed prompt, real trunk forwards, real KV cache
stepping, real released FAST codec; only the emission is forced, so
the loop/extraction/decode/tail wiring is what the assertions pin.

Covered: (1) end-to-end — a scripted well-formed emission
(``<action_start>`` bins ``<action_end>`` EOS) round-trips through
span-extraction + OpenFAST decode + their output tail (n_obs_steps
chunk slice, q01/q99 unnormalize) to exactly the hand-composed
actions, with the raw emission and bins surfaced for the parity
harness; (2) the reference's tolerant span semantics (missing
markers widen the span; non-action ids inside are dropped) as pure
cases over ``extract_action_bins``; (3) a non-decodable emission is
LOUD by default and matches the reference's silent zeros fallback
only when asked (``on_undecodable="zeros"``); (4) the generation cap
raises when EOS never arrives (the reference raises too); (5) the
config/resource guards. The GPU-scale token-for-token parity read vs
the released HF reference on banked anchor rows rides the
``molmoact2_e2e_parity.py`` extension, not this file."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from test_molmoact2_predictor import _observation, predictor  # noqa: F401

from bijou.molmoact2 import MolmoAct2Predictor, unnormalize_action
from bijou.molmoact2.fast_codec import MolmoAct2FastCodec
from bijou.molmoact2.predictor import DiscreteActionResult, extract_action_bins

FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_fast_tokenizer"
ACTION_TOKEN_START = 151_934
ACTION_START = 151_932
ACTION_END = 151_933
EOS = 151_645  # the fixture predictor's eos_token_id (BOS_ID)
HORIZON, DIM, N_OBS, N_STEPS = 4, 3, 2, 2  # the fixture tag's facts


class _ScriptedHead(torch.nn.Module):
    """Forces the emission while every trunk forward stays real: call
    ``k`` puts its one-hot at the LAST position — exactly the id the
    greedy loop consumes there."""

    def __init__(self, script: list[int], vocab: int) -> None:
        super().__init__()
        self.script = script
        self.vocab = vocab
        self.calls = 0

    @override
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        logits = torch.zeros((batch, length, self.vocab))
        index = min(self.calls, len(self.script) - 1)
        logits[0, -1, self.script[index]] = 100.0
        self.calls += 1
        return logits


def codec() -> MolmoAct2FastCodec:
    return MolmoAct2FastCodec.load(FIXTURE)


def decodable_bins(loaded: MolmoAct2FastCodec) -> list[int]:
    """Real release-BPE ids whose symbol lengths sum to exactly the
    fixture tag's HORIZON×DIM = 12."""
    lengths = loaded.symbol_lengths
    three = int(np.nonzero(lengths == 3)[0][0])
    four = int(np.nonzero(lengths == 4)[0][0])
    singles = [int(i) for i in np.nonzero(lengths == 1)[0][:5]]
    bins = [three, four, *singles]
    assert int(lengths[bins].sum()) == HORIZON * DIM
    return bins


def discrete_predictor(
    base: MolmoAct2Predictor,
    loaded: MolmoAct2FastCodec,
) -> MolmoAct2Predictor:
    return dataclasses.replace(
        base,
        action_mode="both",
        action_token_start_id=ACTION_TOKEN_START,
        fast_codec=loaded,
    )


def run_scripted(
    base: MolmoAct2Predictor,
    loaded: MolmoAct2FastCodec,
    script: list[int],
    monkeypatch: pytest.MonkeyPatch,
    on_undecodable: str = "raise",
) -> DiscreteActionResult:
    subject = discrete_predictor(base, loaded)
    wte = subject.trunk.text.transformer.wte
    vocab = wte.embedding.shape[0] + wte.new_embedding.shape[0]
    monkeypatch.setattr(subject.trunk.text, "lm_head", _ScriptedHead(script, vocab))
    observation = _observation()
    return subject.predict_action_discrete(
        images=observation["images"],
        task=observation["task"],
        state=observation["state"],
        on_undecodable=on_undecodable,
    )


def test_discrete_end_to_end_scripted(
    predictor: MolmoAct2Predictor,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = codec()
    bins = decodable_bins(loaded)
    script = [
        ACTION_START,
        *[ACTION_TOKEN_START + b for b in bins],
        ACTION_END,
        EOS,
    ]
    result = run_scripted(predictor, loaded, script, monkeypatch)
    assert result.token_ids[0].tolist() == script
    assert result.bins == bins
    normalized = torch.from_numpy(
        loaded.decode(bins, time_horizon=HORIZON, action_dim=DIM),
    ).to(torch.float32)[None]
    expected = unnormalize_action(
        normalized[:, N_OBS - 1 : N_OBS - 1 + N_STEPS],
        predictor.action_stats,
    ).to(torch.float32)
    assert result.actions.shape == (1, N_STEPS, DIM)
    assert torch.equal(result.actions, expected)


def test_extract_action_bins_tolerant_semantics() -> None:
    kwargs = {
        "action_start_id": ACTION_START,
        "action_end_id": ACTION_END,
        "action_token_start_id": ACTION_TOKEN_START,
        "block_vocab": 2048,
    }
    a = ACTION_TOKEN_START
    # Well-formed, with a stray non-action id inside (dropped) and
    # trailing ids past <action_end> (ignored).
    assert extract_action_bins(
        [ACTION_START, a + 5, 42, a + 9, ACTION_END, a + 1, EOS],
        **kwargs,
    ) == [5, 9]
    # No <action_start>: span opens at 0 (their fallback).
    assert extract_action_bins([a + 3, a + 4, ACTION_END], **kwargs) == [3, 4]
    # No <action_end>: span runs to the end, EOS dropped by the filter.
    assert extract_action_bins([ACTION_START, a + 7, EOS], **kwargs) == [7]
    # <action_end> BEFORE <action_start> is not a closer (their scan
    # starts after the opener) — span runs to the end.
    assert extract_action_bins([ACTION_END, ACTION_START, a + 2], **kwargs) == [2]
    # Ids beyond the block are not action tokens.
    assert extract_action_bins([a + 2047, a + 2048], **kwargs) == [2047]
    assert extract_action_bins([], **kwargs) == []


def test_non_decodable_is_loud_by_default_zeros_on_request(
    predictor: MolmoAct2Predictor,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = codec()
    short = decodable_bins(loaded)[1:]  # 9 symbols, not 12
    script = [
        ACTION_START,
        *[ACTION_TOKEN_START + b for b in short],
        ACTION_END,
        EOS,
    ]
    with pytest.raises(ValueError, match="expected 12"):
        run_scripted(predictor, loaded, script, monkeypatch)
    result = run_scripted(
        predictor,
        loaded,
        script,
        monkeypatch,
        on_undecodable="zeros",
    )
    expected = unnormalize_action(
        torch.zeros((1, N_STEPS, DIM)),
        predictor.action_stats,
    ).to(torch.float32)
    assert torch.equal(result.actions, expected)


def test_missing_eos_hits_the_cap_loudly(
    predictor: MolmoAct2Predictor,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = codec()
    with pytest.raises(RuntimeError, match="did not emit EOS"):
        run_scripted(predictor, loaded, [ACTION_TOKEN_START], monkeypatch)


def test_discrete_guards(
    predictor: MolmoAct2Predictor,  # noqa: F811
) -> None:
    loaded = codec()
    observation = _observation()
    call = {
        "images": observation["images"],
        "task": observation["task"],
        "state": observation["state"],
    }
    with pytest.raises(ValueError, match="action_mode"):
        dataclasses.replace(
            discrete_predictor(predictor, loaded),
            action_mode="continuous",
        ).predict_action_discrete(**call)
    with pytest.raises(ValueError, match="no FAST codec"):
        dataclasses.replace(
            discrete_predictor(predictor, loaded),
            fast_codec=None,
        ).predict_action_discrete(**call)
    with pytest.raises(ValueError, match="action_token_start"):
        dataclasses.replace(
            discrete_predictor(predictor, loaded),
            action_token_start_id=None,
        ).predict_action_discrete(**call)
    with pytest.raises(ValueError, match="on_undecodable"):
        discrete_predictor(predictor, loaded).predict_action_discrete(
            **call,
            on_undecodable="quiet",
        )
