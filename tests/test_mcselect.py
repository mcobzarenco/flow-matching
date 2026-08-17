"""#6 rung (c) producer instrument oracles — the masked-contrast
(mcselect) eval path (pre-reg 2026-08-09-prereg-subgoal-mcselect.md;
the frozen READS are oracled in fontaine/scripts/mcselect_results.py
--oracle, this file gates the PRODUCER):

- planted-informative fixture (draft oracle 1): a scripted model whose
  "sharp" candidate's conditional distribution is strictly more
  informative than the masked reference must earn the larger KL — with
  exact hand arithmetic at tau=1 (the reference-identical candidate
  lands exactly 0), NaN pinned at the truncated candidate.
- tau degeneracy (draft oracle 2): tau -> infinity tempers the masked
  reference to uniform-over-legal, so the score must collapse to
  log|legal| - H(p_cond), checked exactly.
- decode-vs-teacher-forced identity (draft oracle 3's op-identity
  core): on the REAL tiny Gemma decoder, the logits captured DURING a
  greedy decode must reproduce under teacher_forced_block_logits over
  the decoded ids against the restored prefill — same memory, same
  scaffold, so any drift is an instrument break. (The live launcher
  additionally spot-checks candidate-0 conditioning against the banked
  rung-(a) self arm on real data.)
- scaffold/plumbing guards: filler rows, empty sequences, planner-less
  base enforcement, missing candidate rows, CLI flag matrix.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, cast

import pytest
import torch
from test_ar_backbone import BATCH, build, encode_memory
from test_ar_backbone import batch as tiny_batch
from test_collator import CHUNK, DIM
from test_selfsubgoal import FakeBase, eval_collator, labeled_item
from test_selfsubgoal import _parse as parse_cli

from bijou.eval.policies import BijouPolicy, MaskedContrastSubgoalPolicy
from bijou.modelling.interface import ActionCaptureStep, ARSampling
from bijou.models.ar_suffix_ops import batch_action_quantiles
from bijou.vla import ARPrediction

# ------------------------------------------------- scripted harness

FAKE_VOCAB = 4  # scripted block width; last id kept illegal so the
# grammar mask is exercised on both KL sides


class ScriptedModel:
    """Speaks exactly the trait surface MaskedContrastSubgoalPolicy
    consumes (predict_ar + teacher_forced_block_logits). The decoder is
    the REAL tiny GemmaARDecoder (the AR gate + block_base/PAD
    arithmetic are exercised for real); the decodes are scripted: one
    active action step whose conditional block logits are keyed on the
    RENDERED condition text (so a candidate that fails to reach the
    prompt slot is a loud KeyError, not a silent wrong number)."""

    def __init__(
        self,
        script: dict[str, torch.Tensor],
        masked_logits: torch.Tensor,
    ) -> None:
        _, self.decoder, _ = build()
        self.script = script
        self.masked_logits = masked_logits
        self.conditioned_prompts: list[str] = []

    def _logits_for(self, condition_text: str) -> torch.Tensor:
        for needle, logits in self.script.items():
            if needle in condition_text:
                return logits
        raise KeyError(
            f"no scripted candidate rendered in {condition_text!r} — the "
            "conditioned collator did not inject the candidate text",
        )

    def predict_ar(
        self,
        batch: Any,
        *,
        sampling: ARSampling | None = None,
        capture: list[ActionCaptureStep] | None = None,
    ) -> ARPrediction:
        samples = tuple(batch.encoder_inputs.samples)
        rows = len(samples)
        if capture is not None:
            # Conditioned pass: one active step of scripted logits.
            for sample in samples:
                self.conditioned_prompts.append(sample.condition_text)
            logits = torch.stack(
                [self._logits_for(s.condition_text) for s in samples],
            )
            allowed = torch.ones(rows, FAKE_VOCAB, dtype=torch.bool)
            allowed[:, -1] = False
            base = self.decoder.config.block_base
            chosen = (
                logits.masked_fill(~allowed, torch.finfo(torch.float32).min).argmax(-1)
                + base
            )
            capture.append(
                ActionCaptureStep(
                    block_logits=logits.float(),
                    allowed=allowed,
                    active=torch.ones(rows, dtype=torch.bool),
                    chosen=chosen,
                ),
            )
        return ARPrediction(actions=torch.zeros(rows, CHUNK, DIM))

    def teacher_forced_block_logits(
        self,
        batch: Any,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        rows, width = action_ids.shape
        return self.masked_logits.expand(rows, width, FAKE_VOCAB).clone()


def build_mcselect(
    *,
    tau: float,
    candidates: list[dict[str, Any]],
    script: dict[str, torch.Tensor],
    masked_logits: torch.Tensor,
    index: int = 7,
) -> tuple[ScriptedModel, MaskedContrastSubgoalPolicy]:
    model = ScriptedModel(script, masked_logits)
    base = cast(
        BijouPolicy,
        FakeBase(cast(Any, model)),
    )  # planner-less, subgoal trained
    policy = MaskedContrastSubgoalPolicy(
        base,
        candidates_by_index={index: candidates},
        tau=tau,
    )
    return model, policy


def kl_legal(
    cond: torch.Tensor,
    ref: torch.Tensor,
    tau: float,
) -> float:
    """Hand arithmetic over the LEGAL slice (last id illegal)."""
    lp_c = cond[:-1].double().log_softmax(-1)
    lp_r = (ref[:-1].double() / tau).log_softmax(-1)
    return float((lp_c.exp() * (lp_c - lp_r)).sum())


REF = torch.tensor([2.0, 0.5, -1.0, 9.0])  # 9.0 sits at the ILLEGAL id
# Peaked on an id the reference gives little mass — the informative
# plant (peaking where the reference already peaks would be a weak KL).
SHARP = torch.tensor([-4.0, 8.0, -4.0, 0.0])
CANDS = [
    {"text": "match the reference", "truncated": False},
    {"text": "sharpen the plan", "truncated": False},
    {"text": "budget ran out", "truncated": True},
]
SCRIPT = {"match the reference": REF.clone(), "sharpen the plan": SHARP.clone()}


def test_planted_informative_candidate_wins_the_kl() -> None:
    """Draft oracle 1: the informative candidate's KL must dominate,
    and at tau=1 the reference-identical candidate lands EXACTLY 0 —
    both against exact hand arithmetic; the truncated slot stays NaN
    (a finite value there is the read script's abort condition)."""
    _, policy = build_mcselect(
        tau=1.0,
        candidates=CANDS,
        script=SCRIPT,
        masked_logits=REF.clone(),
    )
    chunks = policy.predict([labeled_item()], [7])
    assert len(chunks) == 1 and chunks[0].shape == (CHUNK, DIM)
    kl = policy.kl[7]
    assert len(kl) == 3
    assert kl[0] == pytest.approx(0.0, abs=1e-12)  # KL(p || p) == 0
    assert kl[1] == pytest.approx(kl_legal(SHARP, REF, 1.0), rel=1e-9)
    assert kl[1] > 0.5  # the plant is decisive, not epsilon
    assert math.isnan(kl[2])
    assert kl[1] > kl[0]  # the argmax (reader-side) lands on the plant


def test_tau_four_matches_hand_arithmetic() -> None:
    """The pre-reg tau: both eligible scores against exact tempered
    hand arithmetic (the tempering must hit the REFERENCE side only)."""
    _, policy = build_mcselect(
        tau=4.0,
        candidates=CANDS,
        script=SCRIPT,
        masked_logits=REF.clone(),
    )
    policy.predict([labeled_item()], [7])
    kl = policy.kl[7]
    assert kl[0] == pytest.approx(kl_legal(REF, REF, 4.0), rel=1e-9)
    assert kl[1] == pytest.approx(kl_legal(SHARP, REF, 4.0), rel=1e-9)


def test_tau_degeneracy_uniform_reference() -> None:
    """Draft oracle 2: tau -> infinity flattens the tempered reference
    to uniform-over-legal, so KL must equal log|legal| - H(p_cond)."""
    _, policy = build_mcselect(
        tau=1e12,
        candidates=CANDS,
        script=SCRIPT,
        masked_logits=REF.clone(),
    )
    policy.predict([labeled_item()], [7])
    for slot, cond in ((0, REF), (1, SHARP)):
        lp = cond[:-1].double().log_softmax(-1)
        entropy = float(-(lp.exp() * lp).sum())
        expected = math.log(FAKE_VOCAB - 1) - entropy
        assert policy.kl[7][slot] == pytest.approx(expected, rel=1e-6)


def test_dump_rows_shapes_and_nan_placement() -> None:
    _, policy = build_mcselect(
        tau=4.0,
        candidates=CANDS,
        script=SCRIPT,
        masked_logits=REF.clone(),
    )
    policy.predict([labeled_item()], [7])
    cand_pred = policy.cand_pred[7]
    assert cand_pred.shape == (3, CHUNK, DIM)
    assert not cand_pred[0].isnan().any() and not cand_pred[1].isnan().any()
    assert cand_pred[2].isnan().all()  # ineligible: never decoded
    assert policy.pred_masked[7].shape == (CHUNK, DIM)
    assert not policy.pred_masked[7].isnan().any()


def test_conditioned_prompts_render_candidate_texts() -> None:
    """The candidate text must travel through the trained [subgoal|…]
    prompt slot (the SelfSubgoalPolicy rendering path) — and the
    truncated candidate must never be rendered at all."""
    model, policy = build_mcselect(
        tau=4.0,
        candidates=CANDS,
        script=SCRIPT,
        masked_logits=REF.clone(),
    )
    policy.predict([labeled_item()], [7])
    rendered = "\n".join(model.conditioned_prompts)
    assert "match the reference" in rendered
    assert "sharpen the plan" in rendered
    assert "budget ran out" not in rendered


def test_missing_candidates_row_aborts() -> None:
    _, policy = build_mcselect(
        tau=4.0,
        candidates=CANDS,
        script=SCRIPT,
        masked_logits=REF.clone(),
        index=7,
    )
    with pytest.raises(SystemExit, match="no row in the candidates"):
        policy.predict([labeled_item()], [8])


def test_planner_less_base_enforced() -> None:
    model = ScriptedModel(SCRIPT, REF.clone())
    conditioned_base = cast(
        BijouPolicy,
        FakeBase(cast(Any, model), include_subgoal_condition=True),
    )
    with pytest.raises(SystemExit, match="PLANNER-LESS"):
        MaskedContrastSubgoalPolicy(
            conditioned_base,
            candidates_by_index={7: CANDS},
            tau=4.0,
        )


def test_bad_tau_and_bad_candidates_abort() -> None:
    model = ScriptedModel(SCRIPT, REF.clone())
    base = cast(BijouPolicy, FakeBase(cast(Any, model)))
    with pytest.raises(SystemExit, match="tau"):
        MaskedContrastSubgoalPolicy(base, candidates_by_index={7: CANDS}, tau=0.0)
    with pytest.raises(SystemExit, match="empty candidates"):
        MaskedContrastSubgoalPolicy(base, candidates_by_index={}, tau=4.0)
    with pytest.raises(SystemExit, match="'text' and 'truncated'"):
        MaskedContrastSubgoalPolicy(
            base,
            candidates_by_index={7: [{"wrong": 1}]},
            tau=4.0,
        )


def test_name_provenance() -> None:
    _, policy = build_mcselect(
        tau=4.0,
        candidates=CANDS,
        script=SCRIPT,
        masked_logits=REF.clone(),
    )
    assert policy.name == "bijou@100000_mcselectsubgoal"


# --------------------------------------- real tiny decoder identity


def test_capture_matches_teacher_forced_reforward() -> None:
    """Draft oracle 3's op-identity core, on the REAL decoder: the
    block logits captured DURING the greedy decode must reproduce
    under teacher_forced_block_logits over the decoded id sequences
    against the restored prefill — same memory, same
    [opener][BOA][ids] scaffold. Also pins the capture's bookkeeping:
    active steps reconstruct exactly the emitted symbol sequence."""
    backbone, decoder, loaded = build()
    memory = encode_memory(backbone)
    snapshot = decoder.cache_snapshot(memory)
    capture: list[ActionCaptureStep] = []
    sample = tiny_batch(loaded)
    decoder.predict_chunk(
        backbone,
        memory,
        sample,
        quantiles=batch_action_quantiles(sample),
        action_capture=capture,
    )
    assert capture, "action phase must capture at least one step"
    base = decoder.config.block_base
    total = decoder.config.chunk_size * decoder.config.action_dim
    ids: list[list[int] | None] = []
    for row in range(BATCH):
        row_ids = [
            int(step.chosen[row]) - base for step in capture if bool(step.active[row])
        ]
        # The active-step reconstruction IS the emitted sequence: its
        # symbol lengths must consume the whole chunk exactly.
        assert sum(int(decoder.symbol_lengths[i]) for i in row_ids) == total
        ids.append(row_ids)
    decoder.cache_restore(memory, snapshot)
    reference = decoder.teacher_forced_block_logits(backbone, memory, ids)
    for row in range(BATCH):
        ref_row = reference[row]
        assert ref_row is not None
        captured = torch.stack(
            [step.block_logits[row] for step in capture if bool(step.active[row])],
        )
        assert ref_row.shape == captured.shape
        assert torch.allclose(ref_row, captured, atol=1e-4), (
            f"row {row}: teacher-forced reference drifted from the "
            f"decode's own logits (max |d| "
            f"{(ref_row - captured).abs().max().item():.2e})"
        )


def test_teacher_forced_filler_and_empty_rows() -> None:
    backbone, decoder, loaded = build()
    memory = encode_memory(backbone)
    snapshot = decoder.cache_snapshot(memory)
    capture: list[ActionCaptureStep] = []
    sample = tiny_batch(loaded)
    decoder.predict_chunk(
        backbone,
        memory,
        sample,
        quantiles=batch_action_quantiles(sample),
        action_capture=capture,
    )
    base = decoder.config.block_base
    row_ids = [int(step.chosen[0]) - base for step in capture if bool(step.active[0])]
    decoder.cache_restore(memory, snapshot)
    out = decoder.teacher_forced_block_logits(backbone, memory, [row_ids, None])
    assert out[1] is None
    assert out[0] is not None and out[0].shape[0] == len(row_ids)
    decoder.cache_restore(memory, snapshot)
    with pytest.raises(ValueError, match="empty action-id sequence"):
        decoder.teacher_forced_block_logits(backbone, memory, [[], None])
    decoder.cache_restore(memory, snapshot)
    assert decoder.teacher_forced_block_logits(backbone, memory, [None, None]) == [
        None,
        None,
    ]


def test_capture_off_leaves_decode_untouched() -> None:
    """No-capture decode is byte-identical to a captured one — the
    capture is observation, never intervention."""
    backbone, decoder, loaded = build()
    memory = encode_memory(backbone)
    snapshot = decoder.cache_snapshot(memory)
    sample = tiny_batch(loaded)
    quantiles = batch_action_quantiles(sample)
    plain, _ = decoder.predict_chunk(backbone, memory, sample, quantiles=quantiles)
    decoder.cache_restore(memory, snapshot)
    capture: list[ActionCaptureStep] = []
    captured, _ = decoder.predict_chunk(
        backbone,
        memory,
        sample,
        quantiles=quantiles,
        action_capture=capture,
    )
    assert torch.equal(plain, captured)


# ------------------------------------------------------- CLI guards


def test_cli_mcselect_flag_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = parse_cli(
        monkeypatch,
        "--subgoal-mode",
        "mcselect",
        "--checkpoint",
        "ckpt",
        "--subgoal-candidates-file",
        "cands.json",
        "--mcselect-tau",
        "4.0",
        "--dump-predictions",
        "out.npz",
    )
    assert ok.subgoal_mode == "mcselect" and ok.mcselect_tau == 4.0
    for bad in (
        # no candidates file
        (
            "--subgoal-mode",
            "mcselect",
            "--checkpoint",
            "c",
            "--mcselect-tau",
            "4.0",
            "--dump-predictions",
            "o.npz",
        ),
        # no tau (no silent default on a pre-registered knob)
        (
            "--subgoal-mode",
            "mcselect",
            "--checkpoint",
            "c",
            "--subgoal-candidates-file",
            "f.json",
            "--dump-predictions",
            "o.npz",
        ),
        # no dump — the measurement would be thrown away
        (
            "--subgoal-mode",
            "mcselect",
            "--checkpoint",
            "c",
            "--subgoal-candidates-file",
            "f.json",
            "--mcselect-tau",
            "4.0",
        ),
        # mcselect flags outside the mode
        ("--checkpoint", "c", "--subgoal-candidates-file", "f.json"),
        ("--checkpoint", "c", "--mcselect-tau", "4.0"),
        (
            "--subgoal-mode",
            "self",
            "--checkpoint",
            "c",
            "--mcselect-tau",
            "4.0",
        ),
        # draws-family flags refused in mcselect mode
        (
            "--subgoal-mode",
            "mcselect",
            "--checkpoint",
            "c",
            "--subgoal-candidates-file",
            "f.json",
            "--mcselect-tau",
            "4.0",
            "--dump-predictions",
            "o.npz",
            "--subgoal-draws",
            "8",
        ),
    ):
        with pytest.raises(SystemExit):
            parse_cli(monkeypatch, *bad)


def test_fake_base_is_argparse_namespace_shaped() -> None:
    """FakeBase.info must keep quacking like the policy's info record
    for the condition-fields set the policy consumes (guards the
    fixture)."""
    model = ScriptedModel(SCRIPT, REF.clone())
    base = cast(BijouPolicy, FakeBase(cast(Any, model)))
    assert isinstance(base.info, argparse.Namespace)
    assert "subgoal" in base.info.condition_fields


def test_eval_collator_available() -> None:
    # Imported fixture sanity — the conditioned collator derives from it.
    collator = eval_collator(())
    assert collator.generate_override == ()
