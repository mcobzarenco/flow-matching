"""Subgoal-draws selection instrument oracles (#6 rung (b)), pure CPU.

The pre-registration (posts/2026-08-08-prereg-subgoal-draws.md) names
the abort-on-red oracles; these tests pin their CPU-checkable halves
(the rung-(a) precedent — the real-checkpoint halves run as the
execution item's preflight):

- exact-arithmetic scorer fixtures (oracle iii): self-certainty from
  hand-built per-step stats, token-F1 fractions, the frozen degenerate
  conventions (both-empty F1 = 1, one-empty = 0), single-candidate and
  exact-tie cases (ties → lowest index, greedy first);
- provenance separation (oracle v): the self-certainty pick consumes
  distribution stats only — its signature has no label or text
  argument; the ceiling pick REQUIRES an explicit true label; the ceil
  arm's name carries _ceilsubgoal;
- pass 2's request set excludes subgoal while conditioned (oracle iv),
  and the conditioned prompt renders through the one shared Collator
  (oracle vi), for both selection arms;
- the forced-empty / label-less-ceil rows collate the planner-less
  baseline prompt byte-exact (oracle ii);
- the REAL decode loop on the tiny fixture model: the text-only greedy
  candidate byte-matches the full pass's parsed value off one restored
  prefill (oracle i's CPU half), restore + re-decode is bit-stable,
  sampled draws are RNG-deterministic, and the recorded stats have the
  greedy argmax property (chosen ≥ mean per step);
- CLI flag interactions for the draws mode;
- the rung-(b') clean-list filter (pre-reg …-cleanlist oracles viii/ix
  CPU halves): planted filter-binds worlds on BOTH scorers, the
  all-truncated greedy fallback, no-truncation pick invariance, the
  single-candidate (draws-0) limit, and name/flag provenance. The
  banked-table halves (vii/x) run in
  fontaine/scripts/subgoal_draws_cleanlist_stage1.py.

Fixture families: test_selfsubgoal (FakeModel/FakeBase over the real
Collator) and test_ar_backbone (tiny 256-vocab Gemma, real decoder).
"""

from __future__ import annotations

import inspect
import math
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_ar_backbone import (
    batch as tiny_batch,
)
from test_ar_backbone import (
    codec,
    encode_memory,
)
from test_ar_backbone_aux import build_with_aux
from test_selfsubgoal import FakeBase, FakeModel, eval_collator, labeled_item
from test_selfsubgoal import _parse as parse_cli

from bijou.eval.policies import (
    BijouPolicy,
    SelectedSubgoalPolicy,
    SelfSubgoalPass1Policy,
)
from bijou.eval.subgoal_scoring import (
    ceiling_pick,
    eligible_indices,
    likelihood_pick,
    mean_chosen_logprob,
    medoid_pick,
    self_certainty,
    self_certainty_pick,
    token_f1,
)
from bijou.modelling.aux_text import AuxField, AuxGeneration
from bijou.modelling.interface import ARSampling, ValueCandidate
from bijou.vla import NarratedPrediction

# ------------------------------------------------------------- scorers


def test_self_certainty_exact_arithmetic() -> None:
    """SC = −mean(mean_logprob) − log V, exactly; a peaked distribution
    (very negative mean over the vocab) scores HIGH."""
    assert self_certainty([-2.0, -3.0], 8) == 2.5 - math.log(8)
    assert self_certainty([-1.0], 2) == 1.0 - math.log(2)
    peaked, flat = self_certainty([-9.0], 16), self_certainty([-3.0], 16)
    assert peaked > flat
    with pytest.raises(ValueError, match="zero decode steps"):
        self_certainty([], 8)
    with pytest.raises(ValueError, match="not a text vocabulary"):
        self_certainty([-1.0], 1)


def test_mean_chosen_logprob_exact() -> None:
    assert mean_chosen_logprob([-1.0, -3.0]) == -2.0
    with pytest.raises(ValueError, match="zero decode steps"):
        mean_chosen_logprob([])


def test_token_f1_exact_fractions_and_degenerates() -> None:
    # Multiset overlap {reach, the, boat} = 3 of 4 + 3 tokens.
    assert token_f1("reach toward the boat", "Reach the BOAT") == 6 / 7
    assert token_f1("a a b", "a a a") == 2 * 2 / 6  # multiset: two a's
    assert token_f1("same words", "same words") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("something", "") == 0.0
    assert token_f1("", "something") == 0.0
    assert token_f1("no overlap", "different entirely") == 0.0


def test_picks_break_ties_toward_lowest_index() -> None:
    # Identical stats → the greedy candidate (index 0) wins.
    assert self_certainty_pick([[-2.0], [-2.0], [-2.0]], 8) == 0
    # 1 and 2 tie above 0 → 1.
    assert self_certainty_pick([[-1.0], [-3.0], [-3.0]], 8) == 1
    assert likelihood_pick([[-2.0], [-1.0], [-1.0]]) == 1
    assert ceiling_pick(["reach the boat", "reach the boat"], "reach the boat") == 0
    # Single candidate: every pick is index 0.
    assert self_certainty_pick([[-5.0]], 8) == 0
    assert likelihood_pick([[-5.0]]) == 0
    assert medoid_pick(["only one"]) == 0
    assert ceiling_pick(["only one"], "reach the boat") == 0
    with pytest.raises(ValueError, match="zero candidates"):
        self_certainty_pick([], 8)


def test_eligible_indices_rule() -> None:
    """Rung (b') frozen rule: non-truncated original indices, ascending
    (so a sublist argmax mapped back preserves greedy-first ties);
    all-truncated → [0]; zero candidates loud."""
    assert eligible_indices([False, False, False]) == [0, 1, 2]
    assert eligible_indices([False, True, False]) == [0, 2]
    assert eligible_indices([True, False, True]) == [1]
    assert eligible_indices([True, True, True]) == [0]
    assert eligible_indices([True]) == [0]
    with pytest.raises(ValueError, match="zero candidates"):
        eligible_indices([])


def test_medoid_pick_consensus() -> None:
    # Two identical + one outlier: the shared string (lowest of the
    # pair) is the medoid.
    assert medoid_pick(["reach the boat", "grab a tool", "reach the boat"]) == 0
    # Fully symmetric candidates tie → lowest index.
    assert medoid_pick(["a", "b", "c"]) == 0


def test_ceiling_pick_maximizes_f1_vs_label() -> None:
    picks = ceiling_pick(
        ["grab a tool", "reach toward the boat", "reach the boat"],
        "reach the boat",
    )
    assert picks == 2


def test_provenance_separation_is_structural() -> None:
    """Oracle v: the deployment-honest pick cannot see labels or even
    candidate text; the ceiling pick cannot run without an explicit
    label argument."""
    bon_params = inspect.signature(self_certainty_pick).parameters
    assert set(bon_params) == {"per_candidate_mean_logprob", "allowed_vocab"}
    ceil_params = inspect.signature(ceiling_pick).parameters
    assert "true_label" in ceil_params
    default = ceil_params["true_label"].default
    assert default is inspect.Parameter.empty  # never optional


# ------------------------------------------- policies (fake model)


def cand(
    text: str,
    mean_logprob: tuple[float, ...],
    *,
    chosen_logprob: tuple[float, ...] | None = None,
    truncated: bool = False,
) -> ValueCandidate:
    return ValueCandidate(
        text=text,
        truncated=truncated,
        chosen_logprob=chosen_logprob or tuple(m + 0.5 for m in mean_logprob),
        mean_logprob=mean_logprob,
        allowed_vocab=16,
    )


class DrawsFakeModel(FakeModel):
    """FakeModel plus the candidates entry point: canned per-instruction
    candidate lists; the full-pass generation mirrors candidate 0 (the
    model-level greedy-equality oracle, honored by construction)."""

    def __init__(self, candidates: dict[str, list[ValueCandidate]]) -> None:
        super().__init__(
            {task: (cands[0].text or None) for task, cands in candidates.items()},
        )
        self.candidates = candidates
        self.candidate_calls: list[tuple[AuxField, int]] = []

    def predict_with_value_candidates(
        self,
        batch: Any,
        *,
        field: AuxField,
        generate: tuple[AuxField, ...],
        draws: int,
        sampling_for_draw: Any,
    ) -> tuple[NarratedPrediction, list[list[ValueCandidate]]]:
        samples = tuple(batch.encoder_inputs.samples)
        self.calls.append((samples, generate))
        self.candidate_calls.append((field, draws))
        generations = []
        rows = []
        for sample in samples:
            row = self.candidates[sample.instruction]
            rows.append(row)
            generations.append(
                AuxGeneration(
                    text=row[0].text,
                    subgoal=row[0].text or None,
                    holding=None,
                    progress=None,
                    event=None,
                    visible=None,
                ),
            )
        from test_selfsubgoal import CHUNK, DIM

        return (
            NarratedPrediction(
                actions=torch.zeros(len(samples), CHUNK, DIM),
                generations=generations,
            ),
            rows,
        )


# The banked risk case, planted: sampled candidate 2 is the most
# PEAKED (most-negative vocab-mean → highest SC → the bon pick) but
# candidate 1 is the one closest to the true label ("reach toward the
# boat" in labeled_item → the ceil pick) — confidence does not equal
# phase, so bon and ceil must diverge here.
CANDS = {
    "pick up the cube": [
        cand("lower the gripper", (-2.0, -2.0)),
        cand("reach toward the boat", (-4.0, -4.0)),
        cand("spin in place", (-5.0, -5.0)),
    ],
}


def build_draws(
    candidates: dict[str, list[ValueCandidate]],
    *,
    draws: int = 2,
    force_empty: bool = False,
    candidate_filter: str | None = None,
) -> tuple[
    DrawsFakeModel,
    SelfSubgoalPass1Policy,
    SelectedSubgoalPolicy,
    SelectedSubgoalPolicy,
]:
    model = DrawsFakeModel(candidates)
    base = cast(BijouPolicy, FakeBase(model))
    base.seed = 0  # BijouPolicy surface consumed by the draws keying
    pass1 = SelfSubgoalPass1Policy(base, draws=draws, temperature=1.0)
    bon = SelectedSubgoalPolicy(
        base,
        pass1,
        mode="bon",
        force_empty=force_empty,
        candidate_filter=candidate_filter,
    )
    ceil = SelectedSubgoalPolicy(
        base,
        pass1,
        mode="ceil",
        force_empty=force_empty,
        candidate_filter=candidate_filter,
    )
    return model, pass1, bon, ceil


def condition_texts(model: FakeModel, call: int) -> list[str]:
    return [sample.condition_text for sample in model.calls[call][0]]


def test_pass1_draws_mode_captures_candidates() -> None:
    model, pass1, _, _ = build_draws(CANDS)
    frame = labeled_item()
    chunks = pass1.predict([frame], [7])
    assert len(chunks) == 1
    assert model.candidate_calls == [(AuxField.SUBGOAL, 2)]
    assert [c.text for c in pass1.candidates[7]] == [
        "lower the gripper",
        "reach toward the boat",
        "spin in place",
    ]
    # Records ride exactly as in rung (a): greedy text is the record.
    assert pass1.records[7].generated_subgoal == "lower the gripper"
    assert pass1.records[7].true_subgoal == "reach toward the boat"
    # Pass 1 stays planner-less and requests exactly subgoal+actions.
    assert condition_texts(model, 0) == [
        "[outcome|success][generate|subgoal actions]",
    ]


def test_pass1_legacy_mode_is_untouched() -> None:
    """draws=None keeps the rung-(a) single-greedy path: the plain
    narrated pass, no candidate capture."""
    model = DrawsFakeModel(CANDS)
    base = cast(BijouPolicy, FakeBase(model))
    pass1 = SelfSubgoalPass1Policy(base)
    pass1.predict([labeled_item()], [7])
    assert model.candidate_calls == []
    assert pass1.candidates == {}
    assert pass1.records[7].generated_subgoal == "lower the gripper"


def test_bon_conditions_on_self_certainty_pick() -> None:
    """The bon arm picks by SC (candidate 2 here — most peaked) and
    renders it through the shared slot on the fast path (oracles iv+vi);
    no label is consulted anywhere."""
    model, pass1, bon, _ = build_draws(CANDS)
    frame = labeled_item()
    pass1.predict([frame], [7])
    bon.predict([frame], [7])
    [rendered] = condition_texts(model, 1)
    assert rendered == "[subgoal|spin in place][outcome|success][generate|actions]"
    assert bon.picks[7] == 2
    # Oracle iv: the conditioned pass never requests the subgoal.
    assert model.calls[1][1] == ()
    assert bon.collator.generate_override == ()
    # Oracle vi: one rendering path — the train-side collation of the
    # same override produces the same bytes.
    from bijou.annotations import ConditionField

    train_side = eval_collator((ConditionField.SUBGOAL, ConditionField.OUTCOME))(
        [{**labeled_item(), "condition_subgoal": "spin in place"}],
    )
    assert train_side.encoder_inputs.samples[0].condition_text == rendered


def test_bon_pick_works_without_any_label() -> None:
    """Deployment-honest: the bon arm picks identically on a label-less
    frame (rig conditions: no judge labels exist)."""
    model, pass1, bon, _ = build_draws(CANDS)
    frame = labeled_item(subgoal=None)
    pass1.predict([frame], [7])
    bon.predict([frame], [7])
    assert bon.picks[7] == 2
    [rendered] = condition_texts(model, 1)
    assert rendered == "[subgoal|spin in place][outcome|success][generate|actions]"


def test_ceil_conditions_on_token_f1_vs_true_label() -> None:
    model, pass1, _, ceil = build_draws(CANDS)
    frame = labeled_item()  # true label: "reach toward the boat"
    pass1.predict([frame], [7])
    ceil.predict([frame], [7])
    [rendered] = condition_texts(model, 1)
    assert (
        rendered == "[subgoal|reach toward the boat][outcome|success][generate|actions]"
    )
    assert ceil.picks[7] == 1


def test_ceil_labelless_frame_renders_baseline_bytes() -> None:
    """Oracle ii half: a label-less ceil row decodes the planner-less
    prompt byte-exact (the rung-(a) oracle-arm convention)."""
    model, pass1, _, ceil = build_draws(CANDS)
    frame = labeled_item(subgoal=None)
    pass1.predict([frame], [7])
    ceil.predict([frame], [7])
    assert ceil.picks[7] is None
    assert condition_texts(model, 1) == ["[outcome|success][generate|actions]"]


def test_force_empty_reproduces_baseline_bytes_on_both_arms() -> None:
    model, pass1, bon, ceil = build_draws(CANDS, force_empty=True)
    frame = labeled_item()
    pass1.predict([frame], [7])
    bon.predict([frame], [7])
    ceil.predict([frame], [7])
    assert condition_texts(model, 1) == ["[outcome|success][generate|actions]"]
    assert condition_texts(model, 2) == ["[outcome|success][generate|actions]"]


def test_name_provenance() -> None:
    _, _, bon, ceil = build_draws(CANDS)
    assert bon.name == "bijou@100000_bonsubgoal"
    assert ceil.name == "bijou@100000_ceilsubgoal"
    _, _, forced_bon, forced_ceil = build_draws(CANDS, force_empty=True)
    assert forced_bon.name == "bijou@100000_bonsubgoal_emptyhint"
    assert forced_ceil.name == "bijou@100000_ceilsubgoal_emptyhint"


# ------------------------------------- rung (b') clean-list filter


def test_clean_filter_binds_on_bon() -> None:
    """Oracle viii, SC half: the full-list SC argmax (candidate 2) is
    truncated — the filtered pick must move (to candidate 1, the best
    ELIGIBLE SC) and the rendered prompt must carry it."""
    planted = {
        "pick up the cube": [
            cand("lower the gripper", (-2.0, -2.0)),
            cand("reach toward the boat", (-4.0, -4.0)),
            cand("spin in place", (-5.0, -5.0), truncated=True),
        ],
    }
    model, pass1, bon, _ = build_draws(planted, candidate_filter="clean")
    frame = labeled_item()
    pass1.predict([frame], [7])
    bon.predict([frame], [7])
    assert bon.picks[7] == 1  # full-list argmax is 2 (see CANDS tests)
    [rendered] = condition_texts(model, 1)
    assert (
        rendered == "[subgoal|reach toward the boat][outcome|success][generate|actions]"
    )


def test_clean_filter_binds_on_ceil() -> None:
    """Oracle viii, ceiling half: the full-list ceil argmax (candidate
    1, the exact label match) is truncated — the filtered pick must
    move (F1 tie at 0 between the survivors → greedy first)."""
    planted = {
        "pick up the cube": [
            cand("lower the gripper", (-2.0, -2.0)),
            cand("reach toward the boat", (-4.0, -4.0), truncated=True),
            cand("spin in place", (-5.0, -5.0)),
        ],
    }
    model, pass1, bon, ceil = build_draws(planted, candidate_filter="clean")
    frame = labeled_item()  # true label: "reach toward the boat"
    pass1.predict([frame], [7])
    ceil.predict([frame], [7])
    assert ceil.picks[7] == 0
    assert condition_texts(model, 1) == [
        "[subgoal|lower the gripper][outcome|success][generate|actions]",
    ]
    # The bon side still sees candidate 2 (eligible) — unchanged pick.
    bon.predict([frame], [7])
    assert bon.picks[7] == 2


def test_clean_filter_all_truncated_falls_back_to_greedy() -> None:
    """Oracle ix: an all-truncated row yields the greedy candidate AS
    DECODED on both scorers — recorded as pick 0, rendered verbatim."""
    planted = {
        "pick up the cube": [
            cand("lower the gripper", (-2.0, -2.0), truncated=True),
            cand("reach toward the boat", (-4.0, -4.0), truncated=True),
            cand("spin in place", (-5.0, -5.0), truncated=True),
        ],
    }
    model, pass1, bon, ceil = build_draws(planted, candidate_filter="clean")
    frame = labeled_item()
    pass1.predict([frame], [7])
    bon.predict([frame], [7])
    ceil.predict([frame], [7])
    assert bon.picks[7] == 0
    assert ceil.picks[7] == 0
    assert condition_texts(model, 1) == [
        "[subgoal|lower the gripper][outcome|success][generate|actions]",
    ]


def test_clean_filter_without_truncation_is_identity() -> None:
    """No truncated candidate → the filter cannot change any pick (the
    banked-table invariance, oracle vii's CPU shape)."""
    _model, pass1, bon, ceil = build_draws(CANDS, candidate_filter="clean")
    frame = labeled_item()
    pass1.predict([frame], [7])
    bon.predict([frame], [7])
    ceil.predict([frame], [7])
    assert bon.picks[7] == 2  # == the unfiltered pick (CANDS tests)
    assert ceil.picks[7] == 1


def test_clean_filter_single_candidate_limit() -> None:
    """The draws-0 limit under the filter: eligible == [greedy], so the
    bit-exact carry of oracles i–vi holds structurally (pick 0 whether
    or not the lone greedy candidate is truncated)."""
    for truncated in (False, True):
        single = {
            "pick up the cube": [
                cand("lower the gripper", (-2.0, -2.0), truncated=truncated),
            ],
        }
        _model, pass1, bon, _ = build_draws(
            single,
            draws=0,
            candidate_filter="clean",
        )
        frame = labeled_item()
        pass1.predict([frame], [7])
        bon.predict([frame], [7])
        assert bon.picks[7] == 0


def test_clean_name_provenance_and_guard() -> None:
    _, _, bon, ceil = build_draws(CANDS, candidate_filter="clean")
    assert bon.name == "bijou@100000_boncleansubgoal"
    assert ceil.name == "bijou@100000_ceilcleansubgoal"
    _, _, forced_bon, _ = build_draws(
        CANDS,
        candidate_filter="clean",
        force_empty=True,
    )
    assert forced_bon.name == "bijou@100000_boncleansubgoal_emptyhint"
    with pytest.raises(SystemExit, match="candidate filter"):
        build_draws(CANDS, candidate_filter="dirty")


def test_pass2_before_pass1_is_loud() -> None:
    _, _, bon, _ = build_draws(CANDS)
    with pytest.raises(SystemExit, match="before"):
        bon.predict([labeled_item()], [7])


def test_guards() -> None:
    model = DrawsFakeModel(CANDS)
    base = cast(BijouPolicy, FakeBase(model))
    with pytest.raises(SystemExit, match=">= 0"):
        SelfSubgoalPass1Policy(base, draws=-1)
    with pytest.raises(SystemExit, match="temperature"):
        SelfSubgoalPass1Policy(base, draws=2, temperature=0.0)
    legacy = SelfSubgoalPass1Policy(base)  # draws=None
    with pytest.raises(SystemExit, match="candidates mode"):
        SelectedSubgoalPolicy(base, legacy, mode="bon")
    capturing = SelfSubgoalPass1Policy(base, draws=0)
    with pytest.raises(SystemExit, match="'bon' or 'ceil'"):
        SelectedSubgoalPolicy(base, capturing, mode="oracle")


def test_mixed_allowed_vocab_is_loud() -> None:
    mixed = {
        "pick up the cube": [
            cand("a", (-1.0,)),
            ValueCandidate(
                text="b",
                truncated=False,
                chosen_logprob=(-1.0,),
                mean_logprob=(-1.0,),
                allowed_vocab=17,
            ),
        ],
    }
    _model, pass1, bon, _ = build_draws(mixed)
    frame = labeled_item()
    pass1.predict([frame], [7])
    with pytest.raises(SystemExit, match="allowed_vocab"):
        bon.predict([frame], [7])


# --------------------------------------- real decode loop (tiny model)


def test_greedy_candidate_matches_full_pass_off_restored_prefill() -> None:
    """Oracle i's CPU half, on the REAL loop: one prefill, full greedy
    decode, restore, text-only greedy decode — the candidate text
    byte-matches the full pass's parsed subgoal, and a second restored
    decode reproduces the candidate bit-for-bit (stats included)."""
    backbone, decoder = build_with_aux()
    loaded = codec()
    memory = encode_memory(backbone)
    snapshot = decoder.cache_snapshot(memory)
    _, generations = decoder.predict_chunk(
        backbone,
        memory,
        tiny_batch(loaded),
        generate=(AuxField.SUBGOAL,),
    )
    decoder.cache_restore(memory, snapshot)
    first = decoder.decode_value_line(backbone, memory, field=AuxField.SUBGOAL)
    for generation, candidate in zip(generations, first, strict=True):
        assert (generation.subgoal or "") == candidate.text
        # Greedy argmax property: the chosen id's log-prob is >= the
        # mean over the allowed vocabulary, every step.
        assert len(candidate.chosen_logprob) == len(candidate.mean_logprob) >= 1
        assert all(
            c >= m
            for c, m in zip(
                candidate.chosen_logprob,
                candidate.mean_logprob,
                strict=True,
            )
        )
        assert candidate.allowed_vocab == decoder.config.block_base
    decoder.cache_restore(memory, snapshot)
    second = decoder.decode_value_line(backbone, memory, field=AuxField.SUBGOAL)
    assert first == second


def test_sampled_candidates_are_rng_deterministic() -> None:
    backbone, decoder = build_with_aux()
    memory = encode_memory(backbone)
    snapshot = decoder.cache_snapshot(memory)

    def sampled() -> list[ValueCandidate]:
        decoder.cache_restore(memory, snapshot)
        sampling = ARSampling(
            temperature=1.0,
            rngs=tuple(np.random.default_rng(row) for row in range(2)),
        )
        return decoder.decode_value_line(
            backbone,
            memory,
            field=AuxField.SUBGOAL,
            sampling=sampling,
        )

    assert sampled() == sampled()


def test_constrained_and_untrained_fields_are_rejected() -> None:
    backbone, decoder = build_with_aux()
    memory = encode_memory(backbone)
    with pytest.raises(ValueError, match="constrained field"):
        decoder.decode_value_line(backbone, memory, field=AuxField.HOLDING)
    with pytest.raises(ValueError, match="trained aux fields"):
        decoder.decode_value_line(backbone, memory, field=AuxField.EVENT)


# ------------------------------------------------------------- CLI


def test_cli_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = parse_cli(
        monkeypatch,
        "--subgoal-mode",
        "draws",
        "--checkpoint",
        "c",
        "--subgoal-draws",
        "8",
        "--subgoal-temperature",
        "1.0",
    )
    assert ok.subgoal_mode == "draws" and ok.subgoal_draws == 8
    assert ok.subgoal_candidate_filter is None
    preflight = parse_cli(
        monkeypatch,
        "--subgoal-mode",
        "draws",
        "--checkpoint",
        "c",
        "--subgoal-draws",
        "0",
    )
    assert preflight.subgoal_draws == 0 and preflight.subgoal_temperature is None
    # Rung (b'): the clean-list filter parses in draws mode (including
    # the draws-0 preflight limit, where it is inert by the frozen rule).
    clean = parse_cli(
        monkeypatch,
        "--subgoal-mode",
        "draws",
        "--checkpoint",
        "c",
        "--subgoal-draws",
        "8",
        "--subgoal-temperature",
        "1.0",
        "--subgoal-candidate-filter",
        "clean",
    )
    assert clean.subgoal_candidate_filter == "clean"
    for bad in (
        # draws mode without an explicit width
        ("--subgoal-mode", "draws", "--checkpoint", "c"),
        # sampled draws without an explicit temperature
        ("--subgoal-mode", "draws", "--checkpoint", "c", "--subgoal-draws", "8"),
        # temperature on the greedy-only limit
        (
            "--subgoal-mode",
            "draws",
            "--checkpoint",
            "c",
            "--subgoal-draws",
            "0",
            "--subgoal-temperature",
            "1.0",
        ),
        # negative width
        (
            "--subgoal-mode",
            "draws",
            "--checkpoint",
            "c",
            "--subgoal-draws",
            "-1",
            "--subgoal-temperature",
            "1.0",
        ),
        # draws flags outside draws mode
        ("--subgoal-mode", "self", "--checkpoint", "c", "--subgoal-draws", "8"),
        ("--checkpoint", "c", "--subgoal-temperature", "1.0"),
        ("--checkpoint", "c", "--dump-subgoal-candidates", "cands.json"),
        ("--checkpoint", "c", "--subgoal-candidate-filter", "clean"),
        (
            "--subgoal-mode",
            "self",
            "--checkpoint",
            "c",
            "--subgoal-candidate-filter",
            "clean",
        ),
        # draws mode inherits the self-mode incompatibilities
        (
            "--subgoal-mode",
            "draws",
            "--checkpoint",
            "c",
            "--subgoal-draws",
            "8",
            "--subgoal-temperature",
            "1.0",
            "--mask-state",
        ),
        (
            "--subgoal-mode",
            "draws",
            "--checkpoint",
            "c",
            "--subgoal-draws",
            "8",
            "--subgoal-temperature",
            "1.0",
            "--ar-temperature",
            "1.0",
        ),
        (
            "--subgoal-mode",
            "draws",
            "--subgoal-draws",
            "8",
            "--subgoal-temperature",
            "1.0",
        ),  # no checkpoint
    ):
        with pytest.raises(SystemExit):
            parse_cli(monkeypatch, *bad)
    # dump-subgoals and force-empty extend to draws mode.
    extended = parse_cli(
        monkeypatch,
        "--subgoal-mode",
        "draws",
        "--checkpoint",
        "c",
        "--subgoal-draws",
        "0",
        "--dump-subgoals",
        "s.json",
        "--selfsubgoal-force-empty",
    )
    assert extended.selfsubgoal_force_empty


def test_stage1_draws_counts_exact() -> None:
    """The rung-(b) stage-1 mechanical go/no-go counts on exact
    fixtures (bars from the pre-reg: (a) >= 90% rows sampled-clean,
    (b) >= 2 unique strings on >= 50% of frames, (c) no sampled string
    > 50% of the pool). Candidate 0 is greedy and never counts toward
    (a)/(c)."""
    from test_selfsubgoal import stage1

    # 2 rows, 1 greedy + 2 sampled each: row 0 diverse+clean, row 1
    # collapsed (both sampled == greedy) but clean -> a 2/2 PASS,
    # b 1/2 (50%) PASS at the boundary, c top 'x' 3/4 (75%) FAIL.
    counts = stage1.draws_counts(
        [["g", "u", "x"], ["x", "x", "x"]],
        [[False, False, False], [False, False, False]],
    )
    assert counts["a_sampled_clean_rows"] == 2 and counts["a_pass"]
    assert counts["b_diverse_rows"] == 1 and counts["b_pass"]
    assert counts["c_top_sampled"] == {"text": "x", "count": 3}
    assert counts["c_pool"] == 4 and not counts["c_pass"]

    # Truncated or empty sampled candidates break (a); the greedy slot
    # being empty does not.
    counts = stage1.draws_counts(
        [["", "u", "v"], ["g", "", "w"], ["g", "y", "z"]],
        [[False, False, False], [False, False, False], [True, False, False]],
    )
    assert counts["a_sampled_clean_rows"] == 2  # row 1 empty sampled
    assert not counts["a_pass"]  # 2/3 < 90%
    assert counts["b_diverse_rows"] == 3

    # A truncated SAMPLED candidate breaks (a) for its row.
    counts = stage1.draws_counts([["g", "u", "v"]], [[False, True, False]])
    assert counts["a_sampled_clean_rows"] == 0
