"""Self-subgoal probe instrument oracles (#6 rung (a)), pure CPU.

The pre-registration (posts/2026-08-07-prereg-selfsubgoal-probe.md)
names four abort-on-red oracles; these tests pin their CPU-checkable
halves at the prompt/collation layer — the layer where the probe's
semantics live, since both passes decode through the unchanged model
path (the real-checkpoint halves run pre-launch on the live setup, the
--mask-state precedent):

(i)   the self arm with its generated text forced EMPTY collates the
      planner-less baseline prompt byte-exact (the no-hint limit), and
      an absent/empty pass-1 generation degenerates the same way;
(ii)  the oracle arm on a label-less frame collates the baseline
      prompt byte-exact (nothing renders without a judge label);
(iii) the conditioned prompt matches the training collator's rendering
      of the same text — one rendering path (both sides ARE the shared
      Collator), pinned against future divergence;
(iv)  pass 2's request set is exactly [generate|actions] — the subgoal
      is never requested while conditioned (training's anti-copy
      coupling made condition-plus-generate an untrained context).

Plus: pass-1 provenance rows (identity triple, instruction, true
label, generated text), the loud pass-2-before-pass-1 guard, name
provenance (_narrsubgoal/_selfsubgoal/_oraclesubgoal/_emptyhint), the
CLI flag interactions, and the stage-1 stratifier. Fixture family:
tests/test_collator (real Collator, fake encoder strategy).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_collator import CHUNK, DIM, fake_inputs_collator, item

from bijou.annotations import ConditionField
from bijou.eval.cli import parse_args
from bijou.eval.policies import (
    BijouPolicy,
    SelfSubgoalPass1Policy,
    SelfSubgoalPolicy,
)
from bijou.model import SamplingMethod
from bijou.modelling.aux_text import AuxField, AuxGeneration
from bijou.modelling.interface import BijouPrediction, Collator, PromptInputs

STAGE1 = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "selfsubgoal_stage1.py"
)
_spec = importlib.util.spec_from_file_location("selfsubgoal_stage1", STAGE1)
assert _spec is not None and _spec.loader is not None
stage1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage1)
stratify = stage1.stratify


def labeled_item(
    *,
    task: str = "pick up the cube",
    subgoal: str | None = "reach toward the boat",
    episode: int = 3,
    frame: int = 40,
) -> dict[str, Any]:
    """A collatable frame with hindsight outcome + optional judge
    segment label (the test_collator item plus the label surfaces)."""
    payload = item(with_quantiles=True)
    payload["task"] = task
    payload["episode_index"] = episode
    payload["frame_index"] = frame
    payload["condition_outcome"] = "success"
    if subgoal is not None:
        payload["timestamp"] = torch.tensor(6.0)
        payload["language_persistent"] = [
            {
                "role": "assistant",
                "content": subgoal,
                "style": "subtask",
                "timestamp": 0.0,
                "camera": None,
                "tool_calls": None,
            },
        ]
    return payload


def eval_collator(condition_fields: tuple[ConditionField, ...]) -> Collator[Any]:
    """A collator built the way BijouPolicy builds eval collators
    (dropout-0, generate bracket on, fast-path override)."""
    return Collator(
        inputs=fake_inputs_collator,
        instruction=None,
        camera_filter=None,
        max_cameras=None,
        action_codec=None,
        aux=None,
        generate_bracket=True,
        generate_override=(),
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        condition_fields=condition_fields,
        condition_dropout=0.0,
        subgoal_condition_dropout=0.0,
    )


class FakeModel:
    """Stands in for the loaded model: records every batch's prompts
    and request set, generates a canned subgoal per instruction when
    pass 1 asks for one."""

    def __init__(self, subgoals: dict[str, str | None]) -> None:
        self.subgoals = subgoals
        self.calls: list[tuple[tuple[PromptInputs, ...], tuple[AuxField, ...]]] = []

    def predict_chunk(
        self,
        batch: Any,
        *,
        num_steps: int = 5,
        method: SamplingMethod = SamplingMethod.HEUN,
        generate: tuple[AuxField, ...] = (),
    ) -> BijouPrediction:
        samples = tuple(batch.encoder_inputs.samples)
        self.calls.append((samples, generate))
        generations = []
        for sample in samples:
            text = (
                self.subgoals.get(sample.instruction)
                if AuxField.SUBGOAL in generate
                else None
            )
            generations.append(
                AuxGeneration(
                    text=text or "",
                    subgoal=text,
                    holding=None,
                    progress=None,
                    event=None,
                    visible=None,
                ),
            )
        return BijouPrediction(
            actions=torch.zeros(len(samples), CHUNK, DIM),
            generations=generations,
        )


class FakeBase:
    """The BijouPolicy surface the two-pass policies consume — no
    checkpoint load; the collator is the real shared Collator."""

    def __init__(
        self,
        model: FakeModel,
        *,
        trained_condition_fields: tuple[str, ...] = ("subgoal", "outcome"),
        aux_fields: tuple[AuxField, ...] = (AuxField.SUBGOAL, AuxField.HOLDING),
        include_subgoal_condition: bool = False,
    ) -> None:
        self.name = "bijou@100000"
        self.model = model
        self.device = torch.device("cpu")
        self.sample_steps = 10
        self.method = SamplingMethod.HEUN
        self.aux_fields = aux_fields
        self.info = argparse.Namespace(condition_fields=trained_condition_fields)
        self.collator = eval_collator(
            tuple(
                ConditionField(f)
                for f in trained_condition_fields
                if ConditionField(f) is not ConditionField.SUBGOAL
                or include_subgoal_condition
            ),
        )

    def apply_overrides(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return items


def build(
    subgoals: dict[str, str | None],
    *,
    force_empty: bool = False,
) -> tuple[FakeModel, SelfSubgoalPass1Policy, SelfSubgoalPolicy]:
    model = FakeModel(subgoals)
    base = cast(BijouPolicy, FakeBase(model))
    pass1 = SelfSubgoalPass1Policy(base)
    pass2 = SelfSubgoalPolicy(base, pass1, force_empty=force_empty)
    return model, pass1, pass2


def condition_texts(model: FakeModel, call: int) -> list[str]:
    return [sample.condition_text for sample in model.calls[call][0]]


def test_pass1_is_plannerless_and_requests_subgoal() -> None:
    """Pass 1 renders NO [subgoal|…] bracket even on a labeled frame
    (planner-less context) and requests exactly [generate|subgoal
    actions]; provenance rows carry identity, instruction, TRUE label
    and the generation."""
    model, pass1, _ = build({"pick up the cube": "go toward the object"})
    frame = labeled_item()
    chunks = pass1.predict([frame], [7])
    assert len(chunks) == 1 and chunks[0].shape == (CHUNK, DIM)
    assert condition_texts(model, 0) == [
        "[outcome|success][generate|subgoal actions]",
    ]
    record = pass1.records[7]
    assert record.repo_id == "user/rig"
    assert record.episode_index == 3 and record.frame_index == 40
    assert record.instruction == "pick up the cube"
    assert record.true_subgoal == "reach toward the boat"
    assert record.generated_subgoal == "go toward the object"


def test_pass2_feeds_generation_through_slot_on_fast_path() -> None:
    """Oracles (iii)+(iv): pass 2 conditions on pass 1's text through
    the shared rendering path — byte-identical to a training-side
    collation of the same override — and its request set is exactly
    [generate|actions]."""
    model, pass1, pass2 = build({"pick up the cube": "go toward the object"})
    frame = labeled_item()
    pass1.predict([frame], [7])
    pass2.predict([frame], [7])
    [rendered] = condition_texts(model, 1)
    assert (
        rendered == "[subgoal|go toward the object][outcome|success][generate|actions]"
    )
    # One rendering path (oracle iii): the training-side collator on the
    # same override produces the same bytes.
    train_side = eval_collator((ConditionField.SUBGOAL, ConditionField.OUTCOME))(
        [{**labeled_item(), "condition_subgoal": "go toward the object"}],
    )
    assert train_side.encoder_inputs.samples[0].condition_text == rendered
    # Oracle (iv): the conditioned pass never requests the subgoal.
    assert model.calls[1][1] == ()
    assert pass2.collator.generate_override == ()
    assert "[generate|actions]" in rendered


def test_forced_empty_and_absent_generation_reproduce_baseline() -> None:
    """Oracle (i): the no-hint limit — force_empty, an empty
    generation, and a None generation all collate the planner-less
    baseline prompt byte-exact."""
    baseline = eval_collator((ConditionField.OUTCOME,))([labeled_item()])
    expected = baseline.encoder_inputs.samples[0].condition_text
    assert expected == "[outcome|success][generate|actions]"
    # force_empty on a REAL generation.
    model, pass1, pass2 = build(
        {"pick up the cube": "go toward the object"},
        force_empty=True,
    )
    pass1.predict([labeled_item()], [7])
    pass2.predict([labeled_item()], [7])
    assert condition_texts(model, 1) == [expected]
    # A pass-1 generation that came back empty (subgoal=None).
    model, pass1, pass2 = build({"pick up the cube": None})
    pass1.predict([labeled_item()], [7])
    pass2.predict([labeled_item()], [7])
    assert condition_texts(model, 1) == [expected]


def test_oracle_arm_labelless_frame_is_baseline_context() -> None:
    """Oracle (ii): with SUBGOAL in the condition fields and no
    override, a label-less frame renders nothing (baseline bytes) and
    a labeled frame renders its TRUE segment label."""
    with_slot = eval_collator((ConditionField.SUBGOAL, ConditionField.OUTCOME))
    without = eval_collator((ConditionField.OUTCOME,))
    unlabeled = labeled_item(subgoal=None)
    assert (
        with_slot([dict(unlabeled)]).encoder_inputs.samples[0].condition_text
        == without([dict(unlabeled)]).encoder_inputs.samples[0].condition_text
    )
    labeled = labeled_item()
    assert (
        with_slot([labeled]).encoder_inputs.samples[0].condition_text
        == "[subgoal|reach toward the boat][outcome|success][generate|actions]"
    )


def test_pass2_before_pass1_is_loud() -> None:
    _, _, pass2 = build({"pick up the cube": "go"})
    with pytest.raises(SystemExit, match="before"):
        pass2.predict([labeled_item()], [7])


def test_pass1_guards() -> None:
    """No subgoal aux field → loud; a subgoal-conditioned base
    (oracle-mode collator) → loud (pass 1 must be planner-less)."""
    model = FakeModel({})
    no_aux = FakeBase(model, aux_fields=(AuxField.HOLDING,))
    with pytest.raises(SystemExit, match=r"no.*subgoal to decode"):
        SelfSubgoalPass1Policy(cast(BijouPolicy, no_aux))
    conditioned = FakeBase(model, include_subgoal_condition=True)
    with pytest.raises(SystemExit, match="PLANNER-LESS"):
        SelfSubgoalPass1Policy(cast(BijouPolicy, conditioned))


def test_name_provenance() -> None:
    """A conditioned/two-pass read must never be mistakable for the
    deployment read (charter §2) — the names carry the mode."""
    _, pass1, pass2 = build({})
    assert pass1.name == "bijou@100000_narrsubgoal"
    assert pass2.name == "bijou@100000_selfsubgoal"
    _, _, forced = build({}, force_empty=True)
    assert forced.name == "bijou@100000_selfsubgoal_emptyhint"


def _parse(monkeypatch: pytest.MonkeyPatch, *extra: str) -> argparse.Namespace:
    monkeypatch.setattr(
        "sys.argv",
        ["bijou.eval", "--data", "corpus", *extra],
    )
    return parse_args()


def test_cli_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """--subgoal-mode needs a checkpoint and owns the request set /
    inference class; the helper flags need the self mode."""
    ok = _parse(monkeypatch, "--subgoal-mode", "self", "--checkpoint", "ckpt")
    assert ok.subgoal_mode == "self" and not ok.selfsubgoal_force_empty
    for bad in (
        ("--subgoal-mode", "self"),  # no checkpoint
        ("--subgoal-mode", "self", "--checkpoint", "c", "--generate", "subgoal"),
        ("--subgoal-mode", "self", "--checkpoint", "c", "--ar-temperature", "1.0"),
        ("--subgoal-mode", "self", "--checkpoint", "c", "--sample-draws", "4"),
        ("--subgoal-mode", "self", "--checkpoint", "c", "--smolvla", "s"),
        ("--subgoal-mode", "self", "--checkpoint", "c", "--mask-state"),
        (
            "--subgoal-mode",
            "oracle",
            "--checkpoint",
            "c",
            "--condition-override",
            "subgoal=go",
        ),
        ("--dump-subgoals", "s.json", "--checkpoint", "c"),  # needs self mode
        (
            "--dump-subgoals",
            "s.json",
            "--subgoal-mode",
            "oracle",
            "--checkpoint",
            "c",
        ),
        ("--selfsubgoal-force-empty", "--checkpoint", "c"),
        (
            "--selfsubgoal-force-empty",
            "--subgoal-mode",
            "oracle",
            "--checkpoint",
            "c",
        ),
    ):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, *bad)


def test_stratify_is_deterministic_and_covers_repos_and_episodes() -> None:
    identities = (
        [("user/a", 0)] * 30
        + [("user/a", 1)] * 30
        + [("user/b", 5)] * 30
        + [("user/c", 2)] * 3
    )
    picked = stratify(identities, 12, seed=0)
    assert picked == stratify(identities, 12, seed=0)  # deterministic
    assert len(picked) == len(set(picked)) == 12
    chosen = [identities[p] for p in picked]
    repos = {repo for repo, _ in chosen}
    assert repos == {"user/a", "user/b", "user/c"}  # every repo seated
    # Round-robin reaches both of user/a's episodes.
    assert {ep for repo, ep in chosen if repo == "user/a"} == {0, 1}
    assert stratify(identities, 12, seed=1) != picked  # seed moves frames
    # n above the population degenerates to everything, sorted.
    assert stratify(identities, 1000, seed=0) == list(range(len(identities)))


def test_stratify_prefers_breadth_within_a_repo() -> None:
    # 6 episodes x 10 frames, one repo: 6 seats land on 6 episodes.
    identities = [("user/a", episode) for episode in range(6) for _ in range(10)]
    chosen = [identities[p] for p in stratify(identities, 6, seed=0)]
    assert len({ep for _, ep in chosen}) == 6
