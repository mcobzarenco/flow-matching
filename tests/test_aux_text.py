"""Aux text rendering + suffix assembly.

Pure CPU/synthetic: a stub tokenizer (1 token per character) stands in
for the HF tokenizer, and items mirror the REAL annotated rig-v2
surfaces (language_persistent/language_events row dicts, NaN-masked
float32 scalars) as audited 2026-08-02. Label provenance is the
dataset's own stamp (no code-level prompt-hash pin — see
docs/episode-annotations.md). EVENT_STYLE is defined ONCE in
bijou.annotations (the artifact-contract leaf both the judge writer and
these renderers import), which also registers the lerobot style.
"""

from __future__ import annotations

import math
from typing import Any, override

import pytest
import torch
from lerobot.datasets.language import EVENT_ONLY_STYLES, STYLE_REGISTRY

from bijou.annotations import EVENT_STYLE
from bijou.aux_text import (
    AUX_TEMPLATE_VERSION,
    AuxDecodeConfig,
    AuxField,
    AuxSpec,
    assemble_suffix,
    build_aux_runtime,
)


class CharTokenizer:
    """One token per character, id = ord(c): deterministic, reversible,
    and guaranteed sub-block ids for ASCII (< any realistic block_base)."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def spec(**overrides: Any) -> AuxSpec:
    kwargs: dict[str, Any] = {
        "tokenizer_dir": "unused",
        "fields": tuple(AuxField),
        "annotated_repos": frozenset({"mcobzarenco/so101_pick_place_v2"}),
        "block_base": 1000,
        "dropout": 0.0,
        "max_subgoal_tokens": 16,
        "max_event_tokens": 24,
    }
    kwargs.update(overrides)
    built = AuxSpec(**kwargs)
    built._tokenizer = CharTokenizer()  # test injection
    return built


def judged_item(
    *,
    holding: float = 1.0,
    progress: float = 0.3,
    subgoal: str = "grasp the boat",
    event: str | None = None,
) -> dict[str, Any]:
    """Mirrors the real item surfaces (row-dict language columns,
    NaN-masked scalars, repo_id attached by StatsAttachedDataset).
    ``language_events`` is FRAME-LOCAL in lerobot items — a row is
    present iff an event fired on this exact frame (emitted_at ignores
    event-row timestamps by design) — so ``event`` here simulates "this
    frame is the firing frame"."""
    events = (
        [
            {
                "role": "assistant",
                "content": event,
                "style": EVENT_STYLE,
                "timestamp": 6.0,
                "camera": None,
                "tool_calls": None,
            },
        ]
        if event is not None
        else []
    )
    return {
        "repo_id": "mcobzarenco/so101_pick_place_v2",
        "timestamp": torch.tensor(6.0),
        "language_persistent": [
            {
                "role": "assistant",
                "content": "reach toward the toy boat",
                "style": "subtask",
                "timestamp": 0.0,
                "camera": None,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": subgoal,
                "style": "subtask",
                "timestamp": 5.13,
                "camera": None,
                "tool_calls": None,
            },
        ],
        "language_events": events,
        "annotation.holding": torch.tensor(holding),
        "annotation.progress": torch.tensor(progress),
    }


def decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


def test_condition_value_mappings() -> None:
    """Hindsight labels → requestable conditioning values: UNCLEAR is
    unlabeled (not a requestable behavior); smoothness buckets are
    8-10/5-7/1-4 (recorded thresholds — v0 base rates in the doc)."""
    from bijou.annotations import TaskCompletion, outcome_text, smoothness_bucket

    assert outcome_text(TaskCompletion.YES) == "success"
    assert outcome_text(TaskCompletion.PARTIAL) == "partial"
    assert outcome_text(TaskCompletion.NO) == "failure"
    assert outcome_text(TaskCompletion.UNCLEAR) is None
    assert [smoothness_bucket(s) for s in (10, 8, 7, 5, 4, 1)] == [
        "high",
        "high",
        "medium",
        "medium",
        "low",
        "low",
    ]


def test_event_style_is_registered_by_the_contract_leaf() -> None:
    """bijou.annotations registers the style at import — the one
    definition the judge writer and the training readers share (the
    old mirrored-constant parity test dissolved with the move)."""
    assert EVENT_STYLE in STYLE_REGISTRY
    assert EVENT_STYLE in EVENT_ONLY_STYLES
    # v3 = bracket headers + positional visibility (forced-scaffold
    # boundary property; measured on the real E2B tokenizer 2026-08-03).
    assert AUX_TEMPLATE_VERSION == 3


def test_render_all_fields_in_template_order() -> None:
    text = decode(spec().render(judged_item()))
    assert text == "[subgoal]grasp the boat\n[holding]yes\n[progress]30%\n"
    with_event = judged_item(event="boat dropped")
    assert decode(spec().render(with_event)) == (
        "[subgoal]grasp the boat\n[holding]yes\n[progress]30%\n[event]boat dropped\n"
    )


def test_event_is_positives_only() -> None:
    """An event renders iff the frame carries a row (language_events is
    frame-local); frames without one render no event field at all — the
    negative is implicit in the trained transition past the field."""
    firing = judged_item(event="boat dropped")
    assert "[event]boat dropped\n" in decode(spec().render(firing))
    assert "event" not in decode(spec().render(judged_item()))


def test_render_suppresses_subgoal_for_prompt_conditioning() -> None:
    """Anti-copy coupling (C2): when the collator put the subgoal in
    the PROMPT, the aux segment must not train to predict it — the
    remaining fields render unchanged."""
    item = judged_item()
    text = decode(spec().render(item, suppress_subgoal=True))
    assert "subgoal" not in text
    assert "[holding]yes\n" in text and "[progress]30%\n" in text


def test_multi_event_frames_render_all_events() -> None:
    """Two events can fire on ONE frame (drop + progress regression —
    real corpus data; the single-row resolver crashed a corpus run on
    2026-08-02): render them all, joined."""
    both = judged_item(event="boat dropped")
    both["language_events"].append(
        {
            "role": "assistant",
            "content": "progress regressed",
            "style": EVENT_STYLE,
            "timestamp": 6.0,
            "camera": None,
            "tool_calls": None,
        },
    )
    # CharTokenizer is 1 token/char — use a generous cap so the joined
    # text survives (the real tokenizer packs ~4 chars/token).
    text = decode(spec(max_event_tokens=64).render(both))
    assert "[event]boat dropped; progress regressed\n" in text


def test_event_truncation_is_bounded() -> None:
    long = judged_item(event="z" * 100)
    text = decode(spec(max_event_tokens=8).render(long))
    assert "[event]" + "z" * 8 + "\n" in text
    assert text.count("z") == 8


def test_visibility_renders_prompt_positions_and_true_negatives() -> None:
    """Sampled frames render which cameras see object/gripper as PROMPT
    POSITIONS — indices into the (kind, short name) camera order, never
    kind or short-name text (names collide and leak dataset-internal
    vocabulary); all-zeros is a TRUE 'none' (occlusion is signal); NaN
    frames render nothing."""
    item = judged_item()
    # Slot (storage) order = sorted short names: gripper_cam, overhead.
    # PROMPT order sorts by (kind, name): overhead(top)=0,
    # gripper_cam(wrist)=1 — deliberately the REVERSE of storage order.
    item["camera_kinds"] = {"overhead": "top", "gripper_cam": "wrist"}
    item["annotation.visible_object"] = torch.tensor([0.0, 1.0])
    item["annotation.visible_gripper"] = torch.tensor([1.0, 1.0])
    text = decode(spec().render(item))
    assert "[visible]object 0; gripper 0,1\n" in text

    item["annotation.visible_object"] = torch.tensor([0.0, 0.0])
    assert "[visible]object none; gripper 0,1\n" in decode(spec().render(item))

    item["annotation.visible_object"] = torch.tensor([math.nan, math.nan])
    assert "visible" not in decode(spec().render(item))


def test_visibility_handles_single_camera_scalars_and_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # lerobot stores shape-(1,) features as 0-d scalars.
    item = judged_item()
    item["camera_kinds"] = {"cam": "front"}
    item["annotation.visible_object"] = torch.tensor(1.0)
    item["annotation.visible_gripper"] = torch.tensor(0.0)
    assert "[visible]object 0; gripper none\n" in decode(spec().render(item))
    # Kinds/vector slot-count disagreement renders nothing (data issue,
    # not guessed through) — LOUDLY, once per dataset per worker.
    item["camera_kinds"] = {"cam": "front", "ghost": "top"}
    item["repo_id"] = "someone/mismatched"
    mismatched = spec(annotated_repos=frozenset({"someone/mismatched"}))
    assert "visible" not in decode(mismatched.render(item))
    assert "visible" not in decode(mismatched.render(item))
    out = capsys.readouterr().out
    assert out.count("someone/mismatched") == 1  # once, not per frame
    assert "re-materialized" in out


def test_visibility_indices_follow_kind_order_and_dropout_cannot_move_them() -> None:
    """Prompt order = (kind, short name): a wrist camera whose short
    name sorts FIRST still renders at the LAST index; camera_prompt_order
    ignores dropout by construction (it takes the raw map)."""
    from bijou.aux_text import camera_prompt_order

    kinds = {"a_wrist": "wrist", "z_top": "top"}
    assert camera_prompt_order(kinds, sorted(kinds)) == ["z_top", "a_wrist"]
    item = judged_item()
    item["camera_kinds"] = kinds
    # Storage order (sorted names): a_wrist slot 0, z_top slot 1.
    item["annotation.visible_object"] = torch.tensor([1.0, 0.0])
    item["annotation.visible_gripper"] = torch.tensor([0.0, 1.0])
    text = decode(spec().render(item))
    # a_wrist is prompt index 1 (wrist sorts after top), z_top index 0.
    assert "[visible]object 1; gripper 0\n" in text


def test_presence_based_fields() -> None:
    # Unsampled frame: holding/progress NaN -> subgoal only.
    item = judged_item(holding=math.nan, progress=math.nan)
    assert decode(spec().render(item)) == "[subgoal]grasp the boat\n"
    # holding=0 renders "no"; progress rounds to whole percent.
    item = judged_item(holding=0.0, progress=0.666)
    text = decode(spec().render(item))
    assert "[holding]no\n" in text
    assert "[progress]67%\n" in text


def test_unjudged_sources_render_nothing() -> None:
    # Dataset whose stamp failed verification (not in annotated_repos).
    other = judged_item()
    other["repo_id"] = "someone/unjudged"
    assert spec().render(other) == []
    # Annotated repo but no columns at all (community mixed corpus).
    bare = {
        "repo_id": "mcobzarenco/so101_pick_place_v2",
        "timestamp": torch.tensor(1.0),
    }
    assert spec().render(bare) == []


def test_subgoal_truncation_is_bounded() -> None:
    long = judged_item(subgoal="x" * 100)
    text = decode(spec(max_subgoal_tokens=8).render(long))
    assert text.startswith("[subgoal]" + "x" * 8)
    assert text.count("x") == 8


def test_field_subset_keeps_order_and_rejects_reorder() -> None:
    only = spec(fields=(AuxField.SUBGOAL, AuxField.HOLDING))
    text = decode(only.render(judged_item()))
    assert "progress" not in text
    with pytest.raises(ValueError, match="template order"):
        AuxSpec(
            tokenizer_dir="unused",
            fields=(AuxField.HOLDING, AuxField.SUBGOAL),
            annotated_repos=frozenset(),
            block_base=1000,
            dropout=0.0,
        )


def test_mode_dropout_drops_whole_samples_deterministically() -> None:
    """dropout=p renders a labeled sample as unlabeled with probability
    p (the sample then trains [ACT]); draws come from a generator seeded
    from torch.initial_seed(), so a fixed torch seed fixes the pattern."""
    item = judged_item()
    # Boundary values need no seeding: 0 keeps everything (the probe
    # collator's clone), and dropout=1 is rejected outright.
    assert spec(dropout=0.0).render(item) != []
    with pytest.raises(ValueError, match="outside"):
        spec(dropout=1.0)
    torch.manual_seed(1234)
    dropped = spec(dropout=0.5)
    pattern = [dropped.render(item) == [] for _ in range(32)]
    assert any(pattern) and not all(pattern)  # both outcomes occur
    torch.manual_seed(1234)
    again = spec(dropout=0.5)
    assert [again.render(item) == [] for _ in range(32)] == pattern
    # Unlabeled sources are unaffected (already empty).
    other = judged_item()
    other["repo_id"] = "someone/unjudged"
    assert dropped.render(other) == []


def test_assemble_suffix_layout_and_masks() -> None:
    aux_rows = [[ord(c) for c in "[holding]yes\n"], []]
    action_tokens = torch.tensor(
        [[128, 5, 7, 129], [128, 3, 129, 129]],
    )  # BOA=128 PAD=129
    suffix, is_aux = assemble_suffix(
        aux_rows,
        action_tokens,
        block_base=1000,
        codec_pad=129,
    )
    width = len(aux_rows[0]) + 4
    assert suffix.shape == (2, width)
    # Row 0: aux text ids then block-offset actions.
    assert decode(suffix[0, : len(aux_rows[0])].tolist()) == "[holding]yes\n"
    assert suffix[0, len(aux_rows[0]) :].tolist() == [1128, 1005, 1007, 1129]
    assert (
        is_aux[0, : len(aux_rows[0])].all() and not is_aux[0, len(aux_rows[0]) :].any()
    )
    # Row 1 (no aux): actions first, block-PAD padding to width, no aux mask.
    assert suffix[1, :4].tolist() == [1128, 1003, 1129, 1129]
    assert (suffix[1, 4:] == 1129).all()
    assert not is_aux[1].any()


def test_assemble_suffix_rejects_reserved_tail_ids() -> None:
    """Aux ids must stay below the WHOLE reserved tail: an id in the
    mode strip [block_base−2, block_base) would silently embed as
    [ACT]/[AUX] and become a CE target (train mode prediction)."""
    for bad in (998, 999, 1000):  # both mode ids and the block base
        with pytest.raises(ValueError, match="mode_base"):
            assemble_suffix(
                [[bad]],
                torch.tensor([[128]]),
                block_base=1000,
                codec_pad=129,
            )


class MergingTokenizer(CharTokenizer):
    """CharTokenizer with ONE BPE-style merge, ']'+'y' → a single id —
    the v2 ``" yes"`` boundary merge relocated to the v3 template, so
    the tripwire has something to trip on."""

    MERGED = 700  # any id outside the ASCII range the base stub emits

    @override
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == ord("]") and ids[i + 1] == ord("y"):
                out.append(self.MERGED)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out


def test_build_aux_runtime_asserts_the_boundary_property() -> None:
    """The forced-scaffold property (encode(header) + encode(value) ==
    encode(header + value)) is asserted at construction — the v2 bug
    class (samples_holding_acc scored off-manifold ids) can only recur
    as a LOUD construction error under a future tokenizer swap."""
    config = AuxDecodeConfig(
        template_version=AUX_TEMPLATE_VERSION,
        fields=(AuxField.SUBGOAL, AuxField.HOLDING),
        prompt_hash="hash",
        judge_model="judge",
    )
    runtime = build_aux_runtime(config, CharTokenizer())
    assert runtime.header_ids[AuxField.HOLDING] == tuple(ord(c) for c in "[holding]")
    assert runtime.value_candidates[AuxField.HOLDING] == (
        tuple(ord(c) for c in "no"),
        tuple(ord(c) for c in "yes"),
    )
    with pytest.raises(SystemExit, match="boundary"):
        build_aux_runtime(config, MergingTokenizer())
