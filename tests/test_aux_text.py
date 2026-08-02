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
from typing import Any

import pytest
import torch
from lerobot.datasets.language import EVENT_ONLY_STYLES, STYLE_REGISTRY

from bijou.annotations import EVENT_STYLE
from bijou.aux_text import (
    AUX_TEMPLATE_VERSION,
    AuxField,
    AuxSpec,
    assemble_suffix,
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


def test_event_style_is_registered_by_the_contract_leaf() -> None:
    """bijou.annotations registers the style at import — the one
    definition the judge writer and the training readers share (the
    old mirrored-constant parity test dissolved with the move)."""
    assert EVENT_STYLE in STYLE_REGISTRY
    assert EVENT_STYLE in EVENT_ONLY_STYLES
    assert AUX_TEMPLATE_VERSION == 2  # header bytes unchanged since v2


def test_render_all_fields_in_template_order() -> None:
    text = decode(spec().render(judged_item()))
    assert text == "subgoal: grasp the boat\nholding: yes\nprogress: 30%\n"
    with_event = judged_item(event="boat dropped")
    assert decode(spec().render(with_event)) == (
        "subgoal: grasp the boat\nholding: yes\nprogress: 30%\nevent: boat dropped\n"
    )


def test_event_is_positives_only() -> None:
    """An event renders iff the frame carries a row (language_events is
    frame-local); frames without one render no event field at all — the
    negative is implicit in the trained transition past the field."""
    firing = judged_item(event="boat dropped")
    assert "event: boat dropped\n" in decode(spec().render(firing))
    assert "event" not in decode(spec().render(judged_item()))


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
    assert "event: boat dropped; progress regressed\n" in text


def test_event_truncation_is_bounded() -> None:
    long = judged_item(event="z" * 100)
    text = decode(spec(max_event_tokens=8).render(long))
    assert "event: " + "z" * 8 + "\n" in text
    assert text.count("z") == 8


def test_presence_based_fields() -> None:
    # Unsampled frame: holding/progress NaN -> subgoal only.
    item = judged_item(holding=math.nan, progress=math.nan)
    assert decode(spec().render(item)) == "subgoal: grasp the boat\n"
    # holding=0 renders "no"; progress rounds to whole percent.
    item = judged_item(holding=0.0, progress=0.666)
    text = decode(spec().render(item))
    assert "holding: no\n" in text
    assert "progress: 67%\n" in text


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
    assert text.startswith("subgoal: " + "x" * 8)
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
    aux_rows = [[ord(c) for c in "holding: yes\n"], []]
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
    assert decode(suffix[0, : len(aux_rows[0])].tolist()) == "holding: yes\n"
    assert suffix[0, len(aux_rows[0]) :].tolist() == [1128, 1005, 1007, 1129]
    assert (
        is_aux[0, : len(aux_rows[0])].all() and not is_aux[0, len(aux_rows[0]) :].any()
    )
    # Row 1 (no aux): actions first, block-PAD padding to width, no aux mask.
    assert suffix[1, :4].tolist() == [1128, 1003, 1129, 1129]
    assert (suffix[1, 4:] == 1129).all()
    assert not is_aux[1].any()
