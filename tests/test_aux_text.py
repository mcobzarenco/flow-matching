"""Aux text rendering + suffix assembly (commit ① of the aux feature).

Pure CPU/synthetic: a stub tokenizer (1 token per character) stands in
for the HF tokenizer, and items mirror the REAL annotated rig-v2
surfaces (language_persistent row dicts, NaN-masked float32 scalars) as
audited 2026-08-02. The prompt-hash pin is the drift tripwire: it must
equal bijou.judge.PROMPT_HASH so a judge-prompt change fails here
instead of silently mixing label distributions.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from bijou.aux_text import (
    AUX_TEMPLATE_VERSION,
    PINNED_PROMPT_HASH,
    AuxField,
    AuxSpec,
    assemble_suffix,
)
from bijou.judge import PROMPT_HASH


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
        "fields": (AuxField.SUBGOAL, AuxField.HOLDING, AuxField.PROGRESS),
        "annotated_repos": frozenset({"mcobzarenco/so101_pick_place_v2"}),
        "block_base": 1000,
        "max_subgoal_tokens": 16,
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
) -> dict[str, Any]:
    """Mirrors the real item surfaces (row-dict language columns,
    NaN-masked scalars, repo_id attached by StatsAttachedDataset)."""
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
        "language_events": [],
        "annotation.holding": torch.tensor(holding),
        "annotation.progress": torch.tensor(progress),
    }


def decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


def test_pinned_hash_tracks_judge_prompt() -> None:
    """THE drift tripwire: a judge-prompt change must fail loudly here,
    not silently mix label distributions in training."""
    assert PINNED_PROMPT_HASH == PROMPT_HASH
    assert AUX_TEMPLATE_VERSION == 2  # the opener-prefixed suffix format


def test_render_all_fields_in_template_order() -> None:
    text = decode(spec().render(judged_item()))
    assert text == "subgoal: grasp the boat\nholding: yes\nprogress: 30%\n"


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
        )


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
