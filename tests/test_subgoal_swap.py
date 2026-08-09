"""Subgoal-swap instrument oracles (#6 content read), pure CPU.

The pre-registration (posts/2026-08-09-prereg-subgoal-swap.md) names
four abort-on-red oracles; these tests pin their CPU-checkable halves
(oracle (ii)'s real-checkpoint half — identity map byte-reproducing the
banked oracle npz — runs launcher-side pre-launch, the --mask-state
precedent):

(i)   the derangement is bijective over each dataset's labeled
      episodes with no identity mappings, deterministic in the seed,
      and order-independent (per-repo seeding);
(ii)  CPU half: with the map forced to identity, a labeled frame's
      swap text equals its own active segment label and the collated
      conditioning text is byte-identical to the oracle arm's;
(iii) label-less frames pass through the swap rewrite UNTOUCHED (the
      same dict object — byte-identical to baseline by construction);
(iv)  the recorded per-frame provenance rows carry exactly what the
      slot received (donor episode + rendered text), and unswappable
      labeled frames (single-labeled-episode datasets) render an EMPTY
      slot, never the truth.

Plus: the pinned fraction-matching rule (inside-span, gap-nearest,
ties → earlier frame), the sidecar span builder's materialize-exact
semantics (chain from frame 0, active until superseded, last segment
to episode end, empty labels drop), the loud sidecar/materialized
disagreement guard, and the CLI flag interactions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from test_collator import fake_inputs_collator, item

from bijou.annotations import ConditionField
from bijou.eval.cli import parse_args
from bijou.eval.policies import BijouPolicy
from bijou.eval.subgoal_swap import (
    LabeledEpisode,
    SubgoalSwapMap,
    build_swap_map,
    derangement,
    fraction_matched_label,
    labeled_episodes,
)
from bijou.interface import Collator


def labeled_item(
    *,
    repo_id: str = "acme/pick",
    episode: int = 3,
    frame: int = 40,
    timestamp: float = 6.0,
    rows: list[tuple[float, str]] | None = None,
) -> dict[str, Any]:
    """A collatable frame carrying materialized subtask rows (None =
    label-less: no timestamp, no rows — the baseline context)."""
    payload = item(with_quantiles=True)
    payload["task"] = "pick up the cube"
    payload["repo_id"] = repo_id
    payload["episode_index"] = episode
    payload["frame_index"] = frame
    payload["condition_outcome"] = "success"
    if rows is not None:
        payload["timestamp"] = torch.tensor(timestamp)
        payload["language_persistent"] = [
            {
                "role": "assistant",
                "content": content,
                "style": "subtask",
                "timestamp": row_timestamp,
                "camera": None,
                "tool_calls": None,
            }
            for row_timestamp, content in rows
        ]
    return payload


def swap_policy(swap_map: SubgoalSwapMap) -> BijouPolicy:
    """The minimal BijouPolicy surface the swap rewrite consumes — no
    checkpoint load (the FakeBase precedent, attribute-level)."""
    policy = object.__new__(BijouPolicy)
    policy.mask_state = False
    policy.condition_override = {}
    policy.subgoal_swap = swap_map
    policy.swap_records = {}
    return policy


def one_repo_map(
    episodes: dict[int, LabeledEpisode],
    donors: dict[int, int],
    *,
    repo_id: str = "acme/pick",
    identity: bool = False,
    skipped: tuple[str, ...] = (),
) -> SubgoalSwapMap:
    return SubgoalSwapMap(
        seed=0,
        identity=identity,
        episodes={repo_id: episodes},
        donors={repo_id: donors} if donors else {},
        skipped=skipped,
    )


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


ORACLE_FIELDS = tuple(f for f in ConditionField if f.value in ("subgoal", "outcome"))


# --- oracle (i): the derangement -------------------------------------


def test_derangement_bijective_no_identity_deterministic() -> None:
    for n in (2, 3, 5, 12):
        indices = list(range(0, 3 * n, 3))
        mapping = derangement(indices, random.Random(f"tag:0:repo{n}"))
        assert sorted(mapping) == indices
        assert sorted(mapping.values()) == indices  # bijective
        assert all(donor != receiving for receiving, donor in mapping.items())
        again = derangement(indices, random.Random(f"tag:0:repo{n}"))
        assert mapping == again  # deterministic in the seed


def test_derangement_of_one_is_loud() -> None:
    with pytest.raises(SystemExit, match="skip single-labeled-episode"):
        derangement([7], random.Random("x"))


def test_build_map_seeding_is_order_independent(tmp_path: Path) -> None:
    for name in ("aa", "bb"):
        write_dataset(
            tmp_path / name,
            {
                0: (100, [(40, "reach"), (100, "grasp")]),
                1: (100, [(50, "lift"), (100, "place")]),
                2: (100, [(30, "push"), (100, "pull")]),
            },
        )
    dirs = {"aa": tmp_path / "aa", "bb": tmp_path / "bb"}
    forward = build_swap_map(dirs, seed=0)
    reversed_order = build_swap_map(
        dict(reversed(list(dirs.items()))),
        seed=0,
    )
    assert forward.donors == reversed_order.donors  # per-repo seeding


# --- the pinned fraction-matching rule -------------------------------


def test_fraction_matched_label_pinned_rule() -> None:
    donor = LabeledEpisode(
        length=100,
        # Labeled spans with a gap (an explicitly empty segment
        # 30..49) and an unlabeled reach beyond the last span is
        # impossible by construction (last span runs to length) — so
        # model the tail as labeled 'c'.
        spans=((0, 30, "a"), (50, 80, "b"), (80, 100, "c")),
    )
    assert fraction_matched_label(donor, 0.0) == "a"  # inside first
    assert fraction_matched_label(donor, 0.29) == "a"  # last 'a' frame
    assert fraction_matched_label(donor, 0.55) == "b"  # inside second
    assert fraction_matched_label(donor, 0.99) == "c"  # tail
    # Gap 30..49: target 35 is nearer span 'a' (frame 29, distance 6)
    # than span 'b' (frame 50, distance 15).
    assert fraction_matched_label(donor, 0.35) == "a"
    # Target 45: nearer 'b' (distance 5) than 'a' (distance 16).
    assert fraction_matched_label(donor, 0.45) == "b"
    # Exact tie at target 39.5 (distance 10.5 both ways) → earlier.
    assert fraction_matched_label(donor, 0.395) == "a"


def test_fraction_matched_label_before_first_span() -> None:
    donor = LabeledEpisode(length=100, spans=((60, 100, "late"),))
    assert fraction_matched_label(donor, 0.0) == "late"


# --- the sidecar span builder ----------------------------------------


def write_dataset(
    root: Path,
    episodes: dict[int, tuple[int, list[tuple[int, str]]]],
    *,
    prompt_hash: str = "h1",
    model: str = "m1",
    extra_records: list[dict[str, Any]] | None = None,
) -> None:
    """A metadata-only fixture dataset: ``episodes`` maps episode index
    to (length, [(until_frame, subgoal), …])."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "judge_annotations.json").write_text(
        json.dumps({"prompt_hash": prompt_hash, "model_filter": model}),
    )
    records = [
        {
            "episode_index": index,
            "model": model,
            "prompt_hash": prompt_hash,
            "judged_at": "2026-08-01 00:00:00",
            "num_timesteps": 10,
            "max_image_dim": 512,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "judgment": {
                "subgoals": [
                    {"until_frame": until, "subgoal": text} for until, text in segments
                ],
            },
        }
        for index, (_, segments) in episodes.items()
    ]
    (meta / "judgments.json").write_text(
        json.dumps({"judgments": records + (extra_records or [])}),
    )
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps({"episode_index": index, "length": length})
            for index, (length, _) in episodes.items()
        ),
    )


def test_labeled_episodes_materialize_exact_semantics(tmp_path: Path) -> None:
    write_dataset(
        tmp_path,
        {
            # Chain from 0; empty middle label drops; LAST span runs to
            # episode end (200) despite until_frame 150.
            0: (200, [(30, "reach"), (60, ""), (150, "grasp")]),
            # No non-empty labels at all -> not a labeled episode.
            1: (50, [(50, "")]),
        },
    )
    episodes = labeled_episodes(tmp_path)
    assert set(episodes) == {0}
    assert episodes[0] == LabeledEpisode(
        length=200,
        spans=((0, 30, "reach"), (60, 200, "grasp")),
    )


def test_labeled_episodes_stamp_selection(tmp_path: Path) -> None:
    write_dataset(
        tmp_path,
        {0: (100, [(100, "old")])},
        extra_records=[
            {
                # Same key, later judged_at -> wins.
                "episode_index": 0,
                "model": "m1",
                "prompt_hash": "h1",
                "judged_at": "2026-08-02 00:00:00",
                "num_timesteps": 10,
                "max_image_dim": 512,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "judgment": {"subgoals": [{"until_frame": 100, "subgoal": "new"}]},
            },
            {
                # Different prompt hash -> excluded by the stamp.
                "episode_index": 0,
                "model": "m1",
                "prompt_hash": "OTHER",
                "judged_at": "2026-08-03 00:00:00",
                "num_timesteps": 10,
                "max_image_dim": 512,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "judgment": {"subgoals": [{"until_frame": 100, "subgoal": "wrong"}]},
            },
        ],
    )
    episodes = labeled_episodes(tmp_path)
    assert episodes[0].spans == ((0, 100, "new"),)


def test_labeled_episodes_unstamped_is_empty(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir(parents=True)
    assert labeled_episodes(tmp_path) == {}


def test_build_map_single_labeled_episode_skips(tmp_path: Path) -> None:
    write_dataset(tmp_path / "solo", {0: (100, [(100, "only")])})
    write_dataset(
        tmp_path / "pair",
        {
            0: (100, [(100, "reach")]),
            1: (100, [(100, "grasp")]),
        },
    )
    swap_map = build_swap_map(
        {"solo": tmp_path / "solo", "pair": tmp_path / "pair"},
        seed=0,
    )
    assert swap_map.skipped == ("solo",)
    assert "solo" not in swap_map.donors  # no donor -> EMPTY renders
    assert swap_map.donors["pair"] == {0: 1, 1: 0}
    # Identity mode maps every labeled episode to itself, solo included.
    identity = build_swap_map(
        {"solo": tmp_path / "solo", "pair": tmp_path / "pair"},
        seed=0,
        identity=True,
    )
    assert identity.skipped == ()
    assert identity.donors["solo"] == {0: 0}
    assert identity.donors["pair"] == {0: 0, 1: 1}


# --- the policy rewrite ----------------------------------------------


def test_labelless_frame_untouched_oracle_iii() -> None:
    swap_map = one_repo_map(
        {3: LabeledEpisode(length=100, spans=((0, 100, "x"),))},
        {3: 3},
    )
    policy = swap_policy(swap_map)
    payload = labeled_item(rows=None)  # no judge label
    (out,) = policy.apply_overrides([payload])
    assert out is payload  # the SAME dict — byte-identical downstream
    assert policy.swap_records == {}


def test_identity_map_reproduces_oracle_prompt_oracle_ii() -> None:
    rows = [(0.0, "reach toward the boat"), (2.0, "grasp the hull")]
    swap_map = one_repo_map(
        {
            3: LabeledEpisode(
                length=100,
                spans=((0, 60, "reach toward the boat"), (60, 100, "grasp the hull")),
            ),
        },
        {3: 3},
        identity=True,
    )
    policy = swap_policy(swap_map)
    # timestamp 6.0 with 30 fps-ish rows: frame 40 sits in span 1 via
    # fraction 0.4 while active_at(6.0) resolves row 2 — use a frame
    # whose span and active row AGREE (the real invariant: identical
    # timestamp/frame provenance), here frame 70 / timestamp 6.0 with
    # row 2 active and span (60, 100) containing 70.
    payload = labeled_item(frame=70, timestamp=6.0, rows=rows)
    (swapped,) = policy.apply_overrides([payload])
    assert swapped["condition_subgoal"] == "grasp the hull"
    collator = eval_collator(ORACLE_FIELDS)
    oracle_batch = collator([labeled_item(frame=70, timestamp=6.0, rows=rows)])
    swap_batch = collator([swapped])
    oracle_text = oracle_batch.encoder_inputs.samples[0].condition_text
    swap_text = swap_batch.encoder_inputs.samples[0].condition_text
    assert oracle_text == swap_text  # byte-identical conditioning
    record = policy.swap_records[("acme/pick", 3, 70)]
    assert record.donor_episode_index == 3
    assert record.rendered_subgoal == "grasp the hull"
    assert record.true_subgoal == "grasp the hull"


def test_swap_renders_donor_fraction_matched_label() -> None:
    swap_map = one_repo_map(
        {
            3: LabeledEpisode(length=100, spans=((0, 100, "reach"),)),
            7: LabeledEpisode(
                length=200,
                spans=((0, 100, "donor early"), (100, 200, "donor late")),
            ),
        },
        {3: 7, 7: 3},
    )
    policy = swap_policy(swap_map)
    payload = labeled_item(frame=90, timestamp=6.0, rows=[(0.0, "reach")])
    (swapped,) = policy.apply_overrides([payload])
    # p = 90/100 -> donor frame 180 -> "donor late".
    assert swapped["condition_subgoal"] == "donor late"
    record = policy.swap_records[("acme/pick", 3, 90)]
    assert record.donor_episode_index == 7
    assert record.true_subgoal == "reach"
    assert record.rendered_subgoal == "donor late"


def test_unswappable_labeled_frame_renders_empty_slot() -> None:
    swap_map = one_repo_map(
        {3: LabeledEpisode(length=100, spans=((0, 100, "only"),))},
        {},
        skipped=("acme/pick",),
    )
    policy = swap_policy(swap_map)
    payload = labeled_item(frame=40, timestamp=6.0, rows=[(0.0, "only")])
    (swapped,) = policy.apply_overrides([payload])
    assert swapped["condition_subgoal"] == ""  # never the truth
    record = policy.swap_records[("acme/pick", 3, 40)]
    assert record.donor_episode_index is None
    assert record.rendered_subgoal == ""
    # An explicit EMPTY override renders nothing — baseline context.
    collator = eval_collator(ORACLE_FIELDS)
    baseline = collator([labeled_item(rows=None)])
    swapped_batch = collator([swapped])
    assert (
        baseline.encoder_inputs.samples[0].condition_text
        == swapped_batch.encoder_inputs.samples[0].condition_text
    )


def test_sidecar_materialized_disagreement_is_loud() -> None:
    swap_map = one_repo_map(
        {7: LabeledEpisode(length=100, spans=((0, 100, "x"),))},
        {},
    )
    policy = swap_policy(swap_map)
    # Episode 3 carries a materialized label but is absent from the map.
    payload = labeled_item(episode=3, frame=40, timestamp=6.0, rows=[(0.0, "x")])
    with pytest.raises(SystemExit, match="disagree"):
        policy.apply_overrides([payload])


def test_frame_outside_meta_length_is_loud() -> None:
    swap_map = one_repo_map(
        {3: LabeledEpisode(length=10, spans=((0, 10, "x"),))},
        {3: 3},
        identity=True,
    )
    policy = swap_policy(swap_map)
    payload = labeled_item(frame=40, timestamp=6.0, rows=[(0.0, "x")])
    with pytest.raises(SystemExit, match="outside"):
        policy.apply_overrides([payload])


# --- CLI flag interactions -------------------------------------------


def _parse(monkeypatch: pytest.MonkeyPatch, *extra: str) -> argparse.Namespace:
    monkeypatch.setattr(
        "sys.argv",
        ["bijou.eval", "--data", "/tmp/x", "--checkpoint", "/tmp/ckpt", *extra],
    )
    return parse_args()


def test_cli_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--subgoal-swap-seed", "0")  # no oracle mode
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--subgoal-mode",
            "self",
            "--subgoal-swap-seed",
            "0",
        )
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--subgoal-swap-identity")  # no seed
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--dump-subgoal-swaps", "/tmp/d.json")  # no seed
    args = _parse(
        monkeypatch,
        "--subgoal-mode",
        "oracle",
        "--subgoal-swap-seed",
        "0",
        "--subgoal-swap-identity",
        "--dump-subgoal-swaps",
        "/tmp/d.json",
    )
    assert args.subgoal_swap_seed == 0
    assert args.subgoal_swap_identity
