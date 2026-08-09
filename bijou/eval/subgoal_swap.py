"""The subgoal-swap content probe: the swap-map builder and donor lookup.

The swap arm re-runs the
oracle subgoal-conditioning arm with each labeled frame's segment label replaced by
a DIFFERENT episode's label — format-valid, content-wrong — so the read
separates "the slot consumes content" from "any plausible words help".

The pinned mapping rule (frozen before implementation):

- Episode-level seeded derangement over the labeled episodes of each
  dataset (``repo_id``) — never cross-dataset (plausible-but-wrong
  beats implausible-and-wrong). Sattolo's algorithm: a single seeded
  cycle, bijective, no fixed points by construction.
- For a receiving frame at fractional position p = frame/episode_len,
  the donor text is the donor episode's label at its labeled frame
  nearest to p·donor_len (ties → the earlier frame).
- Datasets with a single labeled episode contribute no swapped frames:
  their labeled frames render NO subgoal (an empty slot, never the
  truth — the swap arm's conditioned frames must be 100% content-wrong)
  and the count is recorded.
- Frames that are label-less under the oracle arm stay label-less
  (byte-identical to baseline).

The builder is a pure-CPU metadata pre-pass: segment spans come from
``meta/judgments.json`` under the dataset's own materialization stamp
(``meta/judge_annotations.json`` — the same record selection
``bijou.judge.materialize`` used to write the language rows the eval
items carry), episode lengths from the LeRobot ``meta/episodes``
tables. No frame or video data is touched. The span model reproduces
the materialized persistent-row semantics exactly (segments tile from
frame 0; a segment stays active until superseded; the last segment runs
to episode end), so forcing the map to identity reproduces the banked
oracle arm byte-exactly — the launcher-side pre-launch check runs
precisely that.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..annotations import load_sidecar

# Seed-domain tag: the derangement RNG is keyed per (tag, seed, repo)
# like the episode-holdout split, so dataset enumeration order can
# never change any dataset's mapping.
_SEED_TAG = "subgoal-swap"


@dataclass(frozen=True, slots=True)
class LabeledEpisode:
    """One labeled episode's piecewise-constant label structure.

    ``spans`` are half-open ``(start_frame, end_frame, text)`` intervals
    in 0-based within-episode frame indices, non-empty text only,
    strictly increasing; the materialized-row semantics (active until
    superseded; last segment to episode end) are already applied, so a
    gap between spans means an explicitly empty segment label."""

    length: int
    spans: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True, slots=True)
class SubgoalSwapMap:
    """The full pinned mapping for one eval run (deterministic in
    ``seed``): per-dataset labeled episodes, donor assignment, and the
    datasets skipped for having a single labeled episode."""

    seed: int
    identity: bool
    # repo_id -> episode_index -> label structure (labeled episodes only).
    episodes: dict[str, dict[int, LabeledEpisode]]
    # repo_id -> receiving episode -> donor episode ({} = skipped repo).
    donors: dict[str, dict[int, int]]
    # repo_ids whose single labeled episode admits no derangement.
    skipped: tuple[str, ...]

    def donor_text(self, repo_id: str, episode: int, frame: int) -> str | None:
        """The pinned swap text for a labeled receiving frame, or None
        when the dataset contributed no swapped frames (single labeled
        episode — the caller renders an EMPTY slot and records it)."""
        receiving = self.episodes[repo_id][episode]
        if not 0 <= frame < receiving.length:
            raise SystemExit(
                f"{repo_id} episode {episode}: frame {frame} outside "
                f"[0, {receiving.length}) — the meta episode length "
                "disagrees with the loaded data; refusing to guess a "
                "fraction",
            )
        donor_index = self.donors.get(repo_id, {}).get(episode)
        if donor_index is None:
            return None
        donor = self.episodes[repo_id][donor_index]
        return fraction_matched_label(donor, frame / receiving.length)


def fraction_matched_label(donor: LabeledEpisode, p: float) -> str:
    """The donor's label at its labeled frame nearest to p·donor_len,
    ties → the earlier frame (the pinned rule). A frame
    inside a span is its own nearest labeled frame, so an identity
    mapping reproduces the active segment label exactly."""
    target = p * donor.length
    best: tuple[float, int, str] | None = None
    for start, end, text in donor.spans:
        # Distance from the real-valued target to the nearest integer
        # frame of the span (frames start..end-1); 0 inside the span.
        distance = max(start - target, target - (end - 1), 0.0)
        if best is None or (distance, start) < (best[0], best[1]):
            best = (distance, start, text)
    assert best is not None  # labeled episodes have >= 1 span
    return best[2]


def episode_lengths(dataset_dir: Path) -> dict[int, int]:
    """Within-episode frame counts from the LeRobot episodes metadata
    (v3 parquet layout, ``episodes.jsonl`` fallback) — metadata only."""
    lengths: dict[int, int] = {}
    episodes_dir = dataset_dir / "meta" / "episodes"
    if episodes_dir.is_dir():
        import pyarrow.parquet as pq

        for path in sorted(episodes_dir.rglob("*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "length"])
            for row in table.to_pylist():
                lengths[int(row["episode_index"])] = int(row["length"])
    else:
        jsonl = dataset_dir / "meta" / "episodes.jsonl"
        if not jsonl.exists():
            raise SystemExit(
                f"{dataset_dir}: neither meta/episodes/ nor "
                "meta/episodes.jsonl — cannot resolve episode lengths",
            )
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
    if not lengths:
        raise SystemExit(f"{dataset_dir}: episodes metadata is empty")
    return lengths


def labeled_episodes(dataset_dir: Path) -> dict[int, LabeledEpisode]:
    """Labeled episodes (≥1 non-empty segment label) under the
    dataset's own materialization stamp — the exact record selection
    ``bijou.judge.materialize`` projects into the language rows the
    eval items carry. {} when the dataset is unstamped/unjudged."""
    stamp_path = dataset_dir / "meta" / "judge_annotations.json"
    if not stamp_path.exists():
        return {}
    stamp = json.loads(stamp_path.read_text())
    prompt_hash = str(stamp.get("prompt_hash") or "")
    model = str(stamp.get("model_filter") or (stamp.get("models") or [""])[0])
    if not prompt_hash or not model:
        raise SystemExit(
            f"{stamp_path}: stamp lacks prompt_hash/model — cannot pick "
            "the blessed judgment records",
        )
    chosen: dict[int, Any] = {}
    chosen_at: dict[int, str] = {}
    for record in load_sidecar(dataset_dir):
        if record.prompt_hash != prompt_hash or record.model != model:
            continue
        if (
            record.episode_index not in chosen
            or record.judged_at > chosen_at[record.episode_index]
        ):
            chosen[record.episode_index] = record.judgment
            chosen_at[record.episode_index] = record.judged_at
    if not chosen:
        return {}
    lengths = episode_lengths(dataset_dir)
    episodes: dict[int, LabeledEpisode] = {}
    for episode_index, judgment in sorted(chosen.items()):
        length = lengths.get(episode_index)
        if length is None:
            raise SystemExit(
                f"{dataset_dir}: judged episode {episode_index} missing "
                "from episodes metadata",
            )
        spans = _label_spans(judgment, length, dataset_dir, episode_index)
        if spans:
            episodes[episode_index] = LabeledEpisode(length=length, spans=spans)
    return episodes


def _label_spans(
    judgment: dict[str, Any],
    length: int,
    dataset_dir: Path,
    episode_index: int,
) -> tuple[tuple[int, int, str], ...]:
    """Segment dicts → non-empty label spans under the materialized
    persistent-row semantics (mirrors ``materialize._persistent_rows``
    plus lerobot's active-until-superseded resolution): segment k
    starts at the previous ``until_frame`` (chain from 0), is
    superseded by segment k+1's start, and the LAST segment stays
    active to episode end regardless of its own ``until_frame``."""
    segments = judgment.get("subgoals") or []
    starts: list[int] = []
    texts: list[str] = []
    previous = 0
    for segment in segments:
        if previous >= length:
            raise SystemExit(
                f"{dataset_dir} episode {episode_index}: segment start "
                f"frame {previous} beyond {length} frames — judgments "
                "disagree with the episodes metadata",
            )
        starts.append(previous)
        texts.append(str(segment.get("subgoal") or ""))
        until = int(segment["until_frame"])
        if until <= previous:
            raise SystemExit(
                f"{dataset_dir} episode {episode_index}: non-increasing "
                f"until_frame {until} after start {previous}",
            )
        previous = until
    spans: list[tuple[int, int, str]] = []
    for k, (start, text) in enumerate(zip(starts, texts, strict=True)):
        end = starts[k + 1] if k + 1 < len(starts) else length
        if text.strip():
            spans.append((start, end, text))
    return tuple(spans)


def derangement(indices: list[int], rng: random.Random) -> dict[int, int]:
    """A seeded derangement via Sattolo's algorithm (one uniform cycle
    over the indices — bijective, no fixed points, deterministic in the
    RNG state). Requires ≥ 2 indices."""
    if len(indices) < 2:
        raise SystemExit(
            f"derangement over {len(indices)} episode(s) is impossible — "
            "the caller must skip single-labeled-episode datasets",
        )
    order = sorted(indices)
    cycle = list(order)
    for i in range(len(cycle) - 1, 0, -1):
        j = rng.randrange(i)
        cycle[i], cycle[j] = cycle[j], cycle[i]
    return dict(zip(order, cycle, strict=True))


def build_swap_map(
    dataset_dirs: dict[str, Path],
    *,
    seed: int,
    identity: bool = False,
) -> SubgoalSwapMap:
    """The full pinned mapping over the eval's selected datasets.
    ``identity`` keeps every episode its own donor (ALL other plumbing
    live) — the launcher's byte-reproduction pre-launch check."""
    episodes: dict[str, dict[int, LabeledEpisode]] = {}
    donors: dict[str, dict[int, int]] = {}
    skipped: list[str] = []
    for repo_id in sorted(dataset_dirs):
        labeled = labeled_episodes(dataset_dirs[repo_id])
        if not labeled:
            continue
        episodes[repo_id] = labeled
        if identity:
            donors[repo_id] = {index: index for index in labeled}
        elif len(labeled) == 1:
            skipped.append(repo_id)
        else:
            rng = random.Random(f"{_SEED_TAG}:{seed}:{repo_id}")
            donors[repo_id] = derangement(sorted(labeled), rng)
    if not episodes:
        raise SystemExit(
            "subgoal swap: no labeled episodes in any selected dataset — "
            "the oracle arm this swaps could never have conditioned",
        )
    return SubgoalSwapMap(
        seed=seed,
        identity=identity,
        episodes=episodes,
        donors=donors,
        skipped=tuple(skipped),
    )


@dataclass(frozen=True, slots=True)
class SwapRecord:
    """Per-frame swap provenance (the --dump-subgoal-swaps row): what
    the collator's slot actually received, addressable by the
    dataset-local identity triple — the mechanical oracle-(iv) check
    recomputes ``rendered_subgoal`` from the sidecars and compares."""

    repo_id: str
    episode_index: int
    frame_index: int
    timestamp: float
    true_subgoal: str
    donor_episode_index: int | None  # None = unswappable (skipped repo)
    rendered_subgoal: str  # "" = empty slot (renders nothing)
