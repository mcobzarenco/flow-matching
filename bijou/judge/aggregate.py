"""Aggregate judge sidecars into calibration evidence (run: ``python -m bijou.judge.aggregate``).

Consumes ``meta/judgments.json`` sidecars across collection roots and
produces the numbers the calibration decisions (curation TODO step 4)
are made from:

- **Per-model summaries** (pinned to one prompt hash): verdict /
  completion / instruction-quality distributions, score histograms,
  holding/progress/event rates, and the impact of candidate filter
  rules in both episodes flagged and datasets killed.
- **Camera-kind majority votes** per (dataset, camera) with agreement
  stats; ``--write-camera-maps`` persists them as
  ``meta/camera_kinds.json`` per dataset — the map train-time camera
  tagging consumes. Ties resolve to ``unknown`` (the honest fallback,
  and already the train-time dropout target).
- **Paired cross-model agreement** on the episodes both models judged
  (the sweep's even-spread selection is deterministic, so a two-model
  pilot yields paired verdicts by construction): verdict confusion,
  discard-set overlap, score MAE/correlation, per-frame holding and
  progress agreement, camera-kind agreement. This is the cascade
  evidence: how much does the cheap judge disagree with the expensive
  one, and in which direction.
- **Failure census** from sweep journals (``--journals``): what failed
  and why, since sidecars only keep successes.
- **Hand-labeling worksheet** (``--review-sample``): a seeded,
  verdict-stratified episode sample as JSONL with empty ``human_*``
  fields — judge-vs-human agreement fills in later.

Judge-vs-human scoring against a completed worksheet is deliberately
not here yet: build it when the labels exist, against their actual
format.

Pure metadata: no video decode, no API calls, runs anywhere the
sidecars are.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data import repo_id_of
from .schema import PROMPT_HASH, CameraKind, EpisodeJudgment, TaskCompletion, Verdict
from .store import discover_datasets, load_sidecar

CAMERA_KINDS_RELPATH = Path("meta") / "camera_kinds.json"

# Candidate judge-gated filter rules (curation TODO step 6). Names are
# report keys; predicates see one parsed judgment. Evaluated per model so
# the paired report shows how the choice of judge moves each rule.
FILTER_RULES: tuple[tuple[str, Callable[[EpisodeJudgment], bool]], ...] = (
    ("verdict_discard", lambda j: j.verdict is Verdict.DISCARD),
    ("verdict_not_keep", lambda j: j.verdict is not Verdict.KEEP),
    (
        "no_completion_and_score<=3",
        lambda j: (
            j.task_completion_visible is TaskCompletion.NO and j.overall_score <= 3
        ),
    ),
    (
        "discard_or_failed",
        lambda j: (
            j.verdict is Verdict.DISCARD
            or (j.task_completion_visible is TaskCompletion.NO and j.overall_score <= 3)
        ),
    ),
    ("score<=2", lambda j: j.overall_score <= 2),
)


@dataclass(frozen=True, slots=True)
class EpisodeVerdict:
    """One parsed sidecar record, flattened for aggregation."""

    repo_id: str
    episode: int
    model: str
    judgment: EpisodeJudgment


@dataclass(frozen=True, slots=True)
class CameraVote:
    """Majority-vote outcome for one (dataset, camera) under one model."""

    kind: CameraKind
    votes: dict[str, int]  # kind value -> count
    unanimous: bool
    tie: bool


@dataclass(frozen=True, slots=True)
class RuleImpact:
    """One filter rule's reach: episodes flagged; datasets where at least
    half / all of the judged episodes are flagged."""

    episodes_flagged: int
    datasets_half_flagged: int
    datasets_all_flagged: int


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """Corpus-level aggregates for one judge model at the pinned hash."""

    model: str
    datasets: int
    episodes: int
    verdicts: dict[str, int]
    task_completion: dict[str, int]
    instruction_quality: dict[str, int]
    overall_hist: dict[int, int]
    overall_mean: float
    subscore_means: dict[str, float]
    sampled_frames: int
    holding_rate: float
    episodes_with_events: int
    events_total: int
    camera_kind_counts: dict[str, int]
    cameras_voted: int
    cameras_unanimous: int
    cameras_tied: int
    rules: dict[str, RuleImpact]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Agreement between two models on the episodes both judged."""

    model_a: str
    model_b: str
    episodes: int
    verdict_agreement: float
    verdict_confusion: dict[str, dict[str, int]]  # a-verdict -> b-verdict -> n
    discard_jaccard: float
    a_discards_b_keeps: int
    b_discards_a_keeps: int
    overall_mae: float
    overall_pearson: float | None
    task_completion_agreement: float
    instruction_quality_agreement: float
    cameras_compared: int
    camera_kind_agreement: float
    holding_frames: int
    holding_agreement: float
    progress_frames: int
    progress_mae: float
    progress_pearson: float | None


def load_verdicts(
    dataset_dirs: Iterable[Path],
    prompt_hash: str,
) -> tuple[list[EpisodeVerdict], int, int]:
    """Parse every sidecar record at ``prompt_hash``.

    Returns (verdicts, datasets_with_records, records_skipped_by_hash).
    Records at other hashes are skipped by design (they obey their own
    prompt's schema); a payload at OUR hash failing to parse is a bug,
    not data — let it raise.
    """
    verdicts: list[EpisodeVerdict] = []
    datasets_with_records = 0
    skipped_by_hash = 0
    for dataset_dir in dataset_dirs:
        records = load_sidecar(dataset_dir)
        if not records:
            continue
        repo_id = repo_id_of(dataset_dir)
        matched = False
        for record in records:
            if record.prompt_hash != prompt_hash:
                skipped_by_hash += 1
                continue
            matched = True
            verdicts.append(
                EpisodeVerdict(
                    repo_id=repo_id,
                    episode=record.episode_index,
                    model=record.model,
                    judgment=record.parsed_judgment(),
                ),
            )
        if matched:
            datasets_with_records += 1
    return verdicts, datasets_with_records, skipped_by_hash


def majority_camera_vote(kinds: list[CameraKind]) -> CameraVote:
    """Strict-majority vote; ties resolve to ``unknown`` and are marked.

    A tie between judged episodes is disagreement worth surfacing, and
    ``unknown`` is exactly the label consumers already treat as "no
    signal" (dropout target).
    """
    counts = Counter(kind.value for kind in kinds)
    ranked = counts.most_common()
    top_kind, top_votes = ranked[0]
    tie = len(ranked) > 1 and ranked[1][1] == top_votes
    return CameraVote(
        kind=CameraKind.UNKNOWN if tie else CameraKind(top_kind),
        votes=dict(counts),
        unanimous=len(counts) == 1,
        tie=tie,
    )


def camera_votes_for_model(
    verdicts: list[EpisodeVerdict],
) -> dict[str, dict[str, CameraVote]]:
    """repo_id -> camera short name -> majority vote (one model's verdicts)."""
    by_camera: dict[str, dict[str, list[CameraKind]]] = {}
    for verdict in verdicts:
        cameras = by_camera.setdefault(verdict.repo_id, {})
        for camera, kind in verdict.judgment.camera_kinds.items():
            cameras.setdefault(camera, []).append(kind)
    return {
        repo_id: {
            camera: majority_camera_vote(kinds) for camera, kinds in cameras.items()
        }
        for repo_id, cameras in by_camera.items()
    }


def rule_impact(
    verdicts: list[EpisodeVerdict],
    predicate: Callable[[EpisodeJudgment], bool],
) -> RuleImpact:
    flagged_by_repo: dict[str, list[bool]] = {}
    for verdict in verdicts:
        flagged_by_repo.setdefault(verdict.repo_id, []).append(
            predicate(verdict.judgment),
        )
    flags = [flag for flags in flagged_by_repo.values() for flag in flags]
    return RuleImpact(
        episodes_flagged=sum(flags),
        datasets_half_flagged=sum(
            1 for flags in flagged_by_repo.values() if sum(flags) * 2 >= len(flags)
        ),
        datasets_all_flagged=sum(1 for flags in flagged_by_repo.values() if all(flags)),
    )


def summarize_model(model: str, verdicts: list[EpisodeVerdict]) -> ModelSummary:
    """Corpus aggregates for one model (callers pass that model's verdicts)."""
    judgments = [verdict.judgment for verdict in verdicts]
    overall = [judgment.overall_score for judgment in judgments]
    annotations = [
        annotation
        for judgment in judgments
        for annotation in judgment.frame_annotations
    ]
    votes = camera_votes_for_model(verdicts)
    flat_votes = [vote for cameras in votes.values() for vote in cameras.values()]
    return ModelSummary(
        model=model,
        datasets=len({verdict.repo_id for verdict in verdicts}),
        episodes=len(verdicts),
        verdicts=dict(Counter(j.verdict.value for j in judgments)),
        task_completion=dict(
            Counter(j.task_completion_visible.value for j in judgments),
        ),
        instruction_quality=dict(
            Counter(j.instruction_quality.value for j in judgments),
        ),
        overall_hist=dict(sorted(Counter(overall).items())),
        overall_mean=float(np.mean(overall)) if overall else float("nan"),
        subscore_means={
            name: float(
                np.mean([getattr(j.scores, name) for j in judgments]),
            )
            for name in ("visual_quality", "smoothness", "efficiency", "camera_framing")
        },
        sampled_frames=len(annotations),
        holding_rate=float(np.mean([a.holding for a in annotations]))
        if annotations
        else float("nan"),
        episodes_with_events=sum(
            1 for j in judgments if any(a.events for a in j.frame_annotations)
        ),
        events_total=sum(len(a.events) for a in annotations),
        camera_kind_counts=dict(
            Counter(vote.kind.value for vote in flat_votes),
        ),
        cameras_voted=len(flat_votes),
        cameras_unanimous=sum(1 for vote in flat_votes if vote.unanimous),
        cameras_tied=sum(1 for vote in flat_votes if vote.tie),
        rules={
            name: rule_impact(verdicts, predicate) for name, predicate in FILTER_RULES
        },
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """None when undefined (fewer than two points or zero variance)."""
    if len(a) < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compare_models(
    model_a: str,
    model_b: str,
    verdicts: list[EpisodeVerdict],
) -> PairedComparison:
    """Paired agreement on (repo, episode) keys both models judged."""
    a_by_key = {
        (v.repo_id, v.episode): v.judgment for v in verdicts if v.model == model_a
    }
    b_by_key = {
        (v.repo_id, v.episode): v.judgment for v in verdicts if v.model == model_b
    }
    common = sorted(set(a_by_key) & set(b_by_key))
    pairs = [(a_by_key[key], b_by_key[key]) for key in common]

    confusion: dict[str, dict[str, int]] = {
        a.value: {b.value: 0 for b in Verdict} for a in Verdict
    }
    for a, b in pairs:
        confusion[a.verdict.value][b.verdict.value] += 1
    a_discards = {key for key in common if a_by_key[key].verdict is Verdict.DISCARD}
    b_discards = {key for key in common if b_by_key[key].verdict is Verdict.DISCARD}
    union = a_discards | b_discards

    overall_a = np.array([a.overall_score for a, _ in pairs], dtype=np.float64)
    overall_b = np.array([b.overall_score for _, b in pairs], dtype=np.float64)

    # Frame annotations pair by frame number; the intersection guards
    # against differing num_timesteps between runs.
    holding_pairs: list[tuple[bool, bool]] = []
    progress_pairs: list[tuple[float, float]] = []
    for a, b in pairs:
        a_frames = {ann.frame: ann for ann in a.frame_annotations}
        b_frames = {ann.frame: ann for ann in b.frame_annotations}
        for frame in sorted(set(a_frames) & set(b_frames)):
            holding_pairs.append((a_frames[frame].holding, b_frames[frame].holding))
            progress_pairs.append(
                (a_frames[frame].progress, b_frames[frame].progress),
            )
    progress_a = np.array([p for p, _ in progress_pairs], dtype=np.float64)
    progress_b = np.array([p for _, p in progress_pairs], dtype=np.float64)

    votes_a = camera_votes_for_model([v for v in verdicts if v.model == model_a])
    votes_b = camera_votes_for_model([v for v in verdicts if v.model == model_b])
    camera_matches = camera_total = 0
    for repo_id in set(votes_a) & set(votes_b):
        for camera in set(votes_a[repo_id]) & set(votes_b[repo_id]):
            camera_total += 1
            if votes_a[repo_id][camera].kind is votes_b[repo_id][camera].kind:
                camera_matches += 1

    def fraction(matches: int, total: int) -> float:
        return matches / total if total else float("nan")

    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        episodes=len(pairs),
        verdict_agreement=fraction(
            sum(1 for a, b in pairs if a.verdict is b.verdict),
            len(pairs),
        ),
        verdict_confusion=confusion,
        discard_jaccard=fraction(len(a_discards & b_discards), len(union)),
        a_discards_b_keeps=len(a_discards - b_discards),
        b_discards_a_keeps=len(b_discards - a_discards),
        overall_mae=float(np.abs(overall_a - overall_b).mean())
        if len(pairs)
        else float("nan"),
        overall_pearson=_pearson(overall_a, overall_b),
        task_completion_agreement=fraction(
            sum(
                1
                for a, b in pairs
                if a.task_completion_visible is b.task_completion_visible
            ),
            len(pairs),
        ),
        instruction_quality_agreement=fraction(
            sum(1 for a, b in pairs if a.instruction_quality is b.instruction_quality),
            len(pairs),
        ),
        cameras_compared=camera_total,
        camera_kind_agreement=fraction(camera_matches, camera_total),
        holding_frames=len(holding_pairs),
        holding_agreement=fraction(
            sum(1 for a, b in holding_pairs if a == b),
            len(holding_pairs),
        ),
        progress_frames=len(progress_pairs),
        progress_mae=float(np.abs(progress_a - progress_b).mean())
        if progress_pairs
        else float("nan"),
        progress_pearson=_pearson(progress_a, progress_b),
    )


def dataset_table(verdicts: list[EpisodeVerdict]) -> dict[str, dict[str, Any]]:
    """Per-dataset actionables (all models pooled; per-model detail lives
    in the summaries): judged count, mean score, discard fraction —
    the raw material of step 6's kill list."""
    by_repo: dict[str, list[EpisodeVerdict]] = {}
    for verdict in verdicts:
        by_repo.setdefault(verdict.repo_id, []).append(verdict)
    table: dict[str, dict[str, Any]] = {}
    for repo_id, repo_verdicts in sorted(by_repo.items()):
        judgments = [verdict.judgment for verdict in repo_verdicts]
        discards = sum(1 for j in judgments if j.verdict is Verdict.DISCARD)
        table[repo_id] = {
            "judged": len(judgments),
            "models": sorted({verdict.model for verdict in repo_verdicts}),
            "episodes": sorted({verdict.episode for verdict in repo_verdicts}),
            "mean_overall": round(
                float(np.mean([j.overall_score for j in judgments])),
                2,
            ),
            "discards": discards,
            "discard_fraction": round(discards / len(judgments), 3),
        }
    return table


def journal_failure_census(journals: list[Path]) -> dict[str, int]:
    """Failed journal lines bucketed by error prefix (sidecars only keep
    successes, so the journals are the only failure record)."""
    census: Counter[str] = Counter()
    for journal in journals:
        with journal.open() as lines:
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") == "failed":
                    census[str(record.get("error", "?"))[:80]] += 1
    return dict(census.most_common())


def write_camera_maps(
    dirs_by_repo: dict[str, Path],
    votes: dict[str, dict[str, CameraVote]],
    *,
    model: str,
    prompt_hash: str,
) -> int:
    """Persist per-dataset ``meta/camera_kinds.json`` (atomic, idempotent)."""
    written = 0
    for repo_id, cameras in sorted(votes.items()):
        payload = {
            "model": model,
            "prompt_hash": prompt_hash,
            "written_at": time.strftime("%F %T", time.gmtime()),
            "cameras": {
                camera: {
                    "kind": vote.kind.value,
                    "votes": vote.votes,
                    "unanimous": vote.unanimous,
                    "tie": vote.tie,
                }
                for camera, vote in sorted(cameras.items())
            },
        }
        path = dirs_by_repo[repo_id] / CAMERA_KINDS_RELPATH
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(path)
        written += 1
    return written


def review_worksheet(
    verdicts: list[EpisodeVerdict],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Verdict-stratified seeded sample as a hand-labeling worksheet.

    Even split across verdict strata (discard/review are the decision
    boundary — uniform sampling would drown them in keeps), remainder
    refilled from the largest strata. ``human_*`` fields start empty.
    """
    rng = random.Random(seed)
    strata: dict[Verdict, list[EpisodeVerdict]] = {v: [] for v in Verdict}
    for verdict in verdicts:
        strata[verdict.judgment.verdict].append(verdict)
    for stratum in strata.values():
        stratum.sort(key=lambda v: (v.repo_id, v.episode, v.model))
        rng.shuffle(stratum)

    per_stratum = sample_size // len(strata)
    chosen: list[EpisodeVerdict] = []
    for stratum in strata.values():
        chosen.extend(stratum[:per_stratum])
    leftovers = [
        verdict for stratum in strata.values() for verdict in stratum[per_stratum:]
    ]
    rng.shuffle(leftovers)
    chosen.extend(leftovers[: sample_size - len(chosen)])

    return [
        {
            "repo_id": verdict.repo_id,
            "episode": verdict.episode,
            "model": verdict.model,
            "judge_verdict": verdict.judgment.verdict.value,
            "judge_overall": verdict.judgment.overall_score,
            "judge_completion": verdict.judgment.task_completion_visible.value,
            "judge_observed_task": verdict.judgment.observed_task,
            "judge_summary": verdict.judgment.summary,
            "human_verdict": "",
            "human_completion": "",
            "human_notes": "",
        }
        for verdict in sorted(chosen, key=lambda v: (v.repo_id, v.episode, v.model))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate judge sidecars into calibration evidence.",
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        required=True,
        help="Collection roots (<root>/<user>/<dataset>) and/or dataset dirs.",
    )
    parser.add_argument(
        "--prompt-hash",
        type=str,
        default=PROMPT_HASH,
        help="Aggregate records judged under this prompt hash only "
        "(default: the running code's, %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full aggregate as JSON here (default: print-only).",
    )
    parser.add_argument(
        "--journals",
        type=Path,
        nargs="*",
        default=[],
        help="Sweep journal JSONLs for the failure census (default: none).",
    )
    parser.add_argument(
        "--write-camera-maps",
        action="store_true",
        help="Write per-dataset meta/camera_kinds.json majority votes "
        "(requires --camera-map-model).",
    )
    parser.add_argument(
        "--camera-map-model",
        type=str,
        default=None,
        help="Model whose votes to persist with --write-camera-maps "
        "(explicit, no default: the map ships to consumers).",
    )
    parser.add_argument(
        "--review-sample",
        type=int,
        default=None,
        help="Emit a verdict-stratified hand-labeling worksheet of N episodes "
        "(default: none).",
    )
    parser.add_argument(
        "--review-model",
        type=str,
        default=None,
        help="Model whose verdicts stratify the worksheet (required with "
        "--review-sample when several models are present).",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=None,
        help="Worksheet JSONL path (required with --review-sample).",
    )
    parser.add_argument(
        "--review-seed",
        type=int,
        default=0,
        help="Worksheet sampling seed (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.write_camera_maps and not args.camera_map_model:
        parser.error("--write-camera-maps requires --camera-map-model")
    if args.review_sample is not None and args.review_output is None:
        parser.error("--review-sample requires --review-output")
    return args


def main() -> None:
    args = parse_args()
    dataset_dirs = discover_datasets(args.roots)
    dirs_by_repo = {repo_id_of(d): d for d in dataset_dirs}
    verdicts, datasets_with_records, skipped_by_hash = load_verdicts(
        dataset_dirs,
        args.prompt_hash,
    )
    if not verdicts:
        raise SystemExit(
            f"no records at prompt_hash={args.prompt_hash} under "
            f"{[str(r) for r in args.roots]} "
            f"({skipped_by_hash} records skipped by hash)",
        )

    models = sorted({verdict.model for verdict in verdicts})
    summaries = {
        model: summarize_model(
            model,
            [verdict for verdict in verdicts if verdict.model == model],
        )
        for model in models
    }
    paired = [
        compare_models(model_a, model_b, verdicts)
        for i, model_a in enumerate(models)
        for model_b in models[i + 1 :]
    ]
    failures = journal_failure_census(args.journals)

    print(
        f"aggregate: {len(dataset_dirs)} datasets discovered | "
        f"{datasets_with_records} with sidecar records | "
        f"{len(verdicts)} verdicts at {args.prompt_hash} | "
        f"{skipped_by_hash} records at other hashes | models: {', '.join(models)}",
    )
    for summary in summaries.values():
        print(f"\n== {summary.model} ==")
        print(
            f"  {summary.episodes} episodes / {summary.datasets} datasets | "
            f"overall mean {summary.overall_mean:.2f}",
        )
        print(f"  verdicts: {summary.verdicts}")
        print(f"  completion: {summary.task_completion}")
        print(f"  instruction: {summary.instruction_quality}")
        print(
            f"  holding rate {summary.holding_rate:.1%} over "
            f"{summary.sampled_frames} sampled frames | "
            f"{summary.events_total} events in "
            f"{summary.episodes_with_events} episodes",
        )
        print(
            f"  cameras: {summary.camera_kind_counts} | "
            f"{summary.cameras_unanimous}/{summary.cameras_voted} unanimous, "
            f"{summary.cameras_tied} ties",
        )
        for name, impact in summary.rules.items():
            print(
                f"  rule {name:<28} flags {impact.episodes_flagged:>5} eps | "
                f"kills {impact.datasets_half_flagged:>4} half / "
                f"{impact.datasets_all_flagged:>4} all datasets",
            )
    for comparison in paired:
        print(f"\n== {comparison.model_a} vs {comparison.model_b} ==")
        print(
            f"  {comparison.episodes} paired episodes | verdict agreement "
            f"{comparison.verdict_agreement:.1%} | discard jaccard "
            f"{comparison.discard_jaccard:.2f} "
            f"(a-only {comparison.a_discards_b_keeps}, "
            f"b-only {comparison.b_discards_a_keeps})",
        )
        print(f"  confusion (a rows -> b cols): {comparison.verdict_confusion}")
        pearson = (
            f"{comparison.overall_pearson:.2f}"
            if comparison.overall_pearson is not None
            else "n/a"
        )
        print(
            f"  overall MAE {comparison.overall_mae:.2f} (r={pearson}) | "
            f"completion agr {comparison.task_completion_agreement:.1%} | "
            f"instruction agr {comparison.instruction_quality_agreement:.1%}",
        )
        progress_pearson = (
            f"{comparison.progress_pearson:.2f}"
            if comparison.progress_pearson is not None
            else "n/a"
        )
        print(
            f"  holding agr {comparison.holding_agreement:.1%} "
            f"({comparison.holding_frames} frames) | progress MAE "
            f"{comparison.progress_mae:.2f} (r={progress_pearson}) | camera agr "
            f"{comparison.camera_kind_agreement:.1%} "
            f"({comparison.cameras_compared} cameras)",
        )
    if failures:
        print("\n== journal failures ==")
        for error, count in failures.items():
            print(f"  {count:>4}  {error}")

    if args.write_camera_maps:
        model_verdicts = [
            verdict for verdict in verdicts if verdict.model == args.camera_map_model
        ]
        if not model_verdicts:
            raise SystemExit(
                f"no verdicts from {args.camera_map_model} to build camera maps",
            )
        written = write_camera_maps(
            dirs_by_repo,
            camera_votes_for_model(model_verdicts),
            model=args.camera_map_model,
            prompt_hash=args.prompt_hash,
        )
        print(f"\ncamera maps written: {written} datasets ({CAMERA_KINDS_RELPATH})")

    if args.review_sample is not None:
        review_model = args.review_model
        if review_model is None:
            if len(models) > 1:
                raise SystemExit(
                    f"--review-model is required with several models: {models}",
                )
            review_model = models[0]
        worksheet = review_worksheet(
            [verdict for verdict in verdicts if verdict.model == review_model],
            sample_size=args.review_sample,
            seed=args.review_seed,
        )
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        with args.review_output.open("w") as sink:
            for row in worksheet:
                sink.write(json.dumps(row) + "\n")
        print(f"review worksheet: {len(worksheet)} episodes -> {args.review_output}")

    if args.output:
        payload = {
            "prompt_hash": args.prompt_hash,
            "generated_at": time.strftime("%F %T", time.gmtime()),
            "roots": [str(root) for root in args.roots],
            "datasets_discovered": len(dataset_dirs),
            "datasets_with_records": datasets_with_records,
            "records_skipped_by_hash": skipped_by_hash,
            "models": {model: asdict(summary) for model, summary in summaries.items()},
            "paired": [asdict(comparison) for comparison in paired],
            "datasets": dataset_table(verdicts),
            "journal_failures": failures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1))
        print(f"aggregate JSON -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
