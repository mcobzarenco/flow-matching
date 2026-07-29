"""Batch episode judging across LeRobot v3.0 collections (run: ``python -m bijou.judge.sweep``).

Drives the Anthropic judge (bijou.judge.claude) over many datasets and
episodes in parallel. Two storage layers:

- The --output JSONL is this run's *journal* (write-ahead log): one line per
  episode as results stream in, ok and failed alike, crash-safe.
- Each dataset's ``meta/judgments.json`` is the *durable store*
  (bijou.judge.store): successful verdicts folded in from the journal
  (auto-merge at the end of every run, or --merge-only to fold a crashed
  run's journal).

Idempotency is keyed on (episode_index, model, prompt_hash): re-running the
same configuration skips everything already judged (on any machine that has
the sidecars); switching model re-judges deliberately, and editing the
prompt changes its hash — stale verdicts invalidate themselves, nothing is
bumped by hand. Failures stay journal-local — they cost nothing to retry
(evidence gathering fails before any API spend) and a fresh machine should
retry transient ones. Episodes shorter than --min-frames are recomputed at
plan time (pure function of episode length), skipped loudly, never stored.

Usage:
    # plan only: what would run, rough token/cost estimate
    uv run python -m bijou.judge.sweep \
        --roots ~/datasets/mcobzarenco/community_dataset_v1_v3 \
        --output verdicts.jsonl --dry-run

    # pilot: 2 episodes per dataset, 4 concurrent workers
    uv run python -m bijou.judge.sweep \
        --roots ~/datasets/mcobzarenco/community_dataset_v1_v3 \
        --output verdicts.jsonl --episodes-per-dataset 2 --workers 4

    # fold an interrupted run's journal into the dataset sidecars
    uv run python -m bijou.judge.sweep --roots ... --output verdicts.jsonl --merge-only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import repo_id_of
from .claude import (
    DEFAULT_MAX_IMAGE_DIM,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_NUM_FRAMES,
)
from .schema import PROMPT_HASH
from .store import JudgmentRecord, load_sidecar, write_sidecar
from .worker import JudgeTask, judge_one

# Sweep-specific CLI defaults (the judge knobs are imported from
# bijou.judge.claude so the two CLIs can never drift apart).
DEFAULT_MIN_FRAMES = 50  # = one action chunk; shorter episodes are filtered
DEFAULT_WORKERS = 4

# Rough $/Mtok (input, output) as of 2026-07; used only for --dry-run and the
# end-of-run summary, clearly labeled as estimates. Unknown models get none.
MODEL_PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# Measured 2026-07 on the opus 4.8 tokenizer via the free count_tokens
# endpoint (640x480 frames thumbnailed to 512): input ~= 1464 + 305/image.
# Older tokenizers run ~10% lighter on the text part; rough by design.
EST_TOKENS_PER_IMAGE = 305
EST_TEXT_TOKENS = 1464
EST_OUTPUT_TOKENS = 500


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    """Planning outcome for one dataset."""

    root: Path
    repo_id: str
    cameras: int
    to_judge: list[int]
    skipped: list[tuple[int, int]]  # (episode, length) below --min-frames


def discover_datasets(roots: list[Path]) -> list[Path]:
    """Dataset dirs under collection roots (or roots that are datasets)."""
    found: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if (root / "meta" / "info.json").exists():
            found.append(root)
            continue
        nested = sorted(p.parent.parent for p in root.glob("*/*/meta/info.json"))
        if not nested:
            raise SystemExit(f"no LeRobot datasets under {root}")
        found.extend(nested)
    return found


def plan_dataset(
    dataset_dir: Path,
    min_frames: int,
    episodes_per_dataset: int | None,
) -> DatasetPlan:
    """Choose episodes to judge from metadata only (no video access).

    Episodes below ``min_frames`` are skipped with a reason. When
    ``episodes_per_dataset`` is set, eligible episodes are subsampled evenly
    across the episode index range (deterministic, covers session drift
    within a recording day better than the first N).
    """
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    features: dict[str, Any] = info.get("features") or {}
    cameras = sum(1 for feature in features.values() if feature.get("dtype") == "video")
    parquets = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not parquets:
        raise ValueError(f"{repo_id_of(dataset_dir)}: no meta/episodes/**/*.parquet")
    episodes = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "length"]) for p in parquets],
    ).sort_values("episode_index")

    eligible: list[int] = []
    skipped: list[tuple[int, int]] = []
    for episode, length in zip(
        episodes["episode_index"],
        episodes["length"],
        strict=True,
    ):
        if int(length) < min_frames:
            skipped.append((int(episode), int(length)))
        else:
            eligible.append(int(episode))

    if episodes_per_dataset is not None and len(eligible) > episodes_per_dataset:
        picks = np.unique(
            np.linspace(0, len(eligible) - 1, episodes_per_dataset).round().astype(int),
        )
        eligible = [eligible[i] for i in picks]

    return DatasetPlan(
        root=dataset_dir,
        repo_id=repo_id_of(dataset_dir),
        cameras=cameras,
        to_judge=eligible,
        skipped=skipped,
    )


def merge_journal(journal: Path, dirs_by_repo: dict[str, Path]) -> None:
    """Fold the journal's ok-records into dataset sidecars (idempotent).

    Last journal line wins per (episode, model, prompt_hash); identical
    records count as unchanged, so re-merging the same journal is a no-op.
    Datasets outside the discovered roots are reported loudly and kept in
    the journal — nothing is dropped.
    """
    if not journal.exists():
        print(f"merge: no journal at {journal}, nothing to fold")
        return
    by_repo: dict[str, dict[tuple[int, str, str], JudgmentRecord]] = {}
    with journal.open() as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("status") != "ok":
                continue
            new = JudgmentRecord.from_journal(raw)
            by_repo.setdefault(raw["dataset"], {})[new.key()] = new
    added = replaced = unchanged = 0
    missing: list[str] = []
    for repo, new_records in sorted(by_repo.items()):
        dataset_dir = dirs_by_repo.get(repo)
        if dataset_dir is None:
            missing.append(repo)
            continue
        existing = {r.key(): r for r in load_sidecar(dataset_dir)}
        changed = False
        for key, record in new_records.items():
            old = existing.get(key)
            if old == record:
                unchanged += 1
                continue
            added += old is None
            replaced += old is not None
            existing[key] = record
            changed = True
        if changed:
            write_sidecar(dataset_dir, list(existing.values()))
    print(
        f"merge: {added} added, {replaced} replaced, {unchanged} unchanged "
        f"across {len(by_repo)} dataset(s) -> meta/judgments.json",
    )
    if missing:
        print(
            f"merge: {len(missing)} journal dataset(s) not under --roots, NOT folded "
            f"(e.g. {missing[:3]}); re-run --merge-only with the right roots",
            file=sys.stderr,
        )


def load_journal_done(
    output: Path,
    *,
    retry_failed: bool,
) -> tuple[set[tuple[str, int, str, str]], set[tuple[str, int]]]:
    """Journal-side skip sets: ok keys (dataset, episode, model, prompt
    hash) — covering results not yet folded into sidecars — and failed
    (dataset, episode) pairs (empty when retrying; failure skip is
    model-agnostic because failures are overwhelmingly evidence-side, e.g.
    corrupt video, and would fail identically under any judge)."""
    ok: set[tuple[str, int, str, str]] = set()
    failed: set[tuple[str, int]] = set()
    if not output.exists():
        return ok, failed
    with output.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                ok.add(
                    (
                        record["dataset"],
                        int(record["episode"]),
                        str(record["model"]),
                        str(record["prompt_hash"]),
                    ),
                )
            elif record.get("status") == "failed" and not retry_failed:
                failed.add((record["dataset"], int(record["episode"])))
    return ok, failed


def estimate_cost(episodes: int, images: int, model: str) -> str:
    input_tokens = images * EST_TOKENS_PER_IMAGE + episodes * EST_TEXT_TOKENS
    output_tokens = episodes * EST_OUTPUT_TOKENS
    tokens = f"~{input_tokens:,} in / ~{output_tokens:,} out tokens"
    for prefix, (in_price, out_price) in MODEL_PRICES.items():
        if model.startswith(prefix):
            dollars = (input_tokens * in_price + output_tokens * out_price) / 1e6
            return f"{tokens}, ~${dollars:,.2f} ({model}, rough)"
    return f"{tokens} (no price table for {model})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge many LeRobot episodes with the Anthropic API, resumably.",
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        required=True,
        help="Collection roots (<root>/<user>/<dataset>) and/or dataset dirs.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Subset as '<user>/<dataset>' repo ids (default: all discovered).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Journal JSONL for this run (appended, crash-safe); successful "
        "verdicts are folded into each dataset's meta/judgments.json at the "
        "end of the run.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="Skip (and count) episodes shorter than this many frames; the default "
        "matches the 50-step action chunk — shorter episodes are mechanically "
        "filtered, no judge needed (default: %(default)s).",
    )
    parser.add_argument(
        "--episodes-per-dataset",
        type=int,
        default=None,
        help="Judge at most N episodes per dataset, evenly spaced over the episode "
        "index range (default: all eligible).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Hard cap on API calls this run, a safety valve for pilots "
        "(default: no cap).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Sampled timesteps per episode, each shown for every camera "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=DEFAULT_MAX_IMAGE_DIM,
        help="Frames are downscaled so the longer side is at most this many pixels "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Anthropic model id (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum response tokens per verdict (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent judge processes; each decodes frames and holds one API "
        "call in flight (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt episodes whose journal record has status=failed.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Fold the journal's verdicts into the dataset sidecars and exit "
        "(no judging; use after an interrupted run). Covers all datasets "
        "under --roots regardless of --datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and estimate only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dirs = discover_datasets(args.roots)
    dirs_by_repo = {repo_id_of(d): d for d in dataset_dirs}
    if args.merge_only:
        merge_journal(args.output, dirs_by_repo)
        return
    if args.datasets is not None:
        wanted = set(args.datasets)
        unknown = wanted - set(dirs_by_repo)
        if unknown:
            raise SystemExit(f"unknown datasets: {sorted(unknown)}")
        dataset_dirs = [dirs_by_repo[name] for name in sorted(wanted)]

    plans: list[DatasetPlan] = []
    plan_failures: list[tuple[str, str]] = []
    for dataset_dir in dataset_dirs:
        try:
            plans.append(
                plan_dataset(dataset_dir, args.min_frames, args.episodes_per_dataset),
            )
        except Exception as error:  # noqa: BLE001 - record and continue planning
            plan_failures.append((repo_id_of(dataset_dir), str(error)))
            print(f"PLAN FAILED {repo_id_of(dataset_dir)}: {error}", file=sys.stderr)

    journal_ok, journal_failed = load_journal_done(
        args.output,
        retry_failed=args.retry_failed,
    )

    cameras_by_repo = {plan.repo_id: plan.cameras for plan in plans}
    tasks: list[JudgeTask] = []
    already = 0
    for plan in plans:
        sidecar_keys = {record.key() for record in load_sidecar(plan.root)}
        for episode in plan.to_judge:
            if (
                (episode, args.model, PROMPT_HASH) in sidecar_keys
                or (plan.repo_id, episode, args.model, PROMPT_HASH) in journal_ok
                or (plan.repo_id, episode) in journal_failed
            ):
                already += 1
                continue
            tasks.append(
                JudgeTask(
                    root=str(plan.root),
                    repo_id=plan.repo_id,
                    episode=episode,
                    num_timesteps=args.num_frames,
                    max_image_dim=args.max_image_dim,
                    model=args.model,
                    max_tokens=args.max_tokens,
                ),
            )

    if args.max_episodes is not None and len(tasks) > args.max_episodes:
        tasks = tasks[: args.max_episodes]
    total_images = sum(
        args.num_frames * cameras_by_repo[task.repo_id] for task in tasks
    )

    planned = sum(len(p.to_judge) for p in plans)
    skipped_total = sum(len(p.skipped) for p in plans)
    print(
        f"plan: {len(plans)} datasets | {planned} episodes eligible | "
        f"{skipped_total} below {args.min_frames} frames | "
        f"{already} already judged for ({args.model}, {PROMPT_HASH}) | "
        f"{len(tasks)} to judge now | {len(plan_failures)} datasets failed to plan",
    )
    print(f"cost: {estimate_cost(len(tasks), total_images, args.model)}")
    if args.dry_run:
        for repo_id, error in plan_failures:
            print(f"  plan failure: {repo_id}: {error}")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (required unless --dry-run)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not tasks:
        print("nothing to judge")
        # Still fold: the journal may hold results from an interrupted run.
        merge_journal(args.output, dirs_by_repo)
        return

    outcomes = {"ok": 0, "failed": 0}
    tokens_in = tokens_out = 0
    # spawn (not fork): workers decode video; forking a torch-loaded
    # parent into AV1 decoders is asking for latent corruption.
    context = mp.get_context("spawn")
    started = time.perf_counter()
    with (
        args.output.open("a") as log,
        ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool,
    ):
        futures = {pool.submit(judge_one, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            log.write(json.dumps(record) + "\n")
            log.flush()
            outcomes[record["status"]] += 1
            usage = record.get("usage")
            if usage:
                tokens_in += usage["input_tokens"]
                tokens_out += usage["output_tokens"]
            if record["status"] == "failed":
                print(
                    f"FAILED {record['dataset']} ep {record['episode']}: {record['error']}",
                    file=sys.stderr,
                )
            if i % 25 == 0 or i == len(tasks):
                rate = i / (time.perf_counter() - started)
                print(
                    f"[{i}/{len(tasks)}] ok={outcomes['ok']} failed={outcomes['failed']} "
                    f"| {tokens_in:,} in / {tokens_out:,} out tokens | {rate:.2f} eps/s",
                    flush=True,
                )

    merge_journal(args.output, dirs_by_repo)
    for prefix, (in_price, out_price) in MODEL_PRICES.items():
        if args.model.startswith(prefix):
            spent = (tokens_in * in_price + tokens_out * out_price) / 1e6
            print(
                f"spent: ~${spent:,.2f} ({tokens_in:,} in / {tokens_out:,} out tokens, rough)",
            )
            break
    print(f"done: {outcomes} -> {args.output} + sidecars")


if __name__ == "__main__":
    main()
