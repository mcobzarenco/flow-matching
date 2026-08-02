"""Render stored judge verdicts for human eyes (run: ``python -m bijou.judge.show``).

Reads sweep journals — which carry the episode context (task, length,
cameras) that the sidecar deliberately omits — and pretty-prints
verdicts in the same format as the live ``bijou.judge.claude`` report.
No API calls, no video decoding: this is the spot-check tool for
sampling what a judge said during and after a sweep.

Usage:
    # three random verdicts from a pilot journal
    uv run python -m bijou.judge.show \
        --journal /durable/judge/pilot_opus5.jsonl --sample 3

    # a specific episode; or every failure with its reason
    ... --dataset user/dataset --episode 4
    ... --status failed
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .claude import format_judgment
from .schema import EpisodeJudgment


def load_journal(
    journals: list[Path],
    *,
    status: str,
    dataset: str | None,
    episode: int | None,
) -> list[dict[str, Any]]:
    """Filtered journal records, last occurrence per (dataset, episode,
    model) — retries supersede earlier attempts, like the sidecar merge."""
    latest: dict[tuple[str, int, str], dict[str, Any]] = {}
    for journal in journals:
        with journal.open() as lines:
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["dataset"], int(record["episode"]), record["model"])
                latest[key] = record
    return [
        record
        for (repo_id, episode_index, _model), record in sorted(latest.items())
        if record.get("status") == status
        and (dataset is None or repo_id == dataset)
        and (episode is None or episode_index == episode)
    ]


def print_record(record: dict[str, Any]) -> None:
    print(
        f"=== {record['dataset']} — episode {record['episode']} "
        f"[{record['model']} @ {record['time']}] ===",
    )
    if record["status"] == "failed":
        print(f"FAILED   : {record['error']}\n")
        return
    print(f'task     : "{record["task"]}"')
    print(
        f"length   : {record['num_frames']} frames "
        f"({record['duration_s']:.1f}s @ {record['fps']:.0f} fps) | "
        f"evidence: {record['num_timesteps']} timesteps x "
        f"{len(record['cameras'])} cameras ({', '.join(record['cameras'])})",
    )
    usage = record.get("usage")
    if usage:
        print(f"tokens   : {usage['input_tokens']} in / {usage['output_tokens']} out")
    print(format_judgment(EpisodeJudgment.from_dict(record["judgment"])))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print stored judge verdicts from sweep journals.",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        nargs="+",
        required=True,
        help="Sweep journal JSONL file(s); later files/lines supersede "
        "earlier ones per (dataset, episode, model).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only this '<user>/<dataset>' repo id (default: all).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Only this episode index (default: all).",
    )
    parser.add_argument(
        "--status",
        choices=("ok", "failed"),
        default="ok",
        help="Which records to show; 'failed' prints errors instead of "
        "verdicts (default: %(default)s).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Show N randomly sampled records instead of all matches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed (default: %(default)s).",
    )
    args = parser.parse_args()

    records = load_journal(
        args.journal,
        status=args.status,
        dataset=args.dataset,
        episode=args.episode,
    )
    if not records:
        raise SystemExit("no matching records")
    total = len(records)
    if args.sample is not None and args.sample < total:
        records = random.Random(args.seed).sample(records, args.sample)
        records.sort(key=lambda record: (record["dataset"], record["episode"]))
    for record in records:
        print_record(record)
    print(f"({len(records)} shown / {total} matching)")


if __name__ == "__main__":
    main()
