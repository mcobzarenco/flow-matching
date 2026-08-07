"""Stage-1 validity table for the self-subgoal probe (#6 rung (a)).

Decodes the model's OWN subgoals (pass 1 of the two-pass instrument,
``bijou.eval --subgoal-mode self``) for a fixed-seed stratified sample
of the panel's CORE frames and writes the pre-registered validity table
(markdown + JSON): frame identity triple, instruction, TRUE segment
label (or —), generated subgoal. NO action scoring and NO scalars are
computed or printed — the pre-registration gates every stage-2 scalar
behind this table's go/no-go (non-empty, non-truncated text on ≥ 90% of
rows; no single string on > 50%; qualitatively subgoal-shaped, judged
by eyes and commented in the results post). The go/no-go judgment
itself is human; this script only produces the evidence plus the two
mechanical counts.

Stratification (an implementation choice the pre-reg left open, fixed
here): round-robin across repos (sorted by id), within a repo
round-robin across its planned episodes, frames within an episode in
seeded-shuffle order — every repo and as many distinct episodes as the
budget allows are represented, deterministically in (plan, n, seed).

Usage (the k4l2 panel's selection args, dropout-free):

    uv run python fontaine/scripts/selfsubgoal_stage1.py \
        --data <corpus dirs> --checkpoint <ar-100k>/step_100000 \
        --sample-plan plans/holdout_curated_v0_k4l2.json \
        --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
        --output-md <table.md> --output-json <table.json>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter, deque
from pathlib import Path

import torch

from bijou.data import EpisodeSplit, select_datasets
from bijou.eval.cli import eval_worker_init, identity_collate
from bijou.eval.plan import (
    PlanFrame,
    SamplePlan,
    episode_tables,
    resolve_plan,
    validate_plan,
)
from bijou.eval.policies import BijouPolicy, SelfSubgoalPass1Policy


def stratify(identities: list[tuple[str, int]], n: int, seed: int) -> list[int]:
    """Positions of a stratified ``n``-row sample of ``identities``
    ((repo_id, episode_index) per row): round-robin across repos, then
    across each repo's episodes, frames within an episode drawn in
    seeded-shuffle order. Deterministic in (identities, n, seed);
    returns sorted positions, ``min(n, len(identities))`` of them."""
    rng = random.Random(seed)
    episodes: dict[tuple[str, int], list[int]] = {}
    for position, key in enumerate(identities):
        episodes.setdefault(key, []).append(position)
    by_repo: dict[str, deque[list[int]]] = {}
    for key in sorted(episodes):
        rows = episodes[key]
        rng.shuffle(rows)
        by_repo.setdefault(key[0], deque()).append(rows)
    ring = deque(by_repo[repo] for repo in sorted(by_repo))
    picked: list[int] = []
    while ring and len(picked) < n:
        repo = ring.popleft()
        episode = repo.popleft()
        picked.append(episode.pop(0))
        if episode:
            repo.append(episode)
        if repo:
            ring.append(repo)
    return sorted(picked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument(
        "--episodes",
        choices=[s.value for s in EpisodeSplit],
        default=EpisodeSplit.ALL.value,
    )
    parser.add_argument("--holdout-episodes", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--fps", type=float, nargs="+", default=None)
    parser.add_argument("--camera-counts", type=int, nargs="+", default=None)
    parser.add_argument("--aux-prompt-hash", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-plan", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")

    selection = select_datasets(
        tuple(args.data),
        tuple(args.exclude),
        args.chunk_size,
        episode_split=EpisodeSplit(args.episodes),
        holdout_fraction=args.holdout_episodes,
        split_seed=args.split_seed,
        allowed_fps=tuple(args.fps) if args.fps else None,
        allowed_camera_counts=(
            tuple(args.camera_counts) if args.camera_counts else None
        ),
        required_prompt_hash=args.aux_prompt_hash,
        load_episode_annotations=True,
    )
    dataset = selection.concat()
    plan = SamplePlan.load(args.sample_plan)
    validate_plan(
        plan,
        episodes=args.episodes,
        holdout_episodes=args.holdout_episodes,
        split_seed=args.split_seed,
        fps=list(args.fps) if args.fps else None,
        camera_counts=list(args.camera_counts) if args.camera_counts else None,
    )
    chosen = stratify(
        [(f.repo_id, f.episode_index) for f in plan.core],
        args.num_frames,
        args.seed,
    )
    subset: list[PlanFrame] = [plan.core[position] for position in chosen]
    # Resolve exactly the chosen frames through the shared plan
    # machinery (concat indices via the one column scan).
    tables = episode_tables(selection)
    indices, _core = resolve_plan(
        dataclasses.replace(plan, core=subset, labeled=[]),
        tables,
    )
    print(
        f"stage 1: {len(indices)} frames stratified from "
        f"{len(plan.core)} core (seed {args.seed}) across "
        f"{len({f.repo_id for f in subset})} repos / "
        f"{len({(f.repo_id, f.episode_index) for f in subset})} episodes",
        flush=True,
    )

    device = torch.device(args.device)
    base = BijouPolicy(
        args.checkpoint,
        device=device,
        seed=args.seed,
        subgoal_mode="self",
    )
    pass1 = SelfSubgoalPass1Policy(base)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
        worker_init_fn=eval_worker_init if args.num_workers > 0 else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    done = 0
    for items in loader:
        batch_indices = indices[done : done + len(items)]
        pass1.predict(items, batch_indices)  # generations retained; chunks unused
        done += len(items)
        print(f"  generated {done}/{len(indices)}", flush=True)

    records = [pass1.records[index] for index in indices]
    # The two mechanical go/no-go counts ((a) and (b)); criterion (c) —
    # subgoal-shaped — is eyes-only and stays with the results post.
    nonempty = sum(1 for r in records if (r.generated_subgoal or "").strip())
    top_text, top_count = ("", 0)
    if records:
        [(top_text, top_count)] = Counter(
            (r.generated_subgoal or "").strip() for r in records
        ).most_common(1)

    lines = [
        "# Self-subgoal stage-1 validity table (#6 rung (a))",
        "",
        (
            f"checkpoint: `{args.checkpoint}` · plan: `{args.sample_plan}` · "
            f"seed {args.seed} · {len(records)} rows"
        ),
        "",
        (
            f"mechanical counts: non-empty {nonempty}/{len(records)} "
            f"(gate (a): ≥ 90% non-empty AND non-truncated — truncation is "
            f"read from the rows, by eyes); most common string "
            f"{top_count}/{len(records)} (gate (b): ≤ 50%): {top_text!r}"
        ),
        "",
        "| # | frame | instruction | true subgoal | generated subgoal |",
        "|---|---|---|---|---|",
    ]

    def cell(text: str | None) -> str:
        return (text or "—").replace("|", "\\|").replace("\n", " ")

    for row, record in enumerate(records):
        frame = f"{record.repo_id} e{record.episode_index} f{record.frame_index}"
        lines.append(
            f"| {row} | {cell(frame)} | {cell(record.instruction)} | "
            f"{cell(record.true_subgoal)} | {cell(record.generated_subgoal)} |",
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "sample_plan": str(args.sample_plan),
                "seed": args.seed,
                "num_frames": len(records),
                "nonempty": nonempty,
                "most_common": {"text": top_text, "count": top_count},
                "rows": [dataclasses.asdict(r) for r in records],
            },
            indent=2,
        ),
    )
    print(f"wrote {args.output_md} and {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
