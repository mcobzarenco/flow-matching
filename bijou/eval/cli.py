"""Open-loop evaluation CLI (invoked via ``python -m bijou.eval``).

Samples K frames (without replacement, seeded) from a dataset selection and
scores every requested policy on the SAME frames: the trivial state-copy
baseline always, a bijou checkpoint when ``--checkpoint`` is given, and a
SmolVLA policy when ``--smolvla`` is given. Reports pad-masked chunk metrics
in raw action units plus paired per-frame comparisons against the baseline.

Ground truth is the recorded action chunk at each frame — this is offline /
open-loop evaluation (no robot, no simulator). For held-out scoring, point
``--data`` at datasets the checkpoint was not trained on.

Usage::

    uv run python -m bijou.eval \
        --data ~/datasets/mcobzarenco/so101_pick_place_clean \
        --checkpoint outputs/train/<run>/step_040000 \
        --smolvla lerobot/smolvla_base \
        --num-samples 256 --device cuda --output-json eval.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..aux_text import AuxDecodeMode
from ..data import EpisodeSplit, select_datasets
from ..model import SamplingMethod
from .metrics import (
    FrameScore,
    PairedComparison,
    PolicySummary,
    compare_paired,
    format_table,
    score_frame,
    summarize,
)
from .policies import (
    BijouPolicy,
    ChunkPolicy,
    NormalizedStateCopyPolicy,
    StateCopyPolicy,
)
from .report import THEMES, ReportSample, render_report
from .smolvla import SmolVLAEvalPolicy


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Schema of the --output-json payload: what was evaluated (data,
    split, sampling, policies) plus the per-policy summaries and paired
    comparisons. The JSON contract other tooling parses — extend, don't
    reshape."""

    data: list[str]
    episodes: str
    holdout_episodes: float
    split_seed: int
    fps: list[float] | None
    num_samples: int
    seed: int
    checkpoint: str | None
    smolvla: str | None
    summaries: list[PolicySummary]
    paired: list[PairedComparison]
    motor_names: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "episodes": self.episodes,
            "holdout_episodes": self.holdout_episodes,
            "split_seed": self.split_seed,
            "fps": self.fps,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "smolvla": self.smolvla,
            "summaries": [s.to_dict() for s in self.summaries],
            "paired": [c.to_dict() for c in self.paired],
            "motor_names": self.motor_names,
        }


def identity_collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep raw items as a list (module-level: spawn workers pickle this)."""
    return items


def eval_worker_init(_worker_id: int) -> None:
    """Single-threaded + file_system tensor sharing, IN the worker: tensors
    are serialized worker-side, so the sharing strategy must be set there —
    the main-process call alone still produced fd-based storages."""
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="dataset dirs and/or collection roots (same semantics as "
        "bijou.train --train-data)",
    )
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument(
        "--episodes",
        choices=[s.value for s in EpisodeSplit],
        default=EpisodeSplit.ALL.value,
        help="which side of the per-dataset episode holdout to evaluate; "
        "'holdout' scores exactly the episodes a training run with the "
        "same --holdout-episodes and --split-seed never saw",
    )
    parser.add_argument(
        "--holdout-episodes",
        type=float,
        default=0.0,
        help="holdout fraction — must match the training run's "
        "--holdout-episodes for the split to line up",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="split seed — must match the training run's --split-seed",
    )
    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=None,
        help="keep only datasets recorded at one of these frame rates; "
        "default keeps every fps. Must match the training run's --fps for "
        "scores to be comparable (any filter changes the concatenated "
        "frame indexing and thus the sampled eval frames)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="bijou checkpoint directory; omit to evaluate the baseline only",
    )
    parser.add_argument(
        "--smolvla",
        default=None,
        help="optional SmolVLA policy id/path (e.g. lerobot/smolvla_base)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="frames sampled without replacement across the selection",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="bijou prefix-encode batch",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=10,
        help="flow ODE solver steps",
    )
    parser.add_argument(
        "--sample-method",
        choices=[m.value for m in SamplingMethod],
        default=SamplingMethod.HEUN.value,
    )
    parser.add_argument(
        "--aux-mode",
        choices=[m.value for m in AuxDecodeMode],
        default=AuxDecodeMode.ACT.value,
        help="ar_backbone decode mode: 'act' scores the deployment fast "
        "path (comparable to aux-less arms); 'free' generates the aux "
        "text first (aux-trained checkpoints only). Other decoder kinds "
        "ignore it",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="write summaries as JSON",
    )
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="write every policy's predicted chunks (plus truth/valid/"
        "frame identity) as a compressed .npz — for offline analyses that "
        "must share this eval's exact prediction conventions (e.g. the "
        "DCT-truncation sweep) instead of re-implementing them",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write a self-contained HTML report (tables + per-datapoint "
        "prediction charts) to this path",
    )
    parser.add_argument(
        "--report-samples",
        type=int,
        default=12,
        help="datapoints charted in the report (evenly spread over the "
        "sampled frames; bounds report size on large evals)",
    )
    parser.add_argument(
        "--report-theme",
        choices=sorted(THEMES),
        default="dark",
        help="report color scheme",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    # Eval workers ship RAW items (10+ tensor storages each, vs training's
    # one collated batch); the default fd-based sharing pins one shm fd per
    # storage and blows the 1024-fd ulimit -> 'received 0 items of ancdata'.
    # The file_system strategy shares via named files instead.
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(args.device)

    episode_split = EpisodeSplit(args.episodes)
    if episode_split is not EpisodeSplit.ALL and args.holdout_episodes <= 0:
        raise SystemExit(f"--episodes {args.episodes} requires --holdout-episodes > 0")
    selection = select_datasets(
        tuple(args.data),
        tuple(args.exclude),
        args.chunk_size,
        episode_split=episode_split,
        holdout_fraction=args.holdout_episodes,
        split_seed=args.split_seed,
        allowed_fps=tuple(args.fps) if args.fps else None,
    )
    dataset = selection.concat()
    print(
        f"eval data: {len(selection.datasets)} datasets, "
        f"{selection.total_episodes} episodes, {len(dataset)} frames, "
        f"action/state dim {selection.action_dim}/{selection.state_dim}",
        flush=True,
    )
    if episode_split is not EpisodeSplit.ALL:
        print(
            f"episode split: {episode_split.value} "
            f"(fraction {args.holdout_episodes}, split seed {args.split_seed}; "
            f"{selection.held_out_episodes} held-out episodes across "
            f"{selection.held_out_datasets} datasets)",
            flush=True,
        )
    if selection.dropped:
        print(f"dropped {len(selection.dropped)} incompatible datasets:", flush=True)
        for reason in selection.dropped:
            print(f"  - {reason}", flush=True)

    num_samples = min(args.num_samples, len(dataset))
    indices = sorted(random.Random(args.seed).sample(range(len(dataset)), num_samples))
    print(f"sampling {num_samples} frames (seed {args.seed})", flush=True)

    policies: list[ChunkPolicy] = [
        StateCopyPolicy(args.chunk_size),
        NormalizedStateCopyPolicy(args.chunk_size),
    ]
    if args.checkpoint is not None:
        policy = BijouPolicy(
            args.checkpoint,
            device=device,
            seed=args.seed,
            sample_steps=args.sample_steps,
            method=SamplingMethod(args.sample_method),
            aux_mode=AuxDecodeMode(args.aux_mode),
        )
        if policy.info.chunk_size != args.chunk_size:
            raise SystemExit(
                f"checkpoint chunk size {policy.info.chunk_size} != "
                f"--chunk-size {args.chunk_size}",
            )
        policies.append(policy)
    if args.smolvla is not None:
        policies.append(
            SmolVLAEvalPolicy(
                args.smolvla,
                device=device,
                seed=args.seed,
                lerobot_stats=selection.lerobot_stats,
            ),
        )

    # Fetch each sampled frame once (parallel decode; spawn context — the
    # main process may hold CUDA state and torchcodec is fork-unsafe) and
    # stream it through every policy so comparisons stay paired.
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
        worker_init_fn=eval_worker_init if args.num_workers > 0 else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    # Datapoints whose predictions are retained for the report: evenly
    # spread over the sampled frames so charts span datasets, bounded so
    # large evals stay renderable.
    report_stride = max(len(indices) // max(args.report_samples, 1), 1)
    report_indices = set(indices[::report_stride][: args.report_samples])
    report_samples: dict[int, ReportSample] = {}

    scores: dict[str, list[FrameScore]] = {p.name: [] for p in policies}
    dump_predictions: dict[str, list[torch.Tensor]] = {p.name: [] for p in policies}
    dump_truth: list[torch.Tensor] = []
    dump_valid: list[torch.Tensor] = []
    dump_repo: list[str] = []
    dump_index: list[int] = []
    done = 0
    for batch_number, items in enumerate(loader):
        batch_indices = indices[done : done + len(items)]
        for policy in policies:
            start = time.perf_counter()
            predictions = policy.predict(items, batch_indices)
            elapsed = (time.perf_counter() - start) / len(items)
            for item, index, predicted in zip(
                items,
                batch_indices,
                predictions,
                strict=True,
            ):
                truth = item["action"]
                predicted = predicted[: truth.shape[0]].float()
                scores[policy.name].append(
                    score_frame(
                        index=index,
                        repo_id=str(item["repo_id"]),
                        predicted=predicted,
                        truth=truth.float(),
                        valid=~item["action_is_pad"],
                        inference_seconds=elapsed,
                    ),
                )
                if args.dump_predictions is not None:
                    dump_predictions[policy.name].append(predicted)
                    if policy is policies[0]:
                        dump_truth.append(truth.float())
                        dump_valid.append(~item["action_is_pad"])
                        dump_repo.append(str(item["repo_id"]))
                        dump_index.append(index)
                if args.report is not None and index in report_indices:
                    sample = report_samples.get(index) or ReportSample(
                        index=index,
                        episode=int(item["episode_index"]),
                        frame_in_episode=int(item["frame_index"]),
                        repo_id=str(item["repo_id"]),
                        task=str(item["task"]),
                        state=item["observation.state"].float(),
                        cameras={
                            k.removeprefix("observation.images."): v
                            for k, v in item.items()
                            if k.startswith("observation.images.")
                        },
                        truth=truth.float(),
                        valid=~item["action_is_pad"],
                        predictions={},
                    )
                    sample.predictions[policy.name] = predicted
                    report_samples[index] = sample
        done += len(items)
        if batch_number % 5 == 0:
            print(f"  scored {done}/{num_samples} frames", flush=True)

    if args.dump_predictions is not None:
        payload: dict[str, np.ndarray] = {
            "truth": torch.stack(dump_truth).numpy(),
            "valid": torch.stack(dump_valid).numpy(),
            "index": np.array(dump_index),
            "repo_id": np.array(dump_repo),
        }
        for name, chunks in dump_predictions.items():
            payload[f"pred:{name}"] = torch.stack(chunks).numpy()
        np.savez_compressed(args.dump_predictions, allow_pickle=False, **payload)
        print(f"dumped predictions to {args.dump_predictions}", flush=True)

    summaries = [summarize(name, frame_scores) for name, frame_scores in scores.items()]
    motor_names = selection.action_names or [
        f"motor_{i}" for i in range(selection.action_dim)
    ]

    print("\n== chunk metrics (raw action units, pad-masked) ==", flush=True)
    print(
        format_table(
            [
                [
                    s.name,
                    f"{s.chunk_mae:.3f}",
                    f"{s.mae_p50:.3f}",
                    f"{s.mae_p90:.3f}",
                    f"{s.first_mae:.3f}",
                    f"{s.chunk_mse:.1f}",
                    f"{s.seconds_per_frame * 1000:.0f}",
                ]
                for s in summaries
            ],
            ["policy", "chunk_mae", "p50", "p90", "first_mae", "chunk_mse", "ms/frame"],
        ),
        flush=True,
    )

    print("\n== per-motor chunk MAE ==", flush=True)
    print(
        format_table(
            [[s.name, *(f"{v:.2f}" for v in s.per_motor_mae)] for s in summaries],
            ["policy", *motor_names],
        ),
        flush=True,
    )

    baseline = policies[0].name
    comparisons = [
        compare_paired(s.name, scores[s.name], baseline, scores[baseline])
        for s in summaries[1:]
    ]
    if comparisons:
        print(f"\n== paired vs {baseline} (negative delta = better) ==", flush=True)
        print(
            format_table(
                [
                    [
                        c.policy,
                        f"{c.mean_delta:+.3f}",
                        f"{c.delta_p50:+.3f}",
                        f"{100 * c.win_rate:.0f}%",
                    ]
                    for c in comparisons
                ],
                ["policy", "mean_delta_mae", "delta_p50", "win_rate"],
            ),
            flush=True,
        )

    if args.output_json is not None:
        report = EvalReport(
            data=[str(p) for p in args.data],
            episodes=args.episodes,
            holdout_episodes=args.holdout_episodes,
            split_seed=args.split_seed,
            fps=list(args.fps) if args.fps else None,
            num_samples=num_samples,
            seed=args.seed,
            checkpoint=str(args.checkpoint) if args.checkpoint else None,
            smolvla=args.smolvla,
            summaries=summaries,
            paired=comparisons,
            motor_names=motor_names,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report.to_json_dict(), indent=2))
        print(f"\nwrote {args.output_json}", flush=True)

    if args.report is not None:
        # Local wall-clock time, offset-aware (naive now() is ambiguous
        # when comparing reports generated on different machines).
        generated = datetime.datetime.now(datetime.UTC).astimezone()
        config_lines = [
            f"generated: {generated.isoformat(timespec='seconds')}",
            f"data: {', '.join(str(p) for p in args.data)}",
            (
                f"selection: {len(selection.datasets)} datasets, "
                f"{selection.total_episodes} episodes, {len(dataset)} frames "
                f"({len(selection.dropped)} dropped)"
            ),
            f"samples: {num_samples} frames, seed {args.seed}",
            f"episodes: {args.episodes}"
            + (
                f" (holdout fraction {args.holdout_episodes}, "
                f"split seed {args.split_seed})"
                if episode_split is not EpisodeSplit.ALL
                else ""
            ),
            f"checkpoint: {args.checkpoint or '-'}",
            f"smolvla: {args.smolvla or '-'}",
            f"sampler: {args.sample_method}-{args.sample_steps}",
            f"fps filter: {args.fps or 'all'}",
        ]
        render_report(
            args.report,
            config_lines,
            summaries,
            comparisons,
            motor_names,
            [report_samples[i] for i in sorted(report_samples)],
            total_scored=num_samples,
            theme=THEMES[args.report_theme],
        )
        print(f"wrote {args.report}", flush=True)
    return 0
