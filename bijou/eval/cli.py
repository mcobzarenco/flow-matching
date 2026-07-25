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
        --data ~/datasets/marius/so101_pick_place_clean \
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
from pathlib import Path
from typing import Any

import torch

from ..data import select_datasets
from ..model import SamplingMethod
from .metrics import (
    FrameScore,
    compare_paired,
    format_table,
    score_frame,
    summarize,
)
from .policies import BijouPolicy, ChunkPolicy, StateCopyPolicy
from .report import ReportSample, render_report
from .smolvla import SmolVLAEvalPolicy


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
        "--batch-size", type=int, default=32, help="bijou prefix-encode batch"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sample-steps", type=int, default=10, help="flow ODE solver steps"
    )
    parser.add_argument(
        "--sample-method",
        choices=[m.value for m in SamplingMethod],
        default=SamplingMethod.HEUN.value,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-json", type=Path, default=None, help="write summaries as JSON"
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

    selection = select_datasets(tuple(args.data), tuple(args.exclude), args.chunk_size)
    dataset = selection.concat()
    print(
        f"eval data: {len(selection.datasets)} datasets, "
        f"{selection.total_episodes} episodes, {len(dataset)} frames, "
        f"action/state dim {selection.action_dim}/{selection.state_dim}",
        flush=True,
    )
    if selection.dropped:
        print(f"dropped {len(selection.dropped)} incompatible datasets:", flush=True)
        for reason in selection.dropped:
            print(f"  - {reason}", flush=True)

    num_samples = min(args.num_samples, len(dataset))
    indices = sorted(random.Random(args.seed).sample(range(len(dataset)), num_samples))
    print(f"sampling {num_samples} frames (seed {args.seed})", flush=True)

    policies: list[ChunkPolicy] = [StateCopyPolicy(args.chunk_size)]
    if args.checkpoint is not None:
        policy = BijouPolicy(
            args.checkpoint,
            device=device,
            seed=args.seed,
            sample_steps=args.sample_steps,
            method=SamplingMethod(args.sample_method),
        )
        if policy.info.chunk_size != args.chunk_size:
            raise SystemExit(
                f"checkpoint chunk size {policy.info.chunk_size} != "
                f"--chunk-size {args.chunk_size}"
            )
        policies.append(policy)
    if args.smolvla is not None:
        policies.append(
            SmolVLAEvalPolicy(
                args.smolvla,
                device=device,
                seed=args.seed,
                lerobot_stats=selection.lerobot_stats,
            )
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
    done = 0
    for batch_number, items in enumerate(loader):
        batch_indices = indices[done : done + len(items)]
        for policy in policies:
            start = time.perf_counter()
            predictions = policy.predict(items, batch_indices)
            elapsed = (time.perf_counter() - start) / len(items)
            for item, index, predicted in zip(
                items, batch_indices, predictions, strict=True
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
                    )
                )
                if args.report is not None and index in report_indices:
                    sample = report_samples.get(index) or ReportSample(
                        index=index,
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
        payload: dict[str, Any] = {
            "data": [str(p) for p in args.data],
            "num_samples": num_samples,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "smolvla": args.smolvla,
            "summaries": [s.to_dict() for s in summaries],
            "paired": [c.to_dict() for c in comparisons],
            "motor_names": motor_names,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output_json}", flush=True)

    if args.report is not None:
        config_lines = [
            f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"data: {', '.join(str(p) for p in args.data)}",
            f"selection: {len(selection.datasets)} datasets, "
            f"{selection.total_episodes} episodes, {len(dataset)} frames "
            f"({len(selection.dropped)} dropped)",
            f"samples: {num_samples} frames, seed {args.seed}",
            f"checkpoint: {args.checkpoint or '-'}",
            f"smolvla: {args.smolvla or '-'}",
            f"sampler: {args.sample_method}-{args.sample_steps}",
        ]
        render_report(
            args.report,
            config_lines,
            summaries,
            comparisons,
            motor_names,
            [report_samples[i] for i in sorted(report_samples)],
            total_scored=num_samples,
        )
        print(f"wrote {args.report}", flush=True)
    return 0
