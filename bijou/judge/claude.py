"""Episode judge backed by the Anthropic API (run: ``python -m bijou.judge.claude``).

Builds the shared evidence + prompt, requests a single verdict, and prints
a report (or the raw response with --raw). Requires ANTHROPIC_API_KEY.

Verdicts are non-deterministic by nature: opus 4.7+ rejects sampling
controls outright, and the API reference never promised determinism even
at temperature=0 — provenance rides in (model, PROMPT_HASH) instead. The
local judge (bijou.judge.gemma, greedy decode) is the path to reproducible
verdicts if that ever becomes load-bearing.

Usage:
    uv run python -m bijou.judge.claude \
        --root ~/datasets/mcobzarenco/community_dataset_v3_v3/<user>/<ds> \
        --episode 3 [--json | --raw] [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import ImageBlockParam, TextBlockParam
from PIL import Image

from .evidence import EpisodeSummary, load_episode_summary
from .schema import SYSTEM_PROMPT, EpisodeJudgment

# CLI defaults. Named so bijou.judge.sweep (which drives this judge over
# whole collections) shares the exact same knobs instead of re-hardcoding
# them; help strings render the live values via argparse's %(default)s.
DEFAULT_MODEL = "claude-opus-4-8"  # $5/$25 per MTok (2026-07)
DEFAULT_NUM_FRAMES = 10  # sampled timesteps per episode
DEFAULT_MAX_IMAGE_DIM = 512  # px, longer side after downscaling
DEFAULT_JPEG_QUALITY = 90
DEFAULT_MAX_TOKENS = 1500  # response budget


def image_to_jpeg_b64(image: Image.Image, quality: int) -> str:
    """JPEG over PNG on the wire: Anthropic token cost depends on pixel
    dimensions only, and the dataset's AV1 compression is already baked in
    either way — one more lossy generation for ~10x less upload."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.standard_b64encode(buffer.getvalue()).decode()


def build_user_content(
    summary: EpisodeSummary,
    context: str | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[TextBlockParam | ImageBlockParam]:
    """Interleaved text/image blocks for the Anthropic messages API."""
    intro = (
        f"Dataset: {summary.repo_id}, episode {summary.episode}\n"
        f'Task instruction: "{summary.task}"\n'
        f"Length: {summary.num_frames} frames = {summary.duration_s:.1f}s "
        f"@ {summary.fps:.0f} fps\n"
        f"Cameras: {', '.join(summary.camera_names)}\n"
        f"Sampled timesteps: {len(summary.frames) // max(len(summary.camera_names), 1)} "
        f"(each shown for every camera, chronological order)"
    )
    content: list[TextBlockParam | ImageBlockParam] = [{"type": "text", "text": intro}]
    if context:
        content.append(
            {
                "type": "text",
                "text": f"Additional context from the dataset owner: {context}",
            },
        )
    for label, camera, image in summary.frames:
        content.append({"type": "text", "text": f"{label} — camera '{camera}'"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_to_jpeg_b64(image, jpeg_quality),
                },
            },
        )
    camera_list = ", ".join(f'"{name}"' for name in summary.camera_names)
    content.append(
        {
            "type": "text",
            "text": "Full-episode trajectory statistics:\n```\n"
            + summary.stats_text
            + "\n```\nNow give your quality assessment as the specified JSON object. "
            f"`camera_kinds` must have exactly these keys: {camera_list}.",
        },
    )
    return content


def request_verdict(
    client: Anthropic,
    model: str,
    max_tokens: int,
    content: list[TextBlockParam | ImageBlockParam],
) -> tuple[str, dict[str, int]]:
    """One judgment request; returns (raw response text, token usage).

    Deliberately no sampling controls: opus 4.7+ rejects ``temperature``
    with a 400 and the API never guaranteed determinism anyway.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return raw, usage


def print_report(
    summary: EpisodeSummary,
    judgment: EpisodeJudgment | None,
    raw: str,
    *,
    as_json: bool,
    usage: dict[str, int] | None = None,
) -> None:
    if as_json:
        payload: dict[str, object] = {
            "dataset": summary.repo_id,
            "episode": summary.episode,
            "task": summary.task,
            "num_frames": summary.num_frames,
            "duration_s": round(summary.duration_s, 2),
            "judge": judgment.to_dict()
            if judgment is not None
            else {"raw_response": raw},
        }
        if usage is not None:
            payload["usage"] = usage
        print(json.dumps(payload, indent=2))
        return

    print(f"=== {summary.repo_id} — episode {summary.episode} ===")
    print(f'task     : "{summary.task}"')
    print(
        f"length   : {summary.num_frames} frames "
        f"({summary.duration_s:.1f}s @ {summary.fps:.0f} fps)",
    )
    print(f"cameras  : {', '.join(summary.camera_names)}")
    if usage is not None:
        print(f"tokens   : {usage['input_tokens']} in / {usage['output_tokens']} out")
    print()
    if judgment is None:
        print("could not parse JSON verdict; raw response:")
        print(raw)
        return
    print(f"overall  : {judgment.overall_score}/10  ->  {judgment.verdict.value}")
    print(f"task done: {judgment.task_completion_visible.value}")
    scores = judgment.scores
    print(
        "scores   : "
        f"visual_quality={scores.visual_quality}  smoothness={scores.smoothness}  "
        f"efficiency={scores.efficiency}  camera_framing={scores.camera_framing}",
    )
    print(f"instr    : {judgment.instruction_quality.value}")
    print(f'observed : "{judgment.observed_task}"')
    start = 1
    for segment in judgment.subgoals:
        print(f'  frames {start:>4}-{segment.until_frame:<4}: "{segment.subgoal}"')
        start = segment.until_frame + 1
    print(
        "cameras  : "
        + "  ".join(
            f"{name}={kind.value}"
            for name, kind in sorted(judgment.camera_kinds.items())
        ),
    )
    for instruction in judgment.suggested_instructions:
        print(f'  suggest: "{instruction}"')
    print("issues   : " + ("none noted" if not judgment.issues else ""))
    for issue in judgment.issues:
        print(f"  - {issue}")
    if judgment.summary:
        print(f"summary  : {judgment.summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge the quality of a LeRobot v3.0 episode with Claude.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset directory containing meta/, data/, videos/ (v3.0 format).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Dataset repo id (default: the last two path components of --root).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to judge (default: %(default)s).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of timesteps to sample, each shown for every camera "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=None,
        help="Camera names to include, short ('wrist') or full "
        "('observation.images.wrist') (default: all cameras).",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=DEFAULT_MAX_IMAGE_DIM,
        help="Frames are downscaled so the longer side is at most this many pixels "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG encoding quality for uploaded frames (default: %(default)s).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Extra context for the judge, e.g. clarifying an ambiguous task "
        "instruction or describing the scene setup (default: none).",
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
        help="Maximum response tokens for the verdict (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of the text report.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the model's verbatim response text instead of any report "
        "(parse problems still warn on stderr).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and describe the payload without requesting a completion. "
        "When ANTHROPIC_API_KEY is set, the exact context length is reported "
        "via the free token-counting endpoint.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    repo_id = args.repo_id or "/".join(root.parts[-2:])

    summary = load_episode_summary(
        root=root,
        repo_id=repo_id,
        episode=args.episode,
        num_timesteps=args.num_frames,
        max_image_dim=args.max_image_dim,
        cameras=args.cameras,
    )
    content = build_user_content(summary, args.context, jpeg_quality=args.jpeg_quality)
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if args.dry_run:
        payload_kb = (
            sum(
                len(block["source"]["data"])  # type: ignore[typeddict-item]  # image blocks only
                for block in content
                if block["type"] == "image"
            )
            * 3
            / 4
            / 1024
        )
        image_count = sum(1 for block in content if block["type"] == "image")
        print(f"[dry run] would send {image_count} images (~{payload_kb:.0f} KB JPEG)")
        print(f'[dry run] task: "{summary.task}"')
        print(f"[dry run] stats block:\n{summary.stats_text}")
        if have_key:
            count = Anthropic().messages.count_tokens(
                model=args.model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            print(
                f"[dry run] context length: {count.input_tokens} input tokens "
                f"({args.model})",
            )
        else:
            print("[dry run] set ANTHROPIC_API_KEY to report the exact context length")
        return

    if not have_key:
        print(
            "error: ANTHROPIC_API_KEY is not set. Export it or use --dry-run.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    raw, usage = request_verdict(Anthropic(), args.model, args.max_tokens, content)
    try:
        judgment: EpisodeJudgment | None = EpisodeJudgment.from_response_text(raw)
        judgment.check_cameras(summary.camera_names)
        judgment.check_subgoals(summary.num_frames)
    except ValueError as error:
        print(f"warning: {error}", file=sys.stderr)
        judgment = None
    if args.raw:
        print(raw)
        return
    print_report(summary, judgment, raw, as_json=args.json, usage=usage)


if __name__ == "__main__":
    main()
