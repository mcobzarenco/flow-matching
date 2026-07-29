"""Local episode judge powered by Gemma 4 (run: ``python -m bijou.judge.gemma``).

Same evidence and verdict schema as the API judge (bijou.judge.claude), so
the two are directly comparable in calibration; generation is greedy by
default, which makes this the reproducible-verdict path the API cannot
offer. Weights load through plain transformers in bf16 onto a single
device — no quantization backends; the 12B model wants ~24 GB, sized for
the training boxes, not laptops.

Usage:
    uv run python -m bijou.judge.gemma \
        --root ~/datasets/mcobzarenco/community_dataset_v3_v3/<user>/<ds> \
        --episode 3 --image-token-budget 280

    # tokenize + report context length without loading model weights
    uv run python -m bijou.judge.gemma --root ... --episode 3 --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
)

from .claude import print_report
from .evidence import EpisodeSummary, load_episode_summary
from .schema import SYSTEM_PROMPT, EpisodeJudgment

GEMMA_MODEL_ID = "google/gemma-4-12B-it"
IMAGE_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)
DEFAULT_NUM_FRAMES = 10  # sampled timesteps per episode
DEFAULT_MAX_IMAGE_DIM = 512  # px, longer side after downscaling
DEFAULT_MAX_NEW_TOKENS = 1200  # generation budget (verdict JSON + slack)


def build_messages(
    summary: EpisodeSummary,
    extra_context: str | None,
) -> list[dict[str, Any]]:
    """Chat messages for the Gemma processor (image content before text)."""
    user_content: list[dict[str, Any]] = []
    for label, camera, image in summary.frames:
        user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": f"{label} — camera '{camera}'"})

    briefing = (
        f"Episode {summary.episode} of dataset {summary.repo_id}: "
        f"{summary.num_frames} frames, {summary.duration_s:.1f}s "
        f"at {summary.fps:.0f} fps, cameras: {', '.join(summary.camera_names)}.\n"
        f'Operator instruction: "{summary.task}"\n'
    )
    if extra_context:
        briefing += f"Context from the dataset owner: {extra_context}\n"
    camera_list = ", ".join(f'"{name}"' for name in summary.camera_names)
    briefing += (
        f"\nFull-trajectory statistics:\n{summary.stats_text}\n\n"
        "Assess this demonstration and reply with the JSON object only. "
        f"`camera_kinds` must have exactly these keys: {camera_list}."
    )
    user_content.append({"type": "text", "text": briefing})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def strip_thought_channel(text: str) -> str:
    """Drop Gemma 4's thought channel and any special-token markup."""
    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
    return re.sub(r"<\|[^>]*\|>", "", text).strip()


def apply_template(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    thinking: bool,
    image_token_budget: int | None,
) -> Any:
    template_kwargs: dict[str, Any] = {"enable_thinking": thinking}
    if image_token_budget is not None:
        # transformers >=5.14 wants per-call processor kwargs nested.
        template_kwargs["processor_kwargs"] = {"max_soft_tokens": image_token_budget}
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )


def run_judge(
    messages: list[dict[str, Any]],
    model_id: str,
    *,
    device: str,
    thinking: bool,
    temperature: float | None,
    max_new_tokens: int,
    image_token_budget: int | None,
) -> tuple[str, dict[str, int], float]:
    """Generate a verdict locally. Returns (text, token counts, seconds)."""
    processor = AutoProcessor.from_pretrained(model_id)

    # `Any`: the HF auto-class stubs don't expose `generate` on their
    # common base, and the two branches return different class families.
    model: Any
    try:
        model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    except ValueError:
        # Non any-to-any checkpoints (useful with --model for smoke tests).
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        )
    model.to(device)
    model.eval()

    inputs = apply_template(
        processor,
        messages,
        thinking=thinking,
        image_token_budget=image_token_budget,
    ).to(device)
    input_len = int(inputs["input_ids"].shape[-1])

    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature is None:
        generate_kwargs["do_sample"] = False
    else:
        # Sampling settings recommended by the Gemma 4 model card.
        generate_kwargs |= {
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
        }

    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - started

    new_tokens = output_ids[0][input_len:]
    decoded = processor.decode(new_tokens, skip_special_tokens=False)
    # The regex strip is the one and only extraction path. The processor's
    # parse_response looks like the official alternative but is not usable
    # here: probed 2026-07-30 under transformers 5.14.1 + gemma-4-12B-it,
    # it raises unconditionally when called without `prefix=` (the tokenized
    # prompt) — an earlier port wrapped it in suppress(Exception), which
    # only hid that the fallback was doing all the work.
    text = strip_thought_channel(decoded)

    usage = {"input_tokens": input_len, "output_tokens": int(new_tokens.shape[-1])}
    return text, usage, elapsed


def count_context_tokens(
    messages: list[dict[str, Any]],
    model_id: str,
    image_token_budget: int | None,
) -> int:
    """Tokenize the prompt (processor only, no model weights)."""
    processor = AutoProcessor.from_pretrained(model_id)
    inputs = apply_template(
        processor,
        messages,
        thinking=False,
        image_token_budget=image_token_budget,
    )
    return int(inputs["input_ids"].shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge a LeRobot v3.0 episode with a local Gemma 4 model.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset directory (v3.0).",
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
        help="Sampled timesteps, each shown for every camera (default: %(default)s).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=None,
        help="Camera names to include (default: all cameras).",
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
        default=GEMMA_MODEL_ID,
        help="Hugging Face model id (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for weights and generation (default: %(default)s).",
    )
    parser.add_argument(
        "--image-token-budget",
        type=int,
        choices=IMAGE_TOKEN_BUDGETS,
        default=None,
        help="Gemma 4 visual token budget per image (default: model's own default).",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Gemma 4's reasoning mode before the final answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Enable sampling at this temperature (default: greedy/deterministic).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens for the verdict (default: %(default)s).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Extra scene context passed to the judge (default: none).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of the text report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tokenize and report context length without loading model weights.",
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
    messages = build_messages(summary, args.context)

    if args.dry_run:
        sizes = {image.size for _, _, image in summary.frames}
        print(
            f"[dry run] {len(summary.frames)} frames at {sizes}, judge model {args.model}",
        )
        print(f'[dry run] task: "{summary.task}"')
        print(f"[dry run] stats block:\n{summary.stats_text}")
        tokens = count_context_tokens(messages, args.model, args.image_token_budget)
        print(f"[dry run] context length: {tokens} input tokens")
        return

    text, usage, seconds = run_judge(
        messages,
        args.model,
        device=args.device,
        thinking=args.thinking,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        image_token_budget=args.image_token_budget,
    )
    print(
        f"generated in {seconds:.1f}s on {args.device} ({args.model})",
        file=sys.stderr,
    )

    try:
        judgment: EpisodeJudgment | None = EpisodeJudgment.from_response_text(text)
        judgment.check_cameras(summary.camera_names)
    except ValueError as error:
        print(f"warning: {error}", file=sys.stderr)
        judgment = None
    print_report(summary, judgment, text, as_json=args.json, usage=usage)


if __name__ == "__main__":
    main()
