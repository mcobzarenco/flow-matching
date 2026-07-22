"""Verify bit-exact parity of the pure-torch Gemma4 vs HF transformers.

Runs three checks against ``Gemma4ForConditionalGeneration`` (eager attention):

1. full-sequence forward: logits of every prompt position,
2. cached greedy decode: per-step logits and chosen tokens,
3. (with ``--image``) the same, with an image in the prompt.

All comparisons are bitwise (``torch.equal``); max abs diff is reported too.

Usage::

    uv run python -m gemma4.verify_parity --max-new-tokens 16
    uv run python -m gemma4.verify_parity --image path/to.jpg --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .cache import KVCache
from .generation import generate
from .loading import load_generation_defaults, load_model, resolve_checkpoint_dir

DEFAULT_MODEL = "google/gemma-4-e2b-it"
DEFAULT_PROMPTS = (
    "The three primary colors are",
    "Instruct: pick up the red cube and place it in the bin.\nPlan:",
)


@dataclass(frozen=True, slots=True)
class Comparison:
    name: str
    bitwise_equal: bool
    max_abs_diff: float

    def report(self) -> str:
        status = "OK (bitwise)" if self.bitwise_equal else "MISMATCH"
        return f"  {self.name:<42} {status:<14} max|Δ|={self.max_abs_diff:.3e}"


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def lap(self, label: str) -> None:
        print(f"  [{time.monotonic() - self.t0:7.1f}s] {label}", flush=True)


def compare(name: str, ours: Tensor, theirs: Tensor) -> Comparison:
    if ours.shape != theirs.shape:
        raise AssertionError(
            f"{name}: shape {tuple(ours.shape)} != {tuple(theirs.shape)}"
        )
    diff = (ours.float() - theirs.float()).abs().max().item()
    return Comparison(name, bool(torch.equal(ours, theirs)), diff)


def check_all(comparisons: list[Comparison]) -> bool:
    ok = True
    for comparison in comparisons:
        print(comparison.report())
        ok &= comparison.bitwise_equal
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument(
        "--image", default=None, help="path to an image for the multimodal check"
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--long-context",
        type=int,
        default=0,
        metavar="N",
        help="also check an N-token synthetic prompt (exercises the sliding "
        "window and its cache once N + decoded tokens exceed 512)",
    )
    args = parser.parse_args()

    import transformers

    device = torch.device(args.device)
    checkpoint_dir = resolve_checkpoint_dir(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)

    watch = Stopwatch()
    print(
        f"loading reference (transformers {transformers.__version__}, eager) ...",
        flush=True,
    )
    # Typed as Any: the HF stubs' protocol checks add noise without value here.
    hf_model: Any = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint_dir, dtype=torch.bfloat16, attn_implementation="eager"
    )
    hf_model = hf_model.to(device).eval()
    watch.lap("reference model loaded")

    print("loading pure-torch implementation ...", flush=True)
    model = load_model(checkpoint_dir, device=device)
    watch.lap("pure-torch model loaded")

    prompts = args.prompt if args.prompt else list(DEFAULT_PROMPTS)
    all_ok = True

    for prompt in prompts:
        print(f"\n=== prompt: {prompt!r}", flush=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        comparisons: list[Comparison] = []

        # 1. Full forward, no cache: all prompt positions.
        with torch.no_grad():
            ours = model(input_ids)
            theirs = hf_model(input_ids=input_ids)
        comparisons.append(
            compare("prefill logits (all positions)", ours.logits, theirs.logits)
        )
        watch.lap("prefill compared")

        # 2. Cached stepwise decode, greedy.
        comparisons.extend(
            stepwise_decode_comparisons(model, hf_model, input_ids, args.max_new_tokens)
        )
        watch.lap("stepwise decode compared")

        # 3. End-to-end generate() token parity. HF's generate stops on the
        # generation_config eos ids, so use the same set.
        gen_defaults = load_generation_defaults(checkpoint_dir)
        eos = gen_defaults.get("eos_token_id", None)
        eos_ids = tuple(eos) if isinstance(eos, list) else None
        result = generate(
            model, input_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids
        )
        hf_tokens = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        ours_text = tokenizer.decode(result.sequences[0, input_ids.shape[1] :])
        match = bool(
            result.sequences.shape == hf_tokens.shape
            and torch.equal(result.sequences, hf_tokens)
        )
        comparisons.append(
            Comparison("generate() token ids", match, 0.0 if match else float("nan"))
        )
        print(f"  generated: {ours_text!r}")
        all_ok &= check_all(comparisons)

    if args.long_context:
        # Deterministic pseudo-random token ids (avoiding special tokens).
        gen = torch.Generator().manual_seed(0)
        body = torch.randint(1_000, 200_000, (1, args.long_context - 1), generator=gen)
        input_ids = torch.cat(
            [torch.tensor([[model.config.text.bos_token_id]]), body], dim=-1
        ).to(device)
        print(f"\n=== long context: {args.long_context} random tokens", flush=True)
        comparisons = []
        with torch.no_grad():
            ours = model(input_ids, logits_to_keep=1)
            theirs = hf_model(input_ids=input_ids, logits_to_keep=1)
        comparisons.append(
            compare("prefill logits (last position)", ours.logits, theirs.logits)
        )
        watch.lap("long-context prefill compared")
        comparisons.extend(
            stepwise_decode_comparisons(model, hf_model, input_ids, args.max_new_tokens)
        )
        watch.lap("long-context stepwise decode compared")
        all_ok &= check_all(comparisons)

    if args.image is not None:
        all_ok &= run_image_check(args, checkpoint_dir, model, hf_model, device)

    print(
        "\nPASS: all comparisons bitwise-identical"
        if all_ok
        else "\nFAIL: mismatches found"
    )
    return 0 if all_ok else 1


def stepwise_decode_comparisons(
    model, hf_model, input_ids: Tensor, max_new_tokens: int
) -> list[Comparison]:
    """Drive both models manually one token at a time and compare each step."""
    from transformers import DynamicCache

    comparisons: list[Comparison] = []
    cache = KVCache(model.config.text)
    hf_cache = DynamicCache(config=hf_model.config.text_config)

    with torch.no_grad():
        ours = model(input_ids, cache=cache, logits_to_keep=1)
        theirs = hf_model(
            input_ids=input_ids,
            past_key_values=hf_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        max_diff = 0.0
        bitwise = True
        token = input_ids  # placeholder
        for step in range(max_new_tokens):
            step_cmp = compare(f"step {step}", ours.logits, theirs.logits)
            bitwise &= step_cmp.bitwise_equal
            max_diff = max(max_diff, step_cmp.max_abs_diff)
            token = ours.logits[:, -1, :].float().argmax(dim=-1)[:, None]
            hf_token = theirs.logits[:, -1, :].float().argmax(dim=-1)[:, None]
            if not torch.equal(token, hf_token):
                bitwise = False
                break
            ours = model(token, cache=cache, logits_to_keep=1)
            theirs = hf_model(
                input_ids=token,
                past_key_values=hf_cache,
                use_cache=True,
                logits_to_keep=1,
            )
    comparisons.append(
        Comparison(f"cached decode logits ({max_new_tokens} steps)", bitwise, max_diff)
    )
    return comparisons


def run_image_check(args, checkpoint_dir, model, hf_model, device) -> bool:
    import transformers
    from PIL import Image

    processor = transformers.AutoProcessor.from_pretrained(checkpoint_dir)
    image = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    print(
        f"\n=== image prompt: {args.image} "
        f"({int((batch['input_ids'] == model.config.image_token_id).sum())} soft tokens)"
    )
    with torch.no_grad():
        ours = model(
            batch["input_ids"],
            pixel_values=batch["pixel_values"],
            image_position_ids=batch["image_position_ids"],
        )
        theirs = hf_model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            image_position_ids=batch["image_position_ids"],
        )
    comparisons = [compare("image prefill logits", ours.logits, theirs.logits)]

    result = generate(
        model,
        batch["input_ids"],
        max_new_tokens=args.max_new_tokens,
        pixel_values=batch["pixel_values"],
        image_position_ids=batch["image_position_ids"],
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    print(
        f"  generated: {tokenizer.decode(result.sequences[0, batch['input_ids'].shape[1] :])!r}"
    )
    return check_all(comparisons)


if __name__ == "__main__":
    sys.exit(main())
