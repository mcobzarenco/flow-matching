"""Verify the pure-torch Gemma4 against HF transformers (eager attention).

Contract (deliberately *not* bitwise, to leave room for kernel optimizations):

- hard gates: greedy tokens must match HF exactly; single-forward logits
  (prefill, and the first cached-decode step) must agree within
  ``--tolerance`` (default 2.0). Measured context for the default: the eager
  backend is bitwise-identical to HF on H100; the SDPA backend lands at
  max|Δ| ≈ 0.6-1.8 on text (≈ 5-14 bf16 ULPs at softcapped-logit scale,
  from 35 layers of fused kernels skipping the fp32 softmax round-trip)
  while producing identical greedy tokens. Structural bugs produce O(10+)
  diffs and wrong tokens, so the gate still bites.
- informational: bitwise status and multi-step decode drift. Once per-step
  kernels differ by a ULP anywhere, the KV-cache feedback loop amplifies
  differences across steps (deterministic chaos), so drift there is expected
  and only token agreement is enforced.
- image prompts gate on *last-position* logits and generated tokens: logits
  at interior image-token positions are never consumed and are hypersensitive
  to ULP-scale soft-token noise.

Measured reference for the drift scale (laptop CPU, oneDNN bf16, HF compared
against *itself* in one process): most decode runs are identical; when one
diverges, the first divergent step's logits differ by median 1.0 / max 5 bf16
ULPs (max|Δ|=0.375 at softcapped-logit scale), then drift persists via the
cache; greedy argmax agreed at every step regardless. On H100 CUDA both
implementations are bitwise-identical and run-to-run stable.

Checks: full prefill forward, cached stepwise greedy decode, end-to-end
``generate()``, optional ``--long-context`` (sliding-window coverage) and
``--image`` (vision path). Prompts are wrapped in the checkpoint's chat
template by default (the correct instruct usage — generations should be
coherent); pass ``--raw`` to feed prompts verbatim instead. ``--attn-backend``
selects our attention implementation (the HF reference always runs eager).

Usage::

    uv run python -m gemma4.verify_parity --device cuda --max-new-tokens 32
    uv run python -m gemma4.verify_parity --device cuda --attn-backend sdpa \
        --long-context 600 --image /tmp/parity_test.png
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
from .config import AttentionBackend
from .generation import generate
from .loading import load_generation_defaults, load_model, resolve_checkpoint_dir
from .model import set_attention_backend

DEFAULT_MODEL = "google/gemma-4-e2b-it"
DEFAULT_PROMPTS = (
    "What are the three primary colors? Answer in one sentence.",
    "You control a robot arm. Plan the steps to pick up the red cube "
    "and place it in the bin.",
)


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def lap(self, label: str) -> None:
        print(f"  [{time.monotonic() - self.t0:7.1f}s] {label}", flush=True)


@dataclass(frozen=True, slots=True)
class Comparison:
    name: str
    passed: bool
    bitwise_equal: bool
    max_abs_diff: float
    detail: str = ""

    def report(self) -> str:
        if self.bitwise_equal:
            status = "OK (bitwise)"
        elif self.passed:
            status = "OK (within tol)"
        else:
            status = "FAILED"
        detail = f"  {self.detail}" if self.detail else ""
        return f"  {self.name:<42} {status:<16} max|Δ|={self.max_abs_diff:.3e}{detail}"


def compare(name: str, ours: Tensor, theirs: Tensor, tolerance: float) -> Comparison:
    if ours.shape != theirs.shape:
        raise AssertionError(
            f"{name}: shape {tuple(ours.shape)} != {tuple(theirs.shape)}"
        )
    bitwise = bool(torch.equal(ours, theirs))
    diff = 0.0 if bitwise else (ours.float() - theirs.float()).abs().max().item()
    return Comparison(name, diff <= tolerance, bitwise, diff)


def check_all(comparisons: list[Comparison]) -> bool:
    ok = True
    for comparison in comparisons:
        print(comparison.report())
        ok &= comparison.passed
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="feed prompts verbatim instead of applying the chat template",
    )
    parser.add_argument(
        "--attn-backend",
        choices=[*[b.value for b in AttentionBackend], "both"],
        default="both",
        help="attention implementation(s) for the pure-torch model",
    )
    parser.add_argument(
        "--image", default=None, help="path to an image for the multimodal check"
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="max abs logit difference allowed for single-forward comparisons",
    )
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

    if args.attn_backend == "both":
        backends = list(AttentionBackend)
    else:
        backends = [AttentionBackend(args.attn_backend)]

    all_ok = True
    for backend in backends:
        set_attention_backend(model, backend)
        print(f"\n######## attention backend: {backend} ########", flush=True)
        all_ok &= run_checks(
            args, model, hf_model, tokenizer, checkpoint_dir, device, watch
        )

    print("\nPASS" if all_ok else "\nFAIL")
    return 0 if all_ok else 1


def run_checks(args, model, hf_model, tokenizer, checkpoint_dir, device, watch) -> bool:
    prompts = args.prompt if args.prompt else list(DEFAULT_PROMPTS)
    all_ok = True

    for prompt in prompts:
        style = "raw" if args.raw else "chat template"
        print(f"\n=== prompt ({style}): {prompt!r}", flush=True)
        if args.raw:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        else:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = templated["input_ids"]
        input_ids = input_ids.to(device)
        comparisons: list[Comparison] = []

        # 1. Full forward, no cache: all prompt positions (hard gate).
        with torch.no_grad():
            ours = model(input_ids)
            theirs = hf_model(input_ids=input_ids)
        comparisons.append(
            compare(
                "prefill logits (all positions)",
                ours.logits,
                theirs.logits,
                args.tolerance,
            )
        )
        watch.lap("prefill compared")

        # 2. Cached stepwise decode, greedy.
        comparisons.extend(
            stepwise_decode_comparisons(
                model, hf_model, input_ids, args.max_new_tokens, args.tolerance
            )
        )
        watch.lap("stepwise decode compared")

        # 3. End-to-end generate() token parity (hard gate). HF's generate
        # stops on the generation_config eos ids, so use the same set.
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
            Comparison(
                "generate() token ids", match, match, 0.0 if match else float("nan")
            )
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
            compare(
                "prefill logits (last position)",
                ours.logits,
                theirs.logits,
                args.tolerance,
            )
        )
        watch.lap("long-context prefill compared")
        comparisons.extend(
            stepwise_decode_comparisons(
                model, hf_model, input_ids, args.max_new_tokens, args.tolerance
            )
        )
        watch.lap("long-context stepwise decode compared")
        all_ok &= check_all(comparisons)

    if args.image is not None:
        all_ok &= run_image_check(args, checkpoint_dir, model, hf_model, device)

    return all_ok


def stepwise_decode_comparisons(
    model, hf_model, input_ids: Tensor, max_new_tokens: int, tolerance: float
) -> list[Comparison]:
    """Drive both models manually one token at a time.

    Hard gates: the first step's logits (cached prefill, no feedback yet)
    within tolerance, and per-step greedy token agreement. Later steps' logit
    drift is reported for information: once any kernel differs by a ULP, the
    cache feedback loop amplifies differences across steps even between two
    runs of the *same* implementation (observed with HF vs itself on CPU).
    """
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
        comparisons.append(
            compare(
                "cached prefill logits (last position)",
                ours.logits,
                theirs.logits,
                tolerance,
            )
        )

        max_drift = 0.0
        first_divergent: int | None = None
        bitwise = True
        tokens_match = True
        steps_run = 0
        for step in range(max_new_tokens):
            steps_run = step + 1
            step_cmp = compare(f"step {step}", ours.logits, theirs.logits, tolerance)
            if not step_cmp.bitwise_equal:
                bitwise = False
                if first_divergent is None:
                    first_divergent = step
            max_drift = max(max_drift, step_cmp.max_abs_diff)
            token = ours.logits[:, -1, :].float().argmax(dim=-1)[:, None]
            hf_token = theirs.logits[:, -1, :].float().argmax(dim=-1)[:, None]
            if not torch.equal(token, hf_token):
                tokens_match = False
                break
            ours = model(token, cache=cache, logits_to_keep=1)
            theirs = hf_model(
                input_ids=token,
                past_key_values=hf_cache,
                use_cache=True,
                logits_to_keep=1,
            )

    detail = (
        "" if first_divergent is None else f"(first drift at step {first_divergent})"
    )
    comparisons.append(
        Comparison(
            f"decode tokens agree ({steps_run} steps)",
            tokens_match,
            bitwise,
            max_drift,
            detail,
        )
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
    # Gate on the last position only: logits at interior image-token slots are
    # never consumed for prediction and are hypersensitive to ULP-scale noise
    # in the 266+ soft tokens. The all-positions diff is reported as info.
    comparisons = [
        compare(
            "image prefill logits (last position)",
            ours.logits[:, -1:, :],
            theirs.logits[:, -1:, :],
            args.tolerance,
        )
    ]
    all_diff = (ours.logits.float() - theirs.logits.float()).abs().max().item()
    print(f"  (info) all-positions max|Δ|={all_diff:.3e}")

    gen_defaults = load_generation_defaults(checkpoint_dir)
    eos = gen_defaults.get("eos_token_id", None)
    eos_ids = tuple(eos) if isinstance(eos, list) else None
    result = generate(
        model,
        batch["input_ids"],
        max_new_tokens=args.max_new_tokens,
        pixel_values=batch["pixel_values"],
        image_position_ids=batch["image_position_ids"],
        eos_token_ids=eos_ids,
    )
    hf_tokens = hf_model.generate(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_position_ids=batch["image_position_ids"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    match = bool(
        result.sequences.shape == hf_tokens.shape
        and torch.equal(result.sequences, hf_tokens)
    )
    comparisons.append(
        Comparison(
            "image generate() token ids", match, match, 0.0 if match else float("nan")
        )
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    print(
        f"  generated: {tokenizer.decode(result.sequences[0, batch['input_ids'].shape[1] :])!r}"
    )
    return check_all(comparisons)


if __name__ == "__main__":
    sys.exit(main())
