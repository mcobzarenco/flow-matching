"""Benchmark eager vs SDPA attention for the pure-torch Gemma4.

Loads the model once and flips the attention backend in place between runs
(`set_attention_backend`), so both backends see identical weights and
allocator state. Workloads:

- prefill: single forward over N random tokens (``logits_to_keep=1`` so the
  lm_head cost does not drown out attention),
- decode: cached greedy generation (includes the per-step host sync of the
  eos check, i.e. realistic single-stream latency),
- image prefill: vision tower + text prefill (a deterministic synthetic
  image by default, ``--image`` to use a file).

Usage::

    uv run python -m bijou.gemma4.bench --device cuda
    uv run python -m bijou.gemma4.bench --device cuda --prefill 512 2048 4096 8192
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
import transformers

from .generation import generate
from .layers import AttentionBackend
from .loading import load_model, resolve_checkpoint_dir
from .model import Gemma4Model, set_attention_backend
from .testing import load_test_image

DEFAULT_MODEL = "google/gemma-4-e2b-it"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def random_ids(n_tokens: int, device: torch.device, bos: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    body = torch.randint(1_000, 200_000, (1, n_tokens - 1), generator=gen)
    return torch.cat([torch.tensor([[bos]]), body], dim=-1).to(device)


def time_prefill(
    model: Gemma4Model, input_ids: torch.Tensor, device: torch.device, iters: int
) -> tuple[float, float]:
    """(median seconds per forward, peak GiB allocated during the forward)."""
    times: list[float] = []
    peak = 0.0
    base = 0
    with torch.no_grad():
        model(input_ids, logits_to_keep=1)  # warmup / autotune / alloc
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
        for _ in range(iters):
            t0 = time.perf_counter()
            model(input_ids, logits_to_keep=1)
            _sync(device)
            times.append(time.perf_counter() - t0)
        if device.type == "cuda":
            peak = (torch.cuda.max_memory_allocated(device) - base) / 2**30
    return statistics.median(times), peak


def time_decode(
    model: Gemma4Model,
    input_ids: torch.Tensor,
    device: torch.device,
    new_tokens: int,
) -> float:
    """Tokens per second of cached greedy decoding."""
    with torch.no_grad():
        # Warmup (also triggers kernel selection for the decode shapes).
        generate(model, input_ids, max_new_tokens=8, eos_token_ids=(-1,))
        _sync(device)
        t0 = time.perf_counter()
        result = generate(
            model, input_ids, max_new_tokens=new_tokens, eos_token_ids=(-1,)
        )
        _sync(device)
        elapsed = time.perf_counter() - t0
    generated = result.sequences.shape[1] - input_ids.shape[1]
    return generated / elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefill", type=int, nargs="*", default=[512, 2048, 4096])
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--decode-prompt", type=int, default=128)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--image",
        default=None,
        help="image file for the image-prefill workload "
        "(default: deterministic synthetic image)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_dir = resolve_checkpoint_dir(args.model)
    print(f"loading {checkpoint_dir.name} on {device} ...", flush=True)
    model = load_model(checkpoint_dir, device=device)
    bos = model.config.text.bos_token_id

    processor = transformers.AutoProcessor.from_pretrained(checkpoint_dir)
    image, image_label = load_test_image(args.image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    image_batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    print(f"image workload: {image_label}", flush=True)

    results: dict[AttentionBackend, dict[str, float]] = {}
    for backend in (AttentionBackend.EAGER, AttentionBackend.SDPA):
        set_attention_backend(model, backend)
        rows: dict[str, float] = {}
        for n in args.prefill:
            ids = random_ids(n, device, bos)
            seconds, peak_gib = time_prefill(model, ids, device, args.iters)
            rows[f"prefill {n} (s)"] = seconds
            rows[f"prefill {n} peak mem (GiB)"] = peak_gib
        rows[f"decode {args.decode_tokens} @ prompt {args.decode_prompt} (tok/s)"] = (
            time_decode(
                model,
                random_ids(args.decode_prompt, device, bos),
                device,
                args.decode_tokens,
            )
        )
        with torch.no_grad():
            t = []
            for _ in range(args.iters + 1):
                t0 = time.perf_counter()
                model(
                    image_batch["input_ids"],
                    pixel_values=image_batch["pixel_values"],
                    image_position_ids=image_batch["image_position_ids"],
                    logits_to_keep=1,
                )
                _sync(device)
                t.append(time.perf_counter() - t0)
            rows["image prefill (s)"] = statistics.median(t[1:])
        results[backend] = rows
        print(f"  {backend}: done", flush=True)

    eager, sdpa = results[AttentionBackend.EAGER], results[AttentionBackend.SDPA]
    width = max(len(k) for k in eager)
    print(f"\n{'workload':<{width}}  {'eager':>12}  {'sdpa':>12}  {'ratio':>8}")
    for key in eager:
        a, b = eager[key], sdpa[key]
        better_high = "tok/s" in key
        ratio = (b / a) if better_high else (a / b if b else float("inf"))
        print(f"{key:<{width}}  {a:>12.4f}  {b:>12.4f}  {ratio:>7.2f}x")
    print("(ratio > 1 means sdpa is better)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
