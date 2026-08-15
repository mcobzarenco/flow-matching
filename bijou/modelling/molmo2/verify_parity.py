"""Verify the pure-torch Molmo2 text decoder against HF transformers.

WP1 scope: the TEXT path only (input_ids -> hidden states -> logits) — the
vision tower does not exist yet, so image parity lands with WP2. Both
models run fp32 (the checkpoint's shipped dtype) with eager attention by
default, so differences are fp-reduction noise, not kernel families:

- hard gates: final logits within ``--tolerance`` at every position, and
  greedy argmax agreement at every position (single forward — the port has
  no KV cache by design, D1);
- informational: per-layer max|Δ| over the residual stream (our
  ``residual_taps`` vs HF ``output_hidden_states``) — the first layer whose
  diff explodes localizes a convention bug immediately;
- ``--check-mount``: additionally load the truncated 15-layer mount from
  the same (sharded, multi-file) checkpoint and require its taps to match
  the full model's prefix taps bitwise — the loader gate on real weights.

CPU-friendly on purpose (2-3 fp32 copies of a 4B model ≈ 40-60 GB RAM):
runs at any time without touching live GPUs.

Usage::

    uv run python -m bijou.modelling.molmo2.verify_parity
    uv run python -m bijou.modelling.molmo2.verify_parity --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ..nn import AttentionBackend
from .loading import load_text_model, resolve_checkpoint_dir
from .text import Molmo2TextModel

DEFAULT_PROMPTS = (
    "The robot arm picks up the red block and places it in the bin.",
    (
        "A single-arm SO-101 manipulator observes the workspace through two "
        "cameras: one overhead, one mounted on the wrist. The task is to "
        "grasp the marker and drop it into the cup without knocking over "
        "the tower of blocks standing between them."
    ),
    "def fibonacci(n):\n    if n < 2:\n        return n",
)


@dataclass(frozen=True, slots=True)
class Comparison:
    name: str
    passed: bool
    max_abs_diff: float
    detail: str = ""

    def report(self) -> str:
        status = "OK" if self.passed else "FAIL"
        detail = f"  {self.detail}" if self.detail else ""
        return f"  {self.name:<46} {status:<5} max|Δ|={self.max_abs_diff:.3e}{detail}"


def max_abs(a: Tensor, b: Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="allenai/Molmo2-4B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help="both models run in this dtype (float32 = the shipped shards)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="hard gate on final-logit max|Δ| (fp32/eager measured headroom; "
        "structural bugs produce O(1)+ diffs)",
    )
    parser.add_argument(
        "--attn-backend",
        type=AttentionBackend,
        default=AttentionBackend.EAGER,
        choices=tuple(AttentionBackend),
    )
    parser.add_argument("--prompts", nargs="*", default=list(DEFAULT_PROMPTS))
    parser.add_argument(
        "--check-mount",
        action="store_true",
        help="also gate the truncated 15-layer mount bitwise vs the full prefix",
    )
    parser.add_argument("--mount-layers", type=int, default=15)
    parser.add_argument(
        "--vision",
        action="store_true",
        help="also gate the vision backbone on real processor inputs "
        "(WP2); the HF reference runs SDPA there — its eager path drops "
        "the pooling attention mask, SDPA is the shipped semantics our "
        "implementation mirrors",
    )
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    checkpoint_dir = resolve_checkpoint_dir(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    start = time.monotonic()
    print(f"loading HF reference ({args.dtype}, eager, trust_remote_code) ...")
    # The checkpoint's auto_map registers only AutoModelForImageTextToText.
    reference = AutoModelForImageTextToText.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
    )
    reference.eval().requires_grad_(False)
    reference.to(device)  # pyright: ignore[reportArgumentType] — HF stub quirk
    print(f"  {time.monotonic() - start:.1f}s")

    start = time.monotonic()
    print(f"loading bijou port ({args.dtype}, {args.attn_backend}) ...")
    ours = load_text_model(
        checkpoint_dir,
        device=device,
        dtype=dtype,
        attn_backend=args.attn_backend,
    )
    print(f"  {time.monotonic() - start:.1f}s")
    num_layers = ours.transformer.config.num_hidden_layers

    mount: Molmo2TextModel | None = None
    if args.check_mount:
        mount = load_text_model(
            checkpoint_dir,
            device=device,
            dtype=dtype,
            attn_backend=args.attn_backend,
            truncate_layers=args.mount_layers,
        )

    comparisons: list[Comparison] = []
    for prompt in args.prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        label = f"[{prompt[:24]!r}… len={input_ids.shape[1]}]"

        with torch.no_grad():
            hf_out = reference(input_ids=input_ids, output_hidden_states=True)
            sink: dict[int, Tensor] = {}
            logits = ours(
                input_ids,
                residual_taps=range(num_layers),
                residual_sink=sink,
            )

        # Informational: first divergence in the residual stream localizes
        # a convention bug to its layer. HF hidden_states[0] is the
        # embedding output; [i + 1] is decoder block i's output — EXCEPT
        # the last entry, which HF stores post-final-norm (our taps are the
        # raw residual stream, so the last one is compared through ln_f).
        # Measured on CPU fp32/eager: every entry is bitwise 0.0.
        embed_diff = max_abs(hf_out.hidden_states[0], ours.transformer.wte(input_ids))
        print(f"{label}")
        print(f"  embeddings                                   max|Δ|={embed_diff:.3e}")
        worst_layer, worst_diff = -1, 0.0
        for i in range(num_layers):
            tap = sink[i]
            if i == num_layers - 1:
                tap = ours.transformer.ln_f(tap)
            diff = max_abs(hf_out.hidden_states[i + 1], tap)
            if diff >= worst_diff:
                worst_layer, worst_diff = i, diff
        print(
            f"  residual stream ({num_layers} layers)                  "
            f"worst layer {worst_layer}: max|Δ|={worst_diff:.3e}",
        )

        hf_logits = hf_out.logits
        logit_diff = max_abs(hf_logits, logits)
        comparisons.append(
            Comparison(
                name=f"logits {label}",
                passed=logit_diff <= args.tolerance,
                max_abs_diff=logit_diff,
            ),
        )
        ours_argmax = logits.argmax(-1)
        hf_argmax = hf_logits.argmax(-1)
        agree = int((ours_argmax == hf_argmax).sum())
        total = ours_argmax.numel()
        comparisons.append(
            Comparison(
                name=f"greedy argmax {label}",
                passed=agree == total,
                max_abs_diff=logit_diff,
                detail=f"{agree}/{total} positions agree",
            ),
        )

        if mount is not None:
            mount_sink: dict[int, Tensor] = {}
            with torch.no_grad():
                mount(
                    input_ids,
                    residual_taps=range(args.mount_layers),
                    residual_sink=mount_sink,
                )
            bitwise = all(
                torch.equal(mount_sink[i], sink[i]) for i in range(args.mount_layers)
            )
            comparisons.append(
                Comparison(
                    name=f"mount[0..{args.mount_layers - 1}] bitwise {label}",
                    passed=bitwise,
                    max_abs_diff=0.0 if bitwise else float("nan"),
                ),
            )

    if args.vision:
        comparisons += verify_vision(
            checkpoint_dir,
            dtype=dtype,
            device=device,
            tolerance=args.tolerance,
        )

    print()
    for comparison in comparisons:
        print(comparison.report())
    ok = all(c.passed for c in comparisons)
    print("PARITY PASSED" if ok else "PARITY FAILED")
    return 0 if ok else 1


def verify_vision(
    checkpoint_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    tolerance: float,
) -> list[Comparison]:
    """Gate the vision backbone on REAL processor inputs.

    The processor builds the crop set + ``pooled_patches_idx`` for the
    synthetic test image; the reference's own ``build_batched_images``
    converts to the backbone's batched inputs (crop geometry stays the
    reference's job until WP3), then both backbones run the same tensors.
    A fresh SDPA reference is loaded for this: the shipped semantics apply
    the pooling attention mask, which the HF eager path silently drops.
    """
    from transformers import AutoProcessor

    from ..gemma4.testing import synthetic_test_image
    from .loading import load_vision_backbone

    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    reference = AutoModelForImageTextToText.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    reference.eval().requires_grad_(False)
    reference.to(device)  # pyright: ignore[reportArgumentType] — HF stub quirk
    ours = load_vision_backbone(checkpoint_dir, device=device, dtype=dtype)

    # The placeholder is load-bearing: without it the processor emits no
    # image tokens and the reference's build_batched_images sees 0 images.
    inputs = processor(
        images=[synthetic_test_image()],
        text="<|image|> What is in this image?",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        images, pooled_patches_idx = reference.model.build_batched_images(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_token_pooling=inputs["image_token_pooling"],
            image_grids=inputs["image_grids"],
            image_num_crops=inputs["image_num_crops"],
        )
        hf_features = reference.model.vision_backbone(images, pooled_patches_idx)
        our_features = ours(images, pooled_patches_idx)

    # The projector output scale is O(1e4) (measured max ≈ 2.7e4), so an
    # absolute gate is meaningless here — gate RELATIVE to the feature
    # scale. Measured (fp32, our eager vs HF SDPA): ≤ 5e-7 relative, with
    # every partial (crop-edge) pooling group agreeing exactly — the mask
    # semantics gate.
    diff = max_abs(hf_features, our_features)
    scale = hf_features.abs().max().item()
    relative = diff / scale if scale else diff
    detail = (
        f"rel={relative:.3e} over scale {scale:.3g}; "
        f"{tuple(our_features.shape)} image tokens (crops + global view)"
    )
    return [
        Comparison(
            name="vision backbone [synthetic 640x480]",
            passed=our_features.shape == hf_features.shape and relative <= tolerance,
            max_abs_diff=diff,
            detail=detail,
        ),
    ]


if __name__ == "__main__":
    sys.exit(main())
