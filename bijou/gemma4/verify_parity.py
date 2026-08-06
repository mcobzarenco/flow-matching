"""Verify the pure-torch Gemma4 against HF transformers (eager attention).

Contract (deliberately *not* bitwise, to leave room for kernel optimizations):

- hard gates: single-forward logits (prefill, and the first cached-decode
  step) must agree within ``--tolerance`` (default 2.0), and greedy tokens
  must match HF exactly — except that a divergence is accepted when it
  happens at a *near-tie*: the two candidate tokens' logits within
  ``--tolerance`` of each other in **both** implementations at the fork step.
  ULP-scale kernel noise legitimately flips such ties (observed on E4B with
  random-token context and image captions); a confident disagreement still
  fails.
- Measured context for the tolerance default: the eager backend is
  bitwise-identical to HF on H100; the SDPA backend lands at max|Δ| ≈ 0.6-1.8
  on text (≈ 5-14 bf16 ULPs at softcapped-logit scale, from the decoder
  stack of fused kernels skipping the fp32 softmax round-trip). Structural
  bugs produce O(10+) diffs and confidently wrong tokens, so the gates bite.
- informational: bitwise status and multi-step decode drift. Once per-step
  kernels differ by a ULP anywhere, the KV-cache feedback loop amplifies
  differences across steps (deterministic chaos), so drift there is expected
  and only (near-tie-aware) token agreement is enforced.
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
``generate()``, an image prompt (vision path; a deterministic synthetic image
by default, ``--image`` to use a file), a **padded 2-sample x 2-image batch**
(the production-encode regime: mixed-length prompts, HF attention mask,
per-sample logical position_ids — gated against HF on the same padded batch
AND against HF's unpadded per-sample forwards; ``--skip-padded-batch`` to
omit) and optional ``--long-context`` (sliding-window coverage). Prompts are
wrapped in the checkpoint's chat template by default (the correct instruct
usage — generations should be coherent); pass ``--raw`` to feed prompts
verbatim instead. ``--attn-backend`` selects our attention implementation
(the HF reference always runs eager).

``--require-bitwise`` escalates every same-shape HF comparison from the
tolerance gate to bitwise equality — the measured eager/H100 contract (the
docstring's "bitwise today" anchor, previously printed but never enforced).
Near-tie token forks are likewise refused under the flag: bitwise logits
cannot fork. Cross-shape comparisons (a padded batch row vs HF's unpadded
forward run different GEMM shapes, so fp reduction order legitimately
differs) stay tolerance-gated and say so in their name.

Usage::

    uv run python -m bijou.gemma4.verify_parity --device cuda --max-new-tokens 32
    uv run python -m bijou.gemma4.verify_parity --device cuda --attn-backend eager \
        --require-bitwise
    uv run python -m bijou.gemma4.verify_parity --device cuda --attn-backend sdpa \
        --long-context 600
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import Tensor
from transformers import DynamicCache

from ..nn import AttentionBackend
from .cache import KVCache
from .generation import GenerationResult, generate
from .loading import load_generation_defaults, load_model, resolve_checkpoint_dir
from .model import Gemma4Model, set_attention_backend
from .testing import load_test_image, synthetic_test_image

DEFAULT_MODEL = "google/gemma-4-e2b-it"
DEFAULT_PROMPTS = (
    "What are the three primary colors? Answer in one sentence.",
    (
        "You control a robot arm. Plan the steps to pick up the red cube "
        "and place it in the bin."
    ),
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


def compare(
    name: str,
    ours: Tensor,
    theirs: Tensor,
    tolerance: float,
    *,
    require_bitwise: bool = False,
) -> Comparison:
    if ours.shape != theirs.shape:
        raise AssertionError(
            f"{name}: shape {tuple(ours.shape)} != {tuple(theirs.shape)}",
        )
    bitwise = bool(torch.equal(ours, theirs))
    diff = 0.0 if bitwise else (ours.float() - theirs.float()).abs().max().item()
    passed = bitwise if require_bitwise else diff <= tolerance
    detail = "(bitwise required)" if require_bitwise and not bitwise else ""
    return Comparison(name, passed, bitwise, diff, detail)


def check_all(comparisons: list[Comparison]) -> bool:
    ok = True
    for comparison in comparisons:
        print(comparison.report())
        ok &= comparison.passed
    return ok


def compare_generated_tokens(
    name: str,
    result: GenerationResult,
    hf_output: Any,
    prompt_len: int,
    tolerance: float,
    *,
    require_bitwise: bool = False,
) -> Comparison:
    """Token-level comparison of our generate() vs HF generate(). Sequences
    must match exactly, except a single fork at a near-tie step is accepted
    (comparison stops there — post-fork continuations legitimately differ).
    Under ``require_bitwise`` the near-tie escape is refused: bitwise-equal
    logits cannot fork, so any disagreement is a failure."""
    ours_tokens: list[int] = result.sequences[0, prompt_len:].tolist()
    theirs_tokens: list[int] = hf_output.sequences[0, prompt_len:].tolist()
    if ours_tokens == theirs_tokens:
        return Comparison(name, True, True, 0.0)
    # strict=False: a length mismatch is a legitimate outcome, reported below.
    for step, (ours_token, theirs_token) in enumerate(
        zip(ours_tokens, theirs_tokens, strict=False),
    ):
        if ours_token == theirs_token:
            continue
        near, gap = is_near_tie(
            result.step_logits[step].float(),
            hf_output.logits[step].float(),
            ours_token,
            theirs_token,
            tolerance,
        )
        detail = (
            f"(forked at step {step}: token {ours_token} vs {theirs_token}, "
            f"top-2 gap {gap:.3f})"
        )
        return Comparison(name, near and not require_bitwise, False, gap, detail)
    return Comparison(
        name,
        False,
        False,
        float("nan"),
        f"(length mismatch: {len(ours_tokens)} vs {len(theirs_tokens)} tokens)",
    )


def is_near_tie(
    ours_logits: Tensor,
    theirs_logits: Tensor,
    ours_token: int,
    theirs_token: int,
    tolerance: float,
) -> tuple[bool, float]:
    """Whether a greedy-token disagreement is a genuine near-tie.

    True iff the two candidate tokens' logits are within ``tolerance`` of
    each other in *both* implementations (ULP-scale noise can then flip the
    argmax legitimately). Returns (near_tie, max_gap). Logits: [1, V] fp32.
    """
    gap_ours = float((ours_logits[0, ours_token] - ours_logits[0, theirs_token]).abs())
    gap_theirs = float(
        (theirs_logits[0, ours_token] - theirs_logits[0, theirs_token]).abs(),
    )
    max_gap = max(gap_ours, gap_theirs)
    return max_gap <= tolerance, max_gap


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
        "--image",
        default=None,
        help="image file for the multimodal check "
        "(default: deterministic synthetic image)",
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
    parser.add_argument(
        "--require-bitwise",
        action="store_true",
        help="escalate same-shape HF comparisons from the tolerance gate to "
        "bitwise equality (the measured eager/H100 contract); near-tie token "
        "forks are refused too",
    )
    parser.add_argument(
        "--skip-padded-batch",
        action="store_true",
        help="skip the padded 2-sample x 2-image batch check",
    )
    args = parser.parse_args()

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
        checkpoint_dir,
        dtype=torch.bfloat16,
        attn_implementation="eager",
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
            args,
            model,
            hf_model,
            tokenizer,
            checkpoint_dir,
            device,
            watch,
        )

    print("\nPASS" if all_ok else "\nFAIL")
    return 0 if all_ok else 1


def run_checks(
    args: argparse.Namespace,
    model: Gemma4Model,
    hf_model: Any,
    tokenizer: Any,
    checkpoint_dir: Path,
    device: torch.device,
    watch: Stopwatch,
) -> bool:
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
                require_bitwise=args.require_bitwise,
            ),
        )
        watch.lap("prefill compared")

        # 2. Cached stepwise decode, greedy.
        comparisons.extend(
            stepwise_decode_comparisons(
                model,
                hf_model,
                input_ids,
                args.max_new_tokens,
                args.tolerance,
                require_bitwise=args.require_bitwise,
            ),
        )
        watch.lap("stepwise decode compared")

        # 3. End-to-end generate() token parity (hard gate, near-tie aware).
        # HF's generate stops on the generation_config eos ids, so use the
        # same set.
        eos_ids = load_generation_defaults(checkpoint_dir).eos_token_ids
        result = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_ids,
        )
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_logits=True,
        )
        ours_text = tokenizer.decode(result.sequences[0, input_ids.shape[1] :])
        comparisons.append(
            compare_generated_tokens(
                "generate() token ids",
                result,
                hf_output,
                input_ids.shape[1],
                args.tolerance,
                require_bitwise=args.require_bitwise,
            ),
        )
        print(f"  generated: {ours_text!r}")
        all_ok &= check_all(comparisons)

    if args.long_context:
        # Deterministic pseudo-random token ids (avoiding special tokens).
        gen = torch.Generator().manual_seed(0)
        body = torch.randint(1_000, 200_000, (1, args.long_context - 1), generator=gen)
        input_ids = torch.cat(
            [torch.tensor([[model.config.text.bos_token_id]]), body],
            dim=-1,
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
                require_bitwise=args.require_bitwise,
            ),
        )
        watch.lap("long-context prefill compared")
        comparisons.extend(
            stepwise_decode_comparisons(
                model,
                hf_model,
                input_ids,
                args.max_new_tokens,
                args.tolerance,
                require_bitwise=args.require_bitwise,
            ),
        )
        watch.lap("long-context stepwise decode compared")
        all_ok &= check_all(comparisons)

    all_ok &= run_image_check(args, checkpoint_dir, model, hf_model, device)

    if not args.skip_padded_batch:
        all_ok &= run_padded_batch_check(args, checkpoint_dir, model, hf_model, device)
        watch.lap("padded batch compared")

    return all_ok


def stepwise_decode_comparisons(
    model: Gemma4Model,
    hf_model: Any,
    input_ids: Tensor,
    max_new_tokens: int,
    tolerance: float,
    *,
    require_bitwise: bool = False,
) -> list[Comparison]:
    """Drive both models manually one token at a time.

    Hard gates: the first step's logits (cached prefill, no feedback yet)
    within tolerance, and per-step greedy token agreement. Later steps' logit
    drift is reported for information: once any kernel differs by a ULP, the
    cache feedback loop amplifies differences across steps even between two
    runs of the *same* implementation (observed with HF vs itself on CPU).
    """
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
                require_bitwise=require_bitwise,
            ),
        )

        max_drift = 0.0
        first_divergent: int | None = None
        bitwise = True
        tokens_match = True
        detail = ""
        steps_run = 0
        for step in range(max_new_tokens):
            steps_run = step + 1
            step_cmp = compare(f"step {step}", ours.logits, theirs.logits, tolerance)
            if not step_cmp.bitwise_equal:
                bitwise = False
                if first_divergent is None:
                    first_divergent = step
            max_drift = max(max_drift, step_cmp.max_abs_diff)
            ours_logits = ours.logits[:, -1, :].float()
            theirs_logits = theirs.logits[:, -1, :].float()
            token = ours_logits.argmax(dim=-1)[:, None]
            hf_token = theirs_logits.argmax(dim=-1)[:, None]
            if not torch.equal(token, hf_token):
                near, gap = is_near_tie(
                    ours_logits,
                    theirs_logits,
                    int(token.item()),
                    int(hf_token.item()),
                    tolerance,
                )
                tokens_match = near and not require_bitwise
                detail = (
                    f"(forked at step {step}, top-2 gap {gap:.3f}"
                    f"{', near-tie' if near else ''})"
                )
                break
            ours = model(token, cache=cache, logits_to_keep=1)
            theirs = hf_model(
                input_ids=token,
                past_key_values=hf_cache,
                use_cache=True,
                logits_to_keep=1,
            )

    if not detail and first_divergent is not None:
        detail = f"(first drift at step {first_divergent})"
    passed = tokens_match and (bitwise or not require_bitwise)
    if require_bitwise and tokens_match and not bitwise:
        detail = f"(bitwise required) {detail}".rstrip()
    comparisons.append(
        Comparison(
            f"decode tokens agree ({steps_run} steps)",
            passed,
            bitwise,
            max_drift,
            detail,
        ),
    )
    return comparisons


def run_image_check(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    model: Gemma4Model,
    hf_model: Any,
    device: torch.device,
) -> bool:
    processor = transformers.AutoProcessor.from_pretrained(checkpoint_dir)
    image, image_label = load_test_image(args.image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        },
    ]
    batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    print(
        f"\n=== image prompt: {image_label} "
        f"({int((batch['input_ids'] == model.config.image_token_id).sum())} soft tokens)",
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
            require_bitwise=args.require_bitwise,
        ),
    ]
    all_diff = (ours.logits.float() - theirs.logits.float()).abs().max().item()
    print(f"  (info) all-positions max|Δ|={all_diff:.3e}")

    eos_ids = load_generation_defaults(checkpoint_dir).eos_token_ids
    result = generate(
        model,
        batch["input_ids"],
        max_new_tokens=args.max_new_tokens,
        pixel_values=batch["pixel_values"],
        image_position_ids=batch["image_position_ids"],
        eos_token_ids=eos_ids,
    )
    hf_output = hf_model.generate(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_position_ids=batch["image_position_ids"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
    )
    comparisons.append(
        compare_generated_tokens(
            "image generate() token ids",
            result,
            hf_output,
            batch["input_ids"].shape[1],
            args.tolerance,
            require_bitwise=args.require_bitwise,
        ),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    print(
        f"  generated: {tokenizer.decode(result.sequences[0, batch['input_ids'].shape[1] :])!r}",
    )
    return check_all(comparisons)


PADDED_BATCH_PROMPTS = (
    "Describe both images in one sentence.",
    (
        "You see two camera views of a robot workspace, one overhead and "
        "one wrist-mounted. Compare what each camera shows, then plan how "
        "to pick up the red cube and place it in the bin, step by step."
    ),
)


def run_padded_batch_check(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    model: Gemma4Model,
    hf_model: Any,
    device: torch.device,
) -> bool:
    """The production-encode regime (deep-dive finding 7): a padded batch of
    two mixed-length prompts with TWO images each, forwarded with the HF
    attention mask and per-sample logical position_ids (cumsum of the mask —
    exactly ``GemmaEncoder.encode_tensors``'s convention, passed to HF too
    since its forward defaults to arange).

    BOTH padding orientations are checked: the processor's native side and
    the per-row roll to the opposite side (the production collator uses the
    processor's native padding; the ar_backbone prompt path collates LEFT —
    both must agree with HF's padded-multimodal conventions). Scope note:
    in a single full forward, positions enter only through RoPE, which is
    relative — so logical-vs-arange is a per-sample constant shift here,
    distinguishable only at fp-rotation-noise scale, and this check gates
    mask + padding + multi-image semantics rather than the position CHAIN;
    position-chain correctness across cached continuation (where the
    convention does bite) is pinned by ``tests/test_backbone_continuation``.

    Gates, per sample at its last REAL position:
      - vs HF on the *same padded batch* (same shapes — bitwise-eligible
        under ``--require-bitwise``);
      - vs HF's *unpadded single-sample* forward (cross-shape: different
        GEMM shapes ⇒ tolerance only) — pins that padding + mask + logical
        positions reproduce the canonical unpadded answer, not merely that
        both implementations share a convention. The last REAL position is
        deliberately the gate: deep-early positions of a long padded row
        legitimately diverge from the unpadded forward on sliding-window
        layers (physical-index windows lose real context to pad slots — the
        same documented effect as right-padded continuation).

    Token identity of each sample against its solo tokenization is asserted
    first (also fixes the padding orientation without assuming one). The
    state-token splice has no HF counterpart and stays covered by bijou's
    own tests."""
    processor = transformers.AutoProcessor.from_pretrained(checkpoint_dir)
    cameras = (synthetic_test_image(), synthetic_test_image(320, 240))
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cameras[0]},
                    {"type": "image", "image": cameras[1]},
                    {"type": "text", "text": text},
                ],
            },
        ]
        for text in PADDED_BATCH_PROMPTS
    ]
    batch = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        # transformers 5.14: per-call processor kwargs must be nested (the
        # production collator's verified pattern).
        processor_kwargs={"padding": True},
    ).to(device)
    input_ids = batch["input_ids"]
    padding_mask = batch["attention_mask"]
    if not bool((padding_mask == 0).any()):
        raise AssertionError(
            "padded-batch check: prompts tokenized to equal lengths — no "
            "padding exercised; make the prompts more different",
        )
    real = padding_mask.to(torch.bool)
    n_images = batch["pixel_values"].shape[0]
    n_pad = int((padding_mask == 0).sum())
    print(
        f"\n=== padded batch: {input_ids.shape[0]} samples x 2 images "
        f"({n_images} images total, seq {input_ids.shape[1]}, "
        f"{n_pad} pad tokens)",
    )

    # Per-sample unpadded HF reference (also proves padding kept token
    # identity: the batch's real tokens must equal the solo tokenization).
    solo_last_logits: list[Tensor] = []
    for i, conversation in enumerate(conversations):
        solo = processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        if not torch.equal(solo["input_ids"][0], input_ids[i][real[i]]):
            raise AssertionError(
                f"padded-batch check: sample {i}'s real tokens differ from "
                "its solo tokenization — padding changed token identity",
            )
        with torch.no_grad():
            solo_out = hf_model(
                input_ids=solo["input_ids"],
                pixel_values=solo["pixel_values"],
                image_position_ids=solo["image_position_ids"],
            )
        solo_last_logits.append(solo_out.logits[:, -1, :])

    def orientation_pass(side: str, ids: Tensor, mask: Tensor) -> bool:
        # Logical positions, exactly as GemmaEncoder.encode_tensors builds
        # them; passed to HF too (its forward defaults to arange).
        position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            ours = model(
                ids,
                pixel_values=batch["pixel_values"],
                image_position_ids=batch["image_position_ids"],
                padding_mask=mask,
                position_ids=position_ids,
            )
            theirs = hf_model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids,
                pixel_values=batch["pixel_values"],
                image_position_ids=batch["image_position_ids"],
            )
        last_real = mask.long().cumsum(-1).argmax(-1)
        rows = torch.arange(ids.shape[0], device=device)
        ours_last = ours.logits[rows, last_real, :]
        theirs_last = theirs.logits[rows, last_real, :]
        comparisons = [
            compare(
                f"{side}-padded batch logits (last real position)",
                ours_last,
                theirs_last,
                args.tolerance,
                require_bitwise=args.require_bitwise,
            ),
        ]
        real_cols = mask.to(torch.bool)
        real_diff = (
            (ours.logits.float() - theirs.logits.float()).abs()[real_cols].max().item()
        )
        print(f"  ({side}) all-real-positions max|Δ|={real_diff:.3e}")
        for i, solo_logits in enumerate(solo_last_logits):
            comparisons.append(
                compare(
                    f"{side}: sample {i} vs unpadded HF (last position, tol only)",
                    ours_last[i : i + 1],
                    solo_logits,
                    args.tolerance,
                ),
            )
        return check_all(comparisons)

    pads_per_row = (~real).long().sum(-1)
    right_padded = bool(real[:, 0].all())
    native = "native-right" if right_padded else "native-left"
    ok = orientation_pass(native, input_ids, padding_mask)

    # The opposite orientation, by rolling each row's pads to the other
    # side (images stay put — image tokens are real tokens and keep their
    # relative order, so pixel_values/image_position_ids are unchanged).
    direction = 1 if right_padded else -1
    flipped_ids = torch.stack(
        [
            row.roll(direction * int(k))
            for row, k in zip(input_ids, pads_per_row, strict=True)
        ],
    )
    flipped_mask = torch.stack(
        [
            row.roll(direction * int(k))
            for row, k in zip(padding_mask, pads_per_row, strict=True)
        ],
    )
    other = "left" if right_padded else "right"
    ok &= orientation_pass(other, flipped_ids, flipped_mask)
    return ok


if __name__ == "__main__":
    sys.exit(main())
