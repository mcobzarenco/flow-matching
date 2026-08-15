"""MolmoAct2 action-side processing — prompt template, discrete state
tokens, q01/q99 norm-stats.

The backbone-side image/prompt machinery is reused from ``bijou.molmo2``;
this module holds the ACTION-side deltas their serving stack applies on
top, mirrored op-for-op from two shipped sources and parity-gated
against them executing on the same inputs
(``fontaine/scripts/molmoact2_processing_goldens.py`` run in their venv
banks the reference outputs; ``tests/test_molmoact2_processing.py``
reproduces them byte-exact here):

- their lerobot policy step ``MolmoAct2PackInputsProcessorStep`` and
  the surrounding normalizer/clamp steps
  (``lerobot/policies/molmoact2/processor_molmoact2.py``): q01/q99
  normalization, task-text normalization, the discrete-state prompt
  clause, the robot prompt template, uint8 image coercion, BOS insert;
- their HF remote-code processor/image processor
  (``processing_molmoact2.py`` / ``image_processing_molmoact2.py``) at
  the shipped operating point ``crop_mode='resize'``: ONE 378x378
  global view per image, 2x2-pooled to a fixed 14x14 = 196 patch
  tokens, no crop tiling, no col tokens (``use_single_crop_col_tokens``
  false), grid row ``(14, 14, 0, 0)``.

Two facts worth calling out because they differ from ``bijou.molmo2``:

1. The MolmoAct2 tokenizer re-homes the image special tokens (its extra
   vocab inserts state/action/setup/control tokens first), so the image
   ids here are NOT molmo2's ``151_936..``-range constants — every id
   below is pinned from the MolmoAct2 checkpoint tokenizer and
   re-verified against the loaded tokenizer in the oracle tests.
2. The image path is uint8 end-to-end: their pack step coerces every
   frame to uint8 RGB HWC (floats <= 1 are scaled by 255) BEFORE the
   resize, and the resize itself runs on the uint8 tensor and rounds
   back to uint8. A float-path resize (the molmo2 convention) is NOT
   equivalent — the double quantization is part of the distribution the
   checkpoint was trained on.

State is DISCRETE at inference (``state_format='discrete'``): the
q01/q99-normalized, clamped state enters the prompt as 256-bin
``<state_N>`` tokens; the action expert's continuous ``state_embeddings``
path is train-only and unused here (see ``wiring.py``).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms
from torch import Tensor

# The quantile primitives moved DOWN to the codec layer 2026-08-14
# (bijou/fast/molmoact2.py — raw-unit glue is codec property, and fast
# sits below encoders in the DAG); re-exported here so prompt-side call
# sites and the port shims keep one import surface.
from ..fast.molmoact2 import (
    QuantileStats,
    normalize_q01q99,
    normalize_state,
    unnormalize_action,
    unnormalize_q01q99,
)
from ..molmo2.processor import ImageCrops, _arange_for_pooling, _pixels_to_patches

__all__ = [
    "ACTION_OUTPUT_ID",
    "BOS_ID",
    "IMAGE_TOKEN_STRINGS",
    "IM_END_ID",
    "IM_PATCH_ID",
    "IM_START_ID",
    "PAD_ID",
    "STATE_TOKEN_0_ID",
    "SUPPORTED_ACTION_MODES",
    "PackedActionExample",
    "QuantileStats",
    "build_robot_prompt",
    "discrete_state_string",
    "encode_action_prompt",
    "encoder_attention_mask",
    "image_token_ids_resize",
    "infer_max_sequence_length",
    "load_norm_stats",
    "normalize_q01q99",
    "normalize_task_text",
    "pack_action_example",
    "process_image_resize",
    "require_single_obs",
    "to_uint8_rgb",
    "unnormalize_action",
    "unnormalize_q01q99",
    "validate_inference_config",
]

# Special-token ids, pinned from the MolmoAct2 checkpoint tokenizer
# (tokenizer.json of the released SO-100/101 checkpoints and the rig-ft
# exports; verified against the loaded tokenizer in the oracles). The
# extra action/state vocab sits BELOW the image tokens — see module
# docstring, these are not molmo2's image ids.
SETUP_START_ID = 151_669
SETUP_END_ID = 151_670
CONTROL_START_ID = 151_671
CONTROL_END_ID = 151_672
STATE_START_ID = 151_673
STATE_END_ID = 151_674
STATE_TOKEN_0_ID = 151_675  # <state_0>; <state_N> = STATE_TOKEN_0_ID + N
ACTION_OUTPUT_ID = 151_931
ACTION_START_ID = 151_932
ACTION_END_ID = 151_933
IM_START_ID = 155_648
IM_END_ID = 155_649
IM_PATCH_ID = 155_650
IM_COL_ID = 155_651
LOW_RES_IM_START_ID = 155_652

#: ``tokenizer_config.json`` sets bos_token == eos_token == <|im_end|>;
#: their processor prepends it to every prompt (``insert_bos``).
BOS_ID = 151_645
PAD_ID = 151_643  # <|endoftext|>

# Their processor's token/placeholder strings (the prompt is built as a
# string and tokenized whole; every name below is a single token).
IMAGE_PLACEHOLDER = "<|image|>"
ACTION_OUTPUT_TOKEN = "<action_output>"
STATE_START_TOKEN = "<state_start>"
STATE_END_TOKEN = "<state_end>"
STATE_TOKEN_PREFIX = "<state_"
SETUP_START_TOKEN = "<setup_start>"
SETUP_END_TOKEN = "<setup_end>"
CONTROL_START_TOKEN = "<control_start>"
CONTROL_END_TOKEN = "<control_end>"

# The shipped image operating point (processor_config.json +
# preprocessor defaults): 378x378 single view, patch 14, 2x2 pooling.
_BASE_INPUT_SIZE = 378
_PATCH_SIZE = 14
_VIEW_PATCHES = _BASE_INPUT_SIZE // _PATCH_SIZE  # 27 patches per dim
_POOLED_DIM = -(-_VIEW_PATCHES // 2)  # 14 pooled tokens per dim
RESIZE_GRID = (_POOLED_DIM, _POOLED_DIM, 0, 0)


# Sequence-budget constants (their configuration_molmoact2.py); the
# inferred cap is a loud guard, not a truncation.
_TOKENS_PER_IMAGE = _POOLED_DIM * _POOLED_DIM  # 196
_FIXED_PROMPT_TOKEN_BUDGET = 80
_TASK_TOKEN_BUDGET = 32
_SEQUENCE_LENGTH_MARGIN = 32
_SEQUENCE_LENGTH_MULTIPLE = 64
_DEFAULT_NUM_IMAGES = 2

# Task-text normalization (their _normalize_question_text): strip
# instruction-prefix boilerplate, surrounding quotes/brackets, trailing
# punctuation; join multi-sentence tasks with '; '; lowercase. The
# curly quotes/ellipsis are escaped only to satisfy RUF001 — the byte
# content is exactly the reference's literals.
_TASK_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,\u2026"
_TASK_TRAILING_CLOSERS = "\"'\u201d\u2019)]}"
_TASK_SURROUNDING_DELIMITERS = "\"'`\u201c\u201d\u2018\u2019[](){}"
_TASK_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def load_norm_stats(
    checkpoint_dir: str | Path,
    tag: str,
) -> tuple[QuantileStats, QuantileStats, dict[str, Any]]:
    """Read ``norm_stats.json`` and return (action, state, metadata) for
    one tag. Metadata carries the prompt facts the pack step consumes
    (setup_type, control_mode, camera_keys, action_horizon); the two
    prompt-load-bearing strings are validated non-empty here — the
    template renders them verbatim, so an empty value would silently
    build an off-distribution prompt ("The setup is .")."""
    stats_path = Path(checkpoint_dir).expanduser() / "norm_stats.json"
    payload = json.loads(stats_path.read_text())
    metadata_by_tag = payload.get("metadata_by_tag")
    if not isinstance(metadata_by_tag, dict) or tag not in metadata_by_tag:
        available = sorted(metadata_by_tag) if isinstance(metadata_by_tag, dict) else []
        raise ValueError(f"norm_tag {tag!r} not in {stats_path} (tags: {available})")
    metadata = metadata_by_tag[tag]
    for key in ("setup_type", "control_mode"):
        value = metadata.get(key)
        if value is None or str(value).strip() == "":
            raise ValueError(
                f"{stats_path}: {tag!r}.{key} is missing/empty — the robot "
                "prompt renders it verbatim; refusing to build degenerate "
                "prompts",
            )

    def quantiles(key: str) -> QuantileStats:
        stats = metadata.get(key)
        if not isinstance(stats, dict) or "q01" not in stats or "q99" not in stats:
            raise ValueError(f"{stats_path}: {tag!r}.{key} has no q01/q99 rows")
        return QuantileStats(
            q01=torch.tensor(stats["q01"], dtype=torch.float32),
            q99=torch.tensor(stats["q99"], dtype=torch.float32),
        )

    return quantiles("action_stats"), quantiles("state_stats"), metadata


def normalize_task_text(text: str) -> str:
    """Their ``_normalize_question_text``, verbatim: whitespace collapse,
    iterated stripping of delimiters / instruction prefixes / trailing
    punctuation, multi-sentence join with '; ', lowercase."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_TASK_SURROUNDING_DELIMITERS).strip()
        for pattern in _TASK_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_TASK_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_TASK_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_TASK_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    chunks = [
        chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()
    ]
    if len(chunks) > 1:
        normalized = "; ".join(chunks)
    return normalized.lower()


def discrete_state_string(
    state: np.ndarray | Tensor,
    *,
    num_state_tokens: int = 256,
) -> str:
    """The normalized state as 256-bin prompt tokens: nan->0 / +-inf->+-1,
    clip to [-1, 1], scale to [0, N-1], round half-to-even (their
    ``np.rint``), emit ``<state_start><state_i>...<state_end>``.

    Shapes:
    - ``state``: [D] (any layout flattens row-major)
    - returns: str with D ``<state_i>`` tokens
    """
    if num_state_tokens <= 0:
        raise ValueError(f"num_state_tokens must be > 0, got {num_state_tokens}")
    arr = np.asarray(
        state.detach().cpu().numpy() if isinstance(state, Tensor) else state,
        dtype=np.float32,
    )
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    token_ids = np.clip(
        np.rint(scaled).astype(np.int64),
        0,
        int(num_state_tokens) - 1,
    ).reshape(-1)
    tokens = "".join(f"{STATE_TOKEN_PREFIX}{int(token_id)}>" for token_id in token_ids)
    return f"{STATE_START_TOKEN}{tokens}{STATE_END_TOKEN}"


def _wrap(text: str, start: str, end: str, *, add_tokens: bool) -> str:
    text = str(text or "")
    if text.startswith(start) and text.endswith(end):
        return text
    if not text or not add_tokens:
        return text
    return f"{start}{text}{end}"


def build_robot_prompt(
    *,
    task: str,
    discrete_state: str,
    setup_type: str,
    control_mode: str,
    num_images: int,
    add_setup_tokens: bool = True,
    add_control_tokens: bool = True,
) -> str:
    """Their ``_build_robot_text``: the fixed robot QA template with the
    (already normalized) task, the discrete-state clause, wrapped
    setup/control descriptors, an ``Image N<|image|>`` prefix per camera
    (bare ``<|image|>`` for one), chat markers, and the trailing
    ``<action_output>`` the expert reads its conditioning from."""
    setup_text = _wrap(
        setup_type,
        SETUP_START_TOKEN,
        SETUP_END_TOKEN,
        add_tokens=add_setup_tokens,
    )
    control_text = _wrap(
        control_mode,
        CONTROL_START_TOKEN,
        CONTROL_END_TOKEN,
        add_tokens=add_control_tokens,
    )
    state_clause = (
        f" The current state of the robot is {discrete_state}."
        if discrete_state
        else ""
    )
    prompt = (
        f"The task is to {task}. The setup is {setup_text}.{state_clause} "
        f"The expected control mode is {control_text}. "
        f"Given these, what action should the robot take to complete the task?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = IMAGE_PLACEHOLDER
    else:
        image_prefix = "".join(
            f"Image {idx + 1}{IMAGE_PLACEHOLDER}" for idx in range(num_images)
        )
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"


def to_uint8_rgb(value: np.ndarray | Tensor) -> np.ndarray:
    """Their pack step's image coercion (``_normalize_image``): any
    reasonable frame layout -> HWC uint8 RGB. Floats with max <= 1 are
    scaled by 255; everything is clipped THEN cast (astype = truncation,
    their op) — this uint8 quantization precedes the resize and is part
    of the reference distribution (see module docstring).

    Shapes:
    - ``value``: [H, W], [C, H, W], [H, W, C] or [1, ...] thereof
      (C in {1, 3, 4}); float 0-1, float 0-255, or integer
    - returns: [H, W, 3] uint8
    """
    arr = (
        value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
    )
    while arr.ndim > 3 and int(arr.shape[0]) == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        raise ValueError(f"unsupported image shape for MolmoAct2: {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype in (np.float16, np.float32, np.float64):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def process_image_resize(image: np.ndarray) -> ImageCrops:
    """One uint8 HWC frame -> the resize-mode ViT inputs: a single
    378x378 view (their uint8 resize: bilinear WITHOUT antialias on the
    uint8 tensor, clipped and rounded back to uint8, then /255 and
    ``x * 2 - 1``), grid ``(14, 14, 0, 0)``, and the 196-row 2x2 pooling
    index. Mirrors ``image_to_patches_and_grids(crop_mode='resize')``.

    Shapes:
    - ``image``: [H, W, 3] uint8 (``to_uint8_rgb`` output)
    - returns: ImageCrops(crops [1, 729, 588], pooled_idx [196, 4],
      grid (14, 14, 0, 0))
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"expected an HWC uint8 RGB frame (to_uint8_rgb), got {image.dtype} {image.shape}",
        )
    chw = torch.permute(torch.from_numpy(image), [2, 0, 1])
    resized = torchvision.transforms.Resize(
        [_BASE_INPUT_SIZE, _BASE_INPUT_SIZE],
        torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=False,
    )(chw)
    resized = torch.clip(resized, 0, 255).to(torch.uint8)
    view = resized.to(torch.float32) / 255.0
    view = view.permute(1, 2, 0) * 2.0 - 1.0  # HWC, mean/std 0.5 shortcut

    resize_idx = np.arange(_VIEW_PATCHES * _VIEW_PATCHES).reshape(
        _VIEW_PATCHES,
        _VIEW_PATCHES,
    )
    pooled_idx = _arange_for_pooling(resize_idx).reshape(-1, 4)
    return ImageCrops(
        grid=RESIZE_GRID,
        crops=_pixels_to_patches(view[None]),
        pooled_idx=torch.from_numpy(pooled_idx),
    )


def image_token_ids_resize() -> list[int]:
    """The token expansion of one ``<|image|>`` in resize mode: the
    single-view branch of their ``get_image_tokens`` with the shipped
    options (no col tokens, plain ``<im_start>``) — 198 ids."""
    return [IM_START_ID, *([IM_PATCH_ID] * _TOKENS_PER_IMAGE), IM_END_ID]


def infer_max_sequence_length(
    *,
    num_images: int,
    state_dim: int,
    include_discrete_action: bool = False,
    action_dim: int = 1,
    action_horizon: int = 1,
) -> int:
    """Their fixed sequence budget (``infer_molmoact2_max_sequence_length``),
    continuous rig scope: discrete-action terms kept for fidelity but off
    by default."""
    if num_images < 1:
        num_images = _DEFAULT_NUM_IMAGES
    state_dim = max(state_dim, 0)
    action_dim = max(action_dim, 1)
    action_horizon = max(action_horizon, 1)
    total = (
        num_images * _TOKENS_PER_IMAGE
        + _FIXED_PROMPT_TOKEN_BUDGET
        + _TASK_TOKEN_BUDGET
        + state_dim
        + _SEQUENCE_LENGTH_MARGIN
    )
    if include_discrete_action:
        per_step = max(6, math.ceil(action_dim * 0.95))
        total += 4 + action_horizon * per_step
    return -(-total // _SEQUENCE_LENGTH_MULTIPLE) * _SEQUENCE_LENGTH_MULTIPLE


def encode_action_prompt(prompt: str, tokenizer: Any) -> list[int]:
    """Tokenize a built prompt: expand each ``<|image|>`` to the resize-
    mode token string (their in-text replacement, one placeholder at a
    time), BPE the whole string, then their ``insert_bos`` — prepend
    ``<|im_end|>`` unless already first. ``tokenizer`` is any
    ``TextTokenizer``-protocol object over the MolmoAct2 checkpoint's
    tokenizer.json (``bijou.molmo2.tokenizer.Molmo2TextTokenizer``)."""
    image_string = "<im_start>" + "<im_patch>" * _TOKENS_PER_IMAGE + "<im_end>"
    while IMAGE_PLACEHOLDER in prompt:
        prompt = prompt.replace(IMAGE_PLACEHOLDER, image_string, 1)
    ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    if not ids or ids[0] != BOS_ID:
        ids = [BOS_ID, *ids]
    return ids


@dataclass(frozen=True, slots=True)
class PackedActionExample:
    """One inference example, packed: the prompt ids and the per-camera
    resize-mode image inputs (grids are all ``RESIZE_GRID``)."""

    input_ids: Tensor  # [S] long, BOS first
    images: tuple[ImageCrops, ...]
    normalized_state: Tensor  # [D] float32, post-clamp (the prompt state)


def pack_action_example(
    *,
    images: list[np.ndarray | Tensor],
    state: Tensor,
    task: str,
    tokenizer: Any,
    state_stats: QuantileStats,
    setup_type: str,
    control_mode: str,
    normalize_language: bool = True,
    num_state_tokens: int = 256,
    max_sequence_length: int | None = None,
) -> PackedActionExample:
    """The full input-side path for one rig observation, in their order:
    normalize+clamp state -> discrete state string -> task normalization
    -> prompt -> uint8 image coercion + resize-mode processing -> token
    expansion + BOS -> loud sequence-budget guard.

    Shapes:
    - ``images``: per camera [H, W, 3]-coercible frames (``to_uint8_rgb``)
    - ``state``: [D] raw joint units
    - returns: PackedActionExample(input_ids [S], images tuple of
      ImageCrops, normalized_state [D])
    """
    if not images:
        raise ValueError("MolmoAct2 requires at least one camera frame")
    state = torch.as_tensor(state, dtype=torch.float32)
    if state.ndim != 1:
        raise ValueError(f"expected a single [D] state row, got {tuple(state.shape)}")
    normalized_state = normalize_state(state, state_stats)
    prompt = build_robot_prompt(
        task=normalize_task_text(task) if normalize_language else task,
        discrete_state=discrete_state_string(
            normalized_state,
            num_state_tokens=num_state_tokens,
        ),
        setup_type=setup_type,
        control_mode=control_mode,
        num_images=len(images),
    )
    input_ids = encode_action_prompt(prompt, tokenizer)
    cap = (
        int(max_sequence_length)
        if max_sequence_length is not None
        else infer_max_sequence_length(
            num_images=len(images),
            state_dim=int(state.shape[0]),
        )
    )
    if len(input_ids) > cap:
        raise ValueError(
            f"sequence length {len(input_ids)} exceeds max_sequence_length={cap}",
        )
    return PackedActionExample(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        images=tuple(process_image_resize(to_uint8_rgb(image)) for image in images),
        normalized_state=normalized_state,
    )


#: Their processor's ``IMAGE_TOKENS`` membership strings (the
#: ``token_type_ids`` set), resolved per checkpoint through its
#: tokenizer — ids cannot be pinned (their releases re-home the image
#: specials; see encoders/molmoact2.py).
IMAGE_TOKEN_STRINGS = (
    "<im_patch>",
    "<im_col>",
    "<im_start>",
    "<low_res_im_start>",
    "<frame_start>",
    "<im_end>",
    "<frame_end>",
    "<im_low>",
)

#: Action-path scope this stack implements: checkpoints whose
#: ``action_mode`` is ``'continuous'`` (rig-ft exports) or ``'both'``
#: (the released SO-100/101 — under 'both' the encoder mask
#: additionally strips EOS positions and discrete action spans, see
#: :func:`encoder_attention_mask`). Discrete-only checkpoints and the
#: depth gate stay out of scope.
SUPPORTED_ACTION_MODES = ("continuous", "both")


def validate_inference_config(config: Mapping[str, object]) -> None:
    """Loud guard over a SOURCE checkpoint's top-level ``config.json``:
    raise on any action-path feature this stack does not implement
    (moved from the port's wiring at its retirement, phase 5 — the
    converter's P-guard)."""
    if not config.get("add_action_expert", False):
        raise ValueError("checkpoint has no action expert (add_action_expert falsy)")
    if config.get("action_expert_depth_gate", False):
        raise NotImplementedError(
            "action_expert_depth_gate=true is not wired (off in the released "
            "SO-100/101 and rig-ft checkpoints)",
        )
    mode = config.get("action_mode", "continuous")
    if mode not in SUPPORTED_ACTION_MODES:
        raise NotImplementedError(
            f"action_mode={mode!r} is not wired (continuous-path scope: "
            f"{SUPPORTED_ACTION_MODES})",
        )


def require_single_obs(config: dict[str, Any]) -> int:
    """Guard a checkpoint config's ``n_obs_steps``: this stack packs
    exactly ONE observation per prompt, so only 1 is loadable.

    Refuses a MISSING key too, loudly: their HF config class defaults to
    30 while training used 1 — under their reference a missing key
    silently shifts chunk slicing to start at index 29, and silently
    picking either side of that divergence is worse than stopping."""
    value = config.get("n_obs_steps")
    if value is None or int(value) != 1:
        raise NotImplementedError(
            f"n_obs_steps={value!r}: this stack packs exactly one observation "
            "per prompt (all released/rig checkpoints ship 1). A missing key "
            "is refused rather than defaulted — their HF config class "
            "defaults to 30, which shifts the chunk slice to index 29",
        )
    return 1


def _mask_discrete_output_span(
    row_ids: Tensor,
    row_mask: Tensor,
    start_id: int | None,
    end_id: int | None,
) -> None:
    """Their ``_mask_discrete_output_span`` verbatim: each ``start``
    pairs with the next ``end`` at-or-after it (inclusive); an unmatched
    start masks through the end of the row.

    Shapes:
    - ``row_ids``: [S] one row's token ids
    - ``row_mask``: [S] bool, mutated in place
    """
    if start_id is None or end_id is None:
        return
    start_positions = (row_ids == start_id).nonzero(as_tuple=False).flatten().tolist()
    if not start_positions:
        return
    end_positions = (row_ids == end_id).nonzero(as_tuple=False).flatten().tolist()
    end_ptr = 0
    for start_pos in start_positions:
        while end_ptr < len(end_positions) and end_positions[end_ptr] < start_pos:
            end_ptr += 1
        if end_ptr >= len(end_positions):
            row_mask[start_pos:] = False
            break
        end_pos = end_positions[end_ptr]
        row_mask[start_pos : end_pos + 1] = False
        end_ptr += 1


def encoder_attention_mask(
    input_ids: Tensor | None,
    attention_mask: Tensor | None,
    *,
    action_mode: str = "continuous",
    eos_token_id: int | None = None,
    action_start_token_id: int | None = None,
    action_end_token_id: int | None = None,
) -> Tensor | None:
    """Bool mask over the prompt for cross-attention. Mirror of their
    ``_get_encoder_attention_mask``: the base mask is the prompt
    ``attention_mask`` (or ``input_ids != -1``); under ``action_mode
    'both'`` every EOS position is additionally excluded — including
    the leading BOS, which IS ``<|im_end|>`` under their convention —
    along with any discrete ``<action_start>..<action_end>`` spans.

    Shapes:
    - ``input_ids``: [B, S] (or None)
    - ``attention_mask``: [B, S], 1 = real token (or None)
    - returns: [B, S] bool (True = attendable), or None if neither input
    """
    if action_mode not in SUPPORTED_ACTION_MODES:
        raise NotImplementedError(
            f"action_mode={action_mode!r} encoder masking is not wired",
        )
    if attention_mask is not None:
        mask = attention_mask.to(dtype=torch.bool).clone()
    elif input_ids is not None:
        mask = input_ids != -1
    else:
        return None
    if action_mode != "both" or input_ids is None:
        return mask
    if eos_token_id is not None:
        mask &= input_ids != int(eos_token_id)
    for batch_idx in range(input_ids.shape[0]):
        _mask_discrete_output_span(
            input_ids[batch_idx],
            mask[batch_idx],
            action_start_token_id,
            action_end_token_id,
        )
    return mask
