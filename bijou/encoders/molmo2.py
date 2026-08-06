"""The Molmo2 prompt-side inputs collator (WP3).

Prompt format — namespaced to this trunk (``MOLMO2_PROMPT_FORMAT``), NOT a
bump of the Gemma format: the ChatML/Qwen convention hoists every image to
the very front of the sequence, so the Gemma "sandwich with inline camera
groups" cannot be rendered here. The format-3 semantic content carries
over; the mechanics change:

    <bos> {hoisted images} <|im_start|>user\\n
    {task}[kind1 camera|Image 1][kind2 camera|Image 2]{condition}{task}
    <state> <|im_end|>\\n

- Images hoist per the shipped chat template's exact bytes: a bare
  expansion for one image, ``Image 1<img>Image 2<img>`` labels for
  several. Camera KINDS bind to images by those shipped labels — the
  ``[kind camera|Image i]`` bracket groups replace Gemma's inline tags
  (pipe-delimited, format-3 style; single-camera prompts still say
  "Image 1", which is unambiguous).
- ``<bos>`` is ``<|im_end|>`` (id 151645) — the checkpoint's own quirky
  convention (``tokenizer_config.json``; the reference processor inserts
  it in front of every sequence).
- The soft state token is spliced just inside the user-turn close exactly
  like Gemma's (a pad-id placeholder whose embedding the encoder
  overwrites; attention-mask 1 distinguishes it from real padding).
- LEFT padding, same rationale as the Gemma collator (suffix continuation
  needs the suffix physically adjacent to the prompt — the AR path).

Tokenization is native: the checkpoint's ``tokenizer.json`` through the
``tokenizers`` backend (no remote code, no AutoProcessor — the shipped
processor is pinned to transformers 4.x). Assembling text segments around
the special ids is EXACTLY equivalent to tokenizing the full templated
string, because the specials are added tokens that always split first
(verified in the golden-fixture test against the reference processor).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import torch
from torch import Tensor

from ..gemma4.loading import resolve_checkpoint_dir
from ..interface import PromptInputs
from ..molmo2.processor import (
    IMAGE_TYPE_IDS,
    ImageCrops,
    image_token_ids,
    process_image,
)

MOLMO2_PROMPT_FORMAT = 1

# ChatML text-side ids (pinned; verified against the loaded tokenizer).
BOS_ID = 151_645  # <|im_end|> — the shipped bos convention
IM_START_TEXT_ID = 151_644  # <|im_start|>
IM_END_TEXT_ID = 151_645  # <|im_end|>
PAD_ID = 151_643  # <|endoftext|>
_PINNED_SPECIALS = {
    "<|im_start|>": IM_START_TEXT_ID,
    "<|im_end|>": IM_END_TEXT_ID,
    "<|endoftext|>": PAD_ID,
    "<im_start>": 151_936,
    "<im_end>": 151_937,
    "<im_patch>": 151_938,
    "<im_col>": 151_939,
    "<low_res_im_start>": 151_940,
}


def camera_tag_text(kind: str, image_index: int) -> str:
    """The per-camera bracket group binding a semantic kind to a hoisted
    image by its shipped label, e.g. ``[wrist camera|Image 1]`` (1-based —
    the chat template's own numbering). The exact bytes are a trained
    contract — change only with a MOLMO2_PROMPT_FORMAT bump."""
    return f"[{kind} camera|Image {image_index}]"


def hoist_text(num_images: int) -> str:
    """The image hoist in the shipped chat template's exact bytes: one
    bare placeholder, or ``Image {i}<|image|>`` per image for several.
    Used by the golden-banking script; the collator assembles the same
    layout directly in id space."""
    if num_images == 1:
        return "<|image|>"
    return "".join(f"Image {i + 1}<|image|>" for i in range(num_images))


def user_turn_text(sample: PromptInputs) -> str:
    """The user turn's text content for one sample: the format-3 sandwich
    minus inline images — instruction, camera bindings, the pre-rendered
    condition/[generate|…] block, instruction again. The soft state token
    is spliced in id space, not here."""
    tags = "".join(
        camera_tag_text(camera.kind, i + 1) for i, camera in enumerate(sample.cameras)
    )
    return f"{sample.instruction}{tags}{sample.condition_text}{sample.instruction}"


@dataclass(frozen=True, slots=True)
class Molmo2Inputs:
    """The Molmo2-specific half of a collated batch.

    ``image_type_mask`` marks the image special-token positions (patch,
    col, start/end markers) — attention between two marked positions is
    bidirectional, everything else causal (the reference's
    ``token_type_ids`` mask). ``crops``/``pooled_patches_idx`` are the
    vision-backbone inputs (``Molmo2VisionBackbone.forward``), padded
    per-sample with -1 exactly like the reference's batching; the pooled
    rows are ordered image-by-image, global view first, matching the
    ``input_ids == IM_PATCH_ID`` injection positions.

    Shapes (M = max crops+views per sample, P = max pooled tokens):
      - input_ids: [B, S]  (left-padded)
      - attention_mask: [B, S]  (1 = real token, 0 = left padding)
      - image_type_mask: [B, S]  (bool)
      - crops: [B, M, patches, patch_dim]  (float32, -1 padded)
      - pooled_patches_idx: [B, P, pool_group]  (long, -1 padded)
      - state: [B, state_dim]  (normalized)
    """

    input_ids: Tensor
    attention_mask: Tensor
    image_type_mask: Tensor
    crops: Tensor
    pooled_patches_idx: Tensor
    state: Tensor
    state_slot: int
    has_padding: bool

    def tensors(self) -> dict[str, Tensor]:
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if isinstance(value := getattr(self, field.name), Tensor)
        }

    def pin_memory(self) -> Molmo2Inputs:
        return dataclasses.replace(
            self,
            **{name: t.pin_memory() for name, t in self.tensors().items()},
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Molmo2Inputs:
        return dataclasses.replace(
            self,
            **{
                name: t.to(device, non_blocking=non_blocking)
                for name, t in self.tensors().items()
            },
        )


class Molmo2InputsCollator:
    """InputsCollator for the Molmo2 trunk (module docstring has the
    format). ``max_crops`` is the ONE image knob: 1 = the port plan's
    operating point (global view + a single full-image crop, the smallest
    layout the shipped distribution contains)."""

    def __init__(self, checkpoint: str, max_crops: int) -> None:
        if max_crops < 1:
            raise ValueError(f"max_crops must be >= 1, got {max_crops}")
        self.checkpoint = checkpoint
        self.max_crops = max_crops
        self._tokenizer: Any = None
        self._newline_id: int | None = None

    @override
    def __getstate__(self) -> dict[str, Any]:
        # Rebuilt lazily in spawned dataloader workers.
        return {**self.__dict__, "_tokenizer": None}

    def _build_tokenizer(self) -> None:
        from tokenizers import Tokenizer

        checkpoint_dir = resolve_checkpoint_dir(self.checkpoint)
        tokenizer_file = Path(checkpoint_dir) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise SystemExit(f"no tokenizer.json in {checkpoint_dir}")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        # The pinned ids are a load-bearing contract (the token layout and
        # the FAST block base both anchor on them) — verify, never assume.
        for token, expected in _PINNED_SPECIALS.items():
            actual = tokenizer.token_to_id(token)
            if actual != expected:
                raise SystemExit(
                    f"tokenizer maps {token!r} to {actual}, expected "
                    f"{expected} — not a Molmo2-4B tokenizer; the prompt "
                    "layout contract does not hold",
                )
        newline = tokenizer.encode("\n", add_special_tokens=False).ids
        if len(newline) != 1:
            raise SystemExit(
                f"'\\n' tokenizes to {newline} — the turn close is no "
                "longer a fixed two-token tail; re-verify the state splice",
            )
        self._newline_id = newline[0]
        self._tokenizer = tokenizer

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def __call__(self, samples: list[PromptInputs]) -> Molmo2Inputs:
        if self._tokenizer is None:
            self._build_tokenizer()
        assert self._newline_id is not None  # _build_tokenizer set it

        sequences: list[list[int]] = []
        crops_per_sample: list[list[Tensor]] = []
        pooled_per_sample: list[list[Tensor]] = []
        for sample in samples:
            if not sample.cameras:
                raise ValueError("Molmo2 prompts require at least one camera")
            images: list[ImageCrops] = [
                process_image(camera.image, max_crops=self.max_crops)
                for camera in sample.cameras
            ]
            ids: list[int] = [BOS_ID]
            crop_base = 0
            pooled: list[Tensor] = []
            for i, image in enumerate(images):
                if len(images) > 1:
                    ids.extend(self._encode(f"Image {i + 1}"))
                ids.extend(image_token_ids(image.grid))
                # Patch indices are per-image; shift into the sample's
                # concatenated (view, patch) grid, preserving -1 markers.
                idx = image.pooled_idx
                pooled.append(torch.where(idx >= 0, idx + crop_base, idx))
                crop_base += image.crops.shape[0] * image.crops.shape[1]
            ids.append(IM_START_TEXT_ID)
            ids.extend(self._encode("user\n" + user_turn_text(sample)))
            # The soft state token, just inside the turn close (pad id —
            # its embedding is overwritten by the encoder's state
            # projection; attention-mask 1, unlike actual padding).
            ids.extend((PAD_ID, IM_END_TEXT_ID, self._newline_id))
            sequences.append(ids)
            crops_per_sample.append([image.crops for image in images])
            pooled_per_sample.append(pooled)

        batch_size = len(samples)
        width = max(len(ids) for ids in sequences)
        input_ids = torch.full((batch_size, width), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        for row, ids in enumerate(sequences):
            input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, width - len(ids) :] = 1

        image_type_ids = torch.tensor(sorted(IMAGE_TYPE_IDS), dtype=torch.long)
        image_type_mask = torch.isin(input_ids, image_type_ids)
        # Left padding uses a REAL text id (pad); only real tokens may
        # count as image positions.
        image_type_mask &= attention_mask.bool()

        sample_crops = [torch.cat(crops, dim=0) for crops in crops_per_sample]
        sample_pooled = [torch.cat(pooled, dim=0) for pooled in pooled_per_sample]
        max_views = max(c.shape[0] for c in sample_crops)
        max_pooled = max(p.shape[0] for p in sample_pooled)
        patches, patch_dim = sample_crops[0].shape[1:]
        pool_group = sample_pooled[0].shape[1]
        crops = torch.full(
            (batch_size, max_views, patches, patch_dim),
            -1.0,
            dtype=torch.float32,
        )
        pooled_patches_idx = torch.full(
            (batch_size, max_pooled, pool_group),
            -1,
            dtype=torch.long,
        )
        for row in range(batch_size):
            crops[row, : sample_crops[row].shape[0]] = sample_crops[row]
            pooled_patches_idx[row, : sample_pooled[row].shape[0]] = sample_pooled[row]

        return Molmo2Inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_type_mask=image_type_mask,
            crops=crops,
            pooled_patches_idx=pooled_patches_idx,
            state=torch.stack([sample.state for sample in samples]),
            state_slot=-3,  # just inside the (<|im_end|>, \n) close
            has_padding=bool((attention_mask == 0).any()),
        )
