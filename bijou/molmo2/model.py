"""Full-model Molmo2 assembly (WP4 slice): vision injection + multimodal
mask + compose + a cache-free greedy decode.

Faithful to the checkpoint's ``modeling_molmo2.py``, conventions read
from the raw file:

- Injection: pooled/projected image features are **added** (``+=``) into
  the input-embedding sequence at ``input_ids == image_patch_id``
  positions — single injection at layer 0, count asserted against the
  vision backbone's valid-token output exactly like the reference
  (``build_input_embeddings``). Only ``<im_patch>`` positions receive
  features; the other image specials (col/start/end markers) keep their
  extension-matrix embeddings.
- Mask: the reference composes ``create_causal_mask`` with the
  ``token_type_ids`` function as an **OR** — a position pair is allowed
  iff ``kv <= q`` (causal) OR both positions are image-typed — then key
  padding is excluded. Any two image-typed positions in a sequence are
  mutually visible (the function does not segment per image block).
- Greedy decode: ``greedy_generate`` is cache-free — it recomputes the
  full forward per emitted token. Probe/parity-scale tool, not a serving
  path; newly emitted tokens are text-typed (causal) by construction.
  (The cached paths live elsewhere: ``bijou.decoders.ar_molmo2`` and the
  MolmoAct2 predictor drive ``Molmo2KVCache`` directly; the FLOW path
  stays cache-free per design decision D1, taps only.)
"""

from __future__ import annotations

from pathlib import Path
from typing import override

import torch
from torch import Tensor, nn

from ..gemma4.loading import resolve_checkpoint_dir
from ..nn import MaskSpec
from .config import Molmo2Config
from .loading import load_config, load_text_model, load_vision_backbone
from .text import Molmo2TextModel
from .vision import Molmo2VisionBackbone


def build_multimodal_mask(
    *,
    image_type_mask: Tensor,  # [B, S] bool, True = image-typed position
    padding_mask: Tensor | None,  # [B, S], True/1 = real token
    dtype: torch.dtype,
    device: torch.device,
) -> MaskSpec:  # .tensor: [B, 1, S, S] additive (0 / dtype-min)
    """Causal-or-image-block attention mask for one no-cache forward.

    ``image_type_mask``: [B, S] bool, True at image-typed positions (the
    collator's mirror of the reference ``token_type_ids == 1``; already
    excludes padding). ``padding_mask``: [B, S], True/1 = real token.
    The pattern is never lower-triangular, so ``is_causal`` is False and
    every backend consumes the additive tensor.
    """
    batch, seq_len = image_type_mask.shape
    q_idx = torch.arange(seq_len, device=device)[None, None, :, None]
    kv_idx = torch.arange(seq_len, device=device)[None, None, None, :]
    image = image_type_mask.to(device=device, dtype=torch.bool)  # [B, S]
    image_block = image[:, None, :, None] & image[:, None, None, :]
    allowed = (kv_idx <= q_idx) | image_block
    if padding_mask is not None:
        cols = padding_mask.to(device=device, dtype=torch.bool)
        allowed = allowed & cols[:, None, None, :]
    allowed = allowed.expand(batch, 1, seq_len, seq_len)
    return MaskSpec(
        tensor=torch.where(
            allowed,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.finfo(dtype).min,
        ),
        is_causal=False,
    )


def ensure_per_sample_patch_alignment(
    input_ids: Tensor,
    pooled_patches_idx: Tensor,
    *,
    image_patch_id: int,
) -> None:
    """Per-sample injection contract: each row's ``image_patch_id`` token
    count must equal its valid pooled rows (rows with any member >= 0).

    ``build_input_embeddings``' device-side guard checks the GLOBAL count
    only (async by design, no host sync) — a per-sample mismatch that
    conserves the total would cross-assign features between batch rows
    there and never fire it. Collators call this CPU-side at batch
    assembly, where the check is free and the failure is loud.

    Shapes:
    - ``input_ids``: [B, S] long
    - ``pooled_patches_idx``: [B, P, G] long, -1 = missing member /
      padding row
    """
    patch_counts = (input_ids == image_patch_id).sum(-1)
    valid_pooled = (pooled_patches_idx >= 0).any(-1).sum(-1)
    if not torch.equal(patch_counts, valid_pooled):
        raise ValueError(
            f"per-sample patch-token counts {patch_counts.tolist()} != "
            f"valid pooled rows {valid_pooled.tolist()} — inputs and "
            "vision grid disagree",
        )


class Molmo2Model(nn.Module):
    """Vision backbone + full text decoder, composed per the reference.

    The forward consumes the ``Molmo2InputsCollator`` batch fields by
    name (the encoder seam stays one level up — this module knows tensors,
    not ``PromptInputs``). Logical positions for left-padded rows are
    derived from ``attention_mask`` exactly like the reference
    (``cumsum - 1``, clamped).
    """

    def __init__(
        self,
        text: Molmo2TextModel,
        vision: Molmo2VisionBackbone,
        *,
        image_patch_id: int,
    ) -> None:
        super().__init__()
        if text.lm_head is None:
            raise ValueError("Molmo2Model needs the full decoder with lm_head")
        self.text = text
        self.vision = vision
        self.image_patch_id = image_patch_id

    def build_input_embeddings(
        self,
        input_ids: Tensor,  # [B, S] long
        *,
        crops: Tensor,  # [B, M, patches, patch_dim] (-1 padded)
        pooled_patches_idx: Tensor,  # [B, P, pool_group] long (-1 padded)
    ) -> Tensor:  # [B, S, hidden]
        """Token embeddings with image features added at patch positions.

        Shapes:
        - ``input_ids``: [B, S] long
        - ``crops``: [B, M, patches, patch_dim] fp32, -1-filled pad views
        - ``pooled_patches_idx``: [B, P, pool_group] long, -1 padded
        - returns: [B, S, hidden]
        """
        embeds = self.text.transformer.wte(input_ids)
        features = self.vision(crops, pooled_patches_idx).to(
            device=embeds.device,
            dtype=embeds.dtype,
        )
        is_patch = (input_ids == self.image_patch_id).view(-1)
        # Device-side guard: int(sum()) forced a host sync on every
        # encode (x chunks/step). The abort survives as a CUDA device
        # assert — message quality traded for the sync; this guard has
        # never fired in any run.
        torch._assert_async(  # pyright: ignore[reportPrivateImportUsage] — public per pytorch docs, stub gap
            is_patch.sum() == features.shape[0],
            "image-patch positions and pooled feature rows disagree — "
            "inputs and vision grid disagree",
        )
        # In-place masked add on the fresh wte output (non-leaf, nothing
        # else aliases it, embedding backward never reads its output
        # value) — the former .clone() copied ~60 MB bf16 per call for
        # no semantic difference; the grads oracle gates bitwise.
        flat = embeds.view(-1, embeds.shape[-1])
        flat[is_patch] += features
        return flat.view_as(embeds)

    @staticmethod
    def logical_positions(attention_mask: Tensor) -> Tensor:
        """Positions of real tokens under left padding.

        Shapes:
        - ``attention_mask``: [B, S], 1 = real token
        - returns: [B, S] long (pad positions clamp to 0)
        """
        return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    @override
    def forward(
        self,
        input_ids: Tensor,  # [B, S] long
        *,
        crops: Tensor,  # [B, M, patches, patch_dim] (-1 padded)
        pooled_patches_idx: Tensor,  # [B, P, pool_group] long (-1 padded)
        image_type_mask: Tensor,  # [B, S] bool, True = image-typed
        attention_mask: Tensor | None = None,  # [B, S], 1 = real; None = no pad
    ) -> Tensor:
        """One full multimodal forward.

        ``attention_mask`` is the collator's field; None means no padding.

        Shapes:
        - ``input_ids``: [B, S] long
        - ``crops``: [B, M, patches, patch_dim] (-1 padded)
        - ``pooled_patches_idx``: [B, P, pool_group] long (-1 padded)
        - ``image_type_mask``: [B, S] bool, True = image-typed
        - ``attention_mask``: [B, S], 1 = real (or None)
        - returns: [B, S, vocab] logits
        """
        embeds = self.build_input_embeddings(
            input_ids,
            crops=crops,
            pooled_patches_idx=pooled_patches_idx,
        )
        position_ids = None
        if attention_mask is not None:
            position_ids = self.logical_positions(attention_mask)
        mask = build_multimodal_mask(
            image_type_mask=image_type_mask,
            padding_mask=attention_mask,
            dtype=embeds.dtype,
            device=embeds.device,
        )
        return self.text(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=mask,
        )

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: Tensor,  # [B, S] long
        *,
        crops: Tensor,  # [B, M, patches, patch_dim] (-1 padded)
        pooled_patches_idx: Tensor,  # [B, P, pool_group] long (-1 padded)
        image_type_mask: Tensor,  # [B, S] bool, True = image-typed
        attention_mask: Tensor | None = None,  # [B, S], 1 = real token
        max_new_tokens: int,
        stop_ids: frozenset[int] = frozenset(),
    ) -> list[list[int]]:
        """Cache-free greedy continuation; returns the NEW ids per row
        (a finished row stops contributing after its stop id).

        Shapes:
        - ``input_ids``: [B, S] long (left-padded)
        - ``crops``: [B, M, patches, patch_dim] (-1 padded)
        - ``pooled_patches_idx``: [B, P, pool_group] long (-1 padded)
        - ``image_type_mask``: [B, S] bool
        - ``attention_mask``: [B, S], 1 = real (or None = no padding)
        - returns: B lists of emitted token ids
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        batch = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        emitted: list[list[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        text_type = torch.zeros((batch, 1), dtype=torch.bool, device=input_ids.device)
        real = torch.ones(
            (batch, 1),
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        for _ in range(max_new_tokens):
            logits = self.forward(
                input_ids,
                crops=crops,
                pooled_patches_idx=pooled_patches_idx,
                image_type_mask=image_type_mask,
                attention_mask=attention_mask,
            )
            next_ids = logits[:, -1].argmax(-1)
            for row in range(batch):
                token = int(next_ids[row])
                if not finished[row]:
                    emitted[row].append(token)
                    finished[row] = token in stop_ids
            if all(finished):
                break
            input_ids = torch.cat([input_ids, next_ids[:, None]], dim=1)
            image_type_mask = torch.cat([image_type_mask, text_type], dim=1)
            attention_mask = torch.cat([attention_mask, real], dim=1)
        return emitted


def load_model(
    model_id_or_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Molmo2Model:
    """Load the full multimodal model (decoder + head + vision) from a
    Molmo2 checkpoint, eval mode, gradients off."""
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    config: Molmo2Config = load_config(checkpoint_dir)
    if config.image_patch_id < 0:
        raise ValueError(f"{checkpoint_dir} config has no image_patch_id")
    text = load_text_model(checkpoint_dir, device=device, dtype=dtype)
    vision = load_vision_backbone(checkpoint_dir, device=device, dtype=dtype)
    model = Molmo2Model(text, vision, image_patch_id=config.image_patch_id)
    model.eval()
    model.requires_grad_(False)
    return model
