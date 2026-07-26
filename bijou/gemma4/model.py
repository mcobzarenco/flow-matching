"""Top-level Gemma 4 model: text decoder + vision tower + LM head.

Mirrors HF's ``Gemma4ForConditionalGeneration`` for text-only and text+image
inputs. The audio tower is intentionally not implemented (not needed for the
VLA); audio inputs are rejected at the loader level by simply not existing.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import override

import torch
from torch import Tensor, nn

from .cache import KVCache
from .config import Gemma4Config
from .layers import DEFAULT_ATTENTION_BACKEND, AttentionBackend, DeviceLike
from .text import TextAttention, TextModel
from .vision import MultimodalEmbedder, VisionAttention, VisionModel


@dataclass(frozen=True, slots=True)
class Gemma4Output:
    """Result of a forward pass. ``logits`` are softcapped, in model dtype."""

    logits: Tensor
    last_hidden_state: Tensor


class Gemma4Model(nn.Module):
    """``device``/``dtype`` are forwarded to every submodule (torch factory
    convention), so parameters are created directly on the target device.
    ``dtype`` defaults to the dtype declared in the checkpoint config;
    submodules can be instantiated individually with different settings if a
    future config splits dtypes per component. ``attn_backend`` is the
    runtime attention implementation (see ``layers.AttentionBackend``)."""

    def __init__(
        self,
        config: Gemma4Config,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = config.dtype
        self.config = config
        self.language_model = TextModel(
            config.text,
            attn_backend=attn_backend,
            device=device,
            dtype=dtype,
        )
        self.vision_tower = (
            VisionModel(
                config.vision,
                attn_backend=attn_backend,
                device=device,
                dtype=dtype,
            )
            if config.vision is not None
            else None
        )
        self.embed_vision = (
            MultimodalEmbedder(
                multimodal_hidden_size=config.vision.hidden_size,
                text_hidden_size=config.text.hidden_size,
                eps=config.vision.rms_norm_eps,
                device=device,
                dtype=dtype,
            )
            if config.vision is not None
            else None
        )
        self.lm_head = nn.Linear(
            config.text.hidden_size,
            config.text.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def get_image_features(
        self,
        pixel_values: Tensor,
        image_position_ids: Tensor,
    ) -> Tensor:
        """Soft tokens projected into LM space: [num_soft_tokens, hidden]."""
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError("model was built without a vision tower")
        soft_tokens = self.vision_tower(pixel_values, image_position_ids)
        return self.embed_vision(soft_tokens)

    @override
    def forward(
        self,
        input_ids: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: KVCache | None = None,
        logits_to_keep: int = 0,
    ) -> Gemma4Output:
        """``input_ids`` [B, S]; image placeholder positions (id
        ``config.image_token_id``) are replaced by vision soft tokens when
        ``pixel_values``/``image_position_ids`` are given.
        """
        text_config = self.config.text
        inputs_embeds, per_layer_inputs = self.embed_multimodal(
            input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )

        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            padding_mask=padding_mask,
            cache=cache,
        )

        if logits_to_keep:
            hidden_for_logits = hidden_states[:, -logits_to_keep:, :]
        else:
            hidden_for_logits = hidden_states
        logits = self.lm_head(hidden_for_logits)
        if (softcap := text_config.final_logit_softcapping) is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap

        return Gemma4Output(logits=logits, last_hidden_state=hidden_states)

    def embed_multimodal(
        self,
        input_ids: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Assemble decoder inputs: (inputs_embeds, per-layer PLE inputs).

        Embeds ``input_ids`` (image placeholders replaced by the pad token,
        then overwritten by vision soft tokens) and returns the raw
        token-identity PLE component. This is the full multimodal front-end
        of the model without running the decoder — used by e.g. the Bijou
        prefix encoder.
        """
        image_mask = input_ids == self.config.image_token_id

        # Multimodal placeholder ids are out of the embedding's vocabulary:
        # embed the pad token there instead, then scatter the image features.
        llm_input_ids = torch.where(
            image_mask,
            self.config.text.pad_token_id,
            input_ids,
        )
        inputs_embeds = self.language_model.embed_tokens(llm_input_ids)
        per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids)

        if pixel_values is not None:
            if image_position_ids is None:
                raise ValueError("image_position_ids is required with pixel_values")
            image_features = self.get_image_features(pixel_values, image_position_ids)
            image_features = image_features.to(inputs_embeds.dtype)
            n_slots = int(image_mask.sum())
            if n_slots * inputs_embeds.shape[-1] != image_features.numel():
                raise ValueError(
                    f"image token slots ({n_slots}) do not match soft tokens "
                    f"({image_features.shape[0]})",
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_features,
            )
        elif bool(image_mask.any()):
            raise ValueError("input contains image tokens but no pixel_values given")

        return inputs_embeds, per_layer_inputs


def set_attention_backend(module: nn.Module, backend: AttentionBackend) -> None:
    """Switch the attention implementation of all attention submodules in
    place (cheap; useful for A/B benchmarking a loaded model)."""
    for submodule in module.modules():
        if isinstance(submodule, (TextAttention, VisionAttention)):
            submodule.attn_backend = backend
