"""Bijou: an observation encoder composed with an action decoder.

The composition root: ``BijouModel`` owns one :class:`ObservationEncoder`
(today the truncated Gemma trunk) and one :class:`ActionDecoder` (today
the flow-matching expert), delegating encode / velocity / sampling. The
prefix (chat-templated instruction + camera images) is encoded once per
observation and cached as an :class:`EncodedPrefix`; the decoder then
denoises a chunk of actions against it, with fresh robot state, at ~10
model evaluations per chunk.

``backbone``/``expert`` are the historical names of the two halves and
remain the public accessors (checkpoint files are keyed by them:
``expert.safetensors``, ``backbone.safetensors``).
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn

from .decoders.flow import FlowDecoder, SamplingMethod
from .encoders.gemma4 import GemmaEncoder, GemmaInputs
from .gemma4.model import Gemma4Model
from .interface import CollatedBatch, EncodedPrefix


class BijouModel(nn.Module):
    """One encoder + one decoder (see the module docstring)."""

    def __init__(self, encoder: GemmaEncoder, decoder: FlowDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def backbone(self) -> Gemma4Model:
        return self.encoder.backbone

    @property
    def expert(self) -> FlowDecoder:
        return self.decoder

    def encode_prefix(
        self,
        input_ids: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> EncodedPrefix:
        """Tensor-level prefix encode (shapes in GemmaEncoder.encode_tensors);
        grad-transparent — training wraps it in autocast, eval in no_grad."""
        return self.encoder.encode_tensors(
            input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            padding_mask=padding_mask,
        )

    @torch.no_grad()
    def predict_chunk(
        self,
        batch: CollatedBatch[GemmaInputs],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        num_steps: int = 5,
        method: SamplingMethod = SamplingMethod.HEUN,
    ) -> Tensor:
        """Collated batch → RAW-unit action chunk [B, chunk, action_dim]:
        encode the prefix (no grad) and run the decoder's chunk-space
        inference with the batch's per-sample stats."""
        prefix = self.encoder.encode(batch.encoder_inputs, with_grad=False)
        return self.decoder.predict_chunk(
            prefix,
            batch,
            generator=generator,
            noise=noise,
            num_steps=num_steps,
            method=method,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        prefix: EncodedPrefix,
        state: Tensor,
        *,
        num_steps: int = 5,
        method: SamplingMethod = SamplingMethod.HEUN,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Normalized-unit sampling against an already-encoded prefix (see
        FlowDecoder.sample_actions for the solver contract and shapes)."""
        return self.decoder.sample_actions(
            prefix,
            state,
            num_steps=num_steps,
            method=method,
            noise=noise,
            generator=generator,
        )

    @override
    def forward(
        self,
        prefix: EncodedPrefix,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Velocity of the action chunk at flow time ``time`` (see
        ``bijou.decoders.flow`` for the flow convention); returns
        [B, chunk, action_dim].

        Shapes:
          - prefix.streams[name].key/value: [B, kv_heads, P, head_dim]
          - state: [B, state_dim]
          - noisy_actions: [B, chunk, action_dim]
          - time: [B]
        """
        return self.decoder(prefix, state, noisy_actions, time)
