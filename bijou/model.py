"""Bijou: an observation encoder composed with an action decoder.

The composition root: ``BijouModel`` owns one :class:`ObservationEncoder`
(today the truncated Gemma trunk) and one :class:`ActionDecoder` (today
the flow-matching expert), delegating encode / velocity / sampling. The
observation (chat-templated instruction + camera images) is encoded once
and cached as an :class:`ObservationMemory`; the decoder then
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

from .decoders.ar_fast import ARFastDecoder
from .decoders.flow import FlowDecoder, SamplingMethod
from .encoders.gemma4 import GemmaEncoder, GemmaInputs
from .gemma4.model import Gemma4Model
from .interface import CollatedBatch, ObservationMemory


class BijouModel(nn.Module):
    """One encoder + one decoder (see the module docstring)."""

    def __init__(
        self,
        encoder: GemmaEncoder,
        decoder: FlowDecoder | ARFastDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def backbone(self) -> Gemma4Model:
        return self.encoder.backbone

    @property
    def expert(self) -> FlowDecoder | ARFastDecoder:
        return self.decoder

    def _flow_decoder(self) -> FlowDecoder:
        if not isinstance(self.decoder, FlowDecoder):
            raise TypeError(
                "this operation integrates the flow velocity field; the "
                f"loaded decoder is {type(self.decoder).__name__}",
            )
        return self.decoder

    def encode_observation(
        self,
        input_ids: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> ObservationMemory:
        """Tensor-level observation encode (shapes in
        GemmaEncoder.encode_tensors);
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
        encode the observation (no grad) and run the decoder's chunk-space
        inference with the batch's per-sample stats. ``num_steps``/
        ``method``/``noise`` are flow solver knobs; an AR decoder decodes
        greedily and ignores them (``noise`` must then be None)."""
        memory = self.encoder.encode(batch.encoder_inputs, with_grad=False)
        if isinstance(self.decoder, FlowDecoder):
            return self.decoder.predict_chunk(
                memory,
                batch,
                generator=generator,
                noise=noise,
                num_steps=num_steps,
                method=method,
            )
        return self.decoder.predict_chunk(
            memory,
            batch,
            generator=generator,
            noise=noise,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        memory: ObservationMemory,
        state: Tensor,
        *,
        num_steps: int = 5,
        method: SamplingMethod = SamplingMethod.HEUN,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Normalized-unit sampling against an already-encoded observation
        (see FlowDecoder.sample_actions for the solver contract and
        shapes); flow decoders only."""
        return self._flow_decoder().sample_actions(
            memory,
            state,
            num_steps=num_steps,
            method=method,
            noise=noise,
            generator=generator,
        )

    @override
    def forward(
        self,
        memory: ObservationMemory,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Velocity of the action chunk at flow time ``time`` (see
        ``bijou.decoders.flow`` for the flow convention); returns
        [B, chunk, action_dim].

        Shapes:
          - memory.streams[name].key/value: [B, kv_heads, P, head_dim]
          - state: [B, state_dim]
          - noisy_actions: [B, chunk, action_dim]
          - time: [B]
        """
        return self._flow_decoder()(memory, state, noisy_actions, time)
