"""Bijou: one trunk, a prompt-side encoder strategy, an action decoder.

The composition root: ``BijouModel`` owns the trunk network ONCE (the
truncated Gemma backbone), plus one prompt-side encoder strategy (the
collation + prefix-encode + unfreeze surface, which receives the trunk
as an argument) and one :class:`ActionDecoder`. The observation
(chat-templated instruction + camera images) is encoded once and cached
as an :class:`ObservationMemory`; the decoder then denoises a chunk of
actions against it, with fresh robot state, at ~10 model evaluations per
chunk.

Trunk ownership lives here — not in the encoder — because one network
can serve several roles: the prefix encoder for the cross-attention
decoders today, and prefix + suffix runner for the decoder-only path.
The root is the one place with every role in scope, so it also owns the
objective dispatch (:meth:`loss`) and the trainable-group routing
(:meth:`param_groups`).

``backbone``/``expert`` are the historical names of the two halves and
remain the public accessors (checkpoint files are keyed by them:
``expert.safetensors``, ``backbone.safetensors``).
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn

from .decoders.ar_fast import ARFastDecoder, ar_fast_loss
from .decoders.flow import FlowDecoder, SamplingMethod, flow_matching_loss
from .encoders.gemma4 import GemmaEncoder, GemmaInputs
from .gemma4.model import Gemma4Model
from .interface import CollatedBatch, ObservationMemory


class BijouModel(nn.Module):
    """One trunk + one encoder strategy + one decoder (see the module
    docstring)."""

    def __init__(
        self,
        trunk: Gemma4Model,
        encoder: GemmaEncoder,
        decoder: FlowDecoder | ARFastDecoder,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.encoder = encoder
        self.decoder = decoder

    @property
    def backbone(self) -> Gemma4Model:
        return self.trunk

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named unfreezable trunk groups ("text", "vision") — the
        component-lr flags route here (see GemmaEncoder.param_groups)."""
        return self.encoder.param_groups(self.trunk)

    def encode(self, inputs: GemmaInputs, *, with_grad: bool) -> ObservationMemory:
        """Encode one collated batch of encoder inputs against the trunk.
        ``with_grad=False`` runs under no_grad (eval/rollout/frozen
        training); True leaves autograd on for live-trunk training."""
        return self.encoder.encode(self.trunk, inputs, with_grad=with_grad)

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

    def loss(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[GemmaInputs],
    ) -> Tensor:
        """Scalar training loss of this model's decoder for one batch
        against its observation memory — the objective dispatch (each
        decoder kind's objective is a module-level function beside it)."""
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                return flow_matching_loss(decoder, memory, batch)
            case ARFastDecoder():
                return ar_fast_loss(decoder, memory, batch)

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
            self.trunk,
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
        memory = self.encode(batch.encoder_inputs, with_grad=False)
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
