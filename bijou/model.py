"""Bijou: one backbone, a prompt-side encoder strategy, an action decoder.

The composition root: ``BijouModel`` owns the backbone network ONCE (the
truncated Gemma), plus one prompt-side encoder strategy (the collation +
prefix-encode + unfreeze surface, which receives the backbone as an
argument) and one action decoder. The observation
(chat-templated instruction + camera images) is encoded once and cached
as an :class:`ObservationMemory`; the decoder then denoises a chunk of
actions against it, with fresh robot state, at ~10 model evaluations per
chunk.

Backbone ownership lives here — not in the encoder — because one network
can serve several roles: the prefix encoder for the cross-attention
decoders today, and prefix + suffix runner for the decoder-only path.
The root is the one place with every role in scope, so it also owns the
objective dispatch (:meth:`loss`) and the trainable-group routing
(:meth:`param_groups`).

Naming: "backbone" is the ONE word for the Gemma network, in both its
senses — the pretrained artifact (``--backbone``, ``BackboneConfig.id``,
``backbone.safetensors``) and the mounted module (``model.backbone``).
``expert`` is the decoder's historical name (checkpoint file
``expert.safetensors``).
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn

from .aux_text import AuxField
from .decoders.ar_backbone import ARBackboneDecoder, ar_backbone_losses
from .decoders.ar_fast import ARFastDecoder, ar_fast_loss
from .decoders.flow import FlowDecoder, SamplingMethod, flow_matching_loss
from .encoders.gemma4 import GemmaEncoder, GemmaInputs
from .gemma4.model import Gemma4Model
from .interface import BijouPrediction, CollatedBatch, ObservationMemory


class BijouModel(nn.Module):
    """One backbone + one encoder strategy + one decoder (see the module
    docstring)."""

    def __init__(
        self,
        backbone: Gemma4Model,
        encoder: GemmaEncoder,
        decoder: FlowDecoder | ARFastDecoder | ARBackboneDecoder,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named trainable-parameter groups — the routing vocabulary for
        component learning rates: ``"decoder"`` (always trained: the
        decoder's own parameters PLUS the encoder's prompt-side ones —
        state_proj — which are "new parameters" in the same sense and
        want the same LR), ``"backbone_text"``/``"backbone_vision"``
        (unfreezable backbone subsets; see GemmaEncoder.param_groups for
        the exactness contract). Groups are disjoint by construction —
        the encoder module carries only prompt-side parameters, never
        backbone ones (the backbone is passed as an argument). Encoder
        params are filtered by ``requires_grad``: frozen-backbone runs
        freeze state_proj (no gradient path through a no-grad prefix
        encode — train.py sets this, loudly), and DDP's exactness
        contract needs it OUT of the group there."""
        backbone_groups = self.encoder.param_groups(self.backbone)
        return {
            "decoder": list(self.decoder.parameters())
            + [p for p in self.encoder.parameters() if p.requires_grad],
            "backbone_text": backbone_groups["text"],
            "backbone_vision": backbone_groups["vision"],
        }

    def encode(self, inputs: GemmaInputs, *, with_grad: bool) -> ObservationMemory:
        """Encode one collated batch of encoder inputs against the
        backbone. ``with_grad=False`` runs under no_grad (eval/rollout/
        frozen training); True leaves autograd on for live-backbone
        training. The full prefix cache is retained iff this model's
        decoder consumes it (the ar_backbone suffix role)."""
        return self.encoder.encode(
            self.backbone,
            inputs,
            with_grad=with_grad,
            retain_cache=isinstance(self.decoder, ARBackboneDecoder),
        )

    @property
    def expert(self) -> FlowDecoder | ARFastDecoder | ARBackboneDecoder:
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
        return self.loss_components(memory, batch)[0]

    def loss_components(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[GemmaInputs],
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """(total, action component, aux CE sum | None, aux position
        count | None) — the total carries the graph; the rest arrive
        detached for logging (aux as sum+count so the train loop can
        aggregate a position-weighted mean across batches and ranks).
        Flow and ar_fast have a single-component objective (aux None)."""
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                total = flow_matching_loss(decoder, memory, batch)
                return total, total.detach(), None, None
            case ARFastDecoder():
                total = ar_fast_loss(decoder, memory, batch)
                return total, total.detach(), None, None
            case ARBackboneDecoder():
                total, action, aux_sum, aux_count = ar_backbone_losses(
                    self.backbone,
                    decoder,
                    memory,
                    batch,
                )
                return (
                    total,
                    action.detach(),
                    None if aux_sum is None else aux_sum.detach(),
                    aux_count,
                )

    def encode_observation(
        self,
        input_ids: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
        state: Tensor | None = None,
        state_slot: int | None = None,
    ) -> ObservationMemory:
        """Tensor-level observation encode (shapes in
        GemmaEncoder.encode_tensors);
        grad-transparent — training wraps it in autocast, eval in no_grad."""
        return self.encoder.encode_tensors(
            self.backbone,
            input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            padding_mask=padding_mask,
            state=state,
            state_slot=state_slot,
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
        generate: tuple[AuxField, ...] = (),
    ) -> BijouPrediction:
        """Collated batch → :class:`BijouPrediction` (RAW-unit chunks
        [B, chunk, action_dim] + per-row aux generations for decoders
        with a text surface): encode the observation (no grad) and run
        the decoder's chunk-space inference with the batch's per-sample
        stats. ``num_steps``/``method``/``noise`` are flow solver knobs;
        an AR decoder decodes greedily and ignores them (``noise`` must
        then be None). ``generate`` is ar_backbone's request set — it
        must match the request the batch's prompts were collated with
        (Collator.generate_override); () = the deployment fast path;
        ignored by other decoder kinds."""
        memory = self.encode(batch.encoder_inputs, with_grad=False)
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                return decoder.predict_chunk(
                    memory,
                    batch,
                    generator=generator,
                    noise=noise,
                    num_steps=num_steps,
                    method=method,
                )
            case ARFastDecoder():
                return decoder.predict_chunk(
                    memory,
                    batch,
                    generator=generator,
                    noise=noise,
                )
            case ARBackboneDecoder():
                return decoder.predict_chunk(
                    self.backbone,
                    memory,
                    batch,
                    generate=generate,
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
