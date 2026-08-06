"""Bijou: one backbone, a prompt-side encoder strategy, an action decoder.

The composition root: ``BijouModel`` owns the backbone network ONCE
(today the truncated Gemma; the seam is trunk-generic), plus one
prompt-side encoder strategy (the collation + prefix-encode + unfreeze
surface, which receives the backbone as an argument) and one action
decoder. The observation
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

The root is generic over the encoder's collated-inputs type ``I`` and
the trunk type ``B`` (``ObservationEncoder[I, B]`` pairs them by
construction). Decoder kinds that run the trunk ITSELF (ar_backbone's
suffix continuation) are Gemma-only today and narrow loudly at their
dispatch arms — composing them over another trunk is a wiring bug, not
a silent fallback.

Naming: "backbone" is the ONE word for the trunk network, in both its
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
from .decoders.ar_backbone import (
    ARBackboneDecoder,
    ARSampling,
    ARSuffixDecoder,
    ar_backbone_counts,
    ar_backbone_loss_sums,
    ar_backbone_losses,
)
from .decoders.ar_fast import (
    ARFastDecoder,
    ar_fast_counts,
    ar_fast_loss,
    ar_fast_loss_sums,
)
from .decoders.ar_molmo2 import Molmo2ARDecoder
from .decoders.flow import (
    FlowDecoder,
    SamplingMethod,
    flow_matching_loss,
    flow_matching_loss_sums,
    snapflow_distill_loss,
    snapflow_distill_loss_sums,
)
from .encoders.gemma4 import GemmaEncoder
from .gemma4.model import Gemma4Model
from .interface import (
    BatchInputs,
    BijouPrediction,
    CollatedBatch,
    ObservationEncoder,
    ObservationMemory,
)
from .molmo2.model import Molmo2Model


class BijouModel[I: BatchInputs, B: nn.Module](nn.Module):
    """One backbone + one encoder strategy + one decoder (see the module
    docstring)."""

    def __init__(
        self,
        backbone: B,
        encoder: ObservationEncoder[I, B],
        decoder: FlowDecoder | ARFastDecoder | ARBackboneDecoder | Molmo2ARDecoder,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        # Training-only objective variant (bijou.train --distill): None =
        # each decoder's standard objective; "snapflow" = the SnapFlow
        # self-distillation mix (flow decoders with φ_s only). Never
        # serialized — a run property, not a checkpoint property.
        self.distill: str | None = None

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named trainable-parameter groups — the routing vocabulary for
        component learning rates: ``"decoder"`` (always trained: the
        decoder's own parameters PLUS the encoder's prompt-side ones —
        state_proj — which are "new parameters" in the same sense and
        want the same LR), ``"backbone_text"``/``"backbone_vision"``
        (unfreezable backbone subsets; see ObservationEncoder.param_groups
        for the exactness contract). Groups are disjoint by construction —
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

    def encode(self, inputs: I, *, with_grad: bool) -> ObservationMemory:
        """Encode one collated batch of encoder inputs against the
        backbone. ``with_grad=False`` runs under no_grad (eval/rollout/
        frozen training); True leaves autograd on for live-backbone
        training. The full prefix cache is retained iff this model's
        decoder consumes it (the ar_backbone suffix role).

        Residual-conditioned flow experts project the encoder's raw taps
        into conditioning streams HERE — after the (possibly no-grad)
        prefix encode, in the caller's grad context, once per observation
        — so the adapters train under a frozen backbone and eval pays the
        projection once, not per velocity evaluation."""
        memory = self.encoder.encode(
            self.backbone,
            inputs,
            with_grad=with_grad,
            retain_cache=isinstance(self.decoder, ARSuffixDecoder),
        )
        if isinstance(self.decoder, FlowDecoder):
            memory = self.decoder.attach_residual_streams(memory)
        return memory

    @property
    def expert(
        self,
    ) -> FlowDecoder | ARFastDecoder | ARBackboneDecoder | Molmo2ARDecoder:
        return self.decoder

    def _flow_decoder(self) -> FlowDecoder:
        if not isinstance(self.decoder, FlowDecoder):
            raise TypeError(
                "this operation integrates the flow velocity field; the "
                f"loaded decoder is {type(self.decoder).__name__}",
            )
        return self.decoder

    def _molmo2_backbone(self) -> Molmo2Model:
        """Narrow the generic trunk for the operations that run the
        Molmo2 stack itself (its suffix continuation) — loud if composed
        over another trunk."""
        backbone = self.backbone
        if not isinstance(backbone, Molmo2Model):
            raise TypeError(
                "this operation runs the Molmo2 trunk; the mounted backbone "
                f"is {type(backbone).__name__}",
            )
        return backbone

    def _gemma_backbone(self) -> Gemma4Model:
        """Narrow the generic trunk for the operations that run the Gemma
        stack itself (ar_backbone's suffix continuation, the tensor-level
        Gemma encode) — Gemma-only paths today, loud if composed over
        another trunk."""
        backbone = self.backbone
        if not isinstance(backbone, Gemma4Model):
            raise TypeError(
                "this operation runs the Gemma trunk; the mounted backbone "
                f"is {type(backbone).__name__}",
            )
        return backbone

    def loss(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
    ) -> Tensor:
        """Scalar training loss of this model's decoder for one batch
        against its observation memory — the objective dispatch (each
        decoder kind's objective is a module-level function beside it)."""
        return self.loss_components(memory, batch)[0]

    def loss_components(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """(total, action component, aux CE sum | None, aux position
        count | None) — the total carries the graph; the rest arrive
        detached for logging (aux as sum+count so the train loop can
        aggregate a position-weighted mean across batches and ranks).
        Flow and ar_fast have a single-component objective (aux None)."""
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                objective = (
                    snapflow_distill_loss
                    if self.distill == "snapflow"
                    else flow_matching_loss
                )
                total = objective(decoder, memory, batch)
                return total, total.detach(), None, None
            case ARFastDecoder():
                total = ar_fast_loss(decoder, memory, batch)
                return total, total.detach(), None, None
            case ARBackboneDecoder():
                total, action, aux_sum, aux_count = ar_backbone_losses(
                    self._gemma_backbone(),
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
            case Molmo2ARDecoder():
                total, action, aux_sum, aux_count = ar_backbone_losses(
                    self._molmo2_backbone(),
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

    def loss_count_normalizers(
        self,
        batch: CollatedBatch[I],
    ) -> tuple[Tensor, Tensor | None]:
        """(action normalizer count, aux normalizer count | None) for one
        batch — data-only tensor ops, NO model forward. Chunked backward
        sums these over all chunks BEFORE the first forward, so each
        chunk's sum-form loss divides by the full-batch normalizer (the
        exact unchunked gradient; see loss_component_sums)."""
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                # Every element of [B, chunk, action_dim] weighs equally.
                return (
                    torch.tensor(
                        batch.actions.numel(),
                        device=batch.actions.device,
                    ),
                    None,
                )
            case ARFastDecoder():
                return ar_fast_counts(decoder, batch), None
            case ARBackboneDecoder() | Molmo2ARDecoder():
                return ar_backbone_counts(decoder, batch)

    def loss_component_sums(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Sum-form objective for chunked backward: (action loss SUM with
        graph, action count, aux CE SUM with graph | None, aux count |
        None). With A/X the FULL-batch normalizers from
        loss_count_normalizers, summing ``action_sum / A +
        aux_loss_weight * aux_sum / max(X, 1)`` over chunks reproduces
        loss_components' total exactly (up to fp reduction order) — the
        chunk decomposition is exact even when chunks carry unequal
        valid-token counts, which a plain mean-of-chunk-means is not."""
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                sums = (
                    snapflow_distill_loss_sums
                    if self.distill == "snapflow"
                    else flow_matching_loss_sums
                )
                loss_sum, count = sums(decoder, memory, batch)
                return loss_sum, count, None, None
            case ARFastDecoder():
                loss_sum, count = ar_fast_loss_sums(decoder, memory, batch)
                return loss_sum, count, None, None
            case ARBackboneDecoder():
                return ar_backbone_loss_sums(
                    self._gemma_backbone(),
                    decoder,
                    memory,
                    batch,
                )
            case Molmo2ARDecoder():
                return ar_backbone_loss_sums(
                    self._molmo2_backbone(),
                    decoder,
                    memory,
                    batch,
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
        grad-transparent — training wraps it in autocast, eval in no_grad.
        Residual taps are attached exactly as in :meth:`encode`. The
        signature is Gemma's tensor-level convention — a convenience for
        Gemma compositions, loud otherwise (encode_tensors is not seam
        surface; the seam-level path is :meth:`encode`)."""
        encoder = self.encoder
        if not isinstance(encoder, GemmaEncoder):
            raise TypeError(
                "encode_observation speaks the Gemma tensor-level encode "
                f"convention; the mounted encoder is {type(encoder).__name__}",
            )
        memory = encoder.encode_tensors(
            self._gemma_backbone(),
            input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            padding_mask=padding_mask,
            state=state,
            state_slot=state_slot,
        )
        if isinstance(self.decoder, FlowDecoder):
            memory = self.decoder.attach_residual_streams(memory)
        return memory

    @torch.no_grad()
    def predict_chunk(
        self,
        batch: CollatedBatch[I],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        num_steps: int = 5,
        method: SamplingMethod = SamplingMethod.HEUN,
        target_time: float | None = None,
        generate: tuple[AuxField, ...] = (),
    ) -> BijouPrediction:
        """Collated batch → :class:`BijouPrediction` (RAW-unit chunks
        [B, chunk, action_dim] + per-row aux generations for decoders
        with a text surface): encode the observation (no grad) and run
        the decoder's chunk-space inference with the batch's per-sample
        stats. ``num_steps``/``method``/``noise``/``target_time`` are
        flow solver knobs;
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
                    target_time=target_time,
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
                    self._gemma_backbone(),
                    memory,
                    batch,
                    generate=generate,
                    generator=generator,
                    noise=noise,
                )
            case Molmo2ARDecoder():
                return decoder.predict_chunk(
                    self._molmo2_backbone(),
                    memory,
                    batch,
                    generate=generate,
                    generator=generator,
                    noise=noise,
                )

    @torch.no_grad()
    def ar_predict_sampled(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
        *,
        generate: tuple[AuxField, ...] = (),
        sampling: ARSampling,
    ) -> BijouPrediction:
        """AR suffix decode against an ALREADY-ENCODED memory with the
        action block temperature-sampled — the sampled-draws eval
        instrument's per-draw call: the caller encodes once, snapshots
        the prefix cache, and restores between draws
        (:meth:`ARSuffixDecoder.cache_snapshot`/``cache_restore``), so
        N draws share one prefill. Loud on decoders without a suffix
        cache to share (flow samples noise; ar_fast has no trunk)."""
        match self.decoder:
            case ARBackboneDecoder():
                return self.decoder.predict_chunk(
                    self._gemma_backbone(),
                    memory,
                    batch,
                    generate=generate,
                    sampling=sampling,
                )
            case Molmo2ARDecoder():
                return self.decoder.predict_chunk(
                    self._molmo2_backbone(),
                    memory,
                    batch,
                    generate=generate,
                    sampling=sampling,
                )
            case _:
                raise TypeError(
                    "sampled AR decode continues a trunk suffix; the "
                    f"loaded decoder is {type(self.decoder).__name__}",
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
