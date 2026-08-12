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

import dataclasses
from collections.abc import Callable
from typing import override

import torch
from torch import Tensor, nn

from .aux_text import AuxField
from .decoders.ar_backbone import (
    ActionCaptureStep,
    ARBackboneDecoder,
    ARSampling,
    ARSuffixDecoder,
    ValueCandidate,
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
from .decoders.molmo_flow import (
    MolmoFlowDecoder,
    molmo_flow_loss,
    molmo_flow_loss_sums,
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
        decoder: (
            FlowDecoder
            | ARFastDecoder
            | ARBackboneDecoder
            | Molmo2ARDecoder
            | MolmoFlowDecoder
        ),
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
        # The KI-joint CE rider (joint flow+CE training): a
        # Molmo2ARDecoder whose phase-1 CE objective continues
        # verbatim beside the flow decoder's objective at fixed weight
        # 1.0 (KI's no-tuning result — deliberately NOT a knob). None =
        # every existing composition, byte-identical behavior. train.py's
        # joint-ce flag assigns it post-construction, and module
        # attribute assignment registers it, so DDP/param_groups/
        # state_dict all see it.
        self.joint_ce: Molmo2ARDecoder | None = None
        # Stop-gradient on the expert→trunk seam (the π0.5/KI recipe):
        # raw residual taps are detached before adapter projection, so
        # flow-loss gradients into every trunk parameter are exactly
        # zero while the (with_grad) trunk still receives CE gradients.
        # Run property like ``distill`` — never serialized.
        self.seam_stop_grad: bool = False
        # Knowledge insulation on the molmo_flow KV seam (§8.13 decision
        # 8, their post-train): extracted per-layer K/V detach before
        # the expert, so flow gradients into every trunk parameter are
        # exactly zero while a live trunk still trains through the
        # joint_ce rider. Run property — never serialized.
        self.insulate_expert: bool = False

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
            # The joint-CE rider's tables continue at the decoder LR —
            # phase 1 trained them there (--decoder-lr), and "the CE
            # objective continuing verbatim" includes its optimizer
            # routing.
            # requires_grad-filtered: molmo_flow carries construction-
            # frozen compat tensors (kv_proj, state_encoder — the
            # reference trainable set); every other decoder kind is
            # fully trainable, so the filter is a no-op there.
            "decoder": [p for p in self.decoder.parameters() if p.requires_grad]
            + (list(self.joint_ce.parameters()) if self.joint_ce is not None else [])
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
            # The joint-CE rider is a suffix consumer too: its CE branch
            # continues the prefix cache exactly like a phase-1 step;
            # molmo_flow CONDITIONS on the whole cache (§8.13).
            retain_cache=isinstance(self.decoder, ARSuffixDecoder | MolmoFlowDecoder)
            or self.joint_ce is not None,
        )
        if isinstance(self.decoder, FlowDecoder):
            if self.seam_stop_grad and memory.residuals is not None:
                # The stop-grad seam: taps enter the expert as constants.
                # Detached HERE — before adapter projection — so the cut
                # covers the whole tap-consumption path while the same
                # live-trunk encode still carries CE gradients through
                # the retained cache.
                memory = dataclasses.replace(
                    memory,
                    residuals={
                        name: tap.detach() for name, tap in memory.residuals.items()
                    },
                )
            memory = self.decoder.attach_residual_streams(memory)
        return memory

    @property
    def expert(
        self,
    ) -> (
        FlowDecoder
        | ARFastDecoder
        | ARBackboneDecoder
        | Molmo2ARDecoder
        | MolmoFlowDecoder
    ):
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
        Flow and ar_fast have a single-component objective (aux None).

        The joint-CE arm (``joint_ce`` set) returns (CE total + flow
        total, detached FLOW loss, detached CE ACTION component, count 1)
        — the aux slots carry the CE branch's action CE (the phase-1
        ``loss_action`` analog, the pinned CE-health read), not
        the rider's own aux fields (those contribute to the total and
        stay unlogged)."""
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                objective = (
                    snapflow_distill_loss
                    if self.distill == "snapflow"
                    else flow_matching_loss
                )
                total = objective(decoder, memory, batch)
                if self.joint_ce is not None:
                    # The phase-1 objective verbatim — ar_backbone_losses
                    # is the exact function a phase-1 step calls, so the
                    # CE-only α-edge oracle is bitwise by construction.
                    # Weight 1.0 fixed (KI), deliberately not a knob.
                    ce_total, ce_action, _, _ = ar_backbone_losses(
                        self._molmo2_backbone(),
                        self.joint_ce,
                        memory,
                        batch,
                    )
                    return (
                        ce_total + total,
                        total.detach(),
                        ce_action.detach(),
                        torch.ones((), device=ce_action.device),
                    )
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
            case MolmoFlowDecoder():
                total = molmo_flow_loss(
                    decoder,
                    memory,
                    batch,
                    insulate=self.insulate_expert,
                )
                return total, total.detach(), None, None

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
            case MolmoFlowDecoder():
                # Position count B*T (the per-position valid-dim mean is
                # the inner reduction — molmo_flow_loss_sums' contract).
                return (
                    torch.tensor(
                        batch.actions.shape[0] * batch.actions.shape[1],
                        device=batch.actions.device,
                    ),
                    None,
                )

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
            case MolmoFlowDecoder():
                loss_sum, count = molmo_flow_loss_sums(
                    decoder,
                    memory,
                    batch,
                    insulate=self.insulate_expert,
                )
                return loss_sum, count, None, None

    def joint_loss_count_normalizers(
        self,
        batch: CollatedBatch[I],
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """(flow element count, CE action-token count, CE aux position
        count | None) for one batch/chunk — the joint arm's three
        normalizers (data-only, no model forward). The chunked-backward
        contract mirrors ``loss_count_normalizers``: summed over chunks
        BEFORE the first forward, each chunk's sum-form share divides by
        the full-batch normalizers."""
        joint_ce = self.joint_ce
        assert joint_ce is not None  # joint-arm-only path (train.py routes)
        ce_action_count, ce_aux_count = ar_backbone_counts(joint_ce, batch)
        return (
            torch.tensor(batch.actions.numel(), device=batch.actions.device),
            ce_action_count,
            ce_aux_count,
        )

    def joint_ce_loss_sums(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """The joint arm's CE branch in sum form — exactly the phase-1
        chunked objective (``ar_backbone_loss_sums`` on the rider), split
        out so BijouTrainStep can run the suffix forward INSIDE the
        autocast region (the [B, S, 153k] logits want bf16) while the
        flow branch stays fp32-by-design outside it."""
        joint_ce = self.joint_ce
        assert joint_ce is not None  # joint-arm-only path (train.py routes)
        return ar_backbone_loss_sums(
            self._molmo2_backbone(),
            joint_ce,
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
        num_steps: int | None = None,
        method: SamplingMethod | None = None,
        target_time: float | None = None,
        generate: tuple[AuxField, ...] = (),
    ) -> BijouPrediction:
        """Collated batch → :class:`BijouPrediction` (RAW-unit chunks
        [B, chunk, action_dim] + per-row aux generations for decoders
        with a text surface): encode the observation (no grad) and run
        the decoder's chunk-space inference. ``num_steps``/``method``
        default to each flow kind's OWN operating point when None
        (flow: Heun-5; molmo_flow: the checkpoint's recorded steps,
        Euler — their serving semantics); ``noise``/``target_time`` are
        flow solver knobs. An AR decoder decodes greedily and ignores
        them (``noise`` must then be None). ``generate`` is
        ar_backbone's request set — it must match the request the
        batch's prompts were collated with (Collator.generate_override);
        () = the deployment fast path; ignored by other decoder kinds."""
        memory = self.encode(batch.encoder_inputs, with_grad=False)
        decoder = self.decoder
        match decoder:
            case FlowDecoder():
                return decoder.predict_chunk(
                    memory,
                    batch,
                    generator=generator,
                    noise=noise,
                    num_steps=5 if num_steps is None else num_steps,
                    method=SamplingMethod.HEUN if method is None else method,
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
            case MolmoFlowDecoder():
                if target_time is not None:
                    raise ValueError(
                        "target_time is the SnapFlow φ_s knob — molmo_flow "
                        "has no shortcut embedding (§8.13 step 7)",
                    )
                return decoder.predict_chunk(
                    memory,
                    batch,
                    generator=generator,
                    noise=noise,
                    num_steps=num_steps,
                    method=SamplingMethod.EULER if method is None else method,
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
    def ar_predict_greedy(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
        *,
        generate: tuple[AuxField, ...] = (),
        action_capture: list[ActionCaptureStep] | None = None,
    ) -> BijouPrediction:
        """Greedy AR suffix decode against an ALREADY-ENCODED memory —
        the masked-contrast (mcselect) per-candidate call: the caller encodes each
        conditioned prompt once and decodes with ``action_capture``
        recording the conditional scoring surface from the decode's own
        logits (:class:`ActionCaptureStep`). ``ar_predict_sampled``'s
        decoder dispatch, minus the sampling."""
        match self.decoder:
            case ARBackboneDecoder():
                return self.decoder.predict_chunk(
                    self._gemma_backbone(),
                    memory,
                    batch,
                    generate=generate,
                    action_capture=action_capture,
                )
            case Molmo2ARDecoder():
                return self.decoder.predict_chunk(
                    self._molmo2_backbone(),
                    memory,
                    batch,
                    generate=generate,
                    action_capture=action_capture,
                )
            case _:
                raise TypeError(
                    "greedy AR decode continues a trunk suffix; the "
                    f"loaded decoder is {type(self.decoder).__name__}",
                )

    @torch.no_grad()
    def ar_teacher_forced_block_logits(
        self,
        memory: ObservationMemory,
        action_ids: list[list[int] | None],
    ) -> list[Tensor | None]:
        """Teacher-forced BLOCK logits over per-row action-id sequences
        against an already-encoded memory (consumes its cache —
        snapshot/restore around calls): the masked-contrast (mcselect)
        masked-reference forward. See
        :meth:`ARSuffixDecoder.teacher_forced_block_logits`."""
        match self.decoder:
            case ARBackboneDecoder():
                return self.decoder.teacher_forced_block_logits(
                    self._gemma_backbone(),
                    memory,
                    action_ids,
                )
            case Molmo2ARDecoder():
                return self.decoder.teacher_forced_block_logits(
                    self._molmo2_backbone(),
                    memory,
                    action_ids,
                )
            case _:
                raise TypeError(
                    "teacher-forced suffix scoring needs an AR suffix "
                    f"decoder; loaded: {type(self.decoder).__name__}",
                )

    @torch.no_grad()
    def ar_predict_with_value_candidates(
        self,
        batch: CollatedBatch[I],
        *,
        field: AuxField,
        generate: tuple[AuxField, ...],
        draws: int,
        sampling_for_draw: Callable[[int], ARSampling],
    ) -> tuple[BijouPrediction, list[list[ValueCandidate]]]:
        """The subgoal-draws pass-1 decode (candidate selection): ONE
        prefill, then (a) the full greedy decode — actions + value lines,
        op-identical to :meth:`predict_chunk` so the draws-0 limit stays
        bit-exact against the self-subgoal probe's pass 1 — and (b) ``draws + 1``
        text-only decodes of ``field`` against the restored prefix
        cache: candidate 0 greedy (its per-step stats make the greedy
        candidate scorable), candidates 1..draws temperature-sampled
        under ``sampling_for_draw(draw)`` (the caller keys each draw's
        per-row RNGs — eval's ``stable_sample_rng`` convention).

        Returns (full-pass prediction, per-row candidate lists). The
        greedy candidate's text must equal the full pass's parsed field
        value — same restored cache state, same ops — and a mismatch is
        a loud instrument break, not a warning."""
        if draws < 0:
            raise ValueError(f"draws must be >= 0, got {draws}")

        def run[T: nn.Module](
            decoder: ARSuffixDecoder[T],
            backbone: T,
        ) -> tuple[BijouPrediction, list[list[ValueCandidate]]]:
            memory = self.encode(batch.encoder_inputs, with_grad=False)
            snapshot = decoder.cache_snapshot(memory)
            full = decoder.predict_chunk(backbone, memory, batch, generate=generate)
            rows: list[list[ValueCandidate]] = [[] for _ in range(batch.state.shape[0])]
            for draw in range(draws + 1):
                decoder.cache_restore(memory, snapshot)
                sampling = None if draw == 0 else sampling_for_draw(draw)
                for row, candidate in enumerate(
                    decoder.decode_value_line(
                        backbone,
                        memory,
                        field=field,
                        sampling=sampling,
                    ),
                ):
                    rows[row].append(candidate)
            return full, rows

        match self.decoder:
            case ARBackboneDecoder():
                prediction, candidates = run(self.decoder, self._gemma_backbone())
            case Molmo2ARDecoder():
                prediction, candidates = run(self.decoder, self._molmo2_backbone())
            case _:
                raise TypeError(
                    "value-candidate decode continues a trunk suffix; the "
                    f"loaded decoder is {type(self.decoder).__name__}",
                )
        assert prediction.generations is not None  # AR suffix always generates
        for row, (generation, row_candidates) in enumerate(
            zip(prediction.generations, candidates, strict=True),
        ):
            parsed = getattr(generation, field.value)
            if parsed is not None and not isinstance(parsed, str):
                raise TypeError(
                    f"{field.value!r} parses to {type(parsed).__name__}, not "
                    "text — candidate decoding covers free-text fields only",
                )
            full_value = parsed or ""
            if full_value != row_candidates[0].text:
                raise SystemExit(
                    f"candidate-0 drift on batch row {row}: text-only greedy "
                    f"decode {row_candidates[0].text!r} != full-pass value "
                    f"{full_value!r} — the shared-prefill contract broke, stop",
                )
        return prediction, candidates

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
