"""MolmoAct2JointVLA — the MolmoAct2 trunk with BOTH action decoders:
the molmo_flow expert and the parameterless discrete rider, trained
jointly (L = mean(flow) + ce_weight·mean(CE)).

Assembly: the molmoact2 flow family's parts plus the discrete decoder
riding the same prefix cache. The rider owns ZERO parameters (trunk-
native rows), so it contributes no checkpoint section file and no
optimizer parameters — its trainable surface is the trunk. ONE
family-owned merged q01/q99 table (``action_quantiles``, a
:class:`~bijou.fast.molmoact2.QuantileStats`) feeds both heads: flow
targets normalize and samples denormalize through it, and the discrete
decode detokenizes under its rows — the heads-share-one-table
invariant is structural, not a discipline.

Objective: :class:`JointObjective` (defined here — family-unique),
optionally with the flow gradients stopped at the KV seam so the trunk
learns only from CE (knowledge insulation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, override

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ..checkpoint import read_metadata
from ..fast.molmoact2 import QuantileStats
from ..modelling.decoders.ar_molmoact2 import MolmoAct2ARDecoder
from ..modelling.decoders.ar_suffix import ar_backbone_counts, ar_backbone_loss_sums
from ..modelling.decoders.molmo_flow import MolmoFlowDecoder, molmo_flow_loss_sums
from ..modelling.encoders.molmo2 import Molmo2Memory
from ..modelling.encoders.molmoact2 import MolmoAct2Encoder, MolmoAct2Inputs
from ..modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    CollatedBatch,
    InputsCollator,
    SamplingMethod,
)
from ..modelling.molmo2.model import Molmo2Model
from ..vla import (
    ARVLA,
    ARPrediction,
    FlowPrediction,
    FlowVLA,
    Loss,
    LossReport,
    VLAFamily,
    VLASpec,
)
from .ar_suffix_ops import ar_block_logits, ar_block_prediction
from .molmoact2_ar import build_molmoact2_ar_component
from .molmoact2_flow import (
    build_molmoact2_flow_component,
    flow_denormalize_chunk,
    flow_normalize_targets,
    load_molmoact2_backbone,
    molmoact2_action_quantiles,
    molmoact2_prompt_of,
    per_dataset_flow_scheme,
)
from .serving import FlowServing


@dataclass(frozen=True, slots=True)
class JointObjective:
    """L = mean(flow) + ce_weight·mean(CE), optionally with the flow
    gradients stopped at the KV seam so the trunk learns only from CE
    (knowledge insulation). Under insulation the two terms reach
    disjoint parameter sets, so ``ce_weight`` is an LR-relative knob,
    not a tuned constant."""

    ce_weight: float
    insulate_flow: bool

    def __post_init__(self) -> None:
        if not self.ce_weight > 0:
            raise ValueError(
                f"ce_weight must be > 0, got {self.ce_weight} — a "
                "zero-weight CE term is the flow objective; construct "
                "MolmoAct2FlowVLA instead",
            )


def parse_joint_objective(data: dict[str, Any]) -> JointObjective:
    """The joint family's payload from the metadata's tagged dict."""
    kind = data.get("kind")
    if kind != "joint":
        raise SystemExit(
            f"objective kind {kind!r} is not the joint objective (flow + ce_weight·CE)",
        )
    return JointObjective(
        ce_weight=float(data["ce_weight"]),
        insulate_flow=bool(data["insulate_flow"]),
    )


class MolmoAct2JointVLA(ARVLA[MolmoAct2Inputs], FlowVLA[MolmoAct2Inputs]):
    """MolmoAct2 trunk with both action decoders (module docstring).
    forward owns the joint precision policy: the prompt-only KV is
    extracted for the flow decoder BEFORE the CE suffix extends the
    cache (ordering is a trained contract, not an implementation
    accident); the flow branch runs fp32 outside autocast; the CE
    branch re-enters the trunk's regime — bf16 autocast iff the trunk
    is live on CUDA (its [B, S, 154k] logits want bf16 there; a frozen
    trunk constructs the context disabled, byte-identical to frozen
    math). Insulation is a detach at exactly the KV seam, inside the
    flow branch."""

    def __init__(
        self,
        backbone: Molmo2Model,
        encoder: MolmoAct2Encoder,
        flow_decoder: MolmoFlowDecoder,
        ar_decoder: MolmoAct2ARDecoder,
        *,
        action_quantiles: QuantileStats,
        objective: JointObjective,
        serving: FlowServing,
        per_dataset_flow_norm: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.flow_decoder = flow_decoder
        # The rider owns zero parameters (trunk-native rows), so its
        # registration matters for dispatch, never for the optimizer,
        # DDP buckets, or state-dict contents.
        self.ar_decoder = ar_decoder
        self.action_quantiles = action_quantiles
        self.objective = objective
        self.serving = serving
        # The FLOW leg's normalization scheme (section tag
        # "q01q99_per_dataset"): flow targets and sampled chunks use
        # each item's own dataset row. The discrete head is untouched —
        # CE tokenized under the merged table, so predict_ar keeps
        # ``action_quantiles.rows``.
        self.per_dataset_flow_norm = per_dataset_flow_norm

    @property
    @override
    def spec(self) -> VLASpec:
        runtime = self.flow_decoder.runtime
        assert runtime is not None  # configure() ran at build
        return VLASpec(
            family=VLAFamily.MOLMOACT2_JOINT,
            chunk_size=runtime.action_horizon,
            action_dim=runtime.action_dim,
        )

    @override
    def collator(self) -> InputsCollator[MolmoAct2Inputs]:
        return self.encoder.inputs_collator()

    def _encode(
        self,
        batch: CollatedBatch[MolmoAct2Inputs],
        *,
        with_grad: bool,
    ) -> Molmo2Memory:
        # Both decoders consume the whole prefix cache (the memory always
        # carries it).
        return self.encoder.encode(
            self.backbone,
            batch.encoder_inputs,
            with_grad=with_grad,
        )

    @override
    def loss_counts(self, batch: CollatedBatch[MolmoAct2Inputs]) -> dict[str, Tensor]:
        ce_count, ce_aux_count = ar_backbone_counts(self.ar_decoder, batch)
        assert ce_aux_count is None  # rider constructed aux-None (format 6)
        return {
            # Flow normalizer = B·T positions (the per-position
            # valid-dim mean is the inner reduction).
            "action_flow": torch.tensor(
                batch.actions.shape[0] * batch.actions.shape[1],
                device=batch.actions.device,
            ),
            # The CE branch's action-token count — logged as the "action_ar" component
            # (the joint arm's historical component convention: the CE
            # read is the pinned CE-health metric).
            "action_ar": ce_count,
        }

    @override
    def forward(
        self,
        batch: CollatedBatch[MolmoAct2Inputs],
        *,
        counts: dict[str, Tensor],
    ) -> LossReport:
        inputs = batch.encoder_inputs
        device_type = next(iter(inputs.tensors().values())).device.type
        # Live iff optimizer policy unfroze trunk subsets for this run.
        live = any(p.requires_grad for p in self.backbone.parameters())
        autocast_on = device_type == "cuda" and live
        with torch.autocast(device_type, torch.bfloat16, enabled=autocast_on):
            memory = self._encode(batch, with_grad=live)
        # The flow branch FIRST (fp32, outside autocast): it extracts
        # its prompt-only KV pairs before the CE rider's suffix forward
        # appends teacher-forced action K/V to the same cache — the
        # expert must never condition on them. Insulation detaches at
        # exactly this extraction.
        flow_sum, flow_count = molmo_flow_loss_sums(
            self.flow_decoder,
            memory,
            actions_norm=flow_normalize_targets(
                batch,
                self.action_quantiles,
                per_dataset=self.per_dataset_flow_norm,
            ),
            insulate=self.objective.insulate_flow,
        )
        with torch.autocast(device_type, torch.bfloat16, enabled=autocast_on):
            ce_sum, ce_count, ce_aux_sum, _ = ar_backbone_loss_sums(
                self.backbone,
                self.ar_decoder,
                memory,
                batch,
            )
        assert ce_aux_sum is None  # rider constructed aux-None (format 6)
        world = dist.get_world_size() if dist.is_initialized() else 1
        # Per-rank scalar whose DDP MEAN is the global objective:
        # sum_r · W / global_count per term.
        objective = flow_sum * world / counts[
            "action_flow"
        ] + self.objective.ce_weight * (ce_sum * world / counts["action_ar"])
        return LossReport(
            objective=objective,
            components={
                "action_flow": Loss(sum=flow_sum, count=flow_count),
                "action_ar": Loss(sum=ce_sum, count=ce_count),
            },
        )

    @override
    def predict(self, batch: CollatedBatch[MolmoAct2Inputs]) -> Tensor:
        # The deployment path is the flow decoder at the RECORDED
        # operating point (the reference serving semantics).
        return self.predict_flow(
            batch,
            num_steps=self.serving.num_steps,
            method=self.serving.method,
        ).actions

    @override
    @torch.no_grad()
    def predict_flow(
        self,
        batch: CollatedBatch[MolmoAct2Inputs],
        *,
        num_steps: int,
        method: SamplingMethod,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowPrediction:
        memory = self._encode(batch, with_grad=False)
        chunk, drawn = self.flow_decoder.sample_chunk(
            memory,
            generator=generator,
            noise=noise,
            num_steps=num_steps,
            method=method,
        )
        # The reference's output dtype path: denormalize in fp32, cast
        # BACK to the sampled dtype, then fp32 — the bf16 quantization
        # is part of the reference output when the expert runs bf16.
        unnormalized = flow_denormalize_chunk(
            batch,
            chunk.cpu(),
            self.action_quantiles,
            per_dataset=self.per_dataset_flow_norm,
        )
        actions = unnormalized.to(chunk.dtype).to(torch.float32)
        return FlowPrediction(actions=actions, noise=drawn)

    @override
    @torch.no_grad()
    def predict_ar(
        self,
        batch: CollatedBatch[MolmoAct2Inputs],
        *,
        sampling: ARSampling | None = None,
        capture: list[ActionCaptureStep] | None = None,
    ) -> ARPrediction:
        return ar_block_prediction(
            self.backbone,
            self.ar_decoder,
            self._encode(batch, with_grad=False),
            batch,
            # Both heads share the ONE family table: the discrete decode
            # detokenizes under the same rows the flow head clamps to.
            quantiles=self.action_quantiles.rows(batch.state.shape[0]),
            sampling=sampling,
            capture=capture,
        )

    @override
    @torch.no_grad()
    def teacher_forced_block_logits(
        self,
        batch: CollatedBatch[MolmoAct2Inputs],
        action_ids: Tensor,
    ) -> Tensor:
        return ar_block_logits(
            self.backbone,
            self.ar_decoder,
            self._encode(batch, with_grad=False),
            action_ids,
        )

    @override
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone_groups = self.encoder.param_groups(self.backbone)
        return {
            # The rider contributes nothing (zero parameters); under
            # insulation the trunk still trains via CE, so the backbone
            # groups stay offered.
            "decoder": [p for p in self.flow_decoder.parameters() if p.requires_grad]
            + list(self.ar_decoder.parameters())
            + [p for p in self.encoder.parameters() if p.requires_grad],
            "backbone_text": backbone_groups["text"],
            "backbone_vision": backbone_groups["vision"],
        }

    @override
    def output_head_parameters(self) -> list[nn.Parameter]:
        raise SystemExit(
            "the corrected/standard weight-decay partition is not audited "
            "for the molmo_flow decoder + discrete rider — audit the "
            "split before training this family with adamc",
        )

    @override
    def checkpoint_components(self) -> dict[str, nn.Module]:
        # ar_decoder is parameterless → no section file (its config
        # rides the metadata's ar_decoder component, weights: false);
        # same for the encoder.
        return {"flow_decoder": self.flow_decoder}

    @classmethod
    @override
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Self:
        metadata = read_metadata(checkpoint)
        if metadata.family is not VLAFamily.MOLMOACT2_JOINT:
            raise SystemExit(
                f"{checkpoint} records family {metadata.family.value!r}, "
                "not molmoact2_joint — load through bijou.loading.load_vla",
            )
        prompt = molmoact2_prompt_of(metadata)
        backbone, encoder, tokenizer_dir = load_molmoact2_backbone(
            checkpoint,
            metadata,
            prompt,
            device=device,
            dtype=dtype,
        )
        flow_decoder = build_molmoact2_flow_component(
            checkpoint,
            metadata,
            device=device,
        )
        ar_decoder = build_molmoact2_ar_component(
            metadata,
            prompt,
            tokenizer_dir,
            "ar_decoder",
        )
        model = cls(
            backbone,
            encoder,
            flow_decoder,
            ar_decoder,
            action_quantiles=molmoact2_action_quantiles(metadata),
            objective=parse_joint_objective(metadata.objective),
            serving=FlowServing.from_dict(metadata.serving),
            per_dataset_flow_norm=per_dataset_flow_scheme(metadata),
        )
        model.eval()
        return model
