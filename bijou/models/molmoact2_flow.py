"""MolmoAct2FlowVLA — the MolmoAct2 trunk with the molmo_flow action
decoder (the per-layer-KV-conditioned DiT).

Assembly: the full Molmo2 trunk speaking the MolmoAct2 prompt format
(discrete state tokens, uint8 single-view images — the encoder owns no
parameters), and a
:class:`~bijou.modelling.decoders.molmo_flow.MolmoFlowDecoder`
conditioning on the whole prefix cache. Normalization is decoder-owned
(the recorded q01/q99 clamp table); the trunk trains only when
optimizer policy unfreezes it.

Objective: :class:`~bijou.models.objectives.FlowObjective`.

The three MolmoAct2 families share their assembly through this module's
free functions (:func:`load_molmoact2_backbone`,
:func:`build_molmoact2_flow_component`) — composition, never a shared
base class."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, override

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import Tensor, nn

from ..checkpoint import VLAMetadata, backbone_directory, read_metadata
from ..modelling.decoders.molmo_flow import (
    MolmoFlowDecoder,
    load_expert_state,
    molmo_flow_loss_sums,
)
from ..modelling.encoders.molmoact2 import MolmoAct2Encoder, MolmoAct2Inputs
from ..modelling.interface import CollatedBatch, InputsCollator, SamplingMethod
from ..modelling.molmo2.model import Molmo2Model
from ..modelling.molmo2.model import load_model as load_molmo2_model
from ..sections import (
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    build_molmo_flow_decoder,
    load_backbone_state,
    parse_decoder_config,
    parse_prompt_config,
)
from ..vla import FlowPrediction, FlowVLA, Loss, LossReport, VLAFamily, VLASpec
from .objectives import FlowObjective
from .serving import FlowServing


def molmoact2_prompt_of(metadata: VLAMetadata) -> MolmoAct2PromptConfig:
    """The molmoact2 families' prompt section, type-checked."""
    prompt = parse_prompt_config(metadata.components["prompt"]["config"])
    if not isinstance(prompt, MolmoAct2PromptConfig):
        raise SystemExit(
            f"checkpoint records a {type(prompt).__name__} prompt — the "
            "molmoact2 families ride the molmoact2 prompt format",
        )
    return prompt


def load_molmoact2_backbone(
    checkpoint: Path,
    metadata: VLAMetadata,
    prompt: MolmoAct2PromptConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[Molmo2Model, MolmoAct2Encoder, Path]:
    """The trunk + prompt-side encoder pair every molmoact2 family
    mounts (the encoder owns zero parameters — nothing further to
    load), plus the resolved trunk directory (tokenizer source for the
    discrete decoder). Trained trunks reload their state from the
    checkpoint's ``backbone.safetensors``."""
    trunk_dir = backbone_directory(checkpoint, metadata)
    backbone = load_molmo2_model(trunk_dir, device=device, dtype=dtype)
    encoder = MolmoAct2Encoder(
        str(trunk_dir),
        setup_type=prompt.setup_type,
        control_mode=prompt.control_mode,
        num_state_tokens=prompt.num_state_tokens,
        action_mode=prompt.action_mode,
        narration=prompt.narration,
    )
    if metadata.backbone_trained:
        load_backbone_state(backbone, checkpoint)
        print(f"loaded trained backbone from {checkpoint}", flush=True)
    return backbone, encoder, trunk_dir


def build_molmoact2_flow_component(
    checkpoint: Path,
    metadata: VLAMetadata,
    *,
    device: torch.device | str,
) -> MolmoFlowDecoder:
    """The flow decoder from its recorded section + trained weights:
    geometry/t-law/serving from the section, the q01/q99 clamp table
    from the metadata's normalization row, converter-export weights
    injected through :func:`load_expert_state` (compat tensors added
    exactly like the reference loader). fp32 — the expert's precision
    by design."""
    section = parse_decoder_config(metadata.components["flow_decoder"]["config"])
    if not isinstance(section, MolmoFlowDecoderConfig):
        raise SystemExit(
            f"checkpoint records a {type(section).__name__} as "
            "flow_decoder — the molmoact2 flow pathway carries the "
            "molmo_flow section",
        )
    decoder = build_molmo_flow_decoder(
        section,
        metadata.stats,
        device=device,
        dtype=torch.float32,
    )
    load_expert_state(
        decoder,
        load_file(str(checkpoint / "flow_decoder.safetensors"), device="cpu"),
    )
    decoder.to(device=device, dtype=torch.float32)
    return decoder


def parse_flow_objective(data: dict[str, Any]) -> FlowObjective:
    """The molmoact2 flow family's objective payload (the unit flow
    variant — snapflow needs a φ_s-extended decoder this pathway does
    not have)."""
    kind = data.get("kind")
    if kind != "flow":
        raise SystemExit(
            f"objective kind {kind!r} is not the flow objective — "
            "molmoact2_flow trains plain flow matching only",
        )
    return FlowObjective()


class MolmoAct2FlowVLA(FlowVLA[MolmoAct2Inputs]):
    """MolmoAct2 trunk + molmo_flow decoder (module docstring). forward
    owns the precision policy: the prefix encode runs inside bf16
    autocast iff the trunk is live on CUDA (a frozen trunk constructs
    the context disabled — byte-identical to frozen math); the flow
    decoder is fp32-by-design and runs OUTSIDE the region."""

    def __init__(
        self,
        backbone: Molmo2Model,
        encoder: MolmoAct2Encoder,
        flow_decoder: MolmoFlowDecoder,
        *,
        objective: FlowObjective,
        serving: FlowServing,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.flow_decoder = flow_decoder
        self.objective = objective
        self.serving = serving

    @property
    @override
    def spec(self) -> VLASpec:
        runtime = self.flow_decoder.runtime
        assert runtime is not None  # configure() ran at build
        return VLASpec(
            family=VLAFamily.MOLMOACT2_FLOW,
            chunk_size=runtime.action_horizon,
            action_dim=runtime.action_dim,
        )

    @override
    def collator(self) -> InputsCollator[MolmoAct2Inputs]:
        return self.encoder.inputs_collator()

    @override
    def loss_counts(self, batch: CollatedBatch[MolmoAct2Inputs]) -> dict[str, Tensor]:
        # Position count B·T (the per-position valid-dim mean is the
        # inner reduction — molmo_flow_loss_sums' contract).
        return {
            "action": torch.tensor(
                batch.actions.shape[0] * batch.actions.shape[1],
                device=batch.actions.device,
            ),
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
        with torch.autocast(
            device_type,
            torch.bfloat16,
            enabled=device_type == "cuda" and live,
        ):
            memory = self.encoder.encode(
                self.backbone,
                inputs,
                with_grad=live,
                retain_cache=True,
            )
        loss_sum, count = molmo_flow_loss_sums(self.flow_decoder, memory, batch)
        world = dist.get_world_size() if dist.is_initialized() else 1
        objective = loss_sum * world / counts["action"]
        return LossReport(
            objective=objective,
            components={"action": Loss(sum=loss_sum, count=count)},
        )

    @override
    def predict(self, batch: CollatedBatch[MolmoAct2Inputs]) -> Tensor:
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
        memory = self.encoder.encode(
            self.backbone,
            batch.encoder_inputs,
            with_grad=False,
            retain_cache=True,
        )
        prediction = self.flow_decoder.predict_chunk(
            memory,
            batch,
            generator=generator,
            noise=noise,
            num_steps=num_steps,
            method=method,
        )
        assert prediction.noise is not None  # flow decodes always carry the draw
        return FlowPrediction(actions=prediction.actions, noise=prediction.noise)

    @override
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone_groups = self.encoder.param_groups(self.backbone)
        return {
            # requires_grad-filtered: molmo_flow carries construction-
            # frozen compat tensors (kv_proj, state_encoder — the
            # reference trainable set). The encoder owns no parameters.
            "decoder": [p for p in self.flow_decoder.parameters() if p.requires_grad]
            + [p for p in self.encoder.parameters() if p.requires_grad],
            "backbone_text": backbone_groups["text"],
            "backbone_vision": backbone_groups["vision"],
        }

    @override
    def output_head_parameters(self) -> list[nn.Parameter]:
        raise SystemExit(
            "the corrected/standard weight-decay partition is not audited "
            "for the molmo_flow decoder — audit its output projection "
            "before training this family with adamc",
        )

    @override
    def checkpoint_components(self) -> dict[str, nn.Module]:
        # The encoder owns zero parameters — its config rides the
        # metadata's prompt component (weights: false), never a file.
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
        if metadata.family is not VLAFamily.MOLMOACT2_FLOW:
            raise SystemExit(
                f"{checkpoint} records family {metadata.family.value!r}, "
                "not molmoact2_flow — load through bijou.loading.load_vla",
            )
        prompt = molmoact2_prompt_of(metadata)
        backbone, encoder, _ = load_molmoact2_backbone(
            checkpoint,
            metadata,
            prompt,
            device=device,
            dtype=dtype,
        )
        decoder = build_molmoact2_flow_component(
            checkpoint,
            metadata,
            device=device,
        )
        model = cls(
            backbone,
            encoder,
            decoder,
            objective=parse_flow_objective(metadata.objective),
            serving=FlowServing.from_dict(metadata.serving),
        )
        model.eval()
        return model
