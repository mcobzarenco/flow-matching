"""MolmoAct2ARVLA — the MolmoAct2 trunk decoding its action chunk as
in-base discrete action tokens (the format-6 emission: no value lines,
the prompt carries the whole scaffold).

Assembly: the full Molmo2 trunk speaking the MolmoAct2 prompt format
and a
:class:`~bijou.modelling.decoders.ar_molmoact2.MolmoAct2ARDecoder`,
which owns ZERO parameters (trunk-native rows) — the trainable surface
is the trunk itself.

Objective: :class:`~bijou.models.objectives.ARObjective`; the format-6
emission has no aux fields, so ``aux_loss_weight`` is inert and the
objective is single-component CE."""

from __future__ import annotations

from pathlib import Path
from typing import Self, override

import torch
from torch import Tensor, nn

from ..checkpoint import VLAMetadata, read_metadata
from ..modelling.decoders.ar_molmoact2 import MolmoAct2ARDecoder
from ..modelling.decoders.ar_suffix import MOLMOACT2_SUFFIX_FORMAT
from ..modelling.encoders.molmoact2 import MolmoAct2Encoder, MolmoAct2Inputs
from ..modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    CollatedBatch,
    InputsCollator,
    ObservationMemory,
)
from ..modelling.molmo2.loading import load_config as load_molmo2_config
from ..modelling.molmo2.model import Molmo2Model
from ..sections import (
    MOLMOACT2_FAST_TOKENIZER_REF,
    ARDecoderConfig,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    build_molmoact2_ar_decoder,
    molmoact2_ar_config_from_flow_section,
    parse_decoder_config,
)
from ..vla import ARVLA, ARPrediction, LossReport, VLAFamily, VLASpec
from .ar_suffix_ops import (
    ar_block_logits,
    ar_block_prediction,
    ar_loss_counts,
    ar_suffix_report,
)
from .molmoact2_flow import load_molmoact2_backbone, molmoact2_prompt_of
from .objectives import ARObjective, parse_ar_objective
from .serving import ARServing


def build_molmoact2_ar_component(
    metadata: VLAMetadata,
    prompt: MolmoAct2PromptConfig,
    trunk_dir: Path,
    component: str,
) -> MolmoAct2ARDecoder:
    """The discrete decoder from its recorded component config — either
    shape the writers produce: a format-6 ar_backbone section
    (train-written discrete runs; other suffix formats are refused —
    value-line emissions ride the Gemma/Molmo2 prompts), or a
    molmo_flow section (release-class conversions record no format-6
    section; the AR config derives from the flow geometry, block ids
    from the trunk tokenizer, block width from the FAST artifact —
    the recorded one, else the canonical release ref)."""
    section = parse_decoder_config(dict(metadata.components[component]["config"]))
    if isinstance(section, ARDecoderConfig):
        if section.suffix_format != MOLMOACT2_SUFFIX_FORMAT:
            raise SystemExit(
                f"format-{section.suffix_format} ar_backbone section under "
                "the molmoact2 prompt — this prompt family's emission is "
                f"format {MOLMOACT2_SUFFIX_FORMAT} (value-line checkpoints "
                "ride the Gemma/Molmo2 prompts)",
            )
        config = section
    elif isinstance(section, MolmoFlowDecoderConfig):
        config = molmoact2_ar_config_from_flow_section(
            section,
            prompt,
            str(trunk_dir),
            fast_tokenizer=metadata.artifacts.get(
                "fast_tokenizer",
                MOLMOACT2_FAST_TOKENIZER_REF,
            ),
        )
    else:
        raise SystemExit(
            f"a {type(section).__name__} cannot back the molmoact2 "
            "discrete decoder — expected a format-6 ar_backbone or a "
            "molmo_flow section",
        )
    return build_molmoact2_ar_decoder(
        config,
        prompt,
        load_molmo2_config(trunk_dir).text,
        str(trunk_dir),
    )


class MolmoAct2ARVLA(ARVLA[MolmoAct2Inputs]):
    """MolmoAct2 trunk + discrete decoder (module docstring). forward
    owns the precision policy: the suffix decoder IS the backbone, so
    prefix encode AND suffix CE share one regime — bf16 autocast iff
    the trunk is live on CUDA (a frozen trunk constructs the context
    disabled — byte-identical to frozen math); the CE itself upcasts
    to fp32 inside the loss. No narration surface: the format-6
    emission never trained value lines."""

    def __init__(
        self,
        backbone: Molmo2Model,
        encoder: MolmoAct2Encoder,
        ar_decoder: MolmoAct2ARDecoder,
        *,
        objective: ARObjective,
        serving: ARServing,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.ar_decoder = ar_decoder
        self.objective = objective
        self.serving = serving

    @property
    @override
    def spec(self) -> VLASpec:
        return VLASpec(
            family=VLAFamily.MOLMOACT2_AR,
            chunk_size=self.ar_decoder.config.chunk_size,
            action_dim=self.ar_decoder.config.action_dim,
        )

    @override
    def collator(self) -> InputsCollator[MolmoAct2Inputs]:
        return self.encoder.inputs_collator()

    def _encode(
        self,
        batch: CollatedBatch[MolmoAct2Inputs],
        *,
        with_grad: bool,
    ) -> ObservationMemory:
        # The suffix role continues the prefix cache — always retained.
        return self.encoder.encode(
            self.backbone,
            batch.encoder_inputs,
            with_grad=with_grad,
            retain_cache=True,
        )

    @override
    def loss_counts(self, batch: CollatedBatch[MolmoAct2Inputs]) -> dict[str, Tensor]:
        return ar_loss_counts(self.ar_decoder, batch)

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
            memory = self._encode(batch, with_grad=live)
            return ar_suffix_report(
                self.backbone,
                self.ar_decoder,
                memory,
                batch,
                counts=counts,
                aux_loss_weight=self.objective.aux_loss_weight,
            )

    @override
    @torch.no_grad()
    def predict(self, batch: CollatedBatch[MolmoAct2Inputs]) -> Tensor:
        # The recorded operating point: the deterministic greedy decode.
        return self.predict_ar(batch).actions

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
            # The decoder and encoder both own zero parameters — the
            # trainable surface is the trunk (filtered lists stay for
            # uniformity; they are empty by construction).
            "decoder": [p for p in self.ar_decoder.parameters() if p.requires_grad]
            + [p for p in self.encoder.parameters() if p.requires_grad],
            "backbone_text": backbone_groups["text"],
            "backbone_vision": backbone_groups["vision"],
        }

    @override
    def output_head_parameters(self) -> list[nn.Parameter]:
        raise SystemExit(
            "the corrected/standard weight-decay partition is not audited "
            "for the MolmoAct2 discrete decoder (its logits are trunk-"
            "native rows) — audit the split before training this family "
            "with adamc",
        )

    @override
    def checkpoint_components(self) -> dict[str, nn.Module]:
        # Decoder and encoder are both parameterless: their configs ride
        # the metadata's components (weights: false), never files.
        return {}

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
        if metadata.family is not VLAFamily.MOLMOACT2_AR:
            raise SystemExit(
                f"{checkpoint} records family {metadata.family.value!r}, "
                "not molmoact2_ar — load through bijou.loading.load_vla",
            )
        prompt = molmoact2_prompt_of(metadata)
        backbone, encoder, trunk_dir = load_molmoact2_backbone(
            checkpoint,
            metadata,
            prompt,
            device=device,
            dtype=dtype,
        )
        decoder = build_molmoact2_ar_component(
            metadata,
            prompt,
            trunk_dir,
            "ar_decoder",
        )
        model = cls(
            backbone,
            encoder,
            decoder,
            objective=parse_ar_objective(metadata.objective),
            serving=ARServing.from_dict(metadata.serving),
        )
        model.eval()
        return model
