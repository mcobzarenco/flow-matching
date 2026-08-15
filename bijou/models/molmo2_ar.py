"""Molmo2ARVLA — the Molmo2 trunk emitting its action chunk (and, when
trained, aux value lines) as tokens through the suffix role.

Assembly: the full Molmo2 stack (always full-depth — the suffix reads
the shipped head), the Molmo2 prompt-side encoder strategy (soft state
token, crop budget), and a
:class:`~bijou.modelling.decoders.ar_molmo2.Molmo2ARDecoder` whose
trainable surface is the untied FAST table/head pair; the trunk trains
only when optimizer policy unfreezes it.

Objective: :class:`~bijou.models.objectives.ARObjective` — next-token
CE over the suffix; ``aux_loss_weight`` mixes the value-line CE when
the checkpoint trained aux fields (an aux-less checkpoint loads fine
and simply has no ``"aux"`` component; its narration surface refuses
requests at decode time, from its own trained-fields record)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Self, override

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from ..checkpoint import backbone_directory, read_metadata
from ..modelling.aux_text import AuxField
from ..modelling.decoders.ar_molmo2 import Molmo2ARDecoder
from ..modelling.encoders.molmo2 import Molmo2Encoder, Molmo2Inputs
from ..modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    CollatedBatch,
    InputsCollator,
    ObservationMemory,
    ValueCandidate,
)
from ..modelling.molmo2.loading import load_config as load_molmo2_config
from ..modelling.molmo2.model import Molmo2Model
from ..modelling.molmo2.model import load_model as load_molmo2_model
from ..sections import (
    ARDecoderConfig,
    Molmo2PromptConfig,
    build_molmo2_ar_decoder,
    load_backbone_state,
    parse_decoder_config,
    parse_prompt_config,
)
from ..vla import (
    ARVLA,
    ARPrediction,
    LossReport,
    NarratedPrediction,
    NarratingVLA,
    VLAFamily,
    VLASpec,
)
from .ar_suffix_ops import (
    ar_block_logits,
    ar_block_prediction,
    ar_loss_counts,
    ar_suffix_report,
    narrated_prediction,
    value_candidates,
)
from .objectives import ARObjective, parse_ar_objective
from .serving import ARServing


class Molmo2ARVLA(ARVLA[Molmo2Inputs], NarratingVLA[Molmo2Inputs]):
    """Molmo2 trunk + suffix decoder (module docstring). forward owns
    the precision policy: the suffix decoder IS the backbone, so prefix
    encode AND suffix CE share one regime — bf16 autocast iff the trunk
    is live on CUDA (a frozen trunk constructs the context disabled —
    byte-identical to frozen math); the CE itself upcasts to fp32
    inside the loss."""

    def __init__(
        self,
        backbone: Molmo2Model,
        encoder: Molmo2Encoder,
        ar_decoder: Molmo2ARDecoder,
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
            family=VLAFamily.MOLMO2_AR,
            chunk_size=self.ar_decoder.config.chunk_size,
            action_dim=self.ar_decoder.config.action_dim,
        )

    @override
    def collator(self) -> InputsCollator[Molmo2Inputs]:
        return self.encoder.inputs_collator()

    def _encode(
        self,
        batch: CollatedBatch[Molmo2Inputs],
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
    def loss_counts(self, batch: CollatedBatch[Molmo2Inputs]) -> dict[str, Tensor]:
        return ar_loss_counts(self.ar_decoder, batch)

    @override
    def forward(
        self,
        batch: CollatedBatch[Molmo2Inputs],
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
    def predict(self, batch: CollatedBatch[Molmo2Inputs]) -> Tensor:
        # The recorded operating point: the deterministic greedy decode.
        return self.predict_ar(batch).actions

    @override
    @torch.no_grad()
    def predict_ar(
        self,
        batch: CollatedBatch[Molmo2Inputs],
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
        batch: CollatedBatch[Molmo2Inputs],
        action_ids: Tensor,
    ) -> Tensor:
        return ar_block_logits(
            self.backbone,
            self.ar_decoder,
            self._encode(batch, with_grad=False),
            action_ids,
        )

    @override
    @torch.no_grad()
    def predict_narrated(
        self,
        batch: CollatedBatch[Molmo2Inputs],
        *,
        generate: tuple[AuxField, ...],
    ) -> NarratedPrediction:
        return narrated_prediction(
            self.backbone,
            self.ar_decoder,
            self._encode(batch, with_grad=False),
            batch,
            generate=generate,
        )

    @override
    @torch.no_grad()
    def predict_with_value_candidates(
        self,
        batch: CollatedBatch[Molmo2Inputs],
        *,
        field: AuxField,
        generate: tuple[AuxField, ...],
        draws: int,
        sampling_for_draw: Callable[[int], ARSampling],
    ) -> tuple[NarratedPrediction, list[list[ValueCandidate]]]:
        return value_candidates(
            self.backbone,
            self.ar_decoder,
            self._encode(batch, with_grad=False),
            batch,
            field=field,
            generate=generate,
            draws=draws,
            sampling_for_draw=sampling_for_draw,
        )

    @override
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone_groups = self.encoder.param_groups(self.backbone)
        return {
            "decoder": [p for p in self.ar_decoder.parameters() if p.requires_grad]
            + [p for p in self.encoder.parameters() if p.requires_grad],
            "backbone_text": backbone_groups["text"],
            "backbone_vision": backbone_groups["vision"],
        }

    @override
    def output_head_parameters(self) -> list[nn.Parameter]:
        # fast_head is the fresh untied logit projection; fast_embed is
        # an input table and stays with the hidden (corrected) matrices.
        return list(self.ar_decoder.fast_head.parameters())

    @override
    def checkpoint_components(self) -> dict[str, nn.Module]:
        return {"prompt": self.encoder, "ar_decoder": self.ar_decoder}

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
        if metadata.family is not VLAFamily.MOLMO2_AR:
            raise SystemExit(
                f"{checkpoint} records family {metadata.family.value!r}, "
                "not molmo2_ar — load through bijou.loading.load_vla",
            )
        prompt = parse_prompt_config(metadata.components["prompt"]["config"])
        if not isinstance(prompt, Molmo2PromptConfig):
            raise SystemExit(
                f"{checkpoint} records a {type(prompt).__name__} prompt — "
                "molmo2_ar rides the molmo2 prompt strategy",
            )
        config = parse_decoder_config(metadata.components["ar_decoder"]["config"])
        if not isinstance(config, ARDecoderConfig):
            raise SystemExit(
                f"{checkpoint} records a {type(config).__name__} as "
                "ar_decoder — molmo2_ar carries the ar_backbone section",
            )
        trunk_dir = backbone_directory(checkpoint, metadata)
        molmo2_config = load_molmo2_config(trunk_dir)
        backbone = load_molmo2_model(trunk_dir, device=device, dtype=dtype)
        encoder = Molmo2Encoder(
            str(trunk_dir),
            max_crops=prompt.max_crops,
            state_dim=prompt.state_dim,
            hidden_size=molmo2_config.text.hidden_size,
            device=device,
            dtype=torch.float32,
        )
        objective = parse_ar_objective(metadata.objective)
        decoder = build_molmo2_ar_decoder(
            str(trunk_dir),
            config,
            molmo2_config.text,
            aux_loss_weight=objective.aux_loss_weight,
            device=device,
            dtype=torch.float32,
        )
        decoder.load_state_dict(
            load_file(str(checkpoint / "ar_decoder.safetensors"), device="cpu"),
            strict=True,
        )
        encoder.load_state_dict(
            load_file(str(checkpoint / "prompt.safetensors"), device="cpu"),
            strict=True,
        )
        model = cls(
            backbone,
            encoder,
            decoder,
            objective=objective,
            serving=ARServing.from_dict(metadata.serving),
        )
        if metadata.backbone_trained:
            load_backbone_state(backbone, checkpoint)
            print(f"loaded trained backbone from {checkpoint}", flush=True)
        model.eval()
        return model
