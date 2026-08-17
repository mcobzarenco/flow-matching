"""GemmaARVLA — the full-depth Gemma trunk emitting its action chunk
(and, when trained, aux value lines) as tokens through the suffix role.

Assembly: the WHOLE Gemma stack (the suffix runs the KV-shared deep
half — prefix-depth checkpoints are refused), the Gemma prompt-side
encoder strategy, and a
:class:`~bijou.modelling.decoders.ar_gemma.GemmaARDecoder` whose
trainable surface is the FAST table patch; the trunk trains only when
optimizer policy unfreezes it.

Objective: :class:`~bijou.models.objectives.ARObjective` — next-token
CE over the suffix; ``narration_weight`` mixes the value-line CE when
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

from ..checkpoint import backbone_files, read_metadata, tokenizer_directory
from ..modelling.aux_text import AuxField
from ..modelling.decoders.ar_gemma import GemmaARDecoder
from ..modelling.encoders.gemma4 import GemmaEncoder, GemmaInputs, GemmaMemory
from ..modelling.gemma4.config import Gemma4Config
from ..modelling.gemma4.model import Gemma4Model
from ..modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    CollatedBatch,
    InputsCollator,
    ValueCandidate,
)
from ..sections import (
    ARDecoderConfig,
    BackboneDepth,
    GemmaPromptConfig,
    build_gemma_ar_decoder,
    build_gemma_encoder,
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
    batch_action_quantiles,
    narrated_prediction,
    value_candidates,
)
from .objectives import ARObjective, parse_ar_objective
from .serving import ARServing


class GemmaARVLA(ARVLA[GemmaInputs], NarratingVLA[GemmaInputs]):
    """Gemma trunk + suffix decoder (module docstring). forward owns the
    precision policy: the suffix decoder IS the backbone, so prefix
    encode AND suffix CE share one regime — bf16 autocast iff the trunk
    is live on CUDA (a frozen trunk constructs the context disabled —
    byte-identical to frozen math); the CE itself upcasts to fp32
    inside the loss."""

    def __init__(
        self,
        backbone: Gemma4Model,
        encoder: GemmaEncoder,
        ar_decoder: GemmaARDecoder,
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
            family=VLAFamily.GEMMA_AR,
            chunk_size=self.ar_decoder.config.chunk_size,
            action_dim=self.ar_decoder.config.action_dim,
        )

    @override
    def collator(self) -> InputsCollator[GemmaInputs]:
        return self.encoder.inputs_collator()

    def _encode(
        self,
        batch: CollatedBatch[GemmaInputs],
        *,
        with_grad: bool,
    ) -> GemmaMemory:
        # The suffix role continues the prefix cache — always retained.
        return self.encoder.encode(
            self.backbone,
            batch.encoder_inputs,
            with_grad=with_grad,
            retain_cache=True,
        )

    @override
    def loss_counts(self, batch: CollatedBatch[GemmaInputs]) -> dict[str, Tensor]:
        return ar_loss_counts(self.ar_decoder, batch)

    @override
    def forward(
        self,
        batch: CollatedBatch[GemmaInputs],
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
                narration_weight=self.objective.narration_weight,
            )

    @override
    @torch.no_grad()
    def predict(self, batch: CollatedBatch[GemmaInputs]) -> Tensor:
        # The recorded operating point: the deterministic greedy decode.
        return self.predict_ar(batch).actions

    @override
    @torch.no_grad()
    def predict_ar(
        self,
        batch: CollatedBatch[GemmaInputs],
        *,
        sampling: ARSampling | None = None,
        capture: list[ActionCaptureStep] | None = None,
    ) -> ARPrediction:
        return ar_block_prediction(
            self.backbone,
            self.ar_decoder,
            self._encode(batch, with_grad=False),
            batch,
            quantiles=batch_action_quantiles(batch),
            sampling=sampling,
            capture=capture,
        )

    @override
    @torch.no_grad()
    def teacher_forced_block_logits(
        self,
        batch: CollatedBatch[GemmaInputs],
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
        batch: CollatedBatch[GemmaInputs],
        *,
        generate: tuple[AuxField, ...],
    ) -> NarratedPrediction:
        return narrated_prediction(
            self.backbone,
            self.ar_decoder,
            self._encode(batch, with_grad=False),
            batch,
            quantiles=batch_action_quantiles(batch),
            generate=generate,
        )

    @override
    @torch.no_grad()
    def predict_with_value_candidates(
        self,
        batch: CollatedBatch[GemmaInputs],
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
            quantiles=batch_action_quantiles(batch),
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
        # fast_embed doubles as the block-logits head (hidden @ Wᵀ) — a
        # TIED embedding/head pair: one parameter, standard AdamW decay.
        return list(self.ar_decoder.fast_embed.parameters())

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
        if metadata.family is not VLAFamily.GEMMA_AR:
            raise SystemExit(
                f"{checkpoint} records family {metadata.family.value!r}, "
                "not gemma_ar — load through bijou.loading.load_vla",
            )
        prompt = parse_prompt_config(metadata.components["prompt"]["config"])
        if not isinstance(prompt, GemmaPromptConfig):
            raise SystemExit(
                f"{checkpoint} records a {type(prompt).__name__} prompt — "
                "gemma_ar rides the gemma4 prompt strategy",
            )
        config = parse_decoder_config(metadata.components["ar_decoder"]["config"])
        if not isinstance(config, ARDecoderConfig):
            raise SystemExit(
                f"{checkpoint} records a {type(config).__name__} as "
                "ar_decoder — gemma_ar carries the ar_backbone section",
            )
        depth = BackboneDepth(metadata.backbone_depth)
        if depth is not BackboneDepth.FULL:
            raise SystemExit(
                f"{checkpoint} records an ar_backbone decoder with a "
                f"'{depth}' backbone — its suffix runs the KV-shared deep "
                "half, which only the full stack has",
            )
        backbone_config = Gemma4Config.from_dict(metadata.backbone_config)
        tokenizer_dir = tokenizer_directory(checkpoint)
        backbone, encoder = build_gemma_encoder(
            backbone_files(checkpoint),
            backbone_config,
            tokenizer_dir=tokenizer_dir,
            exports=prompt.exports,
            max_soft_tokens=prompt.max_soft_tokens,
            state_dim=prompt.state_dim,
            device=device,
            dtype=dtype,
            depth=depth,
        )
        objective = parse_ar_objective(metadata.objective)
        decoder = build_gemma_ar_decoder(
            tokenizer_dir,
            config,
            backbone_config.text,
            narration_weight=objective.narration_weight,
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
        model.eval()
        return model
