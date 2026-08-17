"""GemmaFlowVLA — the Gemma trunk with the cross-attention flow-matching
action decoder.

Assembly: a Gemma backbone truncated to its non-KV-shared prefix, the
Gemma prompt-side encoder strategy (soft state token, K/V exports), and
a :class:`~bijou.modelling.decoders.flow.FlowDecoder` cross-attending
the exported streams. Decoder and prompt-side parameters are fp32 ("new
parameters"); the trunk mounts at the requested dtype and trains only
when optimizer policy unfreezes it.

Objective union: :class:`~bijou.models.objectives.FlowObjective` (plain
flow matching) | :class:`~bijou.models.objectives.SnapflowObjective`
(the self-distillation mix — requires a φ_s-extended decoder, validated
at construction)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, override

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import Tensor, nn

from ..checkpoint import backbone_files, read_metadata, tokenizer_directory
from ..modelling.decoders.flow import (
    FlowDecoder,
    flow_matching_loss_sums,
    snapflow_distill_loss_sums,
)
from ..modelling.encoders.gemma4 import GemmaEncoder, GemmaInputs, GemmaMemory
from ..modelling.gemma4.config import Gemma4Config
from ..modelling.gemma4.model import Gemma4Model
from ..modelling.interface import (
    CollatedBatch,
    InputsCollator,
    NormStats,
    SamplingMethod,
)
from ..sections import (
    FlowDecoderSection,
    GemmaPromptConfig,
    build_gemma_flow_parts,
    expert_config_from_architecture,
    parse_decoder_config,
    parse_prompt_config,
)
from ..vla import FlowPrediction, FlowVLA, Loss, LossReport, VLAFamily, VLASpec
from .objectives import FlowObjective, SnapflowObjective
from .serving import FlowServing


def normalized_state(batch: CollatedBatch[Any]) -> Tensor:
    """The gemma per-sample stats policy, state side: each sample
    z-scores under its OWN dataset's mean/std.

    Shapes:
    - ``batch.state``/stats rows: [B, state_dim]
    - returns: [B, state_dim] normalized units
    """
    return (batch.state - batch.state_stats.mean) / batch.state_stats.std


def normalized_actions(batch: CollatedBatch[Any]) -> Tensor:
    """The gemma per-sample stats policy, action-target side.

    Shapes:
    - ``batch.actions``: [B, chunk, action_dim] raw units
    - returns: [B, chunk, action_dim] normalized units
    """
    return (
        batch.actions - batch.action_stats.mean[:, None, :]
    ) / batch.action_stats.std[:, None, :]


def raw_chunks(sampled: Tensor, stats: NormStats) -> Tensor:
    """Sampled normalized chunks back to raw units under per-sample
    stats (the affine inverse of :func:`normalized_actions`; the mean
    over draws commutes with it).

    Shapes:
    - ``sampled``: [B, chunk, action_dim] normalized units
    - stats rows: [B, action_dim]
    - returns: [B, chunk, action_dim] float32 raw units
    """
    return sampled.float() * stats.std[:, None, :] + stats.mean[:, None, :]


@torch.no_grad()
def decode_chunk(
    decoder: FlowDecoder,
    memory: GemmaMemory,
    batch: CollatedBatch[Any],
    *,
    generator: torch.Generator | None = None,
    noise: Tensor | None = None,
    num_steps: int = 5,
    method: SamplingMethod = SamplingMethod.HEUN,
    target_time: float | None = None,
    sde_noise_level: float | None = None,
    sde_step_noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """RAW-unit chunk decode — the family-owned boundary around the pure
    decoder: normalize the batch's state with its per-sample stats,
    integrate the field, and denormalize with the action stats.
    ``num_steps``/``method``/``target_time`` are flow-specific solver
    knobs.

    Returns ``(actions, noise)``. ``noise`` is ALWAYS the initial draw
    the solver actually integrated (supplied or drawn) — paired
    re-decodes must reuse it, or sampling variance floors any
    conditioning-sensitivity signal.

    ``sde_noise_level`` routes the decode through
    :meth:`FlowDecoder.sample_actions_sde` (Euler-only, no φ_s) with the
    same normalization seam — ``method`` must be EULER and
    ``target_time`` None so an SDE read can never silently wear ODE
    solver knobs.

    Shapes:
      - returns actions: [B, chunk, action_dim] — RAW action units
      - returns noise: [B, chunk, action_dim] — normalized units
    """
    state = normalized_state(batch)
    if noise is None:
        # The identical draw sample_actions would make (same shape/
        # dtype/device/generator ⇒ bit-exact result and generator
        # consumption) — made here so the return can carry it.
        noise = torch.randn(
            state.shape[0],
            decoder.config.chunk_size,
            decoder.config.action_dim,
            dtype=state.dtype,
            device=state.device,
            generator=generator,
        )
    if sde_noise_level is not None:
        if method is not SamplingMethod.EULER:
            raise ValueError(
                "the SDE decode is Euler-only (a Heun corrector breaks "
                f"the Gaussian transition), got method={method.name}",
            )
        if target_time is not None:
            raise ValueError(
                "sample_actions_sde offers no φ_s shortcut conditioning "
                f"— target_time must be None, got {target_time}",
            )
        sampled = decoder.sample_actions_sde(
            memory,
            state,
            noise_level=sde_noise_level,
            num_steps=num_steps,
            noise=noise,
            step_noise=sde_step_noise,
            generator=generator,
        )
        assert isinstance(sampled, Tensor)  # return_logprob not requested
    elif sde_step_noise is not None:
        raise ValueError("sde_step_noise without sde_noise_level")
    else:
        sampled = decoder.sample_actions(
            memory,
            state,
            num_steps=num_steps,
            method=method,
            noise=noise,
            generator=generator,
            target_time=target_time,
        )
    return raw_chunks(sampled, batch.action_stats), noise


def parse_gemma_flow_objective(
    data: dict[str, Any],
) -> FlowObjective | SnapflowObjective:
    """The family's objective union from the metadata's tagged dict."""
    kind = data.get("kind")
    match kind:
        case "flow":
            return FlowObjective()
        case "snapflow":
            return SnapflowObjective(
                alpha=float(data["alpha"]),
                shortcut_weight=float(data["shortcut_weight"]),
            )
        case _:
            raise SystemExit(
                f"objective kind {kind!r} is not a gemma_flow objective "
                "(flow | snapflow)",
            )


class GemmaFlowVLA(FlowVLA[GemmaInputs]):
    """Gemma trunk + flow decoder (module docstring). forward owns the
    precision policy: the prefix encode runs inside bf16 autocast iff
    the trunk is live on CUDA (a frozen trunk constructs the context
    disabled — byte-identical to frozen math); the flow decoder is
    fp32-by-design and runs OUTSIDE the region, casting the exported
    K/V streams to its own dtype."""

    def __init__(
        self,
        backbone: Gemma4Model,
        encoder: GemmaEncoder,
        flow_decoder: FlowDecoder,
        *,
        objective: FlowObjective | SnapflowObjective,
        serving: FlowServing,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.flow_decoder = flow_decoder
        self.objective = objective
        self.serving = serving
        if isinstance(objective, SnapflowObjective) and (
            not flow_decoder.config.target_time_embed
        ):
            raise SystemExit(
                "the snapflow objective needs a φ_s-extended decoder "
                "(target_time_embed) — extend the checkpoint at init "
                "(the φ_s MLP is zero-initialized, so extension is "
                "function-preserving)",
            )

    @property
    @override
    def spec(self) -> VLASpec:
        return VLASpec(
            family=VLAFamily.GEMMA_FLOW,
            chunk_size=self.flow_decoder.config.chunk_size,
            action_dim=self.flow_decoder.config.action_dim,
        )

    @override
    def collator(self) -> InputsCollator[GemmaInputs]:
        return self.encoder.inputs_collator()

    @override
    def loss_counts(self, batch: CollatedBatch[GemmaInputs]) -> dict[str, Tensor]:
        # Every element of [B, chunk, action_dim] weighs equally.
        return {
            "action_flow": torch.tensor(
                batch.actions.numel(),
                device=batch.actions.device,
            ),
        }

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
            memory = self.encoder.encode(
                self.backbone,
                inputs,
                with_grad=live,
                retain_cache=False,
            )
        state_norm = normalized_state(batch)
        actions_norm = normalized_actions(batch)
        if isinstance(self.objective, SnapflowObjective):
            loss_sum, count = snapflow_distill_loss_sums(
                self.flow_decoder,
                memory,
                state_norm=state_norm,
                actions_norm=actions_norm,
                alpha=self.objective.alpha,
                shortcut_weight=self.objective.shortcut_weight,
            )
        else:
            loss_sum, count = flow_matching_loss_sums(
                self.flow_decoder,
                memory,
                state_norm=state_norm,
                actions_norm=actions_norm,
            )
        world = dist.get_world_size() if dist.is_initialized() else 1
        # Per-rank scalar whose DDP MEAN is the global objective:
        # sum_r · W / global_count.
        objective = loss_sum * world / counts["action_flow"]
        return LossReport(
            objective=objective,
            components={"action_flow": Loss(sum=loss_sum, count=count)},
        )

    @override
    def predict(self, batch: CollatedBatch[GemmaInputs]) -> Tensor:
        return self.predict_flow(
            batch,
            num_steps=self.serving.num_steps,
            method=self.serving.method,
        ).actions

    @override
    @torch.no_grad()
    def predict_flow(
        self,
        batch: CollatedBatch[GemmaInputs],
        *,
        num_steps: int,
        method: SamplingMethod,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        target_time: float | None = None,
    ) -> FlowPrediction:
        """The trait decode, plus this family's φ_s shortcut read:
        ``target_time`` (a family-concrete widening — φ_s-extended
        decoders only, refused otherwise) conditions every solver
        forward on the constant jump target s; None is the standard s=t
        integration every family serves. The 1-NFE read is
        ``target_time=0.0`` with euler/1."""
        memory = self.encoder.encode(
            self.backbone,
            batch.encoder_inputs,
            with_grad=False,
            retain_cache=False,
        )
        actions, drawn = decode_chunk(
            self.flow_decoder,
            memory,
            batch,
            generator=generator,
            noise=noise,
            num_steps=num_steps,
            method=method,
            target_time=target_time,
        )
        return FlowPrediction(actions=actions, noise=drawn)

    @torch.no_grad()
    def predict_flow_sde(
        self,
        batch: CollatedBatch[GemmaInputs],
        *,
        noise_level: float,
        num_steps: int,
        noise: Tensor | None = None,
        step_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowPrediction:
        """The Euler–Maruyama stochastic decode — a family-concrete
        instrument, not a capability (Flow-GRPO's trainable sampler on
        this decoder lineage): Euler-only by construction, no φ_s.
        ``step_noise`` supplies every step's ε explicitly (the
        batch-composition-invariant keyed path); ``noise_level=0``
        reproduces the Euler ODE decode bit-for-bit.

        Shapes:
          - ``noise``/returned noise: [B, chunk, action_dim] —
            normalized units (the initial draw, ALWAYS returned)
          - ``step_noise`` (when given): [num_steps, B, chunk,
            action_dim]
        """
        memory = self.encoder.encode(
            self.backbone,
            batch.encoder_inputs,
            with_grad=False,
            retain_cache=False,
        )
        actions, drawn = decode_chunk(
            self.flow_decoder,
            memory,
            batch,
            generator=generator,
            noise=noise,
            num_steps=num_steps,
            method=SamplingMethod.EULER,
            sde_noise_level=noise_level,
            sde_step_noise=step_noise,
        )
        return FlowPrediction(actions=actions, noise=drawn)

    @override
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone_groups = self.encoder.param_groups(self.backbone)
        return {
            # The decoder's own parameters plus the prompt-side ones
            # (state_proj) — "new parameters" at the same LR; both
            # requires_grad-filtered (frozen-trunk runs freeze
            # state_proj: no gradient path through a no-grad encode).
            "decoder": [p for p in self.flow_decoder.parameters() if p.requires_grad]
            + [p for p in self.encoder.parameters() if p.requires_grad],
            "backbone_text": backbone_groups["text"],
            "backbone_vision": backbone_groups["vision"],
        }

    @override
    def output_head_parameters(self) -> list[nn.Parameter]:
        raise SystemExit(
            "the corrected/standard weight-decay partition is not audited "
            "for the flow decoder — audit its output projection before "
            "training this family with adamc",
        )

    @override
    def checkpoint_components(self) -> dict[str, nn.Module]:
        return {"prompt": self.encoder, "flow_decoder": self.flow_decoder}

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
        if metadata.family is not VLAFamily.GEMMA_FLOW:
            raise SystemExit(
                f"{checkpoint} records family {metadata.family.value!r}, "
                "not gemma_flow — load through bijou.loading.load_vla",
            )
        prompt = parse_prompt_config(metadata.components["prompt"]["config"])
        if not isinstance(prompt, GemmaPromptConfig):
            raise SystemExit(
                f"{checkpoint} records a {type(prompt).__name__} prompt — "
                "gemma_flow rides the gemma4 prompt strategy",
            )
        section = parse_decoder_config(metadata.components["flow_decoder"]["config"])
        if not isinstance(section, FlowDecoderSection):
            raise SystemExit(
                f"{checkpoint} records a {type(section).__name__} as "
                "flow_decoder — gemma_flow carries the flow section",
            )
        backbone_config = Gemma4Config.from_dict(metadata.backbone_config)
        expert_config = expert_config_from_architecture(
            prompt,
            section,
            backbone_config,
        )
        backbone, encoder, decoder = build_gemma_flow_parts(
            backbone_files(checkpoint),
            backbone_config,
            expert_config,
            tokenizer_dir=tokenizer_directory(checkpoint),
            max_soft_tokens=prompt.max_soft_tokens,
            device=device,
            dtype=dtype,
            expert_dtype=torch.float32,
        )
        # CPU-load + copy-in (transient-memory discipline; the copy
        # semantics cast into the built dtypes).
        decoder.load_state_dict(
            load_file(str(checkpoint / "flow_decoder.safetensors"), device="cpu"),
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
            objective=parse_gemma_flow_objective(metadata.objective),
            serving=FlowServing.from_dict(metadata.serving),
        )
        model.eval()
        return model
