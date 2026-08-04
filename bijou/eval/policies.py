"""Chunk-prediction policies for open-loop evaluation.

Every policy consumes raw LeRobot items (with per-dataset stats attached by
``StatsAttachedDataset``) in batches and returns per-item action chunks in
raw action units, [chunk, action_dim] each. Batching is up to the policy —
the runner hands over the same items to every policy, so comparisons are
paired by construction.

Determinism: policies that sample flow-matching noise derive it per item
from ``seed + global_index`` on CPU, so predictions are independent of batch
composition, evaluation order and device.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from ..annotations import ConditionField
from ..aux_text import AuxField, AuxGeneration
from ..decoders.ar_backbone import ARBackboneDecoder
from ..decoders.flow import FlowDecoder
from ..interface import Collator
from ..loading import CheckpointInfo, from_checkpoint
from ..model import BijouModel, SamplingMethod


class ChunkPolicy(Protocol):
    """A named policy that predicts action chunks for a batch of items."""

    name: str

    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        """Raw-unit chunk predictions, one [chunk, action_dim] per item."""
        ...


class StateCopyPolicy:
    """The trivial baseline: hold the current joint positions for the whole
    chunk. Near-optimal on quasi-static segments; collapses on motion —
    beating it in aggregate is the minimum bar for a learned policy."""

    name = "state-copy"

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        return [
            item["observation.state"][None, :].expand(self.chunk_size, -1).clone()
            for item in items
        ]


class NormalizedStateCopyPolicy:
    """State-copy through the per-dataset stats: normalize the state with the
    STATE stats, unnormalize with the ACTION stats —
    ``â = μ_a + σ_a · (s − μ_s)/σ_s``, held for the whole chunk.

    On SO-100 teleop data, action (leader/commanded) and state (follower/
    measured) differ by quasi-constant offsets: gravity droop on loaded
    joints, leader-follower calibration deltas, gripper clamp force. Those
    offsets live in μ_a − μ_s, so this baseline hits offset-but-constant
    traces that raw state-copy misses — with zero learning. It is the
    correct trivial policy under per-dataset normalization, and the honest
    reference for how much of a learned model's edge is stats arithmetic."""

    name = "state-copy-norm"

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        predictions: list[Tensor] = []
        for item in items:
            normalized = (item["observation.state"] - item["state_mean"]) / item[
                "state_std"
            ]
            action = normalized * item["action_std"] + item["action_mean"]
            predictions.append(action[None, :].expand(self.chunk_size, -1).clone())
        return predictions


def sample_noise(seed: int, shape: tuple[int, ...]) -> Tensor:
    """Seeded on CPU so values are identical regardless of device."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


class BijouPolicy:
    """A bijou training checkpoint. Normalization is per dataset with the
    stats attached to each item (identical to training; works on held-out
    datasets because stats travel with the data, not the checkpoint)."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device,
        seed: int,
        sample_steps: int = 10,
        method: SamplingMethod = SamplingMethod.HEUN,
        expert_dtype: torch.dtype = torch.float32,
        generate: tuple[AuxField, ...] = (),
        condition_override: dict[str, str] | None = None,
        include_subgoal_condition: bool = False,
    ) -> None:
        self.name = f"bijou@{checkpoint.name.removeprefix('step_').lstrip('0') or '0'}"
        self.device = device
        self.seed = seed
        self.sample_steps = sample_steps
        self.method = method
        self.generate = generate
        self.model: BijouModel
        self.info: CheckpointInfo
        self.model, self.info = from_checkpoint(
            checkpoint,
            device=device,
            expert_dtype=expert_dtype,
        )
        is_ar_backbone = isinstance(self.model.decoder, ARBackboneDecoder)
        if generate and not is_ar_backbone:
            raise SystemExit(
                "--generate is ar_backbone-only (the request rides its "
                "prompt); this checkpoint's decoder is "
                f"{type(self.model.decoder).__name__}",
            )
        # Counterfactual conditioning (the Q3 diagnostic): force given
        # fields to a value regardless of the items' hindsight labels.
        # Only meaningful on condition-trained checkpoints — loud
        # otherwise (the --aux-mode free precedent).
        self.condition_override = condition_override or {}
        unknown = set(self.condition_override) - set(self.info.condition_fields)
        if unknown:
            raise SystemExit(
                f"--condition-override for {sorted(unknown)}, but the "
                f"checkpoint trained condition fields "
                f"{list(self.info.condition_fields) or 'NONE'} — overriding "
                "an untrained field would render text the model never saw",
            )
        self.collator = Collator(
            inputs=self.model.encoder.inputs_collator(),
            instruction=None,
            camera_filter=None,
            max_cameras=None,
            # Inference never tokenizes actions — AR decoding needs only the
            # batch quantiles; CE eval during training uses the train
            # collator, which does carry the codec.
            action_codec=None,
            aux=None,
            # ar_backbone prompts always carry [generate|…] (the
            # request is the caller's ask; () = the fast path and must
            # match the decode's ``generate`` — same tuple, one
            # source). Other decoders render it iff training did
            # (info.generate_bracket — the --prompt-generate-bracket
            # record; inference reproduces the training prompt).
            generate_bracket=is_ar_backbone or self.info.generate_bracket,
            generate_override=generate if is_ar_backbone else None,
            # Kinds travel with the items (StatsAttachedDataset attaches
            # them; rollout items carry an explicit map); never dropped
            # at inference — dropout is a train-time regularizer. Same
            # for instruction augmentation: inference scores/serves the
            # instruction it was GIVEN. Conditioning renders the
            # HINDSIGHT fields the checkpoint trained from each item's
            # TRUE labels (dropout 0 — score against truth ⇒ condition
            # on truth); the SUBGOAL hint is an operator INPUT, not a
            # hindsight label — eval runs the deployment-default
            # (planner-less) context unless an item carries an explicit
            # condition_subgoal (rollout --subgoal / condition_override).
            camera_kind_dropout=0.0,
            instruction_augment=0.0,
            condition_fields=tuple(
                ConditionField(f)
                for f in self.info.condition_fields
                if ConditionField(f) is not ConditionField.SUBGOAL
                or include_subgoal_condition
            ),
            condition_dropout=0.0,
            subgoal_condition_dropout=0.0,
        )

    @property
    def aux_fields(self) -> tuple[AuxField, ...]:
        """The aux fields this checkpoint trained (empty for aux-less /
        non-ar_backbone) — what a narrated pass can request."""
        decoder = self.model.decoder
        if isinstance(decoder, ARBackboneDecoder) and decoder.config.aux is not None:
            return decoder.config.aux.fields
        return ()

    def apply_overrides(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The counterfactual-conditioning item rewrite (shared with the
        narrated pass so both decode under identical conditioning)."""
        if not self.condition_override:
            return items
        return [
            {
                **item,
                **{
                    f"condition_{field}": value
                    for field, value in self.condition_override.items()
                },
            }
            for item in items
        ]

    @torch.no_grad()
    def predict_with_text(
        self,
        items: list[dict[str, Any]],
        indices: list[int],
    ) -> tuple[list[Tensor], list[AuxGeneration] | None]:
        """Chunks plus the decode's aux generations — one per item for
        ar_backbone (empty-text on the fast path), None for decoders
        that generate no text. Rollout prints these live; eval's
        ChunkPolicy protocol keeps consuming ``predict``."""
        items = self.apply_overrides(items)
        batch = self.collator(items).to(self.device)
        # Flow integrates from per-item seeded noise (deterministic and
        # batch-composition-independent); AR decodes greedily and takes
        # none.
        noise: Tensor | None = None
        if isinstance(self.model.decoder, FlowDecoder):
            chunk = batch.actions.shape[1]
            noise = torch.stack(
                [
                    sample_noise(self.seed + index, (chunk, batch.actions.shape[2]))
                    for index in indices
                ],
            ).to(self.device)
        prediction = self.model.predict_chunk(
            batch,
            noise=noise,
            num_steps=self.sample_steps,
            method=self.method,
            generate=self.generate,
        )
        return [chunk.cpu() for chunk in prediction.actions], prediction.generations

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        chunks, _generations = self.predict_with_text(items, indices)
        return chunks


class NarratedBijouPolicy:
    """The all-fields pass of an already-loaded BijouPolicy: same model,
    same conditioning, but the prompt requests every trained aux field
    (``[generate|… actions]``) and the actions follow the model's OWN
    generated value lines. Slots into the runner as one more policy —
    its chunk MAE vs the base policy IS the full-sample-size
    does-narration-help comparison — and retains per-frame generations
    (``self.generations[index]``) for the aux metrics (holding/progress
    vs weak labels) and the report blocks. No second model load: shares
    ``base.model``; the collator is a generate_override clone."""

    def __init__(self, base: BijouPolicy) -> None:
        fields = base.aux_fields
        if not fields:
            raise SystemExit(
                "narrated pass on a checkpoint without trained aux fields "
                "— nothing to request",
            )
        self.base = base
        self.fields = fields
        self.name = f"{base.name}+fields"
        self.collator = dataclasses.replace(
            base.collator,
            generate_override=fields,
        )
        self.generations: dict[int, AuxGeneration] = {}

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        items = self.base.apply_overrides(items)
        batch = self.collator(items).to(self.base.device)
        prediction = self.base.model.predict_chunk(
            batch,
            num_steps=self.base.sample_steps,
            method=self.base.method,
            generate=self.fields,
        )
        assert prediction.generations is not None  # ar_backbone always generates
        for index, generation in zip(indices, prediction.generations, strict=True):
            self.generations[index] = generation
        return [chunk.cpu() for chunk in prediction.actions]
