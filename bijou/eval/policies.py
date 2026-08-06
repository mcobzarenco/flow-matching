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
import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np
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


def mask_state_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """State-reliance probe rewrite: replace each item's proprioceptive
    state with its dataset's state mean, so the normalized soft state
    token collates to EXACTLY zero (x − x ≡ 0 bitwise) — the prompt
    stays well-formed but carries zero state information. Items are
    rebuilt, never mutated: baseline policies (state-copy is the
    intact-state reference) and the truth actions see the originals."""
    return [{**item, "observation.state": item["state_mean"].clone()} for item in items]


def sample_noise(seed: int, shape: tuple[int, ...]) -> Tensor:
    """Seeded on CPU so values are identical regardless of device."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


# Draw d of item i seeds sample_noise(seed + i + d·STRIDE): draw 0 is
# byte-identical to the historical single-draw path (paired
# comparisons against every prior flow eval stay valid), and the
# stride sits above any frame index (~2e7 on curated-v0), so no draw
# of one item can reuse another item's draw-0 noise. 2**26, NOT a
# 64-bit stride: torch's CPU Generator.manual_seed ignores bits ≥32
# (measured 2026-08-05 — manual_seed(s) == manual_seed(s + 2**32)
# stream-for-stream), so a wider stride silently collapses every draw
# to draw 0; the tripwire test would catch a recurrence.
DRAW_SEED_STRIDE = 2**26


def draw_noise(seed: int, index: int, draw: int, shape: tuple[int, ...]) -> Tensor:
    """Per-(item, draw) flow noise — deterministic, batch-composition-
    independent, draw 0 identical to the single-draw convention."""
    return sample_noise(seed + index + draw * DRAW_SEED_STRIDE, shape)


# "index" keys noise to the corpus-relative concat index, so ANY change
# to eval-corpus composition (a dataset added, removed, or regrown)
# silently redraws every flow policy's noise downstream of the edit —
# flow numbers are only comparable at frozen corpus composition.
# "stable" keys each draw to the frame's identity triple
# (repo_id, episode_index, frame_index): blake2b of the triple feeds a
# numpy SeedSequence together with the run seed and draw number, giving
# 128-bit keying — no birthday collisions across the panel, and no
# torch manual_seed involved (its CPU generator ignores seed bits ≥32,
# which is what forced DRAW_SEED_STRIDE above to stay narrow).
# Flipping the default is a versioned instrument break (posted
# amendment + re-banked flow anchors); until that executes, "index"
# stays the default and is byte-identical to the historical path.
NOISE_KEYS = ("index", "stable")


def stable_noise(
    seed: int,
    repo_id: str,
    episode_index: int,
    frame_index: int,
    draw: int,
    shape: tuple[int, ...],
) -> Tensor:
    """Per-(frame-identity, draw) flow noise — invariant to corpus
    composition, evaluation order, batch composition and device."""
    digest = hashlib.blake2b(
        f"{repo_id}\x1f{episode_index}\x1f{frame_index}".encode(),
        digest_size=16,
    ).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]
    sequence = np.random.SeedSequence([seed, draw, *words])
    values = np.random.Generator(np.random.PCG64(sequence)).standard_normal(
        shape,
        dtype=np.float32,
    )
    return torch.from_numpy(values)


def noise_for_item(
    noise_key: str,
    seed: int,
    item: dict[str, Any],
    index: int,
    draw: int,
    shape: tuple[int, ...],
) -> Tensor:
    """One frame's noise for one draw under the chosen keying scheme.

    Under "index" this is byte-identical to the historical path:
    draw 0 reproduces ``sample_noise(seed + index)`` exactly."""
    if noise_key == "stable":
        return stable_noise(
            seed,
            str(item["repo_id"]),
            int(item["episode_index"]),
            int(item["frame_index"]),
            draw,
            shape,
        )
    if noise_key != "index":
        raise ValueError(f"unknown noise key {noise_key!r} (choose from {NOISE_KEYS})")
    return draw_noise(seed, index, draw, shape)


def collapse_draws(stacked: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    """Split a [draws, batch, chunk, dim] stack into the per-item ensemble
    means (the policy's prediction — the mean commutes with the affine
    unnormalization, so raw-degree averaging is exact) and the per-item
    [draws, chunk, dim] stacks that --dump-draws persists. The mean is
    taken once, on the full stack, so dumping cannot perturb the
    prediction path."""
    means = [row.cpu() for row in stacked.mean(dim=0)]
    per_item = [row.cpu() for row in stacked.permute(1, 0, 2, 3)]
    return means, per_item


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
        sample_draws: int = 1,
        target_time: float | None = None,
        expert_dtype: torch.dtype = torch.float32,
        generate: tuple[AuxField, ...] = (),
        condition_override: dict[str, str] | None = None,
        include_subgoal_condition: bool = False,
        offload_ple: bool = False,
        noise_key: str = "index",
        mask_state: bool = False,
    ) -> None:
        self.name = f"bijou@{checkpoint.name.removeprefix('step_').lstrip('0') or '0'}"
        if sample_draws > 1:
            # The name carries the draw count: an ensembled number must
            # never be mistakable for a deployment-class read in a
            # report or ledger row (charter §2 budget classes).
            self.name += f"_draws{sample_draws}"
        if mask_state:
            # Same convention: a state-blind diagnostic read must never
            # be mistakable for a deployment read.
            self.name += "_state-masked"
        self.mask_state = mask_state
        self.device = device
        self.seed = seed
        self.sample_steps = sample_steps
        self.method = method
        self.sample_draws = sample_draws
        # SnapFlow shortcut conditioning s (None = standard s=t): passed
        # to every flow solver forward; validated against the checkpoint
        # below (needs φ_s).
        self.target_time = target_time
        # Per-item [draws, chunk, dim] stacks from the LAST ensembled
        # batch (--dump-draws reads them right after predict; one batch
        # of chunks, so retention is trivial). None until the first
        # ensembled batch — and always None at draws=1.
        self.last_draws: list[Tensor] | None = None
        self.generate = generate
        if noise_key not in NOISE_KEYS:
            raise SystemExit(
                f"--noise-key must be one of {NOISE_KEYS}, got {noise_key!r}",
            )
        self.noise_key = noise_key
        self.model: BijouModel
        self.info: CheckpointInfo
        self.model, self.info = from_checkpoint(
            checkpoint,
            device=device,
            expert_dtype=expert_dtype,
            offload_ple=offload_ple,
        )
        is_ar_backbone = isinstance(self.model.decoder, ARBackboneDecoder)
        if generate and not is_ar_backbone:
            raise SystemExit(
                "--generate is ar_backbone-only (the request rides its "
                "prompt); this checkpoint's decoder is "
                f"{type(self.model.decoder).__name__}",
            )
        if sample_draws < 1:
            raise SystemExit(f"--sample-draws must be >= 1, got {sample_draws}")
        if sample_draws > 1 and not isinstance(self.model.decoder, FlowDecoder):
            raise SystemExit(
                "--sample-draws > 1 averages flow noise draws; this "
                f"checkpoint's decoder is {type(self.model.decoder).__name__} "
                "(greedy AR decode has no noise to draw)",
            )
        if target_time is not None:
            decoder_module = self.model.decoder
            if not isinstance(decoder_module, FlowDecoder):
                raise SystemExit(
                    "--target-time zero drives the flow shortcut field; "
                    "this checkpoint's decoder is "
                    f"{type(decoder_module).__name__}",
                )
            if not decoder_module.config.target_time_embed:
                raise SystemExit(
                    "--target-time zero requires a φ_s-extended checkpoint "
                    "(config target_time_embed); this checkpoint has none — "
                    "only SnapFlow-distilled models take shortcut reads",
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
        """The per-item input rewrite — state masking plus counterfactual
        conditioning (shared with the narrated pass so both decodes see
        identical inputs)."""
        if self.mask_state:
            items = mask_state_items(items)
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
        decoder = self.model.decoder
        if isinstance(decoder, FlowDecoder) and self.sample_draws > 1:
            # Unconstrained-class noise-draw ensembling (charter §8
            # item 1): encode the prefix ONCE, integrate one chunk per
            # draw against the shared memory, average in raw degrees
            # (unnormalization is affine, so the mean commutes with it).
            memory = self.model.encode(batch.encoder_inputs, with_grad=False)
            shape = (batch.actions.shape[1], batch.actions.shape[2])
            stacked = torch.stack(
                [
                    decoder.predict_chunk(
                        memory,
                        batch,
                        noise=torch.stack(
                            [
                                noise_for_item(
                                    self.noise_key,
                                    self.seed,
                                    item,
                                    index,
                                    draw,
                                    shape,
                                )
                                for item, index in zip(items, indices, strict=True)
                            ],
                        ).to(self.device),
                        num_steps=self.sample_steps,
                        method=self.method,
                        target_time=self.target_time,
                    ).actions
                    for draw in range(self.sample_draws)
                ],
            )
            means, self.last_draws = collapse_draws(stacked)
            return means, None
        # Flow integrates from per-item seeded noise (deterministic and
        # batch-composition-independent); AR decodes greedily and takes
        # none.
        noise: Tensor | None = None
        if isinstance(decoder, FlowDecoder):
            shape = (batch.actions.shape[1], batch.actions.shape[2])
            noise = torch.stack(
                [
                    noise_for_item(self.noise_key, self.seed, item, index, 0, shape)
                    for item, index in zip(items, indices, strict=True)
                ],
            ).to(self.device)
        prediction = self.model.predict_chunk(
            batch,
            noise=noise,
            num_steps=self.sample_steps,
            method=self.method,
            target_time=self.target_time,
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
