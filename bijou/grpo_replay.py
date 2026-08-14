"""First-class GRPO serving + replay for the MolmoAct2 discrete head —
the `bijou/molmoact2/{predictor,replay}` re-point (docs/
molmoact2-retirement.md phase 4, fontaine sign-off 2026-08-14).

**What is frozen** (decision 10): the row NPZ format (TrainingRowWriter
output — frames + sampled bins + per-token logprobs + packed masks +
MODEL-unit state) and the loop's ``step_NNNN.pt`` checkpoints. This
module reads the same rows and exposes the same functional surface the
loop consumes (``load_training_rows``, ``grammar_masks_from_bins``,
``verify_recorded_masks``, ``replay_logprobs``, ``molmoact2_grpo_sums``
/ ``_loss``), swapping the port predictor for the first-class stack:

- serving: :class:`MolmoAct2DiscreteStack` — the AR read of a BIJOU
  molmoact2-family checkpoint (release-class molmo_flow section or a
  format-6 ar/joint descendant) behind the port predictor's
  duck-typed attribute surface (``trunk``, ``fast_codec``,
  ``action_token_start_id``, ``metadata``, ``action_stats``…), so the
  loop's freeze/anchor/row-span machinery runs verbatim;
- rollout decode: ``predict_action_discrete`` = the scaffold's
  grammar-masked ``predict_chunk`` (phase-2 byte-parity-gated against
  the port's decode) with the same keyed-Gumbel sampling scheme and
  the same [B, 2048] block-relative capture surface
  (``token_rows_from_capture`` consumes it unchanged);
- replay: one teacher-forced suffix forward per row through the SAME
  decoder (``ARSuffixDecoder.forward`` — position t of
  ``[<action_start>, bins…]`` predicts bin t), block columns reduced
  under the bins-recomputed grammar mask at the sampling temperature —
  fontaine's exact reduction ops.

Numerics note (registered): the replay bound (1e-5 + the JPEG budget)
was measured with the TEXT stack fp32 for both rollout and replay —
the loop's ``stack.trunk.text.float()`` convention. Mount bf16 and the
teacher-forced-vs-incremental comparison sits at the batch-shape
reduction-order floor instead (measured ≤5.6e-2 worst-step; see
probes/probe_molmoact2_ar_parity.py's fp32 diagnostic).

RNG note (decision 11): sampled draws consume full-width Gumbel
vectors per step (the scaffold's device-agnostic scheme) where the
port drew 2048 per step — bit-identical GREEDY and identical masked
softmax, different sample streams under the same seed. Replay of
banked rows is unaffected (rows carry their bins and π_old);
post-migration runs start fresh, never resume across the re-point.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .decoders.ar_backbone import ActionCaptureStep, ARSampling
from .decoders.ar_molmoact2 import MolmoAct2ARDecoder
from .encoders.molmoact2 import MolmoAct2Encoder, MolmoAct2InputsCollator
from .fast.molmoact2 import QuantileStats, normalize_state
from .interface import CameraFrame, CollatedBatch, NormStats, PromptInputs
from .loading import (
    MOLMOACT2_FAST_TOKENIZER_REF,
    ARBackboneConfig,
    CheckpointInfo,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    ar_backbone_config_from_dict,
    build_molmoact2_ar_decoder,
    checkpoint_sections,
    load_adapted_backbone,
    molmoact2_ar_config_from_flow_section,
    read_checkpoint_info,
    resolve_checkpoint_dir,
)
from .model import BijouModel
from .molmo2.loading import load_config as load_molmo2_config
from .molmo2.model import load_model as load_molmo2_model
from .train_grpo import GRPOConfig, GRPOStats, grpo_objective_sums


@dataclass(frozen=True, slots=True)
class DiscreteActionResult:
    """The rollout decode's full record — the port result's shape
    (``masked_violations`` retired with the unconstrained mode: the
    first-class decode is masked-only)."""

    actions: Tensor  # [1, n_action_steps, action_dim] fp32 CPU
    token_ids: Tensor  # [1, K] long CPU — [<action_start>, bins…, <action_end>]
    bins: list[int]
    masked_violations: int | None = None


class MolmoAct2DiscreteStack:
    """The GRPO loop's policy object: first-class serving + the port
    predictor's duck-typed attribute surface (see module docstring)."""

    def __init__(
        self,
        model: BijouModel,
        info: CheckpointInfo,
        prompt: MolmoAct2PromptConfig,
    ) -> None:
        decoder = model.decoder
        if not isinstance(decoder, MolmoAct2ARDecoder):
            raise TypeError(
                f"MolmoAct2DiscreteStack serves the discrete head; the "
                f"model carries {type(decoder).__name__}",
            )
        self.model = model
        self.info = info
        self.decoder = decoder
        self.collator: MolmoAct2InputsCollator = model.encoder.inputs_collator()  # type: ignore[assignment]  # the molmoact2 encoder's collator by construction
        # The port surface the loop's freeze/anchor/row-span machinery
        # reads (duck-typed over `.trunk` etc.).
        self.trunk = model.backbone
        self.fast_codec = decoder.codec.tokenizer  # type: ignore[union-attr]  # MolmoAct2ActionCodec by construction
        self.action_token_start_id: int = decoder.config.block_base
        self.action_start_token_id: int = decoder.config.block_base - 2
        self.action_end_token_id: int = decoder.config.block_base - 1
        self.metadata: dict[str, Any] = {
            "action_horizon": decoder.config.chunk_size,
            "n_action_steps": decoder.config.chunk_size,
        }
        normalization = info.normalization
        assert normalization.action_q01 is not None  # loading guards
        assert normalization.action_q99 is not None
        assert normalization.state_q01 is not None
        assert normalization.state_q99 is not None
        self.action_stats = QuantileStats(
            q01=torch.tensor(normalization.action_q01, dtype=torch.float32),
            q99=torch.tensor(normalization.action_q99, dtype=torch.float32),
        )
        self.state_stats = QuantileStats(
            q01=torch.tensor(normalization.state_q01, dtype=torch.float32),
            q99=torch.tensor(normalization.state_q99, dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.trunk.parameters()).device

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fast_tokenizer: str = MOLMOACT2_FAST_TOKENIZER_REF,
    ) -> MolmoAct2DiscreteStack:
        """The AR read of a BIJOU molmoact2-family checkpoint:
        release-class (molmo_flow section — the format-6 config derives
        from its geometry + the trunk tokenizer) or an ar/joint
        descendant (its own recorded section). Trunk deltas
        (``backbone.safetensors``) load when present."""
        checkpoint = Path(checkpoint)
        info = read_checkpoint_info(checkpoint)
        meta = json.loads((checkpoint / "bijou_config.json").read_text())
        sections = checkpoint_sections(meta)
        prompt = sections.prompt
        if not isinstance(prompt, MolmoAct2PromptConfig):
            raise SystemExit(
                f"{checkpoint} is not a molmoact2-family checkpoint "
                f"(prompt {type(prompt).__name__})",
            )
        if isinstance(sections.decoder, ARBackboneConfig):
            # An ar-only descendant: its own recorded format-6 section.
            config = sections.decoder
        elif isinstance(sections.decoder, MolmoFlowDecoderConfig):
            # A joint descendant records the rider's section verbatim;
            # a release-class checkpoint derives it (geometry from the
            # flow section, block_base from the trunk tokenizer).
            joint_section = meta.get("joint_ce")
            config = (
                ar_backbone_config_from_dict(joint_section)
                if joint_section is not None
                else molmoact2_ar_config_from_flow_section(
                    sections.decoder,
                    prompt,
                    info.backbone,
                    fast_tokenizer=fast_tokenizer,
                )
            )
        else:
            raise SystemExit(
                f"{checkpoint} records {type(sections.decoder).__name__} — "
                "not a molmoact2-family decoder section",
            )
        trunk_dir = resolve_checkpoint_dir(info.backbone)
        decoder = build_molmoact2_ar_decoder(
            config,
            prompt,
            load_molmo2_config(trunk_dir).text,
            info.backbone,
        )
        backbone = load_molmo2_model(trunk_dir, device=device, dtype=dtype)
        encoder = MolmoAct2Encoder(
            info.backbone,
            setup_type=prompt.setup_type,
            control_mode=prompt.control_mode,
            num_state_tokens=prompt.num_state_tokens,
            action_mode=prompt.action_mode,
            narration=prompt.narration,
        )
        model = BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)
        if (checkpoint / "backbone.safetensors").exists():
            load_adapted_backbone(model, checkpoint)
            print(f"loaded adapted backbone from {checkpoint}", flush=True)
        model.eval()
        return cls(model, info, prompt)

    def _batch(self, inputs: Any, state: Tensor) -> CollatedBatch[Any]:
        chunk = self.decoder.config.chunk_size
        dim = self.decoder.config.action_dim
        stats = NormStats(
            mean=torch.zeros(1, dim),
            std=torch.ones(1, dim),
            q01=self.action_stats.q01[None].clone(),
            q99=self.action_stats.q99[None].clone(),
        )
        return CollatedBatch(
            encoder_inputs=inputs,
            state=state[None].to(self.device),
            actions=torch.zeros(1, chunk, dim),
            action_is_pad=torch.zeros(1, chunk, dtype=torch.bool),
            action_stats=stats,
            state_stats=stats,
            action_tokens=None,
            suffix_tokens=None,
            suffix_is_aux=None,
        )

    def prompt_inputs(
        self,
        images: list[Any],
        task: str,
        state: Tensor,
    ) -> tuple[Any, Tensor]:
        """(collated inputs on device, clamp-normalized state) — the
        rollout AND replay prompt packing, one function so the two
        sides cannot drift."""
        cameras = []
        for index, image in enumerate(images):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    f"camera {index} is not [H, W, 3]-coercible: {array.shape}",
                )
            if not array.flags.writeable:
                # PIL-decoded replay frames arrive read-only;
                # torch.from_numpy warns on non-writable bases.
                array = array.copy()
            cameras.append(
                CameraFrame(
                    name=("top", "wrist")[index] if index < 2 else f"cam{index}",
                    kind="unknown",
                    image=torch.from_numpy(array).permute(2, 0, 1).float() / 255.0,
                ),
            )
        normalized = normalize_state(
            torch.as_tensor(state, dtype=torch.float32),
            self.state_stats,
        )
        inputs = self.collator(
            [
                PromptInputs(
                    instruction=task,
                    cameras=tuple(cameras),
                    condition_text="",
                    state=normalized,
                ),
            ],
        ).to(self.device)
        return inputs, normalized

    def predict_action_discrete(
        self,
        *,
        images: list[Any],
        task: str,
        state: Tensor,
        grammar_masked: bool = True,
        on_undecodable: str = "raise",
        temperature: float | None = None,
        sample_rng: np.random.Generator | None = None,
        action_capture: list[ActionCaptureStep] | None = None,
    ) -> DiscreteActionResult:
        """One observation → the executed chunk through the discrete
        head — the port's masked mode on the first-class stack (byte
        parity gated in phase 2). The unconstrained reference mode
        retired with the port; ``grammar_masked`` must be True."""
        if not grammar_masked:
            raise ValueError(
                "the unconstrained (zeros-fallback) mode retired with the "
                "port package — the first-class decode is masked-only",
            )
        if on_undecodable not in ("raise", "zeros"):
            raise ValueError(
                f"on_undecodable must be 'raise' or 'zeros', got "
                f"{on_undecodable!r} (moot either way: the masked decode "
                "decodes by construction)",
            )
        if (temperature is None) != (sample_rng is None):
            raise ValueError(
                "sampled decode takes temperature AND sample_rng together",
            )
        inputs, _ = self.prompt_inputs(images, task, torch.as_tensor(state))
        batch = self._batch(inputs, torch.as_tensor(state, dtype=torch.float32))
        with torch.no_grad():
            memory = self.model.encode(inputs, with_grad=False)
            capture: list[ActionCaptureStep] = (
                action_capture if action_capture is not None else []
            )
            if temperature is None:
                prediction = self.model.ar_predict_greedy(
                    memory,
                    batch,
                    action_capture=capture,
                )
            else:
                assert sample_rng is not None  # guarded above
                prediction = self.model.ar_predict_sampled(
                    memory,
                    batch,
                    sampling=ARSampling(temperature=temperature, rngs=(sample_rng,)),
                    action_capture=capture,
                )
        base = self.action_token_start_id
        bins = [int(step.chosen[0]) - base for step in capture if bool(step.active[0])]
        token_ids = torch.tensor(
            [
                [
                    self.action_start_token_id,
                    *(base + b for b in bins),
                    self.action_end_token_id,
                ],
            ],
            dtype=torch.long,
        )
        return DiscreteActionResult(
            actions=prediction.actions.cpu().float(),
            token_ids=token_ids,
            bins=bins,
        )


# --- the frozen row format (decision 10) --------------------------------


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """One stored rollout predict, decoded back from the writer's NPZ —
    the port replay's row, format-frozen (see its docstring)."""

    top: np.ndarray  # [H, W, 3] uint8
    wrist: np.ndarray  # [H, W, 3] uint8
    state: Tensor  # [D] float32, MODEL units
    ids: np.ndarray  # [T] int64 — codec space (block-relative bins)
    logprobs: np.ndarray  # [T] float32
    allowed_packed: np.ndarray  # [T, ceil(block_vocab/8)] uint8
    vocab_total: int
    temperature: float
    seed: int
    draw: int
    replan: int


def load_training_rows(root: Path) -> tuple[dict[str, Any], list[ReplayRow]]:
    """(meta.json dict, rows in index order) from a TrainingRowWriter
    directory. Loud on an empty index — a rows directory with no rows
    is a broken capture, not an empty batch."""
    from PIL import Image

    meta = json.loads((root / "meta.json").read_text())
    entries = [
        json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()
    ]
    if not entries:
        raise ValueError(f"empty training-rows index in {root}")
    rows: list[ReplayRow] = []
    for entry in entries:
        data = np.load(root / entry["path"])
        rows.append(
            ReplayRow(
                top=np.asarray(Image.open(io.BytesIO(data["top_jpeg"].tobytes()))),
                wrist=np.asarray(
                    Image.open(io.BytesIO(data["wrist_jpeg"].tobytes())),
                ),
                state=torch.from_numpy(data["state"]),
                ids=data["ids"],
                logprobs=data["logprobs"],
                allowed_packed=data["allowed_packed"],
                vocab_total=int(entry["vocab_total"]),
                temperature=float(entry["temperature"]),
                seed=int(entry["seed"]),
                draw=int(entry["draw"]),
                replan=int(entry["replan"]),
            ),
        )
    return meta, rows


def grammar_masks_from_bins(
    stack: MolmoAct2DiscreteStack,
    bins: list[int],
) -> Tensor:
    """[T, block_vocab] bool legality masks recomputed from the bins
    alone — pure budget arithmetic over the codec's symbol lengths
    (the port replay's contract, verbatim on the first-class codec)."""
    codec = stack.fast_codec
    total = stack.decoder.config.chunk_size * stack.decoder.config.action_dim
    lengths = torch.from_numpy(codec.symbol_lengths)
    ids = torch.tensor(bins, dtype=torch.long)
    if ids.numel() == 0:
        raise ValueError("empty bin stream — not a masked decode's output")
    step_lengths = lengths[ids]
    if bool((step_lengths == 0).any()):
        raise ValueError(
            f"zero-length (untrained) bins in the stream at steps "
            f"{(step_lengths == 0).nonzero().flatten().tolist()} — the "
            "legality mask never admits them; corrupt row",
        )
    consumed = int(step_lengths.sum())
    if consumed != total:
        raise ValueError(
            f"recorded bins consume {consumed} symbols, the masked "
            f"decode consumes exactly {total} — corrupt row",
        )
    remaining = total - (step_lengths.cumsum(dim=0) - step_lengths)
    return (lengths > 0)[None, :] & (lengths[None, :] <= remaining[:, None])


def verify_recorded_masks(
    stack: MolmoAct2DiscreteStack,
    row: ReplayRow,
) -> None:
    """Bit-equality of the row's packbits mask surface against the
    bins-only recomputation — the loud pre-training guard."""
    if row.vocab_total != stack.fast_codec.block_vocab:
        raise ValueError(
            f"row vocab_total {row.vocab_total} != codec block width "
            f"{stack.fast_codec.block_vocab} — artifact mismatch",
        )
    recomputed = grammar_masks_from_bins(stack, [int(i) for i in row.ids])
    recorded = torch.from_numpy(
        np.unpackbits(row.allowed_packed, axis=1, count=row.vocab_total).astype(
            bool,
        ),
    )
    if not torch.equal(recorded, recomputed):
        raise ValueError(
            f"recorded grammar masks diverge from the bins-only "
            f"recomputation (seed {row.seed} draw {row.draw} replan "
            f"{row.replan}) — rollout and trainer would score different "
            "distributions",
        )


def replay_logprobs(
    stack: MolmoAct2DiscreteStack,
    rows: list[ReplayRow],
    *,
    task: str,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """(new chosen logprobs [B, T] fp32 WITH graph, decision mask
    [B, T] bool) — the teacher-forced training forward on the
    first-class stack: per row, the rollout's exact prompt packing
    (collator over the stored frames + MODEL-unit state), then ONE
    suffix forward ``[<action_start>, bins…]`` through the scaffold
    (position t predicts bin t — the decode's incremental positions),
    block columns reduced under the recomputed grammar mask at the
    sampling temperature. Runs WITH grad (the trainable surface is
    trunk rows); wrap in ``torch.no_grad()`` for scoring."""
    if not rows:
        raise ValueError("replay_logprobs on an empty row batch")
    base = stack.action_token_start_id
    device = stack.device
    decoder = stack.decoder
    per_row: list[Tensor] = []
    for row in rows:
        bins = [int(i) for i in row.ids]
        allowed = grammar_masks_from_bins(stack, bins).to(device)
        inputs, _ = stack.prompt_inputs(
            [row.top, row.wrist],
            task,
            row.state,
        )
        memory = stack.model.encode(inputs, with_grad=True)
        suffix = torch.tensor(
            [[stack.action_start_token_id, *(base + b for b in bins)]],
            dtype=torch.long,
            device=device,
        )
        logits = decoder(stack.trunk, memory, suffix)
        block = logits[0, : len(bins), base : base + row.vocab_total]
        logprobs = (
            (block.float() / temperature)
            .masked_fill(~allowed, float("-inf"))
            .log_softmax(-1)
        )
        chosen = torch.tensor(bins, dtype=torch.long, device=device)
        per_row.append(logprobs.gather(-1, chosen[:, None]).squeeze(-1))
    width = max(int(p.shape[0]) for p in per_row)
    decisions = torch.zeros((len(per_row), width), dtype=torch.bool, device=device)
    padded: list[Tensor] = []
    for index, values in enumerate(per_row):
        decisions[index, : values.shape[0]] = True
        padded.append(
            torch.nn.functional.pad(values, (0, width - int(values.shape[0]))),
        )
    return torch.stack(padded), decisions


def molmoact2_grpo_sums(
    stack: MolmoAct2DiscreteStack,
    rows: list[ReplayRow],
    *,
    task: str,
    advantages: Tensor,
    config: GRPOConfig,
    anchor_logprobs: Tensor | None = None,
    kl_beta: float = 0.0,
) -> tuple[Tensor, Tensor, GRPOStats]:
    """Sum-form replay step — the port function's contract on the
    first-class stack (guards, teacher-forced forward, the
    decoder-generic clipped surrogate)."""
    for row in rows:
        if float(row.temperature) != config.temperature:
            raise ValueError(
                f"row (seed {row.seed} draw {row.draw} replan "
                f"{row.replan}) sampled at T={row.temperature}, config "
                f"trains at T={config.temperature} — the ratio is only "
                "π_new/π_old under the SAME masked softmax",
            )
        verify_recorded_masks(stack, row)
    new_logprobs, decisions = replay_logprobs(
        stack,
        rows,
        task=task,
        temperature=config.temperature,
    )
    old_logprobs = torch.zeros_like(new_logprobs)
    for index, row in enumerate(rows):
        old_logprobs[index, : row.logprobs.shape[0]] = torch.from_numpy(
            row.logprobs,
        )
    return grpo_objective_sums(
        new_logprobs,
        old_logprobs,
        advantages,
        decisions,
        config,
        anchor_logprobs=anchor_logprobs,
        kl_beta=kl_beta,
    )


def molmoact2_grpo_loss(
    stack: MolmoAct2DiscreteStack,
    rows: list[ReplayRow],
    *,
    task: str,
    advantages: Tensor,
    config: GRPOConfig,
) -> tuple[Tensor, GRPOStats]:
    """Single-batch form of :func:`molmoact2_grpo_sums` — the NEGATED
    token-weighted mean, ``advantages`` one scalar per row."""
    objective_sum, count, stats = molmoact2_grpo_sums(
        stack,
        rows,
        task=task,
        advantages=advantages,
        config=config,
    )
    return -(objective_sum / count.clamp(min=1)), stats
