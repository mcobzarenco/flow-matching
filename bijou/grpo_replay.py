"""GRPO serving + replay for the MolmoAct2 discrete head.

The RL loop (``sim/grpo_loop.py``) needs three things from a policy:
a rollout decode that records its own scoring surface, a
teacher-forced replay that re-scores stored rows under the live
weights, and the grammar-mask arithmetic that keeps the two on the
SAME masked softmax. This module provides all three:

- serving: :class:`MolmoAct2DiscreteStack` — a thin adapter over the
  loaded molmoact2 family (``bijou.loading.load_vla``): the ar and
  joint families carry the discrete decoder already (their recorded
  format-6 section); the flow family (release-class conversions)
  gets it derived from the recorded molmo_flow section's geometry.
  The adapter exposes the attribute surface the loop's
  freeze/anchor/row-span machinery reads (``trunk``, ``fast_codec``,
  ``action_token_start_id``, ``metadata``, ``action_stats``…);
- rollout decode: ``predict_action_discrete`` = the family's
  grammar-masked :meth:`~bijou.vla.ARVLA.predict_ar` with keyed-Gumbel
  sampling and the [B, 2048] block-relative capture surface
  (``token_rows_from_capture`` consumes it unchanged);
- replay: one teacher-forced suffix forward per row through the SAME
  decoder (``ARSuffixDecoder.forward`` — position t of
  ``[<action_start>, bins…]`` predicts bin t), block columns reduced
  under the bins-recomputed grammar mask at the sampling temperature.

The row NPZ format (``TrainingRowWriter`` output — frames + sampled
bins + per-token logprobs + packed masks + MODEL-unit state) and the
loop's ``step_NNNN.pt`` checkpoints are FROZEN artifact formats:
banked rows stay readable forever, and this module's functional
surface (``load_training_rows``, ``grammar_masks_from_bins``,
``verify_recorded_masks``, ``replay_logprobs``,
``molmoact2_grpo_sums`` / ``_loss``) is what the loop consumes.

Numerics: replay-vs-rollout logprob agreement assumes the TEXT stack
runs fp32 for both (the loop's ``stack.trunk.text.float()``
convention) — under a bf16 trunk the teacher-forced-vs-incremental
comparison sits at the batch-shape reduction-order floor instead
(see probes/probe_molmoact2_ar_parity.py's fp32 diagnostic).

RNG: sampled draws consume one full-vocabulary-width Gumbel vector
per step (the scaffold's device-agnostic scheme). Greedy decodes and
the masked softmax itself are unaffected; two policies drawing from
the same seed agree only if they consume identically, so runs are
never resumed across policy-stack changes — banked-row replay is safe
regardless (rows carry their bins and π_old).
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

from .checkpoint import read_metadata, tokenizer_directory
from .data import DatasetStats
from .fast.molmoact2 import QuantileStats
from .loading import (
    MOLMOACT2_FAST_TOKENIZER_REF,
    MolmoFlowDecoderConfig,
    build_molmoact2_ar_decoder,
    load_vla,
    molmoact2_ar_config_from_flow_section,
    parse_decoder_config,
)
from .modelling.decoders.ar_molmoact2 import MolmoAct2ARDecoder
from .modelling.encoders.molmo2 import Molmo2Memory
from .modelling.encoders.molmoact2 import MolmoAct2InputsCollator
from .modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    CameraFrame,
    CollatedBatch,
    NormStats,
    PromptInputs,
)
from .modelling.molmo2.config import Molmo2Config
from .models.molmoact2_ar import MolmoAct2ARVLA
from .models.molmoact2_flow import MolmoAct2FlowVLA, molmoact2_prompt_of
from .models.molmoact2_joint import MolmoAct2JointVLA
from .models.objectives import ARObjective
from .models.serving import ARServing
from .train_grpo import GRPOConfig, GRPOStats, grpo_objective_sums


@dataclass(frozen=True, slots=True)
class DiscreteActionResult:
    """One rollout decode's full record: the executed chunk plus the
    raw emission the RL instruments consume. ``masked_violations`` is
    always None — the decode is grammar-masked only, so there is no
    unconstrained argmax to diverge from."""

    actions: Tensor  # [1, n_action_steps, action_dim] fp32 CPU
    token_ids: Tensor  # [1, K] long CPU — [<action_start>, bins…, <action_end>]
    bins: list[int]
    masked_violations: int | None = None


class MolmoAct2DiscreteStack:
    """The GRPO loop's policy object: a thin adapter over the loaded
    molmoact2 family, exposing serving plus the duck-typed attribute
    surface the loop's freeze/anchor/row-span machinery reads (see the
    module docstring)."""

    def __init__(
        self,
        vla: MolmoAct2ARVLA | MolmoAct2JointVLA,
        stats: DatasetStats,
    ) -> None:
        self.vla = vla
        self.decoder: MolmoAct2ARDecoder = vla.ar_decoder
        self.encoder = vla.encoder
        self.collator: MolmoAct2InputsCollator = vla.encoder.inputs_collator()  # type: ignore[assignment]  # the molmoact2 encoder's collator by construction
        # The surface the loop's freeze/anchor/row-span machinery reads
        # (duck-typed over `.trunk` etc.).
        self.trunk = vla.backbone
        self.fast_codec = self.decoder.codec.tokenizer  # type: ignore[union-attr]  # MolmoAct2ActionCodec by construction
        self.action_token_start_id: int = self.decoder.config.block_base
        self.action_start_token_id: int = self.decoder.config.block_base - 2
        self.action_end_token_id: int = self.decoder.config.block_base - 1
        self.metadata: dict[str, Any] = {
            "action_horizon": self.decoder.config.chunk_size,
            "n_action_steps": self.decoder.config.chunk_size,
        }
        assert stats.action_q01 is not None  # family loaders guard
        assert stats.action_q99 is not None
        assert stats.state_q01 is not None
        assert stats.state_q99 is not None
        self.action_stats = QuantileStats(
            q01=torch.tensor(stats.action_q01, dtype=torch.float32),
            q99=torch.tensor(stats.action_q99, dtype=torch.float32),
        )
        self.state_stats = QuantileStats(
            q01=torch.tensor(stats.state_q01, dtype=torch.float32),
            q99=torch.tensor(stats.state_q99, dtype=torch.float32),
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
        """The AR read of a bijou molmoact2-family checkpoint, loaded
        through the family registry (``load_vla``): the ar and joint
        families carry the discrete decoder already (their recorded
        format-6 section, ``fast_tokenizer`` unused); the flow family
        (release-class conversions record no format-6 section) derives
        it — geometry from the recorded molmo_flow section, block ids
        from the trunk tokenizer, block width from ``fast_tokenizer``."""
        checkpoint = Path(checkpoint)
        metadata = read_metadata(checkpoint)
        vla = load_vla(checkpoint, device=device, dtype=dtype)
        if isinstance(vla, MolmoAct2ARVLA | MolmoAct2JointVLA):
            return cls(vla, metadata.stats)
        if isinstance(vla, MolmoAct2FlowVLA):
            prompt = molmoact2_prompt_of(metadata)
            section = parse_decoder_config(
                dict(metadata.components["flow_decoder"]["config"]),
            )
            assert isinstance(section, MolmoFlowDecoderConfig)  # the family parsed it
            tokenizer_dir = tokenizer_directory(checkpoint)
            config = molmoact2_ar_config_from_flow_section(
                section,
                prompt,
                str(tokenizer_dir),
                fast_tokenizer=fast_tokenizer,
            )
            ar = MolmoAct2ARVLA(
                vla.backbone,
                vla.encoder,
                build_molmoact2_ar_decoder(
                    config,
                    prompt,
                    Molmo2Config.from_dict(metadata.backbone_config).text,
                    str(tokenizer_dir),
                ),
                # The flow family already built the ONE merged table —
                # the AR read detokenizes under the same rows.
                action_quantiles=vla.action_quantiles,
                # Adapter construction facts, not checkpoint records:
                # the AR read decodes greedily unless sampled (the unit
                # ARServing) and format 6 has no aux for the weight to
                # mix — both inert for serving and replay.
                objective=ARObjective(narration_weight=1.0),
                serving=ARServing(),
            )
            ar.eval()
            return cls(ar, metadata.stats)
        raise SystemExit(
            f"{checkpoint} records family {vla.spec.family.value!r} — the "
            "discrete stack serves the MolmoAct2 trunk's AR read "
            "(molmoact2_flow/ar/joint checkpoints)",
        )

    def encode(self, inputs: Any, *, with_grad: bool) -> Molmo2Memory:
        """Prefix-encode collated inputs against the trunk (the memory
        carries the cache the suffix decoder consumes) — the ONE
        encode both the rollout decode and the teacher-forced replay
        ride, so the two sides cannot drift."""
        return self.encoder.encode(
            self.trunk,
            inputs,
            with_grad=with_grad,
        )

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
        normalized = self.state_stats.normalize(
            torch.as_tensor(state, dtype=torch.float32),
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
        head: ``<action_start>`` fed, bins decoded under the
        symbol-budget grammar mask (greedy, or keyed-Gumbel sampled at
        ``temperature``), raw units via the checkpoint's merged table.
        Masked-only — ``grammar_masked`` must be True (an unconstrained
        full-vocabulary decode has a zeros-fallback failure mode this
        stack deliberately does not implement)."""
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
        capture: list[ActionCaptureStep] = (
            action_capture if action_capture is not None else []
        )
        sampling: ARSampling | None = None
        if temperature is not None:
            assert sample_rng is not None  # guarded above
            sampling = ARSampling(temperature=temperature, rngs=(sample_rng,))
        # The trait decode (no_grad by contract): prompt encode, BOA
        # forced, grammar-masked block decode — greedy at sampling=None.
        prediction = self.vla.predict_ar(batch, sampling=sampling, capture=capture)
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


# --- the frozen row format ----------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """One stored rollout predict, decoded back from the writer's NPZ
    (a FROZEN artifact format): the two frames (JPEG-decoded — a lossy
    leg the replay bounds account for), the MODEL-unit state the
    policy consumed, and the token surface — block-relative bins, the
    rollout's chosen logprobs under the decode's own masked softmax,
    bit-packed legality masks."""

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
    alone — pure budget arithmetic over the codec's symbol lengths.
    The decode's mask at step ``t`` is a function of the bin prefix
    (legal ids are trained rows whose expansion fits the remaining
    T×D budget), so a row's masks are exactly recomputable; loud on
    streams that cannot be a real masked decode's output."""
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
        memory = stack.encode(inputs, with_grad=True)
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
    """Sum-form replay step for a row batch: guards (every row sampled
    at ``config.temperature``; recorded masks reproduce bit-for-bit
    from bins), the teacher-forced forward, then the decoder-generic
    clipped surrogate — (objective SUM with graph, decision count,
    detached stats). The caller owns normalization: chunked backward
    divides each chunk's sum by the FULL-batch token count, so
    chunking never changes the gradient."""
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
