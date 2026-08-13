"""MolmoAct2 token-GRPO replay collator (phase-2 instrument item 3,
retargeted to the molmoact2 discrete surface per the owner steering of
2026-08-13 10:02Z; design memo posts/2026-08-13-token-grpo-phase2-design.md
§8 item 3).

The rollout side (``predict_action_discrete(grammar_masked=True,
action_capture=...)``) records per bin step the pre-mask block logits,
the applied legality mask and the chosen backbone id;
``token_rows_from_capture`` reduces those to TokenRow records and the
driver's ``TrainingRowWriter`` persists them with the two observation
frames and the MODEL-unit state vector. This module is the trainer
side: it rebuilds the decode's inputs from a stored row, runs ONE
teacher-forced trunk forward over ``prompt + [<action_start>] + bins``
(same packing, same multimodal mask — suffix positions causal), and
reduces the block columns under the grammar mask RECOMPUTED from the
bins alone — the same masked softmax the decode sampled from, so at an
unchanged policy every ratio is 1 to reduction-shape noise only (the
§8 amended bound: one-shot forward vs the decode's incremental cache
feeding).

Contracts pinned by tests/test_molmoact2_replay.py:

- the mask at step ``t`` is budget arithmetic over the bin prefix — a
  pure function of the recorded ids — and must land bit-for-bit on the
  rollout's packbits surface (both directions of item 2's "train-time
  grammar mask == rollout mask");
- replayed chosen logprobs reproduce the rollout's recorded logprobs
  within the registered reduction-noise bound on the CPU fixture;
- glued into :func:`bijou.train_grpo.grpo_objective_sums`, a fresh
  policy shows ratio ≈ 1, clip fraction 0, k3 ≈ 0.

Batch discipline: rows forward one at a time (the rollout's batch-1
prompt packing, exactly reproduced) and pad into ``[B, T]`` tensors —
correctness first; packed batching is a measured-pace decision for the
loop harness (item 4) if the training forward, not the rollout, turns
out to gate the ladder.
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

from ..molmo2.model import build_multimodal_mask
from ..train_grpo import GRPOConfig, GRPOStats, grpo_objective_sums
from .predictor import MolmoAct2Predictor


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """One stored rollout predict, decoded back from the writer's NPZ:
    the two frames (JPEG-decoded — the registered lossy budget), the
    MODEL-unit state the predictor consumed (the driver applies the
    official shim BEFORE writing), and the TokenRow surface (codec-space
    bins, rollout chosen logprobs under the decode's own masked softmax,
    bit-packed legality masks)."""

    top: np.ndarray  # [H, W, 3] uint8
    wrist: np.ndarray  # [H, W, 3] uint8
    state: Tensor  # [D] float32, MODEL units
    ids: np.ndarray  # [T] int64 — codec space
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
    predictor: MolmoAct2Predictor,
    bins: list[int],
) -> Tensor:
    """[T, block_vocab] bool legality masks recomputed from the bins
    alone — the trainer's half of "train-time grammar mask == rollout
    mask". The decode's mask at step ``t`` is budget arithmetic over
    the bin prefix: legal ids are the trained BPE rows whose symbol
    expansion fits the remaining T×D budget. Every recorded step is a
    decision (the masked decode stops at budget 0), so there is no
    decisions vector to return — a row's masks are exactly its bins'
    steps. Loud on streams that do not consume the budget exactly or
    carry zero-length (untrained) bins: those are not a real masked
    decode's output."""
    codec = predictor.fast_codec
    if codec is None:
        raise ValueError("no FAST codec attached — load(..., fast_tokenizer=...)")
    horizon = int(predictor.metadata.get("action_horizon") or 0) or (
        predictor.max_action_horizon
    )
    total = horizon * int(predictor.action_stats.q01.numel())
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
    predictor: MolmoAct2Predictor,
    row: ReplayRow,
) -> None:
    """Bit-equality of the row's packbits mask surface against the
    bins-only recomputation — the loud pre-training guard (a divergence
    means the rollout and trainer would score different distributions;
    stop, never silently continue)."""
    codec = predictor.fast_codec
    assert codec is not None  # grammar_masks_from_bins guards first
    if row.vocab_total != codec.block_vocab:
        raise ValueError(
            f"row vocab_total {row.vocab_total} != codec block width "
            f"{codec.block_vocab} — artifact mismatch",
        )
    recomputed = grammar_masks_from_bins(predictor, [int(i) for i in row.ids])
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
    predictor: MolmoAct2Predictor,
    rows: list[ReplayRow],
    *,
    task: str,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """(new chosen logprobs [B, T] fp32 WITH graph, decision mask
    [B, T] bool — True on a row's real steps, False on pad) — the
    teacher-forced training forward. Per row: the rollout's exact
    prompt packing (``batch_inputs`` over the stored frames/state),
    suffix ``[<action_start>] + bins`` appended, one one-shot trunk
    forward under the multimodal mask (suffix positions causal, same
    positions the decode's incremental feeding used), block columns at
    the bin-decision positions reduced under the recomputed grammar
    mask: ``log_softmax((block/temperature) | mask)`` — the decode's
    own distribution. Runs WITH grad (the phase-2 trainable surface is
    trunk rows); wrap in ``torch.no_grad()`` for scoring."""
    if not rows:
        raise ValueError("replay_logprobs on an empty row batch")
    base = predictor.action_token_start_id
    codec = predictor.fast_codec
    if base is None or codec is None or predictor.action_start_token_id is None:
        raise ValueError(
            "replay needs the discrete resources (action_token_start_id, "
            "fast codec, action_start id) on the predictor",
        )
    lm_head = predictor.trunk.text.lm_head
    assert lm_head is not None
    device = predictor.device
    per_row: list[Tensor] = []
    for row in rows:
        bins = [int(i) for i in row.ids]
        allowed = grammar_masks_from_bins(predictor, bins).to(device)
        inputs = predictor.batch_inputs([row.top, row.wrist], task, row.state)
        prompt_length = int(inputs["input_ids"].shape[1])
        suffix = torch.tensor(
            [predictor.action_start_token_id, *[base + b for b in bins]],
            dtype=torch.long,
            device=device,
        )[None]
        full_ids = torch.cat([inputs["input_ids"], suffix], dim=1)
        embeds = predictor.trunk.build_input_embeddings(
            full_ids,
            crops=inputs["crops"],
            pooled_patches_idx=inputs["pooled_patches_idx"],
        )
        type_mask = torch.cat(
            [
                inputs["image_type_mask"],
                torch.zeros_like(suffix, dtype=torch.bool),
            ],
            dim=1,
        )
        mask = build_multimodal_mask(
            image_type_mask=type_mask,
            padding_mask=None,
            dtype=embeds.dtype,
            device=embeds.device,
        )
        hidden = predictor.trunk.text.transformer(
            inputs_embeds=embeds,
            attention_mask=mask,
        )
        # Position prompt_length + t (the opener for t = 0, bin t−1
        # after) predicts bin t — the decode's incremental positions
        # exactly.
        block = lm_head(hidden[:, prompt_length : prompt_length + len(bins)])[
            ...,
            base : base + codec.block_vocab,
        ]
        logprobs = (
            (block.float() / temperature)
            .masked_fill(~allowed[None], float("-inf"))
            .log_softmax(-1)
        )
        chosen = torch.tensor(bins, dtype=torch.long, device=device)
        per_row.append(logprobs[0].gather(-1, chosen[:, None]).squeeze(-1))
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
    predictor: MolmoAct2Predictor,
    rows: list[ReplayRow],
    *,
    task: str,
    advantages: Tensor,
    config: GRPOConfig,
) -> tuple[Tensor, Tensor, GRPOStats]:
    """Sum-form replay step for a row batch: guards (every row sampled
    at ``config.temperature``; recorded masks reproduce bit-for-bit
    from bins), teacher-forced new logprobs, then the decoder-generic
    clipped surrogate — (objective SUM with graph, decision count,
    detached stats). The caller owns normalization: the loop harness's
    chunked backward divides each chunk's sum by the FULL-batch token
    count (the ar_backbone_loss_sums discipline), so chunking never
    changes the gradient."""
    for row in rows:
        if float(row.temperature) != config.temperature:
            raise ValueError(
                f"row (seed {row.seed} draw {row.draw} replan "
                f"{row.replan}) sampled at T={row.temperature}, config "
                f"trains at T={config.temperature} — the ratio is only "
                "π_new/π_old under the SAME masked softmax",
            )
        verify_recorded_masks(predictor, row)
    new_logprobs, decisions = replay_logprobs(
        predictor,
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
    )


def molmoact2_grpo_loss(
    predictor: MolmoAct2Predictor,
    rows: list[ReplayRow],
    *,
    task: str,
    advantages: Tensor,
    config: GRPOConfig,
) -> tuple[Tensor, GRPOStats]:
    """Single-batch form of :func:`molmoact2_grpo_sums` — the NEGATED
    token-weighted mean, ``advantages`` one scalar per row."""
    objective_sum, count, stats = molmoact2_grpo_sums(
        predictor,
        rows,
        task=task,
        advantages=advantages,
        config=config,
    )
    return -(objective_sum / count.clamp(min=1)), stats
