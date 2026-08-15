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
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from ..annotations import ConditionField
from ..aux_text import AuxField, AuxGeneration, subgoal_text
from ..data import DatasetStats
from ..decoders.ar_backbone import (
    ActionCaptureStep,
    ARSampling,
    ARSuffixDecoder,
    ValueCandidate,
)
from ..decoders.flow import FlowDecoder
from ..decoders.molmo_flow import MolmoFlowDecoder
from ..interface import (
    CollatedBatch,
    Collator,
    MemoryStream,
    NormStats,
    ObservationMemory,
    mask_state_item,
)
from ..loading import CheckpointInfo, from_checkpoint, molmo_flow_state_table
from ..model import BijouModel, SamplingMethod
from .molmo_norm import ItemMaps, MolmoNorm, fit_item_maps
from .subgoal_scoring import ceiling_pick, eligible_indices, self_certainty_pick
from .subgoal_swap import SubgoalSwapMap, SwapRecord


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
    """State-reliance probe rewrite: every item through
    ``mask_state_item`` (the shared masking primitive — see its
    docstring; ``--state-dropout`` applies the same rewrite per sample
    at train time, so probe and regularizer semantics can never
    drift)."""
    return [mask_state_item(item) for item in items]


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


# AR sampled-draws RNGs reuse the stable frame-identity keying
# UNCONDITIONALLY — the instrument is new (2026-08-06), so there is no
# legacy index-keyed AR path to preserve and corpus-composition
# invariance costs nothing; --noise-key governs flow noise only. The
# domain constant separates these streams from stable_noise's (same
# frame, same seed, same draw must not replay the flow bitstream).
AR_SAMPLE_DOMAIN = 0x41525344  # "ARSD"


def stable_sample_rng(
    seed: int,
    repo_id: str,
    episode_index: int,
    frame_index: int,
    draw: int,
) -> np.random.Generator:
    """One frame's action-sampling stream for one draw — invariant to
    corpus composition, evaluation order, batch composition and device
    (the RNG is consumed CPU-side; see ARSampling). Unlike flow noise
    the consumption LENGTH is decode-dependent, so this returns the
    generator itself, not a fixed-shape tensor."""
    digest = hashlib.blake2b(
        f"{repo_id}\x1f{episode_index}\x1f{frame_index}".encode(),
        digest_size=16,
    ).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]
    sequence = np.random.SeedSequence([AR_SAMPLE_DOMAIN, seed, draw, *words])
    return np.random.Generator(np.random.PCG64(sequence))


# SDE per-STEP noise reuses the stable frame-identity keying with its
# own domain constant: the Euler–Maruyama decode consumes one ε per
# solver step ON TOP of the initial noise (which stays stable_noise's
# stream — same key, different domain, so the two must never replay
# each other). Per-item keying is what makes the batched SDE decode
# batch-composition-invariant: a shared generator would hand each item
# different ε depending on who else sits in the batch.
SDE_STEP_DOMAIN = 0x53444553  # "SDES"


def stable_sde_step_noise(
    seed: int,
    repo_id: str,
    episode_index: int,
    frame_index: int,
    draw: int,
    num_steps: int,
    shape: tuple[int, int],
) -> Tensor:
    """One frame's [num_steps, chunk, dim] SDE step noise for one draw —
    invariant to corpus composition, evaluation order, batch composition
    and device (generated CPU-side like stable_noise)."""
    digest = hashlib.blake2b(
        f"{repo_id}\x1f{episode_index}\x1f{frame_index}".encode(),
        digest_size=16,
    ).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]
    sequence = np.random.SeedSequence([SDE_STEP_DOMAIN, seed, draw, *words])
    values = np.random.Generator(np.random.PCG64(sequence)).standard_normal(
        (num_steps, *shape),
        dtype=np.float32,
    )
    return torch.from_numpy(values)


# Golden-ticket noise mode: a frozen
# bank of candidate noise vectors replaces per-frame keying entirely —
# draw d at EVERY frame integrates from tickets[d] (the defining
# ticket property), so one batched draws-M eval scores every candidate
# on every frame. The domain constant separates the bank's generation
# stream from stable_noise's and stable_sample_rng's; the middle 0 in
# [TICKET_DOMAIN, 0, m] reserves a family slot for any future bank.
TICKET_DOMAIN = 0x54434B54  # "TCKT"


def generate_tickets(count: int, shape: tuple[int, int]) -> np.ndarray:
    """The candidate bank: ticket m ~ N(0, I) float32 [chunk, dim] from
    SeedSequence [TICKET_DOMAIN, 0, m] — deterministic, generated once,
    committed, sha-pinned (tests pin both the array bytes and the
    committed file's sha256)."""
    return np.stack(
        [
            np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([TICKET_DOMAIN, 0, m])),
            ).standard_normal(shape, dtype=np.float32)
            for m in range(count)
        ],
    )


def load_tickets(path: Path) -> tuple[Tensor, str]:
    """(bank [count, chunk, dim] float32, sha256 of the file bytes).
    The sha is provenance: every ticket read quotes it — a ticket read
    must never pass as a keyed-noise read."""
    data = np.load(path, allow_pickle=False)
    if "tickets" not in data.files:
        raise SystemExit(
            f"--noise-tickets {path} carries no 'tickets' array (keys: {data.files})",
        )
    bank = data["tickets"]
    if bank.ndim != 3 or bank.dtype != np.float32:
        raise SystemExit(
            f"tickets must be float32 [count, chunk, dim], got "
            f"{bank.dtype} {tuple(bank.shape)}",
        )
    return torch.from_numpy(bank), hashlib.sha256(path.read_bytes()).hexdigest()


def load_ticket_map(path: Path, bank_count: int) -> tuple[dict[str, int], str]:
    """(dataset → bank-index routing map, sha256 of its canonical form).

    Accepts the committed stage-0/1 analysis json (map under
    ``stage1.routing_map`` — the per-dataset routing artifact) or a bare
    ``{repo_id: index}`` json. The sha is computed over
    ``json.dumps(map, sort_keys=True)`` — byte-identical to the
    committing script's — so a routed run's provenance can be checked
    against the pre-registered map sha without trusting file paths."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "stage1" in data:
        data = data["stage1"].get("routing_map")
    if not isinstance(data, dict) or not data:
        raise SystemExit(
            f"--noise-ticket-map {path} carries no routing map (expected "
            "a non-empty {repo_id: ticket index} dict, bare or under "
            "stage1.routing_map)",
        )
    mapping: dict[str, int] = {}
    for repo_id, ticket in data.items():
        if not isinstance(ticket, int) or isinstance(ticket, bool):
            raise SystemExit(
                f"--noise-ticket-map {path}: dataset {repo_id!r} routes "
                f"to non-integer ticket {ticket!r}",
            )
        if not 0 <= ticket < bank_count:
            raise SystemExit(
                f"--noise-ticket-map {path}: dataset {repo_id!r} routes "
                f"to ticket {ticket} outside the bank [0, {bank_count})",
            )
        mapping[str(repo_id)] = ticket
    blob = json.dumps(mapping, sort_keys=True).encode()
    return mapping, hashlib.sha256(blob).hexdigest()


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


def tile_memory(memory: ObservationMemory, draws: int) -> ObservationMemory:
    """Draws-major tiling of an encoded observation for batched
    ensembling: every K/V stream and the padding mask repeat along the
    batch dim ([B, …] → [draws·B, …], whole-batch-major, so row
    d·B + i is (draw d, item i) — the collapse_draws layout). A KV
    cache cannot be tiled (AR-only surface) and must be absent."""
    if memory.cache is not None:
        raise ValueError("cannot tile an ObservationMemory carrying a KV cache")
    return dataclasses.replace(
        memory,
        streams={
            name: MemoryStream(
                key=stream.key.repeat(draws, 1, 1, 1),
                value=stream.value.repeat(draws, 1, 1, 1),
            )
            for name, stream in memory.streams.items()
        },
        padding_mask=(
            memory.padding_mask.repeat(draws, 1)
            if memory.padding_mask is not None
            else None
        ),
    )


def tile_stats(batch: CollatedBatch[Any], draws: int) -> CollatedBatch[Any]:
    """The FLOW-DECODE view of a batch at draws x B: only the fields
    FlowDecoder.predict_chunk reads — ``state`` and the two stats — are
    tiled (draws-major, matching :func:`tile_memory`). Every other
    field keeps its B rows and must not be consumed at draws scale."""

    def tile(stats: NormStats) -> NormStats:
        return dataclasses.replace(
            stats,
            mean=stats.mean.repeat(draws, 1),
            std=stats.std.repeat(draws, 1),
            q01=stats.q01.repeat(draws, 1) if stats.q01 is not None else None,
            q99=stats.q99.repeat(draws, 1) if stats.q99 is not None else None,
        )

    return dataclasses.replace(
        batch,
        state=batch.state.repeat(draws, 1),
        action_stats=tile(batch.action_stats),
        state_stats=tile(batch.state_stats),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TokenRow:
    """One frame's action-token record for RL replay — the token-GRPO
    training-rows instrument. Per
    emitted symbol (active decode steps only, in emission order): the
    CODEC id (backbone id − block_base), its chosen log-probability
    under the decode's OWN sampling distribution — fp32
    ``log_softmax(block_logits / temperature)`` with illegal columns at
    −inf, i.e. exactly the masked softmax :func:`_sample_action_ids`
    drew from — and the applied grammar mask, bit-packed per step
    (``np.packbits`` over the vocab axis). Greedy decodes record
    temperature 1.0: their logprobs are the plain masked softmax a
    teacher-forced re-forward recomputes (the item-1 oracle)."""

    ids: np.ndarray  # [T] int64 — codec space
    logprobs: np.ndarray  # [T] float32
    allowed_packed: np.ndarray  # [T, ceil(vocab_total/8)] uint8
    vocab_total: int
    temperature: float


def token_rows_from_capture(
    capture: list[ActionCaptureStep],
    *,
    block_base: int,
    temperature: float | None,
) -> list[TokenRow]:
    """Reduce a decode's :class:`ActionCaptureStep` list to per-row
    :class:`TokenRow` records. Pure CPU post-processing of the captured
    surface — the decode itself is untouched (capture is observation,
    never intervention). ``temperature`` None means a greedy decode
    (recorded as 1.0). One .cpu() per step, mcselect's sync discipline —
    never per row."""
    scale = 1.0 if temperature is None else temperature
    steps = [
        (
            step.block_logits.float().cpu(),
            step.allowed.cpu(),
            step.active.cpu(),
            step.chosen.cpu(),
        )
        for step in capture
    ]
    if not steps:
        raise ValueError("token_rows_from_capture on an empty capture")
    batch_size, vocab_total = steps[0][0].shape
    rows: list[TokenRow] = []
    for row in range(batch_size):
        ids: list[int] = []
        logprobs: list[float] = []
        masks: list[np.ndarray] = []
        for logits, allowed, active, chosen in steps:
            if not bool(active[row]):
                continue
            legal = allowed[row]
            step_logprobs = (
                (logits[row] / scale).masked_fill(~legal, float("-inf")).log_softmax(-1)
            )
            codec_id = int(chosen[row]) - block_base
            ids.append(codec_id)
            logprobs.append(float(step_logprobs[codec_id]))
            masks.append(np.packbits(legal.numpy()))
        rows.append(
            TokenRow(
                ids=np.asarray(ids, dtype=np.int64),
                logprobs=np.asarray(logprobs, dtype=np.float32),
                allowed_packed=np.stack(masks),
                vocab_total=vocab_total,
                temperature=scale,
            ),
        )
    return rows


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
        ar_temperature: float | None = None,
        sde_noise_level: float | None = None,
        target_time: float | None = None,
        expert_dtype: torch.dtype = torch.float32,
        generate: tuple[AuxField, ...] = (),
        condition_override: dict[str, str] | None = None,
        include_subgoal_condition: bool = False,
        subgoal_mode: str | None = None,
        subgoal_swap: SubgoalSwapMap | None = None,
        offload_ple: bool = False,
        noise_key: str = "stable",
        tickets: Path | None = None,
        ticket_map: Path | None = None,
        mask_state: bool = False,
        molmo_norm: MolmoNorm = MolmoNorm.CHECKPOINT,
    ) -> None:
        self.name = f"bijou@{checkpoint.name.removeprefix('step_').lstrip('0') or '0'}"
        if sample_draws > 1:
            # The name carries the draw count: an ensembled number must
            # never be mistakable for a deployment-class read in a
            # report or ledger row (charter §2 budget classes).
            self.name += f"_draws{sample_draws}"
        if ar_temperature is not None:
            # Same convention: a temperature-sampled AR read is a
            # different voice from the greedy deployment decode.
            self.name += f"_t{ar_temperature:g}"
        if sde_noise_level is not None:
            # Same convention: an Euler–Maruyama SDE decode is a
            # different voice from the deterministic keyed-noise ODE
            # read (even at a=0, which reproduces euler bit-for-bit).
            self.name += f"_sde{sde_noise_level:g}"
        if mask_state:
            # Same convention: a state-blind diagnostic read must never
            # be mistakable for a deployment read.
            self.name += "_state-masked"
        self.tickets: Tensor | None = None
        self.tickets_sha256: str | None = None
        self.ticket_map: dict[str, int] | None = None
        self.ticket_map_sha256: str | None = None
        if ticket_map is not None and tickets is None:
            raise SystemExit(
                "--noise-ticket-map routes datasets INTO a ticket bank — "
                "it requires --noise-tickets",
            )
        if tickets is not None:
            # Same convention: a searched-noise read must never be
            # mistakable for a keyed-noise read (the sha rides in the
            # report/dump provenance).
            self.tickets, self.tickets_sha256 = load_tickets(tickets)
            if ticket_map is not None:
                # Per-dataset ticket routing. The
                # distinct suffix keeps a routed read from ever pooling
                # as a single-ticket (_ticket) read.
                if sample_draws != 1:
                    raise SystemExit(
                        "--noise-ticket-map is a deterministic single "
                        "decode (each frame integrates from its dataset's "
                        f"routed ticket) — --sample-draws {sample_draws} "
                        "has no routed semantics",
                    )
                self.ticket_map, self.ticket_map_sha256 = load_ticket_map(
                    ticket_map,
                    self.tickets.shape[0],
                )
                self.name += "_ticketmap"
            else:
                self.name += "_ticket"
        if subgoal_mode not in (None, "oracle", "self", "draws", "mcselect"):
            raise SystemExit(
                "subgoal_mode must be None, 'oracle', 'self', 'draws' or "
                f"'mcselect', got {subgoal_mode!r}",
            )
        if subgoal_mode is not None and "subgoal" in (condition_override or {}):
            raise SystemExit(
                "--subgoal-mode and --condition-override subgoal=… are two "
                "sources for the same prompt slot — pick one",
            )
        if subgoal_swap is not None and subgoal_mode != "oracle":
            raise SystemExit(
                "--subgoal-swap-seed rides the oracle conditioning path "
                "(the swap arm IS the oracle arm with deranged labels) — "
                f"it requires --subgoal-mode oracle, got {subgoal_mode!r}",
            )
        if subgoal_mode == "oracle":
            # Same convention (charter §2): a truth-conditioned read must
            # never pass as the planner-less deployment read — and a
            # content-wrong (or identity-check) swap read must never pass
            # as either of the other two.
            if subgoal_swap is None:
                self.name += "_oraclesubgoal"
            elif subgoal_swap.identity:
                self.name += "_swapidentity"
            else:
                self.name += "_swapsubgoal"
        self.subgoal_swap = subgoal_swap
        # Per-frame swap provenance keyed by the dataset-local identity
        # triple — --dump-subgoal-swaps serializes it (oracle iv).
        self.swap_records: dict[tuple[str, int, int], SwapRecord] = {}
        self.subgoal_mode = subgoal_mode
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
        # Training-rows instrument (--emit-training-rows): when the flag
        # is set, every AR predict retains per-item TokenRows from the
        # decode's own capture surface (same retention pattern as
        # last_draws). Guarded in predict_with_text: AR-suffix decoders
        # at sample_draws 1 only — every other path is loud.
        self.capture_token_rows = False
        self.last_token_rows: list[TokenRow] | None = None
        self.generate = generate
        # Per-frame generations under an EXPLICIT --generate (the main
        # arm narrates; no second pass runs) — the NarratedBijouPolicy
        # retention pattern, so aux metrics and --dump-generations
        # survive without re-running a narrated arm. Empty when
        # generate=() (the deployment fast path emits no value lines).
        self.generations: dict[int, AuxGeneration] = {}
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
        # The whole AR-suffix family (--decoder ar_backbone on any
        # trunk: Gemma's ARBackboneDecoder AND Molmo2ARDecoder) — the
        # Gemma-concrete isinstance here silently dropped the narrated
        # pass (and refused --generate) on molmo2 checkpoints.
        is_ar_backbone = isinstance(self.model.decoder, ARSuffixDecoder)
        if generate and not is_ar_backbone:
            raise SystemExit(
                "--generate is ar_backbone-only (the request rides its "
                "prompt); this checkpoint's decoder is "
                f"{type(self.model.decoder).__name__}",
            )
        if sample_draws < 1:
            raise SystemExit(f"--sample-draws must be >= 1, got {sample_draws}")
        if ar_temperature is not None:
            if not ar_temperature > 0:
                raise SystemExit(
                    f"--ar-temperature must be > 0, got {ar_temperature} "
                    "(the greedy read is the default, not a temperature "
                    "limit)",
                )
            if not isinstance(self.model.decoder, ARSuffixDecoder):
                raise SystemExit(
                    "--ar-temperature samples the backbone-suffix AR "
                    "action decode; this checkpoint's decoder is "
                    f"{type(self.model.decoder).__name__}",
                )
        self.ar_temperature = ar_temperature
        if sde_noise_level is not None:
            if sde_noise_level < 0:
                raise SystemExit(
                    f"--sde-noise-level must be >= 0, got {sde_noise_level} "
                    "(0 is the bit-identity-with-euler oracle setting)",
                )
            if not isinstance(self.model.decoder, FlowDecoder):
                raise SystemExit(
                    "--sde-noise-level is the Euler–Maruyama flow decode; "
                    "this checkpoint's decoder is "
                    f"{type(self.model.decoder).__name__}",
                )
            if ar_temperature is not None:
                raise SystemExit(
                    "--sde-noise-level and --ar-temperature sample different "
                    "decoder families — pick one",
                )
            if sample_draws != 1:
                raise SystemExit(
                    "--sample-draws ensembling averages ODE draws; SDE "
                    "groups ride the rollout driver's per-draw keying "
                    f"instead — got --sample-draws {sample_draws}",
                )
            if tickets is not None:
                raise SystemExit(
                    "--noise-tickets substitutes the INITIAL noise bank; "
                    "the SDE decode's step noise has no ticket semantics",
                )
            if target_time is not None:
                raise SystemExit(
                    "the SDE sampler offers no φ_s shortcut conditioning — "
                    f"--target-time must be unset, got {target_time}",
                )
            if method is not SamplingMethod.EULER:
                raise SystemExit(
                    "the SDE decode is Euler-only (a Heun corrector breaks "
                    f"the Gaussian transition) — got method {method.name}; "
                    "pass --method euler",
                )
            if noise_key != "stable":
                raise SystemExit(
                    "SDE step noise is keyed by the frame identity triple — "
                    f"it requires --noise-key stable, got {noise_key!r}",
                )
        self.sde_noise_level = sde_noise_level
        if self.tickets is not None:
            if not isinstance(self.model.decoder, FlowDecoder):
                raise SystemExit(
                    "--noise-tickets substitutes flow initial noise; this "
                    "checkpoint's decoder is "
                    f"{type(self.model.decoder).__name__} (AR decodes take "
                    "no flow noise)",
                )
            if sample_draws > self.tickets.shape[0]:
                raise SystemExit(
                    f"--sample-draws {sample_draws} > {self.tickets.shape[0]} "
                    "tickets in the bank — each draw IS one ticket, so the "
                    "draw count cannot exceed the bank",
                )
            if self.tickets.shape[1] != self.info.chunk_size:
                raise SystemExit(
                    f"tickets shaped for chunk {self.tickets.shape[1]}, "
                    f"checkpoint chunk size {self.info.chunk_size}",
                )
        self.molmo_norm = molmo_norm
        # Per-repo fitted maps (stats are per-dataset constants, so one
        # fit serves every frame of a dataset) - see molmo_norm.py for
        # the two modes' semantics.
        self._molmo_norm_maps: dict[str, ItemMaps] = {}
        if molmo_norm is not MolmoNorm.CHECKPOINT:
            if not isinstance(self.model.decoder, MolmoFlowDecoder):
                raise SystemExit(
                    "--molmo-norm re-maps a molmo_flow checkpoint's global "
                    "q01/q99 table; this checkpoint's decoder is "
                    f"{type(self.model.decoder).__name__} (per-dataset "
                    "normalization is already its native scheme)",
                )
            # Off-contract inference class: the suffix keeps these reads
            # from ever pooling with contract reads (charter §2).
            self.name += (
                "_pdnorm" if molmo_norm is MolmoNorm.PER_DATASET else "_convmap"
            )
        if sample_draws > 1 and isinstance(self.model.decoder, MolmoFlowDecoder):
            # The decode IS stochastic (flow noise), but draw ensembling
            # tiles the prefix memory and the molmo2 KV cache has no
            # tile path yet (§8.13 loose end) — refuse with the real
            # reason, not the greedy-AR message below.
            raise SystemExit(
                "--sample-draws > 1 is not implemented for molmo_flow: "
                "draw ensembling tiles the prefix memory, and the "
                "molmo2 KV cache has no tile path yet — run single-draw",
            )
        if sample_draws > 1 and not (
            isinstance(self.model.decoder, FlowDecoder) or ar_temperature is not None
        ):
            raise SystemExit(
                "--sample-draws > 1 needs a stochastic decode: flow noise "
                "draws, or an ar_backbone checkpoint with --ar-temperature; "
                f"this checkpoint's decoder is "
                f"{type(self.model.decoder).__name__} (greedy AR decode has "
                "no noise to draw)",
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
        if subgoal_mode is not None:
            # Self-subgoal probe: both modes need the trained [subgoal|…] slot;
            # the self mode additionally decodes the model's own subgoal.
            if ConditionField.SUBGOAL.value not in self.info.condition_fields:
                raise SystemExit(
                    "--subgoal-mode needs a checkpoint that trained the "
                    "[subgoal|…] condition slot; this one trained "
                    f"{list(self.info.condition_fields) or 'NONE'}",
                )
            if (
                subgoal_mode in ("self", "draws")
                and AuxField.SUBGOAL not in self.aux_fields
            ):
                raise SystemExit(
                    f"--subgoal-mode {subgoal_mode} decodes the model's OWN "
                    "subgoal; this checkpoint trained aux fields "
                    f"{[f.value for f in self.aux_fields] or 'NONE'} — no "
                    "subgoal to generate",
                )
        # The oracle mode is per-frame TRUE-label conditioning: SUBGOAL
        # joins the condition fields with no override, so each frame
        # renders its judge segment label (label-less frames render
        # nothing and decode identically to baseline). The self mode
        # keeps the base collator PLANNER-LESS — pass 2 clones it with
        # the slot added and feeds pass 1's text through the override.
        include_subgoal_condition = (
            include_subgoal_condition or subgoal_mode == "oracle"
        )
        # molmo_flow state tokens are BINNED from the checkpoint's merged
        # q01/q99 table (§8.13) — the same wiring as training's
        # collator; without it the per-sample mean/std path would silently
        # shift every state bin off its trained meaning.
        state_table = (
            molmo_flow_state_table(self.info.normalization)
            if isinstance(self.model.decoder, MolmoFlowDecoder)
            else None
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
            state_q01=state_table[0] if state_table is not None else None,
            state_q99=state_table[1] if state_table is not None else None,
        )

    @property
    def aux_fields(self) -> tuple[AuxField, ...]:
        """The aux fields this checkpoint trained (empty for aux-less /
        non-ar_backbone) — what a narrated pass can request."""
        decoder = self.model.decoder
        if isinstance(decoder, ARSuffixDecoder) and decoder.config.aux is not None:
            return decoder.config.aux.fields
        return ()

    def apply_overrides(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The per-item input rewrite — state masking plus counterfactual
        conditioning (shared with the narrated pass so both decodes see
        identical inputs)."""
        if self.mask_state:
            items = mask_state_items(items)
        if self.subgoal_swap is not None:
            items = [self._swap_subgoal(item) for item in items]
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

    def _swap_subgoal(self, item: dict[str, Any]) -> dict[str, Any]:
        """The pinned swap rewrite for one frame: label-less frames pass
        UNTOUCHED (byte-identical to baseline); labeled
        frames render the donor's fraction-matched label through the
        ``condition_subgoal`` override — or an EMPTY slot when the
        dataset contributed no swapped frames (single labeled episode:
        the slot must never fall back to the truth). Every labeled
        frame's outcome is recorded for --dump-subgoal-swaps (a
        mechanical offline check recomputes the dump from sidecars)."""
        assert self.subgoal_swap is not None
        true_label = subgoal_text(item)
        if true_label is None:
            return item
        repo_id = str(item["repo_id"])
        episode = int(item["episode_index"])
        frame = int(item["frame_index"])
        episodes = self.subgoal_swap.episodes.get(repo_id)
        if episodes is None or episode not in episodes:
            raise SystemExit(
                f"{repo_id} episode {episode}: the frame carries a "
                "materialized segment label but the swap map (built from "
                "meta/judgments.json under the dataset's own stamp) has "
                "no labeled episode for it — sidecar and materialized "
                "language rows disagree; re-materialize before swapping",
            )
        donor_index = self.subgoal_swap.donors.get(repo_id, {}).get(episode)
        text = self.subgoal_swap.donor_text(repo_id, episode, frame) or ""
        self.swap_records[(repo_id, episode, frame)] = SwapRecord(
            repo_id=repo_id,
            episode_index=episode,
            frame_index=frame,
            timestamp=float(item["timestamp"]),
            true_subgoal=true_label,
            donor_episode_index=donor_index,
            rendered_subgoal=text,
        )
        return {**item, "condition_subgoal": text}

    def _molmo_norm_rewrite(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[ItemMaps]]:
        """The off-contract normalization seam (--molmo-norm): fit each
        item's dataset -> checkpoint-table affine maps (cached per repo
        — stats are per-dataset constants), rewrite the raw state
        through A on the way IN (the collator then runs the standard
        checkpoint-table path on it), and hand back the maps so predict
        can pull chunks through A⁻¹ on the way OUT. Items are copied,
        never mutated — the runner hands the same dicts to every
        policy."""
        rewritten: list[dict[str, Any]] = []
        maps: list[ItemMaps] = []
        for item in items:
            repo_id = str(item["repo_id"])
            fitted = self._molmo_norm_maps.get(repo_id)
            if fitted is None:
                stats = DatasetStats(
                    action_mean=tuple(item["action_mean"].tolist()),
                    action_std=tuple(item["action_std"].tolist()),
                    state_mean=tuple(item["state_mean"].tolist()),
                    state_std=tuple(item["state_std"].tolist()),
                    action_q01=(
                        tuple(item["action_q01"].tolist())
                        if "action_q01" in item
                        else None
                    ),
                    action_q99=(
                        tuple(item["action_q99"].tolist())
                        if "action_q99" in item
                        else None
                    ),
                    state_q01=(
                        tuple(item["state_q01"].tolist())
                        if "state_q01" in item
                        else None
                    ),
                    state_q99=(
                        tuple(item["state_q99"].tolist())
                        if "state_q99" in item
                        else None
                    ),
                )
                fitted, fit = fit_item_maps(
                    stats,
                    self.info.normalization,
                    self.molmo_norm,
                )
                self._molmo_norm_maps[repo_id] = fitted
                if fit is not None and (any(fit.translated) or any(fit.needs_affine)):
                    # One reviewable line per dataset, first visit — the
                    # chosen convention is part of the read's provenance.
                    print(
                        f"convention-map {repo_id}: "
                        f"scale {fit.map.scale.tolist()} "
                        f"offset {fit.map.offset.tolist()} "
                        f"(identity floor {fit.identity_floor.tolist()}, "
                        f"residual {fit.snapped_floor.tolist()}"
                        + (
                            f"; UNTRANSLATABLE joints "
                            f"{[i for i, n in enumerate(fit.needs_affine) if n]}"
                            if any(fit.needs_affine)
                            else ""
                        )
                        + ")",
                        flush=True,
                    )
            maps.append(fitted)
            rewritten.append(
                {
                    **item,
                    "observation.state": fitted.state.apply(
                        item["observation.state"],
                    ),
                },
            )
        return rewritten, maps

    def _flow_noise(
        self,
        items: list[dict[str, Any]],
        indices: list[int],
        draws: int,
        shape: tuple[int, int],
    ) -> Tensor:
        """Draws-major [draws·B, chunk, dim] flow noise (row d·B + i is
        (draw d, item i) — the tile_memory/collapse_draws layout).
        Ticket mode substitutes the bank row for the per-frame keying:
        draw d at every frame IS tickets[d] — the golden-ticket oracles
        assert this on the produced tensor, not by construction."""
        if self.tickets is not None:
            if draws > self.tickets.shape[0]:
                raise SystemExit(
                    f"{draws} draws > {self.tickets.shape[0]} tickets in the bank",
                )
            if tuple(self.tickets.shape[1:]) != shape:
                raise SystemExit(
                    f"tickets shaped {tuple(self.tickets.shape[1:])}, "
                    f"batch actions shaped {shape}",
                )
            if self.ticket_map is not None:
                # Routed mode: each frame integrates from its DATASET's
                # ticket. Coverage is a hard abort — silently falling
                # back to any default would blend routed and unrouted
                # rows inside one npz.
                if draws != 1:
                    raise SystemExit(
                        f"routed ticket decode is single-draw, got {draws}",
                    )
                unmapped = sorted(
                    {
                        str(item["repo_id"])
                        for item in items
                        if str(item["repo_id"]) not in self.ticket_map
                    },
                )
                if unmapped:
                    raise SystemExit(
                        f"--noise-ticket-map covers no route for {unmapped} "
                        "— every eval dataset must appear in the map",
                    )
                return torch.stack(
                    [
                        self.tickets[self.ticket_map[str(item["repo_id"])]]
                        for item in items
                    ],
                )
            return self.tickets[:draws].repeat_interleave(len(items), dim=0)
        return torch.cat(
            [
                torch.stack(
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
                )
                for draw in range(draws)
            ],
        )

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
        self.last_token_rows = None
        if self.capture_token_rows:
            # Loud before any compute: the flag promises rows, so a
            # predict that cannot produce them must never run quietly.
            if not isinstance(self.model.decoder, ARSuffixDecoder):
                raise SystemExit(
                    "capture_token_rows needs an AR-suffix decoder (the "
                    "token stream IS the trainable surface); this "
                    "checkpoint's decoder is "
                    f"{type(self.model.decoder).__name__}",
                )
            if self.sample_draws > 1:
                raise SystemExit(
                    "capture_token_rows at --sample-draws "
                    f"{self.sample_draws}: the ensembled mean executes no "
                    "single token stream — rollout draws ride (seed, "
                    "draw) work units instead",
                )
        item_maps: list[ItemMaps] | None = None
        if self.molmo_norm is not MolmoNorm.CHECKPOINT:
            # Off-contract normalization (molmo_flow only — __init__
            # guard): state passes through A here; chunks come back
            # through A⁻¹ below. Only the plain single-decode branch is
            # reachable with the modes active (draws/temperature arms
            # are refused for molmo_flow at construction).
            items, item_maps = self._molmo_norm_rewrite(items)
        batch = self.collator(items).to(self.device)
        decoder = self.model.decoder
        if isinstance(decoder, FlowDecoder) and self.sample_draws > 1:
            # Unconstrained-class noise-draw ensembling (charter §8
            # item 1): encode the prefix ONCE, then integrate ALL draws
            # in ONE solver call at batch draws x B — draws are
            # independent batch rows, and at rollout's B=1 a sequential
            # loop leaves the GPU starved (N x num_steps tiny forwards).
            # Draws-major tiling keeps collapse_draws/--dump-draws
            # layouts unchanged. Average in raw degrees (unnormalization
            # is affine, so the mean commutes with it).
            memory = self.model.encode(batch.encoder_inputs, with_grad=False)
            shape = (batch.actions.shape[1], batch.actions.shape[2])
            noise = self._flow_noise(
                items,
                indices,
                self.sample_draws,
                shape,
            ).to(self.device)
            stacked = decoder.predict_chunk(
                tile_memory(memory, self.sample_draws),
                tile_stats(batch, self.sample_draws),
                noise=noise,
                num_steps=self.sample_steps,
                method=self.method,
                target_time=self.target_time,
            ).actions.reshape(self.sample_draws, len(items), *shape)
            means, self.last_draws = collapse_draws(stacked)
            return means, None
        if self.ar_temperature is not None and isinstance(decoder, ARSuffixDecoder):
            # AR sampled-draws instrument (the flow ensembling's
            # mirror): encode the prefix ONCE, snapshot the cache,
            # temperature-sample one chunk per draw against the restored
            # prefill, average the decoded chunks in raw units. Value
            # lines stay greedy, so generations are draw-invariant —
            # draw 0's are returned.
            memory = self.model.encode(batch.encoder_inputs, with_grad=False)
            snapshot = decoder.cache_snapshot(memory)
            draws: list[Tensor] = []
            generations: list[AuxGeneration] | None = None
            for draw in range(self.sample_draws):
                if draw:
                    decoder.cache_restore(memory, snapshot)
                sampling = ARSampling(
                    temperature=self.ar_temperature,
                    rngs=tuple(
                        stable_sample_rng(
                            self.seed,
                            str(item["repo_id"]),
                            int(item["episode_index"]),
                            int(item["frame_index"]),
                            draw,
                        )
                        for item in items
                    ),
                )
                capture: list[ActionCaptureStep] | None = (
                    [] if self.capture_token_rows else None
                )
                prediction = self.model.ar_predict_sampled(
                    memory,
                    batch,
                    generate=self.generate,
                    sampling=sampling,
                    action_capture=capture,
                )
                if capture is not None:
                    # sample_draws == 1 here (the top guard), so this
                    # single-draw retention is the whole batch's rows.
                    self.last_token_rows = token_rows_from_capture(
                        capture,
                        block_base=decoder.config.block_base,
                        temperature=self.ar_temperature,
                    )
                if generations is None:
                    generations = prediction.generations
                draws.append(prediction.actions)
            stacked = torch.stack(draws)
            if self.sample_draws > 1:
                means, self.last_draws = collapse_draws(stacked)
                return means, generations
            return [chunk.cpu() for chunk in stacked[0]], generations
        if self.sde_noise_level is not None and isinstance(decoder, FlowDecoder):
            # SDE stochastic decode (Euler–Maruyama, the trainable
            # sampler): initial noise stays the stable keyed stream;
            # per-step ε comes pre-drawn per item from its own keyed
            # domain, so the batched decode stays batch-composition-
            # invariant like every other stable stream. Draw identity
            # rides the item's repo_id (the rollout drivers' draw-suffix
            # convention), so the in-call draw index is always 0.
            memory = self.model.encode(batch.encoder_inputs, with_grad=False)
            shape = (batch.actions.shape[1], batch.actions.shape[2])
            sde_noise = self._flow_noise(items, indices, 1, shape).to(self.device)
            step_noise = torch.stack(
                [
                    stable_sde_step_noise(
                        self.seed,
                        str(item["repo_id"]),
                        int(item["episode_index"]),
                        int(item["frame_index"]),
                        0,
                        self.sample_steps,
                        shape,
                    )
                    for item in items
                ],
                dim=1,
            ).to(self.device)
            prediction = decoder.predict_chunk(
                memory,
                batch,
                noise=sde_noise,
                num_steps=self.sample_steps,
                method=self.method,
                sde_noise_level=self.sde_noise_level,
                sde_step_noise=step_noise,
            )
            return [chunk.cpu() for chunk in prediction.actions], None
        if self.capture_token_rows and isinstance(decoder, ARSuffixDecoder):
            # Greedy decode WITH the capture surface: encode + greedy
            # dispatch is compute-identical to model.predict_chunk's AR
            # case (generator/noise are None there), the capture is
            # observation only — the training-rows oracle path.
            memory = self.model.encode(batch.encoder_inputs, with_grad=False)
            greedy_capture: list[ActionCaptureStep] = []
            prediction = self.model.ar_predict_greedy(
                memory,
                batch,
                generate=self.generate,
                action_capture=greedy_capture,
            )
            self.last_token_rows = token_rows_from_capture(
                greedy_capture,
                block_base=decoder.config.block_base,
                temperature=None,
            )
            return [chunk.cpu() for chunk in prediction.actions], prediction.generations
        # Flow integrates from per-item seeded noise (deterministic and
        # batch-composition-independent); AR decodes greedily and takes
        # none.
        noise: Tensor | None = None
        if isinstance(decoder, FlowDecoder):
            shape = (batch.actions.shape[1], batch.actions.shape[2])
            noise = self._flow_noise(items, indices, 1, shape).to(self.device)
        elif isinstance(decoder, MolmoFlowDecoder):
            # Same determinism contract, THEIR geometry: noise lives in
            # [action_horizon, max_action_dim] (32 padded dims, not the
            # tag's action_dim) and must match the expert dtype — the
            # decoder consumes a supplied tensor as-is (parity surface),
            # so the cast happens here. Without this branch the decoder
            # draws from ambient RNG: irreproducible and
            # batch-composition-dependent.
            runtime = decoder.runtime
            assert runtime is not None  # from_checkpoint configures at load
            shape = (runtime.action_horizon, decoder.config.max_action_dim)
            noise = self._flow_noise(items, indices, 1, shape).to(
                device=self.device,
                dtype=decoder.action_embed.weight.dtype,
            )
        prediction = self.model.predict_chunk(
            batch,
            noise=noise,
            num_steps=self.sample_steps,
            method=self.method,
            target_time=self.target_time,
            generate=self.generate,
        )
        chunks = [chunk.cpu() for chunk in prediction.actions]
        if item_maps is not None:
            # Decoded raw chunks are in checkpoint-table units — pull
            # them back into each item's dataset units.
            chunks = [
                maps.action.invert(chunk)
                for maps, chunk in zip(item_maps, chunks, strict=True)
            ]
        return chunks, prediction.generations

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        chunks, generations = self.predict_with_text(items, indices)
        if self.generate and generations is not None:
            for index, generation in zip(indices, generations, strict=True):
                self.generations[index] = generation
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


@dataclasses.dataclass(frozen=True, slots=True)
class SubgoalRecord:
    """One frame's self-subgoal provenance row: the frame
    identity triple, the instruction, the TRUE segment label (None =
    unjudged frame) and what pass 1 generated — machine-readable for
    the stage-1 validity table and the results post's qualitative
    block. ``generated_text`` is the raw generation (the ground truth
    for reports); ``generated_subgoal`` is the parsed value line."""

    index: int
    repo_id: str
    episode_index: int
    frame_index: int
    instruction: str
    true_subgoal: str | None
    generated_subgoal: str | None
    generated_text: str


class SelfSubgoalPass1Policy:
    """Pass 1 of the self-subgoal probe — and the
    narrated-subgoal-only arm, free: the PLANNER-LESS prompt requests
    ``[generate|subgoal actions]``, the greedy decode's actions are
    this policy's prediction, and each frame's generated subgoal is
    retained (``records[index]``) for pass 2 to feed back through the
    prompt slot. Shares the base policy's loaded model (the
    NarratedBijouPolicy pattern); the collator is a generate_override
    clone of the base's, which must EXCLUDE the SUBGOAL condition
    field: training's anti-copy coupling suppressed the subgoal request
    whenever the prompt rendered the hint, so condition-plus-generate
    is a context the model never trained."""

    def __init__(
        self,
        base: BijouPolicy,
        *,
        draws: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        if AuxField.SUBGOAL not in base.aux_fields:
            raise SystemExit(
                "self-subgoal pass 1 requests the model's own subgoal; "
                "this checkpoint trained aux fields "
                f"{[f.value for f in base.aux_fields] or 'NONE'} — no "
                "subgoal to decode",
            )
        if ConditionField.SUBGOAL in base.collator.condition_fields:
            raise SystemExit(
                "self-subgoal pass 1 must collate the PLANNER-LESS "
                "context, but the base collator renders the [subgoal|…] "
                "condition — construct the base with subgoal_mode='self', "
                "not 'oracle'",
            )
        if draws is not None and draws < 0:
            raise SystemExit(f"--subgoal-draws must be >= 0, got {draws}")
        if draws is not None and draws > 0 and not temperature > 0:
            raise SystemExit(
                f"--subgoal-temperature must be > 0, got {temperature} "
                "(greedy is the draws-0 limit, not a temperature limit)",
            )
        self.base = base
        # None = plain self-subgoal probe: the legacy single-greedy
        # path, byte-for-byte.
        # An int >= 0 = subgoal-draws candidates mode: the same greedy full
        # pass PLUS `draws + 1` text-only candidate decodes off one
        # shared prefill (0 sampled draws still captures the greedy
        # candidate's stats — the bit-exactness preflight limit).
        self.draws = draws
        self.temperature = temperature
        self.name = f"{base.name}_narrsubgoal"
        self.collator = dataclasses.replace(
            base.collator,
            generate_override=(AuxField.SUBGOAL,),
        )
        self.records: dict[int, SubgoalRecord] = {}
        # Candidates mode only: per-frame candidate lists, index-keyed
        # like `records` (candidate 0 greedy, then the sampled draws).
        self.candidates: dict[int, list[ValueCandidate]] = {}

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        items = self.base.apply_overrides(items)
        batch = self.collator(items).to(self.base.device)
        if self.draws is None:
            prediction = self.base.model.predict_chunk(
                batch,
                num_steps=self.base.sample_steps,
                method=self.base.method,
                generate=(AuxField.SUBGOAL,),
            )
        else:
            # Sampled-draw RNGs reuse the draws10_t1 stable frame-keying
            # verbatim (stable_sample_rng, run seed, draw index d for
            # sampled candidate d in 1..draws; the greedy candidate 0
            # consumes no RNG).
            def sampling_for_draw(draw: int) -> ARSampling:
                return ARSampling(
                    temperature=self.temperature,
                    rngs=tuple(
                        stable_sample_rng(
                            self.base.seed,
                            str(item["repo_id"]),
                            int(item["episode_index"]),
                            int(item["frame_index"]),
                            draw,
                        )
                        for item in items
                    ),
                )

            prediction, candidates = self.base.model.ar_predict_with_value_candidates(
                batch,
                field=AuxField.SUBGOAL,
                generate=(AuxField.SUBGOAL,),
                draws=self.draws,
                sampling_for_draw=sampling_for_draw,
            )
            for index, row_candidates in zip(indices, candidates, strict=True):
                self.candidates[index] = row_candidates
        assert prediction.generations is not None  # ar_backbone always generates
        for item, index, generation in zip(
            items,
            indices,
            prediction.generations,
            strict=True,
        ):
            self.records[index] = SubgoalRecord(
                index=index,
                repo_id=str(item["repo_id"]),
                episode_index=int(item["episode_index"]),
                frame_index=int(item["frame_index"]),
                instruction=str(item["task"]),
                true_subgoal=subgoal_text(item),
                generated_subgoal=generation.subgoal,
                generated_text=generation.text,
            )
        return [chunk.cpu() for chunk in prediction.actions]


class SelfSubgoalPolicy:
    """Pass 2 — the self-subgoal arm: re-encode each frame with pass 1's
    generated subgoal rendered through the trained ``[subgoal|…]``
    prompt slot and decode actions on the deployment fast path
    ``[generate|actions]``. The request NEVER includes subgoal
    (condition-plus-generate is an untrained
    context); rendering goes through the one shared Collator path, never a
    re-implementation. An empty or absent pass-1
    generation renders nothing, so the prompt is byte-identical to the
    planner-less baseline's (the no-hint limit);
    ``force_empty`` forces that limit on EVERY frame, the live
    pre-launch check, and suffixes the name so a forced run can never
    pass as the self arm."""

    def __init__(
        self,
        base: BijouPolicy,
        pass1: SelfSubgoalPass1Policy,
        *,
        force_empty: bool = False,
    ) -> None:
        self.base = base
        self.pass1 = pass1
        self.force_empty = force_empty
        self.name = f"{base.name}_selfsubgoal" + ("_emptyhint" if force_empty else "")
        self.collator = dataclasses.replace(
            base.collator,
            # The trained condition fields WITH the subgoal slot, in
            # template order (the base excluded it — planner-less).
            condition_fields=tuple(
                f for f in ConditionField if f.value in set(base.info.condition_fields)
            ),
            generate_override=(),
        )

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        items = self.base.apply_overrides(items)
        conditioned: list[dict[str, Any]] = []
        for item, index in zip(items, indices, strict=True):
            record = self.pass1.records.get(index)
            if record is None:
                raise SystemExit(
                    f"self-subgoal pass 2 reached frame {index} before "
                    "pass 1 — the runner must score the pass-1 policy "
                    "first in every batch",
                )
            text = "" if self.force_empty else (record.generated_subgoal or "")
            # An explicit EMPTY override means "no hint" to the collator
            # — it must never fall through to the frame's true label.
            conditioned.append({**item, "condition_subgoal": text})
        batch = self.collator(conditioned).to(self.base.device)
        prediction = self.base.model.predict_chunk(
            batch,
            num_steps=self.base.sample_steps,
            method=self.base.method,
            generate=(),
        )
        return [chunk.cpu() for chunk in prediction.actions]


class MaskedContrastSubgoalPolicy:
    """Masked-contrast (mcselect) producer — the masked-contrast
    (MG-Select-form)
    scorer's measurement pass (the frozen reads and the ARGMAX live in
    ``fontaine/scripts/mcselect_results.py``, never here).

    Candidates come from the banked clean-list candidates FILE — no
    in-run sampling, so the scorer re-ranks exactly the width whose
    ceiling and floor are on the board. Per frame: ONE planner-less
    greedy decode (the masked reference prediction, record-only), then
    per ELIGIBLE (non-truncated) candidate a greedy decode conditioned
    on the candidate's text through the trained ``[subgoal|…]`` prompt
    slot (the SelfSubgoalPolicy rendering path, byte-shared) with the
    ACTION phase's own logits captured, plus one teacher-forced
    planner-less forward over that candidate's decoded action-id
    sequence (one shared masked prefill, snapshot/restored — the
    sampled-draws convention). Score
    ``KL( p_cond(·|c) ‖ p_masked^{1/τ} )`` averaged over the
    candidate's active decode steps, both sides masked by the SAME
    grammar mask the decode applied. Ineligible candidates stay NaN —
    a finite score there is a contract violation the read script
    aborts on."""

    def __init__(
        self,
        base: BijouPolicy,
        *,
        candidates_by_index: dict[int, list[dict[str, Any]]],
        tau: float,
    ) -> None:
        if not isinstance(base.model.decoder, ARSuffixDecoder):
            raise SystemExit(
                "masked-contrast scoring continues a trunk suffix; the "
                f"loaded decoder is {type(base.model.decoder).__name__}",
            )
        if ConditionField.SUBGOAL in base.collator.condition_fields:
            raise SystemExit(
                "masked-contrast needs the PLANNER-LESS base context "
                "(the masked reference IS the deployment default), but "
                "the base collator renders the [subgoal|…] condition — "
                "construct the base with subgoal_mode='mcselect'",
            )
        if not tau > 0:
            raise SystemExit(f"mcselect tau must be > 0, got {tau}")
        if not candidates_by_index:
            raise SystemExit("mcselect got an empty candidates file")
        first = next(iter(candidates_by_index.values()))
        if not first or not {"text", "truncated"} <= set(first[0]):
            raise SystemExit(
                "candidates file rows must carry candidate dicts with "
                "'text' and 'truncated' — not a clean-list candidates dump",
            )
        self.base = base
        self.candidates_by_index = candidates_by_index
        self.tau = float(tau)
        # One width for every dumped row (the reader's [N, C] contract);
        # the banked file is homogeneous, but never assume per batch.
        self.n_candidates = max(len(c) for c in candidates_by_index.values())
        self.name = f"{base.name}_mcselectsubgoal"
        self.collator = base.collator  # planner-less, deployment fast path
        self.cond_collator = dataclasses.replace(
            base.collator,
            # The trained condition fields WITH the subgoal slot, in
            # template order (the SelfSubgoalPolicy construction).
            condition_fields=tuple(
                f for f in ConditionField if f.value in set(base.info.condition_fields)
            ),
            generate_override=(),
        )
        # Per-frame measurement rows, keyed by global frame index —
        # the --dump-predictions npz serializes these (mcselect:* keys).
        self.kl: dict[int, list[float]] = {}
        self.cand_pred: dict[int, Tensor] = {}
        self.pred_masked: dict[int, Tensor] = {}

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        items = self.base.apply_overrides(items)
        rows_cands: list[list[dict[str, Any]]] = []
        for index in indices:
            cands = self.candidates_by_index.get(index)
            if cands is None:
                raise SystemExit(
                    f"frame index {index} has no row in the candidates "
                    "file — this run's plan does not match the banked "
                    "candidate-list width, stop",
                )
            rows_cands.append(cands)
        model = self.base.model
        decoder = model.decoder
        assert isinstance(decoder, ARSuffixDecoder)  # __init__ guarded
        block_base = decoder.config.block_base

        # Masked pass: ONE planner-less prefill for the reference
        # prediction AND (snapshot/restored) every candidate's
        # teacher-forced reference forward.
        batch = self.collator(items).to(self.base.device)
        memory = model.encode(batch.encoder_inputs, with_grad=False)
        snapshot = decoder.cache_snapshot(memory)
        masked_chunks = [
            chunk.cpu() for chunk in model.ar_predict_greedy(memory, batch).actions
        ]

        n_rows = len(items)
        kl = torch.full((n_rows, self.n_candidates), float("nan"), dtype=torch.float64)
        chunk_shape = tuple(masked_chunks[0].shape)  # (S, D)
        cand_pred = torch.full(
            (n_rows, self.n_candidates, *chunk_shape),
            float("nan"),
        )
        for j in range(self.n_candidates):
            eligible = [
                row
                for row in range(n_rows)
                if j < len(rows_cands[row]) and not rows_cands[row][j]["truncated"]
            ]
            if not eligible:
                continue
            cond_items = [
                {**items[row], "condition_subgoal": str(rows_cands[row][j]["text"])}
                for row in eligible
            ]
            cbatch = self.cond_collator(cond_items).to(self.base.device)
            cmemory = model.encode(cbatch.encoder_inputs, with_grad=False)
            capture: list[ActionCaptureStep] = []
            conditioned = model.ar_predict_greedy(
                cmemory,
                cbatch,
                action_capture=capture,
            )
            # CPU once per step — the per-row loop below must not sync
            # the device thousands of times.
            steps = [
                (
                    step.block_logits.cpu(),
                    step.allowed.cpu(),
                    step.active.cpu(),
                    step.chosen.cpu(),
                )
                for step in capture
            ]
            active_steps: list[list[int]] = [
                [t for t, (_, _, active, _) in enumerate(steps) if bool(active[sub])]
                for sub in range(len(eligible))
            ]
            sequences: list[list[int] | None] = [None] * n_rows
            for sub, row in enumerate(eligible):
                sequences[row] = [
                    int(steps[t][3][sub]) - block_base for t in active_steps[sub]
                ]
            decoder.cache_restore(memory, snapshot)
            reference = model.ar_teacher_forced_block_logits(memory, sequences)
            for sub, row in enumerate(eligible):
                ref_row = reference[row]
                assert ref_row is not None  # sequences[row] was non-None
                own_steps = active_steps[sub]
                if ref_row.shape[0] != len(own_steps):
                    raise SystemExit(
                        f"row {row} candidate {j}: reference forward "
                        f"returned {ref_row.shape[0]} positions for "
                        f"{len(own_steps)} decoded steps — the "
                        "teacher-forced scaffold drifted from the decode, "
                        "stop",
                    )
                ref_row = ref_row.cpu()
                total = 0.0
                for position, t in enumerate(own_steps):
                    step_logits, step_allowed, _, _ = steps[t]
                    legal = step_allowed[sub]
                    # float64: the score is tiny CPU math and lands in
                    # a float64 dump column — no fp32 noise in the rank.
                    log_cond = step_logits[sub][legal].double().log_softmax(-1)
                    log_ref = (
                        ref_row[position][legal].double() / self.tau
                    ).log_softmax(
                        -1,
                    )
                    total += float((log_cond.exp() * (log_cond - log_ref)).sum())
                kl[row, j] = total / len(own_steps)
            actions = conditioned.actions.cpu()
            for sub, row in enumerate(eligible):
                cand_pred[row, j] = actions[sub]

        for i, (item, index) in enumerate(zip(items, indices, strict=True)):
            length = int(item["action"].shape[0])
            self.kl[index] = [float(v) for v in kl[i]]
            self.cand_pred[index] = cand_pred[i, :, :length].clone()
            self.pred_masked[index] = masked_chunks[i][:length].float().clone()
        return masked_chunks


class SelectedSubgoalPolicy:
    """Pass 2 of subgoal-draws candidate selection: condition each
    frame on a SELECTED candidate from pass 1's candidate list and
    decode actions on the deployment fast path — the SelfSubgoalPolicy
    machinery with the selection swapped in.

    ``mode='bon'`` (primary, deployment-honest): the frozen
    self-certainty pick — distribution stats only, no candidate text
    comparison, no label access anywhere on the path (the scorer
    function's signature has no label argument).
    ``mode='ceil'`` (record-only ceiling): the candidate maximizing
    token-F1 vs the frame's TRUE segment label; label-less frames
    render NO subgoal (the oracle-arm convention). The name
    carries the mode (``_bonsubgoal``/``_ceilsubgoal``) so an
    oracle-informed read can never pass as a deployment read.

    ``candidate_filter='clean'`` applies the clean-list frozen
    eligible-list rule: every scorer
    operates on the non-truncated candidates only, an all-truncated row
    falls back to the greedy candidate as decoded (recorded), and the
    name carries the filter (``_boncleansubgoal``/``_ceilcleansubgoal``)
    so a filtered read can never pool as an unfiltered read. Pass 1 and
    the dump bytes are untouched — the filter is selection-side only.

    ``force_empty`` forces the no-hint limit on every frame (the
    pre-launch consistency run) and suffixes the name."""

    def __init__(
        self,
        base: BijouPolicy,
        pass1: SelfSubgoalPass1Policy,
        *,
        mode: str,
        force_empty: bool = False,
        candidate_filter: str | None = None,
    ) -> None:
        if mode not in ("bon", "ceil"):
            raise SystemExit(f"selection mode must be 'bon' or 'ceil', got {mode!r}")
        if candidate_filter not in (None, "clean"):
            raise SystemExit(
                f"candidate filter must be None or 'clean', got {candidate_filter!r}",
            )
        if pass1.draws is None:
            raise SystemExit(
                "selected-subgoal pass 2 needs pass 1 in candidates mode — "
                "construct SelfSubgoalPass1Policy with draws >= 0",
            )
        self.base = base
        self.pass1 = pass1
        self.mode = mode
        self.force_empty = force_empty
        self.candidate_filter = candidate_filter
        tag = "clean" if candidate_filter == "clean" else ""
        self.name = f"{base.name}_{mode}{tag}subgoal" + (
            "_emptyhint" if force_empty else ""
        )
        self.collator = dataclasses.replace(
            base.collator,
            # The trained condition fields WITH the subgoal slot, in
            # template order (the SelfSubgoalPolicy construction).
            condition_fields=tuple(
                f for f in ConditionField if f.value in set(base.info.condition_fields)
            ),
            generate_override=(),
        )
        # Per-frame pick provenance (None = label-less ceil row): the
        # candidates dump cross-checks these against offline recomputes.
        self.picks: dict[int, int | None] = {}

    def _pick(self, index: int) -> int | None:
        candidates = self.pass1.candidates.get(index)
        if candidates is None:
            raise SystemExit(
                f"selected-subgoal pass 2 reached frame {index} before "
                "pass 1 — the runner must score the pass-1 policy first "
                "in every batch",
            )
        vocabs = {c.allowed_vocab for c in candidates}
        if len(vocabs) != 1:
            raise SystemExit(
                f"frame {index}: candidates disagree on allowed_vocab "
                f"{sorted(vocabs)} — mixed decode masks, stop",
            )
        eligible = (
            eligible_indices([c.truncated for c in candidates])
            if self.candidate_filter == "clean"
            else list(range(len(candidates)))
        )
        if self.mode == "bon":
            return eligible[
                self_certainty_pick(
                    [candidates[i].mean_logprob for i in eligible],
                    candidates[0].allowed_vocab,
                )
            ]
        record = self.pass1.records[index]
        if record.true_subgoal is None:
            return None
        return eligible[
            ceiling_pick(
                [candidates[i].text for i in eligible],
                record.true_subgoal,
            )
        ]

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        items = self.base.apply_overrides(items)
        conditioned: list[dict[str, Any]] = []
        for item, index in zip(items, indices, strict=True):
            pick = self._pick(index)
            self.picks[index] = pick
            text = (
                ""
                if self.force_empty or pick is None
                else self.pass1.candidates[index][pick].text
            )
            # An explicit EMPTY override means "no hint" to the collator
            # — it must never fall through to the frame's true label.
            conditioned.append({**item, "condition_subgoal": text})
        batch = self.collator(conditioned).to(self.base.device)
        prediction = self.base.model.predict_chunk(
            batch,
            num_steps=self.base.sample_steps,
            method=self.base.method,
            generate=(),
        )
        return [chunk.cpu() for chunk in prediction.actions]
