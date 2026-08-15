"""Decoder-only action decoder: the backbone's suffix role (ar_backbone).

The prompt — which carries the ``[generate|…]`` request conditioning
and the projected state token (prompt format 3) — is prefill-encoded
once (the ObservationMemory retains the full prefix KV cache); this
decoder continues the suffix-format-5 sequence
``[<|turn>model\\n][value\\n per requested][BOA][t_1..t_k]`` through ALL
backbone layers — the KV-shared deep half included — and reads
FULL-VOCABULARY logits from the frozen tied LM head with the FAST
block's columns supplied by a trainable patch. Aux value lines
(headerless, request-ordered — bijou.modelling.aux_text) and action tokens share
that one softmax. What speaks is commanded by the PROMPT's request
set: the model learns p(value | observation, asked) and is never asked
to infer judged-ness from appearance.

Ownership: this module owns ONLY the new parameters (~11M at E2B scale)
— the FAST block's input-embedding + per-layer-embedding rows. The
backbone is owned by the family class and passed into every call; the
state projection is PROMPT-side and lives in GemmaEncoder (format 3
moved state into the user turn).

Id spaces: the collator's ``action_tokens`` and the codec speak CODEC
ids [0, vocab_total); the head/targets speak backbone ids
``block_base + codec_id`` — the block is TAIL-anchored at
``vocab_size − vocab_total`` (E2B: 261118..262143, inside the 3259-id
reserved-unused run starting at 258885). Aux text ids are ordinary
sub-block text ids. The backbone never consumes FAST ids as ids: suffix
tokens enter as embeddings via ``inputs_embeds``/``per_layer_inputs``
(text ids through the frozen tables, block ids through the patch); the
block ids exist so actions and text share one softmax under
full-vocabulary CE.

Sequence/loss conventions: all-but-the-last opener positions predict
constants and are IGNOREd; BOA IS predicted (the trained transition
out of the value lines — or directly after the opener on
``[generate|actions]`` samples — and the action block's single-id
begin-marker); PAD is batch padding only and always ignored; there is
no EOA (action length is fixed by the FAST grammar). Decoding
(:meth:`GemmaARDecoder.predict_chunk`) is fully scaffolded by the
request set: per requested field, greedy value ids under a per-field
budget until the ``\\n`` terminator (constrained candidates where
defined), then BOA is FORCED and the FAST grammar mask decodes the
chunk by remaining symbol budget.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, override

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..aux_text import (
    SUFFIX_FORMAT,
    VALUE_BUDGETS,
    AuxDecodeConfig,
    AuxField,
    AuxGeneration,
    AuxRuntime,
    TextTokenizer,
    display_text,
)
from ..codecs import ActionCodec
from ..interface import (
    ActionCaptureStep,
    ARSampling,
    BijouPrediction,
    CollatedBatch,
    ObservationMemory,
    ValueCandidate,
)

# CE positions to skip (opener prefix, PAD padding) — torch's
# cross_entropy convention. Lived in decoders/ar_fast.py until that
# decoder's retirement (tag pre-decoder-simplify).
IGNORE_INDEX = -100

# The MolmoAct2 release's discrete emission format (suffix_format 6):
# <action_start> + bins + <action_end> continued from their serving
# prompt — empty opener, no value lines, specials at negative codec
# offsets (MolmoAct2ActionCodec). Registered beside the config guard
# (aux_text.SUFFIX_FORMAT = 5 is the Gemma/Molmo2 value-line scaffold);
# each concrete asserts ITS format at construction.
MOLMOACT2_SUFFIX_FORMAT = 6


@dataclass(frozen=True, slots=True)
class ARDecoderConfig:
    """Construction config of the decoder-only path. Deliberately tiny:
    the backbone defines all geometry; this records only the action
    vocabulary and its placement. ``vocab_total`` = the width of the
    codec-relative body id space (our artifacts: BPE + BOA + PAD; the
    MolmoAct2 release: the bare 2048 bin block — its specials sit
    below); ``block_base`` is the first backbone vocabulary id of the
    reserved block."""

    tokenizer: str  # artifact ref (local dir or <user>/<repo>/<subfolder>)
    vocab_total: int
    block_base: int
    chunk_size: int
    action_dim: int
    # Suffix format — WHICH suffix grammar the checkpoint trained:
    # 5 (aux_text.SUFFIX_FORMAT) = the request-conditioned headerless
    # format of the Gemma/Molmo2 concretes (opener bytes, value lines,
    # BOA, FAST block); 6 (MOLMOACT2_SUFFIX_FORMAT) = the MolmoAct2
    # release emission (<action_start> + bins + <action_end> against
    # their serving prompt — no opener bytes, no value lines). Each
    # concrete asserts ITS format at construction; formats ≤ 4 (fed
    # mode tokens, suffix state slot, header bytes) are REFUSED — no
    # trained artifact worth loading exists (owner call, 2026-08-03;
    # the parameter sets are incompatible anyway: state_proj moved
    # prompt-side, mode tables deleted).
    suffix_format: int
    # Aux text record: template version + fields + label provenance
    # (None = trained without aux — every sample was [generate|actions]).
    aux: AuxDecodeConfig | None

    def __post_init__(self) -> None:
        if self.vocab_total < 3:  # ≥ 1 body token + the special offsets
            raise ValueError(f"vocab_total {self.vocab_total} is not a FAST vocabulary")
        if self.block_base < 0:
            raise ValueError(f"block_base {self.block_base} must be non-negative")
        if self.suffix_format < SUFFIX_FORMAT:
            raise ValueError(
                f"suffix format {self.suffix_format} < {SUFFIX_FORMAT}: "
                "pre-5 checkpoints trained mode tokens/suffix state/header "
                "bytes this code no longer implements — retrain (no "
                "back-compat, 2026-08-03)",
            )
        if self.suffix_format not in (SUFFIX_FORMAT, MOLMOACT2_SUFFIX_FORMAT):
            raise ValueError(
                f"unknown suffix format {self.suffix_format} — registered "
                f"formats: {SUFFIX_FORMAT} (Gemma/Molmo2 value-line "
                f"scaffold), {MOLMOACT2_SUFFIX_FORMAT} (MolmoAct2 release "
                "emission)",
            )


def _sample_action_ids(
    logits: Tensor,
    allowed: Tensor,
    sampling: ARSampling,
) -> Tensor:
    """One categorical draw per row from softmax(logits/T) restricted
    to the grammar mask, via Gumbel-max: argmax(logits/T + G) with G
    i.i.d. Gumbel(0,1) samples the masked softmax exactly; illegal ids
    sit at -inf and can never win. Gumbel noise G = -log(E), E ~ Exp(1)
    drawn fp32 from each row's own CPU RNG (clamped away from 0 —
    torch/numpy exponentials may return exact zeros)."""
    scaled = (logits / sampling.temperature).masked_fill(~allowed, float("-inf"))
    exponential = np.stack(
        [
            rng.standard_exponential(scaled.shape[1], dtype=np.float32)
            for rng in sampling.rngs
        ],
    )
    gumbel = -np.log(np.maximum(exponential, np.finfo(np.float32).tiny))
    return (scaled + torch.from_numpy(gumbel).to(scaled.device)).argmax(dim=-1)


class ARSuffixDecoder[B: nn.Module](nn.Module, abc.ABC):
    """Trunk-generic half of the backbone-suffix role: the format-5
    scaffold (opener, value lines, BOA, grammar-masked action decode),
    the codec/config guards, and the teacher-forced forward shape.

    Generic over the trunk type ``B``: the trunk-specific compute — how
    suffix ids become embeddings, how they continue through the stack
    against the prefix cache, and how full-id-space logits are read —
    lives in the three abstract methods. The Gemma concrete is
    :class:`GemmaARDecoder`; the Molmo2 concrete rides the same
    scaffold (checkpoint decoder kind stays ``ar_backbone`` — the trunk
    axis is the PROMPT kind).

    ``codec`` is a runtime resource (BPE + quantile glue), not a module;
    checkpoints reference the tokenizer artifact by id. ``opener_text``
    is the trunk's assistant-turn opener bytes (Gemma:
    ``aux_text.GENERATION_OPENER``; ChatML trunks pass their own).

    ``newline_carrier_ids``: text ids whose decoded bytes CONTAIN the
    field terminator (``\\n``) without being it — banned during value
    decoding so termination is always the single trained terminator id.
    Empty for Gemma (its tokenizer satisfies the split==joint boundary
    contract on every probe value, so the ban would be a no-op risk);
    tokenizers with merged ``…\\n`` pieces (Qwen's ``'%\\n'``) populate
    it and skip the boundary SystemExit instead."""

    def __init__(
        self,
        config: ARDecoderConfig,
        codec: ActionCodec,
        *,
        tokenizer: TextTokenizer | None,
        opener_text: str,
        aux_runtime: AuxRuntime | None = None,
        aux_loss_weight: float = 1.0,
        newline_carrier_ids: frozenset[int] = frozenset(),
    ) -> None:
        super().__init__()
        self.config = config
        self.codec = codec
        # Runtime resources, not modules (the codec convention): the text
        # tokenizer (opener ids, generation text), aux decode runtime,
        # and the aux loss mixture weight.
        if tokenizer is None:
            raise ValueError(
                "ar_backbone needs the backbone's text tokenizer (the "
                "generation opener is tokenized at construction)",
            )
        self.tokenizer: TextTokenizer = tokenizer
        self.aux_runtime = aux_runtime
        self.aux_loss_weight = aux_loss_weight
        self.newline_carrier_ids = newline_carrier_ids
        # Cumulative value-budget exhaustions (decode health metric).
        self.fallback_count = 0
        self.opener_ids: tuple[int, ...] = tuple(
            tokenizer.encode(opener_text, add_special_tokens=False),
        )
        if codec.vocab_total != config.vocab_total:
            raise ValueError(
                f"codec vocab_total {codec.vocab_total} != config "
                f"{config.vocab_total} — the checkpoint was trained with a "
                "different tokenizer artifact",
            )
        if (codec.time_horizon, codec.action_dim) != (
            config.chunk_size,
            config.action_dim,
        ):
            raise ValueError(
                f"tokenizer horizon/dim ({codec.time_horizon}, "
                f"{codec.action_dim}) != decoder chunk/action_dim "
                f"({config.chunk_size}, {config.action_dim})",
            )
        # Constrained decoding needs each token's symbol expansion
        # length. CODEC-owned (the measurement is artifact-family
        # specific: our fit reads BPE piece lengths, the released
        # byte-level BPE needs decoded-string lengths). Copied into a
        # fresh tensor — plain attribute, not a buffer: derived from
        # the codec, never saved.
        symbol_lengths = torch.tensor(codec.symbol_lengths, dtype=torch.long)
        if symbol_lengths.shape[0] != config.vocab_total:
            raise ValueError(
                f"codec symbol_lengths covers {symbol_lengths.shape[0]} "
                f"ids, config vocab_total is {config.vocab_total} — "
                "mismatched codec/config pairing",
            )
        if not bool((symbol_lengths == 1).any()):
            raise ValueError(
                "BPE vocabulary has no single-symbol token — exact fill "
                "(and decode termination) cannot be guaranteed",
            )
        self.symbol_lengths = symbol_lengths

    @abc.abstractmethod
    def init_tables_from_backbone(self, backbone: B) -> None:
        """Re-init the trainable FAST tables around the trunk's REAL
        tables' statistics (see each concrete) — full-vocabulary CE
        competes block logits against text priors, so an arbitrary
        offset start matters."""

    @abc.abstractmethod
    def _suffix_hidden(
        self,
        backbone: B,
        memory: ObservationMemory,
        tokens: Tensor,
        fed: int,
    ) -> Tensor:
        """Embed BACKBONE-id suffix ``tokens`` [B, T] (text ids through
        the frozen tables, block ids through the trainable patch) and run
        them through ALL trunk layers against the prefix cache (which
        they extend in place). ``fed`` = suffix positions already in the
        cache from previous calls (decode loop). Returns final-normed
        hidden states [B, T, hidden]."""

    @abc.abstractmethod
    def _logits(self, backbone: B, hidden: Tensor) -> Tensor:
        """Full-id-space logits [B, T, V] for suffix hidden states — the
        FAST block's columns computed from the trainable patch, text
        columns from the trunk's (frozen) head; every id in
        ``[0, block_base + vocab_total)`` must be a valid column."""

    @override
    def forward(
        self,
        backbone: B,
        memory: ObservationMemory,
        tokens: Tensor,
    ) -> Tensor:
        """Next-token logits over the suffix (BACKBONE ids — mixed value
        lines + FAST block).

        CONSUMES the memory's cache (suffix K/V are appended): encode a
        fresh memory per call — training does, and the decode loop is
        exactly the incremental consumer this enables.

        Shapes:
          - memory.cache: the trunk-private prefix cache
          - tokens: [B, T]  (long; backbone ids)
          - returns: [B, T, V]  (full id space incl. the FAST block;
            position j predicts suffix position j + 1)
        """
        hidden = self._suffix_hidden(backbone, memory, tokens, fed=0)
        return self._logits(backbone, hidden)

    @torch.no_grad()
    def predict_chunk(
        self,
        backbone: B,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generate: tuple[AuxField, ...] = (),
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        sampling: ARSampling | None = None,
        action_capture: list[ActionCaptureStep] | None = None,
    ) -> BijouPrediction:
        """The single decode path, fully scaffolded by the request set.
        Deterministic greedy by default; ``sampling`` switches the
        ACTION block (only) to per-row temperature sampling — the
        sampled-draws eval instrument. ``generator``/``noise``
        unused/must be None. ``action_capture`` (mcselect): a list
        the ACTION phase appends one :class:`ActionCaptureStep` to per
        decode step — the conditional scoring surface, captured from
        the very logits the decode chose from (no re-forward, no
        numeric drift vs the executed decode).

        ``generate`` must equal the request the PROMPT was collated with
        (the collator's ``generate_override``; the memory was encoded
        from that prompt — mismatched scaffolds would sit off the
        conditioning). Per requested field, in order: greedy text-id
        decode under the field's VALUE_BUDGETS cap until the ``\\n``
        terminator (budget exhaustion forces it and counts a loud
        fallback; HOLDING constrains the first token to its candidate
        set). After the last field — immediately, for ``generate=()``,
        the deployment fast path — BOA is FORCED (its target is trained;
        its identity is not a decision) and the FAST grammar mask
        decodes the chunk by remaining symbol budget, PAD once finished.

        Requires an aux-trained checkpoint for non-empty ``generate``
        (requested-but-untrained fields would be elicited off-manifold).

        Returns a BijouPrediction: chunks [B, chunk, action_dim] raw
        units + one AuxGeneration per row (empty for ``generate=()``)."""
        if noise is not None:
            raise ValueError("GemmaARDecoder.predict_chunk takes no noise")
        config = self.config
        if generate:
            trained = () if config.aux is None else config.aux.fields
            untrained = [f.value for f in generate if f not in trained]
            if untrained:
                raise ValueError(
                    f"generate={untrained} requested, but the checkpoint "
                    f"trained aux fields {[f.value for f in trained] or 'NONE'} "
                    "— an untrained field would be elicited off-manifold",
                )
            ordered = tuple(f for f in AuxField if f in generate)
            if ordered != generate:
                raise ValueError(
                    f"generate must keep template order; got "
                    f"{[f.value for f in generate]}",
                )
        stats = batch.action_stats
        if stats.q01 is None or stats.q99 is None:
            raise SystemExit(
                "batch stats carry no action quantiles — AR decoding needs "
                "the exact q01/q99 the tokenizer was fitted under (old "
                "checkpoint stats tables cannot drive AR inference)",
            )
        batch_size = batch.state.shape[0]
        if sampling is not None and len(sampling.rngs) != batch_size:
            raise ValueError(
                f"sampling carries {len(sampling.rngs)} row RNGs for a "
                f"batch of {batch_size} — one keyed stream per row",
            )
        device = batch.state.device
        base = config.block_base
        boa_backbone = base + self.codec.boa
        pad_backbone = base + self.codec.pad
        lengths = self.symbol_lengths.to(device)
        total_symbols = config.chunk_size * config.action_dim
        min_value: float = torch.finfo(torch.float32).min
        runtime = self.aux_runtime
        if generate and runtime is None:
            raise ValueError(
                "aux fields requested but the decoder has no AuxRuntime — "
                "loading built the model without the aux record",
            )

        # Text-only mask for value decoding (block ids illegal — the
        # terminator ends a value, BOA is forced by the scaffold).
        vocab = base + config.vocab_total
        text_allowed = torch.zeros(vocab, dtype=torch.bool, device=device)
        text_allowed[:base] = True
        if self.newline_carrier_ids:
            # Ban text ids whose decoded bytes carry the terminator
            # inside a merged piece (e.g. Qwen's '%\n') — termination
            # must always be the single trained terminator id.
            text_allowed[
                torch.tensor(
                    sorted(self.newline_carrier_ids),
                    dtype=torch.long,
                    device=device,
                )
            ] = False

        fed = 0
        fallback_count = 0
        value_ids: list[list[list[int]]] = [[] for _ in range(batch_size)]

        # Value phase: rows decode in lockstep, one requested field at a
        # time — rows that terminate early keep feeding the terminator
        # until the batch max (their cache gains benign repeat-\n
        # positions; never recorded — the accepted imprecision of the
        # old free phase's PAD-after-finish, and rollout's B=1 never
        # hits it).
        feed: Tensor = (
            torch.tensor([list(self.opener_ids)], dtype=torch.long, device=device)
            .expand(batch_size, -1)
            .contiguous()
        )
        for aux_field in generate:
            assert runtime is not None  # guarded above
            terminator = runtime.terminator_id
            candidates = runtime.value_candidates.get(aux_field)
            if candidates is not None:
                # Constrained field = classification: score the
                # candidates' first ids once, then FORCE the winning
                # candidate + terminator (deterministic length, exactly
                # the trained bytes — no free phase to wander in).
                logits, fed = self._step(backbone, memory, feed, fed)
                firsts = torch.tensor(
                    [c[0] for c in candidates],
                    dtype=torch.long,
                    device=device,
                )
                picks = logits[:, firsts].argmax(dim=-1).tolist()
                width = max(len(c) for c in candidates) + 1
                forced_rows = torch.full(
                    (batch_size, width),
                    terminator,
                    dtype=torch.long,
                    device=device,
                )
                for row, pick in enumerate(picks):
                    chosen = [*candidates[pick], terminator]
                    forced_rows[row, : len(chosen)] = torch.tensor(
                        chosen,
                        dtype=torch.long,
                        device=device,
                    )
                    value_ids[row].append(list(candidates[pick]))
                feed = forced_rows
                continue
            budget = VALUE_BUDGETS[aux_field]
            done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            row_ids: list[list[int]] = [[] for _ in range(batch_size)]
            for step in range(budget + 1):
                logits, fed = self._step(backbone, memory, feed, fed)
                logits = logits[:, :vocab].masked_fill(~text_allowed, min_value)
                next_ids = logits.argmax(dim=-1)
                if step == budget:
                    # Budget exhausted: force the terminator on rows
                    # still talking.
                    forced = ~done & (next_ids != terminator)
                    fallback_count += int(forced.sum())
                    next_ids = torch.where(
                        forced,
                        torch.full_like(next_ids, terminator),
                        next_ids,
                    )
                # Early-terminated rows keep feeding the terminator.
                next_ids = torch.where(
                    done,
                    torch.full_like(next_ids, terminator),
                    next_ids,
                )
                for row, next_id in enumerate(next_ids.tolist()):
                    if not done[row] and next_id != terminator:
                        row_ids[row].append(next_id)
                done |= next_ids == terminator
                feed = next_ids[:, None]
                if bool(done.all()):
                    break
            for row in range(batch_size):
                value_ids[row].append(row_ids[row])

        # Action phase: BOA forced (the scaffold's transition — its
        # target is trained, its identity is not a decision), fed
        # together with the not-yet-consumed last feed (the final
        # terminator, or the whole opener when generate=()).
        boa = torch.full(
            (batch_size, 1),
            boa_backbone,
            dtype=torch.long,
            device=device,
        )
        feed = torch.cat([feed, boa], dim=1)
        remaining = torch.full((batch_size,), total_symbols, device=device)
        action_ids: list[list[int]] = [[] for _ in range(batch_size)]
        while not bool((remaining == 0).all()):
            logits, fed = self._step(backbone, memory, feed, fed)
            logits = logits[:, :vocab]
            action_allowed = (lengths[None, :] > 0) & (
                lengths[None, :] <= remaining[:, None]
            )
            action_allowed = torch.cat(
                [
                    torch.zeros(
                        (batch_size, base),
                        dtype=torch.bool,
                        device=device,
                    ),
                    action_allowed,
                ],
                dim=1,
            )
            action_allowed[:, pad_backbone] = remaining == 0
            if sampling is None:
                next_ids = logits.masked_fill(~action_allowed, min_value).argmax(
                    dim=-1,
                )
            else:
                # Finished rows have only PAD legal, so they keep
                # sampling PAD; their recorded ids are already fixed,
                # and each active row's ids depend only on its OWN
                # stream position — batch-composition-independent.
                next_ids = _sample_action_ids(logits, action_allowed, sampling)
            if action_capture is not None:
                # Active BEFORE the remaining update: exactly the rows
                # whose emitted id this step is a real symbol (finished
                # rows can only emit PAD — never recorded).
                action_capture.append(
                    ActionCaptureStep(
                        block_logits=logits[:, base:].float(),
                        allowed=action_allowed[:, base:].clone(),
                        active=(remaining > 0).clone(),
                        chosen=next_ids.clone(),
                    ),
                )
            for row, next_id in enumerate(next_ids.tolist()):
                codec_id = next_id - base
                if codec_id != self.codec.pad:
                    action_ids[row].append(codec_id)
                    remaining[row] -= int(lengths[codec_id])
            feed = next_ids[:, None]
        if fallback_count:
            self.fallback_count += fallback_count
            print(
                f"[ar_backbone] value budget exhausted on "
                f"{fallback_count} field value(s) — terminator forced "
                f"(cumulative {self.fallback_count}); a persistent rate "
                "means the model stopped closing its value lines",
                flush=True,
            )

        q01 = stats.q01.cpu().numpy()
        q99 = stats.q99.cpu().numpy()
        chunks = [
            torch.from_numpy(
                self.codec.decode(row_ids, q01[row], q99[row]),
            ).float()
            for row, row_ids in enumerate(action_ids)
        ]
        generations = [
            _parse_aux(
                generate,
                [self.tokenizer.decode(ids) for ids in row],
            )
            for row in value_ids
        ]
        return BijouPrediction(
            actions=torch.stack(chunks).to(device),
            generations=generations,
        )

    @torch.no_grad()
    def teacher_forced_block_logits(
        self,
        backbone: B,
        memory: ObservationMemory,
        action_ids: list[list[int] | None],
    ) -> list[Tensor | None]:
        """Teacher-forced next-token BLOCK logits over given per-row
        action-id sequences (CODEC space) against an already-encoded
        memory — the masked-contrast (mcselect) masked-reference forward: the
        planner-less prompt's distribution at every position of a
        sequence DECODED ELSEWHERE (under a candidate's conditioning).
        The suffix is exactly the deployment fast path's —
        ``[opener][BOA][a_1..a_T]`` — so row ``r``'s output ``t``
        predicts ``action_ids[r][t]`` from the same scaffold the
        conditioned decode ran. CONSUMES the memory's cache
        (snapshot/restore around calls — the sampled-draws convention).
        ``None`` rows are batch filler (PAD-padded, causal attention
        keeps them from touching anything) and return ``None``; an
        EMPTY non-None sequence is a caller bug (every decode emits at
        least one symbol) and raises. Returns per-row
        ``[len(ids), vocab_total]`` float32."""
        config = self.config
        base = config.block_base
        boa_backbone = base + self.codec.boa
        pad_backbone = base + self.codec.pad
        opener = list(self.opener_ids)
        k = len(opener)
        t_max = 0
        for ids in action_ids:
            if ids is None:
                continue
            if not ids:
                raise ValueError(
                    "teacher_forced_block_logits got an empty action-id "
                    "sequence — every decode emits at least one symbol; "
                    "pass None for filler rows",
                )
            t_max = max(t_max, len(ids))
        if t_max == 0:
            return [None for _ in action_ids]
        # The trunk's device, not the decoder's: parameterless concretes
        # (MolmoAct2 — trunk-native rows, zero own parameters) have no
        # parameters to read a device from, and the suffix runs through
        # the trunk either way.
        device = next(backbone.parameters()).device
        width = k + 1 + t_max
        rows: list[list[int]] = []
        for ids in action_ids:
            seq = [*opener, boa_backbone, *(base + i for i in (ids or []))]
            seq.extend([pad_backbone] * (width - len(seq)))
            rows.append(seq)
        tokens = torch.tensor(rows, dtype=torch.long, device=device)
        logits = self.forward(backbone, memory, tokens)
        return [
            (
                None
                if ids is None
                else logits[row, k : k + len(ids), base : base + config.vocab_total]
                .detach()
                .float()
            )
            for row, ids in enumerate(action_ids)
        ]

    @torch.no_grad()
    def decode_value_line(
        self,
        backbone: B,
        memory: ObservationMemory,
        *,
        field: AuxField,
        sampling: ARSampling | None = None,
    ) -> list[ValueCandidate]:
        """Text-ONLY decode of one free-text value line per row against
        the memory's CURRENT cache state — the subgoal-draws candidate
        instrument. The caller owns the cache protocol:
        ``cache_restore`` to the post-prefill snapshot before every
        call, so all candidates (and the full greedy pass) share one
        prefill. The value loop mirrors :meth:`predict_chunk`'s free
        value phase op-for-op (same text mask, same min_value greedy
        argmax, same budget/terminator forcing), so the greedy
        (``sampling=None``) candidate's text is bit-identical to the
        full pass's parsed field value — asserted by the model-level
        caller, never assumed. ``sampling`` switches emission to the
        temperature Gumbel-max draw (:func:`_sample_action_ids`, the
        trained sampled-draws machinery); the recorded distributions
        are the same masked value softmax either way (sampling changes
        which id is emitted, never the stats). BOA is never fed and no
        actions decode — the candidate texts are the entire output."""
        config = self.config
        runtime = self.aux_runtime
        if runtime is None:
            raise ValueError(
                "decode_value_line on a decoder without an AuxRuntime — "
                "loading built the model without the aux record",
            )
        trained = () if config.aux is None else config.aux.fields
        if field not in trained:
            raise ValueError(
                f"decode_value_line for {field.value!r}, but the checkpoint "
                f"trained aux fields {[f.value for f in trained] or 'NONE'}",
            )
        if field in runtime.value_candidates:
            raise ValueError(
                f"{field.value!r} is a constrained field (classification "
                "over fixed candidates) — free-text candidate decoding "
                "does not apply",
            )
        streams = next(iter(memory.streams.values()))
        batch_size = int(streams.key.shape[0])
        device = streams.key.device
        if sampling is not None and len(sampling.rngs) != batch_size:
            raise ValueError(
                f"sampling carries {len(sampling.rngs)} row RNGs for a "
                f"batch of {batch_size} — one keyed stream per row",
            )
        vocab = config.block_base + config.vocab_total
        min_value: float = torch.finfo(torch.float32).min
        # The text-only mask, exactly predict_chunk's construction.
        text_allowed = torch.zeros(vocab, dtype=torch.bool, device=device)
        text_allowed[: config.block_base] = True
        if self.newline_carrier_ids:
            text_allowed[
                torch.tensor(
                    sorted(self.newline_carrier_ids),
                    dtype=torch.long,
                    device=device,
                )
            ] = False
        allowed_vocab = int(text_allowed.sum())
        terminator = runtime.terminator_id
        budget = VALUE_BUDGETS[field]
        fed = 0
        feed: Tensor = (
            torch.tensor([list(self.opener_ids)], dtype=torch.long, device=device)
            .expand(batch_size, -1)
            .contiguous()
        )
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        row_ids: list[list[int]] = [[] for _ in range(batch_size)]
        chosen_stats: list[list[float]] = [[] for _ in range(batch_size)]
        mean_stats: list[list[float]] = [[] for _ in range(batch_size)]
        truncated = [False] * batch_size
        for step in range(budget + 1):
            logits, fed = self._step(backbone, memory, feed, fed)
            logits = logits[:, :vocab]
            masked = logits.masked_fill(~text_allowed, min_value)
            if sampling is None:
                next_ids = masked.argmax(dim=-1)
            else:
                next_ids = _sample_action_ids(
                    logits,
                    text_allowed.expand(batch_size, -1),
                    sampling,
                )
            # Stats from the SAME masked softmax the emission came from
            # (fp32; illegal ids carry ~-inf log-probs and are excluded
            # from the mean by indexing the allowed columns).
            logprobs = torch.log_softmax(masked, dim=-1)
            step_mean = logprobs[:, text_allowed].mean(dim=-1)
            step_chosen = logprobs.gather(1, next_ids[:, None]).squeeze(1)
            if step == budget:
                forced = ~done & (next_ids != terminator)
                for row in forced.nonzero().flatten().tolist():
                    truncated[row] = True
                next_ids = torch.where(
                    forced,
                    torch.full_like(next_ids, terminator),
                    next_ids,
                )
            next_ids = torch.where(
                done,
                torch.full_like(next_ids, terminator),
                next_ids,
            )
            for row, next_id in enumerate(next_ids.tolist()):
                if done[row] or (truncated[row] and step == budget):
                    continue
                chosen_stats[row].append(float(step_chosen[row]))
                mean_stats[row].append(float(step_mean[row]))
                if next_id != terminator:
                    row_ids[row].append(next_id)
            done |= next_ids == terminator
            feed = next_ids[:, None]
            if bool(done.all()):
                break
        return [
            ValueCandidate(
                # The _parse_aux stripped convention: conditioning on a
                # candidate is byte-compatible with conditioning on a
                # pass-1 parsed value.
                text=self.tokenizer.decode(row_ids[row]).strip(),
                truncated=truncated[row],
                chosen_logprob=tuple(chosen_stats[row]),
                mean_logprob=tuple(mean_stats[row]),
                allowed_vocab=allowed_vocab,
            )
            for row in range(batch_size)
        ]

    def _step(
        self,
        backbone: B,
        memory: ObservationMemory,
        feed: Tensor,
        fed: int,
    ) -> tuple[Tensor, int]:
        """Feed BACKBONE-id tokens against the growing cache; returns
        (last-position fp32 logits [B, vocab], new fed count)."""
        hidden = self._suffix_hidden(backbone, memory, feed, fed)
        logits = self._logits(backbone, hidden)[:, -1, :].float()
        return logits, fed + feed.shape[1]

    @staticmethod
    def cache_snapshot(
        memory: ObservationMemory,
    ) -> tuple[int, list[tuple[Tensor | None, Tensor | None]]]:
        """Capture the prefix cache by REFERENCE so N sampled decodes
        share one prefill (draws differ only in suffix K/V — the
        cheaper-per-draw fairness caveat of the sampled-draws
        instrument, made literal). Sound because every trunk cache is
        append-only: ``update()`` rebinds ``layer.keys``/``values`` to
        NEW tensors (cat/slice) and never writes into stored ones, so
        restoring the old references recovers the exact prefill state
        at zero copy cost. A future in-place (preallocated) cache
        breaks this contract — its snapshot must copy."""
        cache: Any = memory.cache
        if cache is None:
            raise ValueError(
                "cache_snapshot on a memory without a retained prefix "
                "cache — encode with an ARSuffixDecoder composed",
            )
        return cache.seen_tokens, [(layer.keys, layer.values) for layer in cache.layers]

    @staticmethod
    def cache_restore(
        memory: ObservationMemory,
        snapshot: tuple[int, list[tuple[Tensor | None, Tensor | None]]],
    ) -> None:
        """Rewind the memory's cache to a :meth:`cache_snapshot` — the
        suffix K/V appended since are dropped by rebinding."""
        cache: Any = memory.cache
        if cache is None:
            raise ValueError("cache_restore on a memory without a cache")
        seen_tokens, layers = snapshot
        cache.seen_tokens = seen_tokens
        for layer, (keys, values) in zip(cache.layers, layers, strict=True):
            layer.keys, layer.values = keys, values


def _parse_aux(
    generate: tuple[AuxField, ...],
    texts: list[str],
) -> AuxGeneration:
    """Zip one row's decoded value lines with the request order (the
    headerless protocol: which field a line answers is pinned by the
    request, not by generated bytes); lenient value parses — failures
    are None, never exceptions. ``text`` is the display form (field
    names re-attached) for reports."""
    # An EMPTY line (budget-0 fallback, degenerate emission) is a
    # missing value, not "" — the display then shows the field's
    # absence honestly.
    values: dict[AuxField, str] = {
        aux_field: stripped
        for aux_field, text in zip(generate, texts, strict=True)
        if (stripped := text.strip()) != ""
    }
    holding = values.get(AuxField.HOLDING)
    progress = values.get(AuxField.PROGRESS, "")
    parsed_progress: float | None = None
    if progress.endswith("%"):
        try:
            parsed_progress = float(progress[:-1]) / 100.0
        except ValueError:
            parsed_progress = None
    return AuxGeneration(
        text=display_text(values),
        subgoal=values.get(AuxField.SUBGOAL),
        holding=None if holding is None else holding == "yes",
        progress=parsed_progress,
        event=values.get(AuxField.EVENT),
        visible=values.get(AuxField.VISIBLE),
    )


def suffix_targets(
    decoder: ARSuffixDecoder[Any],
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor, Tensor | None]:
    """(full suffix ids [B, 1+S], shifted CE targets [B, S], aux-position
    mask [B, S] | None) — the data-only half of the objective (no model
    forward), shared by the loss and by chunked backward's normalizer
    counts. Semantics in :func:`ar_backbone_losses`'s docstring."""
    base = decoder.config.block_base
    if batch.suffix_tokens is not None:
        is_aux_content = batch.suffix_is_aux
        assert is_aux_content is not None  # collator invariant
        content = batch.suffix_tokens
    else:
        tokens = batch.action_tokens
        if tokens is None:
            raise SystemExit(
                "batch carries no action_tokens — build the Collator with "
                "an ActionCodec (--fast-tokenizer) to train an AR decoder",
            )
        content = tokens + base
        is_aux_content = None
    prefix = torch.tensor(
        [decoder.opener_ids],
        dtype=torch.long,
        device=content.device,
    ).expand(content.shape[0], -1)
    full = torch.cat([prefix, content], dim=1)
    pad_id = base + decoder.codec.pad
    # forward: position j predicts suffix position j + 1, so
    # logits[:, j] pairs with full[:, j + 1] — targets are the SHIFTED
    # sequence. Opener tokens are constants: ignore every target that
    # IS an opener token (shifted indices 0..len(prefix)−2); the last
    # opener position's target — content[0], the first value token or
    # BOA on [generate|actions] rows — stays trained. max(…, 0): an
    # EMPTY opener (the MolmoAct2 format — the prompt carries the whole
    # scaffold) must mask nothing; the raw -1 would slice [:, :-1] and
    # silently ignore every target but the last.
    targets = full[:, 1:].clone()
    targets[:, : max(prefix.shape[1] - 1, 0)] = IGNORE_INDEX
    targets[targets == pad_id] = IGNORE_INDEX
    if is_aux_content is None:
        return full, targets, None
    is_aux = torch.cat(
        [torch.zeros_like(prefix, dtype=torch.bool), is_aux_content],
        dim=1,
    )[:, 1:]
    return full, targets, is_aux


def ar_backbone_counts(
    decoder: ARSuffixDecoder[Any],
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor | None]:
    """(action target count, aux target count | None) for one batch —
    cheap tensor ops only, no backbone forward: chunked backward reads
    the FULL-batch normalizers off every chunk before the first
    forward, so per-chunk sums divide by global counts (exactly the
    unchunked token-weighted mean, not a chunk-mean-of-means)."""
    _, targets, is_aux = suffix_targets(decoder, batch)
    valid = targets != IGNORE_INDEX
    if is_aux is None:
        return valid.sum(), None
    return (valid & ~is_aux).sum(), (valid & is_aux).sum()


def ar_backbone_loss_sums[B: nn.Module](
    backbone: B,
    decoder: ARSuffixDecoder[B],
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Sum-form objective for chunked backward: (action CE SUM with
    graph, action target count, aux CE SUM with graph | None, aux
    target count | None). The caller owns normalization — dividing by
    FULL-batch counts and summing over chunks reproduces
    :func:`ar_backbone_losses`'s token-weighted means exactly (up to fp
    reduction order). The mean-form stays the single-batch path
    untouched (its reductions are byte-anchored by the loss oracles)."""
    full, targets, is_aux = suffix_targets(decoder, batch)
    logits = decoder(backbone, memory, full[:, :-1])
    elementwise = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(targets.shape)
    valid = targets != IGNORE_INDEX
    # Mask-multiply, never boolean-index: advanced indexing runs
    # nonzero + a host sync per reduction (x chunks/step on the live
    # path). CE with reduction="none" writes exact 0.0 at IGNORE_INDEX
    # positions, so these sums equal the indexed sums up to fp
    # reduction order (the CPU
    # loss oracles re-pin on the sum form, mean form untouched).
    if is_aux is None:
        return elementwise.sum(), valid.sum(), None, None
    aux_positions = valid & is_aux
    action_positions = valid & ~is_aux
    return (
        (elementwise * action_positions).sum(),
        action_positions.sum(),
        (elementwise * aux_positions).sum(),
        aux_positions.sum(),
    )


def ar_backbone_losses[B: nn.Module](
    backbone: B,
    decoder: ARSuffixDecoder[B],
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Teacher-forced FULL-VOCABULARY cross-entropy over the format-5
    suffix ``[opener][value lines][BOA][actions]`` — (total with
    graph, action component, aux CE SUM | None, aux position count |
    None). Aux rides as sum+count (not a mean) so the train loop can
    aggregate a position-weighted mean across batches and ranks — a
    per-batch mean dilutes toward 0 on sparsely-labeled corpora.

    The opener is prepended HERE (the collator supplies content —
    request-consistent value lines + BOA + actions — and the request
    itself rides the PROMPT). Opener positions predict constants and
    are IGNOREd except the last, whose target is content[0]: the first
    value token, or BOA on ``[generate|actions]`` rows. Component
    split: ``action`` = mean CE over action targets incl. BOA
    (pretrain scale); total = action + aux_loss_weight * (aux_sum /
    aux_count) — batch-mean semantics, 0-safe when a batch has no
    labeled sample.
    """
    full, targets, is_aux = suffix_targets(decoder, batch)
    logits = decoder(backbone, memory, full[:, :-1])
    if is_aux is None:
        action = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return action, action, None, None
    elementwise = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(targets.shape)
    valid = targets != IGNORE_INDEX
    aux_positions = valid & is_aux
    action_positions = valid & ~is_aux
    action = elementwise[action_positions].sum() / action_positions.sum().clamp(min=1)
    aux_count = aux_positions.sum()
    aux_sum = elementwise[aux_positions].sum()
    total = action + decoder.aux_loss_weight * (aux_sum / aux_count.clamp(min=1))
    return total, action, aux_sum, aux_count


def ar_backbone_loss[B: nn.Module](
    backbone: B,
    decoder: ARSuffixDecoder[B],
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> Tensor:
    """Scalar objective (see :func:`ar_backbone_losses`; the sum forms
    below are what the family loss paths compose — this is the plain
    mean-form entry)."""
    total, _, _, _ = ar_backbone_losses(backbone, decoder, memory, batch)
    return total
