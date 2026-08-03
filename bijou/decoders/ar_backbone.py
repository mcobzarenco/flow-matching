"""Decoder-only action decoder: the backbone's suffix role (ar_backbone).

The prompt — which carries the ``[generate|…]`` request conditioning
and the projected state token (prompt format 3) — is prefill-encoded
once (the ObservationMemory retains the full prefix KV cache); this
decoder continues the suffix-format-5 sequence
``[<|turn>model\\n][value\\n per requested][BOA][t_1..t_k]`` through ALL
backbone layers — the KV-shared deep half included — and reads
FULL-VOCABULARY logits from the frozen tied LM head with the FAST
block's columns supplied by a trainable patch. Aux value lines
(headerless, request-ordered — bijou.aux_text) and action tokens share
that one softmax. What speaks is commanded by the PROMPT's request
set: the model learns p(value | observation, asked) and is never asked
to infer judged-ness from appearance.

Ownership: this module owns ONLY the new parameters (~11M at E2B scale)
— the FAST block's input-embedding + per-layer-embedding rows. The
backbone is owned by BijouModel and passed into every call; the state
projection is PROMPT-side and lives in GemmaEncoder (format 3 moved
state into the user turn).

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
(:meth:`ARBackboneDecoder.predict_chunk`) is fully scaffolded by the
request set: per requested field, greedy value ids under a per-field
budget until the ``\\n`` terminator (constrained candidates where
defined), then BOA is FORCED and the ar_fast-style grammar mask
decodes the chunk by remaining symbol budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..aux_text import (
    GENERATION_OPENER,
    SUFFIX_FORMAT,
    VALUE_BUDGETS,
    AuxDecodeConfig,
    AuxField,
    AuxGeneration,
    AuxRuntime,
    TextTokenizer,
    display_text,
)
from ..fast.codec import ActionCodec
from ..gemma4.config import Gemma4TextConfig
from ..gemma4.model import Gemma4Model
from ..interface import BijouPrediction, CollatedBatch, ObservationMemory
from ..nn import DeviceLike
from .ar_fast import IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class ARBackboneConfig:
    """Construction config of the decoder-only path. Deliberately tiny:
    the backbone defines all geometry; this records only the action
    vocabulary and its placement. ``vocab_total`` = BPE vocabulary +
    BOA + PAD (ActionCodec's convention); ``block_base`` is the first
    backbone vocabulary id of the reserved block."""

    tokenizer: str  # artifact ref (local dir or <user>/<repo>/<subfolder>)
    vocab_total: int
    block_base: int
    chunk_size: int
    action_dim: int
    # Suffix format (aux_text.SUFFIX_FORMAT when written): 5 = the
    # request-conditioned headerless format every new run trains.
    # Formats ≤ 4 (fed mode tokens, suffix state slot, header bytes)
    # are REFUSED at construction — no trained artifact worth loading
    # exists (owner call, 2026-08-03; the parameter sets are
    # incompatible anyway: state_proj moved prompt-side, mode tables
    # deleted).
    suffix_format: int
    # Aux text record: template version + fields + label provenance
    # (None = trained without aux — every sample was [generate|actions]).
    aux: AuxDecodeConfig | None

    def __post_init__(self) -> None:
        if self.vocab_total < 3:  # ≥ 1 body token + BOA + PAD
            raise ValueError(f"vocab_total {self.vocab_total} is not a FAST vocabulary")
        if self.block_base < 0:
            raise ValueError(f"block_base {self.block_base} must be non-negative")
        if self.suffix_format != SUFFIX_FORMAT:
            raise ValueError(
                f"suffix format {self.suffix_format} != {SUFFIX_FORMAT}: "
                "pre-5 checkpoints trained mode tokens/suffix state/header "
                "bytes this code no longer implements — retrain (no "
                "back-compat, 2026-08-03)",
            )


class ARBackboneDecoder(nn.Module):
    """The backbone's suffix role (see the module docstring).

    ``codec`` is a runtime resource (BPE + quantile glue), not a module;
    checkpoints reference the tokenizer artifact by id. ``text_config``
    is the FULL backbone architecture — construction validates the block
    placement and that the KV-shared deep half is present (a truncated
    backbone has none; this decoder is definitionally the full stack's
    suffix role)."""

    def __init__(
        self,
        config: ARBackboneConfig,
        text_config: Gemma4TextConfig,
        codec: ActionCodec,
        *,
        tokenizer: TextTokenizer | None,
        aux_runtime: AuxRuntime | None = None,
        aux_loss_weight: float = 1.0,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
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
        # Cumulative value-budget exhaustions (decode health metric).
        self.fallback_count = 0
        self.opener_ids: tuple[int, ...] = tuple(
            tokenizer.encode(GENERATION_OPENER, add_special_tokens=False),
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
        if config.block_base + config.vocab_total > text_config.vocab_size:
            raise ValueError(
                f"FAST block [{config.block_base}, "
                f"{config.block_base + config.vocab_total}) does not fit the "
                f"backbone vocabulary ({text_config.vocab_size})",
            )
        if text_config.num_kv_shared_layers == 0:
            raise ValueError(
                "ar_backbone needs the FULL backbone (its suffix runs the "
                "KV-shared deep half); this config is a truncated prefix — "
                "load with depth=full",
            )
        hidden = text_config.hidden_size
        self.num_layers = text_config.num_hidden_layers
        self.ple_dim = text_config.hidden_size_per_layer_input
        # Same scale asymmetry as the backbone's own tying: input lookups
        # are multiplied by √dim (ScaledEmbedding), the head uses raw rows.
        self.embed_scale = hidden**0.5
        self.ple_scale = self.ple_dim**0.5
        self.fast_embed = nn.Embedding(
            config.vocab_total,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.fast_ple = nn.Embedding(
            config.vocab_total,
            self.num_layers * self.ple_dim,
            device=device,
            dtype=dtype,
        )
        # Constrained decoding needs each token's symbol expansion length
        # (one BPE piece = a run of quantized DCT coefficients). Specials
        # stay 0 and are handled explicitly in the decode mask. Plain
        # attribute, not a buffer: derived from the codec, never saved.
        symbol_lengths = torch.zeros(config.vocab_total, dtype=torch.long)
        for token_id in range(codec.tokenizer.vocab_size):
            piece = codec.tokenizer.bpe.id_to_token(token_id)
            assert piece is not None, f"BPE id {token_id} has no piece"
            symbol_lengths[token_id] = len(piece)
        if not bool((symbol_lengths == 1).any()):
            raise ValueError(
                "BPE vocabulary has no single-symbol token — exact fill "
                "(and decode termination) cannot be guaranteed",
            )
        self.symbol_lengths = symbol_lengths
        if device is None or torch.device(device).type != "meta":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Fallback init: patch tables at the text-embedding-typical 0.02
        std. Training warm-up should prefer
        :meth:`init_tables_from_backbone`."""
        nn.init.normal_(self.fast_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fast_ple.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def init_tables_from_backbone(self, backbone: Gemma4Model) -> None:
        """Re-init the patch rows around the REAL tables' row mean
        (+0.02 noise) — the block's logits then start near the average
        text logit instead of at an arbitrary offset, which matters under
        full-vocabulary CE (the data contract's extended-vocab recipe)."""
        text = backbone.language_model
        embed_mean = text.embed_tokens.weight.float().mean(dim=0)
        ple_mean = text.embed_tokens_per_layer.weight.float().mean(dim=0)
        for table, mean in (
            (self.fast_embed.weight, embed_mean),
            (self.fast_ple.weight, ple_mean),
        ):
            noise = torch.randn_like(table) * 0.02
            table.copy_(mean.to(table.dtype)[None, :] + noise)

    def _suffix_inputs_backbone_ids(
        self,
        backbone: Gemma4Model,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """(inputs_embeds, per_layer_inputs) for BACKBONE-id ``tokens``
        [B, T], routed by range: text ids (< block_base — opener + value
        lines) embed through the frozen embed_tokens/PLE tables, block
        ids through the FAST patch. An all-block suffix reproduces the
        pre-aux computation bitwise (torch.where with an all-False mask
        returns the block side elementwise). Returns
        ([B, T, hidden], [B, T, num_layers, ple_dim]) in the backbone's
        dtype."""
        text = backbone.language_model
        target_dtype = text.embed_tokens.weight.dtype
        is_text = (tokens < self.config.block_base)[..., None]
        block_ids = (tokens - self.config.block_base).clamp(min=0)
        # Text-side lookups use the pad row at block positions
        # (discarded by the select) — every id stays in range for every
        # table.
        text_ids = torch.where(
            is_text[..., 0],
            tokens,
            torch.full_like(tokens, text.config.pad_token_id),
        )
        embeds = torch.where(
            is_text,
            text.embed_tokens(text_ids).float(),
            self.fast_embed(block_ids) * self.embed_scale,
        )
        ple = torch.where(
            is_text[..., None],
            text.get_per_layer_inputs(text_ids).float(),
            (self.fast_ple(block_ids) * self.ple_scale).view(
                tokens.shape[0],
                tokens.shape[1],
                self.num_layers,
                self.ple_dim,
            ),
        )
        return embeds.to(target_dtype), ple.to(target_dtype)

    def _patched_logits(self, backbone: Gemma4Model, hidden: Tensor) -> Tensor:
        """Full-vocabulary logits with the FAST block's columns computed
        from the trainable patch, softcapped AFTER the overwrite so the
        block is capped identically to text (Gemma4Model.forward
        semantics). hidden [B, S, hidden] → [B, S, vocab_size].

        Memory discipline (a [B, S, 262k] tensor is ~1.2 GiB fp32 at
        B10 — an out-of-place chain here OOM'd the first full-recipe
        run): the block columns are written IN PLACE (identical values
        and gradient routing to the old cat splice — the overwritten
        head columns received no gradient there either); the softcap's
        div runs in place (scalar backward saves nothing) and tanh_ in
        place, but the trailing scale must be OUT-of-place — tanh_
        saves its output for backward, and a further in-place op on it
        trips the version counter (measured, not theorized). Two
        full-vocab tensors live instead of the old ~four."""
        base = self.config.block_base
        end = base + self.config.vocab_total
        logits = backbone.lm_head(hidden)
        block = hidden @ self.fast_embed.weight.to(hidden.dtype).T
        logits[..., base:end] = block
        softcap = backbone.config.text.final_logit_softcapping
        if softcap is not None:
            logits = logits.div_(softcap).tanh_() * softcap
        return logits

    def _continue_suffix(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        embeds: Tensor,
        per_layer: Tensor,
        fed: int,
    ) -> Tensor:
        """Run S suffix embeddings through ALL layers against the prefix
        cache (which they extend in place). ``fed`` = suffix positions
        already in the cache from previous calls (decode loop); positions
        continue per-sample after each REAL prompt length + fed.
        Returns final-normed hidden states [B, S, hidden]."""
        cache = memory.cache
        if cache is None:
            raise ValueError(
                "ObservationMemory carries no prefix cache — encode with "
                "retain_cache=True (BijouModel does this for ar_backbone)",
            )
        batch, seq_len, _ = embeds.shape
        device = embeds.device
        offsets = torch.arange(seq_len, device=device)[None, :] + fed
        if memory.padding_mask is not None:
            real = memory.padding_mask.to(device=device, dtype=torch.bool)
            positions = real.long().sum(dim=1, keepdim=True) + offsets
            full_mask = torch.cat(
                [
                    real,
                    torch.ones(
                        (batch, fed + seq_len),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )
        else:
            positions = torch.full((batch, 1), memory.length, device=device) + offsets
            full_mask = None
        # cuDNN's fused-attention graph intermittently fails to EXECUTE
        # its backward on the suffix geometry (bf16 head_dim-512 queries
        # at ragged lengths against the prefix cache) — the
        # pytorch/pytorch#122695 'mha_graph.execute is_good()' assert
        # family. It killed the fullstack run twice (steps 10440,
        # ~20500) starting exactly when the suffix went bf16 and thus
        # became cuDNN-eligible; the TORCH_CUDNN_SDPA_ENABLED env var is
        # NOT honored by this torch (2.11) — verified the hard way. Pin
        # THIS call to the non-cuDNN kernels (the backend chosen at
        # forward time also selects the backward); the prefix encode —
        # ~80% of compute, crash-free for >100k steps — keeps the full
        # dispatcher including cuDNN. No-op on CPU/eager paths.
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
        ):
            return backbone.language_model(
                inputs_embeds=embeds,
                per_layer_inputs=per_layer,
                position_ids=positions,
                padding_mask=full_mask,
                cache=cache,
            )

    @override
    def forward(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        tokens: Tensor,
    ) -> Tensor:
        """Next-token logits over the suffix (BACKBONE ids — mixed value
        lines + FAST block).

        CONSUMES the memory's cache (suffix K/V are appended): encode a
        fresh memory per call — training does, and the decode loop is
        exactly the incremental consumer this enables.

        Shapes:
          - memory.cache: prefix K/V of every non-shared layer, [B, ...]
          - tokens: [B, T]  (long; backbone ids)
          - returns: [B, T, vocab_size]  (FULL backbone vocabulary;
            position j predicts suffix position j + 1)
        """
        embeds, per_layer = self._suffix_inputs_backbone_ids(backbone, tokens)
        hidden = self._continue_suffix(backbone, memory, embeds, per_layer, fed=0)
        return self._patched_logits(backbone, hidden)

    @torch.no_grad()
    def predict_chunk(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generate: tuple[AuxField, ...] = (),
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> BijouPrediction:
        """The single decode path, fully scaffolded by the request set.
        Deterministic greedy; ``generator``/``noise`` unused/must be
        None.

        ``generate`` must equal the request the PROMPT was collated with
        (the collator's ``generate_override``; the memory was encoded
        from that prompt — mismatched scaffolds would sit off the
        conditioning). Per requested field, in order: greedy text-id
        decode under the field's VALUE_BUDGETS cap until the ``\\n``
        terminator (budget exhaustion forces it and counts a loud
        fallback; HOLDING constrains the first token to its candidate
        set). After the last field — immediately, for ``generate=()``,
        the deployment fast path — BOA is FORCED (its target is trained;
        its identity is not a decision) and the ar_fast grammar mask
        decodes the chunk by remaining symbol budget, PAD once finished.

        Requires an aux-trained checkpoint for non-empty ``generate``
        (requested-but-untrained fields would be elicited off-manifold).

        Returns a BijouPrediction: chunks [B, chunk, action_dim] raw
        units + one AuxGeneration per row (empty for ``generate=()``)."""
        if noise is not None:
            raise ValueError("ARBackboneDecoder.predict_chunk takes no noise")
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
            next_ids = logits.masked_fill(~action_allowed, min_value).argmax(dim=-1)
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

    def _step(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        feed: Tensor,
        fed: int,
    ) -> tuple[Tensor, int]:
        """Feed BACKBONE-id tokens against the growing cache; returns
        (last-position fp32 logits [B, vocab], new fed count)."""
        embeds, per_layer = self._suffix_inputs_backbone_ids(backbone, feed)
        hidden = self._continue_suffix(backbone, memory, embeds, per_layer, fed)
        logits = self._patched_logits(backbone, hidden)[:, -1, :].float()
        return logits, fed + embeds.shape[1]


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


def ar_backbone_losses(
    backbone: Gemma4Model,
    decoder: ARBackboneDecoder,
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
    logits = decoder(backbone, memory, full[:, :-1])
    pad_id = base + decoder.codec.pad
    # forward: position j predicts suffix position j + 1, so
    # logits[:, j] pairs with full[:, j + 1] — targets are the SHIFTED
    # sequence. Opener tokens are constants: ignore every target that
    # IS an opener token (shifted indices 0..len(prefix)−2); the last
    # opener position's target — content[0], the first value token or
    # BOA on [generate|actions] rows — stays trained.
    targets = full[:, 1:].clone()
    targets[:, : prefix.shape[1] - 1] = IGNORE_INDEX
    targets[targets == pad_id] = IGNORE_INDEX
    if is_aux_content is None:
        action = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return action, action, None, None
    is_aux = torch.cat(
        [torch.zeros_like(prefix, dtype=torch.bool), is_aux_content],
        dim=1,
    )[:, 1:]
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


def ar_backbone_loss(
    backbone: Gemma4Model,
    decoder: ARBackboneDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> Tensor:
    """Scalar objective (see :func:`ar_backbone_losses`; BijouModel.loss
    dispatches here — the train step calls the tuple form for component
    logging)."""
    total, _, _, _ = ar_backbone_losses(backbone, decoder, memory, batch)
    return total
