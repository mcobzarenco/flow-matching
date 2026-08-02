"""Decoder-only action decoder: the backbone's suffix role (ar_backbone).

The prompt is prefill-encoded once (the ObservationMemory retains the
full prefix KV cache); this decoder continues ``[state][BOA][t_1..t_k]``
through ALL backbone layers — the KV-shared deep half included — and
reads FULL-VOCABULARY logits from the frozen tied LM head with the FAST
block's columns supplied by a trainable patch. Aux text outputs (future)
extend the same head; today every predicted position is an action token
and decoding is grammar-constrained inside the block from the seeded
BOA onward. (Aux + actions will share one softmax: text first, then the
grammar-fixed action block, one head.)

Ownership: this module owns ONLY the new parameters (~11M at E2B scale)
— the state projection (suffix position 0, zero-initialized so the
prompt-conditioned computation starts undisturbed) and the FAST block's
input-embedding + per-layer-embedding rows. The backbone is owned by
BijouModel and passed into every call; ``expert.safetensors`` therefore
stays exactly "the new parameters".

Id spaces: the collator's ``action_tokens`` and the codec speak CODEC
ids [0, vocab_total); the head/targets speak backbone ids
``block_base + codec_id`` (a contiguous run of reserved-unused vocabulary
ids — E2B: 258885.., inside the 3259-id unused tail). The backbone never
consumes FAST ids as ids: suffix tokens enter as patch embeddings via
``inputs_embeds``/``per_layer_inputs``; the block ids exist so action
and text tokens share one softmax (full-vocab CE now, aux text later).

Sequence/loss conventions mirror ``ar_fast``: BOA seeds and is never
predicted (the state position's constant target is ignored), PAD is
batch padding only, there is no EOA (length is fixed by the FAST
grammar), and constrained greedy decode masks to body tokens whose BPE
symbol expansion fits the remaining chunk*dim budget.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..aux_text import AuxDecodeConfig, AuxField, AuxGeneration, AuxRuntime
from ..fast.codec import ActionCodec
from ..gemma4.cache import KVCache
from ..gemma4.config import Gemma4TextConfig
from ..gemma4.model import Gemma4Model
from ..interface import CollatedBatch, MemoryStream, NormStats, ObservationMemory
from ..nn import DeviceLike
from .ar_fast import IGNORE_INDEX

# Free-text aux value caps (safety net at decode; training truncates via
# AuxSpec.max_subgoal_tokens).
MAX_SUBGOAL_DECODE_TOKENS = 24
MAX_PROGRESS_DECODE_TOKENS = 6


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
    state_dim: int
    chunk_size: int
    action_dim: int
    # Aux text record: template version + fields + label provenance
    # (None = trained without aux; forced-scaffold decode then refuses).
    aux: AuxDecodeConfig | None

    def __post_init__(self) -> None:
        if self.vocab_total < 3:  # ≥ 1 body token + BOA + PAD
            raise ValueError(f"vocab_total {self.vocab_total} is not a FAST vocabulary")
        if self.block_base < 0:
            raise ValueError(f"block_base {self.block_base} must be non-negative")


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
        aux_runtime: AuxRuntime | None = None,
        aux_loss_weight: float = 1.0,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.codec = codec
        # Runtime resources, not modules (the codec convention): scaffold
        # ids for forced-field decoding + the aux loss mixture weight.
        self.aux_runtime = aux_runtime
        self.aux_loss_weight = aux_loss_weight
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
        self.state_proj = nn.Linear(
            config.state_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
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
        std; the state projection at ZERO so suffix position 0 starts as
        an inert token (it still receives gradients through its K/V use —
        unlike a zero output head, nothing is blocked). Training warm-up
        should prefer :meth:`init_tables_from_backbone`."""
        nn.init.normal_(self.fast_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fast_ple.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.state_proj.weight)
        assert self.state_proj.bias is not None
        nn.init.zeros_(self.state_proj.bias)

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

    def _suffix_inputs(
        self,
        backbone: Gemma4Model,
        state: Tensor | None,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """(inputs_embeds, per_layer_inputs) for ``[state?][tokens...]``,
        in the backbone's dtype. The state slot takes the pad token's PLE
        row — the exact precedent image soft tokens set in
        embed_multimodal.

        Shapes: state [B, state_dim] or None (decode steps past the
        first feed); tokens [B, T] CODEC ids; returns
        ([B, S, hidden], [B, S, num_layers, ple_dim]) with
        S = T (+1 with state)."""
        return self._suffix_inputs_backbone_ids(
            backbone,
            state,
            tokens + self.config.block_base,
        )

    def _suffix_inputs_backbone_ids(
        self,
        backbone: Gemma4Model,
        state: Tensor | None,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """As :meth:`_suffix_inputs` but ``tokens`` are BACKBONE ids: text
        ids (< block_base — the aux segment) embed through the frozen
        embed_tokens/PLE tables; block ids through the patch. An all-block
        suffix reproduces the pre-aux computation bitwise (torch.where
        with an all-False mask returns the block side elementwise)."""
        text = backbone.language_model
        target_dtype = text.embed_tokens.weight.dtype
        is_text = (tokens < self.config.block_base)[..., None]
        block_ids = (tokens - self.config.block_base).clamp(min=0)
        # Text-side lookups use the pad row at block positions (discarded
        # by the select) — every id stays in range for both tables.
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
        if state is not None:
            state_embed = self.state_proj(state.to(self.state_proj.weight.dtype))
            embeds = torch.cat([state_embed[:, None, :], embeds], dim=1)
            pad_ids = torch.full(
                (tokens.shape[0], 1),
                text.config.pad_token_id,
                dtype=torch.long,
                device=tokens.device,
            )
            ple = torch.cat([text.get_per_layer_inputs(pad_ids).float(), ple], dim=1)
        return embeds.to(target_dtype), ple.to(target_dtype)

    def _patched_logits(self, backbone: Gemma4Model, hidden: Tensor) -> Tensor:
        """Full-vocabulary logits with the FAST block's columns computed
        from the trainable patch, softcapped AFTER the overwrite so the
        block is capped identically to text (Gemma4Model.forward
        semantics). hidden [B, S, hidden] → [B, S, vocab_size]."""
        base = self.config.block_base
        end = base + self.config.vocab_total
        logits = backbone.lm_head(hidden)
        block = hidden @ self.fast_embed.weight.to(hidden.dtype).T
        logits = torch.cat([logits[..., :base], block, logits[..., end:]], dim=-1)
        softcap = backbone.config.text.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
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
        state: Tensor,
        tokens: Tensor,
    ) -> Tensor:
        """Next-token logits over the ``[state][tokens...]`` suffix.

        CONSUMES the memory's cache (suffix K/V are appended): encode a
        fresh memory per call — training does, and the decode loop is
        exactly the incremental consumer this enables.

        Shapes:
          - memory.cache: prefix K/V of every non-shared layer, [B, ...]
          - state: [B, state_dim]  (normalized)
          - tokens: [B, T]  (long; CODEC ids — BOA/body/PAD)
          - returns: [B, 1 + T, vocab_size]  (FULL backbone vocabulary;
            position j predicts the token at suffix position j + 1)
        """
        return self.forward_backbone_ids(
            backbone,
            memory,
            state,
            tokens + self.config.block_base,
        )

    def forward_backbone_ids(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        state: Tensor,
        tokens: Tensor,
    ) -> Tensor:
        """As :meth:`forward` but ``tokens`` are BACKBONE ids (mixed aux
        text + FAST block — the aux training path)."""
        embeds, per_layer = self._suffix_inputs_backbone_ids(backbone, state, tokens)
        hidden = self._continue_suffix(backbone, memory, embeds, per_layer, fed=0)
        return self._patched_logits(backbone, hidden)

    @torch.no_grad()
    def predict_chunk(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """CONSTRAINED greedy decode, then detokenize + denormalize with
        the batch's per-sample q01/q99 (the ar_fast contract: grammar
        mask by remaining symbol budget, no EOA, exactly one chunk,
        deterministic — ``generator``/``noise`` unused/must be None).

        Incremental: the first feed is ``[state][BOA]``, every later step
        feeds ONE token against the growing cache — the payoff of running
        the suffix natively on the backbone."""
        if noise is not None:
            raise ValueError("ARBackboneDecoder.predict_chunk takes no noise")
        stats = batch.action_stats
        if stats.q01 is None or stats.q99 is None:
            raise SystemExit(
                "batch stats carry no action quantiles — AR decoding needs "
                "the exact q01/q99 the tokenizer was fitted under (old "
                "checkpoint stats tables cannot drive AR inference)",
            )
        state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
        config = self.config
        batch_size = state.shape[0]
        device = state.device
        lengths = self.symbol_lengths.to(device)
        total_symbols = config.chunk_size * config.action_dim
        remaining = torch.full(
            (batch_size,),
            total_symbols,
            dtype=torch.long,
            device=device,
        )
        boa = self.codec.boa
        pad = self.codec.pad
        tokens = torch.full((batch_size, 1), boa, dtype=torch.long, device=device)
        min_value: float = torch.finfo(torch.float32).min
        base = config.block_base
        feed = tokens + base  # first feed carries [state][BOA]
        feed_state: Tensor | None = state
        fed = 0
        for _ in range(total_symbols):
            if bool((remaining == 0).all()):
                break
            logits, fed = self._step(backbone, memory, feed_state, feed, fed)
            # Grammar mask over the block slice: greedy never leaves the
            # block after BOA (aux text decoding runs the pre-BOA segment
            # over the text vocabulary instead — decode_with_aux).
            block = logits[:, base : base + config.vocab_total]
            allowed = (lengths[None, :] > 0) & (lengths[None, :] <= remaining[:, None])
            allowed[:, pad] = remaining == 0
            block = block.masked_fill(~allowed, min_value)
            next_token = block.argmax(dim=-1)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            remaining = remaining - lengths[next_token]
            feed = (next_token + base)[:, None]
            feed_state = None

        q01 = stats.q01.cpu().numpy()
        q99 = stats.q99.cpu().numpy()
        token_rows = tokens.cpu().tolist()
        chunks = [
            torch.from_numpy(
                self.codec.decode(
                    [t for t in row[1:] if t != pad],  # drop seed BOA
                    q01[row_index],
                    q99[row_index],
                ),
            ).float()
            for row_index, row in enumerate(token_rows)
        ]
        return torch.stack(chunks).to(device)

    def _step(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        state: Tensor | None,
        feed: Tensor,
        fed: int,
    ) -> tuple[Tensor, int]:
        """Feed BACKBONE-id tokens (plus the state token on the first
        call) against the growing cache; returns (last-position fp32
        logits [B, vocab], new fed count)."""
        embeds, per_layer = self._suffix_inputs_backbone_ids(backbone, state, feed)
        hidden = self._continue_suffix(backbone, memory, embeds, per_layer, fed)
        logits = self._patched_logits(backbone, hidden)[:, -1, :].float()
        return logits, fed + embeds.shape[1]

    @torch.no_grad()
    def decode_with_aux(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
    ) -> tuple[Tensor, list[AuxGeneration]]:
        """Forced-scaffold aux decode, then grammar-constrained actions —
        actions here are CONDITIONED on the self-generated aux text (the
        deliberate contrast with :meth:`predict_chunk`, which forces BOA
        immediately; comparing the two MAEs measures whether the
        scratchpad helps). Row-at-a-time (value lengths are ragged and
        forced headers must not interleave across rows); batch the
        callers, not this loop.

        Requires ``aux_runtime`` (a checkpoint trained without aux has
        nothing to elicit — loud error)."""
        runtime = self.aux_runtime
        if runtime is None:
            raise SystemExit(
                "decode_with_aux on a decoder without aux_runtime — the "
                "checkpoint's decoder config carries no aux section (it "
                "was trained before/without aux)",
            )
        chunks: list[Tensor] = []
        generations: list[AuxGeneration] = []
        for row in range(batch.state.shape[0]):
            row_batch = _row_slice(batch, row)
            row_memory = _row_memory(memory, row)
            chunk, generation = self._decode_row_with_aux(
                backbone,
                row_memory,
                row_batch,
                runtime,
            )
            chunks.append(chunk[0])
            generations.append(generation)
        return torch.stack(chunks), generations

    def _decode_row_with_aux(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        runtime: AuxRuntime,
    ) -> tuple[Tensor, AuxGeneration]:
        state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
        device = state.device
        base = self.config.block_base
        min_value: float = torch.finfo(torch.float32).min
        fed = 0
        generated: list[int] = []
        feed_state: Tensor | None = state
        pending = torch.empty((1, 0), dtype=torch.long, device=device)

        def feed_ids(ids: list[int]) -> None:
            nonlocal pending
            pending = torch.cat(
                [pending, torch.tensor([ids], dtype=torch.long, device=device)],
                dim=1,
            )

        def step() -> Tensor:
            """Flush pending feed (+state on the first call), return
            last-position logits."""
            nonlocal fed, feed_state, pending
            logits, fed = self._step(backbone, memory, feed_state, pending, fed)
            feed_state = None
            pending = torch.empty((1, 0), dtype=torch.long, device=device)
            return logits

        caps = {
            AuxField.SUBGOAL: MAX_SUBGOAL_DECODE_TOKENS,
            AuxField.PROGRESS: MAX_PROGRESS_DECODE_TOKENS,
        }
        for aux_field in runtime.config.fields:
            header = list(runtime.header_ids[aux_field])
            generated.extend(header)
            feed_ids(header)
            candidates = runtime.value_candidates.get(aux_field)
            if candidates is not None:
                logits = step()
                firsts = [c[0] for c in candidates]
                pick = int(torch.argmax(logits[0, firsts]))
                value = list(candidates[pick])
                generated.extend(value)
                feed_ids(value)
            else:
                for _ in range(caps[aux_field]):
                    logits = step()
                    # Aux values are TEXT: the FAST block (and beyond)
                    # is masked out during free decoding.
                    logits[0, base:] = min_value
                    next_id = int(logits[0].argmax())
                    if next_id == runtime.terminator_id:
                        break
                    generated.append(next_id)
                    feed_ids([next_id])
            generated.append(runtime.terminator_id)
            feed_ids([runtime.terminator_id])
        # Aux done: force BOA, then the standard grammar-masked loop.
        feed_ids([base + self.codec.boa])
        lengths = self.symbol_lengths.to(device)
        remaining = self.config.chunk_size * self.config.action_dim
        action_ids: list[int] = []
        while remaining > 0:
            logits = step()
            block = logits[:, base : base + self.config.vocab_total]
            allowed = (lengths > 0) & (lengths <= remaining)
            block = block.masked_fill(~allowed[None, :], min_value)
            next_token = int(block[0].argmax())
            action_ids.append(next_token)
            remaining -= int(lengths[next_token])
            feed_ids([base + next_token])
        stats = batch.action_stats
        assert stats.q01 is not None and stats.q99 is not None  # caller-checked
        chunk = torch.from_numpy(
            self.codec.decode(
                action_ids,
                stats.q01[0].cpu().numpy(),
                stats.q99[0].cpu().numpy(),
            ),
        ).float()[None]
        text = runtime.tokenizer.decode(generated)
        return chunk.to(device), _parse_aux(text, runtime.config.fields)


def _parse_aux(text: str, fields: tuple[AuxField, ...]) -> AuxGeneration:
    """Lenient field parsing of the generated aux text; raw text is kept
    verbatim for reports (parse failures are None, never exceptions)."""
    values: dict[AuxField, str] = {}
    for line in text.splitlines():
        for aux_field in fields:
            prefix = f"{aux_field.value}: "
            if line.startswith(prefix):
                values[aux_field] = line[len(prefix) :].strip()
    holding = values.get(AuxField.HOLDING)
    progress = values.get(AuxField.PROGRESS, "")
    parsed_progress: float | None = None
    if progress.endswith("%"):
        try:
            parsed_progress = float(progress[:-1]) / 100.0
        except ValueError:
            parsed_progress = None
    return AuxGeneration(
        text=text,
        subgoal=values.get(AuxField.SUBGOAL),
        holding=None if holding is None else holding == "yes",
        progress=parsed_progress,
    )


def _row_slice(batch: CollatedBatch[Any], row: int) -> CollatedBatch[Any]:
    """One-sample view of a collated batch (stats/state sliced; encoder
    inputs irrelevant post-encode)."""

    def cut(stats: NormStats) -> NormStats:
        return NormStats(
            mean=stats.mean[row : row + 1],
            std=stats.std[row : row + 1],
            q01=None if stats.q01 is None else stats.q01[row : row + 1],
            q99=None if stats.q99 is None else stats.q99[row : row + 1],
        )

    return dataclasses.replace(
        batch,
        state=batch.state[row : row + 1],
        actions=batch.actions[row : row + 1],
        action_is_pad=batch.action_is_pad[row : row + 1],
        action_stats=cut(batch.action_stats),
        state_stats=cut(batch.state_stats),
    )


def _row_memory(memory: ObservationMemory, row: int) -> ObservationMemory:
    """One-sample view of an encoded memory: streams sliced, the prefix
    cache RE-SLICED per layer (decode appends to it, so each row gets its
    own copy of the cache structure over row-sliced tensors)."""
    cache = memory.cache
    if cache is None:
        raise ValueError(
            "ObservationMemory carries no prefix cache — encode with "
            "retain_cache=True (BijouModel does this for ar_backbone)",
        )
    sliced = KVCache(cache.config)
    for idx, layer in enumerate(cache.layers):
        if layer.keys is not None and layer.values is not None:
            sliced.layers[idx].keys = layer.keys[row : row + 1]
            sliced.layers[idx].values = layer.values[row : row + 1]
    sliced.seen_tokens = cache.seen_tokens
    return ObservationMemory(
        streams={
            name: MemoryStream(
                key=s.key[row : row + 1],
                value=s.value[row : row + 1],
            )
            for name, s in memory.streams.items()
        },
        length=memory.length,
        padding_mask=(
            None if memory.padding_mask is None else memory.padding_mask[row : row + 1]
        ),
        cache=sliced,
    )


def ar_backbone_losses(
    backbone: Gemma4Model,
    decoder: ARBackboneDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Teacher-forced FULL-VOCABULARY cross-entropy — (total, action
    component, aux component | None).

    Without aux (``batch.suffix_tokens`` None) this is EXACTLY the
    pre-aux objective, single-call CE, byte-identical (oracle-gated):
    inputs ``action_tokens[:, :-1]``, targets ``block_base + codec_id``,
    the state position's constant seed BOA and PADs IGNOREd.

    With aux, the suffix is the collator's mixed-id assembly and the
    loss is componentized: ``action`` = mean CE over action targets
    (same scale as the pretrain objective — curves stay comparable),
    ``aux`` = mean CE over aux-text targets (0-sample-safe), total =
    ``action + aux_loss_weight · aux``. The state position stays IGNOREd
    in both modes: whether aux follows depends on LABEL AVAILABILITY,
    which pixels cannot predict — training it would inject pipeline
    noise. Inference forces the scaffold instead.
    """
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
    if batch.suffix_tokens is None:
        tokens = batch.action_tokens
        if tokens is None:
            raise SystemExit(
                "batch carries no action_tokens — build the Collator with "
                "an ActionCodec (--fast-tokenizer) to train an AR decoder",
            )
        logits = decoder(backbone, memory, state, tokens[:, :-1])
        targets = tokens + decoder.config.block_base
        targets[:, 0] = IGNORE_INDEX  # the seed BOA, a constant
        targets[tokens == decoder.codec.pad] = IGNORE_INDEX
        action = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return action, action, None

    suffix = batch.suffix_tokens
    is_aux = batch.suffix_is_aux
    assert is_aux is not None  # collator invariant: the fields travel together
    logits = decoder.forward_backbone_ids(backbone, memory, state, suffix[:, :-1])
    pad_id = decoder.config.block_base + decoder.codec.pad
    targets = suffix.clone()
    targets[:, 0] = IGNORE_INDEX
    targets[suffix == pad_id] = IGNORE_INDEX
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
    aux = elementwise[aux_positions].sum() / aux_positions.sum().clamp(min=1)
    total = action + decoder.aux_loss_weight * aux
    return total, action, aux


def ar_backbone_loss(
    backbone: Gemma4Model,
    decoder: ARBackboneDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> Tensor:
    """Scalar objective (see :func:`ar_backbone_losses`; BijouModel.loss
    dispatches here — the train step calls the tuple form for component
    logging)."""
    total, _, _ = ar_backbone_losses(backbone, decoder, memory, batch)
    return total
