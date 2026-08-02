"""Decoder-only action decoder: the backbone's suffix role (ar_backbone).

The prompt is prefill-encoded once (the ObservationMemory retains the
full prefix KV cache); this decoder continues the suffix-format-3
sequence ``[state][<start_of_turn>model\\n][MODE][aux text?][BOA]
[t_1..t_k]`` through ALL backbone layers — the KV-shared deep half
included — and reads FULL-VOCABULARY logits from the frozen tied LM
head with the FAST block's columns supplied by a trainable patch. Aux
text (subgoal / holding / progress rendered from judge annotations,
bijou.aux_text) and action tokens share that one softmax. The MODE
token ([ACT] | [AUX]) is FED, never predicted: whether a sample speaks
is decided by its label presence (and aux dropout) at collation, so
the model is never asked to infer judged-ness from appearance, and
inference COMMANDS the mode — [ACT] goes straight to actions, [AUX]
elicits the aux segment. Aux-less runs feed [ACT] everywhere and stay
the same model family.

Ownership: this module owns ONLY the new parameters (~11M at E2B scale)
— the state projection (suffix position 0, zero-initialized so the
prompt-conditioned computation starts undisturbed) and the FAST block's
input-embedding + per-layer-embedding rows. The backbone is owned by
BijouModel and passed into every call; ``expert.safetensors`` therefore
stays exactly "the new parameters".

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

Sequence/loss conventions: state and all-but-the-last opener positions
predict constants and are IGNOREd; BOA IS predicted (the decision
point's target on aux-less samples, the aux segment's terminator
otherwise); PAD is batch padding only and always ignored; there is no
EOA (action length is fixed by the FAST grammar). Decoding is the ONE
free-until-BOA path (:meth:`ARBackboneDecoder.predict_chunk`):
text-or-BOA mask under a token budget, then the ar_fast-style
grammar-constrained greedy mask by remaining symbol budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..aux_text import (
    ACT_MODE,
    AUX_MODE,
    GENERATION_OPENER,
    MAX_FREE_TOKENS,
    NUM_MODES,
    OPENER_SUFFIX_FORMAT,
    SUFFIX_FORMAT,
    AuxDecodeConfig,
    AuxDecodeMode,
    AuxGeneration,
    AuxRuntime,
    TextTokenizer,
)
from ..fast.codec import ActionCodec
from ..gemma4.config import Gemma4TextConfig
from ..gemma4.model import Gemma4Model
from ..interface import ChunkPrediction, CollatedBatch, ObservationMemory
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
    state_dim: int
    chunk_size: int
    action_dim: int
    # Suffix format (aux_text.SUFFIX_FORMAT when written): 3 = the
    # opener+mode format every new run trains; parsed 2 (opener, no
    # mode) / 1 (pre-opener) = legacy checkpoints — loadable for warm
    # starts (fresh mode rows), decoded via their own mode-less path
    # with a loud warning.
    suffix_format: int
    # Aux text record: template version + fields + label provenance
    # (None = trained without aux — the decision point trains to BOA).
    aux: AuxDecodeConfig | None

    def __post_init__(self) -> None:
        if self.vocab_total < 3:  # ≥ 1 body token + BOA + PAD
            raise ValueError(f"vocab_total {self.vocab_total} is not a FAST vocabulary")
        if self.block_base < 0:
            raise ValueError(f"block_base {self.block_base} must be non-negative")
        if self.aux is not None and self.suffix_format < OPENER_SUFFIX_FORMAT:
            raise ValueError(
                f"aux config on suffix format {self.suffix_format}: aux "
                "text shipped WITH the opener format — no such checkpoint "
                "exists, and the decode path could not elicit it",
            )
        if self.suffix_format >= SUFFIX_FORMAT and self.block_base < NUM_MODES:
            raise ValueError(
                f"block_base {self.block_base} leaves no room for the "
                f"{NUM_MODES} mode ids directly below the FAST block",
            )

    @property
    def mode_base(self) -> int:
        """First backbone id of the mode-token pair ([ACT], [AUX]) —
        directly below the FAST block."""
        return self.block_base - NUM_MODES


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
        # tokenizer (opener ids, generation text), aux metrics runtime,
        # and the aux loss mixture weight.
        self.tokenizer = tokenizer
        self.aux_runtime = aux_runtime
        self.aux_loss_weight = aux_loss_weight
        # Cumulative free-phase budget exhaustions (decode health metric).
        self.fallback_count = 0
        # Per-feature format gating: the opener arrived with format 2,
        # the mode token with format 3 — a format-2 checkpoint keeps its
        # trained opener and decodes via the mode-less legacy path.
        self.uses_modes = config.suffix_format >= SUFFIX_FORMAT
        if config.suffix_format >= OPENER_SUFFIX_FORMAT:
            if tokenizer is None:
                raise ValueError(
                    "opener-format checkpoints need the backbone's text "
                    "tokenizer (the generation opener is tokenized at "
                    "construction)",
                )
            self.opener_ids: tuple[int, ...] = tuple(
                tokenizer.encode(GENERATION_OPENER, add_special_tokens=False),
            )
        else:
            self.opener_ids = ()
        if not self.uses_modes:
            print(
                "[ar_backbone] LEGACY suffix format "
                f"{config.suffix_format} checkpoint (pre-mode): decoding "
                "uses its own mode-less path — fine for --init-from warm "
                "starts, re-fine-tune before trusting inference",
                flush=True,
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
        # Mode-token tables ([ACT], [AUX]) — same patch mechanism as the
        # FAST block, two rows directly below it. Present (and saved)
        # for every format so state-dict shapes are format-independent;
        # legacy checkpoints load them fresh and never feed them.
        self.mode_embed = nn.Embedding(NUM_MODES, hidden, device=device, dtype=dtype)
        self.mode_ple = nn.Embedding(
            NUM_MODES,
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
        nn.init.normal_(self.mode_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_ple.weight, mean=0.0, std=0.02)
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
            (self.mode_embed.weight, embed_mean),
            (self.mode_ple.weight, ple_mean),
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
        """As :meth:`_suffix_inputs` but ``tokens`` are BACKBONE ids,
        routed by range: text ids (< mode_base — the aux segment) embed
        through the frozen embed_tokens/PLE tables, the two mode ids
        through the mode tables, block ids through the FAST patch. An
        all-block suffix reproduces the pre-aux computation bitwise
        (nested torch.where with all-False masks returns the block side
        elementwise)."""
        text = backbone.language_model
        target_dtype = text.embed_tokens.weight.dtype
        mode_base = self.config.mode_base
        is_block = tokens >= self.config.block_base
        is_mode = ((tokens >= mode_base) & ~is_block)[..., None]
        is_text = (tokens < mode_base)[..., None]
        block_ids = (tokens - self.config.block_base).clamp(min=0)
        mode_ids = (tokens - mode_base).clamp(min=0, max=NUM_MODES - 1)
        # Text-side lookups use the pad row at block/mode positions
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
            torch.where(
                is_mode,
                self.mode_embed(mode_ids) * self.embed_scale,
                self.fast_embed(block_ids) * self.embed_scale,
            ),
        )
        ple = torch.where(
            is_text[..., None],
            text.get_per_layer_inputs(text_ids).float(),
            torch.where(
                is_mode[..., None],
                (self.mode_ple(mode_ids) * self.ple_scale).view(
                    tokens.shape[0],
                    tokens.shape[1],
                    self.num_layers,
                    self.ple_dim,
                ),
                (self.fast_ple(block_ids) * self.ple_scale).view(
                    tokens.shape[0],
                    tokens.shape[1],
                    self.num_layers,
                    self.ple_dim,
                ),
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
        mode: AuxDecodeMode = AuxDecodeMode.ACT,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> ChunkPrediction:
        """The single decode path for every ar_backbone checkpoint,
        batched with per-row phases. Deterministic greedy;
        ``generator``/``noise`` unused/must be None.

        ``mode`` picks the fed mode token (format 3): ACT feeds
        ``[state][opener][ACT][BOA]`` and goes straight to the ACTION
        phase (the ar_fast grammar mask by remaining symbol budget; PAD
        once finished, inert) — the deployment fast path. FREE feeds
        ``[state][opener][AUX]`` then runs the FREE phase per row (text
        ids and BOA only — mode ids and the rest of the FAST block are
        masked; budget MAX_FREE_TOKENS, exhaustion forces BOA and
        increments ``fallback_count``, printed loudly) until BOA, then
        actions. FREE requires an aux-trained checkpoint ([AUX] is
        untrained otherwise — loud error). Pre-mode legacy checkpoints
        ignore ``mode`` and decode their own trained format:
        ``[state][opener?]`` then free-until-BOA.

        Returns a ChunkPrediction: chunks [B, chunk, action_dim] raw
        units + one AuxGeneration per row (empty text under ACT and
        whenever the model went straight to BOA)."""
        if noise is not None:
            raise ValueError("ARBackboneDecoder.predict_chunk takes no noise")
        if self.uses_modes and mode is AuxDecodeMode.FREE and self.config.aux is None:
            raise ValueError(
                "FREE decode on an aux-less checkpoint: every training "
                "sample fed [ACT], so the [AUX] mode is untrained — "
                "decode with mode=act",
            )
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
        base = config.block_base
        boa_backbone = base + self.codec.boa
        pad_backbone = base + self.codec.pad
        lengths = self.symbol_lengths.to(device)
        total_symbols = config.chunk_size * config.action_dim
        min_value: float = torch.finfo(torch.float32).min

        # Static free-phase mask: text ids + BOA allowed; mode ids and
        # the rest of the block (and beyond) illegal before BOA. (Mode
        # ids are fed only — masking them is a no-op for legacy layouts.)
        vocab = base + config.vocab_total
        free_allowed = torch.zeros(vocab, dtype=torch.bool, device=device)
        free_allowed[:base] = True
        free_allowed[config.mode_base : base] = False
        free_allowed[boa_backbone] = True

        # The fed prefix and starting phase are the mode (format 3); a
        # pre-mode legacy checkpoint feeds its own trained prefix.
        prefix_ids = list(self.opener_ids)
        start_in_action = False
        if self.uses_modes:
            match mode:
                case AuxDecodeMode.ACT:
                    prefix_ids += [config.mode_base + ACT_MODE, boa_backbone]
                    start_in_action = True
                case AuxDecodeMode.FREE:
                    prefix_ids.append(config.mode_base + AUX_MODE)

        in_action = torch.full(
            (batch_size,),
            start_in_action,
            dtype=torch.bool,
            device=device,
        )
        remaining = torch.full((batch_size,), total_symbols, device=device)
        free_spent = 0
        free_ids: list[list[int]] = [[] for _ in range(batch_size)]
        action_ids: list[list[int]] = [[] for _ in range(batch_size)]

        prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        feed = prefix.expand(batch_size, -1)
        feed_state: Tensor | None = state
        fed = 0
        fallback_count = 0
        for _ in range(MAX_FREE_TOKENS + total_symbols + 1):
            if bool(in_action.all()) and bool((remaining == 0).all()):
                break
            logits, fed = self._step(backbone, memory, feed_state, feed, fed)
            feed_state = None
            logits = logits[:, :vocab]
            # Per-row mask by phase.
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
            allowed = torch.where(
                in_action[:, None],
                action_allowed,
                free_allowed[None, :],
            )
            next_ids = logits.masked_fill(~allowed, min_value).argmax(dim=-1)
            if free_spent >= MAX_FREE_TOKENS:
                # Budget exhausted: force BOA on rows still talking.
                forced = ~in_action & (next_ids != boa_backbone)
                fallback_count += int(forced.sum())
                next_ids = torch.where(
                    forced,
                    torch.full_like(next_ids, boa_backbone),
                    next_ids,
                )
            free_spent += int((~in_action).any())
            rows = next_ids.tolist()
            for row, next_id in enumerate(rows):
                if in_action[row]:
                    codec_id = next_id - base
                    if codec_id != self.codec.pad:
                        action_ids[row].append(codec_id)
                        remaining[row] -= int(lengths[codec_id])
                elif next_id == boa_backbone:
                    in_action[row] = True
                else:
                    free_ids[row].append(next_id)
            feed = next_ids[:, None]
        if fallback_count:
            self.fallback_count += fallback_count
            print(
                f"[ar_backbone] free-phase budget exhausted on "
                f"{fallback_count} row(s) — BOA forced (cumulative "
                f"{self.fallback_count}); a persistent rate means the "
                "model stopped closing its aux segment",
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
                self.tokenizer.decode(ids) if self.tokenizer is not None else "",
            )
            for ids in free_ids
        ]
        return ChunkPrediction(
            chunks=torch.stack(chunks).to(device),
            generations=generations,
        )

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


def _parse_aux(text: str) -> AuxGeneration:
    """Lenient field parsing of the generated aux text; raw text is kept
    verbatim for reports (parse failures are None, never exceptions)."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        for field_name in ("subgoal", "holding", "progress"):
            prefix = f"{field_name}: "
            if line.startswith(prefix):
                values[field_name] = line[len(prefix) :].strip()
    holding = values.get("holding")
    progress = values.get("progress", "")
    parsed_progress: float | None = None
    if progress.endswith("%"):
        try:
            parsed_progress = float(progress[:-1]) / 100.0
        except ValueError:
            parsed_progress = None
    return AuxGeneration(
        text=text,
        subgoal=values.get("subgoal"),
        holding=None if holding is None else holding == "yes",
        progress=parsed_progress,
    )


def ar_backbone_losses(
    backbone: Gemma4Model,
    decoder: ARBackboneDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Teacher-forced FULL-VOCABULARY cross-entropy over the format-3
    suffix ``[state][opener][MODE][aux?][BOA][actions]`` — (total with
    graph, action component, aux CE SUM | None, aux position count |
    None). Aux rides as sum+count (not a mean) so the train loop can
    aggregate a position-weighted mean across batches and ranks — a
    per-batch mean dilutes toward 0 on sparsely-labeled corpora.

    The opener and per-row MODE token are prepended HERE (the collator
    supplies content only; a row's mode is derived from its content —
    [AUX] iff it carries aux positions, which is also how aux dropout
    lands on [ACT]). State, opener and the fed MODE position predict
    constants and are IGNOREd; the MODE token's own logits are the
    trained transition — first aux token on [AUX] rows, BOA on [ACT]
    rows. Component split: ``action`` = mean CE over action targets
    (pretrain scale); total = action + aux_loss_weight * (aux_sum /
    aux_count) — batch-mean semantics, 0-safe when a batch has no
    labeled sample. Legacy pre-mode decoders (warm-start loads) train
    their own opener-only sequence.
    """
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
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
    opener = torch.tensor(
        [decoder.opener_ids],
        dtype=torch.long,
        device=content.device,
    ).expand(content.shape[0], -1)
    if decoder.uses_modes:
        mode_base = decoder.config.mode_base
        if is_aux_content is None:
            modes = torch.full(
                (content.shape[0], 1),
                mode_base + ACT_MODE,
                dtype=torch.long,
                device=content.device,
            )
        else:
            modes = torch.where(
                is_aux_content.any(dim=1),
                mode_base + AUX_MODE,
                mode_base + ACT_MODE,
            )[:, None].to(device=content.device, dtype=torch.long)
        prefix = torch.cat([opener, modes], dim=1)
    else:
        prefix = opener
    full = torch.cat([prefix, content], dim=1)
    logits = decoder.forward_backbone_ids(backbone, memory, state, full[:, :-1])
    pad_id = base + decoder.codec.pad
    targets = full.clone()
    # State + opener + fed-MODE positions carry no signal; the MODE
    # position's own logits (targets index len(prefix)-1... its target
    # is full[len(prefix)] = content[0]) stay trained. Note the shift:
    # logits[j] predicts full[j], so IGNORE full[0..len(prefix)-1] keeps
    # exactly the constant prefix targets out and content[0] in.
    targets[:, : prefix.shape[1]] = IGNORE_INDEX
    targets[full == pad_id] = IGNORE_INDEX
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
    )
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
