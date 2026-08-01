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

from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..fast.codec import ActionCodec
from ..gemma4.config import Gemma4TextConfig
from ..gemma4.model import Gemma4Model
from ..interface import CollatedBatch, ObservationMemory
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
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.codec = codec
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
        first feed); tokens [B, T] codec ids; returns
        ([B, S, hidden], [B, S, num_layers, ple_dim]) with
        S = T (+1 with state)."""
        text = backbone.language_model
        target_dtype = text.embed_tokens.weight.dtype
        embeds = self.fast_embed(tokens) * self.embed_scale
        ple = (self.fast_ple(tokens) * self.ple_scale).view(
            tokens.shape[0],
            tokens.shape[1],
            self.num_layers,
            self.ple_dim,
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
        embeds, per_layer = self._suffix_inputs(backbone, state, tokens)
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
        feed = tokens  # first feed carries [state][BOA]
        feed_state: Tensor | None = state
        fed = 0
        for _ in range(total_symbols):
            if bool((remaining == 0).all()):
                break
            embeds, per_layer = self._suffix_inputs(backbone, feed_state, feed)
            hidden = self._continue_suffix(backbone, memory, embeds, per_layer, fed)
            fed += embeds.shape[1]
            logits = self._patched_logits(backbone, hidden)[:, -1, :].float()
            # Grammar mask over the block slice: greedy never leaves the
            # block after BOA today (aux text decoding, when it lands,
            # runs the pre-BOA segment over the full vocabulary instead).
            block = logits[:, base : base + config.vocab_total]
            allowed = (lengths[None, :] > 0) & (lengths[None, :] <= remaining[:, None])
            allowed[:, pad] = remaining == 0
            block = block.masked_fill(~allowed, min_value)
            next_token = block.argmax(dim=-1)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            remaining = remaining - lengths[next_token]
            feed = next_token[:, None]
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


def ar_backbone_loss(
    backbone: Gemma4Model,
    decoder: ARBackboneDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> Tensor:
    """Teacher-forced FULL-VOCABULARY cross-entropy.

    Targets are backbone ids ``block_base + codec_id`` — action tokens
    compete against the whole text vocabulary (the model learns to
    suppress text priors at action positions, the capability aux text
    outputs will rely on). Conventions mirror :func:`ar_fast_loss`:
    inputs are ``action_tokens[:, :-1]``, the state position's constant
    seed BOA and all PAD positions are IGNORE_INDEX, CE averages over
    real target tokens.
    """
    tokens = batch.action_tokens
    if tokens is None:
        raise SystemExit(
            "batch carries no action_tokens — build the Collator with an "
            "ActionCodec (--fast-tokenizer) to train an AR decoder",
        )
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
    logits = decoder(backbone, memory, state, tokens[:, :-1])
    targets = tokens + decoder.config.block_base
    targets[:, 0] = IGNORE_INDEX  # the seed BOA, a constant
    targets[tokens == decoder.codec.pad] = IGNORE_INDEX
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
