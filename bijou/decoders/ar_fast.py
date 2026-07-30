"""Autoregressive FAST-token action decoder (the flow decoder's causal
sibling).

Same skeleton as the flow decoder — a stack of shared sandwich blocks
cross-attending ObservationMemory streams — but the suffix is
``[state][BOA][t_1..t_k]`` under a plain causal mask, the head is a
token LM head over the FAST vocabulary + BOA/PAD, and there is no
flow-time machinery at all. There is no EOA: a valid sequence expands
to exactly chunk_size * action_dim quantized coefficients, so length is
fixed by the FAST grammar — training is teacher-forced cross-entropy on
the collator's ``action_tokens``, and inference seeds with BOA, never
samples BOA/PAD, and decodes greedily under the grammar mask until the
symbol budget reaches zero — always emitting exactly one chunk.

Baseline intent (docs/plan.md): tests whether the exported K/V interface
carries enough for discrete action prediction, paired head-to-head with
the flow decoder on identical memory and eval frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..fast.codec import ActionCodec
from ..interface import (
    ActionDecoder,
    CollatedBatch,
    MemoryStream,
    ObservationMemory,
    StreamGeometry,
)
from ..nn import (
    DEFAULT_ATTENTION_BACKEND,
    AttentionBackend,
    DeviceLike,
    MaskSpec,
    RMSNorm,
    RopeParameters,
    RopeType,
    buffer_device,
    rope_cos_sin,
    rope_inv_freq_from_params,
)
from .blocks import (
    ExpertConfig,
    ExpertLayer,
    SelfAttentionMode,
    TimeConditioning,
    cross_attention_mask,
)

# CE positions to skip: the state position (its target, the seed BOA, is
# a constant) and PAD padding. torch's cross_entropy convention.
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class ARFastConfig:
    """Construction config of the AR decoder. ``schedule`` references
    encoder stream names (length = depth); cross-attention geometry comes
    from the encoder's StreamGeometry at build time, never from here.
    ``vocab_total`` = BPE vocabulary + BOA + PAD (appended after the BPE
    ids, in that order — ActionCodec's convention)."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    hidden_activation: str
    rms_norm_eps: float
    self_attention_rope_theta: float
    cross_attention_heads: int
    schedule: tuple[str, ...]
    tokenizer: str  # artifact ref (local dir or <user>/<repo>/<subfolder>)
    vocab_total: int
    state_dim: int
    chunk_size: int
    action_dim: int

    @property
    def boa(self) -> int:
        return self.vocab_total - 2

    @property
    def pad(self) -> int:
        return self.vocab_total - 1


class ARFastDecoder(ActionDecoder):
    """Causal token decoder over ObservationMemory (see module docstring).

    ``codec`` is a runtime resource (BPE + quantile glue), not a module:
    it never enters the state_dict, and checkpoints reference the
    tokenizer artifact by id instead."""

    cross_inv_freq: Tensor
    self_inv_freq: Tensor

    def __init__(
        self,
        config: ARFastConfig,
        geometries: dict[str, StreamGeometry],
        codec: ActionCodec,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
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
        missing = [name for name in config.schedule if name not in geometries]
        if missing:
            raise ValueError(
                f"schedule references unknown stream(s) {sorted(set(missing))}; "
                f"encoder exports {sorted(geometries)}",
            )
        scheduled = {name: geometries[name] for name in config.schedule}
        distinct = {(g.kv_heads, g.head_dim, g.rope) for g in scheduled.values()}
        if len(distinct) != 1:
            raise ValueError(
                f"scheduled streams have mixed geometries ({distinct}); "
                "per-layer geometry is not implemented",
            )
        geometry = next(iter(scheduled.values()))
        if geometry.rope is None:
            raise ValueError(
                "rope-free memory streams are not implemented (the decoder "
                "RoPEs its cross-attention queries)",
            )
        # The shared blocks read geometry through ExpertConfig; fields the
        # AR decoder does not use (chunk/time/self-attention mode) are
        # inert placeholders — masks and heads are built here, not there.
        self._block_config = ExpertConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            rms_norm_eps=config.rms_norm_eps,
            self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
            self_attention_rope_theta=config.self_attention_rope_theta,
            cross_attention_heads=config.cross_attention_heads,
            cross_attention_head_dim=geometry.head_dim,
            cross_attention_rope=geometry.rope,
            cross_attention_schedule=(0,) * len(config.schedule),
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            chunk_size=config.chunk_size,
            time_embed_dim=2,
            time_conditioning=TimeConditioning.ADDITIVE,
        )
        hidden = config.hidden_size
        self.token_embedding = nn.Embedding(
            config.vocab_total,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.state_proj = nn.Linear(
            config.state_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            ExpertLayer(
                self._block_config,
                attn_backend=attn_backend,
                device=device,
                dtype=dtype,
            )
            for _ in config.schedule
        )
        self.norm = RMSNorm(hidden, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            hidden,
            config.vocab_total,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "cross_inv_freq",
            rope_inv_freq_from_params(
                geometry.rope,
                geometry.head_dim,
                device=buffer_device(device),
            ),
            persistent=False,
        )
        self.register_buffer(
            "self_inv_freq",
            rope_inv_freq_from_params(
                RopeParameters(
                    rope_type=RopeType.DEFAULT,
                    rope_theta=config.self_attention_rope_theta,
                    factor=1.0,
                    partial_rotary_factor=1.0,
                ),
                hidden // config.num_attention_heads,
                device=buffer_device(device),
            ),
            persistent=False,
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
        assert bool((symbol_lengths == 1).any()), (
            "BPE vocabulary has no single-symbol token — exact fill (and "
            "decode termination) cannot be guaranteed"
        )
        self.symbol_lengths = symbol_lengths
        if device is None or torch.device(device).type != "meta":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                bias = getattr(module, "bias", None)
                if bias is not None:
                    nn.init.zeros_(bias)

    def _causal_mask(
        self,
        batch: int,
        length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> MaskSpec:
        idx = torch.arange(length, device=device)
        allowed = idx[None, :] <= idx[:, None]
        min_value = torch.finfo(dtype).min
        tensor = torch.where(
            allowed,
            torch.tensor(0.0, device=device, dtype=dtype),
            min_value,
        )
        return MaskSpec(tensor=tensor[None, None].expand(batch, 1, length, length))

    @override
    def forward(
        self,
        memory: ObservationMemory,
        state: Tensor,
        tokens: Tensor,
    ) -> Tensor:
        """Next-token logits over the ``[state][tokens...]`` suffix.

        Shapes:
          - memory.streams[name].key/value: [B, kv_heads, P, head_dim]
          - state: [B, state_dim]  (normalized)
          - tokens: [B, T]  (long; BOA/body/PAD ids)
          - returns: [B, 1 + T, vocab_total]  (logits at suffix position j
            predict the token at position j + 1; the BOA position predicts
            t_1)
        """
        dtype = self.state_proj.weight.dtype
        embeds = torch.cat(
            [
                self.state_proj(state.to(dtype))[:, None, :],
                self.token_embedding(tokens),
            ],
            dim=1,
        )
        batch, length, _ = embeds.shape
        device = embeds.device
        streams = {
            name: MemoryStream(key=s.key.to(dtype), value=s.value.to(dtype))
            for name, s in memory.streams.items()
        }
        positions = torch.arange(length, device=device)
        if memory.padding_mask is not None:
            real_lengths = memory.padding_mask.to(device=device, dtype=torch.long).sum(
                dim=1,
            )
            cross_positions = real_lengths[:, None] + positions[None, :]
        else:
            cross_positions = (memory.length + positions)[None, :]
        cross_position_embeddings = rope_cos_sin(
            self.cross_inv_freq,
            cross_positions,
            dtype,
        )
        self_position_embeddings = rope_cos_sin(
            self.self_inv_freq,
            positions[None, :],
            dtype,
        )
        self_mask = self._causal_mask(batch, length, dtype, device)
        cross_mask = cross_attention_mask(memory, dtype, device)

        hidden_states = embeds
        for layer, stream_name in zip(self.layers, self.config.schedule, strict=True):
            hidden_states = layer(
                hidden_states,
                streams[stream_name],
                cross_position_embeddings,
                cross_mask,
                self_position_embeddings,
                self_mask,
                None,
            )
        return self.lm_head(self.norm(hidden_states))

    @override
    def loss(self, memory: ObservationMemory, batch: CollatedBatch[Any]) -> Tensor:
        """Teacher-forced CE (see :func:`ar_fast_loss`). DDP training calls
        the module-level function with the wrapper instead — gradient hooks
        require the forward to go through DDP's __call__."""
        return ar_fast_loss(self, memory, batch)

    @override
    @torch.no_grad()
    def predict_chunk(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """CONSTRAINED greedy decode, then detokenize + denormalize with
        the batch's per-sample q01/q99. Deterministic: ``generator``/
        ``noise`` are unused (greedy has no randomness) and ``noise`` must
        be None. Always emits exactly one chunk.

        The constraint is the FAST grammar: a valid generation expands to
        exactly chunk_size * action_dim quantized coefficients, so each
        step masks to body tokens whose symbol length fits the remaining
        budget — BOA/PAD are never sampled (BOA only seeds the sequence);
        a row is finished when its budget reaches zero (no EOA — length
        is fixed by the grammar), after which it emits PAD (inert,
        stripped before decode). BPE's single-symbol base tokens make
        exact fill always reachable, so the loop terminates in ≤
        chunk*dim steps and every generation decodes by construction
        (typical sequences are ~50-60 tokens). Malformed generations are
        impossible by construction — a decode error here is a bug and
        propagates."""
        if noise is not None:
            raise ValueError("ARFastDecoder.predict_chunk takes no noise")
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
        # Every sequence opens with BOA; body tokens follow.
        tokens = torch.full(
            (batch_size, 1),
            config.boa,
            dtype=torch.long,
            device=device,
        )
        min_value: float = torch.finfo(torch.float32).min
        for _ in range(total_symbols):
            if bool((remaining == 0).all()):
                break
            logits = self(memory, state, tokens)[:, -1, :].float()
            # The FAST grammar mask (see docstring). Specials have length
            # 0, so the > 0 term keeps BOA/PAD unsampleable for live rows;
            # finished rows (budget 0) emit PAD.
            allowed = (lengths[None, :] > 0) & (lengths[None, :] <= remaining[:, None])
            allowed[:, config.pad] = remaining == 0
            logits = logits.masked_fill(~allowed, min_value)
            next_token = logits.argmax(dim=-1)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            remaining = remaining - lengths[next_token]

        q01 = stats.q01.cpu().numpy()
        q99 = stats.q99.cpu().numpy()
        token_rows = tokens.cpu().tolist()
        chunks = [
            torch.from_numpy(
                self.codec.decode(
                    [t for t in row[1:] if t != config.pad],  # drop seed BOA
                    q01[row_index],
                    q99[row_index],
                ),
            ).float()
            for row_index, row in enumerate(token_rows)
        ]
        return torch.stack(chunks).to(device)


def ar_fast_loss(
    model: torch.nn.Module,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> Tensor:
    """Teacher-forced token cross-entropy.

    ``model`` is the ARFastDecoder (or its DDP wrapper — training forwards
    must go through the wrapper for gradient hooks); call convention
    (memory, state, tokens) -> logits.

    Inputs are ``action_tokens[:, :-1]`` (= [BOA, t_1..t_{k-1}] plus
    padding; the last column is PAD for every sample but the
    batch-longest, whose final token needs no successor); logits at
    suffix position j predict ``action_tokens[:, j]`` — the state
    position's target (the constant seed BOA) and all PAD positions are
    IGNORE_INDEX. CE averages over real target tokens, so longer token
    sequences weigh proportionally more — standard LM behavior.
    """
    tokens = batch.action_tokens
    if tokens is None:
        raise SystemExit(
            "batch carries no action_tokens — build the Collator with an "
            "ActionCodec (--fast-tokenizer) to train an AR decoder",
        )
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
    logits = model(memory, state, tokens[:, :-1])
    targets = tokens.clone()
    targets[:, 0] = IGNORE_INDEX  # the seed BOA, a constant
    pad_id = int(logits.shape[-1] - 1)  # PAD is the last id by convention
    targets[targets == pad_id] = IGNORE_INDEX
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
