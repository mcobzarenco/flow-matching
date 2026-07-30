"""Autoregressive FAST-token action decoder (the flow decoder's causal
sibling).

Same skeleton as the flow decoder — a stack of shared sandwich blocks
cross-attending ObservationMemory streams — but the suffix is
``[state][BOA][t_1..t_k]`` under a plain causal mask, the head is a token
LM head over the FAST vocabulary + specials, and there is no flow-time
machinery at all. Training is teacher-forced cross-entropy on the
collator's ``action_tokens``; inference decodes greedily until EOA and
detokenizes through the ActionCodec with the batch's per-sample q01/q99.

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
from ..fast.tokenizer import FastDecodeError
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

# CE positions to skip: the state position (its target, BOA, is a
# constant) and PAD padding. torch's cross_entropy convention.
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class ARFastConfig:
    """Construction config of the AR decoder. ``schedule`` references
    encoder stream names (length = depth); cross-attention geometry comes
    from the encoder's StreamGeometry at build time, never from here.
    ``vocab_total`` = BPE vocabulary + 3 specials (BOA, EOA, PAD appended
    after the BPE ids, in that order — ActionCodec's convention)."""

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
    max_tokens: int
    state_dim: int
    chunk_size: int
    action_dim: int

    @property
    def boa(self) -> int:
        return self.vocab_total - 3

    @property
    def eoa(self) -> int:
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
        # Malformed-decode telemetry (predict_chunk substitutes state-copy
        # and counts here — same pattern as FastTokenizer.clipped_coefficients).
        self.malformed_decodes = 0
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
          - tokens: [B, T]  (long; BOA/body/EOA/PAD ids)
          - returns: [B, 1 + T, vocab_total]  (logits at position j predict
            the token at suffix position j + 1; the state position predicts
            BOA, which training ignores)
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
        """Greedy decode until EOA (or max_tokens), then detokenize +
        denormalize with the batch's per-sample q01/q99. Deterministic:
        ``generator``/``noise`` are unused (greedy has no randomness) and
        ``noise`` must be None.

        Malformed generations (FastDecodeError: wrong coefficient count,
        unknown ids) substitute the state-copy chunk for that sample and
        increment ``self.malformed_decodes`` — the pre-registered health
        metric is the RATE, so it must stay visible, and eval requires a
        chunk per frame."""
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
        tokens = torch.full(
            (batch_size, 1),
            config.boa,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(config.max_tokens):
            logits = self(memory, state, tokens)[:, -1, :]
            # BOA opens the sequence and PAD is a batching artifact —
            # neither is ever a valid continuation.
            logits[:, config.boa] = torch.finfo(logits.dtype).min
            logits[:, config.pad] = torch.finfo(logits.dtype).min
            next_token = logits.argmax(dim=-1)
            # Finished rows keep appending EOA (inert; stripped on decode).
            next_token = torch.where(
                finished,
                torch.full_like(next_token, config.eoa),
                next_token,
            )
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            finished = finished | (next_token == config.eoa)
            if bool(finished.all()):
                break

        q01 = stats.q01.cpu().numpy()
        q99 = stats.q99.cpu().numpy()
        token_rows = tokens.cpu().tolist()
        chunks: list[Tensor] = []
        malformed = 0
        for row_index, row in enumerate(token_rows):
            body = row[1:]  # drop BOA
            if config.eoa in body:
                body = body[: body.index(config.eoa)]
            try:
                decoded = self.codec.decode(body, q01[row_index], q99[row_index])
                chunks.append(torch.from_numpy(decoded).float())
            except FastDecodeError:
                malformed += 1
                chunks.append(
                    batch.state[row_index]
                    .cpu()
                    .float()[None, :]
                    .expand(config.chunk_size, -1)
                    .clone(),
                )
        if malformed:
            self.malformed_decodes += malformed
            print(
                f"AR decode: {malformed}/{batch_size} malformed generation(s) "
                f"substituted with state-copy ({self.malformed_decodes} total "
                "in this process)",
                flush=True,
            )
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

    Inputs are ``action_tokens[:, :-1]`` (the last column is PAD for every
    sample but the batch-longest, whose final EOA needs no successor);
    targets align logits[j] with action_tokens[j]: the state position's
    target (BOA, a constant) and all PAD positions are IGNORE_INDEX. CE
    averages over real target tokens, so longer token sequences weigh
    proportionally more — standard LM behavior.
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
    targets[:, 0] = IGNORE_INDEX
    pad_id = int(logits.shape[-1] - 1)  # PAD is the last id by convention
    targets[targets == pad_id] = IGNORE_INDEX
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
