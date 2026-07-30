"""Bijou action expert: a narrow flow-matching decoder over action chunks.

The expert consumes ``[state][action_1..action_chunk]`` tokens and, per layer,
cross-attends one *global-attention* K/V stream exported from the frozen
Gemma 4 trunk (layers 4/9/14 for E2B), then self-attends the suffix, then
runs a gated MLP. It predicts the flow-matching velocity of the action chunk
at flow time τ.

Design notes (see the design discussion in the repo history):

- Cross-attention queries adopt the backbone's global-attention geometry so
  the exported K/V are consumed exactly as the backbone's own deep layers
  consume them: head_dim = ``global_head_dim`` (512), q-RMSNorm, p-RoPE at
  positions continuing after each sample's REAL (unpadded) memory width,
  attention scaling 1.0.
- The per-layer stream assignment is the ``cross_attention_schedule`` tuple
  (its length is the expert depth), e.g. blocks ``(4,4,4,4, 9,9,9,9,
  14,...)``; cycle/hybrid schedules are config diffs, not code paths.
- Self-attention over the suffix is bidirectional or causal-over-actions
  (state visible to and from everything in both modes) — an explicit ablation
  knob, ``SelfAttentionMode``.
- Flow time τ enters via one of two schemes (``TimeConditioning``): ADDITIVE
  (π0-style — sinusoidal embedding, MLP-transformed, added to the action
  token embeddings; state token not time-conditioned) or ADARMS (DiT-style
  adaptive RMSNorm — per-layer scale on the norm outputs and gate on the
  sublayer contributions, identity at init). adaRMS is the intended
  successor; the additive path is kept for loading pre-adaRMS checkpoints
  and is deletable as a unit (its ``None``-guarded branches).

Flow-matching convention (matches lerobot's π0/SmolVLA):
``x_τ = τ·ε + (1−τ)·actions`` with ε ~ N(0, I), so τ=1 is pure noise; the
velocity target is ``u = ε − actions``; sampling integrates from τ=1 to 0
with steps of ``dτ = −1/num_steps``.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, cast, override

import torch
from torch import Tensor, nn

from ..interface import (
    ActionDecoder,
    CollatedBatch,
    MemoryStream,
    ObservationMemory,
    kv_stream_name,
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
    apply_scale,
    cross_attention_mask,
)


class SamplingMethod(Enum):
    """ODE solver for integrating the velocity field from noise to actions.

    EULER: 1 model evaluation per step, first-order (global error O(1/n)).
    HEUN: explicit trapezoidal predictor-corrector, 2 evaluations per step,
    second-order (O(1/n²)); the better quality-per-evaluation trade for all
    but the very smallest step counts (Karras et al., EDM).
    """

    EULER = "euler"
    HEUN = "heun"


def sinusoidal_time_embedding(
    time: Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """[B] flow times in [0, 1] -> [B, dim] float32 sin/cos features.

    Geometric period range tuned for the unit interval (π0's choice), rather
    than the 10k-period convention used for token positions.
    """
    half = dim // 2
    fraction = torch.arange(half, dtype=torch.float32, device=time.device) / half
    period = min_period * (max_period / min_period) ** fraction
    angle = time[:, None].float() / period[None, :] * (2 * math.pi)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class FlowDecoder(ActionDecoder):
    """Flow-matching action decoder: a velocity network over an action
    chunk, conditioned on observation-memory streams, robot state and flow
    time.

    Attribute names are frozen — they are the safetensors keys of every
    existing checkpoint (gated by tests/test_state_dict_keys.py)."""

    cross_inv_freq: Tensor
    self_inv_freq: Tensor

    def __init__(
        self,
        config: ExpertConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # Stream names this expert's layers cross-attend, one per layer
        # (the int schedule is legacy config; lookups are by name).
        self.schedule_names = tuple(
            kv_stream_name(idx) for idx in config.cross_attention_schedule
        )
        hidden = config.hidden_size

        self.state_proj = nn.Linear(
            config.state_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.action_in_proj = nn.Linear(
            config.action_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.time_in_proj = nn.Linear(
            config.time_embed_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.time_out_proj = nn.Linear(
            hidden,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.time_act = nn.SiLU()

        self.layers = nn.ModuleList(
            ExpertLayer(config, attn_backend=attn_backend, device=device, dtype=dtype)
            for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(hidden, eps=config.rms_norm_eps, device=device, dtype=dtype)
        # adaRMS only: a final scale on the output norm (no gate — no
        # residual there), zero-initialized => identity at init.
        self.final_modulation: nn.Sequential | None = None
        if config.time_conditioning is TimeConditioning.ADARMS:
            self.final_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden, hidden, device=device, dtype=dtype),
            )
        # Zero-initialized so the initial velocity field is 0 (standard for
        # flow/diffusion heads).
        self.action_out_proj = nn.Linear(
            hidden,
            config.action_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.register_buffer(
            "cross_inv_freq",
            rope_inv_freq_from_params(
                config.cross_attention_rope,
                config.cross_attention_head_dim,
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
                config.hidden_size // config.num_attention_heads,
                device=buffer_device(device),
            ),
            persistent=False,
        )
        if device is None or torch.device(device).type != "meta":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # cast past the stubs: torch types Linear.bias as Parameter,
                # but bias=False modules carry None at runtime.
                bias = cast("nn.Parameter | None", module.bias)
                if bias is not None:
                    nn.init.zeros_(bias)
        nn.init.zeros_(self.action_out_proj.weight)
        assert self.action_out_proj.bias is not None
        nn.init.zeros_(self.action_out_proj.bias)
        # adaRMS heads zero-initialized (overriding the normal-init above):
        # scale=0 => norm output unchanged, gate=0 => sublayer contributes
        # nothing => every layer is the identity at init, so the residual
        # stream passes through untouched and the velocity field is 0.
        for module in self.modules():
            if isinstance(module, ExpertLayer) and module.modulation is not None:
                head = module.modulation[1]
                assert isinstance(head, nn.Linear)
                nn.init.zeros_(head.weight)
                assert head.bias is not None
                nn.init.zeros_(head.bias)
        if self.final_modulation is not None:
            head = self.final_modulation[1]
            assert isinstance(head, nn.Linear)
            nn.init.zeros_(head.weight)
            assert head.bias is not None
            nn.init.zeros_(head.bias)

    def _self_attention_mask(
        self,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> MaskSpec:
        if self.config.self_attention_mode is SelfAttentionMode.BIDIRECTIONAL:
            return MaskSpec()
        length = self.config.suffix_length
        idx = torch.arange(length, device=device)
        # State (position 0) attends and is attended by everything; actions
        # are causal among themselves.
        allowed = (
            (idx[:, None] == 0) | (idx[None, :] == 0) | (idx[None, :] <= idx[:, None])
        )
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
        noisy_actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Velocity of the action chunk at flow time τ; returns
        [B, chunk, action_dim].

        Inputs and memory streams are cast to the expert's own dtype (the
        backbone may run in a different precision, e.g. bf16 vs fp32 expert).

        Shapes:
          - memory.streams[name].key/value: [B, kv_heads, P, head_dim]
          - memory.padding_mask (when present): [B, P]  (True = real token)
          - state: [B, state_dim]
          - noisy_actions: [B, chunk, action_dim]
          - time: [B]  (flow times in [0, 1])
        """
        config = self.config
        batch = state.shape[0]
        if noisy_actions.shape[1] != config.chunk_size:
            raise ValueError(
                f"expected chunk of {config.chunk_size} actions, "
                f"got {noisy_actions.shape[1]}",
            )
        dtype = self.state_proj.weight.dtype

        state_embeds = self.state_proj(state.to(dtype))[:, None, :]
        action_embeds = self.action_in_proj(noisy_actions.to(dtype))
        time_embeds = sinusoidal_time_embedding(time, config.time_embed_dim)
        time_embeds = self.time_out_proj(
            self.time_act(self.time_in_proj(time_embeds.to(dtype))),
        )
        # ADDITIVE: fold τ into the action tokens, layers unconditioned.
        # ADARMS: τ conditions each layer's scale/gate head instead.
        adarms = config.time_conditioning is TimeConditioning.ADARMS
        condition = time_embeds if adarms else None
        if not adarms:
            action_embeds = action_embeds + time_embeds[:, None, :]
        hidden_states = torch.cat([state_embeds, action_embeds], dim=1)

        device = hidden_states.device
        streams = {
            name: MemoryStream(key=s.key.to(dtype), value=s.value.to(dtype))
            for name, s in memory.streams.items()
        }
        suffix_positions = torch.arange(config.suffix_length, device=device)
        # Cross-attention queries continue after each sample's REAL prefix.
        # Using the padded batch width here would shift every query->key
        # RoPE distance by that sample's padding, making predictions depend
        # on batch-mates' prompt lengths (measured: max|delta| 0.55 on the
        # expert alone, outputs/probe_effect1_fix.py).
        if memory.padding_mask is not None:
            real_lengths = memory.padding_mask.to(device=device, dtype=torch.long).sum(
                dim=1,
            )
            cross_positions = real_lengths[:, None] + suffix_positions[None, :]
        else:
            cross_positions = (memory.length + suffix_positions)[None, :]
        cross_position_embeddings = rope_cos_sin(
            self.cross_inv_freq,
            cross_positions,
            dtype,
        )
        self_position_embeddings = rope_cos_sin(
            self.self_inv_freq,
            suffix_positions[None, :],
            dtype,
        )
        self_attention_mask = self._self_attention_mask(batch, dtype, device)
        cross_mask = cross_attention_mask(memory, dtype, device)

        for layer, stream_name in zip(
            self.layers,
            self.schedule_names,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                streams[stream_name],
                cross_position_embeddings,
                cross_mask,
                self_position_embeddings,
                self_attention_mask,
                condition,
            )

        hidden_states = self.norm(hidden_states)
        if self.final_modulation is not None:
            assert condition is not None
            hidden_states = apply_scale(
                hidden_states,
                self.final_modulation(condition),
            )
        return self.action_out_proj(hidden_states[:, 1:, :])

    @torch.no_grad()
    def sample_actions(
        self,
        memory: ObservationMemory,
        state: Tensor,
        *,
        num_steps: int = 5,
        method: SamplingMethod = SamplingMethod.HEUN,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Integrate the velocity field from τ=1 (noise) to τ=0.

        The default (Heun, 5 steps = 10 model evaluations) costs the same as
        the Euler/10 convention of π0/SmolVLA and integrates more accurately:
        on a trained checkpoint vs a Heun-64 reference, Heun-5 halves the
        worst-case error (0.24 vs 0.53 normalized units) at equal cost.
        ``num_steps`` counts solver steps: Euler does 1 evaluation per step,
        Heun 2 — and below ~4 steps Heun's corrector is wasted (use Euler if
        you must go that low).

        The model is trained on τ ∈ (0.001, 1]; Heun's final corrector
        evaluates at exactly τ=0, a negligible extrapolation for the smooth
        sinusoidal time embedding.

        Pass ``noise`` (or a seeded ``generator``) for deterministic
        evaluation. ``state`` and the returned chunk are NORMALIZED units
        (the raw-unit wrapper is :meth:`predict_chunk`).

        Shapes:
          - memory.streams[name].key/value: [B, kv_heads, P, head_dim]
          - state: [B, state_dim]
          - noise (when given): [B, chunk, action_dim]
          - returns: [B, chunk, action_dim]
        """
        config = self.config
        batch = state.shape[0]
        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(
                batch,
                config.chunk_size,
                config.action_dim,
                dtype=dtype,
                device=device,
                generator=generator,
            )
        actions = noise
        for k in range(num_steps):
            # Exact endpoints (no accumulated float drift): τ goes
            # 1 -> 1-1/n -> ... -> 0.
            t = 1.0 - k / num_steps
            t_next = 1.0 - (k + 1) / num_steps
            dt = t_next - t
            time = torch.full((batch,), t, dtype=dtype, device=device)
            velocity = self(memory, state, actions, time)
            if method is SamplingMethod.HEUN:
                predicted = actions + dt * velocity.to(actions.dtype)
                time_next = torch.full((batch,), t_next, dtype=dtype, device=device)
                velocity_next = self(memory, state, predicted, time_next)
                velocity = 0.5 * (velocity + velocity_next)
            actions = actions + dt * velocity.to(actions.dtype)
        return actions

    @override
    def loss(self, memory: ObservationMemory, batch: CollatedBatch[Any]) -> Tensor:
        """Scalar flow-matching loss (see :func:`flow_matching_loss`). DDP
        training calls the module-level function with the wrapper instead —
        gradient hooks require the forward to go through DDP's __call__."""
        return flow_matching_loss(self, memory, batch)

    @override
    @torch.no_grad()
    def predict_chunk(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        num_steps: int = 5,
        method: SamplingMethod = SamplingMethod.HEUN,
    ) -> Tensor:
        """RAW-unit chunk prediction [B, chunk, action_dim]: normalize the
        batch's state with its per-sample stats, integrate the field, and
        unnormalize with the action stats. ``num_steps``/``method`` are
        flow-specific knobs beyond the ActionDecoder minimum."""
        state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
        sampled = self.sample_actions(
            memory,
            state,
            num_steps=num_steps,
            method=method,
            noise=noise,
            generator=generator,
        )
        return (
            sampled.float() * batch.action_stats.std[:, None, :]
            + batch.action_stats.mean[:, None, :]
        )


def flow_matching_loss(
    velocity_model: torch.nn.Module,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> Tensor:
    """``batch`` must already be device-resident; no transfers happen here.
    Actions/state are normalized with each sample's own dataset stats.

    Episode-boundary chunks train on the full ``chunk`` length with
    repeat-last-action targets: lerobot's delta-timestamps query clamps
    indices to the episode range (dataset_reader._get_query_indices), so
    positions past the end already hold the final real action — the
    desired "reach and hold" target — and ``action_is_pad`` is deliberately
    ignored here (decision 2026-07-29: full-chunk targets over masking;
    the expert attends every position, so masked-out padding was still
    silently shaping predictions). Eval stays real-steps-only.

    ``velocity_model`` is the decoder (or its DDP wrapper under torchrun —
    training forwards must go through the wrapper for gradient hooks);
    call convention (memory, state, noisy_actions, tau) is shared by
    FlowDecoder, BijouModel and DDP(FlowDecoder).

    Shapes (batch fields in CollatedBatch's docstring):
      - memory.streams[name]: key/value each [B, kv_heads, P, head_dim]
      - velocity/target: [B, chunk, action_dim]; tau: [B]
      - returns: scalar loss
    """
    actions = (
        batch.actions - batch.action_stats.mean[:, None, :]
    ) / batch.action_stats.std[:, None, :]
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std

    noise = torch.randn_like(actions)
    # π0's time distribution: Beta(1.5, 1) squeezed into (0, 1).
    tau = (
        torch.distributions.Beta(1.5, 1.0)
        .sample((actions.shape[0],))
        .to(actions.device)
    )
    tau = tau * 0.999 + 0.001
    tau_ = tau[:, None, None]
    noisy_actions = tau_ * noise + (1 - tau_) * actions
    target = noise - actions

    velocity = velocity_model(memory, state, noisy_actions, tau)
    return (velocity.float() - target.float()).pow(2).mean()
