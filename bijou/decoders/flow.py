"""Bijou action expert: a narrow flow-matching decoder over action chunks.

The expert consumes ``[state][action_1..action_chunk]`` tokens and, per layer,
cross-attends one *global-attention* K/V stream exported from the frozen
Gemma 4 backbone (layers 4/9/14 for E2B), then self-attends the suffix, then
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

CONVENTION FREEZE (architecture.md §8.13 decision 2): this π 0-style
parametrization is legacy-frozen with this module and its checkpoint
lineage. New flow code uses the ASCENDING convention (t=0 noise → t=1
data, target ``x − ε`` — ``decoders/molmo_flow.py``); the two modules
deliberately share no code, so a sign can never be "harmonized" across
them by accident. An exact weight-space converter exists on paper
(sinusoid reflection into ``time_in_proj``, sign into
``action_out_proj``) if unification is ever worth a lineage migration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast, override

import torch
from torch import Tensor, nn

from ..interface import (
    BijouPrediction,
    CollatedBatch,
    MemoryStream,
    ObservationMemory,
    SamplingMethod,
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
    SuffixBlock,
    apply_scale,
    cross_attention_mask,
)

_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


class SelfAttentionMode(StrEnum):
    """Masking of the expert's self-attention over ``[state][actions]``.

    The state token attends and is attended by everything in both modes;
    in CAUSAL_ACTIONS each action token only attends earlier actions (the
    SmolVLA ablation found this beats bidirectional; π0 uses bidirectional).
    """

    BIDIRECTIONAL = "bidirectional"
    CAUSAL_ACTIONS = "causal_actions"


class TimeConditioning(StrEnum):
    """How flow time τ conditions the expert.

    ADDITIVE: the π0/SmolVLA scheme currently shipped — τ's MLP embedding
    is added to the action token embeddings at the input; the state token
    is not time-conditioned; layers are unconditioned.
    ADARMS: DiT-style adaptive RMSNorm — τ's embedding drives a per-layer
    zero-initialized head producing a per-channel SCALE on each sublayer's
    (RMS)norm output and a GATE on its residual contribution (no shift —
    RMSNorm carries no bias; the gate is the additive-injection route).
    Identity at init. The intended successor to ADDITIVE.
    """

    ADDITIVE = "additive"
    ADARMS = "adarms"


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    """Architecture of the flow-matching action expert. Use
    :func:`bijou.loading.default_expert_config` to derive one from a backbone
    config with the blocks-schedule knobs."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    hidden_activation: str
    rms_norm_eps: float

    self_attention_mode: SelfAttentionMode
    self_attention_rope_theta: float

    # Cross-attention geometry, copied from the backbone's global layers.
    cross_attention_heads: int
    cross_attention_head_dim: int
    cross_attention_rope: RopeParameters
    # Backbone layer index each expert layer cross-attends; the length of
    # this tuple is the expert depth.
    cross_attention_schedule: tuple[int, ...]

    action_dim: int
    state_dim: int
    chunk_size: int
    time_embed_dim: int
    time_conditioning: TimeConditioning
    # SnapFlow φ_s: a zero-initialized target-time embedding added to the
    # τ embedding, enabling shortcut (s=0, one-step) conditioning. Default
    # off — checkpoints predating the field load unchanged, and the
    # zero-init makes an extended model exactly the unextended one until
    # trained (the distill warm-start identity).
    target_time_embed: bool = False

    def __post_init__(self) -> None:
        if not self.cross_attention_schedule:
            raise ValueError("cross_attention_schedule must not be empty")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("self-attention head_dim must be even (RoPE)")
        if self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be even")

    @property
    def num_layers(self) -> int:
        return len(self.cross_attention_schedule)

    @property
    def streams(self) -> tuple[int, ...]:
        """Backbone layers the expert conditions on, ascending (their K/V
        exports)."""
        return tuple(sorted(set(self.cross_attention_schedule)))

    def stream_name(self, layer_idx: int) -> str:
        """The stream name a schedule entry resolves to."""
        return kv_stream_name(layer_idx)

    @property
    def suffix_length(self) -> int:
        return 1 + self.chunk_size


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


class FlowDecoder(nn.Module):
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
            config.stream_name(idx) for idx in config.cross_attention_schedule
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

        # SnapFlow φ_s target-time MLP (config.target_time_embed): same
        # two-layer shape as the τ path, output zero-initialized (see
        # reset_parameters) so s has no effect until trained.
        self.target_time_in_proj: nn.Linear | None = None
        self.target_time_out_proj: nn.Linear | None = None
        if config.target_time_embed:
            self.target_time_in_proj = nn.Linear(
                config.time_embed_dim,
                hidden,
                bias=True,
                device=device,
                dtype=dtype,
            )
            self.target_time_out_proj = nn.Linear(
                hidden,
                hidden,
                bias=True,
                device=device,
                dtype=dtype,
            )

        self.layers = nn.ModuleList(
            SuffixBlock(
                hidden_size=hidden,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_activation=config.hidden_activation,
                rms_norm_eps=config.rms_norm_eps,
                cross_attention_heads=config.cross_attention_heads,
                cross_attention_head_dim=config.cross_attention_head_dim,
                modulated=config.time_conditioning is TimeConditioning.ADARMS,
                attn_backend=attn_backend,
                device=device,
                dtype=dtype,
            )
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
            if isinstance(module, SuffixBlock) and module.modulation is not None:
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
        # φ_s output layer zero-initialized: the target-time embedding
        # contributes exactly 0 at init, so an extended model loaded from
        # an unextended checkpoint IS that checkpoint (the SnapFlow
        # warm-start identity; the in_proj keeps its normal init so
        # gradients reach it through the zero out_proj immediately).
        if self.target_time_out_proj is not None:
            nn.init.zeros_(self.target_time_out_proj.weight)
            assert self.target_time_out_proj.bias is not None
            nn.init.zeros_(self.target_time_out_proj.bias)

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
        target_time: Tensor | None = None,
    ) -> Tensor:
        """Velocity of the action chunk at flow time τ; returns
        [B, chunk, action_dim].

        Inputs and memory streams are cast to the expert's own dtype (the
        backbone may run in a different precision, e.g. bf16 vs fp32 expert).

        ``target_time`` (SnapFlow shortcut conditioning, φ_s-extended
        models only): the time s the caller intends to jump to — None
        means s=t, the standard forward; s=0 is one-step mode. Refused
        loudly on models without the embedding.

        Shapes:
          - memory.streams[name].key/value: [B, kv_heads, P, head_dim]
          - memory.padding_mask (when present): [B, P]  (True = real token)
          - state: [B, state_dim]
          - noisy_actions: [B, chunk, action_dim]
          - time: [B]  (flow times in [0, 1])
          - target_time (when given): [B]  (target times in [0, 1])
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
        if self.target_time_in_proj is not None:
            assert self.target_time_out_proj is not None
            target = time if target_time is None else target_time
            target_embeds = sinusoidal_time_embedding(
                target,
                config.time_embed_dim,
            )
            time_embeds = time_embeds + self.target_time_out_proj(
                self.time_act(self.target_time_in_proj(target_embeds.to(dtype))),
            )
        elif target_time is not None:
            raise ValueError(
                "target_time conditioning requires a φ_s-extended decoder "
                "(config target_time_embed); this checkpoint has none",
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
        target_time: float | None = None,
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

        ``target_time`` (φ_s-extended models only): constant SnapFlow
        shortcut conditioning s passed to every solver forward; None =
        standard s=t. The 1-NFE read is ``target_time=0.0`` with
        Euler/1 — from pure noise, ``x̂ = ε − F(ε, s=0, t=1)``.

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
        target = (
            None
            if target_time is None
            else torch.full((batch,), target_time, dtype=dtype, device=device)
        )
        for k in range(num_steps):
            # Exact endpoints (no accumulated float drift): τ goes
            # 1 -> 1-1/n -> ... -> 0.
            t = 1.0 - k / num_steps
            t_next = 1.0 - (k + 1) / num_steps
            dt = t_next - t
            time = torch.full((batch,), t, dtype=dtype, device=device)
            velocity = self(memory, state, actions, time, target)
            if method is SamplingMethod.HEUN:
                predicted = actions + dt * velocity.to(actions.dtype)
                time_next = torch.full((batch,), t_next, dtype=dtype, device=device)
                velocity_next = self(memory, state, predicted, time_next, target)
                velocity = 0.5 * (velocity + velocity_next)
            actions = actions + dt * velocity.to(actions.dtype)
        return actions

    @torch.no_grad()
    def sample_actions_sde(
        self,
        memory: ObservationMemory,
        state: Tensor,
        *,
        noise_level: float = 0.5,
        num_steps: int = 10,
        noise: Tensor | None = None,
        step_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_logprob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Euler–Maruyama decode on Flow-GRPO's marginal-preserving SDE
        (2505.05470 §4/App. A) — the *trainable* stochastic sampler: each
        step's transition is an explicit Gaussian, so its logprob (the
        GRPO ratio numerator) is exact. Fresh-noise ODE re-decodes give
        diversity with no logprob; this is the sampler phase 2 would
        actually train, and the signal probe's cell 5.

        Same τ convention and uniform grid as :meth:`sample_actions`
        (τ=1 noise → τ=0 data). With σ_τ = a·√(τ/(1−τ)) (a =
        ``noise_level``) and signed dt = −1/num_steps:

            x_next = x + [v + (σ_τ²/2τ)·(x + (1−τ)·v)]·dt + σ_τ·√|dt|·ε

        The τ=1 endpoint follows the reference implementation
        (flow_grpo sd3 sde step): the divergent 1/(1−τ) is evaluated at
        the SECOND grid point, so σ at τ=1 is a·√num_steps here. Noise ε
        is drawn per step from ``generator``; ``noise_level=0`` adds
        exactly-zero terms and reproduces Euler :meth:`sample_actions`
        bit-for-bit (oracle: tests/test_flow_sde.py). Euler-only by
        construction — a Heun corrector would break the Gaussian
        transition. φ_s shortcut conditioning is not offered (standard
        s=t models only).

        ``step_noise`` [num_steps, B, chunk, action_dim] supplies every
        step's ε explicitly instead of drawing from ``generator`` — the
        batch-composition-invariant path (per-item keyed streams stacked
        by the caller; a shared generator makes step ε depend on batch
        membership).

        ``return_logprob`` (requires ``noise_level > 0``): also return
        [B] per-sample logprob summed over steps and chunk/action dims.
        """
        config = self.config
        batch = state.shape[0]
        dtype = state.dtype
        device = state.device
        if return_logprob and noise_level == 0.0:
            raise ValueError("logprob is undefined at noise_level=0 (deterministic)")
        if step_noise is not None:
            expected = (num_steps, batch, config.chunk_size, config.action_dim)
            if tuple(step_noise.shape) != expected:
                raise ValueError(
                    f"step_noise shaped {tuple(step_noise.shape)}, expected {expected}",
                )
            step_noise = step_noise.to(device=device, dtype=dtype)
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
        logprob = torch.zeros(batch, dtype=dtype, device=device)
        for k in range(num_steps):
            t = 1.0 - k / num_steps
            t_next = 1.0 - (k + 1) / num_steps
            dt = t_next - t
            time = torch.full((batch,), t, dtype=dtype, device=device)
            velocity = self(memory, state, actions, time, None).to(actions.dtype)
            # Endpoint rule: at k=0 (τ=1) the 1−τ denominator comes from
            # the second grid point (see docstring).
            denominator = 1.0 - t if k > 0 else 1.0 - (1.0 - 1.0 / num_steps)
            sigma = noise_level * math.sqrt(t / denominator)
            drift = velocity + (sigma**2 / (2.0 * t)) * (actions + (1.0 - t) * velocity)
            mean = actions + dt * drift
            if noise_level == 0.0:
                actions = mean
                continue
            std = sigma * math.sqrt(-dt)
            epsilon = (
                step_noise[k]
                if step_noise is not None
                else torch.randn(
                    batch,
                    config.chunk_size,
                    config.action_dim,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
            )
            actions = mean + std * epsilon
            if return_logprob:
                logprob += (
                    -0.5 * epsilon.square() - math.log(std) - _HALF_LOG_TWO_PI
                ).sum(dim=(1, 2))
        if return_logprob:
            return actions, logprob
        return actions

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
        target_time: float | None = None,
        sde_noise_level: float | None = None,
        sde_step_noise: Tensor | None = None,
    ) -> BijouPrediction:
        """RAW-unit chunk prediction (BijouPrediction, generations None —
        flow has no text surface): normalize the batch's state with its
        per-sample stats, integrate the field, and unnormalize with the
        action stats. ``num_steps``/``method``/``target_time`` are
        flow-specific solver knobs (other decoder kinds have none).

        ``sde_noise_level`` routes the decode through
        :meth:`sample_actions_sde` (Euler-only, no φ_s) with the same
        normalization seam — ``method`` must be EULER and ``target_time``
        None so an SDE read can never silently wear ODE solver knobs."""
        state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
        if noise is None:
            # The identical draw sample_actions would make (same shape/
            # dtype/device/generator ⇒ bit-exact result and generator
            # consumption) — made here so the prediction can carry it.
            noise = torch.randn(
                state.shape[0],
                self.config.chunk_size,
                self.config.action_dim,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
        if sde_noise_level is not None:
            if method is not SamplingMethod.EULER:
                raise ValueError(
                    "the SDE decode is Euler-only (a Heun corrector breaks "
                    f"the Gaussian transition), got method={method.name}",
                )
            if target_time is not None:
                raise ValueError(
                    "sample_actions_sde offers no φ_s shortcut conditioning "
                    f"— target_time must be None, got {target_time}",
                )
            sampled = self.sample_actions_sde(
                memory,
                state,
                noise_level=sde_noise_level,
                num_steps=num_steps,
                noise=noise,
                step_noise=sde_step_noise,
                generator=generator,
            )
            assert isinstance(sampled, Tensor)  # return_logprob not requested
        elif sde_step_noise is not None:
            raise ValueError("sde_step_noise without sde_noise_level")
        else:
            sampled = self.sample_actions(
                memory,
                state,
                num_steps=num_steps,
                method=method,
                noise=noise,
                generator=generator,
                target_time=target_time,
            )
        chunks = (
            sampled.float() * batch.action_stats.std[:, None, :]
            + batch.action_stats.mean[:, None, :]
        )
        return BijouPrediction(actions=chunks, generations=None, noise=noise)


def flow_matching_loss(
    decoder: FlowDecoder,
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

    The flow objective as a module-level function (BijouModel.loss
    dispatches here); training's single DDP wrapper hooks gradients at the
    train-step level, so the decoder is always the raw module.

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

    velocity = decoder(memory, state, noisy_actions, tau)
    return (velocity.float() - target.float()).pow(2).mean()


def flow_matching_loss_sums(
    decoder: FlowDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor]:
    """Sum-form objective for chunked backward: (squared-error SUM with
    graph, element count). Dividing by the FULL-batch element count and
    summing over chunks reproduces :func:`flow_matching_loss`'s mean
    over ``[B, chunk, action_dim]`` exactly (up to fp reduction order)
    — every element weighs equally, so the count is just numel. Noise
    and tau draw per CALL: a chunked step consumes the RNG stream in
    chunk-shaped draws (a different, equally-distributed realization
    than one full-batch draw — same law, not the same bytes)."""
    actions = (
        batch.actions - batch.action_stats.mean[:, None, :]
    ) / batch.action_stats.std[:, None, :]
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std

    noise = torch.randn_like(actions)
    tau = (
        torch.distributions.Beta(1.5, 1.0)
        .sample((actions.shape[0],))
        .to(actions.device)
    )
    tau = tau * 0.999 + 0.001
    tau_ = tau[:, None, None]
    noisy_actions = tau_ * noise + (1 - tau_) * actions
    target = noise - actions

    velocity = decoder(memory, state, noisy_actions, tau)
    squared = (velocity.float() - target.float()).pow(2)
    return squared.sum(), torch.tensor(squared.numel(), device=squared.device)


# SnapFlow (arXiv:2604.05656) loss mix, frozen in
# code: L = α·L_FM + (1−α)·λ·L_shortcut.
SNAPFLOW_ALPHA = 0.5
SNAPFLOW_LAMBDA = 0.1


def _snapflow_squared_errors(
    decoder: FlowDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor]:
    """Per-element squared errors of the two SnapFlow terms, each
    [B, chunk, action_dim]: (flow-matching, shortcut). Shared machinery
    of the mean- and sum-form objectives so they consume RNG identically.

    L_FM is the standard objective (s=t forwards — φ_s trained, not
    bypassed). L_shortcut regresses the one-step field at the pure-noise
    end (the 1-NFE deployment point, t=1): a fresh ε ~ N(0, I), a
    stop-gradient two-step-Euler estimate of the full-jump mean velocity
    — no EMA teacher, the model distills from itself —
      x_mid    = ε − ½·sg F(ε,     s=1, t=1)      (midpoint, τ=½)
      v_target = ½·[sg F(ε, s=1, t=1) + sg F(x_mid, s=½, t=½)]
      L        = ‖F(ε, s=0, t=1) − v_target‖²
    so 1-NFE inference is exactly ``x̂ = ε − F(ε, s=0, t=1)``. Three
    expert forwards (2 sg + 1 grad) per shortcut sample, all against the
    ONE observation memory the caller encoded — the prefix encode is
    shared across every term."""
    actions = (
        batch.actions - batch.action_stats.mean[:, None, :]
    ) / batch.action_stats.std[:, None, :]
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std

    # Flow-matching term: identical draws/machinery to flow_matching_loss.
    noise = torch.randn_like(actions)
    tau = (
        torch.distributions.Beta(1.5, 1.0)
        .sample((actions.shape[0],))
        .to(actions.device)
    )
    tau = tau * 0.999 + 0.001
    tau_ = tau[:, None, None]
    noisy_actions = tau_ * noise + (1 - tau_) * actions
    target = noise - actions
    velocity = decoder(memory, state, noisy_actions, tau)
    fm_squared = (velocity.float() - target.float()).pow(2)

    # Shortcut term.
    batch_size = actions.shape[0]
    epsilon = torch.randn_like(actions)
    ones = torch.ones(batch_size, device=actions.device, dtype=tau.dtype)
    halves = torch.full_like(ones, 0.5)
    zeros = torch.zeros_like(ones)
    with torch.no_grad():
        v_start = decoder(memory, state, epsilon, ones, ones)
        x_mid = epsilon - 0.5 * v_start.to(epsilon.dtype)
        v_mid = decoder(memory, state, x_mid, halves, halves)
        v_target = 0.5 * (v_start.float() + v_mid.float())
    v_one_step = decoder(memory, state, epsilon, ones, zeros)
    shortcut_squared = (v_one_step.float() - v_target).pow(2)
    return fm_squared, shortcut_squared


def snapflow_distill_loss(
    decoder: FlowDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> Tensor:
    """SnapFlow self-distillation objective (mean form):
    ``SNAPFLOW_ALPHA·mean(fm) + (1−SNAPFLOW_ALPHA)·SNAPFLOW_LAMBDA·
    mean(shortcut)``. Requires a φ_s-extended decoder (the s=0 forward
    refuses otherwise). Same contract as :func:`flow_matching_loss`."""
    fm_squared, shortcut_squared = _snapflow_squared_errors(
        decoder,
        memory,
        batch,
    )
    return (
        SNAPFLOW_ALPHA * fm_squared.mean()
        + (1 - SNAPFLOW_ALPHA) * SNAPFLOW_LAMBDA * shortcut_squared.mean()
    )


def snapflow_distill_loss_sums(
    decoder: FlowDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
) -> tuple[Tensor, Tensor]:
    """Sum-form SnapFlow objective for chunked backward: (weighted
    squared-error SUM with graph, element count). Both terms share one
    element count (each is a full [B, chunk, action_dim] grid), so
    ``sum over chunks / full-batch count`` reproduces
    :func:`snapflow_distill_loss` exactly (up to fp reduction order) —
    the same contract as :func:`flow_matching_loss_sums`."""
    fm_squared, shortcut_squared = _snapflow_squared_errors(
        decoder,
        memory,
        batch,
    )
    weighted_sum = (
        SNAPFLOW_ALPHA * fm_squared.sum()
        + (1 - SNAPFLOW_ALPHA) * SNAPFLOW_LAMBDA * shortcut_squared.sum()
    )
    return weighted_sum, torch.tensor(
        fm_squared.numel(),
        device=fm_squared.device,
    )
