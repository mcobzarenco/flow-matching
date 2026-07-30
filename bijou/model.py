"""Bijou: frozen Gemma 4 prefix encoder + flow-matching action expert.

The prefix (chat-templated instruction + camera images, assembled by the
Gemma4 processor) runs once per observation through the truncated backbone —
only the non-KV-shared layers, e.g. layers 0–14 for E2B. The K/V of the
*global-attention* layers (4/9/14 for E2B) are exported as an
:class:`EncodedPrefix` and cached; the expert then denoises a chunk of
actions against them, with fresh robot state, at ~10 Euler steps per chunk.
"""

from __future__ import annotations

from enum import Enum
from typing import override

import torch
from torch import Tensor, nn

from .expert import ActionExpert
from .gemma4.cache import KVCache
from .gemma4.model import Gemma4Model
from .interface import EncodedPrefix, MemoryStream, kv_stream_name


class SamplingMethod(Enum):
    """ODE solver for integrating the velocity field from noise to actions.

    EULER: 1 model evaluation per step, first-order (global error O(1/n)).
    HEUN: explicit trapezoidal predictor-corrector, 2 evaluations per step,
    second-order (O(1/n²)); the better quality-per-evaluation trade for all
    but the very smallest step counts (Karras et al., EDM).
    """

    EULER = "euler"
    HEUN = "heun"


class BijouModel(nn.Module):
    """Composed VLA. Build from a checkpoint with
    :func:`bijou.loading.from_backbone`."""

    def __init__(self, backbone: Gemma4Model, expert: ActionExpert) -> None:
        super().__init__()
        text = backbone.config.text
        for layer_idx in expert.config.streams:
            if not 0 <= layer_idx < text.num_hidden_layers:
                raise ValueError(
                    f"cross-attention stream {layer_idx} outside backbone "
                    f"(has {text.num_hidden_layers} layers)",
                )
            if (
                text.head_dim_for_layer(layer_idx)
                != expert.config.cross_attention_head_dim
            ):
                raise ValueError(
                    f"stream {layer_idx} head_dim "
                    f"{text.head_dim_for_layer(layer_idx)} != expert cross-attention "
                    f"head_dim {expert.config.cross_attention_head_dim}",
                )
        self.backbone = backbone
        self.expert = expert

    def encode_prefix(
        self,
        input_ids: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> EncodedPrefix:
        """Run the truncated backbone over the multimodal prefix and export
        the expert's K/V streams (returns an EncodedPrefix of width P).
        Cache the result across flow steps (and, if the observation is
        unchanged, across replans).

        For right-padded batches (mixed-length instructions), pass the HF
        ``attention_mask`` (True/1 = real token) as ``padding_mask``; it masks
        both the backbone's self-attention and the expert's cross-attention.

        Shapes (P = prompt/prefix tokens = the encoded sequence length;
        ``images`` = Σ per-sample camera images, not B):
          - input_ids: [B, P]
          - pixel_values: [images, patches, 3·patch_size²]
          - image_position_ids: [images, patches, 2]  ((x, y) spatial ids)
          - padding_mask (when present): [B, P]  (True = real token)
          - returns EncodedPrefix: streams["kv{layer}"].key/value each
            [B, kv_heads, P, head_dim]; padding_mask [B, P] or None
        """
        inputs_embeds, per_layer_inputs = self.backbone.embed_multimodal(
            input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        cache = KVCache(self.backbone.config.text)
        self.backbone.language_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            padding_mask=padding_mask,
            cache=cache,
            # The deepest exported layer's K/V depend only on its input:
            # its attention/MLP and any deeper layers are dead weight here
            # (~1/15 of decoder compute at the default E2B schedule).
            kv_stop_layer=max(self.expert.config.streams),
        )
        streams: dict[str, MemoryStream] = {}
        for layer_idx in self.expert.config.streams:
            layer = cache.layers[layer_idx]
            assert layer.keys is not None and layer.values is not None
            streams[kv_stream_name(layer_idx)] = MemoryStream(
                key=layer.keys,
                value=layer.values,
            )
        return EncodedPrefix(
            streams=streams,
            length=input_ids.shape[1],
            padding_mask=padding_mask,
        )

    @override
    def forward(
        self,
        prefix: EncodedPrefix,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Velocity of the action chunk at flow time ``time`` (see
        ``bijou.expert`` for the flow convention); returns
        [B, chunk, action_dim].

        Shapes:
          - prefix.streams[name].key/value: [B, kv_heads, P, head_dim]
          - state: [B, state_dim]
          - noisy_actions: [B, chunk, action_dim]
          - time: [B]
        """
        return self.expert(prefix, state, noisy_actions, time)

    @torch.no_grad()
    def sample_actions(
        self,
        prefix: EncodedPrefix,
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
        evaluation.

        Shapes:
          - prefix.streams[name].key/value: [B, kv_heads, P, head_dim]
          - state: [B, state_dim]
          - noise (when given): [B, chunk, action_dim]
          - returns: [B, chunk, action_dim]
        """
        config = self.expert.config
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
            velocity = self.expert(prefix, state, actions, time)
            if method is SamplingMethod.HEUN:
                predicted = actions + dt * velocity.to(actions.dtype)
                time_next = torch.full((batch,), t_next, dtype=dtype, device=device)
                velocity_next = self.expert(prefix, state, predicted, time_next)
                velocity = 0.5 * (velocity + velocity_next)
            actions = actions + dt * velocity.to(actions.dtype)
        return actions
