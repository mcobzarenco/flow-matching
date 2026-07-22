"""Bijou: frozen Gemma 4 prefix encoder + flow-matching action expert.

The prefix (chat-templated instruction + camera images, assembled by the
Gemma4 processor) runs once per observation through the truncated backbone —
only the non-KV-shared layers, e.g. layers 0–14 for E2B. The K/V of the
*global-attention* layers (4/9/14 for E2B) are exported as :class:`PrefixKV`
and cached; the expert then denoises a chunk of actions against them, with
fresh robot state, at ~10 Euler steps per chunk.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .expert import ActionExpert, PrefixKV, StreamKV
from .gemma4.cache import KVCache
from .gemma4.model import Gemma4Model


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
                    f"(has {text.num_hidden_layers} layers)"
                )
            if (
                text.head_dim_for_layer(layer_idx)
                != expert.config.cross_attention_head_dim
            ):
                raise ValueError(
                    f"stream {layer_idx} head_dim "
                    f"{text.head_dim_for_layer(layer_idx)} != expert cross-attention "
                    f"head_dim {expert.config.cross_attention_head_dim}"
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
    ) -> PrefixKV:
        """Run the truncated backbone over the multimodal prefix and export
        the expert's K/V streams. Cache the result across flow steps (and, if
        the observation is unchanged, across replans).

        For right-padded batches (mixed-length instructions), pass the HF
        ``attention_mask`` (True/1 = real token) as ``padding_mask``; it masks
        both the backbone's self-attention and the expert's cross-attention.
        """
        inputs_embeds, per_layer_inputs = self.backbone.embed_multimodal(
            input_ids, pixel_values=pixel_values, image_position_ids=image_position_ids
        )
        cache = KVCache(self.backbone.config.text)
        self.backbone.language_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            padding_mask=padding_mask,
            cache=cache,
        )
        streams: StreamKV = {}
        for layer_idx in self.expert.config.streams:
            layer = cache.layers[layer_idx]
            assert layer.keys is not None and layer.values is not None
            streams[layer_idx] = (layer.keys, layer.values)
        return PrefixKV(
            streams=streams, length=input_ids.shape[1], padding_mask=padding_mask
        )

    def forward(
        self,
        prefix: PrefixKV,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Velocity of the action chunk at flow time ``time`` (see
        ``bijou.expert`` for the flow convention). Shapes: state
        [B, state_dim], noisy_actions [B, chunk, action_dim], time [B]."""
        return self.expert(prefix, state, noisy_actions, time)

    @torch.no_grad()
    def sample_actions(
        self,
        prefix: PrefixKV,
        state: Tensor,
        *,
        num_steps: int = 10,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Euler integration of the velocity field from τ=1 (noise) to τ=0.

        Pass ``noise`` of shape [B, chunk, action_dim] (or a seeded
        ``generator``) for deterministic evaluation. Returns the action chunk
        [B, chunk_size, action_dim].
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
        dt = -1.0 / num_steps
        time = torch.ones(batch, dtype=dtype, device=device)
        for _ in range(num_steps):
            velocity = self.expert(prefix, state, actions, time)
            actions = actions + dt * velocity.to(actions.dtype)
            time = time + dt
        return actions
