"""Backbone↔action-expert wiring — the inference glue their
``modeling_molmoact2.py`` runs between the VLM trunk and the expert.

Pre-reg posts/2026-08-10-prereg-molmoact2-firstclass-port.md, item 1
remainder. Semantics mirrored off their HF remote-code model (the
`MolmoAct2ForConditionalGeneration` action path) and parity-gated
against it executing on the same inputs
(``fontaine/scripts/molmoact2_wiring_parity.py``):

- KV extraction: one post-RoPE ``(K, V)`` pair per trunk layer off the
  prompt-forward cache, truncated to the prompt length and flattened
  ``[B, kv_heads, S, head_dim] -> [B, S, kv_heads * head_dim]``
  (1024 for the Molmo2 4B trunk); 36 trunk layers condition 36 expert
  blocks 1:1, count hard-checked.
- Encoder mask: the prompt ``attention_mask`` as bool (or
  ``input_ids != -1``). EOS/discrete-span masking exists only under
  ``action_mode='both'`` — out of the rig-path scope, loud guard.
- Sampling: plain ascending-Euler flow integration — ``x ~ N(0, 1)``
  in the expert dtype, ``t = idx/steps`` in fp32, ``x += v/steps`` for
  ``flow_matching_num_steps`` (10) steps. ``mask_action_dim_padding``
  zeroes padded action dims (beyond the 6 real joints of 32) on the
  initial noise AND on both velocity and trajectory every step.
- ``flow_matching_time_offset/scale/beta_*`` are train-side timestep-
  sampling params; the inference loop does not use them. The depth
  gate is OFF in the released/rig checkpoints (guarded below); the
  expert's continuous-state path is unused at inference (state enters
  as discrete prompt tokens — item 2 territory).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from bijou.molmo2.cache import Molmo2KVCache
from bijou.molmoact2.action_expert import ActionExpert

#: Rig-path inference scope this wiring implements. ``action_mode
#: 'both'`` (discrete+continuous heads) and the depth gate are absent
#: from the released SO-100/101 + rig-ft checkpoints.
_SUPPORTED_ACTION_MODE = "continuous"


def validate_inference_config(config: Mapping[str, object]) -> None:
    """Loud guard over a checkpoint's top-level ``config.json``: raise
    on any action-path feature this wiring does not implement."""
    if not config.get("add_action_expert", False):
        raise ValueError("checkpoint has no action expert (add_action_expert falsy)")
    if config.get("action_expert_depth_gate", False):
        raise NotImplementedError(
            "action_expert_depth_gate=true is not wired (off in the released "
            "SO-100/101 and rig-ft checkpoints)",
        )
    mode = config.get("action_mode", _SUPPORTED_ACTION_MODE)
    if mode != _SUPPORTED_ACTION_MODE:
        raise NotImplementedError(
            f"action_mode={mode!r} is not wired (rig-path scope is "
            f"{_SUPPORTED_ACTION_MODE!r}; 'both' needs EOS/discrete-span "
            "encoder masking)",
        )


def layer_kv_to_sequence(
    cache_tensor: Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> Tensor:
    """Flatten one cached layer's K or V to ``[B, S, heads * head_dim]``.

    Mirror of their ``_cache_to_sequence`` including its layout
    inference: a dim matching a known head count identifies the head
    axis; otherwise the smaller of dims 1/2 is assumed to be heads.
    """
    if cache_tensor.dim() != 4:
        raise ValueError(
            f"expected a 4-dim KV cache tensor, got shape {tuple(cache_tensor.shape)}",
        )
    head_candidates = {num_key_value_heads, num_attention_heads}
    heads_first = cache_tensor.shape[1] in head_candidates or (
        cache_tensor.shape[2] not in head_candidates
        and cache_tensor.shape[1] <= cache_tensor.shape[2]
    )
    if heads_first:
        bsz, n_heads, seq_len, head_dim = cache_tensor.shape
        return cache_tensor.permute(0, 2, 1, 3).reshape(
            bsz,
            seq_len,
            n_heads * head_dim,
        )
    bsz, seq_len, n_heads, head_dim = cache_tensor.shape
    return cache_tensor.reshape(bsz, seq_len, n_heads * head_dim)


def extract_kv_states(
    cache: Molmo2KVCache,
    *,
    num_expert_blocks: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    seq_len: int | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """Per-layer conditioning states off a filled prompt cache:
    ``[(K, V)] * num_layers`` in ``[B, S, kv_dim]`` layout, truncated
    to ``seq_len`` (default: the cache's seen length). Trunk layers
    map 1:1 onto expert blocks — the count is a hard check."""
    limit = cache.seen_tokens if seq_len is None else seq_len
    kv_states: list[tuple[Tensor, Tensor]] = []
    for layer in cache.layers:
        keys, values = layer.keys, layer.values
        if keys is None or values is None:
            continue
        if keys.shape[-2] > limit:
            keys = keys[..., :limit, :]
            values = values[..., :limit, :]
        kv_states.append(
            (
                layer_kv_to_sequence(
                    keys,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                ),
                layer_kv_to_sequence(
                    values,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                ),
            ),
        )
    if len(kv_states) != num_expert_blocks:
        raise ValueError(
            f"expected {num_expert_blocks} KV layers (one per expert block), "
            f"got {len(kv_states)}",
        )
    return kv_states


def encoder_attention_mask(
    input_ids: Tensor | None,
    attention_mask: Tensor | None,
    *,
    action_mode: str = _SUPPORTED_ACTION_MODE,
) -> Tensor | None:
    """Bool mask over the prompt for cross-attention. Mirror of their
    ``_get_encoder_attention_mask`` restricted to the continuous
    action mode (the 'both' mode additionally strips EOS + discrete
    action-output spans — out of the rig-path scope)."""
    if action_mode != _SUPPORTED_ACTION_MODE:
        raise NotImplementedError(
            f"action_mode={action_mode!r} encoder masking is not wired",
        )
    if attention_mask is not None:
        return attention_mask.to(dtype=torch.bool).clone()
    if input_ids is not None:
        return input_ids != -1
    return None


def _action_dim_valid_mask(
    target: Tensor,
    action_dim_is_pad: Tensor | None,
) -> Tensor | None:
    """Mirror of their ``_action_dim_valid_mask``: broadcastable bool
    mask of VALID action dims, or None when nothing is padded."""
    if action_dim_is_pad is None:
        return None
    mask = ~action_dim_is_pad.to(device=target.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"action_dim_is_pad width {mask.shape[-1]} does not match "
            f"action width {target.shape[-1]}",
        )
    if mask.shape[0] == 1 and target.shape[0] != 1:
        mask = mask.expand(target.shape[0], -1)
    if mask.shape[0] != target.shape[0]:
        raise ValueError(
            f"action_dim_is_pad batch {mask.shape[0]} does not match "
            f"batch {target.shape[0]}",
        )
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)
    return mask


def _mask_action_dims(
    tensor: Tensor,
    *,
    action_dim_is_pad: Tensor | None,
    enabled: bool,
) -> Tensor:
    if not enabled:
        return tensor
    valid = _action_dim_valid_mask(tensor, action_dim_is_pad)
    if valid is None:
        return tensor
    return tensor.masked_fill(~valid, 0)


def flow_timesteps(
    num_steps: int,
    batch_size: int,
    device: torch.device,
) -> list[Tensor]:
    """Ascending Euler grid ``t = idx/steps`` (0 … 1 − 1/steps), one
    fp32 ``[B]`` tensor per step — fp32 regardless of the expert dtype
    (the sinusoid runs at this precision; the expert casts after)."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    return [
        torch.full((batch_size,), idx / num_steps, device=device, dtype=torch.float32)
        for idx in range(num_steps)
    ]


@torch.no_grad()
def generate_actions(
    expert: ActionExpert,
    *,
    encoder_kv_states: Sequence[tuple[Tensor, Tensor]],
    encoder_attention_mask: Tensor | None = None,
    action_horizon: int | None = None,
    action_dim_is_pad: Tensor | None = None,
    num_steps: int = 10,
    mask_action_dim_padding: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample an action chunk ``[B, horizon, max_action_dim]`` from the
    expert conditioned on extracted trunk KV — their
    ``generate_actions_from_inputs`` flow loop on our modules."""
    if len(encoder_kv_states) == 0:
        raise ValueError("expected at least one encoder KV state")
    horizon = expert.config.max_horizon if action_horizon is None else action_horizon
    if not 1 <= horizon <= expert.config.max_horizon:
        raise ValueError(
            f"action_horizon must be in [1, {expert.config.max_horizon}], "
            f"got {horizon}",
        )
    source = encoder_kv_states[0][0]
    batch_size, device = source.shape[0], source.device
    trajectory = torch.randn(
        (batch_size, horizon, expert.config.max_action_dim),
        device=device,
        dtype=expert.action_embed.weight.dtype,
        generator=generator,
    )
    trajectory = _mask_action_dims(
        trajectory,
        action_dim_is_pad=action_dim_is_pad,
        enabled=mask_action_dim_padding,
    )
    timesteps = flow_timesteps(num_steps, batch_size, device)
    dt = 1.0 / num_steps
    for step_t in timesteps:
        velocity = expert(
            trajectory,
            step_t,
            encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        velocity = _mask_action_dims(
            velocity,
            action_dim_is_pad=action_dim_is_pad,
            enabled=mask_action_dim_padding,
        )
        trajectory = trajectory + dt * velocity
        trajectory = _mask_action_dims(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=mask_action_dim_padding,
        )
    return trajectory
