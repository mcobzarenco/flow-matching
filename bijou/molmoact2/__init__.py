"""MolmoAct2 first-class port (rig-path scope).

Pre-reg: fontaine/blog/src/posts/2026-08-10-prereg-molmoact2-firstclass-port.md
(owner GO 2026-08-10 20:06Z, items 1-4). The Molmo2 backbone is reused
from ``bijou.molmo2``; this package holds the genuinely new pieces: the
flow-matching action expert + backbone wiring (item 1) and the
action-side processing — prompt template, discrete state tokens,
q01/q99 norm stats (item 2).
"""

from bijou.molmoact2.action_expert import (
    ActionExpert,
    ActionExpertConfig,
    load_action_expert_state,
)
from bijou.molmoact2.predictor import (
    IMAGE_TOKEN_STRINGS,
    MolmoAct2Predictor,
    action_expert_from_config,
    load_action_expert,
    resolve_image_token_ids,
)
from bijou.molmoact2.processing import (
    PackedActionExample,
    QuantileStats,
    build_robot_prompt,
    discrete_state_string,
    encode_action_prompt,
    load_norm_stats,
    normalize_state,
    normalize_task_text,
    pack_action_example,
    process_image_resize,
    to_uint8_rgb,
    unnormalize_action,
)
from bijou.molmoact2.wiring import (
    encoder_attention_mask,
    extract_kv_states,
    flow_timesteps,
    generate_actions,
    layer_kv_to_sequence,
    validate_inference_config,
)

__all__ = [
    "IMAGE_TOKEN_STRINGS",
    "ActionExpert",
    "ActionExpertConfig",
    "MolmoAct2Predictor",
    "PackedActionExample",
    "QuantileStats",
    "action_expert_from_config",
    "build_robot_prompt",
    "discrete_state_string",
    "encode_action_prompt",
    "encoder_attention_mask",
    "extract_kv_states",
    "flow_timesteps",
    "generate_actions",
    "layer_kv_to_sequence",
    "load_action_expert",
    "load_action_expert_state",
    "load_norm_stats",
    "normalize_state",
    "normalize_task_text",
    "pack_action_example",
    "process_image_resize",
    "resolve_image_token_ids",
    "to_uint8_rgb",
    "unnormalize_action",
    "validate_inference_config",
]
