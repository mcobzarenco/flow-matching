"""MolmoAct2 first-class port (rig-path scope).

Pre-reg: fontaine/blog/src/posts/2026-08-10-prereg-molmoact2-firstclass-port.md
(owner GO 2026-08-10 20:06Z, items 1-4). The Molmo2 backbone is reused
from ``bijou.molmo2``; this package holds the genuinely new pieces,
starting with the flow-matching action expert (item 1).
"""

from bijou.molmoact2.action_expert import (
    ActionExpert,
    ActionExpertConfig,
    load_action_expert_state,
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
    "ActionExpert",
    "ActionExpertConfig",
    "encoder_attention_mask",
    "extract_kv_states",
    "flow_timesteps",
    "generate_actions",
    "layer_kv_to_sequence",
    "load_action_expert_state",
    "validate_inference_config",
]
