"""Checkpoint-compatibility gate for the modularity refactor.

The expert's state_dict key set must never drift: safetensors keys are
attribute paths, and every existing checkpoint (cont45k, the adaRMS
lineage, the rig fine-tunes) must keep loading strictly. The fixtures
were captured from the pre-refactor ActionExpert (one per
time-conditioning mode); any rename of a module attribute shows up here
as a diff against them.
"""

from __future__ import annotations

import json
from pathlib import Path

from bijou.modelling.decoders.flow import (
    FlowDecoder,
    FlowDecoderConfig,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.modelling.nn import RopeParameters, RopeType

FIXTURES = Path(__file__).parent / "fixtures"


def build(time_conditioning: TimeConditioning) -> FlowDecoder:
    config = FlowDecoderConfig(
        hidden_size=64,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
        self_attention_rope_theta=10_000.0,
        cross_attention_heads=2,
        cross_attention_head_dim=32,
        cross_attention_rope=RopeParameters(
            rope_type=RopeType.PROPORTIONAL,
            rope_theta=1_000_000.0,
            factor=1.0,
            partial_rotary_factor=0.62,
        ),
        cross_attention_schedule=(1, 3, 5),
        action_dim=6,
        state_dim=6,
        chunk_size=50,
        time_embed_dim=256,
        time_conditioning=time_conditioning,
    )
    return FlowDecoder(config)


def test_expert_state_dict_keys_are_stable() -> None:
    for mode, fixture in (
        (TimeConditioning.ADDITIVE, "expert_keys_additive.json"),
        (TimeConditioning.ADARMS, "expert_keys_adarms.json"),
    ):
        expected = json.loads((FIXTURES / fixture).read_text())
        actual = sorted(build(mode).state_dict().keys())
        assert actual == expected, (
            f"{mode}: decoder state_dict keys drifted from the pre-refactor "
            f"ActionExpert fixture — existing checkpoints would no longer "
            f"load. missing={sorted(set(expected) - set(actual))} "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
