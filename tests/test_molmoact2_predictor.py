"""Oracles for the MolmoAct2 first-class predictor (port item 3, gate G3).

CPU-scale tests over a tiny trunk + tiny expert wearing the REAL token-id
layout (wide-vocab embedding: base 151,936 + 4,096 extension rows covers
BOS 151,645, the state/action vocab at 151,669+, and the rig-ft image
specials at 155,648+), so the packed prompt from ``bijou.molmoact2``
flows through ``build_input_embeddings`` unchanged. The GPU-scale parity
read vs the banked 240-row HF anchors stays a script
(``fontaine/scripts/molmoact2_e2e_parity.py``, gate G2).

Covered here:

- predict_action == the composition of its published parts (pack ->
  trunk KV -> generate -> their slice/unnormalize tail) with a shared
  generator seed — the tail ORDER (width slice, ``n_obs_steps`` chunk
  slice, clamp+unnormalize, dtype round-trip) can't silently drift;
- noise determinism (same seed byte-equal, different seed different);
- checkpoint-dependent image-token id resolution (the released-vs-rig-ft
  re-homing class) + the ``image_type_mask`` in the collated batch;
- horizon/metadata guards raise loudly.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
import torch

from bijou.molmo2.config import Molmo2Config
from bijou.molmo2.model import Molmo2Model
from bijou.molmo2.text import Molmo2TextModel
from bijou.molmo2.vision import Molmo2VisionBackbone
from bijou.molmoact2 import (
    ActionExpertConfig,
    MolmoAct2Predictor,
    QuantileStats,
    pack_action_example,
    resolve_image_token_ids,
    unnormalize_action,
)
from bijou.molmoact2.predictor import action_expert_from_config
from bijou.molmoact2.processing import (
    ACTION_OUTPUT_ID,
    BOS_ID,
    IM_END_ID,
    IM_PATCH_ID,
    IM_START_ID,
    PAD_ID,
    STATE_TOKEN_0_ID,
)
from bijou.molmoact2.wiring import encoder_attention_mask, generate_actions

# The rig-ft export's token layout (the goldens' layout): base vocab
# 151,936 + extension rows through the 155,648+ image specials.
_BASE_VOCAB = 151_936
_EXTRA_VOCAB = 4_096

_SPECIAL_IDS: dict[str, int] = {
    "<|endoftext|>": PAD_ID,
    "<|im_start|>": 151_644,
    "<|im_end|>": BOS_ID,
    "<setup_start>": 151_669,
    "<setup_end>": 151_670,
    "<control_start>": 151_671,
    "<control_end>": 151_672,
    "<state_start>": 151_673,
    "<state_end>": 151_674,
    "<action_output>": ACTION_OUTPUT_ID,
    "<action_start>": 151_932,
    "<action_end>": 151_933,
    "<im_start>": IM_START_ID,
    "<im_end>": IM_END_ID,
    "<im_patch>": IM_PATCH_ID,
    "<im_col>": 151_939 + 3_712,  # 155_651
    "<low_res_im_start>": 155_652,
}
for _n in range(256):
    _SPECIAL_IDS[f"<state_{_n}>"] = STATE_TOKEN_0_ID + _n


class _StubTokenizerBackend:
    def token_to_id(self, token: str) -> int | None:
        return _SPECIAL_IDS.get(token)


class _StubTokenizer:
    """Molmo2TextTokenizer-protocol stand-in: special tokens split first
    (like `tokenizers` added tokens) at their REAL ids; plain text maps
    one id per whitespace-separated word (BPE-like compression, so their
    sequence budget holds). Deterministic."""

    tokenizer = _StubTokenizerBackend()

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens  # the stub has no auto-specials
        ids: list[int] = []
        specials = sorted(_SPECIAL_IDS, key=len, reverse=True)
        word = 0
        while text:
            for token in specials:
                if text.startswith(token):
                    if word:
                        ids.append(word % 50_000)
                        word = 0
                    ids.append(_SPECIAL_IDS[token])
                    text = text[len(token) :]
                    break
            else:
                char, text = text[0], text[1:]
                if char.isspace():
                    if word:
                        ids.append(word % 50_000)
                        word = 0
                else:
                    word = word * 31 + ord(char)
        if word:
            ids.append(word % 50_000)
        return ids


def _tiny_trunk_config() -> dict[str, Any]:
    return {
        "model_type": "molmoact2",
        "dtype": "float32",
        "tie_word_embeddings": False,
        "image_patch_id": IM_PATCH_ID,
        "text_config": {
            "model_type": "molmo2_text",
            "vocab_size": _BASE_VOCAB,
            "additional_vocab_size": _EXTRA_VOCAB,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_act": "silu",
            "layer_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "use_qk_norm": True,
            "qk_norm_type": "qwen3",
            "qkv_bias": False,
            "norm_after": False,
            "rope_scaling": None,
            "rope_scaling_layers": None,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "residual_dropout": 0.0,
        },
        "vit_config": {
            "model_type": "molmo2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "image_patch_size": 14,  # the real 378/14 resize-mode grid
            "image_num_pos": 729,
            "float32_attention": True,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
        },
        "adapter_config": {
            "model_type": "molmo2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "text_hidden_size": 32,
            "vit_layers": [-1, -2],
            "float32_attention": True,
            "pooling_attention_mask": True,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
            "image_feature_dropout": 0.0,
        },
    }


_MAX_ACTION_DIM = 8
_ACTION_DIM = 3  # < max: the padded-dim path is real
_MAX_HORIZON = 4
_N_OBS_STEPS = 2  # > 1: the chunk slice drops a leading row
_N_ACTION_STEPS = 2
_FLOW_STEPS = 3


def _tiny_stats(dim: int, offset: float) -> QuantileStats:
    return QuantileStats(
        q01=torch.arange(dim, dtype=torch.float32) - offset,
        q99=torch.arange(dim, dtype=torch.float32) + offset,
    )


@pytest.fixture(scope="module")
def predictor() -> MolmoAct2Predictor:
    torch.manual_seed(0)
    config = Molmo2Config.from_dict(_tiny_trunk_config())
    assert config.vit is not None and config.adapter is not None
    text = Molmo2TextModel(config.text, lm_head=True)
    vision = Molmo2VisionBackbone(config.vit, config.adapter)
    trunk = Molmo2Model(text, vision, image_patch_id=config.image_patch_id)
    trunk.eval()
    trunk.requires_grad_(False)
    expert = ActionExpertConfig(
        max_horizon=_MAX_HORIZON,
        max_action_dim=_MAX_ACTION_DIM,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        timestep_embed_dim=8,
    ).build(llm_kv_dim=config.text.head_dim * config.text.num_key_value_heads)
    expert.eval()
    expert.requires_grad_(False)
    return MolmoAct2Predictor(
        trunk=trunk,
        expert=expert,
        tokenizer=_StubTokenizer(),  # type: ignore[arg-type] — protocol stand-in
        action_stats=_tiny_stats(_ACTION_DIM, 2.0),
        state_stats=_tiny_stats(_ACTION_DIM, 1.5),
        metadata={
            "setup_type": "tiny rig",
            "control_mode": "absolute joint pose",
            "action_horizon": _MAX_HORIZON,
            "n_action_steps": _N_ACTION_STEPS,
        },
        image_token_ids=(IM_START_ID, IM_END_ID, IM_PATCH_ID),
        action_mode="continuous",
        eos_token_id=BOS_ID,
        action_start_token_id=151_932,
        action_end_token_id=151_933,
        max_action_horizon=_MAX_HORIZON,
        max_action_dim=_MAX_ACTION_DIM,
        n_obs_steps=_N_OBS_STEPS,
        num_state_tokens=256,
        flow_matching_num_steps=_FLOW_STEPS,
        mask_action_dim_padding=True,
    )


def _observation() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    return {
        "images": [
            rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8),
        ],
        "task": "Pick up the cube.",
        "state": torch.tensor([0.5, -0.25, 1.75]),
    }


def test_predict_action_matches_composed_parts(
    predictor: MolmoAct2Predictor,
) -> None:
    """The one-call path equals pack -> trunk KV -> generate -> THEIR
    tail applied by hand, byte-for-byte, given the same noise."""
    obs = _observation()
    pred = predictor.predict_action(
        **obs,
        generator=torch.Generator().manual_seed(11),
    )

    inputs = predictor.batch_inputs(obs["images"], obs["task"], obs["state"])
    kv_states, enc_mask = predictor.prompt_kv(inputs)
    chunk = generate_actions(
        predictor.expert,
        encoder_kv_states=kv_states,
        encoder_attention_mask=enc_mask,
        action_horizon=_MAX_HORIZON,
        action_dim_is_pad=torch.tensor([[False] * _ACTION_DIM + [True] * 5]),
        num_steps=_FLOW_STEPS,
        generator=torch.Generator().manual_seed(11),
    )
    start = _N_OBS_STEPS - 1
    sliced = chunk[..., :_ACTION_DIM][:, start : start + _N_ACTION_STEPS]
    expected = (
        unnormalize_action(sliced, predictor.action_stats)
        .to(sliced.dtype)
        .to(torch.float32)
    )
    assert pred.shape == (1, _N_ACTION_STEPS, _ACTION_DIM)
    torch.testing.assert_close(pred, expected, rtol=0, atol=0)


def test_noise_determinism(predictor: MolmoAct2Predictor) -> None:
    obs = _observation()
    first = predictor.predict_action(
        **obs,
        generator=torch.Generator().manual_seed(3),
    )
    again = predictor.predict_action(
        **obs,
        generator=torch.Generator().manual_seed(3),
    )
    other = predictor.predict_action(
        **obs,
        generator=torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    assert not torch.equal(first, other)


def test_batch_inputs_layout(predictor: MolmoAct2Predictor) -> None:
    """Prompt facts in the collated batch: BOS first, one 198-token image
    expansion per camera marked image-typed, the discrete state clause,
    the trailing ``<action_output>``, and per-image pooled-index shift."""
    obs = _observation()
    inputs = predictor.batch_inputs(obs["images"], obs["task"], obs["state"])
    ids = inputs["input_ids"][0]
    assert ids[0].item() == BOS_ID
    assert ids[-1].item() == ACTION_OUTPUT_ID
    assert int((ids == IM_PATCH_ID).sum()) == 196 * 2
    assert int(inputs["image_type_mask"].sum()) == 198 * 2
    state_tokens = (ids >= STATE_TOKEN_0_ID) & (ids < STATE_TOKEN_0_ID + 256)
    assert int(state_tokens.sum()) == _ACTION_DIM
    assert inputs["crops"].shape == (1, 2, 729, 14 * 14 * 3)
    pooled = inputs["pooled_patches_idx"][0]
    assert pooled.shape == (196 * 2, 4)
    # Second image's pooled rows index into its own view's patch range;
    # -1 ragged-pool markers (27 is odd) survive the shift untouched.
    second = pooled[196:]
    assert int(second[second >= 0].min()) >= 729
    assert int(pooled[:196].max()) < 729
    assert bool((second == -1).any())


def test_resolve_image_token_ids_is_checkpoint_dependent() -> None:
    """The id set comes from the tokenizer at hand — a vocabulary without
    the optional video tokens still resolves (the released-checkpoint
    re-homing class is exercised at GPU scale by the parity script)."""
    ids = resolve_image_token_ids(_StubTokenizer())  # type: ignore[arg-type]
    assert IM_PATCH_ID in ids
    assert set(ids) == {
        IM_PATCH_ID,
        155_651,
        IM_START_ID,
        155_652,
        IM_END_ID,
    }

    class _Empty:
        class tokenizer:  # noqa: N801 — attribute stand-in
            @staticmethod
            def token_to_id(token: str) -> None:
                return None

    with pytest.raises(ValueError, match="none of the MolmoAct2 image tokens"):
        resolve_image_token_ids(_Empty())  # type: ignore[arg-type]


def test_both_mode_encoder_mask() -> None:
    """The released-checkpoint (``action_mode='both'``) branch: every EOS
    position drops out of the cross-attention mask — including the
    leading BOS, which IS ``<|im_end|>`` under their convention — and
    each ``<action_start>..<action_end>`` span masks inclusively, with
    an unmatched start masking through the row end (their pairing)."""
    eos, a0, a1 = 9, 20, 21
    ids = torch.tensor(
        [
            [eos, 5, 6, eos, 7, a0, 8, a1, 3, 4],
            [1, 2, a0, 3, 4, 5, 6, 7, 8, 9],  # unmatched start
        ],
    )
    mask = encoder_attention_mask(
        ids,
        None,
        action_mode="both",
        eos_token_id=eos,
        action_start_token_id=a0,
        action_end_token_id=a1,
    )
    assert mask is not None
    expected = torch.tensor(
        [
            [False, True, True, False, True, False, False, False, True, True],
            [True, True, False, False, False, False, False, False, False, False],
        ],
    )
    torch.testing.assert_close(mask, expected)
    # Continuous mode leaves the base mask untouched even with ids set.
    plain = encoder_attention_mask(
        ids,
        None,
        action_mode="continuous",
        eos_token_id=eos,
        action_start_token_id=a0,
        action_end_token_id=a1,
    )
    assert plain is not None and bool(plain.all())


def test_horizon_guards(predictor: MolmoAct2Predictor) -> None:
    obs = _observation()
    over_horizon = dataclasses.replace(
        predictor,
        metadata={**predictor.metadata, "action_horizon": _MAX_HORIZON + 1},
    )
    with pytest.raises(ValueError, match="exceeds checkpoint max_action_horizon"):
        over_horizon.predict_action(**obs)
    over_steps = dataclasses.replace(
        predictor,
        metadata={**predictor.metadata, "n_action_steps": _MAX_HORIZON + 1},
    )
    with pytest.raises(ValueError, match="exceeds action_horizon"):
        over_steps.predict_action(**obs)
    wide_stats = dataclasses.replace(
        predictor,
        action_stats=_tiny_stats(_MAX_ACTION_DIM + 1, 2.0),
    )
    with pytest.raises(ValueError, match="exceeds max_action_dim"):
        wide_stats.predict_action(**obs)


def test_action_expert_from_config_derives_kv_dim() -> None:
    expert = action_expert_from_config(
        {
            "max_action_horizon": _MAX_HORIZON,
            "max_action_dim": _MAX_ACTION_DIM,
            "action_expert_config": {
                "hidden_size": 16,
                "num_layers": 2,
                "num_heads": 2,
                "mlp_ratio": 4.0,
                "ffn_multiple_of": 16,
                "timestep_embed_dim": 8,
                "context_layer_norm": True,
                "qk_norm": True,
                "qk_norm_eps": 1e-6,
                "rope": True,
                "causal_attn": False,
            },
            "text_config": {"head_dim": 8, "num_key_value_heads": 2},
        },
    )
    assert expert.llm_kv_dim == 16
    assert expert.config.max_horizon == _MAX_HORIZON
    assert len(expert.blocks) == 2


def test_pack_rejects_missing_camera(predictor: MolmoAct2Predictor) -> None:
    with pytest.raises(ValueError, match="at least one camera frame"):
        pack_action_example(
            images=[],
            state=torch.zeros(_ACTION_DIM),
            task="x",
            tokenizer=predictor.tokenizer,
            state_stats=predictor.state_stats,
            setup_type="tiny rig",
            control_mode="absolute joint pose",
        )
