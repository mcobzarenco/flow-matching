"""Oracles for the MolmoAct2 encoder mode (§8.13 step 4).

The collator OWNS the prompt assembly (template, split point, batching)
and imports the golden-pinned leaf transforms from the frozen port —
so the byte gate here compares ASSEMBLY against the port's
``pack_action_example``, which stays meaningful precisely because only
the leaves are shared (the leaves themselves are gated by the port's
golden fixtures). Pinned:

- narration-off assembly byte-equals the port's batch-1 pack per row
  (ids, crops, pooled indices), left padding and all;
- BOTH split-point layouts: off-ids == on-ids + [<action_output>], the
  opener bytes end the narration-on prefill (decision 7);
- the conditioning mask carries the ``action_mode`` flavor
  ('continuous' = attention mask; 'both' additionally drops EOS
  positions, the leading BOS included — load-bearing for converted
  expert weights) while ``attention_mask`` keeps counting them (the
  positions source must not change);
- bijou-only prompt surfaces are refused loudly (condition/[generate|…]
  has no bytes in this format), as are empty setup/control strings;
- ``MolmoAct2Encoder.encode``: prefix cache retained and filled, the
  conditioning mask rides ``ObservationMemory``, and NO state splice
  happens (state lives in the ids — there is no ``state_proj``).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from bijou.encoders.molmoact2 import (
    GENERATION_OPENER,
    MolmoAct2Encoder,
    MolmoAct2InputsCollator,
    robot_prompt,
)
from bijou.interface import CameraFrame, PromptInputs
from bijou.molmo2.cache import Molmo2KVCache
from bijou.molmo2.config import Molmo2Config
from bijou.molmo2.model import Molmo2Model
from bijou.molmo2.text import Molmo2TextModel
from bijou.molmo2.vision import Molmo2VisionBackbone
from bijou.molmoact2.processing import (
    ACTION_OUTPUT_ID,
    BOS_ID,
    IM_END_ID,
    IM_PATCH_ID,
    IM_START_ID,
    PAD_ID,
    STATE_TOKEN_0_ID,
    QuantileStats,
    build_robot_prompt,
    pack_action_example,
)

# The rig-ft token layout, borrowed from tests/test_molmoact2_predictor.py
# (tests/ is not a package, so the stub is restated).
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
    "<im_col>": 155_651,
    "<low_res_im_start>": 155_652,
}
for _n in range(256):
    _SPECIAL_IDS[f"<state_{_n}>"] = STATE_TOKEN_0_ID + _n


class _StubTokenizerBackend:
    def token_to_id(self, token: str) -> int | None:
        return _SPECIAL_IDS.get(token)


class _StubTokenizer:
    tokenizer = _StubTokenizerBackend()

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
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


def _collator(
    *,
    action_mode: str = "continuous",
    narration: bool = False,
) -> MolmoAct2InputsCollator:
    collator = MolmoAct2InputsCollator(
        "unused",
        setup_type="tiny rig",
        control_mode="absolute joint pose",
        num_state_tokens=256,
        action_mode=action_mode,
        narration=narration,
    )
    # The worker-side lazy build, stubbed (the real path needs the HF
    # checkpoint's tokenizer.json) — the train-collator test precedent.
    collator._tokenizer = _StubTokenizer()  # type: ignore[assignment]
    collator._image_ids = (IM_START_ID, IM_END_ID, IM_PATCH_ID)
    collator._patch_id = IM_PATCH_ID
    collator._eos_id = BOS_ID
    collator._action_start_id = 151_932
    collator._action_end_id = 151_933
    return collator


def _sample(task: str, seed: int, cameras: int = 2) -> PromptInputs:
    generator = torch.Generator().manual_seed(seed)
    frames = tuple(
        CameraFrame(
            name=f"cam{index}",
            kind="unknown",
            image=torch.rand((3, 48, 64), generator=generator),
        )
        for index in range(cameras)
    )
    state = torch.rand((6,), generator=generator) * 2 - 1  # clamp-normalized
    return PromptInputs(
        instruction=task,
        cameras=frames,
        condition_text="",
        state=state,
    )


def _port_pack(sample: PromptInputs) -> Any:
    """The port's batch-1 pack on the SAME inputs (uint8 leaf coercion
    included — the collator must match its float->uint8 path too)."""
    return pack_action_example(
        images=[camera.image for camera in sample.cameras],
        state=sample.state,
        task=sample.instruction,
        tokenizer=_StubTokenizer(),
        state_stats=_IDENTITY_STATS,
        setup_type="tiny rig",
        control_mode="absolute joint pose",
        num_state_tokens=256,
    )


# Maps normalize_state to the identity on already-normalized inputs:
# q01=-1, q99=1 -> 2*(x+1)/2-1 = x.
_IDENTITY_STATS = QuantileStats(
    q01=torch.full((6,), -1.0),
    q99=torch.full((6,), 1.0),
)


def test_narration_off_assembly_matches_port_pack() -> None:
    """Per-row byte equality with the port's batch-1 pack: ids (left-
    padded tails), crops, shifted pooled indices — the step-4 gate."""
    collator = _collator()
    samples = [_sample("Pick up the cube.", 3), _sample("Pick.", 4, cameras=1)]
    batch = collator(samples)
    for row, sample in enumerate(samples):
        pack = _port_pack(sample)
        length = pack.input_ids.shape[0]
        assert torch.equal(batch.input_ids[row, -length:], pack.input_ids)
        assert bool(batch.attention_mask[row, -length:].all())
        assert not bool(batch.attention_mask[row, :-length].any())
        assert bool((batch.input_ids[row, :-length] == PAD_ID).all())
        views = torch.cat([image.crops for image in pack.images], dim=0)
        assert torch.equal(batch.crops[row, : views.shape[0]], views)
        crop_base = 0
        shifted = []
        for image in pack.images:
            idx = image.pooled_idx
            shifted.append(torch.where(idx >= 0, idx + crop_base, idx))
            crop_base += image.crops.shape[0] * image.crops.shape[1]
        expected_pooled = torch.cat(shifted, dim=0)
        assert torch.equal(
            batch.pooled_patches_idx[row, : expected_pooled.shape[0]],
            expected_pooled,
        )


def test_prompt_text_matches_port_template_bytes() -> None:
    """robot_prompt(narration=False) == the port's build_robot_prompt,
    byte-for-byte, for 1 and 2 cameras (assembly owned here, template
    bytes pinned against the reference renderer)."""
    for cameras in (1, 2):
        ours = robot_prompt(
            task="pick up the cube",
            discrete_state="<state_start><state_1><state_end>",
            setup_type="tiny rig",
            control_mode="absolute joint pose",
            num_images=cameras,
            narration=False,
        )
        port = build_robot_prompt(
            task="pick up the cube",
            discrete_state="<state_start><state_1><state_end>",
            setup_type="tiny rig",
            control_mode="absolute joint pose",
            num_images=cameras,
        )
        assert ours == port


def test_split_point_layouts() -> None:
    """decision 7, pinned: narration-off ids == narration-on ids + the
    single ``<action_output>`` token; the narration-on prompt STRING
    ends at the ChatML opener."""
    off = _collator(narration=False)
    on = _collator(narration=True)
    sample = _sample("Pick up the cube.", 5)
    off_ids = off([sample]).input_ids[0]
    on_ids = on([sample]).input_ids[0]
    assert off_ids.shape[0] == on_ids.shape[0] + 1
    assert torch.equal(off_ids[:-1], on_ids)
    assert int(off_ids[-1]) == ACTION_OUTPUT_ID
    on_prompt = robot_prompt(
        task="pick up the cube",
        discrete_state="<state_start><state_1><state_end>",
        setup_type="tiny rig",
        control_mode="absolute joint pose",
        num_images=2,
        narration=True,
    )
    assert on_prompt.endswith(GENERATION_OPENER)


def test_conditioning_mask_carries_action_mode_flavor() -> None:
    """'continuous' = the attention mask verbatim; 'both' additionally
    drops EOS positions (leading BOS included) — while attention_mask
    keeps counting them (the positions source must not change)."""
    sample = _sample("Pick.", 6, cameras=1)
    plain = _collator(action_mode="continuous")([sample])
    assert torch.equal(plain.conditioning_mask, plain.attention_mask.bool())
    both = _collator(action_mode="both")([sample])
    assert torch.equal(both.attention_mask, plain.attention_mask)
    ids = both.input_ids[0]
    eos_positions = ids == BOS_ID  # bos == eos == <|im_end|>
    assert bool(eos_positions.any())  # leading BOS + turn close at least
    assert not bool(both.conditioning_mask[0][eos_positions].any())
    non_eos_real = both.attention_mask[0].bool() & ~eos_positions
    assert bool(both.conditioning_mask[0][non_eos_real].all())


def test_bijou_prompt_surfaces_refused() -> None:
    collator = _collator()
    sample = _sample("Pick.", 7)
    conditioned = PromptInputs(
        instruction=sample.instruction,
        cameras=sample.cameras,
        condition_text="[outcome|success][generate|actions]",
        state=sample.state,
    )
    with pytest.raises(ValueError, match="no bytes for the bijou condition"):
        collator([conditioned])
    with pytest.raises(ValueError, match="at least one camera"):
        collator(
            [
                PromptInputs(
                    instruction="x",
                    cameras=(),
                    condition_text="",
                    state=sample.state,
                ),
            ],
        )
    with pytest.raises(ValueError, match="setup_type/control_mode"):
        MolmoAct2InputsCollator(
            "unused",
            setup_type=" ",
            control_mode="absolute joint pose",
            num_state_tokens=256,
            action_mode="continuous",
            narration=False,
        )


def _tiny_trunk() -> Molmo2Model:
    config = Molmo2Config.from_dict(
        {
            "model_type": "molmoact2",
            "dtype": "float32",
            "tie_word_embeddings": False,
            "image_patch_id": IM_PATCH_ID,
            "text_config": {
                "model_type": "molmo2_text",
                "vocab_size": 151_936,
                "additional_vocab_size": 4_096,
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
                "image_patch_size": 14,
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
        },
    )
    torch.manual_seed(0)
    assert config.vit is not None and config.adapter is not None
    text = Molmo2TextModel(config.text, lm_head=True)
    vision = Molmo2VisionBackbone(config.vit, config.adapter)
    trunk = Molmo2Model(text, vision, image_patch_id=config.image_patch_id)
    trunk.eval()
    trunk.requires_grad_(False)
    return trunk


def test_encoder_encode_retains_cache_and_conditioning_mask() -> None:
    """The encode product: a filled prefix cache (every layer), the
    conditioning mask threaded onto ObservationMemory, empty streams —
    and no state splice anywhere (this encoder has no parameters)."""
    trunk = _tiny_trunk()
    encoder = MolmoAct2Encoder(
        "unused",
        setup_type="tiny rig",
        control_mode="absolute joint pose",
        num_state_tokens=256,
        action_mode="both",
        narration=False,
    )
    assert len(list(encoder.parameters())) == 0
    collator = _collator(action_mode="both")
    inputs = collator([_sample("Pick up the cube.", 8), _sample("Pick.", 9)])
    memory = encoder.encode(trunk, inputs, with_grad=False, retain_cache=True)
    assert memory.streams == {}
    assert memory.residuals is None
    assert isinstance(memory.cache, Molmo2KVCache)
    assert memory.cache.seen_tokens == inputs.input_ids.shape[1]
    for layer in memory.cache.layers:
        assert layer.keys is not None and layer.values is not None
    assert memory.conditioning_mask is not None
    assert torch.equal(memory.conditioning_mask, inputs.conditioning_mask)
    assert memory.padding_mask is not None  # mixed prompt lengths
    without = encoder.encode(trunk, inputs, with_grad=False, retain_cache=False)
    assert without.cache is None
