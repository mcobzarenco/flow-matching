"""Cross-family fixture builders for tests and CPU oracles — the
DAG-top testing home (this module imports the loading schema, so it
lives beside ``train``/``loading``; ``bijou/molmo2/testing.py`` stays
the trunk-only builder underneath).

``write_tiny_molmoact2_release`` fabricates a loadable, TRAINABLE tiny
molmoact2-family artifact pair in the CONVERTED layout — the fixture
behind the objective-matrix CPU loss oracles and the GRPO suite's
subject. See ``probes/generate_tiny_molmoact2.py`` (the per-checkout
CLI wrapper) for the anchor-recording discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from .data import DatasetStats
from .encoders.molmoact2 import MOLMOACT2_PROMPT_FORMAT
from .encoders.molmoact2_processing import BOS_ID, PAD_ID
from .loading import (
    BackboneConfig,
    BackboneDepth,
    CheckpointMetadata,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    build_molmo_flow_decoder,
)
from .molmo2.testing import tiny_config_json, write_tiny_text_checkpoint

TINY_MOLMOACT2_BLOCK_BASE = 151_934  # the release's action_token_start_id
TINY_MOLMOACT2_VOCAB = 154_048  # ≥ block end 153,982, multiple of 64
TINY_MOLMOACT2_T = 30
TINY_MOLMOACT2_D = 6

# Rig-plausible per-dim quantile rows (so normalization is non-trivial).
TINY_MOLMOACT2_Q01 = (-92.3, -104.1, -3.7, -88.0, -45.5, 2.1)
TINY_MOLMOACT2_Q99 = (88.9, 102.6, 178.2, 91.4, 47.0, 97.3)


def tiny_molmoact2_flow_section() -> MolmoFlowDecoderConfig:
    """A tiny expert whose num_layers matches the tiny trunk's 6 blocks
    (layer_kv_pairs pins one conditioning pair per expert block) and
    whose llm_kv_dim matches its 2×16 KV geometry."""
    return MolmoFlowDecoderConfig(
        max_horizon=TINY_MOLMOACT2_T,
        max_action_dim=8,
        hidden_size=32,
        num_layers=6,
        num_heads=2,
        mlp_ratio=2.0,
        ffn_multiple_of=16,
        timestep_embed_dim=16,
        context_layer_norm=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        rope=True,
        causal_attn=False,
        llm_kv_dim=32,
        num_flow_steps=4,
        mask_action_dim_padding=True,
        action_dim=TINY_MOLMOACT2_D,
        action_horizon=TINY_MOLMOACT2_T,
        n_action_steps=TINY_MOLMOACT2_T,
        normalization="q01q99",
        time_offset=0.001,
        time_scale=0.999,
        beta_alpha=1.0,
        beta_beta=1.5,
    )


def _write_tiny_molmoact2_tokenizer(directory: Path) -> None:
    """tokenizer.json written DIRECTLY (the builder API cannot place
    added tokens at arbitrary ids; the file format can — Qwen's own
    extras work this way): a dense filler vocab over the full trunk
    width with the anchor/template tokens at their REAL ids as ADDED
    tokens (added tokens split regardless of pre-tokenization).
    Everything unregistered falls to <unk> (id 0), deterministically."""
    block_base = TINY_MOLMOACT2_BLOCK_BASE
    added = {
        "<|endoftext|>": PAD_ID,
        "<|im_end|>": BOS_ID,
        "<|im_start|>": 151_644,
        "<action_output>": 151_646,
        "<state_start>": 151_647,
        "<state_end>": 151_648,
        "<setup_start>": 151_649,
        "<setup_end>": 151_650,
        "<control_start>": 151_651,
        "<control_end>": 151_652,
        "<im_start>": 151_653,
        "<im_end>": 151_654,
        "<im_patch>": 151_655,
        "<im_col>": 151_656,
        "<action_start>": block_base - 2,
        "<action_end>": block_base - 1,
        "<action_0>": block_base,
    }
    # State bins render as <state_N> text tokens; register the full
    # 256-bin range so prompts tokenize to STABLE ids. BELOW the ChatML
    # ids — a 151_700 anchoring collided with <action_start>
    # (151_700 + 232 == 151_932).
    for bin_index in range(256):
        added[f"<state_{bin_index}>"] = 151_300 + bin_index
    by_id = {token_id: content for content, token_id in added.items()}
    words = ("The", "task", "is", "to", "robot", "state", "control", "mode", "Image")
    vocab: dict[str, int] = {}
    for index in range(TINY_MOLMOACT2_VOCAB):
        if index in by_id:
            vocab[by_id[index]] = index
        elif index == 0:
            vocab["<unk>"] = 0
        elif index <= len(words):
            vocab[words[index - 1]] = index
        else:
            vocab[f"<w{index}>"] = index
    payload = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for content, token_id in sorted(added.items(), key=lambda kv: kv[1])
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>",
        },
    }
    (directory / "tokenizer.json").write_text(json.dumps(payload))
    # Loud round-trip: the pins the collator enforces must hold on the
    # written artifact, not by construction hope.
    check = Tokenizer.from_file(str(directory / "tokenizer.json"))
    assert check.token_to_id("<|im_end|>") == BOS_ID
    assert check.token_to_id("<action_0>") == block_base
    assert check.encode("", add_special_tokens=False).ids == []


def write_tiny_molmoact2_release(root: Path) -> tuple[Path, Path]:
    """(trunk_dir, checkpoint_dir) — a tiny random molmoact2-family
    artifact pair in the CONVERTED layout: the trunk (real-id vocab
    hosting the release action block in-base, ChatML pins at Qwen's
    real ids, ViT speaking the real 588-dim/27×27 patch geometry) and
    a release-class bijou checkpoint (molmo_flow section + quantiled
    stats + fresh expert in converted-export shape). The two dirs are
    SPLIT because the trunk loader globs ``*.safetensors`` — the expert
    file must never sit beside the trunk shards."""
    trunk = root / "trunk"
    checkpoint = root / "checkpoint"
    config = json.loads(json.dumps(tiny_config_json()))
    config["text_config"]["vocab_size"] = TINY_MOLMOACT2_VOCAB
    # The REAL image path runs through this fixture (genuine 378×378
    # resize-mode crops: 27×27 patches of 14²·3 = 588 dims, pooled
    # 14×14 = 196 tokens/image) — the tiny ViT must speak that patch
    # geometry; only its hidden sizes stay tiny.
    config["vit_config"]["image_patch_size"] = 14
    config["vit_config"]["image_num_pos"] = 729  # 27·27 raw (no CLS row)
    config["image_patch_id"] = 151_655  # the tokenizer's <im_patch> row
    torch.manual_seed(0)
    write_tiny_text_checkpoint(trunk, config_json=config)
    _write_tiny_molmoact2_tokenizer(trunk)
    checkpoint.mkdir(parents=True, exist_ok=True)

    section = tiny_molmoact2_flow_section()
    stats = DatasetStats(
        action_mean=(0.0,) * TINY_MOLMOACT2_D,
        action_std=(1.0,) * TINY_MOLMOACT2_D,
        state_mean=(0.0,) * TINY_MOLMOACT2_D,
        state_std=(1.0,) * TINY_MOLMOACT2_D,
        action_q01=TINY_MOLMOACT2_Q01,
        action_q99=TINY_MOLMOACT2_Q99,
        state_q01=TINY_MOLMOACT2_Q01,
        state_q99=TINY_MOLMOACT2_Q99,
    )
    torch.manual_seed(1)
    expert = build_molmo_flow_decoder(section, stats, device="cpu", dtype=torch.float32)
    # Converted-export shape: the construction-frozen compat tensors
    # are OMITTED (their loader convention; load_expert_state injects).
    frozen = {
        name
        for name, parameter in expert.named_parameters()
        if not parameter.requires_grad
    }
    state = {
        name: tensor.contiguous()
        for name, tensor in expert.state_dict().items()
        if name not in frozen
    }
    save_file(state, str(checkpoint / "expert.safetensors"))

    metadata = CheckpointMetadata(
        backbone=BackboneConfig(id=str(trunk), depth=BackboneDepth.FULL),
        prompt=MolmoAct2PromptConfig(
            format=MOLMOACT2_PROMPT_FORMAT,
            norm_tag="tiny",
            setup_type="tabletop",
            control_mode="joint",
            num_state_tokens=256,
            state_dim=TINY_MOLMOACT2_D,
            action_mode="both",
            n_obs_steps=1,
            camera_keys=("top", "wrist"),
            narration=False,
        ),
        decoder=section.to_dict(),
        normalization=stats,
        per_dataset_normalization={},
        train_args={
            "decoder": "molmo_flow",
            "objective": "flow",
            "decoder_hidden": section.hidden_size,
            "decoder_heads": section.num_heads,
            "decoder_intermediate": 64,
            "decoder_cross_heads": section.num_heads,
            "stream_counts": [],
            "self_attention_mode": "bidirectional",
            "chunk_size": TINY_MOLMOACT2_T,
            "max_soft_tokens": 140,
            "max_crops": 1,
            "time_conditioning": "additive",
            "target_time_embed": False,
            "fast_tokenizer": None,
        },
        step=0,
    )
    (checkpoint / "bijou_config.json").write_text(
        json.dumps(metadata.to_json_dict(), indent=2, default=str),
    )
    return trunk, checkpoint
