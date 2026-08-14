"""Generate outputs/tiny-molmoact2 — a loadable, TRAINABLE tiny
molmoact2-family checkpoint for the phase-3 CPU loss oracles
(docs/molmoact2-retirement.md: "2-step corridor for ar and joint on
the tiny fixture recorded as new anchors").

Per-checkout artifact like outputs/tiny-gemma4 (regenerating it
re-baselines the molmoact2 oracles — loudly, in the ledger). One
self-contained directory, exactly the converted-checkpoint layout:

- config.json + model.safetensors — a tiny random Molmo2 trunk whose
  base vocabulary spans the REAL id layout (154,048 rows ≥ the release
  action block [151934, 153982); ChatML pins at Qwen's real ids), so
  the real collator pins hold and the real block arithmetic runs;
- tokenizer.json — WordLevel over the anchor tokens as ADDED tokens
  (added tokens split regardless of pre-tokenization, which is what
  makes template words tokenizable; everything unregistered falls to
  <unk> deterministically — an oracle wants determinism, not fidelity);
- bijou_config.json — format 3: molmoact2 prompt section + a tiny
  molmo_flow decoder section + a stats table with quantiles;
- expert.safetensors — the tiny expert's fresh state (converted-export
  shape: compat tensors omitted, load_expert_state injects).

Run: uv run python probes/generate_tiny_molmoact2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from bijou.data import DatasetStats
from bijou.encoders.molmoact2 import MOLMOACT2_PROMPT_FORMAT
from bijou.encoders.molmoact2_processing import BOS_ID, PAD_ID
from bijou.loading import (
    BackboneConfig,
    BackboneDepth,
    CheckpointMetadata,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    build_molmo_flow_decoder,
)
from bijou.molmo2.testing import tiny_config_json, write_tiny_text_checkpoint

OUTPUT = Path("outputs/tiny-molmoact2")
# The converted-checkpoint layout is TWO directories: the checkpoint
# (bijou_config.json + expert.safetensors) records the TRUNK dir as its
# backbone ref — the trunk loader globs *.safetensors in its dir, so
# the expert file must never sit beside the trunk shards.
TRUNK = OUTPUT / "trunk"
CHECKPOINT = OUTPUT / "checkpoint"
BLOCK_BASE = 151_934  # the release's action_token_start_id, verbatim
VOCAB_SIZE = 154_048  # ≥ block end 153,982, multiple of 64
T, D = 30, 6

# Rig-plausible per-dim quantile rows (so normalization is non-trivial).
Q01 = (-92.3, -104.1, -3.7, -88.0, -45.5, 2.1)
Q99 = (88.9, 102.6, 178.2, 91.4, 47.0, 97.3)


def tiny_flow_section() -> MolmoFlowDecoderConfig:
    """num_layers matches the tiny trunk's 6 blocks (one conditioning
    pair per expert block); llm_kv_dim its 2×16 KV geometry."""
    return MolmoFlowDecoderConfig(
        max_horizon=T,
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
        action_dim=D,
        action_horizon=T,
        n_action_steps=T,
        normalization="q01q99",
        time_offset=0.001,
        time_scale=0.999,
        beta_alpha=1.0,
        beta_beta=1.5,
    )


def write_tokenizer(directory: Path) -> None:  # a vocab table
    """tokenizer.json written DIRECTLY (the builder API cannot place
    added tokens at arbitrary ids; the file format can — Qwen's own
    extras work this way): a dense tiny WordLevel word vocab, plus the
    anchor/template tokens as ADDED tokens at their REAL ids (added
    tokens split regardless of pre-tokenization). Everything
    unregistered falls to <unk> (id 0), deterministically."""
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
        "<action_start>": BLOCK_BASE - 2,
        "<action_end>": BLOCK_BASE - 1,
        "<action_0>": BLOCK_BASE,
    }
    # State bins render as <state_N> text tokens; register the full
    # 256-bin range so prompts tokenize to STABLE ids (not unk floods
    # whose count depends on digit clumping). BELOW the ChatML ids —
    # 151_300 + 255 = 151_555 < 151_643 (a 151_700 anchoring collided
    # with <action_start>: 151_700 + 232 == 151_932).
    for bin_index in range(256):
        added[f"<state_{bin_index}>"] = 151_300 + bin_index
    # The model vocab is DENSE over the full trunk width with the
    # specials at their REAL ids and filler everywhere else — added
    # tokens must overlay existing vocab ids (the real Qwen layout;
    # sparse added ids beyond the vocab get silently REINDEXED, the
    # first draft's bug).
    by_id = {token_id: content for content, token_id in added.items()}
    words = ("The", "task", "is", "to", "robot", "state", "control", "mode", "Image")
    vocab: dict[str, int] = {}
    for index in range(VOCAB_SIZE):
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
    assert check.token_to_id("<action_0>") == BLOCK_BASE
    assert check.encode("", add_special_tokens=False).ids == []


def main() -> int:
    config = json.loads(json.dumps(tiny_config_json()))
    config["text_config"]["vocab_size"] = VOCAB_SIZE
    # The REAL image path runs through this fixture (the collator
    # produces genuine 378×378 resize-mode crops: 27×27 patches of
    # 14²·3 = 588 dims, pooled 14×14 = 196 tokens/image), so the tiny
    # ViT must speak that patch geometry — only its hidden sizes stay
    # tiny. (The default tiny vit_config's patch_size-2/9-pos geometry
    # fits the fabricated test crops, not real images.)
    config["vit_config"]["image_patch_size"] = 14
    config["vit_config"]["image_num_pos"] = 729  # 27·27 raw patches (no CLS row)
    config["image_patch_id"] = 151_655  # the tokenizer's <im_patch> row
    torch.manual_seed(0)
    write_tiny_text_checkpoint(TRUNK, config_json=config)
    write_tokenizer(TRUNK)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)

    section = tiny_flow_section()
    stats = DatasetStats(
        action_mean=(0.0,) * D,
        action_std=(1.0,) * D,
        state_mean=(0.0,) * D,
        state_std=(1.0,) * D,
        action_q01=Q01,
        action_q99=Q99,
        state_q01=Q01,
        state_q99=Q99,
    )
    torch.manual_seed(1)
    expert = build_molmo_flow_decoder(section, stats, device="cpu", dtype=torch.float32)
    # Converted-export shape: the construction-frozen compat tensors are
    # OMITTED (their loader convention; load_expert_state injects).
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
    save_file(state, str(CHECKPOINT / "expert.safetensors"))

    metadata = CheckpointMetadata(
        backbone=BackboneConfig(id=str(TRUNK), depth=BackboneDepth.FULL),
        prompt=MolmoAct2PromptConfig(
            format=MOLMOACT2_PROMPT_FORMAT,
            norm_tag="tiny",
            setup_type="tabletop",
            control_mode="joint",
            num_state_tokens=256,
            state_dim=D,
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
            "chunk_size": T,
            "max_soft_tokens": 140,
            "max_crops": 1,
            "time_conditioning": "additive",
            "target_time_embed": False,
            "fast_tokenizer": None,
        },
        step=0,
    )
    (CHECKPOINT / "bijou_config.json").write_text(
        json.dumps(metadata.to_json_dict(), indent=2, default=str),
    )
    print(
        f"written {TRUNK} + {CHECKPOINT} (vocab {VOCAB_SIZE}, block base {BLOCK_BASE})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
