"""Hermetic gemma-flow fixtures shared by the family/train suites: a
tiny loadable Gemma4 trunk, a fabricated legacy (format-3)
``bijou_config.json`` checkpoint for ``bijou.convert_legacy``, and
hand-built ``GemmaInputs`` batches. Moved verbatim from the retired
phase-4 parity suite — the converted directory is exactly the artifact
class the family loaders and the train CLI consume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch import Tensor

from bijou.loading import (
    PROMPT_FORMAT,
    FlowDecoderSection,
    GemmaPromptConfig,
    expert_config_from_architecture,
)
from bijou.modelling.codecs import FastActionCodec
from bijou.modelling.decoders.flow import (
    FlowDecoder,
    SelfAttentionMode,
    TimeConditioning,
)
from bijou.modelling.encoders.gemma4 import GemmaInputs
from bijou.modelling.gemma4.config import Gemma4Config
from bijou.modelling.gemma4.loading import load_config
from bijou.modelling.gemma4.model import Gemma4Model
from bijou.modelling.gemma4.testing import tiny_config_json as gemma_tiny_config_json
from bijou.modelling.interface import CollatedBatch, NormStats

TINY_FAST_FIXTURE = Path(__file__).parent / "fixtures" / "tiny_fast_tokenizer"

BATCH = 2
GEMMA_VOCAB = 2048
GEMMA_DIM = 6
GEMMA_CHUNK = 10
GEMMA_PROMPT_LEN = 12


def write_gemma_trunk(directory: Path) -> Path:
    """A loadable tiny Gemma4 trunk: gemma4.testing's architecture at a
    2048 vocabulary (batches use ids < 1000; the real 262k vocab exists
    only to match the real tokenizer, which these tests never run),
    plus a WordLevel tokenizer so AutoTokenizer resolves hermetically
    (the ar_backbone construction path tokenizes its opener)."""
    config_json = gemma_tiny_config_json()
    config_json["text_config"]["vocab_size"] = GEMMA_VOCAB
    config_json["text_config"]["vocab_size_per_layer_input"] = GEMMA_VOCAB
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config_json))
    torch.manual_seed(0)
    model = Gemma4Model(Gemma4Config.from_dict(config_json), device="cpu")
    state = {
        f"model.{name}": tensor.contiguous()
        for name, tensor in model.state_dict().items()
        # lm_head ties to the embedding at load, like the released
        # checkpoints.
        if name != "lm_head.weight"
    }
    save_file(state, str(directory / "model.safetensors"))
    vocab = {"<unk>": 0, "<pad>": 1, "<start_of_turn>": 3, "model": 4, "hello": 5}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
    )
    return directory


def gemma_prompt_config() -> GemmaPromptConfig:
    # Exports = the tiny prefix's global layers (1, 3, 5).
    return GemmaPromptConfig(
        exports=(1, 3, 5),
        max_soft_tokens=8,
        format=PROMPT_FORMAT,
        state_dim=GEMMA_DIM,
        condition_fields=(),
        generate_bracket=False,
    )


def gemma_flow_section() -> FlowDecoderSection:
    return FlowDecoderSection(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_mode=SelfAttentionMode.CAUSAL_ACTIONS,
        self_attention_rope_theta=10_000.0,
        cross_attention_heads=2,
        schedule=("kv1", "kv3", "kv5"),
        action_dim=GEMMA_DIM,
        state_dim=GEMMA_DIM,
        chunk_size=GEMMA_CHUNK,
        time_embed_dim=16,
        time_conditioning=TimeConditioning.ADDITIVE,
    )


def gemma_stats_dict() -> dict[str, Any]:
    return {
        "action": {
            "mean": [0.0] * GEMMA_DIM,
            "std": [1.0] * GEMMA_DIM,
            "q01": [-1.0] * GEMMA_DIM,
            "q99": [1.0] * GEMMA_DIM,
        },
        "observation.state": {
            "mean": [0.0] * GEMMA_DIM,
            "std": [1.0] * GEMMA_DIM,
        },
    }


def gemma_train_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "decoder": "flow",
        "decoder_hidden": 32,
        "decoder_heads": 2,
        "decoder_intermediate": 64,
        "decoder_cross_heads": 2,
        "stream_counts": [1, 1, 1],
        "self_attention_mode": "causal_actions",
        "chunk_size": GEMMA_CHUNK,
        "max_soft_tokens": 8,
        "max_crops": 1,
        "time_conditioning": "additive",
        "target_time_embed": False,
        "fast_tokenizer": None,
        "narration_weight": 1.0,
        "seed": 0,
    }
    args.update(overrides)
    return args


def prompt_state(*, seed: int) -> dict[str, Tensor]:
    """Non-zero state_proj weights so the soft state token genuinely
    conditions the memory (zero-init would make state effects vacuous)."""
    generator = torch.Generator().manual_seed(seed)
    hidden = 64  # the tiny trunk's hidden size
    return {
        "state_proj.weight": torch.randn(hidden, GEMMA_DIM, generator=generator) * 0.05,
        "state_proj.bias": torch.randn(hidden, generator=generator) * 0.05,
    }


def write_gemma_flow_legacy(directory: Path, trunk: Path) -> Path:
    directory.mkdir(parents=True)
    expert_config = expert_config_from_architecture(
        gemma_prompt_config(),
        gemma_flow_section(),
        load_config(trunk),
    )
    torch.manual_seed(3)
    decoder = FlowDecoder(expert_config, device="cpu", dtype=torch.float32)
    save_file(
        {k: v.contiguous() for k, v in decoder.state_dict().items()},
        str(directory / "expert.safetensors"),
    )
    save_file(prompt_state(seed=4), str(directory / "prompt.safetensors"))
    config = {
        "format": 3,
        "backbone": {"id": str(trunk), "depth": "prefix"},
        "prompt": gemma_prompt_config().to_dict(),
        "decoder": gemma_flow_section().to_dict(),
        "step": 3,
        "train_args": gemma_train_args(),
        "normalization": gemma_stats_dict(),
        "per_dataset_normalization": {},
    }
    (directory / "bijou_config.json").write_text(json.dumps(config))
    return directory


def gemma_batch(
    seed: int,
    *,
    chunk_size: int,
    with_tokens: bool,
) -> CollatedBatch[GemmaInputs]:
    """A hand-built Gemma batch: text-only prompt ids (no images — the
    encoder passes pixel_values=None through), the soft state slot just
    inside the sequence end, mean-0/std-1 stats (state and CollatedBatch
    carry the same values), and FAST action tokens when the AR family
    needs them."""
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(BATCH, GEMMA_DIM, generator=generator)
    inputs = GemmaInputs(
        input_ids=torch.randint(
            3,
            1000,
            (BATCH, GEMMA_PROMPT_LEN),
            generator=generator,
        ),
        attention_mask=torch.ones(BATCH, GEMMA_PROMPT_LEN, dtype=torch.long),
        pixel_values=None,  # pyright: ignore[reportArgumentType] — text-only fixture batch; encode passes None through
        image_position_ids=None,  # pyright: ignore[reportArgumentType] — text-only fixture batch
        state=state,
        state_slot=-2,
        has_padding=False,
    )
    actions = torch.cumsum(
        torch.randn(BATCH, chunk_size, GEMMA_DIM, generator=generator) * 0.05,
        dim=1,
    ).clamp(-1, 1)
    action_tokens: Tensor | None = None
    if with_tokens:
        codec = FastActionCodec.load(TINY_FAST_FIXTURE)
        bounds = np.full(GEMMA_DIM, 1.0)
        sequences = [
            codec.encode(actions[row].numpy(), -bounds, bounds) for row in range(BATCH)
        ]
        width = max(len(sequence) for sequence in sequences)
        action_tokens = torch.tensor(
            [
                sequence + [codec.pad] * (width - len(sequence))
                for sequence in sequences
            ],
            dtype=torch.long,
        )
    stats = NormStats(
        mean=torch.zeros(BATCH, GEMMA_DIM),
        std=torch.ones(BATCH, GEMMA_DIM),
        q01=torch.full((BATCH, GEMMA_DIM), -1.0),
        q99=torch.full((BATCH, GEMMA_DIM), 1.0),
    )
    return CollatedBatch(
        encoder_inputs=inputs,
        state=state,
        actions=actions,
        action_is_pad=torch.zeros(BATCH, chunk_size, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=action_tokens,
        suffix_tokens=None,
        suffix_is_aux=None,
    )
