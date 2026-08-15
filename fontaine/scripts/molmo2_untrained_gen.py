"""Untrained-generation probe (owner ask 2026-08-06 18:18Z): what does the
RAW Molmo2-4B checkpoint generate on our exact training-formatted prompt?

Builds one real rig frame through the REAL pipeline — StatsAttachedDataset
item -> Collator (dropout-0, judged camera kinds) -> Molmo2InputsCollator
(WP3, max_crops 1) -> Molmo2Model (WP4) — and greedy-decodes the raw
checkpoint from three starting points:

  1. the exact training prompt, deployment fast path ``[generate|actions]``
     (the AR arm's suffix continuation position),
  2. the exact training prompt with the full aux request
     ``[generate|subgoal holding progress event visible actions]``,
  3. variant 1 plus the natural ChatML assistant opener
     (``<|im_start|>assistant\n``) — the "what does the trunk want to say
     about the scene" read (the form where gemma4 gave refusals).

The prompt's soft state token stays the PAD placeholder: the untrained
trunk has no state projection — that is part of the "untrained" condition
being probed. Doubles as the end-to-end prompt-path test: collator output
feeding the composed model with the real checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.data import EpisodeSplit, select_datasets
from bijou.modelling.aux_text import AuxField
from bijou.modelling.encoders.molmo2 import (
    IM_END_TEXT_ID,
    IM_START_TEXT_ID,
    PAD_ID,
    Molmo2InputsCollator,
)
from bijou.modelling.gemma4.loading import resolve_checkpoint_dir
from bijou.modelling.interface import Collator
from bijou.modelling.molmo2.model import load_model
from bijou.modelling.molmo2.processor import IMAGE_TYPE_IDS

FULL_REQUEST = (
    AuxField.SUBGOAL,
    AuxField.HOLDING,
    AuxField.PROGRESS,
    AuxField.EVENT,
    AuxField.VISIBLE,
)


def build_collator(
    checkpoint: str,
    max_crops: int,
    override: tuple[AuxField, ...],
) -> Collator:
    return Collator(
        inputs=Molmo2InputsCollator(checkpoint, max_crops),
        instruction=None,
        camera_filter=None,
        max_cameras=None,
        action_codec=None,
        aux=None,
        generate_bracket=True,
        generate_override=override,
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        condition_fields=(),
        condition_dropout=0.0,
        subgoal_condition_dropout=0.0,
        state_dropout=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="allenai/Molmo2-4B")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2"),
    )
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(Path(checkpoint_dir) / "tokenizer.json"))

    selection = select_datasets(
        (args.dataset,),
        (),
        chunk_size=50,
        episode_split=EpisodeSplit.ALL,
    )
    item = selection.datasets[0][args.frame]
    print(
        f"frame {args.frame}: task={item['task']!r} "
        f"kinds={item.get('camera_kinds')} state={item['observation.state']}",
    )

    print(f"loading {args.checkpoint} (bf16) on {args.device} ...", flush=True)
    model = load_model(
        args.checkpoint,
        device=args.device,
        dtype=torch.bfloat16,
    )

    variants: list[tuple[str, tuple[AuxField, ...], bool]] = [
        ("fast-path [generate|actions], raw continuation", (), False),
        ("full aux request, raw continuation", FULL_REQUEST, False),
        ("fast-path + <|im_start|>assistant opener", (), True),
    ]
    stop_ids = frozenset({IM_END_TEXT_ID, PAD_ID})
    for title, override, assistant_opener in variants:
        collator = build_collator(str(checkpoint_dir), args.max_crops, override)
        batch = collator([item])
        inputs = batch.encoder_inputs
        if assistant_opener:
            opener = [
                IM_START_TEXT_ID,
                *tokenizer.encode("assistant\n", add_special_tokens=False).ids,
            ]
            opener_ids = torch.tensor([opener], dtype=torch.long)
            inputs = dataclasses.replace(
                inputs,
                input_ids=torch.cat([inputs.input_ids, opener_ids], dim=1),
                attention_mask=torch.cat(
                    [inputs.attention_mask, torch.ones_like(opener_ids)],
                    dim=1,
                ),
                image_type_mask=torch.cat(
                    [
                        inputs.image_type_mask,
                        torch.zeros_like(opener_ids, dtype=torch.bool),
                    ],
                    dim=1,
                ),
            )
        inputs = inputs.to(args.device)

        prompt_ids = inputs.input_ids[0].tolist()
        text_ids = [i for i in prompt_ids if i not in IMAGE_TYPE_IDS]
        num_image = len(prompt_ids) - len(text_ids)
        emitted = model.greedy_generate(
            inputs.input_ids,
            crops=inputs.crops,
            pooled_patches_idx=inputs.pooled_patches_idx,
            image_type_mask=inputs.image_type_mask,
            attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_new_tokens,
            stop_ids=stop_ids,
        )[0]
        print(f"\n=== {title} ===")
        print(
            f"prompt: {len(prompt_ids)} ids ({num_image} image-typed); "
            f"text part: {tokenizer.decode(text_ids, skip_special_tokens=False)!r}",
        )
        print(f"generated ({len(emitted)} ids): ")
        print(repr(tokenizer.decode(emitted, skip_special_tokens=False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
