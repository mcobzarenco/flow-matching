"""Materialize the φ_s-extended STEP-0 checkpoint for the SnapFlow drift
gate (2026-08-06 pre-registration, validation gate (b)).

Writes ``<checkpoint>_snapflow_step0`` beside the source: the teacher's
expert weights plus fresh φ_s parameters (zero-initialized output — the
identity oracle proves this is bitwise the teacher), config flagged
``target_time_embed`` so ``bijou.eval`` builds the extended decoder.
The E1-style gate then evals this dir Heun-30 s=t on the stride-7 probe
subset and demands frame-MAE drift < 0.05 vs the banked flow npz.

Backbone/prompt files are hardlinked (bit-identical, no disk cost).

Usage: uv run python fontaine/scripts/materialize_snapflow_init.py \
    [--checkpoint outputs/train/bijou_flow_artrunk_h1024_40k_ddp2/step_080000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.decoders.flow import FlowDecoder
from bijou.gemma4.loading import load_config, resolve_checkpoint_dir
from bijou.loading import (
    FlowDecoderConfig,
    expert_config_from_architecture,
    flow_decoder_config_from_expert,
    parse_decoder_config,
    parse_prompt_config,
)

PHI_S_KEYS = {
    "target_time_in_proj.weight",
    "target_time_in_proj.bias",
    "target_time_out_proj.weight",
    "target_time_out_proj.bias",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/train/bijou_flow_artrunk_h1024_40k_ddp2/step_080000",
        ),
    )
    args = parser.parse_args()
    out_dir = args.checkpoint.parent / f"{args.checkpoint.name}_snapflow_step0"
    if out_dir.exists():
        sys.exit(f"{out_dir} already exists — remove it to re-materialize")

    meta = json.loads((args.checkpoint / "bijou_config.json").read_text())
    decoder_config = parse_decoder_config(meta["decoder"])
    if not isinstance(decoder_config, FlowDecoderConfig):
        sys.exit(f"not a flow checkpoint: {meta['decoder'].get('kind')}")
    if decoder_config.target_time_embed:
        sys.exit("checkpoint is ALREADY φ_s-extended — nothing to do")
    config = expert_config_from_architecture(
        parse_prompt_config(meta["prompt"]),
        decoder_config,
        load_config(resolve_checkpoint_dir(meta["backbone"]["id"])),
    )

    torch.manual_seed(0)  # φ_s in_proj draw; output layer is zeroed
    extended = FlowDecoder(
        dataclasses.replace(config, target_time_embed=True),
        device="cpu",
        dtype=torch.float32,
    )
    state = load_file(str(args.checkpoint / "expert.safetensors"), device="cpu")
    missing, unexpected = extended.load_state_dict(state, strict=False)
    if set(missing) != PHI_S_KEYS or unexpected:
        sys.exit(
            f"key diff beyond φ_s: missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}",
        )

    out_dir.mkdir(parents=True)
    meta["decoder"] = flow_decoder_config_from_expert(extended.config).to_dict()
    (out_dir / "bijou_config.json").write_text(json.dumps(meta, indent=2))
    save_file(
        {k: v.contiguous() for k, v in extended.state_dict().items()},
        str(out_dir / "expert.safetensors"),
    )
    for name in ("prompt.safetensors", "backbone.safetensors"):
        source = args.checkpoint / name
        if source.exists():
            os.link(source, out_dir / name)
    print(f"materialized φ_s-extended step-0 checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
