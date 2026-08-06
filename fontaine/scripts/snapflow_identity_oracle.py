"""SnapFlow pre-launch validation gate (a): zero-init φ_s identity on the
REAL subject checkpoint (2026-08-06 distill pre-registration).

Loads the teacher expert (flow-80k step_080000) twice — unmodified and
φ_s-extended — against a synthetic observation memory (decoder-level: the
identity is a decoder property, no backbone needed) and demands BITWISE
equality of the velocity field on s=t forwards, explicit-s forwards, and
the s=0 one-step forward (φ_s output is exactly zero at init for every
s). Any diff blocks the launch. CPU, ~seconds.

Usage: uv run python fontaine/scripts/snapflow_identity_oracle.py \
    [--checkpoint outputs/train/bijou_flow_artrunk_h1024_40k_ddp2/step_080000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.decoders.flow import FlowDecoder
from bijou.gemma4.loading import load_config, resolve_checkpoint_dir
from bijou.interface import MemoryStream, ObservationMemory
from bijou.loading import (
    FlowDecoderConfig,
    expert_config_from_architecture,
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

    meta = json.loads((args.checkpoint / "bijou_config.json").read_text())
    decoder_config = parse_decoder_config(meta["decoder"])
    if not isinstance(decoder_config, FlowDecoderConfig):
        sys.exit(f"not a flow checkpoint: {meta['decoder'].get('kind')}")
    if decoder_config.target_time_embed:
        sys.exit("checkpoint is ALREADY φ_s-extended — wrong subject")
    prompt_config = parse_prompt_config(meta["prompt"])
    backbone_config = load_config(
        resolve_checkpoint_dir(meta["backbone"]["id"]),
    )
    config = expert_config_from_architecture(
        prompt_config,
        decoder_config,
        backbone_config,
    )

    state = load_file(str(args.checkpoint / "expert.safetensors"), device="cpu")
    teacher = FlowDecoder(config, device="cpu", dtype=torch.float32)
    teacher.load_state_dict(state, strict=True)
    torch.manual_seed(0)  # φ_s in_proj init draw — any seed, output is 0
    extended = FlowDecoder(
        dataclasses.replace(config, target_time_embed=True),
        device="cpu",
        dtype=torch.float32,
    )
    missing, unexpected = extended.load_state_dict(state, strict=False)
    if set(missing) != PHI_S_KEYS or unexpected:
        sys.exit(
            f"FAIL: unexpected key diff — missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}",
        )
    teacher.eval()
    extended.eval()

    generator = torch.Generator().manual_seed(1)
    batch, prefix = 3, 24
    streams = {
        name: MemoryStream(
            key=torch.randn(
                batch,
                1,
                prefix,
                config.cross_attention_head_dim,
                generator=generator,
            ),
            value=torch.randn(
                batch,
                1,
                prefix,
                config.cross_attention_head_dim,
                generator=generator,
            ),
        )
        for name in sorted(set(teacher.schedule_names))
    }
    memory = ObservationMemory(streams=streams, length=prefix, padding_mask=None)
    robot_state = torch.randn(batch, config.state_dim, generator=generator)
    actions = torch.randn(
        batch,
        config.chunk_size,
        config.action_dim,
        generator=generator,
    )

    checks: dict[str, bool] = {}
    with torch.no_grad():
        for label, time in (
            ("tau_mixed", torch.tensor([0.001, 0.5, 1.0])),
            ("tau_one", torch.ones(batch)),
        ):
            reference = teacher(memory, robot_state, actions, time)
            checks[f"{label}_implicit_s"] = torch.equal(
                extended(memory, robot_state, actions, time),
                reference,
            )
            checks[f"{label}_explicit_s=t"] = torch.equal(
                extended(memory, robot_state, actions, time, time),
                reference,
            )
            checks[f"{label}_s=0"] = torch.equal(
                extended(
                    memory,
                    robot_state,
                    actions,
                    time,
                    torch.zeros_like(time),
                ),
                reference,
            )

    for name, ok in checks.items():
        print(f"  {name}: {'BIT-EXACT' if ok else 'DIFF'}")
    if not all(checks.values()):
        sys.exit("VALIDATION GATE (a) FAILED — do not launch")
    print(
        f"GATE (a) PASSED: φ_s-extended {args.checkpoint} is bitwise the "
        f"teacher on {len(checks)}/6 forward checks",
    )


if __name__ == "__main__":
    main()
