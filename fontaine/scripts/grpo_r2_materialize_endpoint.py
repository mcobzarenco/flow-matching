"""Materialize a servable checkpoint dir from a GRPO overlay save.

The loop's ``save_checkpoint`` banks only the TRAINABLE tensors
(``step_NNNN.pt``: option-surface names over the stack, ``text.``-
prefixed — the text stack is GBs per save, so full dirs were never
written). But the A3.4 boundary legs read the endpoint through
``sim.rollout_sim`` / ``BijouPolicy`` — the anchors' serving path —
which loads a self-contained VLA dir. This script closes that seam on
CPU: apply the overlay's tensors onto the pinned base's
``backbone_text.safetensors`` and write a first-class checkpoint via
``bijou.checkpoint.write_checkpoint`` (atomic, validated; every
untouched file hard-links the base, so the copy is one part file).

Guards are loud, never silent: the overlay must be text-surface only
(``text.``-prefixed — anything else means a trainable surface this
mapping does not cover), every key must land on an existing base
tensor with the same shape, and the destination must not exist.
Overlay tensors are cast to the base tensor's dtype (the loop trains
the text stack fp32; the dir serves whatever the base serves).

Provenance lands beside the weights as ``grpo_overlay.json`` (base,
overlay path, loop step, replaced-key count) — metadata.json stays the
base's verbatim, so the sidecar is what names the dir a GRPO endpoint.

Usage:
    uv run python -m fontaine.scripts.grpo_r2_materialize_endpoint \
        --base ~/checkpoints/finetune/fontaine_grasp_sft_joint_corrected/step_002000_v2 \
        --overlay outputs/sim/grpo_r2/loop/step_0010.pt \
        --out outputs/sim/grpo_r2/boundary/endpoint_step_0010
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from bijou.checkpoint import (
    BACKBONE_TEXT_FILENAME,
    BACKBONE_VISION_FILENAME,
    TOKENIZER_DIRNAME,
    read_metadata,
    write_checkpoint,
)

TEXT_PREFIX = "text."


def load_overlay(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    """(trainable tensors, loop step) from a loop ``step_NNNN.pt``;
    loud on a payload that is not a loop save."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    trainable = payload.get("trainable")
    if not isinstance(trainable, dict) or not trainable:
        raise SystemExit(
            f"{path}: no non-empty 'trainable' dict — not a GRPO loop save",
        )
    if "step" not in payload:
        raise SystemExit(f"{path}: no 'step' recorded — not a GRPO loop save")
    return trainable, int(payload["step"])


def apply_overlay(
    base_text: dict[str, torch.Tensor],
    trainable: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The base backbone_text state with the overlay applied — every
    overlay key must be ``text.``-prefixed and land shape-exact on an
    existing tensor (cast to ITS dtype); a miss is a wiring bug."""
    off_surface = sorted(k for k in trainable if not k.startswith(TEXT_PREFIX))
    if off_surface:
        raise SystemExit(
            f"overlay keys outside the text surface: {off_surface[:5]} — "
            "the backbone_text mapping does not cover them; this overlay "
            "came from a different trainable surface",
        )
    merged = dict(base_text)
    for name, tensor in trainable.items():
        key = name.removeprefix(TEXT_PREFIX)
        if key not in merged:
            raise SystemExit(
                f"overlay key {name!r} has no tensor {key!r} in the base's "
                f"{BACKBONE_TEXT_FILENAME} — base/overlay mismatch",
            )
        if merged[key].shape != tensor.shape:
            raise SystemExit(
                f"overlay key {name!r}: shape {tuple(tensor.shape)} vs base "
                f"{tuple(merged[key].shape)} — base/overlay mismatch",
            )
        merged[key] = tensor.to(merged[key].dtype).contiguous()
    return merged


def materialize(base: Path, overlay: Path, out: Path) -> dict[str, object]:
    """Write the endpoint dir; returns the provenance record."""
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing {out}")
    metadata = read_metadata(base)  # loud if base is not a VLA dir
    trainable, step = load_overlay(overlay)
    base_text = load_file(str(base / BACKBONE_TEXT_FILENAME))
    merged = apply_overlay(base_text, trainable)
    tokenizer_dir = base / TOKENIZER_DIRNAME
    write_checkpoint(
        out,
        metadata=metadata,
        components={},
        component_files={
            name: base / f"{name}.safetensors"
            for name, record in metadata.components.items()
            if record["weights"]
        },
        backbone_text=merged,
        backbone_vision=base / BACKBONE_VISION_FILENAME,
        tokenizer_files={
            p.name: p for p in sorted(tokenizer_dir.iterdir()) if p.is_file()
        },
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    record: dict[str, object] = {
        "base": str(base),
        "overlay": str(overlay),
        "grpo_step": step,
        "replaced_keys": len(trainable),
        "commit": commit,
    }
    (out / "grpo_overlay.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = materialize(args.base, args.overlay, args.out)
    print(
        f"materialized {args.out}: GRPO step {record['grpo_step']}, "
        f"{record['replaced_keys']} text tensors applied over "
        f"{record['base']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
