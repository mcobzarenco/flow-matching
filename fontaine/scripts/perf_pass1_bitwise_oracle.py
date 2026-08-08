"""Perf pass-1 bitwise oracle (pre-reg 2026-08-08): P3b + P4 gates.

Builds the deterministic tiny Molmo2 fixture (same construction as
tests/test_molmo2_model.py), runs one multimodal forward + backward in
train mode on CPU fp32, and prints sha256 hashes of the logits, the
wte outputs for a no-extension and a with-extension id batch, and
every parameter gradient. Run once with PYTHONPATH=<HEAD checkout>
and once with PYTHONPATH=<perf-pass1 worktree>; the pre-reg requires
EVERY hash to match bitwise (P3b branchless wte, P4 clone drop are
bitwise claims; P3a changes no values). Usage:

    PYTHONPATH=/path/to/checkout python perf_pass1_bitwise_oracle.py out.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

from bijou.molmo2.config import Molmo2Config
from bijou.molmo2.model import load_model
from bijou.molmo2.testing import tiny_config_json, write_tiny_text_checkpoint


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().to(torch.float32).numpy().tobytes(),
    ).hexdigest()


def main() -> int:
    out_path = Path(sys.argv[1])
    config = Molmo2Config.from_dict(tiny_config_json())
    patch = config.image_patch_id
    marker = patch + 1

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = write_tiny_text_checkpoint(Path(tmp) / "tiny-molmo2", seed=0)
        model = load_model(str(ckpt), dtype=torch.float32)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    row0 = [1, marker, patch, patch, patch, patch, marker, 2, 3, 4]
    row1 = [5, patch, patch, marker, 6, 7]
    width = len(row0)
    input_ids = torch.tensor([row0, [0] * (width - len(row1)) + row1])
    attention_mask = torch.tensor(
        [[1] * width, [0] * (width - len(row1)) + [1] * len(row1)],
    )
    image_type_mask = (
        (input_ids == patch) | (input_ids == marker)
    ) & attention_mask.bool()

    vit_config = config.vit
    assert vit_config is not None
    torch.manual_seed(2)
    num_patch_tokens = [4, 2]
    crops = torch.randn(2, 1, vit_config.image_num_pos, vit_config.patch_dim)
    max_tokens = max(num_patch_tokens)
    pooled_idx = torch.full((2, max_tokens, 1), -1, dtype=torch.long)
    for row, count in enumerate(num_patch_tokens):
        for token in range(count):
            pooled_idx[row, token, 0] = token

    logits = model(
        input_ids,
        crops=crops,
        pooled_patches_idx=pooled_idx,
        image_type_mask=image_type_mask,
        attention_mask=attention_mask,
    )
    loss = logits.float().pow(2).sum()
    loss.backward()

    hashes: dict[str, str] = {"logits": sha(logits), "loss": f"{loss.item():.10e}"}
    wte = model.text.transformer.wte
    base_rows = wte.embedding.shape[0]
    plain_ids = torch.arange(0, 8).reshape(2, 4) % (base_rows - 1)
    hashes["wte_no_extension"] = sha(wte(plain_ids))
    mixed_ids = plain_ids.clone()
    mixed_ids[0, 0] = patch
    hashes["wte_with_extension"] = sha(wte(mixed_ids))
    for name, param in sorted(model.named_parameters()):
        if param.grad is not None:
            hashes[f"grad/{name}"] = sha(param.grad)

    out_path.write_text(json.dumps(hashes, indent=1, sort_keys=True))
    print(f"{len(hashes)} hashes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
