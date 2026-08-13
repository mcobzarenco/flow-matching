"""Upload the R0 rung-boundary checkpoint to fontaine-checkpoints.

Weights-only by necessity: the full ``step_0002.pt`` payload
(trainable fp32 + 2 Adam states) is ~53 GB, over the HF Hub 50 GB
per-file cap — and R1 seeds from the LOCAL full payload, so the
preservation copy only needs the trained weights. The heartbeat
jsonl + run meta ride along (tiny, they contextualize the weights).
Run detached (run_detached.sh) — R1's step-4 save prunes step_0002
from disk ~90 min into R1, so this must land first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

REPO = "mcobzarenco/fontaine-checkpoints"
RUN_DIR = Path("outputs/sim/grpo_phase2")
DEST = "grpo_phase2_r0"


def main() -> int:
    source = RUN_DIR / "step_0002.pt"
    if not source.exists():
        print(f"FATAL: {source} missing", file=sys.stderr)
        return 2
    payload = torch.load(source, map_location="cpu", weights_only=False)
    weights_path = RUN_DIR / "step_0002_weights.pt"
    torch.save(
        {
            "step": payload["step"],
            "trainable": payload["trainable"],
            "baseline": payload["baseline"],
        },
        weights_path,
    )
    print(f"weights-only payload: {weights_path.stat().st_size / 2**30:.1f} GiB")
    api = HfApi()
    for local, remote in [
        (weights_path, f"{DEST}/step_0002_weights.pt"),
        (RUN_DIR / "train.jsonl", f"{DEST}/train.jsonl"),
        (RUN_DIR / "meta.json", f"{DEST}/meta.json"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO,
            repo_type="model",
        )
        print(f"uploaded {remote}")
    weights_path.unlink()
    print("upload complete, local weights-only staging removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
