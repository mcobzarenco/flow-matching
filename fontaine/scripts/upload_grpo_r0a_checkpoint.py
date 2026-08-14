"""Upload the R0-A GO-boundary checkpoint to fontaine-checkpoints.

The GO boundary consumed ``step_0002.pt`` (R1-A resumes it), so the
upload rule triggers (owner standing rule: banked/consumable
checkpoints same-session, weights-only unless seeding training —
R1-A seeds from the LOCAL full payload). Option-A payloads are small
(two untied matrices; weights-only ~3.2 GB, no 50 GB cap concern).
Run detached — R1-A's --save-every 1 --keep 2 prunes step_0002.pt
from disk about two steps (~2 h) into the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

REPO = "mcobzarenco/fontaine-checkpoints"
RUN_DIR = Path("outputs/sim/grpo_phase2_a")
DEST = "grpo_phase2_r0a"


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
