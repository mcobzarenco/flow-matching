"""Upload the R1-B tripwire-boundary checkpoint to fontaine-checkpoints.

R1-B self-stopped at step 7 (exit 3, knock-away tripwire re-fired under
the v2 reward, 2026-08-14 12:40:50Z) before saving the step-7 update —
``step_0006.pt`` is the banked endpoint state (the R1-A pattern). Upload
rule triggers (owner standing rule: banked/consumable checkpoints
same-session, weights-only). The final train.jsonl (incl. the tripwire
row) and meta.json ride along as the run record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

REPO = "mcobzarenco/fontaine-checkpoints"
RUN_DIR = Path("outputs/sim/grpo_phase2_b")
DEST = "grpo_phase2_r1b"


def main() -> int:
    source = RUN_DIR / "step_0006.pt"
    if not source.exists():
        print(f"FATAL: {source} missing", file=sys.stderr)
        return 2
    payload = torch.load(source, map_location="cpu", weights_only=False)
    weights_path = RUN_DIR / "step_0006_weights.pt"
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
        (weights_path, f"{DEST}/step_0006_weights.pt"),
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
