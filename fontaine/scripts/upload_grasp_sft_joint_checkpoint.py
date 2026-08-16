"""Upload the route-C joint endpoint (step 2000) to fontaine-checkpoints.

Weights-only per the owner standing rule (banked/consumable checkpoints
same-session): backbone.safetensors + flow_decoder.safetensors +
metadata.json, NOT optimizer.pt (33.7 GB of offloaded AdamW moments —
only needed to seed further training, which would restart from the
registered amendment anyway). The train_log.jsonl rides along as the
run record (GRPO upload pattern).

Checkpoint provenance: bijou.train --objective joint --joint-ce-weight
1.0 --insulate-flow, init molmoact2_base_corrected_stats_v0_vla
(corrected norm table baked), 2000 steps gb64, launched 01:09:16Z
2026-08-16 per posts/2026-08-16-amendment-grasp-sft-route-c-joint.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "mcobzarenco/fontaine-checkpoints"
RUN_DIR = Path.home() / "checkpoints/finetune/fontaine_grasp_sft_joint_corrected"
STEP_DIR = RUN_DIR / "step_002000"
DEST = "molmoact2_grasp_sft_joint_corrected_step2000"

FILES = [
    (STEP_DIR / "backbone.safetensors", f"{DEST}/backbone.safetensors"),
    (STEP_DIR / "flow_decoder.safetensors", f"{DEST}/flow_decoder.safetensors"),
    (STEP_DIR / "metadata.json", f"{DEST}/metadata.json"),
    (RUN_DIR / "train_log.jsonl", f"{DEST}/train_log.jsonl"),
]


def main() -> int:
    for local, _ in FILES:
        if not local.exists():
            print(f"FATAL: {local} missing", file=sys.stderr)
            return 2
    total = sum(local.stat().st_size for local, _ in FILES)
    print(
        f"uploading {len(FILES)} files, {total / 2**30:.1f} GiB total (optimizer.pt excluded)",
    )
    api = HfApi()
    for local, remote in FILES:
        print(f"  {local.name} -> {remote} ({local.stat().st_size / 2**30:.2f} GiB)")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO,
            repo_type="model",
        )
    print(f"DONE: https://huggingface.co/{REPO}/tree/main/{DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
