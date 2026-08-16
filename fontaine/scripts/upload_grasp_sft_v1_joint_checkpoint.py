"""Upload the grasp_sft_v1_joint endpoint (step 3000) to fontaine-checkpoints.

Runs ON the A100 box (the checkpoint's home; HF token verified there).
Weights-only per the owner standing rule: backbone_text +
backbone_vision + flow_decoder safetensors, metadata.json and the
tokenizer dir, NOT optimizer.pt (33.7 GB of AdamW moments — further
training would restart from a fresh registration anyway). The run's
train_log.jsonl rides along as the run record (GRPO upload pattern).

Checkpoint provenance: bijou.train --objective joint --joint-ce-weight
1.0 --insulate-flow, --init-from the owner's re-converted
molmoact2-so101-released (remap-stats v21-to-v30), 3 datasets / 4551
train episodes / 1.49M frames, 3000 steps eff-96 on 8xA100; run 2 —
--recompute-stats restart launched 21:14:48Z 2026-08-16 (unit
grasp-sft-v1c, owner order 20:51Z; run 1b killed ~1900, saves archived
_run1_remaponly).
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "mcobzarenco/fontaine-checkpoints"
RUN_DIR = Path.home() / "checkpoints/finetune/grasp_sft_v1_joint_8xa100"
STEP_DIR = RUN_DIR / "step_003000"
DEST = "grasp_sft_v1_joint_step3000"

FILES = [
    (STEP_DIR / "backbone_text.safetensors", f"{DEST}/backbone_text.safetensors"),
    (STEP_DIR / "backbone_vision.safetensors", f"{DEST}/backbone_vision.safetensors"),
    (STEP_DIR / "flow_decoder.safetensors", f"{DEST}/flow_decoder.safetensors"),
    (STEP_DIR / "metadata.json", f"{DEST}/metadata.json"),
    (RUN_DIR / "train_log.jsonl", f"{DEST}/train_log.jsonl"),
]


def main() -> int:
    tokenizer_dir = STEP_DIR / "tokenizer"
    files = list(FILES) + [
        (p, f"{DEST}/tokenizer/{p.name}") for p in sorted(tokenizer_dir.iterdir())
    ]
    for local, _ in files:
        if not local.exists():
            print(f"FATAL: {local} missing", file=sys.stderr)
            return 2
    total = sum(local.stat().st_size for local, _ in files)
    print(
        f"uploading {len(files)} files, {total / 2**30:.1f} GiB total (optimizer.pt excluded)",
    )
    api = HfApi()
    for local, remote in files:
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
