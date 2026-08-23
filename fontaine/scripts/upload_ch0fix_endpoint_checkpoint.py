"""Upload the ch0fix (ch0-affine clean) endpoint (step 3000) to fontaine-checkpoints.

Weights-only per the owner standing rule: backbone_text +
backbone_vision + flow_decoder safetensors, metadata.json and the
tokenizer dir, NOT optimizer.pt. train_log.jsonl rides along.

Checkpoint provenance: bijou.train --objective joint --joint-ce-weight
1.0 --insulate-flow --per-dataset-flow-norm --recompute-stats; mix =
grasp_demos_v2/merged + so101_pick_place_clean_ch0fix_n (x4) ONLY —
the ch0 (shoulder-pan) distribution isolation cell, carrier-hunt rung
2: clean's ch0 (action AND state) mapped through the frozen
moment-matched affine x' = 0.0923... + (x - 1.4820...) * 2.7552...,
everything else byte-identical; 3000 steps eff-96 local H100, unit
fontaine-v2-joint-pdnorm-ch0fix launched 22:38:31Z 2026-08-22;
pre-reg posts/2026-08-22-prereg-clean-ch0-affine.md. Banked at the
endpoint verdict (frozen-grid sim100 read) per the standing
banked-checkpoints rule — bankable on either decisive branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "mcobzarenco/fontaine-checkpoints"
RUN_DIR = Path.home() / "checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm_ch0fix"
STEP_DIR = RUN_DIR / "step_003000"
DEST = "grasp_sft_v2_joint_pdnorm_ch0fix_step3000"

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
