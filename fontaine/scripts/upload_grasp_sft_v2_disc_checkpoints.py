"""Upload the 1-GPU discriminator's verdict-evidence saves to fontaine-checkpoints.

Queue item disc-verdict-checkpoint-upload (owner standing rule:
banked/consumable checkpoints leave the host same-session,
weights-only). Steps 500 and 1000 of grasp_sft_v2_demosonly_1gpu_disc
(local H100, unit fontaine-demosonly-1gpu-disc-r2, attempt 2 launched
20:20:55Z 2026-08-17 after the probe-batch OOM fix): these saves are
Amendment 1's disambiguation substrate — the stack-parity probe
(fontaine/scripts/stack_parity_probe.sh) evaluates exactly these
directories on the pre-merge surface, so they are evidence, not just
artifacts. Both jsonls ride along as run records: the fresh attempt-2
log (all eval records) and the preserved attempt-1 OOM log.

Usage: uv run python fontaine/scripts/upload_grasp_sft_v2_disc_checkpoints.py
       (run only after step 1000 writes save-1000; refuses if missing)
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "mcobzarenco/fontaine-checkpoints"
RUN_DIR = Path.home() / "checkpoints/finetune/grasp_sft_v2_demosonly_1gpu_disc"
DEST = "grasp_sft_v2_demosonly_1gpu_disc"
STEPS = ("step_000500", "step_001000")
WEIGHT_FILES = (
    "backbone_text.safetensors",
    "backbone_vision.safetensors",
    "flow_decoder.safetensors",
    "metadata.json",
)


def main() -> int:
    files: list[tuple[Path, str]] = [
        (RUN_DIR / "train_log.jsonl", f"{DEST}/train_log.jsonl"),
        (
            RUN_DIR / "train_log_attempt1_oom250.jsonl",
            f"{DEST}/train_log_attempt1_oom250.jsonl",
        ),
    ]
    for step in STEPS:
        step_dir = RUN_DIR / step
        files += [(step_dir / name, f"{DEST}/{step}/{name}") for name in WEIGHT_FILES]
        tokenizer_dir = step_dir / "tokenizer"
        files += [
            (p, f"{DEST}/{step}/tokenizer/{p.name}")
            for p in sorted(tokenizer_dir.iterdir())
        ]
    for local, _ in files:
        if not local.exists():
            print(f"FATAL: {local} missing", file=sys.stderr)
            return 2
    total = sum(local.stat().st_size for local, _ in files)
    print(
        f"uploading {len(files)} files, {total / 2**30:.1f} GiB total "
        "(optimizer.pt excluded)",
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
