"""Upload the grasp-SFT stage-C endpoint as a weights-only delta to
fontaine-checkpoints (owner standing rule: banked/consumable
checkpoints same-session; standing dedup rule from the rig-r1 upload).

The rig-r1 step2000 upload established the pattern (see the README in
``fontaine-checkpoints/molmoact2_so101_rig_r1_step2000``): the trainer
touches only ``model.action_expert.*`` (AE-only recipe) plus a
load-time vocab resize of ``wte.embedding``/``lm_head.weight``, so the
delta vs the released ``allenai/MolmoAct2-SO100_101`` is ~590 of ~1294
tensors (~5.5 GB fp32) and the trunk dedups against the released repo.
This script recomputes that diff from scratch (per-tensor compare, no
assumption about WHICH tensors differ), refuses on NaN/inf in the
delta, and stages delta + small files + a generated README.

Oracle (run before first real use — the known-good answer):
  uv run python fontaine/scripts/upload_grasp_sft_stagec_delta.py \
      --hf-dir ~/checkpoints/molmoact2-so101-rig-r1-step2000-hf --dry-run
  -> expects 590 differing / 704 identical (the banked rig-r1 counts).

Boundary use (after the stage-D launcher's convert step):
  uv run python fontaine/scripts/upload_grasp_sft_stagec_delta.py \
      --hf-dir ~/checkpoints/molmoact2-grasp-sft-stagec-ar-step3000-hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

BASE_REPO = "allenai/MolmoAct2-SO100_101"
DEST_REPO = "mcobzarenco/fontaine-checkpoints"
SMALL_FILES = (
    "config.json",
    "norm_stats.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def load_tensors(hf_dir: Path) -> dict[str, torch.Tensor]:
    """All tensors from every safetensors shard in the dir (lazy files,
    eager tensors — peak RAM is one checkpoint's worth, ~21 GB fp32)."""
    tensors: dict[str, torch.Tensor] = {}
    shards = sorted(hf_dir.glob("*.safetensors"))
    if not shards:
        msg = f"no safetensors shards under {hf_dir}"
        raise FileNotFoundError(msg)
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safe_open has no __iter__
                tensors[key] = f.get_tensor(key)
    return tensors


def compute_delta(
    ours: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Tensors in `ours` that differ from `base` (missing/reshaped/other
    values). Returns (delta, n_identical)."""
    delta: dict[str, torch.Tensor] = {}
    identical = 0
    for key, tensor in ours.items():
        ref = base.get(key)
        if (
            ref is not None
            and ref.shape == tensor.shape
            and ref.dtype == tensor.dtype
            and torch.equal(ref, tensor)
        ):
            identical += 1
        else:
            delta[key] = tensor
    return delta, identical


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument(
        "--dest",
        default="molmoact2_grasp_sft_stagec_ar_step3000",
        help="folder inside fontaine-checkpoints",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    hf_dir = args.hf_dir.expanduser()

    base_dir = Path(
        snapshot_download(BASE_REPO, allow_patterns=["*.safetensors"]),
    )
    print(f"[delta] loading ours: {hf_dir}")
    ours = load_tensors(hf_dir)
    print(f"[delta] loading base: {base_dir}")
    base = load_tensors(base_dir)
    delta, identical = compute_delta(ours, base)
    n_params = sum(t.numel() for t in delta.values())
    n_bytes = sum(t.numel() * t.element_size() for t in delta.values())
    print(
        f"[delta] {len(delta)} differing / {identical} identical of "
        f"{len(ours)} tensors; {n_params / 1e6:.1f}M params, "
        f"{n_bytes / 2**30:.1f} GiB",
    )
    ae = sum(k.startswith("model.action_expert.") for k in delta)
    other = sorted(k for k in delta if not k.startswith("model.action_expert."))
    print(f"[delta] action_expert tensors: {ae}; other diffs: {other}")
    for key, tensor in delta.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            print(f"FATAL: non-finite values in delta tensor {key}", file=sys.stderr)
            return 2

    if args.dry_run:
        print("[delta] dry run — nothing staged or uploaded")
        return 0

    staging = hf_dir.parent / f"upload_{hf_dir.name}"
    staging.mkdir(exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in delta.items()},
        staging / "model-delta.safetensors",
    )
    staged = ["model-delta.safetensors"]
    for name in SMALL_FILES:
        src = hf_dir / name
        if src.exists():
            (staging / name).write_bytes(src.read_bytes())
            staged.append(name)
    (staging / "README.md").write_text(
        f"# {args.dest}\n\n"
        f"Weights-only DELTA vs the released `{BASE_REPO}` (standing\n"
        f"dedup rule; reconstruct by overlaying `model-delta.safetensors`\n"
        f"on the released shards). {len(delta)} differing tensors\n"
        f"({n_params / 1e6:.1f}M params, {n_bytes / 2**30:.1f} GiB);\n"
        f"{identical} trunk tensors byte-identical to released.\n"
        f"action_expert tensors: {ae}; non-AE diffs: {other}\n"
        f"(vocab-resize rows are load-time re-init, not optimized —\n"
        f"the trainer runs `--ft_embedding=none`).\n\n"
        f"Run: `fontaine_grasp_sft_stagec_ar` (3000 steps, AE-only\n"
        f"5e-5, gb64, 313 demo episodes / 54,101 frames). Pre-reg:\n"
        f"`fontaine/blog/src/posts/2026-08-14-prereg-grasp-sft-bootstrap.md`.\n"
        f"`norm_stats.json` is the DEMO-SET recomputed table under tag\n"
        f"`so100_so101_molmoact2` — the server must use this, never the\n"
        f"released repo's stats.\n",
    )
    print(f"[stage] {staging}: {[*staged, 'README.md']}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(staging),
        repo_id=DEST_REPO,
        repo_type="model",
        path_in_repo=args.dest,
        commit_message=f"{args.dest}: stage-C endpoint delta (weights-only)",
    )
    print(f"[upload] complete -> {DEST_REPO}/{args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
