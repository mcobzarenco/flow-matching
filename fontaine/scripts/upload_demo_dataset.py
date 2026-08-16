"""Upload a merged demo dataset (LeRobot v3 layout) to the HF hub as a
dataset repo, with a generated card (queue item ``demo-gen-sharded-a100``).

Same huggingface_hub route as the checkpoint uploads (HfApi;
``mcobzarenco/*`` is the token's namespace — the checkpoints precedent
is ``mcobzarenco/fontaine-checkpoints``). ``upload_large_folder``
handles the multi-GB video payload with resumable, parallel commits.

``--dry-run`` proves the path without writing: auth whoami, file
census + size, card render to stdout.

Usage (on whichever box holds the merged dataset):
  uv run python fontaine/scripts/upload_demo_dataset.py \
      --root ~/datasets/fontaine/grasp_demos_v1/merged \
      --repo mcobzarenco/fontaine-grasp-demos-v1 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CARD_TEMPLATE = """\
---
license: mit
task_categories:
- robotics
tags:
- lerobot
- so101
- simulation
- imitation-learning
---

# {repo}

Scripted-expert pick-and-place demonstrations from the fontaine SO-101
MuJoCo sim ({substrate}); successes only. LeRobot v3 format:
`action` / `observation.state` are float32[6] absolute joint targets /
positions in degrees (rig motor order), cameras `front` (sim top) and
`wrist` at 480x640 @ 30 fps.

- **Episodes**: {kept} kept of {attempted} attempted (expert success
  rate {rate:.0%}); {frames} frames.
- **Spawn protocol**: {spawn_version} — disk uniform over the measured
  977-cell workspace mask, boat uniform in the annulus around it,
  full-range yaw, upright (pre-reg
  posts/2026-08-16-prereg-sim-spawn-v2.md, finalized 2026-08-16).
- **Boat tint**: {tint_band} (70% rig-gray band / 30% wide-hue when
  `mix70`).
- **Success definition**: {success_definition}.
- **Expert / sim code**: flow-matching @ `{expert_head}`.
- **Kept-seed list + per-shard provenance**: `meta/demo_provenance.json`.

Generated sharded on 8xA100 (`sim.collect_demos_sharded`), merged with
`sim.merge_demo_shards` (merge oracle: bit-identical to a single-process
run over the same seeds). `meta/stats.json` quantile rows are exact
all-frame quantiles (lerobot's count-weighted-mean quantile aggregation
is rewritten away — see `sim/collect_demos.py:rewrite_quantile_stats`).
"""


def render_card(root: Path, repo: str) -> str:
    provenance = json.loads((root / "meta" / "demo_provenance.json").read_text())
    info = json.loads((root / "meta" / "info.json").read_text())
    return CARD_TEMPLATE.format(
        repo=repo,
        kept=provenance["kept"],
        attempted=provenance["attempted"],
        rate=provenance["kept"] / max(provenance["attempted"], 1),
        frames=info["total_frames"],
        spawn_version=provenance.get("spawn_version") or "v1 (fixed disk, band)",
        tint_band=provenance.get("tint_band") or "rig_gray",
        success_definition=provenance["success_definition"],
        expert_head=provenance["expert_head"],
        substrate=provenance["substrate"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser()

    for required in ("meta/info.json", "meta/demo_provenance.json", "data"):
        if not (root / required).exists():
            print(
                f"FATAL: {root / required} missing — not a merged dataset",
                file=sys.stderr,
            )
            return 2
    files = [p for p in root.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"[upload] {root}: {len(files)} files, {total / 2**30:.2f} GiB")

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"[upload] authenticated as {api.whoami()['name']}")
    card = render_card(root, args.repo)
    if args.dry_run:
        print("[upload] DRY RUN — card follows, nothing written:\n")
        print(card)
        return 0

    (root / "README.md").write_text(card)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(root),
    )
    print(f"DONE: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
