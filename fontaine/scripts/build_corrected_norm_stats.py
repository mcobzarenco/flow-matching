"""Build a corrected molmoact2 ``norm_stats.json`` artifact from a
lerobot dataset's (fixed) exact-quantile ``meta/stats.json``.

Why this exists (quantile class bug, 2026-08-15): lerobot's
``aggregate_feature_stats`` merged per-episode quantiles by weighted
MEAN, so any table derived through it carries corrupt q01/q99 rows
(measured on grasp_sft_demos_v0: wrist_roll true action q01/q99
±157° vs the banked [35.5, 94.4]). ``collect_demos.
rewrite_quantile_stats()`` now recomputes every qNN row exactly over
raw frames at finalize — this script projects that corrected table
into the molmoact2 ``norm_stats.json`` convention so
``bijou.convert_molmoact2 --norm-stats-from <out-dir>`` can convert
the released base under the corrected table (their fine-tune recipe's
table-recompute semantics, done right).

The donor artifact (``--source``, default the released base) supplies
the tag's structure and prompt-load-bearing metadata (setup_type,
control_mode, names, mask, geometry fields) verbatim; every numeric
stats row present in BOTH the donor tag and the dataset feature
(min/max/mean/std/count/qNN) is replaced with the dataset's value.
Rows the donor has that the dataset lacks are a hard error — a silent
keep would ship a mixed table.

Usage (real artifact):
  uv run python fontaine/scripts/build_corrected_norm_stats.py \
      --dataset ~/datasets/fontaine/grasp_sft_demos_v0 \
      --out ~/checkpoints/norm_stats_grasp_sft_v0_corrected
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# dataset feature -> norm_stats tag block
FEATURE_TO_BLOCK = {
    "action": "action_stats",
    "observation.state": "state_stats",
}
# rows never replaced: identity/metadata, not statistics
PRESERVED_KEYS = ("names", "mask")


def corrected_tag(
    tag: dict[str, Any],
    dataset_stats: dict[str, Any],
    *,
    tag_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return (deep-copied tag with stats rows replaced, replaced-row
    labels). Refuses on missing features/rows, dim mismatch vs the
    donor's names row, non-finite values, and q01 > q99."""
    out = json.loads(json.dumps(tag))
    replaced: list[str] = []
    for feature, block_name in FEATURE_TO_BLOCK.items():
        if feature not in dataset_stats:
            raise SystemExit(f"dataset stats.json has no {feature!r} feature")
        if block_name not in out:
            raise SystemExit(f"tag {tag_name!r} has no {block_name!r} block")
        feat = dataset_stats[feature]
        block = out[block_name]
        dim = len(block["names"])
        for row in list(block):
            if row in PRESERVED_KEYS:
                continue
            if row not in feat:
                raise SystemExit(
                    f"dataset {feature!r} lacks row {row!r} (donor "
                    f"{block_name} has it) — refusing a mixed table",
                )
            value = feat[row]
            expected = 1 if row == "count" else dim
            if len(value) != expected:
                raise SystemExit(
                    f"{feature}.{row}: dataset dim {len(value)} != "
                    f"expected {expected} (donor names: {block['names']})",
                )
            if not all(math.isfinite(v) for v in value):
                raise SystemExit(f"{feature}.{row}: non-finite value in {value}")
            block[row] = [float(v) for v in value]
            replaced.append(f"{block_name}.{row}")
        for lo, hi in (("q01", "q99"), ("min", "max")):
            if lo in block and hi in block:
                bad = [
                    i
                    for i, (a, b) in enumerate(zip(block[lo], block[hi], strict=True))
                    if a > b
                ]
                if bad:
                    raise SystemExit(
                        f"{block_name}: {lo} > {hi} at dims {bad} — "
                        "corrupt replacement table",
                    )
    return out, replaced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("~/datasets/fontaine/grasp_sft_demos_v0"),
        help="lerobot dataset root (reads meta/stats.json)",
    )
    ap.add_argument(
        "--source",
        default="allenai/MolmoAct2-SO100_101",
        help="donor artifact for norm_stats.json structure/metadata "
        "(HF repo id or local dir)",
    )
    ap.add_argument("--norm-tag", default="so100_so101_molmoact2")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dataset_stats_path = args.dataset.expanduser() / "meta" / "stats.json"
    dataset_stats = json.loads(dataset_stats_path.read_text())

    source_dir = Path(args.source).expanduser()
    if source_dir.is_dir():
        donor_path = source_dir / "norm_stats.json"
    else:
        from huggingface_hub import hf_hub_download

        donor_path = Path(hf_hub_download(args.source, "norm_stats.json"))
    payload = json.loads(donor_path.read_text())
    tags = payload.get("metadata_by_tag")
    if not isinstance(tags, dict) or args.norm_tag not in tags:
        raise SystemExit(
            f"tag {args.norm_tag!r} not in {donor_path} "
            f"(tags: {sorted(tags) if isinstance(tags, dict) else []})",
        )

    tag = tags[args.norm_tag]
    before = {
        b: {r: tag[b][r] for r in ("q01", "q99")}
        for b in ("action_stats", "state_stats")
    }
    tags[args.norm_tag], replaced = corrected_tag(
        tag,
        dataset_stats,
        tag_name=args.norm_tag,
    )
    payload["provenance"] = {
        "built_by": "fontaine/scripts/build_corrected_norm_stats.py",
        "donor": str(args.source),
        "donor_norm_stats_sha256": hashlib.sha256(
            donor_path.read_bytes(),
        ).hexdigest(),
        "dataset_stats": str(dataset_stats_path),
        "dataset_stats_sha256": hashlib.sha256(
            dataset_stats_path.read_bytes(),
        ).hexdigest(),
        "rows_replaced": replaced,
        "context": "quantile class bug fix 2026-08-15 (lerobot "
        "aggregate_feature_stats weighted-mean quantile merge); rows are "
        "exact statistics over the dataset's raw frames",
    }

    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "norm_stats.json"
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    after = tags[args.norm_tag]
    for block in ("action_stats", "state_stats"):
        names = after[block]["names"]
        print(f"[{block}] {len(names)} dims, rows replaced from dataset:")
        for row in ("q01", "q99"):
            print(
                f"  {row}: donor {[round(v, 1) for v in before[block][row]]}"
                f" -> {[round(v, 1) for v in after[block][row]]}",
            )
    print(f"[out] {out_path} ({len(replaced)} rows replaced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
