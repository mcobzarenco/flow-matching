"""Recompute a sealed_v2 pooled anchor from a v1 panel report JSON.

Frame-weighted re-pool of per-dataset means, dropping the removed
repos' (frames x mae) terms. NOT exact: the report's pooled summary
weights by valid chunk elements (horizons truncate near episode ends),
so re-pooling reproduces it only to ~5e-3 (bijou) / ~4e-2 (state-copy)
on the primary panel; the printed repool-check quantifies this per
report. Quote v2 anchors with that method error — it is ~15x smaller
than the v1->v2 shift and negligible vs the 0.15 tolerance band.
See posts/2026-08-05-sealed-plan-v2.md.

Usage: python fontaine/scripts/sealed_v2_anchor.py <report.json>
"""

import json
import sys

REMOVED = [
    "kevin510/lerobot-cat-toy-placement",
    "kevin510/so-100-draw-smiley",
    "willnorris/bbox-2",
]


def main(path: str) -> None:
    r = json.load(open(path))
    pd = r["per_dataset"]
    missing = [d for d in REMOVED if d not in pd]
    if missing:
        sys.exit(f"removed repos absent from report: {missing}")

    policies = list(next(iter(pd.values()))["chunk_mae"])
    total_frames = sum(v["frames"] for v in pd.values())
    print(f"report: {path}")
    print(f"datasets: {len(pd)}, pooled frames: {total_frames}")
    for pol in policies:
        v1_sum = sum(v["frames"] * v["chunk_mae"][pol] for v in pd.values())
        rm_frames = sum(pd[d]["frames"] for d in REMOVED)
        rm_sum = sum(pd[d]["frames"] * pd[d]["chunk_mae"][pol] for d in REMOVED)
        v1_mae = v1_sum / total_frames
        v2_mae = (v1_sum - rm_sum) / (total_frames - rm_frames)
        summary = next(
            (s["chunk_mae"] for s in r["summaries"] if s["policy"] == pol), None
        )
        drift = abs(v1_mae - summary) if summary is not None else float("nan")
        print(
            f"  {pol}: v1 {v1_mae:.4f} (summary {summary:.4f}, "
            f"repool-check {drift:.2e}) -> v2 {v2_mae:.4f} "
            f"on {total_frames - rm_frames} frames (-{rm_frames})"
        )


if __name__ == "__main__":
    main(sys.argv[1])
