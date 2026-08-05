"""Derive the draws-dispersion probe plan from the k4l2 panel plan.

Every 7th core frame (file order, offset 0) of
plans/holdout_curated_v0_k4l2.json, labeled panel empty — the probe
scores no aux metrics, it exists to dump per-draw chunks
(bijou.eval --dump-draws) for the mode-averaging fairness reads at
~1/7th of the full panel's GPU cost. Deterministic: same input file,
same subset, byte-identical output. The parent's metadata (plan_seed,
split, fps, camera_counts, created_at) passes through untouched so the
probe eval's flag cross-checks against the plan still hold; provenance
lives in the file name and the pre-registration amendment.

Frames land in the SAME corpus/selection as the parent plan, so the
probe npz's `index` column joins row-for-row against the banked
AR-100k / flow-80k panel npzs (the paired-analysis join convention).
"""

import json
from pathlib import Path

PARENT = Path("plans/holdout_curated_v0_k4l2.json")
OUT = Path("plans/holdout_curated_v0_k4l2_drawsprobe_s7.json")
STRIDE = 7


def main() -> None:
    plan = json.loads(PARENT.read_text())
    core = plan["core"]
    subset = core[::STRIDE]
    plan["core"] = subset
    plan["labeled"] = []
    OUT.write_text(json.dumps(plan, indent=1) + "\n")

    repos = {frame[0] for frame in subset}
    episodes = {(frame[0], frame[1]) for frame in subset}
    parent_episodes = {(frame[0], frame[1]) for frame in core}
    print(f"parent core {len(core)} frames -> probe {len(subset)} (stride {STRIDE})")
    print(
        f"coverage: {len(repos)} repos, {len(episodes)}/{len(parent_episodes)} "
        "parent episodes",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
