"""Freeze the state-reliance probe's panel subset (ideas #11 rung (a)).

Subset rule (pre-registered): every 4th CORE entry of the frozen k4l2
plan — positions ≡ 0 (mod 4) in the plan file's core enumeration —
labeled panel dropped (the probe reads headline MAE only; judge-labeled
oversampling would buy nothing and cost GPU-minutes). A strict
row-subset of the frozen plan: every banked full-panel npz pools
intact-side numbers over exactly these rows, so the masked runs are the
only GPU work.

Oracle (runs on every invocation): the subset is exactly the mod-4
rows, in plan order; every triple exists in the parent core list at its
claimed position; count == ceil(17204 / 4); zero labeled rows; parent
plan is byte-unchanged after the write.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path

PARENT = Path("plans/holdout_curated_v0_k4l2.json")
SUBSET = Path("plans/holdout_curated_v0_k4l2_stateprobe_q4.json")
STRIDE = 4


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    plan = json.loads(parent_bytes)
    core = plan["core"]
    subset_core = core[::STRIDE]

    # Oracle: exact mod-4 positions, order preserved, membership by
    # position (not just value — duplicate triples would alias).
    assert len(subset_core) == math.ceil(len(core) / STRIDE)
    for k, triple in enumerate(subset_core):
        assert core[k * STRIDE] == triple

    subset = {
        **{key: value for key, value in plan.items() if key not in ("core", "labeled")},
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds",
        ),
        "core": subset_core,
        "labeled": [],
    }
    SUBSET.write_text(json.dumps(subset, indent=2))
    assert PARENT.read_bytes() == parent_bytes  # parent untouched

    reloaded = json.loads(SUBSET.read_text())
    assert reloaded["core"] == subset_core and reloaded["labeled"] == []
    digest = hashlib.sha256(SUBSET.read_bytes()).hexdigest()
    print(
        f"wrote {SUBSET}: {len(subset_core)} core rows "
        f"(parent {len(core)}, stride {STRIDE}), 0 labeled\n"
        f"sha256 {digest}",
    )


if __name__ == "__main__":
    main()
