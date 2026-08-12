"""Registered v3 wrist guard (sim-content-diversity pre-reg,
2026-08-12): for the same (seed, appearance_seed), the v3 wrist frame
must be BIT-IDENTICAL to the v2 wrist frame — v3's content draws may
only touch the top composite. Exits nonzero on any mismatch.

Usage: PYTHONPATH=. MUJOCO_GL=egl uv run python \
    fontaine/scripts/sim_v3_wrist_guard.py
"""

from __future__ import annotations

import sys

import numpy as np

from sim.so101_sim import SO101Sim

PAIRS = ((0, None), (1, None), (2, 17), (5, 999), (13, 1013))


def main() -> int:
    v2 = SO101Sim(render_style="v2")
    v3 = SO101Sim(render_style="v3")
    for seed, appearance in PAIRS:
        kwargs = {} if appearance is None else {"appearance_seed": appearance}
        obs2 = v2.reset(seed, **kwargs)
        obs3 = v3.reset(seed, **kwargs)
        if not np.array_equal(obs2.wrist, obs3.wrist):
            diff = int(np.sum(obs2.wrist != obs3.wrist))
            print(f"FAIL seed={seed} appearance={appearance}: {diff} px differ")
            return 1
        same_top = bool(np.array_equal(obs2.top, obs3.top))
        print(
            f"ok seed={seed} appearance={appearance}: wrist bit-identical"
            f" (top identical: {same_top} — False expected once plate/"
            f"clutter draws differ from the global-plate scene)",
        )
    print("wrist guard GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
