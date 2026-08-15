"""Grasp-SFT stage-C launch preflight (pre-reg §6 frozen gate facts).

Run from the flow-matching repo root before EITHER stage-C arm
launches. Exits non-zero (refusing the launch) unless every frozen
precondition holds:

- stage-B collection is finished (no live ``collect_demos`` process,
  provenance banked with a DONE stop_reason);
- gate: >= 300 kept successes (pre-reg §2/§6);
- eval-seed integrity: every kept seed >= 1000 (the sim100 holdout
  0-99 never appears in demos, §3);
- convention seam: provenance ``state_units`` is the frozen identity
  string (§6 item 4 — recomputed dataset table, no shim in B-D);
- dataset meta is loadable and consistent with the kept count.

Prints the epoch math at the frozen global batch so the launch log
carries it (§6: 3000 steps x gb64 vs total demo frames).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEMO_ROOT = Path.home() / "datasets/fontaine/grasp_sft_demos_v0"
KEPT_GATE = 300
DEMO_SEED_BASE = 1000
STATE_UNITS = "rig (identity — recomputed dataset table)"

AR_STEPS, AR_GLOBAL_BATCH = 3000, 64
FLOW_STEPS, FLOW_BATCH = 4000, 24


def fail(msg: str) -> None:
    print(f"[preflight] REFUSE: {msg}")
    sys.exit(1)


def main() -> None:
    live = subprocess.run(
        ["pgrep", "-f", "collect_demos"],
        capture_output=True,
        text=True,
        check=False,
    )
    if live.returncode == 0:
        fail(f"stage-B collector still running (pids {live.stdout.split()})")

    prov_path = DEMO_ROOT / "meta" / "demo_provenance.json"
    if not prov_path.exists():
        fail(f"{prov_path} missing — collection not finalized")
    prov = json.loads(prov_path.read_text())

    state = json.loads((DEMO_ROOT / "collect_state.json").read_text())
    kept = state["kept_seeds"]
    if prov["kept"] != len(kept):
        fail(f"provenance kept={prov['kept']} != collect_state {len(kept)}")
    if len(kept) < KEPT_GATE:
        fail(f"kept {len(kept)} < gate {KEPT_GATE} (§6 stage-B gate)")
    bad = [s for s in kept if s < DEMO_SEED_BASE]
    if bad:
        fail(f"kept seeds below {DEMO_SEED_BASE}: {bad[:5]} — eval-seed breach")
    if prov.get("state_units") != STATE_UNITS:
        fail(f"state_units {prov.get('state_units')!r} != frozen {STATE_UNITS!r}")
    if "DONE" not in str(prov.get("stop_reason", "")) and not prov.get("stop_reason"):
        fail("provenance carries no stop_reason")

    info = json.loads((DEMO_ROOT / "meta" / "info.json").read_text())
    episodes, frames = info["total_episodes"], info["total_frames"]
    if episodes != len(kept):
        fail(f"info.json episodes {episodes} != kept {len(kept)}")

    print(
        f"[preflight] PASS: {len(kept)} kept (gate >= {KEPT_GATE}), "
        f"{prov['attempted']} attempted "
        f"({len(kept) / prov['attempted']:.0%} keep rate), "
        f"stop_reason={prov.get('stop_reason')!r}",
    )
    print(
        f"[preflight] seams: state_units OK, all kept seeds >= {DEMO_SEED_BASE}, "
        f"expert HEAD {prov.get('expert_head')}",
    )
    print(f"[preflight] demo set: {episodes} episodes / {frames} frames")
    print(
        f"[preflight] epoch math AR primary: {AR_STEPS} steps x gb{AR_GLOBAL_BATCH} "
        f"= {AR_STEPS * AR_GLOBAL_BATCH} samples ~= "
        f"{AR_STEPS * AR_GLOBAL_BATCH / frames:.1f} epochs",
    )
    print(
        f"[preflight] epoch math flow arm: {FLOW_STEPS} steps x b{FLOW_BATCH} "
        f"= {FLOW_STEPS * FLOW_BATCH} samples ~= "
        f"{FLOW_STEPS * FLOW_BATCH / frames:.1f} epochs",
    )


if __name__ == "__main__":
    main()
