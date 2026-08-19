"""GRPO R2 serving-parity verdict (A5) — the GPU parity read's
mechanical judgment.

The wave-0/relaunch kills showed the loop's serving path can be INERT
while every anchor interacts (root cause: the port-era v30→v21 shim on
v3.0-frame tables; fixed by --joint-frame). This read gates any future
R2 launch: the SAME seeds, greedy, through BOTH serving paths —

  discrete leg: sim.rollout_sim_parallel --molmoact2-discrete
      --molmoact2-grammar-masked --joint-frame rig  (the loop's stack
      + frame map, exactly what waves and in-loop eval serve)
  anchor leg:   sim.rollout_sim --checkpoint --serve-head ar  (the
      BijouPolicy path every R2 anchor and the preflight PASS used)

Rule (registered with A5, spelled once here):

  interacted(seed) := min_cm < initial_cm − 1e-6  OR  final_cm !=
  initial_cm (bit-frozen scenes record exact equality — the kill's
  telemetry). PASS requires BOTH:
    (1) |successes_discrete − successes_anchor| <= 2 of N seeds;
    (2) |interacted_frac_discrete − interacted_frac_anchor| <= 0.30.
  The convicted failure mode reads 0.00 interacted vs the anchor's
  ~0.59 — decades outside the band; the band itself is generous to
  decode-stack noise (bf16 batch-shape reduction order, grammar-masked
  vs reference greedy seam) which moves single seeds, not fractions.
  Anything else is FAIL — the launcher refuses to fire.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUCCESS_TOLERANCE = 2
INTERACTED_TOLERANCE = 0.30


def episode_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("episodes")
    if not rows:
        raise SystemExit("no episodes in the json — not a finished read")
    return rows


def interacted(row: dict[str, Any]) -> bool:
    return (
        row["min_cm"] < row["initial_cm"] - 1e-6 or row["final_cm"] != row["initial_cm"]
    )


def leg_facts(payload: dict[str, Any]) -> dict[str, Any]:
    rows = episode_rows(payload)
    seeds = sorted(row["seed"] for row in rows)
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds in a leg: {seeds}")
    return {
        "seeds": seeds,
        "successes": sum(1 for row in rows if row["success_tick"] is not None),
        "success_seeds": sorted(
            row["seed"] for row in rows if row["success_tick"] is not None
        ),
        "interacted": sum(1 for row in rows if interacted(row)),
        "interacted_frac": sum(1 for row in rows if interacted(row)) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discrete-json", type=Path, required=True)
    parser.add_argument("--anchor-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    discrete_payload = json.loads(args.discrete_json.read_text())
    anchor_payload = json.loads(args.anchor_json.read_text())
    # Provenance guard: the discrete leg must actually be the loop's
    # serving stack under the identity frame — a BijouPolicy json here
    # would compare the anchor path with itself and prove nothing.
    record = (discrete_payload.get("config") or {}).get("molmoact2_discrete")
    if record is None:
        raise SystemExit(
            "--discrete-json carries no molmoact2_discrete record — it is "
            "not a loop-stack read (the parity leg must serve through "
            "sim.rollout_sim_parallel --molmoact2-discrete)",
        )
    if record.get("joint_frame") != "rig":
        raise SystemExit(
            f"discrete leg served joint_frame {record.get('joint_frame')!r}, "
            "the parity gate reads the fixed seam — rerun with "
            "--joint-frame rig",
        )

    discrete = leg_facts(discrete_payload)
    anchor = leg_facts(anchor_payload)
    if discrete["seeds"] != anchor["seeds"]:
        raise SystemExit(
            f"seed sets differ: discrete {discrete['seeds']} vs anchor "
            f"{anchor['seeds']} — the read is paired by construction",
        )

    success_delta = abs(discrete["successes"] - anchor["successes"])
    interacted_delta = abs(
        discrete["interacted_frac"] - anchor["interacted_frac"],
    )
    verdict = (
        "PASS"
        if success_delta <= SUCCESS_TOLERANCE
        and interacted_delta <= INTERACTED_TOLERANCE
        else "FAIL"
    )
    payload = {
        "verdict": verdict,
        "rule": {
            "success_tolerance": SUCCESS_TOLERANCE,
            "interacted_tolerance": INTERACTED_TOLERANCE,
        },
        "discrete": discrete,
        "anchor": anchor,
        "success_delta": success_delta,
        "interacted_delta": round(interacted_delta, 4),
        "discrete_json": str(args.discrete_json),
        "anchor_json": str(args.anchor_json),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(
        f"parity verdict: {verdict} — successes "
        f"{discrete['successes']} vs {anchor['successes']} (|Δ| "
        f"{success_delta} <= {SUCCESS_TOLERANCE}), interacted "
        f"{discrete['interacted_frac']:.2f} vs "
        f"{anchor['interacted_frac']:.2f} (|Δ| {interacted_delta:.2f} <= "
        f"{INTERACTED_TOLERANCE})",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
