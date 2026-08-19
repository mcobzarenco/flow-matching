"""R2 boundary verdict (A3.4's three registered endpoint legs, machine-read).

Consumes the three boundary ``sim.rollout_sim`` out-jsons plus the banked
preflight verdict and emits one machine-readable boundary read, so the
boundary session is reads-not-code:

- **PRIMARY** — boundary greedy sim100 vs the frozen greedy anchor 7/100,
  paired per-seed exact test (seeds 0-99 are paired by construction):
  McNemar-style, the test runs on the discordant pairs only — ``wins`` =
  seeds newly succeeding, ``losses`` = anchor successes lost — with the
  exact one-sided sign-test tail P(X >= wins | Bin(wins+losses, 1/2)) at
  the house 5% level. Three-band surface (the stage-D 5-19 style):
  ``IMPROVED`` (materially above -> accumulation confirmed, pending wire
  review), ``REGRESSED`` (materially below -> F-regression), ``FLAT``
  (neither -> F-flat surface; banked negative if the wires ran quiet).
- **Record-only sibling** — boundary sampled T=1.0 sim100 vs the preflight
  leg-0 floor: prices decode-gap movement (base gap = preflight sampled -
  greedy 7); no verdict is minted from this leg.
- **Flow-head F-regression leg** — boundary flow unseen100 (euler-10) vs
  the frozen flow-sibling anchor 44/100. The material-regression surface
  is stated mechanically — exact one-sided binomial tail
  P(X <= k | Bin(100, 44/100)) < 0.05, i.e. k <= 35 — but the final call
  is judged (A3.4: material regression is F-regression evidence even if
  the token side improves).

``overall_surface`` combines the legs mechanically: F-regression evidence
(primary REGRESSED or flow materially below) outranks accumulation;
IMPROVED with the flow leg clean reads accumulation; anything else F-flat.
Wire cleanliness (the §2 tripwire belt) lives in the run heartbeat, not
these jsons — the field names what stays judged.

Provenance is guarded loudly per leg (serve_head, temperature, flow
method/steps, seed window, one shared boundary checkpoint that must not be
the pinned base): the wrong leg's json must not mint a verdict.

Usage:
    uv run python -m fontaine.scripts.grpo_r2_boundary_verdict \
        --greedy-json  outputs/sim/grpo_r2/boundary/token_greedy_sim100.json \
        --sampled-json outputs/sim/grpo_r2/boundary/sampled_t1_sim100.json \
        --flow-json    outputs/sim/grpo_r2/boundary/flow_unseen100.json \
        --preflight-json outputs/sim/grpo_r2/preflight/preflight_verdict.json \
        --out outputs/sim/grpo_r2/boundary/boundary_verdict.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from fontaine.scripts.grpo_r2_preflight_verdict import binom_tail_below

# Frozen anchors (A3.1): both legs measured on the same trunk, seeds 0-99.
GREEDY_ANCHOR = 7
GREEDY_SUCCESS_SEEDS = frozenset({34, 35, 63, 68, 71, 91, 96})
FLOW_ANCHOR = 44
FLOW_SUCCESS_SEEDS = frozenset(
    {0, 3, 4, 5, 8, 11, 17, 20, 22, 23, 24, 26, 29, 30, 32, 35, 36, 37, 40, 43}
    | {49, 50, 52, 54, 56, 60, 64, 65, 66, 67, 69, 73, 74, 75, 77, 80, 86, 87}
    | {90, 93, 96, 97, 98, 99},
)
NUM_SEEDS = 100
FIRST_SEED = 0
# House "materially" line: exact one-sided tails at the 5% level (the
# preflight verdict's convention, inherited).
MATERIAL_ALPHA = 0.05
# The boundary legs read the GRPO endpoint; a json minted from the pinned
# base is the wrong leg by construction.
BASE_CHECKPOINT_NAMES = frozenset({"step_002000", "step_002000_v2"})


def sign_test_tail(wins: int, discordant: int) -> float:
    """Exact one-sided sign-test tail P(X >= wins | Bin(discordant, 1/2));
    1.0 when there are no discordant pairs (no evidence either way)."""
    if discordant == 0:
        return 1.0
    return float(
        sum(math.comb(discordant, i) for i in range(wins, discordant + 1))
        / 2**discordant,
    )


def _guard_seed_window(config: dict, episodes: list[dict], leg: str) -> None:
    if config.get("seed") != FIRST_SEED or config.get("num_seeds") != NUM_SEEDS:
        raise ValueError(
            f"{leg}: boundary legs run seeds {FIRST_SEED}-"
            f"{FIRST_SEED + NUM_SEEDS - 1}, got seed={config.get('seed')} "
            f"num_seeds={config.get('num_seeds')}",
        )
    seeds = sorted(e["seed"] for e in episodes)
    if seeds != list(range(FIRST_SEED, FIRST_SEED + NUM_SEEDS)):
        raise ValueError(
            f"{leg}: episode seed set is not exactly {FIRST_SEED}-"
            f"{FIRST_SEED + NUM_SEEDS - 1}: {len(episodes)} episodes",
        )


def _success_seeds(payload: dict, leg: str) -> set[int]:
    config, episodes = payload["config"], payload["episodes"]
    _guard_seed_window(config, episodes, leg)
    return {e["seed"] for e in episodes if e.get("success_tick") is not None}


def _guard_greedy(payload: dict) -> None:
    config = payload["config"]
    if config.get("serve_head") != "ar":
        raise ValueError(
            f"primary leg is the AR head (--serve-head ar), got "
            f"serve_head={config.get('serve_head')!r}",
        )
    if config.get("ar_temperature") is not None:
        raise ValueError(
            f"primary leg is GREEDY (no --ar-temperature), got "
            f"ar_temperature={config.get('ar_temperature')!r} — a sampled "
            "json cannot mint the serving-convention read",
        )


def _guard_sampled(payload: dict) -> None:
    config = payload["config"]
    if config.get("serve_head") != "ar":
        raise ValueError(
            f"sampled sibling is the AR head (--serve-head ar), got "
            f"serve_head={config.get('serve_head')!r}",
        )
    if config.get("ar_temperature") != 1.0:
        raise ValueError(
            f"sampled sibling runs T=1.0 (--ar-temperature 1.0), got "
            f"ar_temperature={config.get('ar_temperature')!r}",
        )


def _guard_flow(payload: dict) -> None:
    config = payload["config"]
    if config.get("serve_head") not in (None, "flow"):
        raise ValueError(
            f"flow leg serves the flow head, got "
            f"serve_head={config.get('serve_head')!r}",
        )
    if config.get("method") != "euler" or config.get("sample_steps") != 10:
        raise ValueError(
            f"flow leg is euler-10 (the 44/100 anchor's decode), got "
            f"method={config.get('method')!r} "
            f"sample_steps={config.get('sample_steps')!r}",
        )


def _guard_checkpoints(payloads: dict[str, dict]) -> str:
    checkpoints = {
        leg: str(p["config"].get("checkpoint") or "") for leg, p in payloads.items()
    }
    if len(set(checkpoints.values())) != 1:
        raise ValueError(
            f"the three boundary legs must read ONE checkpoint, got {checkpoints}",
        )
    checkpoint = next(iter(checkpoints.values()))
    if Path(checkpoint.rstrip("/")).name in BASE_CHECKPOINT_NAMES:
        raise ValueError(
            f"checkpoint {checkpoint!r} is the pinned BASE — the boundary "
            "verdict reads the GRPO endpoint, not the anchors' checkpoint",
        )
    return checkpoint


def _guard_preflight(preflight: dict) -> int:
    if preflight.get("verdict") != "PASS":
        raise ValueError(
            f"preflight verdict is {preflight.get('verdict')!r}, not PASS — "
            "the A3.4 run must not have launched; no floor to price against",
        )
    return int(preflight["sampled_successes"])


def boundary_verdict(
    greedy_payload: dict,
    sampled_payload: dict,
    flow_payload: dict,
    preflight: dict,
) -> dict:
    """The three A3.4 boundary reads on the rollout_sim out-jsons; raises
    on any provenance mismatch."""
    _guard_greedy(greedy_payload)
    _guard_sampled(sampled_payload)
    _guard_flow(flow_payload)
    checkpoint = _guard_checkpoints(
        {
            "greedy": greedy_payload,
            "sampled": sampled_payload,
            "flow": flow_payload,
        },
    )
    preflight_floor = _guard_preflight(preflight)

    greedy = _success_seeds(greedy_payload, "primary greedy leg")
    sampled = _success_seeds(sampled_payload, "sampled sibling leg")
    flow = _success_seeds(flow_payload, "flow regression leg")

    # PRIMARY: paired per-seed exact test on the discordant pairs.
    wins = sorted(greedy - GREEDY_SUCCESS_SEEDS)
    losses = sorted(GREEDY_SUCCESS_SEEDS - greedy)
    discordant = len(wins) + len(losses)
    p_improve = sign_test_tail(len(wins), discordant)
    p_regress = sign_test_tail(len(losses), discordant)
    if p_improve < MATERIAL_ALPHA:
        band = "IMPROVED"
    elif p_regress < MATERIAL_ALPHA:
        band = "REGRESSED"
    else:
        band = "FLAT"

    # Flow leg: material-regression surface stated, call judged.
    flow_tail = binom_tail_below(len(flow), NUM_SEEDS, FLOW_ANCHOR / NUM_SEEDS)
    flow_material = flow_tail < MATERIAL_ALPHA

    if band == "REGRESSED" or flow_material:
        overall = "f-regression"
    elif band == "IMPROVED":
        overall = "accumulation"
    else:
        overall = "f-flat"

    base_gap = preflight_floor - GREEDY_ANCHOR
    boundary_gap = len(sampled) - len(greedy)
    return {
        "overall_surface": overall,
        "judged_separately": [
            (
                "wire cleanliness (§2 tripwire belt) — lives in the run "
                "heartbeat, not these jsons; accumulation requires it quiet"
            ),
            (
                "flow-leg final call (A3.4: record + judge) — "
                "materially_below states the registered surface only"
            ),
        ],
        "primary": {
            "leg": "boundary greedy sim100 vs frozen 7/100 (paired exact)",
            "band": band,
            "boundary_successes": len(greedy),
            "anchor_successes": GREEDY_ANCHOR,
            "wins": len(wins),
            "losses": len(losses),
            "win_seeds": wins,
            "loss_seeds": losses,
            "held_seeds": sorted(greedy & GREEDY_SUCCESS_SEEDS),
            "p_improve_exact": round(p_improve, 6),
            "p_regress_exact": round(p_regress, 6),
        },
        "sampled_record": {
            "leg": "boundary sampled T=1.0 sim100 vs preflight floor (record-only)",
            "boundary_successes": len(sampled),
            "preflight_floor": preflight_floor,
            "delta_vs_floor": len(sampled) - preflight_floor,
            "decode_gap_base": base_gap,
            "decode_gap_boundary": boundary_gap,
            "decode_gap_movement": boundary_gap - base_gap,
            "success_seeds": sorted(sampled),
        },
        "flow_regression": {
            "leg": "boundary flow unseen100 euler-10 vs frozen 44/100 (record + judge)",
            "boundary_successes": len(flow),
            "anchor_successes": FLOW_ANCHOR,
            "p_below_anchor_exact": round(flow_tail, 6),
            "materially_below": flow_material,
            "material_line_max": 35,
            "win_seeds": sorted(flow - FLOW_SUCCESS_SEEDS),
            "loss_seeds": sorted(FLOW_SUCCESS_SEEDS - flow),
        },
        "material_alpha": MATERIAL_ALPHA,
        "checkpoint": checkpoint,
        "commit": greedy_payload["config"].get("commit"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-json", type=Path, required=True)
    parser.add_argument("--sampled-json", type=Path, required=True)
    parser.add_argument("--flow-json", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = boundary_verdict(
        json.loads(args.greedy_json.read_text()),
        json.loads(args.sampled_json.read_text()),
        json.loads(args.flow_json.read_text()),
        json.loads(args.preflight_json.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    primary, flow = result["primary"], result["flow_regression"]
    print(
        f"boundary surface: {result['overall_surface'].upper()} — PRIMARY "
        f"{primary['band']}: greedy {primary['boundary_successes']}/100 vs "
        f"anchor {GREEDY_ANCHOR} (wins {primary['wins']} / losses "
        f"{primary['losses']}, p_improve {primary['p_improve_exact']}, "
        f"p_regress {primary['p_regress_exact']}); sampled "
        f"{result['sampled_record']['boundary_successes']}/100 vs floor "
        f"{result['sampled_record']['preflight_floor']} (record-only, "
        f"decode-gap movement "
        f"{result['sampled_record']['decode_gap_movement']:+d}); flow "
        f"{flow['boundary_successes']}/100 vs {FLOW_ANCHOR} "
        f"(materially_below={flow['materially_below']}, judge owns the call)",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
