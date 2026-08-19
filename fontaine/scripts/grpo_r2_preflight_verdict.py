"""R2 preflight leg-0 verdict (A3.3 F-premise gate, machine-readable).

Consumes the sampled T=1.0 sim100 out-json from ``sim.rollout_sim``
(``--serve-head ar --ar-temperature 1.0``, seeds 0-99 on the pinned
``step_002000_v2`` base) and emits the registered F-premise verdict —
the mechanized read of A3.3's "materially below greedy 7 -> abort":

- ``ABORT``  — the sampled count is materially below the frozen greedy
  anchor: exact one-sided binomial P(X <= k | X ~ Bin(100, 7/100))
  < 0.05 (i.e. k <= 2). The attenuation-relief premise is wrong; the
  run aborts unstarted and the read banks against the diagnosis.
- ``PASS``   — at-or-above the anchor (k >= 7): the premise holds
  first-contact and k/100 is the recorded training-decode competence
  floor.
- ``BAND``   — below the anchor but not materially (3 <= k <= 6):
  A3.3 names only the two outcomes above; this middle is surfaced, a
  decision post owns it (charter: decide + announce, no GO ask). The
  launcher refuses to fire on BAND without an explicit override.

Provenance is guarded loudly (serve_head, temperature, seed window):
a greedy json or the wrong checkpoint's leg must not mint a verdict.
Also emitted, record-only: the §1 mixed-group prediction
1-(1-p)^8-p^8 at p = k/100 — the wave-0 calibration gate's predicted
value (abort line 0.20 per A3.3).

Usage:
    uv run python fontaine/scripts/grpo_r2_preflight_verdict.py \
        --json outputs/sim/grpo_r2/preflight/sampled_t1_sim100.json \
        --out  outputs/sim/grpo_r2/preflight/preflight_verdict.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Frozen anchors (A3.1): token greedy 7/100 on seeds 0-99; the seeds
# are recorded for the paired context, the verdict runs on the count.
GREEDY_ANCHOR = 7
GREEDY_SUCCESS_SEEDS = (34, 35, 63, 68, 71, 91, 96)
NUM_SEEDS = 100
FIRST_SEED = 0
# The "materially below" line: exact one-sided binomial tail at the
# house 5% level under H0 p = anchor/n.
MATERIAL_ALPHA = 0.05
# §1 group shape (8 draws per seed) for the record-only prediction.
DRAWS = 8


def binom_tail_below(k: int, n: int, p: float) -> float:
    """Exact P(X <= k) for X ~ Bin(n, p)."""
    return float(
        sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1)),
    )


def mixed_group_prediction(p: float, draws: int = DRAWS) -> float:
    """§1 arithmetic: probability a ``draws``-draw group mixes
    successes and failures at per-draw success rate ``p``."""
    return 1.0 - (1.0 - p) ** draws - p**draws


def preflight_verdict(payload: dict) -> dict:
    """The F-premise read on a rollout_sim out-json payload; raises on
    provenance mismatch (the wrong leg must not mint a verdict)."""
    config = payload["config"]
    if config.get("serve_head") != "ar":
        raise ValueError(
            f"preflight leg-0 is the AR head (--serve-head ar), got "
            f"serve_head={config.get('serve_head')!r}",
        )
    if config.get("ar_temperature") != 1.0:
        raise ValueError(
            f"preflight leg-0 samples at T=1.0 (--ar-temperature 1.0), got "
            f"ar_temperature={config.get('ar_temperature')!r} — a greedy "
            "json cannot price the relief premise",
        )
    if config.get("seed") != FIRST_SEED or config.get("num_seeds") != NUM_SEEDS:
        raise ValueError(
            f"preflight leg-0 is seeds {FIRST_SEED}-{FIRST_SEED + NUM_SEEDS - 1}, "
            f"got seed={config.get('seed')} num_seeds={config.get('num_seeds')}",
        )
    episodes = payload["episodes"]
    seeds = sorted(e["seed"] for e in episodes)
    expected = list(range(FIRST_SEED, FIRST_SEED + NUM_SEEDS))
    if seeds != expected:
        raise ValueError(
            f"episode seed set is not exactly {FIRST_SEED}-"
            f"{FIRST_SEED + NUM_SEEDS - 1}: {len(episodes)} episodes",
        )

    success_seeds = sorted(
        e["seed"] for e in episodes if e.get("success_tick") is not None
    )
    k = len(success_seeds)
    p_below = binom_tail_below(k, NUM_SEEDS, GREEDY_ANCHOR / NUM_SEEDS)
    if p_below < MATERIAL_ALPHA:
        verdict = "ABORT"
    elif k >= GREEDY_ANCHOR:
        verdict = "PASS"
    else:
        verdict = "BAND"
    sampled_rate = k / NUM_SEEDS
    return {
        "verdict": verdict,
        "sampled_successes": k,
        "greedy_anchor": GREEDY_ANCHOR,
        "p_below_anchor_exact": round(p_below, 6),
        "material_alpha": MATERIAL_ALPHA,
        "success_seeds": success_seeds,
        "greedy_success_seeds": list(GREEDY_SUCCESS_SEEDS),
        "greedy_overlap": sorted(set(success_seeds) & set(GREEDY_SUCCESS_SEEDS)),
        "training_decode_floor": sampled_rate if verdict == "PASS" else None,
        "predicted_mixed_groups_frac": round(
            mixed_group_prediction(sampled_rate),
            4,
        ),
        "checkpoint": config.get("checkpoint"),
        "commit": config.get("commit"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = preflight_verdict(json.loads(args.json.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(
        f"F-premise verdict: {result['verdict']} — sampled "
        f"{result['sampled_successes']}/100 vs greedy anchor "
        f"{GREEDY_ANCHOR}/100 (exact P(X<={result['sampled_successes']}) = "
        f"{result['p_below_anchor_exact']}); predicted wave-0 mixed "
        f"fraction {result['predicted_mixed_groups_frac']} "
        f"(A3.3 abort line 0.20)",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
