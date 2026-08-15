"""Banked-wave replay integrity through the first-class GRPO stack
(bijou.grpo_replay) — mask bit-equality on every row + banked-vs-replay
logprob spread reporting.

HISTORY: this probe's first life was the phase-4 old-vs-new acceptance
gate (docs/molmoact2-retirement.md), run 2026-08-14 on the box against
fontaine's R1-A/R1-B waves at their collection weights — VERDICT PASS:
masks bit-equal on all rows of both waves; port-vs-first-class
logprobs max |Δ| 4.4e-5 (v1) / 5.7e-5 (v2) within the re-baselined
1e-4 cross-decomposition bound (monolithic port forward vs scaffold
prefill+continuation, fp32 — mechanism note below); per-token
objective deltas 0.0 / 6.5e-8. The old-side (port) comparison retired
with bijou/molmoact2 in phase 5 — rerun it only at tag
`pre-molmoact2-retirement`+phases. What remains is the NEW-stack wave
integrity read, reusable for any future banked wave.

PRE-REGISTERED EXPECTATIONS (re-baselined 2026-08-14 after the first
read — mechanism named, not bumped):
- per-row grammar masks verify bit-equal through BOTH stacks (the
  packbits surface vs bins-only recomputation);
- per-token teacher-forced logprobs, OLD (port predictor + port
  replay) vs NEW (MolmoAct2DiscreteStack + bijou.grpo_replay), on the
  SAME rows at the SAME restored weights, agree within **1e-4**. The
  first registration said 1e-5 — the same-surface bound — but the two
  replays are NOT the same decomposition: the port forwards
  prompt+suffix MONOLITHICALLY, the first-class replay is the
  scaffold's prefill+continuation. Cross-decomposition fp32 drift
  measured 4.4–5.7e-5 worst-token here, the same class and decade as
  the phase-2 fp32 diagnostic (2.8e-5). SECOND occurrence of the
  "1e-5 implies same decomposition" trap — the rule: 1e-5 bounds
  apply between IDENTICAL forward decompositions only. Ratio impact
  exp(1e-4)−1 ≈ 0.01%, three orders below the clip band;
- the banked-vs-replay delta (JPEG + policy-move inclusive) is
  REPORTED, not gated: the loop ITSELF trained under this spread —
  R1-B's own step-7 heartbeat records clip_fraction 0.141 with
  mean_ratio 1.0014, i.e. a fat banked-vs-replay tail was the run's
  operating condition (fontaine's design acknowledges it: the k3
  anchor penalty compares two REPLAY forwards so the JPEG floor
  cancels; the surrogate clips the rest);
- grpo_objective_sums on identical advantages agrees old-vs-new.

Sampling: masks verified on EVERY row of each wave; the logprob A/B
runs on the first ROWS_PER_WAVE rows in index order (deterministic,
every seed represented at 8 draws x 30 replans per seed).

Waves (fontaine's R1-A/R1-B banks, collection policies = the .pt saved
one step earlier, surface a):
- v1: outputs/sim/grpo_phase2_a rows/step_0004 @ step_0003.pt
- v2: outputs/sim/grpo_phase2_b rows/step_0006 @ step_0005.pt

Run on the box from ~/marius-convert-gate:
  PYTHONPATH=. uv run python probes/probe_grpo_replay_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bijou.grpo_replay import (
    MolmoAct2DiscreteStack,
    load_training_rows,
    replay_logprobs,
    verify_recorded_masks,
)
from sim.grpo_loop import apply_option_a_freeze

FONTAINE = Path.home() / "flow-matching"
WAVES = (
    ("v1", FONTAINE / "outputs/sim/grpo_phase2_a", "step_0004", "step_0003.pt"),
    ("v2", FONTAINE / "outputs/sim/grpo_phase2_b", "step_0006", "step_0005.pt"),
)
RELEASE_BIJOU = Path("converted/vla_molmoact2_so100_101_release")
FAST_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--allenai--MolmoAct2-FAST-Tokenizer"
    / "snapshots/d45593b4c863d0bc1ca064f8b352fa16b75c38e8"
)
ROWS_PER_WAVE = 256


def restore_pt(named: list[tuple[str, torch.nn.Parameter]], pt: Path) -> None:
    """The loop checkpoint's named trainable tensors copied in (the
    load_checkpoint core without optimizer state — scoring only)."""
    payload = torch.load(pt, map_location="cpu", weights_only=False)
    tensors: dict[str, torch.Tensor] = payload["trainable"]
    by_name = dict(named)
    if set(tensors) != set(by_name):
        raise SystemExit(
            f"{pt} trainable set does not match the live surface "
            f"({sorted(tensors)} vs {sorted(by_name)})",
        )
    for name, value in tensors.items():
        if name not in by_name:
            raise SystemExit(f"{pt} carries unknown tensor {name!r}")
        with torch.no_grad():
            by_name[name].copy_(value.to(by_name[name].device, by_name[name].dtype))


def main() -> int:
    device = torch.device("cuda")
    stack = MolmoAct2DiscreteStack.load(
        RELEASE_BIJOU,
        device=device,
        dtype=torch.bfloat16,
        fast_tokenizer=str(FAST_SNAPSHOT),
    )
    stack.trunk.text.float()  # the loop's training-dtype convention

    verdict_lines: list[str] = []
    for name, run_dir, wave, pt_name in WAVES:
        meta, rows = load_training_rows(run_dir / "rows" / wave)
        task = str(meta["task"])
        temperature = float(meta["temperature"])
        print(
            f"[{name}] {len(rows)} rows @ T={temperature} "
            f"(reward="
            f"{json.loads((run_dir / 'meta.json').read_text()).get('train_reward', 'v1')})",
            flush=True,
        )
        named = apply_option_a_freeze(stack)
        restore_pt(named, run_dir / pt_name)

        # Masks: EVERY row (bit-equality raises inside).
        for row in rows:
            verify_recorded_masks(stack, row)
        print(f"[{name}] masks bit-equal on all {len(rows)} rows", flush=True)

        subset = rows[:ROWS_PER_WAVE]
        banked_deltas: list[float] = []
        with torch.no_grad():
            for row in subset:
                lp, _ = replay_logprobs(
                    stack,
                    [row],
                    task=task,
                    temperature=temperature,
                )
                width = int(row.ids.shape[0])
                banked_deltas.append(
                    float(
                        (lp[0, :width].cpu() - torch.from_numpy(row.logprobs).float())
                        .abs()
                        .max(),
                    ),
                )
        spread = np.array(banked_deltas)
        line = (
            f"[{name}] banked-vs-replay on {len(subset)} rows "
            f"(JPEG + policy-history inclusive, REPORT-ONLY — the loop's "
            f"clipped surrogate is the consumer): "
            f"median {np.median(spread):.3e}, p90 "
            f"{np.quantile(spread, 0.9):.3e}, max {spread.max():.3e}"
        )
        print(line, flush=True)
        verdict_lines.append(line)

    print("---")
    for line in verdict_lines:
        print(line)
    print("WAVE INTEGRITY: PASS (masks bit-equal; spreads reported above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
