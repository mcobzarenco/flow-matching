"""Phase-4 acceptance gate: the frozen-wave replay, old stack vs new
(docs/molmoact2-retirement.md phase 4; fontaine's v2-wave amendment).

PRE-REGISTERED EXPECTATIONS:
- per-row grammar masks verify bit-equal through BOTH stacks (the
  packbits surface vs bins-only recomputation);
- per-token teacher-forced logprobs, OLD (port predictor + port
  replay) vs NEW (MolmoAct2DiscreteStack + bijou.grpo_replay), on the
  SAME rows at the SAME restored weights, agree within 1e-5 — both
  sides are same-surface (one wide fp32-text forward over identical
  JPEG-decoded inputs), so the JPEG budget cancels in the A/B;
- the banked-vs-replay ratio (JPEG-inclusive, either stack) is
  REPORTED, not gated — the loop's own registered bound governs it;
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

import torch

from bijou.grpo_replay import MolmoAct2DiscreteStack, load_training_rows
from bijou.grpo_replay import replay_logprobs as new_replay_logprobs
from bijou.grpo_replay import verify_recorded_masks as new_verify_masks
from bijou.molmoact2 import MolmoAct2Predictor
from bijou.molmoact2.replay import replay_logprobs as old_replay_logprobs
from bijou.molmoact2.replay import verify_recorded_masks as old_verify_masks
from bijou.train_grpo import GRPOConfig, grpo_objective_sums
from sim.grpo_loop import apply_option_a_freeze

FONTAINE = Path.home() / "flow-matching"
WAVES = (
    ("v1", FONTAINE / "outputs/sim/grpo_phase2_a", "step_0004", "step_0003.pt"),
    ("v2", FONTAINE / "outputs/sim/grpo_phase2_b", "step_0006", "step_0005.pt"),
)
RELEASE_HF = "allenai/MolmoAct2-SO100_101"
RELEASE_BIJOU = Path("converted/molmoact2_so100_101_release")
FAST_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--allenai--MolmoAct2-FAST-Tokenizer"
    / "snapshots/d45593b4c863d0bc1ca064f8b352fa16b75c38e8"
)
ROWS_PER_WAVE = 256
LOGPROB_BOUND = 1e-5


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
    old = MolmoAct2Predictor.load(
        RELEASE_HF,
        "so100_so101_molmoact2",
        device=device,
        dtype=torch.bfloat16,
        fast_tokenizer=str(FAST_SNAPSHOT),
    )
    old.trunk.text.float()  # the loop's training-dtype convention
    new = MolmoAct2DiscreteStack.load(
        RELEASE_BIJOU,
        device=device,
        dtype=torch.bfloat16,
        fast_tokenizer=str(FAST_SNAPSHOT),
    )
    new.trunk.text.float()

    verdict_lines: list[str] = []
    failed = False
    for name, run_dir, wave, pt_name in WAVES:
        meta, rows = load_training_rows(run_dir / "rows" / wave)
        task = str(meta["task"])
        temperature = float(meta["temperature"])
        print(
            f"[{name}] {len(rows)} rows @ T={temperature} "
            f"(reward={json.loads((run_dir / 'meta.json').read_text())['train_reward']})",
            flush=True,
        )
        old_named = apply_option_a_freeze(old)
        new_named = apply_option_a_freeze(new)
        restore_pt(old_named, run_dir / pt_name)
        restore_pt(new_named, run_dir / pt_name)

        # Masks: EVERY row, both stacks (bit-equality raises inside).
        for row in rows:
            # The two ReplayRow dataclasses are the SAME frozen format
            # (decision 10) — nominally distinct until phase 5 deletes
            # the port's; duck-passing is the point of the A/B.
            old_verify_masks(old, row)  # type: ignore[arg-type]  # frozen-format duck rows
            new_verify_masks(new, row)
        print(f"[{name}] masks bit-equal on all {len(rows)} rows", flush=True)

        subset = rows[:ROWS_PER_WAVE]
        deltas: list[float] = []
        banked_deltas: list[float] = []
        with torch.no_grad():
            for row in subset:
                old_lp, _ = old_replay_logprobs(
                    old,
                    [row],  # type: ignore[list-item]  # frozen-format duck rows
                    task=task,
                    temperature=temperature,
                )
                new_lp, _ = new_replay_logprobs(
                    new,
                    [row],
                    task=task,
                    temperature=temperature,
                )
                width = int(row.ids.shape[0])
                delta = float(
                    (old_lp[0, :width] - new_lp[0, :width]).abs().max(),
                )
                deltas.append(delta)
                banked_deltas.append(
                    float(
                        (
                            new_lp[0, :width].cpu()
                            - torch.from_numpy(row.logprobs).float()
                        )
                        .abs()
                        .max(),
                    ),
                )
        worst = max(deltas)
        line = (
            f"[{name}] old-vs-new logprobs on {len(subset)} rows: "
            f"max |delta| {worst:.3e} (bound {LOGPROB_BOUND:.0e}); "
            f"banked-vs-replay (JPEG-inclusive, report-only) max "
            f"{max(banked_deltas):.3e}"
        )
        print(line, flush=True)
        verdict_lines.append(line)
        if worst > LOGPROB_BOUND:
            failed = True

        # The objective composed on identical advantages.
        config = GRPOConfig(temperature=temperature)
        advantages = torch.linspace(-1.0, 1.0, len(subset), device=device)
        old_stack_lp = torch.zeros(
            (len(subset), max(int(r.ids.shape[0]) for r in subset)),
            device=device,
        )
        new_stack_lp = torch.zeros_like(old_stack_lp)
        decisions = torch.zeros_like(old_stack_lp, dtype=torch.bool)
        with torch.no_grad():
            for index, row in enumerate(subset):
                width = int(row.ids.shape[0])
                o, _ = old_replay_logprobs(
                    old,
                    [row],  # type: ignore[list-item]  # frozen-format duck rows
                    task=task,
                    temperature=temperature,
                )
                n, _ = new_replay_logprobs(
                    new,
                    [row],
                    task=task,
                    temperature=temperature,
                )
                old_stack_lp[index, :width] = o[0, :width]
                new_stack_lp[index, :width] = n[0, :width]
                decisions[index, :width] = True
        banked = torch.zeros_like(old_stack_lp)
        for index, row in enumerate(subset):
            banked[index, : row.ids.shape[0]] = torch.from_numpy(
                row.logprobs,
            ).to(device)
        old_sum, old_count, _ = grpo_objective_sums(
            old_stack_lp,
            banked,
            advantages,
            decisions,
            config,
        )
        new_sum, new_count, _ = grpo_objective_sums(
            new_stack_lp,
            banked,
            advantages,
            decisions,
            config,
        )
        objective_delta = float((old_sum / old_count - new_sum / new_count).abs())
        line = (
            f"[{name}] objective (per-token) old-vs-new |delta| {objective_delta:.3e}"
        )
        print(line, flush=True)
        verdict_lines.append(line)

    print("---")
    for line in verdict_lines:
        print(line)
    if failed:
        print("PHASE-4 REPLAY GATE: FAIL — a re-baseline decision, not a bump")
        return 1
    print("PHASE-4 REPLAY GATE: PASS (old-vs-new in-bound on both waves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
