# Pre-registration: GRPO signal probe (5 cells × 15 seeds × K=8, v3 frames)

*2026-08-12 20:1xZ. Finalizes §4 of the
[GRPO-on-sim design memo](2026-08-12-grpo-sim-design-memo.md)
(owner GO 13:16Z: "Yes, let's do this, get everything ready for when I
give you back the GPU"; sequencing 13:36Z: parallel oracle →
molmoact2-ftrig eval → this probe — both predecessors are complete, so
the probe launches on GPU handback). Rollouts only; no training.*

**Plain words.** Group-relative RL (GRPO) only works if, when you run
the same policy several times on the same starting position, the
attempts differ enough to rank them. Our policies barely move the boat
today, so maybe every attempt looks the same — then there is nothing
for RL to learn from. This probe runs each policy 8 times per starting
position under three kinds of randomness (sampling the action tokens,
re-rolling the flow decoder's noise, and a noisy "SDE" decoder that RL
could actually train through) and measures whether the attempts spread
out. If they don't, GRPO-on-sim parks; if they do, the spread tells us
which head (AR or flow) to train first.

## Cells (frozen)

All cells: **seeds 0–14** (contiguous, no selection), v3 frames
(`render_style` default, owner-approved 07:29Z), sim100 episode
conventions with the budget stated in TIME per today's owner catch:
**`--episode-seconds 30`** (= 900 ticks; the sim100 protocol's 30
replans × 30 ticks — note the queue item's "15 replans" was drift, the
registered sim100 protocol is 30). Policy seed 0, bf16 expert,
stats repo `so101_pick_place_v2`, task string unchanged.

| cell | checkpoint | decode | stochasticity | draws |
|---|---|---|---|---|
| 1 | `er60k` (ar_backbone, step_060000) | AR greedy→sampled | `--ar-temperature 1.0` | 8 (all sampled) |
| 2 | `er60k` | AR sampled | `--ar-temperature 1.6` (SimpleVLA-RL) | 8 |
| 3 | `teacher80k` (flow, heun-30) | ODE | fresh keyed noise per draw | 9 (draw 0 = deterministic anchor) |
| 4 | `ftrig4k` (snapflow student rig-ft, euler-1) | ODE | fresh keyed noise per draw | 9 (draw 0 = anchor) |
| 5 | `teacher80k` | **SDE euler-10**, `--sde-noise-level 0.5` (πRL action value) | per-step keyed Gaussian | 8 (all stochastic) |

Anchor passes (deterministic, 15 episodes each, same driver/config):
`er60k` greedy (cells 1–2's competence anchor) and `teacher80k`
**euler-10 ODE** (cell 5's anchor — the a=0 bit-identity class of the
SDE sampler; cell 3's heun-30 rows are NOT cell 5's anchor, different
solver voice).

**Cell 5b (registered hedge, runs only on trigger):** if cell 5's
competence cost reads worse than −1.0 cm (paired CI excluding −1.0 on
the bad side), one additional cell at `--sde-noise-level 0.3` runs
within the same gate, same reads.

## Driver + determinism discipline (frozen)

Everything runs the **parallel driver at workers=8** — including the
anchor passes. The parallel GPU oracle read FAIL on bit-match at
workers>1 (batched bf16 decode drift; frozen rule: parallel is
paired-only), so **every comparison in this probe is within-driver,
within-config**; no pooled claims against banked sequential rows. The
v3-rerun row join from the memo is demoted to a record-only
cross-check if/when that rerun lands. Per-draw streams are keyed by
the identity triple + draw-suffixed repo_id (draw 0 unsuffixed);
SDE step noise rides its own domain (`SDE_STEP_DOMAIN`), initial noise
stays `stable_noise`.

## Instrument (landed before this finalization)

- `80a5388` — `FlowDecoder.sample_actions_sde` (Euler–Maruyama,
  Flow-GRPO marginal-preserving SDE) + a=0 bit-identity/logprob
  oracles.
- `0f7ea86` — sequential `--draws` + `--ar-temperature`, draw keying
  via repo_id suffix, draw-0 banked-identity oracles.
- `8b6d034` — SDE wired end-to-end (`--sde-noise-level` on both
  drivers, per-item keyed step noise, batch-composition-invariant) +
  parallel driver (seed, draw) work units + parity oracles.
- `c26a99e` — `--episode-seconds` (time-stated budget).

check.py 797 green at finalization.

## Reads (frozen)

1. **Primary, per cell**: per-seed within-group std of
   `progress_final_cm` over the K=8 stochastic draws; statistic =
   **median over the 15 seeds**; **signal bar: ≥ 0.25 cm** (a quarter
   of teacher80k's spot20 paired effect, as proposed in the memo).
2. **Non-degeneracy** (record-only): fraction of groups with std
   ≥ 0.05 cm — the dynamic-sampling-filter survival analog.
3. **Competence cost, per cell**: mean over seeds of (group mean −
   its deterministic anchor), 10k-resample bootstrap CI95 paired by
   seed. Anchors: cells 1–2 → er60k greedy pass; cell 3/4 → in-cell
   draw 0; cell 5 → teacher80k euler-10 pass.
4. **Guard rates** (record-only): knock-aways (progress_final ≤
   −1 cm), final_upright < 0.9, reset strikes (validity: must be 0),
   successes; best-point (`progress_cm`) group std alongside the
   primary.

AR token entropy is NOT registered (not instrumented in the rollout
path); it may ride a later amendment if phase 2 wants it.

## Decision rule (frozen, from the approved memo)

- **No cell clears 0.25 cm** → GRPO-on-sim parks; the sim axis
  continues via visuals/task semantics.
- **An AR cell clears it** AND its competence cost CI does not sit
  entirely below −1.0 cm → phase 2 = token-GRPO per the SimpleVLA-RL
  mapping (cheapest infra).
- **Only flow cells (3/4/5) clear** → phase 2 = Flow-GRPO SDE,
  expert-only, on the clearing arm with the best competence/signal
  trade.
- **Both families clear** → AR first (infra), flow second, joint
  parked for the merged molmo_flow model (owner lane).

## Cost + tripwires

660 episodes at 30 s (5 cells: 120+120+135+135+120, + 30 anchor).
Parallel-path estimate ~4 episodes/min at workers=8 → **~2.8 h wall;
gate ≤ 3.5 GPU-h** (includes anchors + slack). Tripwire: if the first
completed cell's pace projects the total past the gate, stop at the
cell boundary and re-scope in-channel (cells are independently
readable; partial cells are discarded, never pooled). The sequential
fallback is NOT viable at v3 render pace (~2 min/episode ⇒ ~22 h) —
if the parallel driver is unavailable the probe shrinks by owner call
rather than silently downgrading.

Launch checklist (at GPU handback): re-pin HEAD + checkpoint paths in
the launcher, babysit entry + registry at launch, per-cell
`--rows-jsonl` stream, results post same session.

## Results (amendment 1, 2026-08-13 01:xxZ — re-scoped run)

**Tripwire fired at the cell-1 boundary** (22:58Z 08-12): measured
pace ~1.13 GPU-h/cell (68 min; the parallel driver runs one seed's 8
draws as a worker-wave, so a cell is 15 waves of ~4.5 min — the ~4
episodes/min estimate assumed cross-seed packing). Full 7-pass plan
projected ~5.9 GPU-h vs the ≤3.5 gate. Re-scope announced in-channel
21:58Z (before the boundary), no objection: **anchors + cells 1, 2, 5
ran; cells 3/4 (flow ODE fresh-noise) parked** — the channel our
banked ceiling-ladder read already measured as NULL for flow;
cell 5's SDE is the channel Flow-GRPO trains through. Partial cells:
none (every reported cell is complete, 15/15 groups).

| read (frozen) | cell 1 AR t=1.0 | cell 2 AR t=1.6 | cell 5 SDE a=0.5 |
|---|---|---|---|
| median group std (ddof0) | **0.771** | **2.461** | CELL5_STD |
| vs 0.25 cm bar | CLEARS 3.1× | CLEARS 9.8× | CELL5_BAR |
| non-degeneracy (≥0.05 cm) | 13/15 | 15/15 | CELL5_NONDEG |
| competence cost (cm) | −0.351 | −1.081 | CELL5_COST |
| cost CI95 (paired) | [−1.117, +0.207] | [−1.556, −0.634] | CELL5_CI |
| knock-aways / tipped | 10 / 6 | 42 / 10 | CELL5_GUARD |
| successes | 1 | 0 | CELL5_SUCC |
| reset strikes (must be 0) | 0 | 0 | CELL5_STRIKES |

Anchors: er60k greedy 15/15, teacher80k euler-10 ODE 15/15 (both
complete before cell 1). Best-point (progress_cm) medians: 0.228 /
0.891 / CELL5_BEST. ddof=1 medians recorded in the reads JSON (ddof
was not frozen in the pre-reg; primary is ddof=0, the population std
GRPO's own advantage normalization uses).

DECISION_BLOCK

Cost: GPU_HOURS GPU-h total vs the 3.5 gate (re-scoped). Instrument:
`read_grpo_signal_probe.py` (frozen reads), `grpo_probe_chart.py`.
Chart + reads JSON on the reports Space. Cells 3/4 re-queue as a
final-word pair (~2.2 GPU-h) on owner call only.
