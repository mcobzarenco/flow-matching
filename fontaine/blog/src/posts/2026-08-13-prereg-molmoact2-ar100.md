# Pre-reg: MolmoAct2 discrete (AR) pathway, 100-seed sim eval — the RL-substrate gate

*2026-08-13 11:2xZ. The `molmoact2-ar-head-port` item (d) gate read,
launched under the owner's 11:07Z delegation ("you make the
decisions, ensure we make progress and GPU is always busy") — the
lane itself was owner-called 10:02Z (focus: release molmoact2 + AR
GRPO). Frozen before launch; launch immediately follows the
commit.*

## Plain words

The released MolmoAct2 robot model can produce actions two ways: a
"flow" pathway (the deployment default — it scored 9/100 successes
in our simulator yesterday, the only successes this task has ever
seen) and a "discrete" pathway that writes actions as tokens, like
text. Reinforcement learning of the kind we're building (token-GRPO)
can only train the token pathway. This run asks the gate question:
**is the token pathway also capable of succeeding in our simulator,
or is all the release's competence locked in the flow pathway?** We
run the exact same 100 test scenarios with the same time budget the
flow pathway got, decoded through the token head instead.

## Arm (single, frozen)

- **Serving**: the first-class port (`MolmoAct2Predictor
  .predict_action_discrete`, parity-gated stack) through
  `sim.rollout_sim_parallel --molmoact2-discrete
  allenai/MolmoAct2-SO100_101` (adapter `931b9a5`, preflight PASS) —
  **reference semantics**: unconstrained greedy to EOS, span
  extraction, OpenFAST decode, zeros-fallback on non-decodable
  emissions (counted per predict), official SO-101 shim (signs
  `1,-1,1,1,1,1`, offsets `0,90,90,0,0,0` — the map amendment 3 of
  the convmap pre-reg validated), norm tag `so100_so101_molmoact2`,
  bf16 trunk.
- **Protocol**: seeds 0–99, `--episode-seconds 30` (900 ticks),
  `--workers 8`, v3 frames, flipped mount (registered geometry),
  sim100 conventions. Videos at draw 0 per driver default.
- **Command** (verbatim):
  `MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel
  --molmoact2-discrete allenai/MolmoAct2-SO100_101 --seed 0
  --num-seeds 100 --workers 8 --episode-seconds 30
  --out-json outputs/sim/molmoact2_ar100/rows.json
  --rows-jsonl outputs/sim/molmoact2_ar100/rows.jsonl`

## Frozen reads

1. **Primary: successes / 100** (the sim's physics criterion,
   unchanged). Comparator: the flow pathway's **9/100** on the SAME
   seeds + budget (`release_officialmap_a_100ep_30s`). RECORD-ONLY
   cross-stack caveat, stated up front: the flow run rode the
   converted-checkpoint BijouPolicy serving; this run rides the
   first-class stack. Both are parity-gated to the same reference,
   but the rows are never pooled.
2. **Validity**: reset strikes must be 0; **zero-fallback emission
   count** (a non-decodable emission = a zero-action chunk under
   reference semantics — a high rate invalidates the competence
   read and becomes its own finding).
3. **Secondary (record-only)**: engagement split (`progress_cm` >
   1 cm count), knock-aways (`progress_final` ≤ −1 cm), mean/median
   `progress_final_cm`, per-seed chart, success-seed overlap with
   the flow run's {42, 47, 55, 56, 57, 73, 85, 92, 93}.

## Gate + tripwires

- **≤ 1.5 GPU-h** (estimate ~0.5–0.9: ~3,000 batch-1 predicts at
  the measured 0.3–0.8 s + sim stepping; the flow run took 51 min).
- Pace tripwire: projected wall > 2.5 h at the first ~10 episodes →
  stop at an episode boundary, re-scope in-channel.
- Babysit entry at launch; results post + entry prune at completion.

## Decision relevance (why this gates the RL lane)

- **≥ 1 success** → the token pathway is a viable RL substrate: the
  token-GRPO pre-reg (memo 2026-08-13, re-pointed per the 10:02Z
  steering) finalizes against THIS checkpoint/pathway.
- **0/100 while flow reads 9/100** → the release's competence is
  flow-locked: the owner decision point becomes AR-SFT-then-RL vs
  Flow-GRPO on the flow head — with these numbers in hand either
  way.

## Results (2026-08-13 12:2xZ, same session — run 11:20–12:26Z, rc 0)

**1/100 successes — the AR pathway IS success-capable in our sim.**
Seed 73 (success tick 622), one of the flow pathway's own nine
success seeds. By the frozen decision rule the token-GRPO lane
proceeds on this checkpoint + pathway.

- **Validity green**: reset strikes 0/100; spend **~1.15 GPU-h ≤
  1.5 gate** (69.1 min wall, 390 predict rounds, mean batch 7.7).
- **The decode-brittleness finding (frozen read 2)**: **202 / 2,991
  predicts (6.8%) hit the zeros-fallback** — the model's
  unconstrained greedy emission failed to decode (short/over-long
  stream or a quantization-hole symbol), so ~1 in 15 chunks executed
  as a ZERO-action chunk under reference semantics. This is exactly
  the class the `grammar_masked` decode repairs by construction
  (and at greedy it changes nothing else — 0 violations on legal
  streams, smoke + fixture oracles).
- Secondary: engagement 13/100 (best-point > 1 cm), knock-aways
  27/100 (≤ −1 cm, worst −11.8 at seed 67), 67 quiet; mean
  `progress_final` −0.87 cm, median −0.00.
- **vs the flow pathway (record-only, cross-stack)**: 1/100 vs
  9/100 successes, mean −0.87 vs −0.27, knock-aways 27 vs 27. The
  token head is ~9× less likely to complete the place at greedy but
  is not inert — and it carries a 6.8% self-inflicted zero-action
  handicap the flow head doesn't have.

## Amendment 1 (frozen 12:3xZ pre-launch): arm B — grammar-masked decode

The 6.8% fallback rate converts a design assumption into a
measurable question: **does repairing the non-decodable emissions
(the RL rollout decode) change deployment competence?** Arm B is
arm A with `--molmoact2-grammar-masked` — identical seeds, budget,
serving, shim; the ONLY change is the decode loop (scaffold fed,
bins budget-masked, end forced; every emission decodes).

- **Primary**: successes / 100; paired per-seed `progress_final_cm`
  delta vs arm A (same seeds → paired bootstrap CI95).
- **Record-only**: masked-violation counts per predict (the
  divergence instrument — expected ≈ the fallback sites), per-seed
  chart, flow-seed overlap.
- **Gate ≤ 1.5 GPU-h** (arm A measured 1.15). Launched detached at
  the amendment commit under the 11:07Z delegation; results ride
  the next session if the wall crosses the session deadline.

### Amendment 1 results (2026-08-13 13:3xZ, same session — arm B 12:29–13:31Z, rc 0)

**The grammar-masked decode is a registered improvement.** Paired
per-seed delta (B − A, same 100 seeds): **+0.728 cm, CI95 [+0.147,
+1.325] — excludes zero**. Knock-aways **27 → 13** (halved); mean
`progress_final` −0.87 → −0.14 cm; 47/100 seeds behaviorally
different; zero-fallbacks **0/2,996 by construction** (vs arm A's
202); strikes 0; ~1.05/1.5 GPU-h (63.1 min — faster than arm A:
masked decode never overruns to EOS). Successes 1/100 on each arm at
different seeds (A: 73, B: 1) — the success count didn't move, the
competence floor did.

**Standing decision (owner delegation 11:07Z/11:18Z): the masked
decode is the default serving mode for the AR pathway** — it is the
RL rollout decode anyway, and it deployment-dominates the reference
on every read that moved.
