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
