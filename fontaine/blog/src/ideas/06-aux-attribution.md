# 6. Aux attribution arms — `confirmed` (aux HELPS actions; results 2026-08-06)

*Tag: `aux-subgoals` · idea #6 · [index](../ideas.md)*

**ANSWERED 2026-08-06 04:2xZ
([results post](../posts/2026-08-06-box-batch-results.md)): the
pre-registered decision rule fired REAL — aux-off costs +0.462 panel
chunk MAE (CI [0.387, 0.537], 7.5× the 0.061 replicate threshold,
leave-one-repo-out coherent). The mainline "within noise" expectation
is falsified: aux supervision shapes the action representation.**
Arms: A-s0/s1/s2 7.7966/7.8052/7.7355, B 8.2989; σ_seed(chunk) 0.038
→ E4B adopt band = 0.15 (floor binds). Twist held up: B's first_mae
3.5009 BEATS aux-on (3.94–4.11) with cond-sensitivity 1.13 vs
1.86–2.00 and predictions 8% closer to state-copy — consistent with
the #11 state-shortcut mechanism; the state-reliance probe is the
falsification instrument (all four npzs now banked). Standing rule:
**aux stays ON in every future recipe; an aux-off arm needs a new
pre-reg citing this result.**

The still-owed paired aux-on vs aux-off arms (does aux supervision
shape the representation, separate from "does narrating help" — the
100k run answered only the latter). Pre-registered mainline
expectation: within probe noise (±0.3). Promoted to arm B of the
paired 40k run after the wrap census killed unwrap-at-load:
[pre-reg](../posts/2026-08-05-prereg-paired-auxoff-40k.md). Primary read:
paired per-frame panel chunk_mae A@40k vs B@40k. **Executing on the
4×H100 box since 17:12Z** (parallel arms + 2 control seed replicates
for the noise floor, with a pre-registered decision rule:
[box batch pre-reg](../posts/2026-08-05-prereg-box-batch-4xh100.md)).
2026-08-06 01:3xZ: all four arms trained (A-s0 formal probe 7.0882@40k,
B 7.702@40k; s1/s2 at their boundary), panel evals chaining; **results
instrument `fontaine/scripts/box_batch_results.py` landed + oracled
before the data** — frozen decision rule, mechanical headline-column
matching vs report JSONs, σ_seed → the E4B adopt band and rig slot 2;
anchors/degenerate/synthetic-inflation oracles all passed. When the
four npz+report pairs land: one command produces the results-post
numbers and both finalization amendments.

- **External replication + a new rung-(a) probe (deep read
  2026-08-07, [post](../posts/2026-08-07-pi05-deep-read.md)):** our
  +0.462 aux-off cost is the same result class as π0.5's Fig. 13 —
  "Implicit HL" (subtask data in training, no runtime decoding) is
  their second-best config, i.e. semantic co-training shapes the
  action representation. Their further increment we have NEVER
  tested: **explicit runtime hierarchy** — decode a subtask first,
  condition actions on it. We own the seam: the `[subgoal|…]`
  conditioning slot (heavily dropped out; planner-less default
  well-trained). **Probe (zero training, quiet-GPU window):** have
  the AR model generate its own subgoal per panel frame, feed it
  back through `[subgoal|…]`, score panel vs no-hint baseline
  5.8026. Validity check first — eyes on a table of self-generated
  subgoals before any scalar (the never-generated-subgoal scar).
  Owner anchor in favor: the 21:43Z steer notes aux subgoals
  generalize strikingly OOD.
- **Lit slice 2026-08-07 03:2xZ — external support + two design
  constraints for rung (a):**
  [Hi-VLA systematic study](https://arxiv.org/html/2606.10267v1)
  (2606.10267) benchmarks hierarchy design and finds explicit
  language subgoals beat flat VLA **largest on long horizon** (flat
  25.30% → naive hierarchy 40.56% → best 67.08%; short-horizon gap
  near zero) — so the probe's per-step decomposition should expect
  the gain concentrated in LATE-horizon chunk_mae, mirroring the #1
  banked-prediction pattern. Two carried constraints: (i) their
  planner/controller are separate models — SELF-generated subgoals
  (our probe) are untested there, so ours is a genuine increment,
  not a replication; (ii) subgoal refresh granularity mattered a lot
  (4–8 s best; model-predicted horizons WORST) — our panel probe
  conditions per-frame, sidestepping refresh policy, but any later
  rollout arm must pre-register the refresh rule. Their hardest-task
  failure mode ("VLMs tend to ignore image inputs as task becomes
  harder") is the #11 state-dominant-bias story from the hierarchy
  side.
- **Rung (a) PRE-REGISTERED 2026-08-07 ~03:5xZ
  ([pre-reg](../posts/2026-08-07-prereg-selfsubgoal-probe.md)):** four
  arms on AR-100k (banked planner-less 5.8026 / oracle-truth
  `[subgoal|…]` / self-generated fed back through the slot /
  narrated-subgoal-only), validity table gated go/no-go BEFORE any
  scalar, frozen Δ-reads + horizon decomposition, ≤ 8 GPU-h with the
  q4-subset fallback. Execution at the first quiet local-GPU
  window ≥ the draws10_t1 boundary + its frozen reads.
- **Instrument LANDED 2026-08-07 ~04:3xZ (oracle-gated, this
  commit):** `bijou.eval --subgoal-mode {oracle,self}` — oracle mode
  renders per-frame TRUE labels through the trained slot (label-less
  frames decode the baseline context); self mode is the two-pass
  loop sharing one model load (pass 1 planner-less
  `[generate|subgoal actions]` = the `_narrsubgoal` arm free, pass 2
  feeds the text back through `[subgoal|…]` on the fast path =
  `_selfsubgoal`). `--dump-subgoals` retains per-frame generations
  (identity triple → text); `--selfsubgoal-force-empty` is the live
  oracle-(i) no-hint-limit run (`_emptyhint`, never a self-arm
  read); report JSON records the mode. Stage-1 validity table:
  `fontaine/scripts/selfsubgoal_stage1.py` (60 stratified frames,
  generation-only — NO scalars before the gate). The four
  pre-registered oracles' CPU halves are pinned in
  `tests/test_selfsubgoal.py` (prompt-byte equality of the no-hint
  limit and label-less oracle frames; one shared rendering path;
  pass 2's request set excludes subgoal); the real-checkpoint halves
  run pre-launch per the pre-reg. No semantic deviation from the
  pre-reg → no amendment needed.
- **Lit slice 2026-08-07 ~04:0xZ — two escalation anchors (radar
  only, no design change to rung (a)):** (i)
  [CAC-VLA](https://arxiv.org/html/2607.04816v1) (2607.04816)
  conditions the action head on VLM-predicted latent actions with a
  LEARNED GATE modulating conditioning strength — and trains on
  ground-truth-encoded conditioning while inferring on
  self-predicted, exactly the truth-vs-self asymmetry our
  Δ_oracle/Δ_self split diagnoses; if rung (a) lands in the
  "Δ_oracle < 0 but Δ_self ≥ 0" cell (generation quality is the
  gap), a gated-strength variant is a named escalation candidate
  (needs its own pre-reg). (ii) π0.7 (via the
  [NVIDIA WAM post](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/))
  escalates explicit-HL beyond text: HL policy emits subtask
  instructions, a BAGEL-based world model renders them as subgoal
  IMAGES, the action expert conditions on obs+subgoal-image —
  reported "necessary for some dataset-bias-breaking tasks where
  no-subgoal variants fail", and subgoal images reportedly speed
  training by making action prediction near-inverse-dynamics. Our
  text-slot probe is the cheap first rung of exactly this ladder.
- **Lit slice 2026-08-07 ~20:0xZ — two same-day releases (both
  announced Fri 08-07, read hours old; page:
  [subgoal-sourcing](../papers/subgoal-sourcing-post-training.md))
  land two fresh directional priors on rung (a) BEFORE its read
  (no design change; the pre-reg is frozen):** (i) HiRoC
  (2608.05999) shows subgoal-source misalignment is a
  cold-start-scale effect — an executor trained to condition on
  task instructions collapses on planner-generated subgoals until a
  dedicated SFT alignment stage retrains it on (obs, subgoal,
  chunk) triples; RL does not recover it. Prior for the probe:
  Δ_self ≤ Δ_oracle; and if the probe lands "oracle helps, self
  doesn't," HiRoC's alignment-SFT joins CAC-VLA's gate on the
  named-escalation list (cheaper: an SFT recipe, no new
  architecture). No oracle-vs-planner ablation in the paper — our
  Δ_oracle/Δ_self split measures the decomposition they skipped.
  (ii) VLA-Talker (2608.05738) at matched evidence: generate+
  supervise text 81.5 / inject+supervise 89.7 / inject+action-only
  97.4 on LIBERO — supervised language regeneration of available
  evidence costs 15.9 pts + 4.6× latency. TENSION with our aux-on
  +0.462 result, resolved (our synthesis, flagged as such): harm
  mechanism = token imbalance + copy-work; our aux fields are
  sparse predictions of latent task structure, not verbose copies —
  "supervise sparse structure prediction, never verbose evidence
  regeneration." Their result also predicts the narrated arm
  (injected, never supervised) is safe-to-helpful.
- **Lit slice 2026-08-08 ~01:1xZ (read while the stage-2 arms
  decoded; page:
  [runtime-plan-verification](../papers/runtime-plan-verification.md))
  — the escalation ladder above rung (a), priced before the
  readout:** three published shapes of the runtime loop any
  escalation would enter. (i) SV-VLA (2604.02965): chunked
  macro-plan + a 17×-cheaper trained verifier carrying the plan
  intent, replan on L1 discrepancy > τ — the ablation that matters:
  verification WITHOUT a recovery path crashes 90.9%→15.5%, so any
  #6 refresh policy must budget the re-decode, not just the gate;
  threshold sensitivity (τ 0.1/0.2/0.4 → 83.1/90.9/77.4) is their
  named open problem. (ii) Do What You Say (2510.16281): embodied
  CoT faithfulness — text right, actions wrong — is the
  execution-side noise source our Δ_oracle/Δ_self split does NOT
  price; their sample-and-align fix needs outcome simulation we
  don't have, but alignment scoring over our existing draws
  machinery is the cheap fragment. (iii) VINE (2512.03913):
  test-time compute scales in candidate-SUBGOAL width (K=1→5:
  28.9→44.4% unseen, peak K=4) — "sample N subgoals, condition on
  best" is implementable depth-1 with the #1 batched-draws
  instrument; their load-bearing failure-aware value function needs
  failure-labeled demos we don't have. All rollout-granularity; no
  change to the frozen reads or E5 — these price the escalation
  pre-regs if Δ_self earns one.
- **2026-08-08 ~02:5xZ — RUNG (a) READ OUT
  ([results post](../posts/2026-08-08-selfsubgoal-results.md), reads
  by `selfsubgoal_results.py`, execution oracles green):** the
  trained `[subgoal|…]` slot is ALIVE — **Δ_oracle −0.290
  [−0.331, −0.225]** on 25,788 labeled panel rows (≈ the whole
  panel), concentrated 6× late-horizon (last-10 −0.480 vs first-10
  −0.081; E3 confirmed) — but the closed loop returns
  **Δ_self −0.018 [−0.052, +0.026]**, a statistical zero at ~3×
  decode cost (E2 point-wise only; E5's falsifier does not fire by
  the letter, the deployment claim is dead anyway). **Channel read
  (the probe's only significant self-text number): narr − self
  +0.043 [+0.023, +0.064]** — identical text, suffix voice loses to
  the condition slot; with stage-1's ~10/60 phase-offset rows this
  locates the bottleneck in single-frame PHASE ESTIMATION, not the
  channel and not language quality. Decode-noise floor context
  −0.0008 ± 0.016 quoted per amendment 1. Cost ~3.2 GPU-h ≤ 8 gate.
  **Disposition: rung (a) closed — "don't deploy"; ceiling −0.29
  banked as the escalation prize. Next rung, own pre-reg required:
  subgoal-DRAWS selection (decode N candidates, condition on
  best-scored — the [runtime-plan-verification
  slice](../papers/runtime-plan-verification.md)'s VINE width
  scaling, depth-1 via the #1 batched-draws machinery); heavier
  siblings: planner-side SFT (HiRoC direction), rollout refresh
  policies (SV-VLA shape, #16-gated).**
- **2026-08-08 ~03:2xZ — RUNG (b) PRE-REGISTERED
  ([pre-reg](../posts/2026-08-08-prereg-subgoal-draws.md)),
  scorer cell settled by a targeted lit check first
  ([Self-Certainty page](../papers/self-certainty.md),
  2502.18581 NeurIPS 2025):** subgoal-DRAWS selection — pass 1
  decodes 9 candidates (greedy + 8 sampled T=1, draws10_t1 seeding
  verbatim), the frozen verifier-free scorer is **self-certainty**
  (mean KL from uniform of the candidate's own decode
  distributions; zero extra forwards, no oracle access; the
  published best reward-free selector on open-ended text — chosen
  over likelihood/medoid, which stay record-only alternates from
  the same dumps). Two conditioned arms: **bon** (scorer pick,
  deployment-honest primary) and **ceil** (token-F1-vs-true-label
  pick, record-only) — the ceiling bounds EVERY scorer at this
  width, so a failed falsifier still adjudicates no-diversity
  (family closes) vs no-scorer (MG-Select-style masked-contrast or
  planner SFT earn a look; masked reference is OOD for us without
  trained image dropout, so it's an escalation, not the primary).
  Head-to-head read: paired per-frame (bon − self) vs rung (a)'s
  banked self npz; falsified unless CI95 entirely below 0.
  Stage-1 candidates table gates stage 2 (diversity ≥ 2 unique
  strings on ≥ 50% of frames, else the rung closes at table cost).
  Gate ≤ 6 GPU-h, q4 fallback; venue local behind the goldenticket
  R1 chain; instrument (subgoal draws + SC dump + two modes) lands
  oracle-gated before launch, draws-0 limit must reproduce the
  rung-(a) self arm bit-exact at matched composition.
