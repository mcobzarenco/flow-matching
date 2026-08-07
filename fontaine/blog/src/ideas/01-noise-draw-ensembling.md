# 1. Inference-time noise-draw ensembling — flow side BANKED (mean-of-10 5.365 beats the AR anchor; fairness + energy-score reads in); open rung: the golden-ticket noise screen (pre-reg POSTED 2026-08-07, execution awaits a quiet local window)

*Tag: `noise-draws` · idea #1 · [index](../ideas.md)*

- **Hypothesis:** mean-of-N noise draws through the flow expert cuts
  panel MAE substantially in the unconstrained class (mainline
  measured 5.30°→2.88° on motion frames for a ft'd model,
  mean-of-10); the stage-2 flow-on-AR-trunk lineage (6.623 panel)
  should benefit similarly.
- **Expected effect:** large on the flow lineage's panel number;
  unconstrained-class only (charter §2) until distilled.
- **Cost:** ~20 lines eval-side + one eval burst per N in {1,5,10}.
  Zero training.
- **Falsification:** paired panel eval, same checkpoint, draws
  stated. Check unimodality of draws first (averaging multi-modal
  draws is wrong): per-frame draw spread on a few hundred panel
  frames. If mean-of-10 does not beat single-draw beyond the panel's
  pairing noise, kill.
- **Open sub-question:** an AR-family analogue (temperature/nucleus
  chunk ensembles, chunk-level medians) — separate screen.
- **Instrument note (2026-08-05):** upstream already ships
  `bijou.eval --sample-draws N` (a16e65a) — verify its semantics
  (independent draws? mean-of-N in action space?) before writing any
  code; the pre-reg's eval-side work may reduce to a flag.
- **Paired-analysis prediction (2026-08-05 ~20:15Z, before the
  draws-10 numbers land):** flow's deficit vs AR is a monotone
  horizon-divergence (crossover at step 2; +1.2 by step 40 —
  [post](../posts/2026-08-05-flow-vs-ar-paired.md)). If per-draw
  spread grows along the horizon, mean-of-N should close the
  *late-horizon* deficit preferentially: chunk_mae moves a lot,
  first_mae barely. Score the draws-10 run per-step, not just
  pooled.
- **Fairness reads pre-registered (2026-08-05 ~22:1xZ,
  [Amendment 1](../posts/2026-08-05-draws-fairness-amendment.md)),
  from the owner's 21:49Z is-MAE-unfair-to-flow challenge:**
  instrument finding — per-draw chunks never left the process
  (`--dump-predictions` stores the post-average), so
  `bijou.eval --dump-draws` landed (tests + bit-exact scoring
  oracle), a 2,458-frame stride-7 probe plan + launcher
  (`~/eval_flow80k_drawsprobe_dump.sh`, ~30 min 1×GPU) is frozen,
  and `fontaine/scripts/draws_fairness.py` computes the three
  pre-declared reads (mean-of-draws / best-of-N / dispersion-
  conditioned deficit; degenerate draws=1 validation reproduces
  6.6232 exactly). Launch at the first quiet local-GPU boundary
  after the draws chain.
- **FIRST DRAWS-10 PANEL READ (2026-08-05 23:31Z, run 2 of the
  chain, full 25.8k-frame panel): chunk_mae 5.365 / first_mae 1.424**
  vs single-draw 6.6232/1.9331 (−19%/−26%) — **mean-of-10 flow beats
  the AR-100k anchor (5.8026/2.1431) on both columns** (unconstrained
  class: 10× NFE; charter §2 caveat until distilled — #12 SnapFlow
  leg). Banked-prediction check: "chunk_mae moves a lot" ✅;
  "first_mae barely" ❌ — first_mae moved 26%, so the gain is NOT
  purely late-horizon; per-step decomposition promoted to a required
  read in the results post (after runs 3–5 + the fairness probe).
- **Read 4 pre-declared (2026-08-05 ~22:5xZ,
  [Amendment 2](../posts/2026-08-05-draws-fairness-amendment2.md), from
  the lit slice):** the **energy score** (RMS-normalized, valid-
  element mask, N=10 vs AR's degenerate N=1) — a strictly proper
  scoring rule where neither mode-averaging nor scatter wins for
  free; the principled middle between MAE and the best-of-N oracle
  bound, and the candidate distributional column for ranking flow
  arms on the comm holdout. Source: Energy Policy
  ([2510.12483](https://arxiv.org/abs/2510.12483)) trains on it;
  we take the metric, computable on CPU from the same
  `--dump-draws` npz. `read4_energy_score` in `draws_fairness.py` +
  degenerate draws=1 validation must land BEFORE the probe npz is
  opened (next CPU work item alongside the E4B launch checklist).
- **FAIRNESS READS IN (2026-08-06 ~07:4xZ,
  [results](../posts/2026-08-06-draws-fairness-results.md)) — the
  unfair-penalty signature FIRED on every declared criterion:** E1
  gate passed (draw-0 drift 0.0145 < 0.05); dispersion-quartile
  deficit monotone 0.23→0.60→0.87→1.42 (Spearman +0.13, q4 = 6.2×
  q1); best-of-10 3.8597 is 2.01 BELOW AR's paired 5.8680;
  **energy score (read 4): flow 5.9308 vs AR 8.7696 — flow wins the
  proper score while losing single-draw MAE**. Honest residual:
  deficit positive even in the tight quartile (+0.23), win rate
  < 0.5 everywhere — partly artifact, not wholly; ES is now the
  candidate distributional column (owner decision to adopt).
  **σ_draw direct = 0.02367 SUPERSEDES the 0.0159 pin** (floors
  0.045/0.05 still bind → both live bands numerically unchanged;
  `sigma_draw_direct.py`, cross-estimator inside the χ²₉ band).
- **Golden-ticket noise search (lit slice 2026-08-06,
  [2603.15757](https://arxiv.org/html/2603.15757v1) "You've Got a
  Golden Ticket"):** a *single searched noise vector* (Monte Carlo
  over candidate tickets, weights frozen, inference-only) improved
  38/43 tasks across diffusion/flow policies incl. SmolVLA-LIBERO,
  with gains *growing at fewer solver steps*. Their search needs env
  rollouts; **our panel gives the offline criterion they lack** —
  score M candidate tickets by probe-subset MAE via
  `sample_actions(noise=...)` (the hook already exists), then
  validate the winner on the full panel. Caveats to carry: their
  LIBERO-Spatial cell *regressed* (−3%), tickets showed limited
  cross-task universality, and a fixed ticket makes the policy
  deterministic. Pairs with #12's 1-NFE distill (fewer-steps trend)
  and with mean-of-N (ticket vs mean-of-10 vs both). Cheap eval-side
  screen; needs its own pre-reg before any number is read.
  **Correction hooks (papers-page deep read 2026-08-07,
  [page](../papers/sampling-beyond-selection.md)):** the 38/43 figure
  was v1 — v3 reports 46/51; "Spatial −3%" was imprecise — per-task
  tickets always gain (+13 Spatial), it's the single *shared* ticket
  per suite that regresses (−2.6 to −12). Design note banked: the
  1-NFE student's draw collapse may have shrunk the searchable
  ticket space — screen the teacher's noise space first, or verify
  the student still responds to noise.
- **Golden-ticket screen PRE-REGISTERED (2026-08-07 ~18:0xZ,
  [pre-reg](../posts/2026-08-07-prereg-golden-ticket-screen.md)):**
  teacher-first, M=64 i.i.d. tickets scored in ONE batched draws-64
  eval on the drawsprobe_s7 subset (the "draws" are the tickets — the
  batched-draws merge makes the search ~1.5 GPU-h); frozen null from
  banked data (σ_probe 0.0669 per-draw pooled spread; null min₆₄ =
  mean − 0.157); staged kill line BEFORE the confirmatory full-panel
  read (winner judged on complement rows only, adopt floor −0.05);
  "both" cell = mean-of-top-10-tickets vs banked mean-of-10; free R4:
  per-dataset argmin tickets = the task-locality read the paper
  predicts. Honest prior *against*: a panel-wide ticket is the
  paper's shared-ticket regime (regressed in all 3 LIBERO suites).
  Instrument = a "ticket" noise-key mode in `bijou.eval` (to land
  oracle-gated; 4 oracles frozen in the post). Gate 6 GPU-h; window
  strictly after tsens rungs + behind the selfsubgoal probe.
- **Batched draws MERGED + speedup measured (2026-08-07,
  [main-sync review](../posts/2026-08-07-main-sync-review.md)):**
  the owner's `2ee2be5` integrates all draws in ONE solver call at
  draws×B; same-harness microbench on the leaderboard configs:
  **mean-of-N now costs single-draw latency** — teacher Heun-30
  mean-of-10 single-stream 11,283.6 → **1,245.0 ms/frame (9.1×)**,
  student 1-NFE mean-of-10 277.9 → **111.2 (2.5×)**; draws=1 control
  cells reproduce ≤0.3%. The unconstrained-class caveat on
  mean-of-10 rows is now almost purely about *panel semantics*, not
  deployment cost — the deployment argument for draws is live.
- **Noise-space steering ladder READ (2026-08-07 ~20:2xZ,
  [papers page](../papers/noise-space-steering.md) — DSRL 2506.15799
  + LP-DS 2606.01151 + FRS 2606.13675):** the rung LAFM only named
  is now mapped. DSRL: RL with the noise AS the action (dual critic
  Q^A→Q^W distilled through the frozen decoder, noise aliasing for
  sample efficiency; steered a real-world π₀/DROID checkpoint,
  black-box access only). LP-DS: names the failure mode — 
  unconstrained noise search drifts off the N(0,I) support and the
  frozen decoder answers with mode collapse; fix = state-conditioned
  residual w = ε + Δ_θ(s) inside a Lagrangian trust region (real
  Franka 33/40 vs 18/40 frozen; preserves action entropy where DSRL
  collapses it). FRS: reverse-ODE the flow to recover the noise
  behind a reference action, then DSBC-distill 10 successful
  trajectories into a tiny noise policy (<1 min, ~1 GB, up to +95%
  absolute on real tasks; explicitly inapplicable to AR policies).
  **Consequences banked**: (a) stage 1 is safe by construction
  (i.i.d. prior candidates can't go off-manifold) but any CEM
  escalation pre-reg MUST carry LP-DS's trust-region clause (‖ε‖
  near the √300 ≈ 17.3 typical shell); (b) R4 gains a third
  interpretation — per-dataset argmin structure is the offline
  shadow of what DSRL/LAFM exploit online; (c) the rig-time story
  from the 18:5x owner exchange now has published shapes one and two
  rungs up (DSRL needs rewards + rollouts → gated on #16's rig
  benchmark; DSBC needs 10 reference demos only → banked as a #16
  lever). Whole ladder stays gated on the screen's R1/R2 verdicts —
  no new arm from this read.
