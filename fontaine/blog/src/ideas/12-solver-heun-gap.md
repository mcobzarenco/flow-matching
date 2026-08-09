# 12. Solver/Heun-gap work — `screening` (1-NFE student BANKED 2026-08-06; rig-ft diagnosis branch — next rung waits on rig data, #16)

*Tag: `one-step` · idea #12 · [index](../ideas.md)*

The h1536 adaRMS Heun-gap collapse did NOT transfer to h1024-on-AR-trunk
(measured −0.28 at 10→30, first_mae −0.46): sampler quality is back on
the table for the best flow lineage. Arms: step-count sweeps, solver
variants, consistency/distillation toward 1–2-step deployment decodes
(the distillation leg pairs with idea 1).

- **Scoring note (2026-08-05 paired analysis):** flow's deficit is
  ~all late-horizon (crossover step 2, monotone to +1.2 @40 —
  [post](../posts/2026-08-05-flow-vs-ar-paired.md)). Score solver arms
  per-step: a solver fixing only late-horizon costs nothing at
  first_mae; pooled-only scoring would misread it.
- **Literature (2026-08-05 slice): SnapFlow (arXiv:2604.05656) is
  the distillation-leg recipe to try first.** Plug-and-play
  SELF-distillation for flow-matching VLAs — no external teacher:
  mixes standard flow-matching samples with consistency samples
  whose targets are two-step Euler shortcut velocities from the
  model's own marginal predictions; zero-init target-time embedding
  switches velocity-estimation vs one-step modes in one network.
  Claimed: ~12 h on ONE GPU, no arch changes; π0.5-3B 1-NFE matches
  the 10-step teacher (98.75% vs 97.75% LIBERO, 274→83 ms); **tested
  on SmolVLA-500M too** (−8.3% MSE, 3.56× e2e) — the closest
  external analogue to our trunk+flow-expert protocol. Cheapest
  falsification here: distill flow-80k, score the panel at 1-NFE vs
  Heun-30 (band: within the σ_draw noise floor of 6.6232). Also
  pairs with #1 (a distilled 1-step model makes mean-of-N nearly
  free).
- **Literature (2026-08-06 slice, ~10:0xZ): the one-step fallback/
  follow-on menu, banked while the SnapFlow run climbs to its 10k
  probe.** If the @30k endpoint misses its band (or to extend a hit),
  three distinct objective families now have external evidence:
  (1) **One-Step Flow Policy** ([2603 era, self-distillation
  w/o pretrained teacher](https://zhaoyang97.github.io/daily-arxiv/2026-03-12/one-step-flow-policy/)) —
  self-consistency loss + self-guided regularization + warm start,
  71.6% avg on 56 sim manipulation tasks at 1-NFE; nearest
  competitor recipe to SnapFlow's. (2) **MeanFlow-based one-step
  VLA** ([arXiv:2603.01469](https://arxiv.org/abs/2603.01469)) —
  average-velocity (MeanFlow) objective, claims to *eliminate the
  consistency constraint* entirely (the constraint whose s=t
  divergence we are currently watching drift); 8.7× vs SmolVLA.
  A MeanFlow arm would be a genuinely different objective, not a
  SnapFlow re-tune — the right shape for a paired follow-up if
  consistency-style distillation is what misses. (3) **"Let It Be
  Simple"** ([arXiv:2606.05737](https://arxiv.org/abs/2606.05737)) —
  claims VLA is image-to-text-like (strong conditioning), so
  HIGH-NOISE TRAINING alone yields one-step decoding (95.6%
  LIBERO-Long, no distillation stage at all); their "irreducible
  velocity loss" framing + the ablation note (weakening the
  condition erases the one-step gain) ties directly to our
  conditioning stack (#11/Q3). Cheapest local probe of (3): score
  the TEACHER at 1-NFE (zero training — we may already have this
  number from the @10k/endpoint probe protocol runs) and read how
  much of the gap distillation actually closed vs what high-noise
  fine-tuning would have to. No new launch implied; feeds the
  SnapFlow results post's discussion + the next pre-reg if the
  endpoint branch fires.
  **Correction hooks (papers-page deep read 2026-08-07,
  [page](../papers/one-step-menu.md)):** MeanFlow-VLA's 8.7× is
  speed-for-accuracy (78% vs SmolVLA's 84.5% avg, loses 2/3 tasks;
  NFE=1 config-sensitive down to 49% in their own sweep); Let It Be
  Simple's one-step win is chiefly **state**-carried (no-state
  ablation ~0% everywhere) and its α=4 schedule *degrades* 10-step
  decoding to 63.4% — a specialization, not a free win; its
  small-irreducible-loss theory retroactively explains our
  student's draw collapse, and teacher-at-1-NFE stays the free
  schedule-vs-distillation decomposition read.
- **PRE-REGISTERED (2026-08-06 ~00:3xZ,
  [pre-reg](../posts/2026-08-06-prereg-snapflow-distill.md)):** SnapFlow
  self-distill of flow-80k — full recipe deep-read and frozen
  (α=0.5/λ=0.1 mix, sg two-step-Euler shortcut targets, zero-init
  φ_s target-time embedding, 30k steps LR 2.5e-5 cosine, trunk
  frozen, ~12–20 h 1×H100). Primary: full panel at 1-NFE vs 6.6232
  (+max(3σ_draw, 0.15) band, σ_draw by finalization amendment from
  draws runs 3–5); deployment headline: mean-of-10@1-NFE vs the AR
  anchor 5.8026 at ~one-Heun-5-draw cost. Fills the local-GPU queue
  slot after the draws chain + fairness probe; pre-launch impl
  checklist (φ_s, `--distill snapflow`, 1-NFE eval switch, oracles)
  = CPU work items.
- **IMPLEMENTATION COMPLETE (2026-08-06 ~00:3x–01:0xZ session): all
  five pre-launch checklist items landed; the launch path is
  zero-CPU.** φ_s config-flagged + checkpoint-compat (sanctioned
  additive warm start in the --init-from guard/loader);
  `--distill snapflow` (α/λ frozen in code, mean- and sum-form so
  chunked backward stays available); `bijou.eval --target-time
  {t,zero}` loud 1-NFE switch recorded through report/npz/banner; 10
  new oracles (validation gate (a) also RUN on the real checkpoint:
  6/6 forwards bit-exact, PASSED); launcher staged + recipe
  diff-verified through the real parse_args (50 teacher fields
  verbatim, 11 pre-registered deltas), chains gates (a)+(b) then
  training then the endpoint 1-NFE panels (1/5/10 draws). Gate (b)
  drift eval + @10k probe script wait on GPU only. check.py 201.
- **σ_draw FINALIZED (2026-08-06 ~05:5xZ,
  [amendment](../posts/2026-08-06-sigma-draw-finalization.md)):**
  σ_draw = 0.0159 from the chain's pooled mean-of-N curve
  (gaussian-bias family, held-out N=5 error 0.087%; CPU-only,
  oracled) — 3σ = 0.048 < 0.15, **the floor binds: endpoint
  adopt-signal iff 1-NFE chunk_mae ≤ 6.7732.** Verdict
  family-independent (even the a-priori-max pure-noise reading gives
  0.040 < 0.045). Fairness-probe direct measurement supersedes if
  larger. The launch's last CPU-side blocker is closed — SnapFlow
  waits only on a quiet local GPU. **Direct measurement IN
  (2026-08-06 ~07:4xZ): σ_draw = 0.02367 supersedes the pin; 3σ =
  0.071 < 0.15 → the floor still binds, adopt band ≤ 6.7732
  UNCHANGED.** SnapFlow queues behind the #18.2 flip eval
  (~09:2xZ boundary) on the local GPU.
- **RESULTS INSTRUMENT BANKED BEFORE THE DATA (2026-08-06 ~09:3xZ,
  `4d48120`):** `fontaine/scripts/snapflow_results.py` — every frozen
  read (probe kill 9.6755, adopt ≤ 6.7732, falsify > 7.1232, edge
  ≤ 1.9831, deploy ≤ 5.8026, per-step horizon read + v2 column),
  oracles (a)–(e) green on banked data only, strict semantics guards
  (1-NFE/euler/target_time/draws/index-keying/full-panel all
  asserted). **Gap caught by banking early: the chained stage-4
  endpoint evals dump no npz** → per-step read had no data source;
  addendum `eval_snapdistill_endpoint_1nfe_npz.sh` staged (noise-key
  pinned `index` explicitly). Standing hold: the `--noise-key`
  default stays `index` until the chain's endpoint evals run at 30k.
  **Hold RELEASED — default flipped to `stable` 2026-08-06 ~15:5xZ
  (#18.2 follow-on, see item 2 in the deep-dive list).**
- **Pointer reads closed (lit slice 2026-08-06):** OFP
  ([2603.12480](https://arxiv.org/abs/2603.12480)) — *from-scratch*
  one-step self-distillation (self-consistency + self-guided
  regularization + warm-start from temporal action correlations);
  π0.5 one-step beats the 10-step teacher on RoboTwin 2.0 — the
  reserve recipe if SnapFlow misses expectation 2, and its
  warm-start trick touches #1's noise structure. GoldenStart
  (2603.14245) — Q-guided VAE priors + entropy control for *online
  RL* distillation — screened out (needs Q-functions/rollouts; not
  our offline setting). Golden Ticket noise search banked in #1.
- **RESULTS (2026-08-06 15:2xZ,
  [post](../posts/2026-08-06-snapflow-results.md)): the 1-NFE student
  HOLDS the panel** — single draw **5.6036 / 1.7039 at ONE expert
  eval** (beats teacher Heun-30 6.6232 AND the AR anchor 5.8026);
  mean-of-10@1-NFE 5.3675/1.5927 edges the teacher's mean-of-10
  (5.365) at 1/30 the compute. Adopt signal fired (≤ 6.7732). v1
  panel / index keying as registered (record-only, stated not
  hidden). This row is the deployment-latency headline the
  leaderboard's compute column measures.
- **Rig fine-tune @4k (2026-08-06 ~18:1xZ,
  [diagnosis](../posts/2026-08-06-ftrig-diagnosis.md)): ship rule
  fired the DIAGNOSIS branch — no upload.** Rig-holdout draws1 got
  worse (+0.09/+0.04); per the owner steer the next rung waits on a
  better rig dataset (#16). The student itself stays banked.
- **A fourth pole on the one-step axis banked (2026-08-09,
  [async execution II](../papers/async-execution-2.md), FASTER
  2603.19199): one-step for the *head* of the chunk, many-step for
  the tail.** A horizon-aware timestep schedule (per-action hit
  times, mixed-schedule fine-tune, no architecture change) finalizes
  action 0 after one flow step of N and streams it while the tail
  keeps refining — TTFA 1.29–3.09× on π0.5/X-VLA. Orthogonal to
  SnapFlow's distill-everything (which we banked); relevant iff we
  ever run multi-step decode for quality at deployment — mean-of-10
  batched draws is exactly that case (see #22's re-ranked arm menu).
  Record-only; no arm here while the 1-NFE student holds the panel.
- **Second production data point for 1-NFE (2026-08-09,
  [RDT2 page](../papers/rdt2-umi-scaling.md)):** 5-step flow →
  1-step distillation with on-the-fly teacher targets holds at 7B /
  10k-hours scale ("UltraFast", fastest inference in their fleet at
  2× π0.5's size, +97 ms button-press reaction vs human). Same
  adopted-signal shape as our SnapFlow row.
