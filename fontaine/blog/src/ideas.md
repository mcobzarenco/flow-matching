# Ideas

The backlog, one page per idea (sidebar, or the index below). Every
idea page carries: hypothesis, expected effect, cost, cheapest
falsification, and the dated record of everything that has happened
to it since. Seeded 2026-08-05 from charter §8 (which distills the
mainline ledger, `docs/architecture.md` §7–8). Status tags: `queued`
/ `screening` / `running` / `confirmed` / `falsified` / `parked`.

This page is the **index**: what is hot right now vs what is on ice.
It is updated whenever an idea moves (the per-idea page is the
record; the line here is the hook). *Index last updated 2026-08-07.*

## Hot — actively pursued

- **`ar-draws` [#19 AR sampled-draws eval](ideas/19-ar-sampled-draws.md)** —
  `screening`. The AR side of the draws fairness programme.
  draws10_t1 read out 2026-08-07: all three pre-registered
  expectations met (Δ_AR −0.145, ~9× smaller than the flow gain —
  the mean-collapse shape). T-sensitivity dT table banked 23:09Z
  same day (record-only): monotone in T — 6.50/6.57/6.78/7.18 at
  T=0.5/0.7/1.0/1.3 on the q4 rows; the T=1.3 asymmetry prior
  confirmed, primary stays T=1.0. Next: the molmo2 arm at its
  endpoint (~04–05Z 08-08).
- **`seam-screen` [#4 Stage-2 attachment seam](ideas/04-stage2-attachment.md)** —
  `screening`. F (frozen) vs K (KI-joint) screen pre-registered;
  instruments, launchers, smoke ladder and frozen-read script all
  landed oracle-gated. Launches at the molmo2 40k endpoint (~08-08).
- **`new-trunks` [#17 New trunks / architectures](ideas/17-new-trunks.md)** —
  standing owner mandate; the Molmo2-4B AR 40k trunk run is LIVE on
  the box (endpoint ~08-08), K1 gate crossed green. New 2026-08-07
  (owner question): the SigLIP-unfreeze evidence got its own slice
  ([vision-encoder-freeze](papers/vision-encoder-freeze.md) — both
  poles are real: unfreeze wins adaptation regimes, freeze/anchor
  wins OOD retention; our rung sits in the adaptation regime; recipe
  prior = full-FT vision at low LR, never LoRA-on-SigLIP);
  vision-unfreeze pre-reg
  [DRAFT](posts/2026-08-07-prereg-molmo2-vision-unfreeze.md) now at
  the owner-agreed amendment-3 design (warm-start two-arm vu5k:
  frozen-continue vs thawed-continue from the 40k endpoint, 5k
  steps, 0.3× LR reheat, vision LR 6e-6 tied to text; superseded
  10k-from-scratch form recorded in its §8); finalization PREP
  landed 08-07 (`485194b`: byte-audit clean, both arm launchers +
  prepared babysit entries) — execution is launch-only after the
  150-step memory smoke + amendment post + owner go, window
  post-attach-screen.
- **`aux-subgoals` [#6 Aux attribution](ideas/06-aux-attribution.md)** —
  `confirmed` (aux HELPS actions, +0.462 cost when off). **Rung (a)
  self-subgoal probe READ OUT 2026-08-08
  ([results](posts/2026-08-08-selfsubgoal-results.md)): the slot is
  ALIVE — Δ_oracle −0.290 [CI −0.331, −0.225], 6× late-horizon,
  twice the AR draws-10 gain — but self-generated subgoals recover
  almost none of it (Δ_self −0.018, CI spans 0; no deployment win at
  3× decode cost). Channel read significant: same text via suffix
  is +0.043 worse than the slot — generation quality (phase
  estimation), not the channel, is the bottleneck. Escalations
  (subgoal-draws selection first) each need a new pre-reg.** New 2026-08-07 (radar, both papers
  announced same day —
  [subgoal-sourcing page](papers/subgoal-sourcing-post-training.md)):
  two fresh directional priors before the read — HiRoC's
  subgoal-source cold start (Δ_self ≤ Δ_oracle expected; its
  alignment-SFT is a new named escalation) and VLA-Talker's
  inject-vs-supervise 15.9-pt gap (predicts the narrated arm is
  safe; tension with our aux-on result recorded + resolved on the
  page). New 2026-08-08 (slice while the arms decoded —
  [runtime-plan-verification page](papers/runtime-plan-verification.md)):
  the escalation ladder above rung (a) priced with published
  numbers — SV-VLA's cheap-gate-heavy-replan (verification without
  recovery crashes 90.9%→15.5%), VINE's subgoal-draws width
  scaling (peak at K=4), Do-What-You-Say's faithfulness gap (the
  execution-side noise our Δ_oracle/Δ_self split doesn't price).
- **`noise-draws` [#1 Noise-draw ensembling](ideas/01-noise-draw-ensembling.md)** —
  flow mean-of-10 banked (5.365, beats the AR anchor on both
  columns); fairness + energy-score reads in; batched draws merged
  2026-08-07 — mean-of-N at single-draw latency (teacher 9.1×,
  student 2.5× single-stream). Next rung: the golden-ticket noise
  screen —
  [pre-reg posted 2026-08-07](posts/2026-08-07-prereg-golden-ticket-screen.md)
  (teacher-first, M=64 tickets in one batched draws-64 probe eval,
  staged kill line, 6 GPU-h gate); **instrument landed 08-07
  (`0acabde`, oracles green)** — execution is launch-only, awaits a
  quiet local window behind tsens + the selfsubgoal probe. Lit
  (08-07, [LAFM page](papers/latent-action-priors.md)): the
  noise-structure ladder above the screen is now mapped — searched
  ticket → per-dataset tickets → LAFM's learned mode-prior library
  (2606.23420, training-time; +10.4 LIBERO-90 over FM at 110M) →
  state-conditioned noise (DSRL + 2026 kin, now read —
  [noise-space-steering page](papers/noise-space-steering.md)):
  DSRL's dual-critic RL-on-noise, LP-DS's off-manifold drift
  diagnosis (trust-region clause banked for any CEM escalation;
  ‖ε‖ ≈ √300 shell), FRS's reverse-ODE noise recovery + 10-demo
  DSBC distillation (also a #16 rig lever). R4 per-dataset argmin
  disagreement would be LAFM's "fragmented action space" showing up
  in our data; the whole ladder stays gated on stage-1 R1/R2. Both
  banked hooks closed same day
  ([part II](papers/noise-space-steering-2.md) — PAINT + UniSteer):
  a probeable prefix-locality property of our teacher noted
  (record-only), the per-step fixed-point inversion primitive is
  the numbers-backed default, no gate change.

## Standing

- **`rig-benchmark` [#16 Few-shot rig-transfer benchmark](ideas/16-rig-transfer-benchmark.md)**
  — **the north star**; execution parked by owner (better rig data
  later), instruments banked. Short-term proxy: comm-holdout MAE +
  attribution. New 2026-08-07: the proxy itself got a lit slice
  ([offline-validation](papers/offline-validation.md) — raw MSE
  measured at ρ −0.61 vs rollout success, sign flips exist);
  critical-frame re-pooling rung **executed same-day — every
  published ranking holds on the critical pool, separation widens**
  ([results](posts/2026-08-07-prereg-critical-frame-repooling.md)).
  Rig-time menu now four deep
  ([noise-steering II](papers/noise-space-steering-2.md)): ticket →
  DSBC (10 demos) → UniSteer (teleop corrections→noise, 20%→90% in
  ~66 min, OOD-robust where DSRL isn't) → DSRL (rewards).
- **`lit-arms` [#15 Literature-sourced arms](ideas/15-literature-arms.md)** —
  the arXiv radar; every borrowed idea cites its source, every
  "novel" idea gets a search first. Feeds the Papers section.
- **`infra-hardening` [#18 Instrument & infra hardening](ideas/18-infra-hardening.md)**
  — the bijou deep-dive fix queue + everything oracle-shaped;
  several items done, rest queued by leverage. New 2026-08-07: item
  9 async checkpoint saves LANDED (owner HIGH; byte-identical
  oracle, ~14% wall-time payoff at the attach screen) + its
  [checkpointing-systems lit page](papers/checkpointing-systems.md).

## On ice — queued or parked, each with its named trigger

- **`throughput-compile` [#2 Throughput: bucketing + compile](ideas/02-throughput-bucketing-compile.md)**
  — 2a landed; GPU A/B conditional on a widened-selection corpus
  (padding ceiling too small under the current recipe).
- **`longer-training` [#3 Longer training](ideas/03-longer-training.md)** — needs the
  own-baseline reference arm first.
- **`tokenizer-v3` [#5 FAST tokenizer v3](ideas/05-fast-tokenizer-v3.md)** — CPU
  refit on curated-v0 quantiles; token metrics reset; entropy/
  utilization gate before any learned-VQ arm.
- **`stream-schedule` [#7 Stream-schedule re-test](ideas/07-stream-schedule.md)** —
  enters at the short-run screen rung.
- **`vocab-head` [#8 Shortlist/output-vocab head](ideas/08-shortlist-vocab-head.md)**
  — VRAM lever for ar_backbone; design concretized, unbuilt.
- **`data-levers` [#9 Data levers](ideas/09-data-levers.md)** — state-dropout arm
  C answered "adopt nothing"; p=0.3 branch survives on our own
  branch rule only; calibrated-noise/GAP are the literature levers.
- **`base-vs-it` [#10 E2B base-vs-IT swap](ideas/10-e2b-base-vs-it.md)** —
  backbone-swap arm, pre-registered prediction ±0.2.
- **`visual-grounding` [#11 Visual grounding arms](ideas/11-visual-grounding.md)** —
  the open front; arch batch #1 pre-registered, arm A (img280) HELD
  for a fresh owner go.
- **`one-step` [#12 Solver/Heun-gap + 1-NFE distill](ideas/12-solver-heun-gap.md)**
  — SnapFlow 1-NFE student banked (holds the panel, single draw
  beats AR); rig fine-tune diagnosed, next rung opens with rig data
  (#16).
- **`sign-convention` [#13 Sign-convention repair](ideas/13-sign-convention.md)** —
  stage 2 hit the escalation branch (3/4 reference populations not
  sign-consistent); parked pending a decision on the reference set.
- **`async-staleness` [#22 Async staleness bridging](ideas/22-async-staleness.md)** —
  RTC-class rollout question; parked, waits on #16 (closed-loop by
  construction). New 2026-08-07: PAINT (2606.19774,
  [noise-steering II page](papers/noise-space-steering-2.md)) —
  training-free initial-noise selection matches RTC on a chunk-50
  π₀ with no gradients; **arm order re-banked: PAINT → A2C2 →
  TT-RTC**.

## Answered — banked results

- **`wrap-census` [#14 ±180° wraparound census](ideas/14-wraparound-census.md)** —
  measured: 1.24% of pooled panel MAE; under the gate, banked.
- **`activation-ckpt` [#20 Activation checkpointing](ideas/20-activation-checkpointing.md)**
  — landed + oracle-gated; the GPU ladder lives on as #4's K smoke
  item.
- **`loop-review` [#21 Agentic-loop deep review](ideas/21-agentic-loop-review.md)**
  — CLOSED: P1–P7 all landed, owner-signed.
