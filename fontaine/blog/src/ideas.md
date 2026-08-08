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
  confirmed, primary stays T=1.0. **Molmo2 arm READ OUT 08-08
  07:2xZ, all expectations met**: Δ_AR −0.154 [CI −0.195, −0.113] —
  the mean-collapse shape replicated on a second AR trunk (AR-100k
  was −0.145); draws10_t1 5.8492/1.9736 → leaderboard row 9, no
  overtake of the flow band; execution oracles byte-green. Decode
  microbench rows measured same window (box H100): greedy
  143.8/678.1, draws10 1191.2/6291.3 ms — row 8's mtime caveat
  retired. Lit (08-08,
  [steering III](papers/noise-space-steering-3.md)): SDN
  (2606.14084) — a judge-free per-step noise selector whose
  smoothness-only ablation carries most of its +18 pp real-robot
  gain; jerk-pick is a pure function of our banked draw stacks →
  record-only ceiling-ladder read EXECUTED same session: flow
  fresh-noise NULL on every diagnostic (agreement at the 10% null,
  −2.3% of the oracle gap; ODE draws uniformly smooth), AR
  real-but-small and T-monotone (5.6/7.5/20.9% of the gap at
  T=0.5/0.7/1.3, Spearman +0.36); never approaches mean-of-N —
  family decodes stand. Molmo2 stack half at the #19 landing.
- **`seam-screen` [#4 Stage-2 attachment seam](ideas/04-stage2-attachment.md)** —
  `screening`. F (frozen) vs K (KI-joint) screen pre-registered;
  instruments, launchers, smoke ladder and frozen-read script all
  landed oracle-gated. Launches at the molmo2 40k endpoint (~08-08).
- **`new-trunks` [#17 New trunks / architectures](ideas/17-new-trunks.md)** —
  standing owner mandate. **Molmo2-4B AR 40k ENDPOINT READ OUT
  2026-08-08 ([results](posts/2026-08-08-molmo2-endpoint-results.md)):
  BEATS — 6.0079/2.1871 vs the E2B own-topology control 7.7966/3.9422,
  paired −1.717 [CI −1.80, −1.63] on 17,204 core frames; frozen
  decision executes, Molmo2 is the phase-2 flow-trunk candidate (the
  #4 attach screen holds this AR-adapted prefix frozen). At 40k it
  sits 0.21 behind AR-100k's greedy at 2.5× fewer steps. Endpoint
  probe 6.2075@40000 = the vu5k amendment's frozen-sanity bar input.** New 2026-08-07
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
  (subgoal-draws selection first) each need a new pre-reg.**
  **Rung (b) PRE-REGISTERED 2026-08-08
  ([pre-reg](posts/2026-08-08-prereg-subgoal-draws.md)): sample 9
  subgoal candidates (greedy + 8 at T=1), condition on the
  self-certainty pick (frozen scorer,
  [Self-Certainty page](papers/self-certainty.md), 2502.18581) —
  plus a record-only oracle-similarity CEILING arm that bounds
  every scorer at this width and adjudicates no-diversity vs
  no-scorer if the falsifier fires; gate ≤ 6 GPU-h; execution
  queued behind the goldenticket R1 chain. Instrument LANDED
  oracle-green 08-08 03:5xZ (draws mode + SC-sufficient stats dump
  + both selection arms + read script w/ 11 abort branches;
  check.py 489) — only the GPU-side preflight oracles remain
  before launch.** New 2026-08-07 (radar, both papers
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
  New 2026-08-08 ~04:1xZ (targeted slice BEFORE the rung-(b) read —
  [progress-from-logits page](papers/progress-from-logits.md),
  TOPReward 2602.19313 + ProgVLA 2605.28231): escalation routing
  pre-mapped for a no-scorer verdict — (1) masked-contrast
  prerequisite VERIFIED MET (MG-Select masks text/state, never
  frames; our subgoal-masked reference = the planner-less path,
  trained at 50% dropout — correction banked on the
  [self-certainty page](papers/self-certainty.md)); (2)
  history-conditioned planning is evidence-backed (TOPReward: phase
  zero-shot recoverable from a video prefix via one completion
  logit, incl. on Molmo2-8B; single frames are the measured rung-(a)
  bottleneck). Lane (a) verdict: nothing published beats SC
  label-free on open-ended text — the frozen scorer cell stands.
  New 2026-08-08 ~19:0xZ (slice while rung-2 stage-2 decoded —
  [RoVer page](papers/rover-learned-verifier.md), 2510.10975): the
  "scorer is the gap" escalation now has a priced recipe — a 0.2B
  PRM (40M trainable) trained offline from demos alone via
  anchor-centered preference pairs, no environment or success
  labels; its stated chunk–step mismatch limitation (gains go
  unstable on chunked policies — all of ours are) is pre-registered
  ammunition: any learned-verifier arm must score the chunk as the
  unit. ELASTIC (same slice) independently names verifier noise as
  THE parallel-scaling bottleneck — the Δ_ceil/Δ_bon split is
  built to price exactly that.
- **`noise-draws` [#1 Noise-draw ensembling](ideas/01-noise-draw-ensembling.md)** —
  flow mean-of-10 banked (5.365); batched draws merged 2026-08-07.
  **GOLDEN-TICKET SCREEN R1+R2 READ OUT 2026-08-08
  ([results](posts/2026-08-08-goldenticket-results.md)): tickets are
  REAL — R1 CONFIRM (sd 0.823 vs null line 0.0785, 12× the null;
  winner ticket 33), R2 REAL on 14,746 complement rows (paired
  −0.924 [CI −0.985, −0.866] vs stable-key; LARGER than the
  selection-biased probe delta): one fixed sha-pinned noise vector =
  5.6468/1.8963 core-pooled, ~75% of the mean-of-10 gain at 1/10th
  the draws — leaderboard row 7. Effect DIRECTIONAL, not norm
  (ticket-33 norm rank 29/64, corr(norm,score) −0.05) — the
  LAFM/DSRL structured-noise premise showing up unprompted.
  **SCREEN CLOSED 08-08 08:2xZ — R3 INTERESTING, 9× the band**:
  mean-of-top-10 **5.1847/1.3831** vs banked mean-of-10
  5.3645/1.4242 (Δ −0.180, record-only; best chunk AND first
  numbers measured on this panel — row-seating needs the paired
  follow-up now folded into the queued noise-ladder pre-reg). R4a:
  ticket 33 argmin in 4.4% of 792 probe datasets (top-10
  containment 29.8% ≈ 2× null; median cell 2 frames —
  selection-noise caveat). R4b: winner gain monotone in draw
  dispersion (−0.35 → −1.44 by quartile). Screen total ~5.55/6
  GPU-h. The
  **Rung-2 pre-reg FINALIZED 08-08 13:2xZ
  ([pre-reg](posts/2026-08-08-prereg-noise-ladder-perdataset.md)):
  stage 0 found a thin floor F=6 on banked data (split-half regret
  n=6 bin 2% under the permutation null; n=4–5 fail — the
  median-2-frame caveat was right), 97 qualifying datasets = 40.8%
  of panel core rows, 88/97 route away from ticket 33 (map sha
  15d92935…).** **Stage-2 READ OUT 08-08 19:4xZ — FALSIFIED
  ([results](posts/2026-08-08-noiseladder-rung2-results.md)):
  Δ_route +0.129 [CI95 +0.060, +0.205] entirely ABOVE zero on the
  6,014 held-out complement core rows (win table 34W/54L, sign
  p 0.042) — the in-sample −0.60 probe delta INVERTED out-of-sample;
  per-dataset argmin on ~6–20-frame cells memorizes its cell even
  past the F=6 permutation floor. Ticket 33 itself re-confirmed
  (routed-vs-stablekey −0.756; board row stays global t33). Rung
  CLOSED; measured prior inherited by every ladder rung above:
  specialization must prove held-out-row transfer AT SELECTION
  TIME. Record-only lead: routing wins chunk steps ~1–8, loses
  ~15+ — a chunk-position noise policy is a different, cheaper
  axis (no arm without its own pre-reg). Seating arm independent,
  in flight.**
  noise-structure ladder (per-dataset tickets → LAFM priors → DSRL
  state-conditioned) has met its entry condition — each rung needs
  its own pre-reg.** Lit
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
  the numbers-backed default, no gate change. Lit (08-08,
  [steering III](papers/noise-space-steering-3.md)): the
  per-dataset-tickets pre-reg inherits three published priors —
  2603.11642's variance decomposition (noise main effect 1.4%,
  context×noise interaction 39.4%, best-shared-noise optimal in
  3.1% of contexts: per-dataset search aims at the interaction
  term); the channel exists because our Heun decode is path-intact
  (DDIM 0.96 vs DDPM 0.11 direction→jerk correlation — any sampler
  change re-tests the ladder); and chunk-boundary artifact is a
  named panel-blind unknown of ticket 33 (rollout-gated jerk read
  banked). Lit (08-08 ~19:0xZ,
  [ELASTIC page](papers/elastic-adaptive-compute.md), 2606.31132):
  a rung-3 candidate named — **dispersion-gated draw allocation**
  (full draws budget only where the banked R4b dispersion quartile
  says ensembling pays, 1 draw elsewhere); ELASTIC learns this
  allocation with per-task online RL and matches best-of-10 at 34%
  lower latency — our version is a zero-training offline re-read of
  banked dumps, gated on the rung-2 verdicts. Directed candidate
  expansion (RoVer, same slice) lands in noise space for us —
  prior art alongside LAFM/DSRL, not a new rung.

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
  New 2026-08-08: **owner-steered molmo2 perf/memory deep review
  SHIPPED** ([review](posts/2026-08-08-molmo2-perf-review.md)) —
  suffix attention on the MATH backend (13×/layer measured), ViT
  eager einsum (13×/block), act-ckpt absent from live launchers;
  S-bundle queued (`molmo2-perf-fix-prereg`, ~8–15% step expected).
  Same-session lit ([loss + mask page](papers/memory-efficient-loss-attention.md)):
  CCE (2411.09009) banked as the CE escalation ladder
  (valid-row → two-segment lse → CCE; entry = wanting
  backward-chunks < 6 or batch > 12), FlexAttention banked as the
  dense-mask successor gated on compile (#2b) or long prefixes.

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
