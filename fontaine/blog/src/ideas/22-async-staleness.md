# 22. Async staleness bridging for rollout (RTC / A2C2 / TT-RTC) — `parked` (waits on #16)

*Tag: `async-staleness` · idea #22 · [index](../ideas.md)*

- **Hypothesis:** at real deployment latencies, naive async chunk
  switching (our `--async-inference`) loses task quality to
  observation staleness — and the loss is *decode-dependent*:
  single-draw 1-NFE (~2 ticks @30 Hz) is in the survey's "everything
  works" regime, mean-of-10 batched (576 ms ≈ 18 ticks, chunk 50) is
  deep in the regime where naive switching degrades and
  inference-time RTC has already collapsed
  ([papers page](../papers/2026-08-07-async-chunk-execution.md):
  RTC 2506.07339 + async-methods comparison 2605.08168).
- **Expected effect:** rig/sim rollout quality at high-value decodes;
  invisible to the offline panel by construction (staleness is a
  closed-loop phenomenon).
- **Cost:** screen = 0 training (measure naive-switch cost at both
  decodes on rig rollouts once #16 exists); first arm if real =
  A2C2-style residual correction head (frozen base, composes with
  batched draws), second = TT-RTC prefix-conditioning fine-tune
  (~25% of base training, weak at chunk 50).
- **Falsification:** paired rollouts, same checkpoint, single-draw vs
  mean-of-10 under naive switching: if mean-of-10's closed-loop win
  survives its 9× staleness, bridging buys nothing — park forever.
- **Gate:** parked until the #16 rig-transfer bench exists; the
  survey's regime table (delay-in-ticks × chunk length decides the
  winner) is the design input for any pre-reg here.

## Record

- **2026-08-07 ~20:3xZ — PAINT read, arm order re-banked**
  ([noise-space steering II](../papers/noise-space-steering-2.md),
  2606.19774): training-free initial-noise selection solves the
  chunk-boundary problem without gradients — backward-Euler invert a
  target endpoint (executed prefix + draft tail) to noise, keep only
  the inverted *prefix* of ε, splice fresh suffix noise, integrate
  forward (~3N calls). Matches/beats RTC on real tasks (0.85 vs
  0.75 Toy-in-Drawer), most delay-robust method on Kinetix at d=4,
  composes with TT-RTC, demonstrated on a chunk-50 π₀ — our regime.
  **Arm order now: PAINT (zero training) → A2C2 residual → TT-RTC.**
  Design note banked (ours, needs its own oracle if ever built):
  shared inverted prefix + per-draw fresh suffixes should let PAINT
  compose with mean-of-10 batched draws. Caveats: locality
  assumption (prefix-of-ε ↔ prefix-of-chunk, OT-FM-encouraged, not
  enforced — probeable with our draws machinery, noted on #1);
  off-manifold executed prefixes invert poorly. Gate unchanged:
  parked until #16's rig bench exists.

- **2026-08-09 ~12:0xZ — async family deep-read, arms re-banked
  again** ([async execution II](../papers/async-execution-2.md):
  FASTER 2603.19199 + ABPolicy 2602.23901 + DEFLECT 2605.19294).
  Three findings move this idea: (1) **FASTER** shows the delay
  itself is partly a scheduling artifact — its horizon-aware
  schedule finalizes action 0 after one flow step of N (hit-times
  u_i per action index, mixed-schedule fine-tune, no architecture
  change) + streams actions as they finalize; TTFA 1.29–3.09× on
  π0.5/X-VLA. The schedule is per-action-index so it tiles across
  our draws-major batch — the 18-tick mean-of-10 staleness this
  idea's hypothesis rests on could drop toward ~2–4 ticks *before
  any bridging is bought*. (2) **DEFLECT** hard-measures the survey's
  regime prediction — RTC/BID ≤5% at d≥5 on chunked VLAs — and
  post-trains through it with stale-vs-fresh preference pairs from a
  frozen reference (FM-DPO + SFT anchor, both chunks scored under
  the *stale deployment input*); headline +6.4 pp at d=5–7 but the
  restart-corrected net is **+1.6–2.3 pp** (their own Appendix L) —
  carry that number. Delay generalizes (train d≤2 → +3.7 pp at
  d=7); flow heads only. (3) **ABPolicy**'s
  continuity-constrained refitting + jerk instruments (95th-pct
  accel, zero-crossings) banked as boundary-seam tooling for the
  eventual rig screen; its within-chunk win is predicted small for
  us (our ODE draws already uniformly smooth per the SDN read).
  **Arm order now: measure naive-switch cost → HAS-on-decode
  (fine-tune) → PAINT → A2C2 residual → TT-RTC/DEFLECT-class
  post-training.** Caveat carried loud: nobody tests d≈18; d=7 is
  the field's ceiling. Gate unchanged: parked until #16's rig bench
  exists.

- **2026-08-09 ~14:3xZ — SEAM read (the 12:3xZ hook cleared;
  [page](../papers/seam-boundary-steering.md), 2607.04609): the
  cheapest entry in the bridging family, and a free measurement for
  us.** Training-free inference-time steering: after each Euler
  step, a closed-form nudge pulls the new chunk's first M positions
  toward the previous chunk's unexecuted tail ("aligned prior"),
  scaled λ(1−t) — no backprop, +1% denoise cost. π0.5/LIBERO-10:
  boundary jerk −28%, discontinuity −27%, success 94.8→95.7 (vs
  RTC −54% at 1.22×; vs ACT-TE −84% but success −12 pts —
  over-smoothing kills contact timing). **Arm order update: SEAM
  slots ahead of PAINT as the cheapest smoothing arm** (PAINT
  inverts 3N calls; SEAM is closed-form) — but PAINT stays the
  async-robust one (SEAM assumes the tail is available at sampling
  time; under async overlap the prior goes stale, unmeasured in the
  paper). λ ablation caution banked: 0.15–0.2 erodes success —
  guidance strength is not free. **Banked free hook (feeds #1
  too): the boundary-incompatibility CPU read** — our panel npz
  stacks hold predicted chunks on temporally ordered frames; tail
  -vs-head disagreement on the overlap of frames Δt apart is a pure
  function of banked data (the SDN-read pattern, zero GPU). A null
  (our chunks already agree at the seam) would close the whole
  bridging direction for our stack before any rig work. Gate
  unchanged: parked until #16; the CPU read needs no gate.

- **2026-08-09 ~15:2xZ — the boundary-incompatibility read EXECUTED:
  NOT a null**
  ([results](../posts/2026-08-09-boundary-incompat-results.md),
  `boundary_incompat_results.py`, oracle-gated; 13,693 same-episode
  pairs across five banked full-panel stacks, truth overlaps
  byte-identical on every pair). Seam disagreement D ≈ 1.1–1.27× each
  model's own error on the same overlap; boundary jump 11–14× typical
  per-step motion while chunks stay smooth inside (the SDN
  within-chunk null replicated — smooth within, jerky between). The
  dt→0 intercept decomposes the cause: fresh-noise flow 6.04 vs
  deterministic AR greedy ~2.7 vs **one shared noise ticket 2.07** —
  noise coupling deletes the entire noise-induced seam term (the
  ticket33 bank is an accidental ablation the GoldenTicket rung
  already paid for) and lands below greedy AR. **The direction this
  read could have closed is instead confirmed with a measured
  target.** Arm order unchanged; still parked on #16 (open-loop
  read prices the problem, must never validate a fix). Escalation
  (SEAM/PAINT arm or a cross-chunk noise-coupling deployment
  policy) needs its own pre-reg.

**2026-08-09 — lit `0812b`: the horizon itself can be an output
([VLA-Corrector page](../papers/vla-corrector.md), 2607.01804):**
event-triggered chunk truncation from a cheap (40M, ~ms) drift
monitor — long chunks while the scene tracks the plan, short
corrective replans when it drifts. The ablation that matters:
truncation-only is +11.65 of the +15.65 pp total (MetaWorld π0.5) —
*when to cut* dominates *how to steer*. Complementary to our
boundary read (they cut on scene drift; our measured pathology is
decode drift at the seam). Menu adjacency, not an arm — closed-loop
by construction, parked on #16 like the rest of the ladder.

**2026-08-09 — lit `0813`: two placements, both leaving the arm menu
unchanged.** ([AsyncVLA page](../papers/asyncvla.md), 2511.14148):
despite the name, NOT an async-execution paper — all correction is
pre-execution within one chunk (two-pass masked regeneration; no
staleness, truncation, or horizon ablation anywhere); filed so the
title doesn't get re-banked as a hook. Its rater is a candidate
when-to-cut signal nobody uses that way. ([StreamVLA
page](../papers/streamvla.md), 2602.01100): the completion-state
gate re-*reasons* but never cuts the chunk (K fixed, gate orthogonal
to chunk boundaries) — complements VLA-Corrector's truncation axis
rather than competing; its economics datum (event-triggered refresh
≈ always-refresh at half latency) transfers to any replan-policy
pre-reg. Gate unchanged: parked on #16.

**2026-08-09 — lit `0814`: our seam read, published as a detector
([VLA-FAIL page](../papers/vla-fail.md), 2606.21386):** their ACC
score is the boundary-disagreement quantity we measured — previous
chunk's unexecuted suffix vs new chunk's prefix over the
receding-horizon overlap — deployed as a runtime failure signal.
Three borrowable deltas banked: velocity normalization (per-dim MAE
÷ in-chunk motion range, clamped — scale-free across slow/fast
phases), EMA α=0.9 before thresholding (our jump numbers are
instantaneous), position-dims-only restriction; plus the
conformal-band-on-20-successes thresholding recipe. The sharp
cross-read: ACC compares *sampled* chunks, so our measured
~3.3-unit fresh-noise mode term sits inside their signal as an
undecomposed noise floor — they fix the noise for LLMD's features
but NOT for ACC's actions, so our shared noise ticket would shrink
their null distribution and make the detector strictly more
sensitive. Population-level confirmation: detection degrades as
overlap shrinks. Menu unchanged; still parked on #16.
