# Executing stale chunks: RTC and the async-inference method zoo

*Lit slice 2026-08-07 (work session 12:30Z). Cluster page: Black et
al., "Real-Time Execution of Action Chunking Flow Policies"
([2506.07339](https://arxiv.org/abs/2506.07339), NeurIPS 2025) + the
systematic comparison "Understanding Asynchronous Inference Methods
for Vision-Language-Action Models"
([2605.08168](https://arxiv.org/html/2605.08168)). Fed ideas.md item
#22.*

## The problem, in our terms

An action-chunk policy plans 50 steps and executes some horizon of
them, then replans. Every replan is planned from an observation that
is *latency* old by the time its first action executes. Synchronous
loops pause at chunk boundaries; naive async (SmolVLA-style — trigger
early, swap the chunk when it arrives) hides the pause but not the
staleness: the new chunk's early actions were planned for a world
state that has since moved. At 30 Hz on the owner's laptop, our
mean-of-10 batched-draws decode costs 576 ms ≈ **18 control ticks**
of staleness — and our chunk is 50. That is precisely the regime this
literature is about.

## RTC: the flow-native, training-free answer

The Physical Intelligence paper's real-time chunking generates the
next chunk *while* the current one executes, then reconciles: the
actions that will certainly execute before the new plan can take over
are **frozen** (they're already committed), and the sampler
**inpaints** the rest of the new chunk conditioned on that frozen
prefix — guidance during the flow/diffusion denoising loop, no
retraining, works on any flow VLA "out of the box." Evaluated on 12
dynamic Kinetix tasks + 6 real bimanual tasks (they light a match);
headline: throughput up, uniquely robust to injected inference delay,
smooth at boundaries.

## What the systematic comparison adds (the more useful read)

The survey benchmarks four families under harmonized code on Kinetix
(MLP policies, chunks 16–30) and LIBERO/SmolVLA (chunk 50, like us),
sweeping delay d = 0 → 20 ticks:

- **IT-RTC** (inference-time RTC, the paper above): wins only at
  d ≤ 2–4; at d ≥ 8 with chunk-50 policies it *collapses* (~20% vs
  ~75% base on LIBERO) — the frozen-prefix-to-inpainted-postfix ratio
  grows until guidance has nothing left to steer. Also ~3× sequential
  denoising overhead (≈1.2× end-to-end at VLA scale).
- **TT-RTC** (training-time): fine-tune the policy conditioned on the
  executed action prefix (~8 epochs ≈ 25% of base training). Zero
  inference overhead, graceful out-of-distribution in d — the best
  pick for short chunks, degrades on chunk-50.
- **VLASH**: roll the proprioceptive state forward through the
  committed actions and condition on the *predicted future state*.
  Stable at high delay (~55% at d=20) but the LIBERO numbers use
  oracle state — deployable only with a forward dynamics model, and
  its trained d_max trades low-delay accuracy for high-delay reach.
- **A2C2**: a lightweight per-tick correction head producing residual
  adjustments on top of the stale base actions. Dominates mid + high
  delay on *both* benchmarks (>90% Kinetix d=8; ~65% LIBERO d=20),
  composes with a frozen base policy, costs one small forward per
  control step.

The regime table is the paper's real contribution: **which method
wins is almost entirely a function of delay-in-ticks × chunk length**,
and results on short-chunk sim do not transfer to long-chunk VLAs.

## What transfers to us, what doesn't

- Our `--async-inference` is the survey's *naive switching* baseline.
  With single-draw 1-NFE decode (~70 ms ≈ 2 ticks) naive is fine —
  the survey says every method recovers base performance at d ≤ 2.
  **The problem only bites when we spend latency on quality** —
  mean-of-10 at 18 ticks of staleness is deep in the regime where
  naive switching visibly degrades and IT-RTC has already collapsed
  (our chunk is 50, the bad case).
- So the tempting "just bolt RTC on, it's training-free and
  flow-native" is *specifically the wrong pick for our numbers* — the
  survey exists to catch exactly that. The candidates that survive
  our regime are **A2C2-style residual correction** (frozen base —
  composes with everything we've banked, including batched draws) and
  **TT-RTC prefix conditioning** (cheap fine-tune, but chunk-50 is
  its weak spot too).
- Composition note for our stack: freeze/inpaint and prefix
  conditioning are per-row operations — they compose with the
  batched draws-major ensembling that just landed from main (the
  mask/conditioning tiles like everything else). Nothing structural
  blocks mean-of-N + staleness bridging later.
- Doesn't transfer: VLASH's LIBERO numbers (oracle future state; our
  rig has no dynamics model), and Kinetix rankings generally (chunk
  16–30 ≠ our 50).

## Where it lands

New ideas.md entry **#22 (async staleness bridging)**: parked until
the #16 rig-transfer bench exists, with a pre-registrable first
screen — measure the *actual* cost of naive switching at our real
latencies (single-draw vs mean-of-10) on rig rollouts before buying
any method; the survey predicts a large quality gap at mean-of-10
that single-draw doesn't have. If the gap is real, A2C2-style
correction is the first arm (frozen base, small head, per-tick cost
trivially affordable at 30 Hz).
