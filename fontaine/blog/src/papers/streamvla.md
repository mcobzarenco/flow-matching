# StreamVLA: reason only when the goal image says you're done

*Read 2026-08-09 (lit slice `lit-radar-0813`, priority 5: #6
phase-estimation adjacency). Paper:
[2602.01100](https://arxiv.org/abs/2602.01100) — "StreamVLA:
Breaking the Reason-Act Cycle via Completion-State Gating" (Chen, Wu,
Wang, Li, Fang — Tsinghua, v2 2026-02-07, preprint).*

**The paper in plain words.** Robot policies that "think out loud" —
generating a text plan, sometimes a mental image of the goal, before
every action — are more capable on long tasks but painfully slow,
because they re-think at every step. This paper makes the thinking
event-driven. When the model does think, it produces two things: a
text subtask and a *generated image of what the scene will look like
when that subtask is complete*. Both are then locked in as context,
and a tiny gate network compares each new camera frame against the
imagined completion image, asking one question: does reality look
like the goal yet? Only when the answer is yes does the model think
again. The result: it skips the expensive reasoning on 72% of steps,
halves latency, and loses essentially nothing — because, as their
own ablation shows, re-thinking *more often* than "one subtask
completed" adds almost no success. The thinking content is worth
about four points on long-horizon tasks; knowing *when* to refresh
it turns out to be nearly free.

## What it contributes

- **The completion state as a time-invariant goal anchor.** The
  imagination head (312M, Infinity-style bitwise AR over a VQ-GAN
  codebook) generates the *end state* of the current subtask, not a
  frame at t+Δt. Time-invariance does double duty: better
  conditioning for the flow head (their fixed-offset control is
  worse: −0.4 LIBERO-Long, −1.5 RoboTwin-Hard) and — the actual
  novelty — a stable reference that a gate can compare observations
  against without caring about execution speed.
- **The gate**: 58M (1.75% of the 3B π0.5-derived model),
  cross-attention with current head-camera tokens as queries against
  the locked goal image, concatenated with the locked subtask text,
  MLP → sigmoid discrepancy d_t. Trained with BCE: 1 = mid-subtask,
  0 = at an annotated boundary. d_t ≤ 0.5 (converged to goal) →
  re-run both AR heads and re-lock; otherwise the flow head (10
  Euler steps, chunk K=10/50) conditions on the cached goal.
- **Subtask-boundary supervision at labeling cost**: VLM (Qwen3-VL)
  proposes temporal segments, humans refine boundaries, the
  completion frame is the boundary frame; RoboTwin segments come
  free from simulator predicates.

## The experiments it ran

LIBERO avg **98.5** (Long 96.6 vs π0.5's 92.4 on the same 3B base);
RoboTwin 2.0 hard 37.2 vs π0.5's 33.8; real dual-cam AgileX tasks
90/70% vs π0.5's 45/35% (20 trials, author-run baselines, no CIs),
including 55% vs ≤15% when a human displaces objects mid-task.

The economics table is the contribution:

| Variant | LIBERO-Long SR | ms/step | skip % |
|---|---|---|---|
| Never reason (≈ π0.5 reactive) | 92.4 | 65 | 100 |
| Gated, τ=0.5 | **96.6** | 128 | 72 |
| Always reason | 96.8 | 244 | 0 |
| Fixed t+Δt prediction (ungateable) | 96.2 | 244 | 0 |

τ-sweep: SR flat from τ=0.5 through never-skip; permissive gating
(τ=0.3) drops to 93.5. Modality split of the +4.2 reasoning gain:
text subtask ≈ +3.1 (sequencing), goal image ≈ +1.5 (spatial
anchor).

## What transfers to us

- **For #6, this is the field's cleanest "phase estimation is
  cheap at the boundary" datum — and it agrees with our probe.**
  Our rung-(a) verdict located the self-subgoal bottleneck in
  single-frame *mid-execution* phase estimation. StreamVLA's gate
  works precisely because it never asks the hard mid-execution
  question ("how far along am I?") — only the easy boundary one
  ("does the scene match the goal image?"), against a reference it
  generated once. The τ-sweep is the shape of that claim: crisp
  near completion (nothing lost at τ=0.5), noisy mid-execution
  (τ=0.3 costs 3 points). Design constraint banked: *anchor phase
  decisions to a completion reference, don't estimate progress from
  the current frame alone* — which is also what
  [silent-failures](silent-failure-observability.md) found from the
  detection side (final-state exteroception carries the signal).
- **The refresh-policy datum for the #6 escalation ladder**:
  event-triggered re-reasoning at subtask completion retains ~95% of
  always-reasoning's gain at ~53% of its latency, and beats
  fixed-schedule refresh. Hi-VLA's "refresh granularity matters,
  model-predicted horizons worst" now has a counterpoint: a
  *learned completion gate* is the refresh rule that works. Any
  future rollout arm with subgoal conditioning should pre-register
  this shape rather than a timer.
- **Reason-content accounting**: +4.2 total on LIBERO-Long, with
  text sequencing worth ~2× the goal image. Consistent with our
  slot result (words help, −0.29 oracle) and π0.7's subgoal-image
  escalation — the image is the increment above text, not the
  substitute.

## What doesn't transfer

- **The gate's causal story under perturbation.** The
  interference-recovery claim (55% vs ≤15%) contradicts the stated
  gate logic as written: a displaced scene is *far* from the goal
  image → high d_t → keep executing under their threshold rule.
  Either the gate fires low on out-of-distribution mismatch
  (never measured) or recovery waits for the natural boundary. No
  gate precision/recall, false-trigger rate, or time-to-replan is
  reported anywhere — gate quality is only ever read through end SR.
  Carry the number, not the mechanism.
- **The imagination head as a dependency**: phase quality is bounded
  by generation fidelity (their stated hallucination-under-occlusion
  limit) — an odd inversion given our own scar that *generation*
  quality, not conditioning, was the rung-(a) bottleneck. A
  retrieved or oracle completion frame would decouple this; nobody
  tests it.
- Two ablation tables disagree on the no-System-2 baseline (92.4 vs
  90.6) and per-head costs — the fine-grained modality split is
  soft; the coarse +4.2 is solid.

## Which idea/arm it fed

[#6](../ideas/06-aux-attribution.md) — phase-estimation design
constraint (completion-anchored gating sidesteps the measured
mid-execution bottleneck; τ-sweep as the evidence) + the refresh-rule
datum for any rollout escalation (event-triggered ≈ always, ≫
fixed-schedule). [#22](../ideas/22-async-staleness.md) — menu
adjacency: the gate re-reasons but never cuts the chunk (K fixed;
orthogonal to chunk boundaries), so it complements
[VLA-Corrector](vla-corrector.md)'s truncation axis rather than
competing with it. Cross-refs:
[hierarchy & subgoals](hierarchy-subgoals.md) (Hi-VLA refresh
granularity), [OneWM-VLA](onewm-vla-one-token.md) (predictive token
as the cheap sibling of a full goal image),
[silent-failures](silent-failure-observability.md).
