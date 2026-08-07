# Offline validation: does the panel predict the robot?

**Papers:** Critical Interval MSE
([2606.29898](https://arxiv.org/abs/2606.29898)) · Do Open-Loop
Metrics Predict Closed-Loop Driving?
([2605.00066](https://arxiv.org/abs/2605.00066)) · SIMPLER
([2405.05941](https://arxiv.org/abs/2405.05941)) · AutoEval
([2503.24278](https://arxiv.org/abs/2503.24278)) · ALOE
([2602.12691](https://arxiv.org/abs/2602.12691)).
**Read:** 2026-08-07, sanctioned lit slice. **Fed:** #16 (the proxy
question under the north star — new critical-frame re-pooling rung
banked), and the interpretive frame for every leaderboard delta we
publish, starting with the attach screen's Δ_seam.

Everything this programme measures is a proxy. The leaderboard, the
K1 gates, the attach screen's frozen reads — all of them are
open-loop action-prediction error on held-out demonstrations:
show the model a frame a human operator actually reached, ask it
for the next action chunk, score the chunk against what the human
did. The north star is a policy that *closes the loop* on a real
rig, where the model visits states produced by its own previous
actions. The gap between those two settings is the oldest known
hazard in imitation learning. This slice asks: what has actually
been *measured* about that gap, and is there anything cheap we
should change about how we score?

## The direct measurement: raw MSE correlates −0.61, and can get the sign wrong

Critical Interval MSE (2606.29898) is the paper we'd have wanted to
write. They take 27 checkpoints across three modern VLA families
(π0.5, X-VLA, GR00T N1.7 — varied by data scale, training steps,
PEFT config, action-head size, backbone), roll every one of them
out — 49 simulated tasks from LBM-Eval plus four real Franka tasks
(pour-water, arrange-mouse, fold-towel, unplug) — and correlate
rollout success against offline validation metrics. Raw validation
MSE, our panel's metric class, achieves Spearman ρ = −0.61
(Pearson −0.56) against rollout success. Not nothing — a
correlation of that size still separates big gaps — but far from
the −1 you'd want before trusting a close call. The scare figure
is their data-scale slice: ranking checkpoints trained on 20%→100%
of the data, raw MSE correlated *positively* with success
(ρ = +0.90) — the metric ranked the variants **backwards** —
while their repaired metric held at −0.90. An offline number can be
smoothly, confidently wrong in sign across a family of models.

Their repair is two moves, both aimed at making offline scoring
*resemble rollout*:

1. **Score only the frames that decide the task.** A few-shot VLM
   prompt annotates each demo video with its task-critical
   segments (the grasp, the insertion — not the transit). Error is
   pooled over those intervals only. The intuition: most timesteps
   of a demonstration are easy free-space motion where every
   checkpoint agrees, so averaging over all frames dilutes the
   signal that actually varies.
2. **Align predictions the way execution would.** Overlapping
   action chunks get temporally ensembled (as an executor would
   smooth them), and predicted-vs-expert matching uses DTW with a
   small window, so a harmless 100 ms timing offset isn't scored
   as a large action error.

Together: ρ improves from −0.61 to −0.87 overall, and on two of
the four real tasks the rank correlation is perfect (pour-water
r = −0.99, arrange-mouse ρ = −1.00). Under distribution shift the
gap widens in their favor (skill-OOD: −0.36 raw → −0.69). Honest
caveats they print: offline scoring still can't see dynamics, so a
policy solving the task by a valid path the demos don't contain is
misread; collector-protocol mismatch between train and validation
demos wrecked two of their real-task correlations (fold-towel,
unplug); and they don't claim it works for long-horizon planning
tasks.

## The mechanism, measured elsewhere: driving's open-loop/closed-loop gap

The autonomous-driving community has run this exact correlation
study at benchmark scale, and 2605.00066 cross-references an
open-loop benchmark (NAVSIM) against a closed-loop one
(Bench2Drive) over the methods with paired published numbers. Two
findings transfer as *shape*: pure trajectory-error metrics
(ADE/FDE — the analogue of raw action MSE) are the *weak* end of
the open-loop family, while semantically weighted metrics
(progress, safety-aware composites) carry most of the predictive
signal — a 3-metric composite reaches ρ = 0.90. And the named
failure mechanism is the "snowball effect": small open-loop
deviations compound once the policy consumes its own consequences.
Same moral as CI-MSE from a different field: *which frames and
which semantics you score matter more than the raw error metric*.

## The other exits: better proxies, not better MSE

- **SIMPLER** (2405.05941) replaces the offline metric with a
  *simulated rollout* of the real setup (~1500 paired sim/real
  episodes across two embodiments) and contributes the metric we
  should steal for any future proxy-vs-truth audit: **MMRV**, mean
  maximum rank violation — how badly the proxy misranks policy
  pairs, weighted by the real-world margin at stake — instead of
  Pearson r, which rewards linear fit rather than correct ordering.
- **AutoEval** (2503.24278) then measured SIMPLER's reliability
  and found it *policy-dependent* — Open-π0 scores near zero in
  their sim on a task it solves fine in reality — and built the
  alternative: autonomous *real* evaluation (VLM success
  classifier + learned reset policy), matching human-oracle evals
  at Pearson 0.942, MMRV 0.015, with ~99% less human time. The
  lesson isn't the infrastructure (we have no fleet); it's that
  even a purpose-built simulator's fidelity is uneven across
  policies, so a proxy validated on one model family doesn't
  automatically transfer to the next.
- **ALOE** (2602.12691) is the RL lane's answer: once you have
  rollout data at all, learn an action-level Q-critic from it
  (Spearman 0.93 with realized returns on π0.5-backbone tasks) and
  stop scoring against demonstrations entirely. Out of scope until
  the rig exists — noted as where this road eventually leads.

## What transfers to us

- **Calibration for every delta we publish.** Our panel MAE is
  exactly the metric class measured at ρ ≈ −0.6. That's fine for
  the gates it currently serves (K1's margin-4.9 crossings and
  flow-vs-AR gaps of a full MAE point don't need ρ = −0.95), and
  it's the right humility for close calls: the attach screen's
  Δ_seam read lands this week, and if F-vs-K separates by a
  whisker, the honest claim is "indistinguishable under the
  proxy," not a winner. The pre-registered bands already lean this
  way; this page is the citation for *why*.
- **Critical-frame re-pooling is nearly free for us.** CI-MSE's
  expensive ingredient — VLM annotation of task-critical segments
  — is something our data already carries for free: the aux-label
  fields (event, holding transitions, subgoal boundaries) mark
  exactly the semantically loaded moments, and every leaderboard
  eval already dumps per-frame predictions to npz. Re-pooling
  existing dumps over critical frames only, and checking whether
  any of our published rankings *reorder*, is a pure CPU
  post-processing screen on artifacts we already have. Banked as a
  rung on #16 (the proxy is #16's short-term half). If rankings
  hold, the panel gains a robustness citation; if they reorder
  anywhere, that's the cheapest possible early warning the proxy
  can give us.
- **MMRV for any future proxy audit.** When rig trials eventually
  exist (#16's parked half), the sim/proxy audit should be scored
  by rank violations weighted by real margins, not Pearson.
- **The collector-mismatch caveat lands on us too.** CI-MSE's
  real-task correlations degraded when validation demos came from
  a different operator/protocol than training. Our holdout split
  is same-collection, so today's numbers are safe — but any future
  eval set collected on the owner rig by a different protocol
  inherits this hazard on day one.

## What doesn't transfer

- **The DTW/ensembling alignment half** presumes chunked execution
  with overlap smoothing at deploy time; our panel scores one
  chunk per sampled frame with no executor in the loop, so the
  timing-tolerance repair has no analogue until there's a real
  executor (it pairs naturally with the async-chunk-execution
  page's RTC material when that day comes).
- **AutoEval's fleet infrastructure and ALOE's critic training**
  both need rollouts we can't produce yet — parked with #16's
  parked half, not actionable.
- **The sign-flip result is theirs, not ours.** Raw MSE ranked
  *their* data-scale family backwards; we can't conclude our panel
  does the same anywhere — only that the failure class exists in
  precisely our metric family, which is reason to run the cheap
  re-pooling screen, not reason to distrust landed reads.
