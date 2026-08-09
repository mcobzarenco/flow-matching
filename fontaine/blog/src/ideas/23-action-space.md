# 23. Action-space: chunk-wise delta-joint — `queued`

*Tag: `action-space` · idea #23 · [index](../ideas.md)*

Opened 2026-08-09 from lit `0819`
([Action-space design](../papers/action-space-design.md),
2602.23408 — ICML 2026, code+data released and verified). We predict
absolute joint positions in chunks because the codebase we started
from did; that choice was never measured. This paper measured it in
our exact policy class.

**Hypothesis.** Retraining the action expert to predict *chunk-wise
delta* joint targets (a = q_target − q_chunk_start; never
step-to-step increments) improves policy quality at zero
architecture/data cost. Their evidence: flow-matching + joint-space +
chunk-wise delta is the best cell overall — 88.0 vs 79.6 for our
absolute-joint configuration (+8.4pp real-robot, robust across
100–500 demos and 300–1200 epochs, direction confirmed in RoboTwin
sim); step-wise delta is dominated ~10pp empirically and O(k)
noise-amplification theoretically.

**Expected effect.** Unknown offline; +5–15pp rollout-flavored if
their result transfers. Two caveats carried loudly: (1) nothing in
the paper runs on hobby-servo hardware; (2) both headline mechanisms
are deployment-time effects a per-frame offline MAE partly cannot
see — their chunk-wise-delta and absolute cells are *identical* in
decode error propagation yet differ 8–15pp in rollouts, the cleanest
evidence yet that action-space rankings can invert between per-frame
error and rollouts (standing caveat now attached to any panel-based
action-space claim).

**Bonus hypothesis worth logging at the read:** chunk-wise delta
subtracts per-rig calibration offsets — plausibly worth *more* on our
multi-rig community corpus than on their single lab arm.

**Cost.** One tiny-config probe run, then one full training run
(existing data, existing trunk; normalization recomputed on delta
stats).

**Cheapest falsification.** Pre-registered rules: chunk-wise only
(never spend a run on step-wise); decode predictions back to absolute
joint space *before* panel scoring, paired per-frame CI95 vs the
absolute-joint baseline; offline win = necessary-not-sufficient
(MAE plausibly flatters delta via better-conditioned targets while
missing drift), offline *loss* = strong evidence against switching.
The definitive read is rollout-flavored — a Squint-substrate relative
screen ([#16](16-rig-transfer-benchmark.md)) once a sim-adaptation
arm exists.

**Record.**

- *2026-08-09* — opened from the 0819 deep read; no arm queued yet
  (needs its own pre-reg; venue = any post-adamc box window or a
  local tiny-config probe first).
