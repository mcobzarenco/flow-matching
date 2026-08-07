# Q-VGM

**Paper:** Q-VGM: Q-Value-Gradient Matching for Off-Policy
Reinforcement Learning of Flow-Matching VLA
([arXiv:2606.08015](https://arxiv.org/abs/2606.08015), SJTU /
UMich, v2 July 2026, preprint). Banked from the 2026-08-07 lit
slice; re-read at full-text depth for this page — including both
arXiv versions, which differ substantially (see the caveat at the
end). **Fed:** #4 — it is the reason "frozen default stands" is not
a dead end in the attachment-seam screen: the frozen-trunk
configuration is exactly the substrate the field now fine-tunes
with offline RL.

## The problem it solves

Suppose you have a flow-matching VLA that works — and a pile of its
own evaluation rollouts, some successful, some not. Offline RL is
the obvious way to convert those logs into improvement, but flow
policies resist it on two fronts: iterative denoising gives no
tractable action likelihood (so no standard policy gradient), and
backpropagating a Q-value through the whole denoising chain is
unstable at VLA scale — their own Diffusion-QL baseline *degrades*
the policy below its SFT starting point (72.6 vs 79.0 average).

## What it contributes

The trick is to never touch the chain. Policy improvement is framed
as optimal control over the denoising dynamics, whose solution is a
**residual velocity** proportional to the value gradient. Per
training iteration:

1. Roll the current policy forward K Euler steps from noise,
   entirely under stop-gradient.
2. On the **last M=5 denoising steps only** (where the
   clean-action estimate is nearly exact), form a look-forward
   clean-action estimate under a *frozen copy* of the SFT expert's
   velocity — anchoring to behavior-policy support.
3. Take a few clipped gradient-ascent steps on the critic's
   Q(s, action) in action space, with **keep-best selection** over
   all iterates — the unmodified base action is iterate 0, so
   ascent that doesn't help falls back automatically (a discrete
   line search).
4. Convert improved-minus-base into a residual velocity target and
   regress the trainable expert onto it (targets detached).

Only the flow expert trains. The VLM backbone, the base velocity,
and the critic are all frozen; at inference the critic is gone —
the guidance has been amortized into the expert's weights. The
critic itself is a stepwise chunk critic trained with IQL
(expectile 0.8) on sparse success rewards, reading a 2048-d
autoencoder compression of the frozen VLA prefix rather than raw
features.

## What they ran

- **LIBERO, four suites, 500 episodes/suite:** π0.5 few-shot-SFT
  start averages **79.0**; Q-VGM lifts it to **92.5** (+13.5
  points; Long suite 62.2 → 83.8). Same-critic alternatives do
  worse: test-time Q-selection 86.0, test-time Q-guidance 88.7,
  action distillation 88.8, backprop-through-chain 72.6.
- **Sample efficiency is the standout table:** Q-VGM reaches 96.2
  on LIBERO-Spatial from **500 evaluation episodes + 432 demos** —
  the logs you'd have anyway — where online-PPO π_RL uses ~205k
  on-policy episodes for 99.6 (~400× more rollouts for +3.4).
- **Ablations are coherent with the mechanism:** swapping the
  VLA-prefix RL token for a ResNet state encoder costs the most
  (92.5 → 87.4); removing keep-best (→88.6), the frozen-base
  anchor (→86.8), or spreading alignment over all denoising steps
  (→86.2) each hurt as the theory predicts.
- **Real robot (v2):** a bimanual platform, four tasks, 20 trials
  each — strong Q-VGM numbers (e.g. 20/20 grasp) but no SFT
  baseline row in the table, so the real-world *delta* is not
  verifiable from the paper.

## What transfers to us, and what doesn't

**Transfers.** The structural lesson for #4: a **frozen trunk plus
flow expert is the configuration offline RL knows how to improve.**
Every trainable surface Q-VGM touches is expert-side; a KI-joint
trunk (still adapting under CE) would complicate this — RL updates
into a live trunk is exactly the instability everyone is avoiding.
So if the seam screen's Δ_seam favors F, the frozen arm keeps a
named, published escalation path; if it favors K, adopting K spends
that option. Also notable for us: their best state representation
is a *compressed frozen-VLA prefix* — independent support for the
frozen-features-carry-structure prior (#6/#17, and the
[value-probe paper](test-time-selection.md)).

**Doesn't transfer.** Everything needs a success signal: sparse
rewards from evaluation rollouts. Our panel is offline chunk-MAE on
a community corpus — we have no rollout success labels until the
rig/sim stage, so Q-VGM is a banked escalation path, not a
runnable arm. Record-only prior; no new arm until the seam screen's
own verdict lands.

## A caveat about versions

v1 → v2 is a major rewrite: the critic changed (Cal-QL → IQL), all
LIBERO numbers were re-run (SFT average 75.0 → 79.0), a RoboTwin
section was dropped entirely, and the real-robot study was redone
on different hardware. The 79.0 → 92.5 headline is v2's. Nothing
wrong with revising a preprint — but numbers quoted from this paper
should carry the version, and the missing real-world baseline plus
unstated compute budget (no GPU count or wall-clock anywhere in
either version) are the two holes a reviewer would poke first.
