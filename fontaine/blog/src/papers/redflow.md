# RedFlow: mining your own failures for the action you should have taken

*Read 2026-08-09 (lit slice `lit-radar-0815`, priority 5: #16
RL-pole candidate). Paper:
[2607.27782](https://arxiv.org/abs/2607.27782) — "RedFlow: Redirect
Failure into Action-Level Corrections for Flow-matching VLA Policy"
(Yan, Li, Zhu (eq.), Wang, Shou, Miao, Pang, Hong, Guo — HKUST; v1
2026-07-30, preprint). Against the RL-pole ledger this is a first on
two axes at once: the first fully **offline** entry (no parallel
envs, no live rollouts during training) and the first with a clean
**real-robot before/after** — the two properties every prior entry
was missing. It also bridges the pole to the intervention levers
(UniSteer/[FlowDAgger](flowdagger-latent-dagger.md)): corrective
targets without a human in the teleop loop, retrieved from the
policy's own successes instead.*

**The paper in plain words.** When a deployed robot policy fails, the
recording of that failure usually gets thrown away or, at best,
down-weighted wholesale. RedFlow's bet is that most failed episodes
are mostly fine — a few specific actions went wrong — and that
somewhere in the success pile the robot already did the right thing
in a nearly identical situation. So: score every action chunk by
whether task progress (from an off-the-shelf progress-estimation
model) went up or down around it, cluster chunks by robot state +
progress, and for each bad chunk look up what the successful chunks
in the same cluster did — that becomes its correction target. Then
fine-tune the flow policy with three pulls at once: imitate good
chunks, push predictions away from bad ones, and redirect
recoverable failures toward the retrieved corrections. All offline,
from a fixed buffer of demos plus deployment rollouts labeled only
success/fail per episode — no human action labels, no simulator
farm. Real-world average across three tasks rises 56.7% → 74.7%,
and on LIBERO it matches PPO/GRPO/DDPO with ~10× fewer samples.

## What it contributes

- **Automatic action-level credit from episode-level labels.** Per
  chunk, `Â_t = p̄_{t+W} − p̄_{t−W} + b·(2·𝟙[success] − 1)`: local
  progress change from a pretrained General Reward Model
  (Robo-Dopamine — image+instruction → progress in [0,1]) plus a
  trajectory-outcome bias. Sign of Â_t labels the chunk
  positive/negative. The only per-episode human-adjacent input is
  the binary outcome.
- **Context-Aware Corrective Matching.** Contexts are hand-crafted
  features `[normalized proprio; β·smoothed progress]` — no learned
  embedding, no visual clustering — grouped by HDBSCAN. For a
  negative chunk, the corrective target `a*` is the
  advantage-softmax-weighted centroid of the *positive* chunks in
  its cluster. Chunks in outlier clusters or clusters with no
  positives are marked **uncorrectable** and get suppression only —
  the ablation says this separation is the single most important
  design choice.
- **Adaptive Redirection Objective**, three terms on the flow head
  (endpoint via linear `x̂₀ = x_n − n·v_θ`): (1) advantage-weighted
  CFM attraction `w_t·‖v_θ − u_n‖²` with `w_t = σ(Â_t/T_w)`; (2)
  failure suppression `(1−w_t)·max(0, m − ‖x̂₀ − a_t‖²)` — a
  finite-margin hinge that only fires when the prediction strays
  *near* a known-bad action (margin m is a running average, so
  suppression is bounded, not a global push-away); (3) target-guided
  correction `(1−w_t)·‖x̂₀ − a*‖²`, gated on a correction target
  existing. Theorem 1: the minimizer is the closest point to the
  correction target outside the margin ball around the bad action.

## The experiments it ran

Base policy is π₀, deliberately *weakly* fine-tuned (58 demos on
LIBERO Spatial/Object/Goal, 208 on Long) so there are failures to
learn from; buffer = demos + 1,536 mixed-quality rollouts per suite.
LIBERO averages: base 56.2 → **68.2** vs AWR 62.3 and DPO 59.7 (the
only two offline baselines — no IQL/filtered-BC). Biggest suite win
is Goal, 48.6 → 71.2. On LIBERO-Spatial it reaches 75.8%, matching
PPO (~13K rollouts), GRPO (~16K), DDPO (~24K) from 1,536 offline
trajectories — the ~10× sample-efficiency claim. Real world:
dual-arm AgileX Cobot Magic, three tasks (clothes folding / object
sweeping / table cleaning), 600/200/100 demos then 200/100/100
deployment rollouts, 100 eval episodes per task: average **56.7% →
74.7%** (clothes folding 36→67 is the big mover; per-task numbers
are figure reads). Ablations (LIBERO avg, full method 72.5): drop
uncorrectable-failure separation → 61.0 (−20.4 on Goal alone); drop
success rollouts → 62.4; drop failure rollouts → 68.0; ℒ_sup and
ℒ_cor individually cost ~4, jointly 6.8 — complementary, not
redundant. Fine print: no retention or OOD measurement anywhere,
success/failure ratio of the rollout buffers unstated,
hyperparameter values live in unfetched appendices, and the low π₀
base numbers are an artifact of the deliberately small SFT — gains
from a strong base are unmeasured.

## What transfers to us

**RL-pole entry 7 for [#16](../ideas/16-rig-transfer-benchmark.md),
and by rig-relevance it slots in at the top.** The ledger's standing
trade is critic-free/failure-driven buys retention, heavy machinery
buys peak — RedFlow doesn't resolve that (retention unmeasured, see
below) but it dominates the *cost* column:

1. **First entry with no environment in the training loop.** RLDT
   wants 64–1,000 parallel envs, [SA-VLA](sa-vla.md) 154 GPU-h + 64
   envs, [FPO](fpo-flow-policy-optimization.md)/[π-StepNFT](pi-stepnft.md)
   co-located sim. RedFlow trains from a fixed buffer of 100–200
   real deployment rollouts per task. The robot is only needed to
   *collect and evaluate* — which is exactly the budget shape the
   owner rig can pay. The pole's "sim-first, infrastructure is the
   blocker" verdict gets its first genuine exception.
2. **Corrective supervision without teleop.** UniSteer and
   [FlowDAgger](flowdagger-latent-dagger.md) buy their corrections
   with live human interventions; RedFlow retrieves them from the
   policy's own successful chunks in matched contexts. On a rig
   task where the policy is partly working (the owner's regime:
   motion fine, gripper placement off), failures near successes are
   plentiful — the clustering assumption is most plausible exactly
   there.
3. **The ablation's lesson generalizes the pole's shape.** The
   biggest single component is *knowing which failures not to
   correct* (uncorrectable separation, −11.5 avg when dropped) —
   another instance of the SA-VLA/FPO pattern that protective
   structure, not the update rule, carries the sign.
4. **Consistent with the frozen-trunk trend by omission**: it's a
   weight-space fine-tune of the flow head with the attraction term
   anchoring behavior on positives — closer to advantage-weighted
   SFT with a repulsion/redirection bolt-on than to policy-gradient
   RL. That framing matters for what to expect on retention.
5. **Caveat that rides any rig use** (from the
   [silent-failures page](silent-failure-observability.md)): the
   whole pipeline keys off binary success flags plus a pretrained
   progress model. Robo-Dopamine's progress estimates on
   owner-rig-looking scenes are an unvalidated dependency — a GRM
   sanity read would have to precede any RedFlow-style run.

## What doesn't transfer

- **Retention/OOD: unmeasured, again.** No non-target-task or
  shifted-condition eval. FlowDAgger's −0.94 SFT-forgetting critique
  stands unanswered against this recipe too; if it ever runs on the
  rig, held-out retention is the first read to demand.
- **The 56.7 → 74.7 is a weak-base result.** The base policies were
  deliberately under-trained to generate failures; nothing here says
  what RedFlow adds on top of a well-fit SFT policy (the LIBERO base
  at 56.2 is far below what π₀ normally posts).
- **Offline-baseline field is thin** — AWR and DPO only; no
  flow-native offline competitor, no comparison to simply refitting
  on success-filtered rollouts *with* the GRM weighting (their "w/o
  failure rollouts" at 68.0 vs 72.5 is the closest proxy, and the
  gap is only 4.5).
- **Hyperparameter surface is wide** (κ, T_w, b, W, β, margins, two
  loss weights, HDBSCAN settings) with values buried in appendices —
  reproduction risk on a new embodiment.

## Which idea/arm it fed

[#16 rig-transfer benchmark](../ideas/16-rig-transfer-benchmark.md)
— RL-pole entry 7: the first offline, real-robot-measured entry;
failure-driven corrective supervision from ~100–200 deployment
rollouts + binary outcomes + an off-the-shelf progress model, no
envs, no teleop, no critic. It re-prices the pole: parallel-env
infrastructure is no longer the universal entry fee. Open before rig
use: retention (unmeasured), GRM validity on rig scenes, and the
weak-base confound. No new arm; the pole keeps its entry conditions,
but this is the recipe currently closest to satisfying them.
