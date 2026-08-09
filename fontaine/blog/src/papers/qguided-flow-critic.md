# Guided Action Flow: a learned chunk critic steering a frozen flow policy

*Read 2026-08-09 (standing lit slice, same session as the rung-(b′)
NO-SCORER escalation map this feeds). Paper: Guided Action Flow
(QGF), [2607.02092](https://arxiv.org/abs/2607.02092), July 2026.*

**The paper in plain words.** A robot policy that generates its
actions by "denoising" (flow matching) can be *steered* while it
generates: a small learned judge — a critic — scores the action
sequence being formed, and its gradient nudges every denoising step
toward actions the judge likes. The base policy stays frozen; the
judge is a small network trained on recorded successes and failures.
On a standard sim benchmark this lifts single-task success rates by
4–14 points — but on genuinely held-out tasks the gain shrinks to
~2.5 points, and the authors say plainly that making the judge
generalize is the unsolved part.

## What it contributes

- **Critic**: an MLP (hidden 768, depth 4) over policy-side
  observation features + the flattened **action chunk** + a task
  embedding (mean-pooled hidden states from the frozen SmolVLA
  language pathway). Trained on 500 rollout episodes with sparse
  **success-to-go** labels (γ^(steps-until-success), 0 if never);
  episode-level splits.
- **Guidance**: at every reverse-flow step, form the clean-action
  estimate â = x_t − t·v_t, take the gradient of a K=3 critic
  ensemble's mean score w.r.t. â, clip it, gate it by ensemble
  *disagreement* (m = max(m_min, exp(−α·σ_Q)) — uncertain critic ⇒
  less steering), and subtract it from the velocity.
- **Results**: single-task 68→82 and 82→86; multi-family validation
  46→56; **held-out test 65→67.5 (+2.5 pts on 40 episodes)**. A
  spatial-only critic *hurt* on transfer (53.3→51.7). Task
  conditioning from frozen-VLM hidden states was the only variant
  that generalized at all.

## What transfers to us, and what doesn't

- **A third learned-scorer shape for the (b′) escalation map.** Our
  NO-SCORER verdict priced two learned routes (RoVer-style
  supervised rerank, uPRM-style set-joint label-free). QGF is a
  distinct third: *continuous gradient guidance* instead of
  discrete rerank — it never enumerates candidates, so it sidesteps
  the fixed-K width question entirely, and it composes with the #1
  noise-ticket machinery (steer the draw rather than pick among
  draws). For the flow board row this is the natural form; for the
  AR subgoal ladder it doesn't apply (no continuous latent to
  steer).
- **The label story is the interesting part for us.** Success-to-go
  from rollouts is data we don't have — but our panel carries weak
  *judge outcome labels* per frame (the Q2 slices), and the (b′)
  run dumped 4,298 picked-vs-oracle pairs. A critic trained on
  weak-label success-to-go over banked episodes is conceivable
  without new GPU collection; carried as a design note on the
  escalation map, not an arm.
- **The disagreement gate travels.** Gating guidance strength by
  ensemble variance is a cheap uncertainty mechanism any scorer
  rung of ours could adopt (score-and-abstain beats score-always —
  the same lesson our SC anti-selection taught).
- **Doesn't transfer / caveats to carry loudly**: evaluation is
  tiny (the headline held-out gain is +2.5 pts on **40 episodes**);
  **no best-of-N or rerank baseline is reported**, so
  guidance-vs-selection is unpriced in the paper (our banked
  best-of-10 ceilings are the missing comparison); guidance is
  parameter-sensitive (β, clip, gate floor); sim-only, SmolVLA-only.

## What it fed

- **#6** — escalation map gains a "gradient-guidance" row (flow-side
  only) with the weak-label success-to-go design note; the
  uncertainty-gate lesson attached to every learned-scorer rung.
- **#1 / #19** — rung-3 candidate family note: critic-guided
  sampling as the continuous alternative to draw selection; any arm
  must be priced against the banked best-of-10 oracle ceilings the
  paper itself lacks.
