# Hierarchy & subgoals — Hi-VLA and CAC-VLA

**Papers:** What Matters in Orchestrating Robot Policies: A
Systematic Study of Hierarchical VLA Agents
([arXiv:2606.10267](https://arxiv.org/abs/2606.10267), Hu et al.,
June 2026 — "Hi-VLA" below) and CAC-VLA: Context-Gated Action
Conditioning for Vision-Language-Action Models
([arXiv:2607.04816](https://arxiv.org/abs/2607.04816), Xiong et
al., July 2026). Banked from the 2026-08-07 lit slices; re-read at
skim depth with number verification. **Fed:** #6 — the design
constraints and the escalation map for the self-subgoal probe
(pre-registered, opens at the first quiet local-GPU window).

## Why this theme is live right now

Our aux-attribution result (#6: aux-off costs +0.462 panel MAE) and
π0.5's Implicit-HL finding agree that *training* on subgoal
prediction shapes the action representation. The open increment is
*runtime* hierarchy: decode a subgoal first, condition the actions
on it. We own the seam — the `[subgoal|…]` conditioning slot — and
the [self-subgoal probe](../posts/2026-08-07-prereg-selfsubgoal-probe.md)
is pre-registered to measure exactly this at zero training cost.
These two papers are the external map: one measures *how much*
explicit hierarchy buys and *where*, the other shows what a
production-grade version of "condition on your own predictions"
looks like when the predictions are unreliable.

## Hi-VLA — the systematic sweep

The setup: a frozen VLM planner (Gemini 2.5 variants) emits
language subgoals; a frozen VLA controller (Gemini Robotics
On-Device, 1–3B) executes them; the paper sweeps the orchestration
choices — planner strength, thinking on/off, subgoal-refresh
policy, what observation text the planner sees, and memory — over
MuJoCo ALOHA suites (5 tasks × 3 categories × 200 trials) plus a
real ALOHA.

The headline table is the cleanest quantification of
where hierarchy pays that we know of:

| config | short-horizon | long-horizon | reasoning |
|---|---|---|---|
| flat VLA | 69.6% | 25.3% | 50.9% |
| naive hierarchy | 69.6% | 40.6% | 66.5% |
| best hierarchy | 78.2% | **67.1%** | 80.9% |

On short-horizon tasks, naive hierarchy buys *nothing* (69.63 vs
69.57); on long-horizon it nearly triples flat performance. The
gain is a long-horizon and indirect-instruction phenomenon.

Second finding we carried: **refresh policy matters a lot.**
Success-detection termination is best (57.4% long-horizon), a fixed
~8 s window is close (52.4%), and **letting the model predict its
own horizon is worst** (43.5%) — VLA stochasticity makes advance
time prediction unreliable. Third: feeding the planner extra
*text* (bounding boxes, contact info) beats raw images alone
(38.8% → 52.4% long-horizon), with the stated failure mode that
"VLMs tend to ignore image inputs as task becomes harder" — the
hierarchy-side rendition of our #11 state-dominant-bias story.

Caveats: planner and controller are separate frozen models — the
language interface is the *only* subgoal representation studied,
and *self*-generated subgoals (one model planning for itself) are
untested. Static environments, latency ignored.

## CAC-VLA — conditioning on your own predictions, gated

CAC-VLA extends π0.5 with latent-action conditioning: a frozen
pretrained action tokenizer encodes future action segments into
compact latents; the VLM learns to *regress* those latents from
learnable query tokens; the action expert consumes them through a
**learned channel-wise gate** at every layer (a gated residual —
the gate reads the current action states and the latent update and
decides how much conditioning to let through). The asymmetry we
banked it for: **training conditions on ground-truth-encoded
latents, inference on the VLM's self-predicted ones**, with no
scheduled sampling — the gate is the stated mitigation for noisy
self-predictions.

Numbers: LIBERO average 98.3% (vs π0.5's 96.9% — a small +1.4
margin in a saturated regime); the meaningful gaps are LIBERO-Plus
robustness (89.5% vs 85.7%) and real-robot (64% full success vs
16% for π0.5 on their pick-and-place; 25 trials/method). Ablations
are thin: removing the gate costs only 0.4 points on saturated
LIBERO; no gate-vs-cross-attention or λ sweep.

## What transfers, and what it fed

**Into the probe's design (already frozen):** Hi-VLA is why the
probe's per-step decomposition expects gains concentrated in
*late-horizon* chunk MAE (their short-horizon null), and why any
future *rollout* arm must pre-register its refresh rule (their
worst-case was exactly the model choosing its own refresh). Since
their planner/controller are separate models, our self-generated
variant is a genuine increment, not a replication.

**Into the escalation map (banked, not built):** if the probe
lands in the "oracle subgoals help but self-generated don't" cell
— the truth-vs-self asymmetry — CAC-VLA's gated-conditioning
pattern is the named escalation: train on truth, infer on self,
let a learned gate modulate trust. Its thin ablations mean we'd
treat it as a design sketch, not evidence of magnitude.

**What doesn't transfer:** all their numbers are closed-loop
success under a runtime planner loop; our panel probe conditions
per-frame and sidesteps refresh policy entirely — which is
precisely what makes it cheap, and also what it cannot measure
(replan-timing effects are invisible offline).
