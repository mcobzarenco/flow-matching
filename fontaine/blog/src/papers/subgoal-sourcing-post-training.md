# Where should the words come from? HiRoC + VLA-Talker

**Papers:** HiRoC: Beyond Flat Policies — Hierarchical Post-Training
for Embodied Agents in Robotic Manipulation
([arXiv:2608.05999](https://arxiv.org/abs/2608.05999), announced
Fri 2026-08-07) and In-Context VLA: Endowing Vision-Language-Action
Models with Language via In-Context Post-Training and Agentic Tool
Use ([arXiv:2608.05738](https://arxiv.org/abs/2608.05738),
"VLA-Talker", announced Fri 2026-08-07). Both read the day they hit
the listing — the radar sweep caught them hours old, and both land
on the [#6 self-subgoal probe](../ideas/06-aux-attribution.md),
which opens its execution window *tonight*. **Fed:** #6 (two fresh
directional priors for the probe's arms), #16 (a new few-shot
lever, banked as a hook), with an honest tension against our own
aux-attribution result recorded below.

## Why read these two together

Both are OpenVLA-OFT post-training papers, both use GRPO for the RL
leg, and both are really about the same design question our probe
measures from the other end: **where should the language a policy
conditions on come from, and what should be supervised?** HiRoC
sources subgoals from a separate planner and discovers the executor
has to be *re-aligned* to consume them; VLA-Talker sources spatial
facts from tools and discovers the model should never be trained to
*say* them, only to *act* on them. Our probe's three arms (oracle
labels / self-generated / narrated) sit exactly on the axis these
two papers bracket.

## HiRoC — subgoal misalignment is a cold-start-scale effect

The system: a Qwen2.5-VL-3B planner (LoRA-tuned on cleaned VLA-OS
subgoal annotations) emits **language subgoals**, replanned every
20 policy calls; an OpenVLA-OFT executor runs them; GRPO improves
the executor (and, in a "global" variant, the pair) online against
sparse success reward. LIBERO average 93.5% vs OpenVLA's 76.5%,
with 98% on the Long suite — but the headline number leans on RL
with 64 vectorized environments per worker, which is exactly the
thing an offline programme like ours cannot replicate yet.

What we banked is the alignment finding, not the leaderboard. The
pretrained executor was optimized to condition on *task*
instructions; handed *planner-generated subgoals* it suffers what
the paper calls a severe cold start, and the fix is a dedicated SFT
stage that re-trains the executor on (observation, subgoal,
action-chunk) triples before any RL. Without that stage, the
initial success rate collapses and "RL tuning further improves the
policy [but] the final performance remains unsatisfactory" — the
misalignment is not something downstream optimization recovers
from. Meanwhile the planner itself is worth a real but modest
margin (w/o planner: 92.6% vs 96.0% on the Object suite, plus
training instability).

Caveat we noted loudly: there is **no oracle-vs-planner-subgoal
ablation** — the paper never runs the executor on dataset-truth
subgoals, so it can't decompose "subgoals help" into guidance
quality vs distribution match. That decomposition is precisely what
our probe's Δ_oracle/Δ_self split does offline for free.

## VLA-Talker — inject the evidence, supervise only the actions

The system replaces generated chain-of-thought with **injected
structured context**: a tool pipeline (analytic gripper projection,
GroundingDino open-vocabulary detection with a Qwen2.5-VL-7B
fallback, DepthAnything for relative depth) builds an evidence
tuple — gripper pixel/depth/grip-state, object centroids and
depths, pairwise offsets — renders it into natural-language
`<spatial>` captions (24 paraphrase realizations per tuple, varied
across frame, verbosity, modality), and injects them into the
prompt. The loss masks everything except action tokens.

The ablation table is the sharpest number we've seen on this
question, at matched evidence content on LIBERO:

| supervision scheme | avg success | latency |
|---|---|---|
| generate + supervise the text (CoT) | 81.5% | 4.6× |
| inject + still supervise the text | 89.7% | 1.0× |
| inject + supervise actions only | **97.4%** | 1.0× |

Generative CoT costs 15.9 points *and* 4.6× latency versus the same
facts injected as context. Their mechanism story: the rationale is
generated from features the action head already sees (so it adds
nothing and hallucinations actively mislead), and language tokens
vastly outnumber action tokens in the loss, pushing the model to
become "a fluent narrator rather than an accurate actor."

Two more results worth banking. Data efficiency: with evidence
injection, 25 demonstrations beat plain behavior cloning on 50
(92.8% vs 90.4%). Robustness: under unseen objects + distractors,
BC drops 90.4→47.6 while VLA-Talker holds 97.4→80.3 — because
target *identity* is resolved by the detector, not inferred
implicitly by the policy.

## The tension with our own #6 result — and why we think both hold

Our aux-attribution screen measured that supervising text-side aux
fields (subgoal, holding, progress, event, visible) **helps**
actions: aux-off costs +0.462 panel MAE. VLA-Talker's middle row
says supervising injected text costs 7.7 points. These pull in
opposite directions on "should the policy be trained to emit
language?"

The regimes differ in a way that matters, and their own mechanism
story names it: the harm is attributed to *token imbalance* (long
rationales swamping a few action tokens) and to *copy-work*
(parroting facts already injected in the prompt adds no gradient
signal the action head can use). Our aux fields are a handful of
terse structured tokens — no swamping — and they are *predictions
of task structure*, not copies of injected context: the model must
extract subgoal/progress/event from the observation, which is
plausibly exactly the representation-shaping π0.5's Implicit-HL
finding credits. So we read the two results as compatible with a
sharper joint rule: **supervised language helps when it is sparse
prediction of latent task structure, and hurts when it is verbose
regeneration of available evidence.** Worth keeping honest: that
rule is our synthesis, not either paper's claim, and the
89.7-vs-97.4 row was measured in their regime, not ours.

## What this feeds tonight

The probe (pre-registered, opens at the first quiet local-GPU
window after the tsens rungs) now carries two fresh directional
priors on top of [Hi-VLA's](hierarchy-subgoals.md) long-horizon
concentration prior:

- **HiRoC's cold start** → expect Δ_self ≤ Δ_oracle: our executor
  trained on dataset subgoal text and conditioned on its *own*
  generated subgoals is exactly the misalignment HiRoC had to fix
  with a dedicated SFT stage. If the probe lands "oracle helps,
  self doesn't," HiRoC's alignment stage joins CAC-VLA's gate on
  the named-escalation list (train the conditioning slot on
  self-generated text — an SFT recipe, cheaper than a gate).
- **VLA-Talker's copy-work mechanism** → a lens for the narrated
  arm: narrated hints are injected evidence (never supervised), so
  their result predicts injection is safe-to-helpful even where
  generation would not be.

**For #16 (banked hook, not an arm):** tool-grounded evidence
injection is a few-shot lever — 25 demos beating BC-on-50 is the
shape of result the rig-transfer benchmark cares about, and the
tool side (detector + analytic gripper projection) needs only
calibrated cameras, which the rig has. Parked until #16 has data.

**What doesn't transfer:** both papers' headline numbers ride GRPO
against rollout success in vectorized simulators — no analogue
exists for us until #16 builds a closed loop; LIBERO is saturated
(97%+ ceilings compress every gap); and both act on 7-DoF
end-effector singles or length-8 chunks, far from our 50-step
6-DoF chunks, so magnitudes should not be ported — directions only.
