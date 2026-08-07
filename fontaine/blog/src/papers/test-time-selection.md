# Test-time selection for VLAs — the six flavors behind #19's ceiling gate

**Papers:** MG-Select
([2510.05681](https://arxiv.org/abs/2510.05681), ICLR 2026),
VLA-ATTC ([2605.01194](https://arxiv.org/abs/2605.01194)),
CoVer ([2602.12281](https://arxiv.org/abs/2602.12281)),
RoboMonkey ([2506.17811](https://arxiv.org/abs/2506.17811)),
TapSampling ([2605.25547](https://arxiv.org/abs/2605.25547),
ICML 2026), Look Before You Leap
([2607.03751](https://arxiv.org/abs/2607.03751)), and What Frozen
VLAs Already Know About Success
([2605.28527](https://arxiv.org/abs/2605.28527)). Banked across the
2026-08-06/07 lit slices; re-read at abstract-plus-skim depth with
number verification for this page. **Fed:** #19 — every one of
these waits behind the oracle best-of-10 ceiling read.

## The theme, and why it keeps showing up on the radar

A policy that samples N candidate actions is only as good as its
ability to *pick one*. The papers here all start from the same
observation: frozen VLAs are much better than their single-draw
numbers — LBYL measures pass@1 = 33% vs pass@32 = 92% on its
benchmarks — so an enormous amount of capability is sitting in the
sampling distribution, unclaimed. The field's compute is visibly
moving to the selection side (CoVer's title is literally "scaling
verification can be more effective than scaling policy learning").

This lands on our #19 directly. We just built the AR sampled-draws
instrument (draws-10, mean-of-samples) for fairness reasons, and the
molmo2 endpoint draws arm will retain a per-draw dump. The
**oracle best-of-10 ceiling read**
(`selection_ceiling_results.py`, landed 08-07) computes, per frame,
what a *perfect* selector would have bought. That single number
gates this entire literature: if the ceiling on our panel is small,
every method below is dead here regardless of how well it worked on
LIBERO — no selector gets built before that number exists.

The six flavors, ordered by what the selector costs to obtain:

## 1. MG-Select — verifier-free, from the policy's own logits

Sample N candidates; for each, run a second forward pass with the
conditioning *masked* (instruction, state, or both) to get a
task-agnostic reference distribution; score each candidate by the
KL between its conditional and masked action-token distributions
(reference tempered at τ=4); execute the argmax — the draw for
which the conditioning was most informative. No verifier, no extra
module — but the headline configuration fine-tunes the policy with
10% condition dropout so the masked distribution is well-trained,
and it needs an AR policy's logits.

Numbers: π0-FAST on RoboCasa pick-and-place 17.0% → 31.0% (100
demos); SIMPLER-WidowX 46.9% → 50.3%; gains saturate around N=4–8.
Caveat: the best masking modality is benchmark-dependent, and bare
masking on a model *not* trained with dropout misbehaves.

Why it's interesting here: **AR-100k already is the
dropout-trained model** (state dropout 0.5, subgoal dropout 0.5),
and `--mask-state` already computes the masked context — so this
flavor is a zero-training read over a draws dump with per-draw
logit retention. The cheapest flavor on the list for us.

## 2. Linear value probes — the cheapest *trained* selector

"What Frozen VLAs Already Know About Success" fits **linear ridge
probes** on frozen VLA features against Monte-Carlo success targets
(success episodes get discounted-outcome values, failures get 0)
and finds strong value-like structure the imitation objective never
asked for: R² ≈ 0.74 on held-out demos for π0.5 features vs 0.20
for proprioception, and ~92% matched-pairwise success ordering
(94.2% for the primary probe; 92.16% mean over ten runs) against a
50% shuffle control.

As a selector over 16 sampled π0.5 chunks it lifts push-plate
success 26.7% → 44.3% (p=0.003; random-choice baseline 33.7%). The
caveat our one-line hook missed: **that result is not probe-only** —
each candidate prefix is rolled out in the simulator from a
snapshot before the probe breaks ties among successes, at ~2.1× the
wall-time of random sampling. The honest reading: frozen features
carry usable value structure (solid), and the probe adds value on
top of rollout screening (real but simulator-assisted). Gains
collapse where the policy already exceeds ~90%, and cross-benchmark
transfer is weak. Also independent evidence for the #6/#17 prior
that frozen trunks carry more task structure than their action
heads use.

## 3. TapSampling — a progress verifier plus cheap candidate generation

Two learned add-ons over a frozen policy: an **action-VAE** that
encodes a handful of true policy samples into a latent posterior
and then decodes ~16 extra candidates cheaply (0.49 s vs 2.64 s for
true policy sampling), and a **task-progress verifier** trained
with labels that cost nothing — normalized progress from expert
demos as positives, *time-reversed* action sequences as negatives.
Surviving candidates are weighted-averaged, not argmaxed.

Numbers: CALVIN ABC→D avg success length 3.30 → 3.51 on OpenVLA;
π0.5 on LIBERO-Long 96.8% → 98.0%; real-Franka π0 78.3% → 83.3%;
claims 12× cheaper verification than RoboMonkey at 16 candidates.
Caveats: near-zero gains on saturated tasks; the linear-progress
label is admittedly a contrastive signal, not calibrated progress;
needs semantically reversible action spaces.

## 4. RoboMonkey — the VLM-verifier original, with a portability lesson

The pattern-setter: sample N̂ actions at temperature, fit a Gaussian
(majority-vote the gripper bit), sample K̂ cheap candidates from
the Gaussian, score each with a fine-tuned VLM verifier (LLaVA-7B,
Bradley-Terry on 20M synthetic pairwise comparisons ranked by RMSE
to ground truth), execute the argmax. Also contributed the
inference-time scaling law (action RMSE vs samples follows a power
law) that CoVer builds on.

Numbers: real-WidowX OOD 60% vs 35% greedy; SIMPLER ID 47.5% vs
38.5%. The portability lesson matters more: **both VLA-ATTC and
CoVer report RoboMonkey transferring poorly to flow-family
policies** (56.5% on LIBERO-Long where bare π0 scores 82.8%;
"failed catastrophically" in CoVer) — it was built around
single-step discrete-token VLAs. Verifier quality is
distribution-bound; a selector must be validated on the policy
family it will select for.

## 5. CoVer — scale the verifier, not the policy

A CLIP-style contrastive verifier (frozen SigLIP2 encoders, action
sequences embedded by a transformer, cosine-similarity score,
InfoNCE training on Bridge V2 + 16× synthetic instruction
rephrases). Deployment is hierarchical: a VLM generates K=8
instruction rephrasings at episode start; each step samples M=5
chunks per rephrase; verifier picks the best rephrase, then the
best chunk. Works on flow policies — actions are scored as
continuous chunks, no logits needed.

Numbers: the headline is a *matched-data* comparison — π0+CoVer
hits 57% ID / 61% OOD on SIMPLER vs 44% / 48.7% for a π0 whose
training consumed the same rephrase data (+22/+13 points, at 3.8×
base-policy FLOPs vs 16× for the training route); +45% absolute on
two real-WidowX tasks; verifier quality scales cleanly with model
size, data, and batch. Caveats: two real tasks on one platform;
rephrase-VLM cost excluded from latency; combining
verification *and* rephrase training is complementary (65.5% ID).

## 6. VLA-ATTC — a pairwise critic behind an uncertainty gate

Two ideas stacked. **Gate:** decode two chunks from the same
context with different seeds; if their DTW distance is below a
calibrated threshold, just execute (most states are easy); only
deliberate on disagreement — 23.3 Hz baseline degrades only to
20.8 Hz. **Deliberation:** 16 candidates (shared prefill, different
noise), ranked by a Relative Action Critic — a transformer that
takes *pairs* of chunks (plus their difference, proprioception, VLM
features) and outputs P(a_i ≻ a_j), run as a single-elimination
tournament. Preference data is automated: degrade expert chunks by
cutting flow-ODE integration steps, label expert ≻ degraded.

Numbers: LIBERO-Long π0 82.8% → 90.6–92.2%; π0.5 90.6% → 94–95.4%
(a ~51% relative failure-rate cut on the average). Caveats:
π0-family only, three real tasks, and gains shrink as the base
policy strengthens. The pairwise framing (rank, don't regress
absolute value) and the deliberate-only-when-uncertain gate are the
two ideas most worth stealing.

## 7. Look Before You Leap — distill tree search into a Q-model

Offline, in a resettable simulator, run MCTS over a *frozen* VLA's
own sampling distribution: candidates sampled from the policy are
the tree's edges, rollouts of the policy itself provide returns,
back-ups produce Q-labels — no human labels, no learned reward.
Distill those labels into a small ensemble Q-model (Qwen-0.8B +
LoRA, 5 bootstrapped heads). At deployment (simulator-free): sample
16–32 candidates, execute argmax of ensemble-mean Q minus an
uncertainty penalty plus the policy log-prior.

Numbers: EB-Habitat +13.8 avg points over five base models;
SimplerEnv π0 38.5% → 50.7%; a 9B model + selection beats a 27B
model at lower latency. Ablations are unusually clean: removing the
Q-model (−12.8 points) or multi-candidate sampling (−16.9) hurts
far more than swapping label source (−5.3). Caveats: needs a
resettable simulator with a success signal for the offline phase;
simulation-only evaluation.

## What transfers to us, and the gate that decides

Common to all seven: the *premise* — a meaningful gap between the
policy's average draw and its best draw. That premise is exactly
what our panel can measure offline, for free, from banked data,
before any verifier exists. The order-statistic best-of-K ladder in
`selection_ceiling_results.py` (K = 1..10, exact, no Monte Carlo)
plus its selector diagnostics (is the best draw concentrated or
uniform? does oracle gain concentrate in high-dispersion frames?)
is the adjudicator. It runs the moment the molmo2 endpoint's
per-draw dump lands (~08-08).

If the ceiling is small, all six flavors are dead on our panel and
we will have spent zero training on finding out. If it is large,
the flavor order for us is roughly the cost order above: MG-Select
first (zero-training — AR-100k is already condition-dropout
trained and `--mask-state` exists), a linear probe on frozen
features second, trained critics last — with RoboMonkey's
portability failure as the standing warning to validate any
selector on our own policy family, and each escalation needing its
own pre-registration.

Two structural caveats to carry: our panel is offline chunk-MAE, so
selector gains that depend on *closed-loop* effects (VLA-ATTC's
gating, TapSampling's per-step filtering, anything
rollout-assisted) are invisible to it — the panel bounds
open-loop selection only. And several headline numbers above come
from settings where the baseline was weak (RoboCasa at 5%, EB at
~38%); the closer a policy is to saturation, the smaller every
reported gain gets, which is also what our ceiling read will tell
us directly.
