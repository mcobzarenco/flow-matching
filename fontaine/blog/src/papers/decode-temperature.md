# Decode-time stochasticity — when temperature matters for tokenized-action policies

**Papers:** Understanding Multimodal Failure in Action-Chunking
Behavioral Cloning ([2605.22493](https://arxiv.org/abs/2605.22493)),
MARS ([2605.29766](https://arxiv.org/abs/2605.29766)), Understanding
Behavior Cloning with Action Quantization
([2603.20538](https://arxiv.org/abs/2603.20538)), BOKBO
([2605.30660](https://arxiv.org/abs/2605.30660)), plus Discrete
Diffusion VLA's decoding ablation
([2508.20072](https://arxiv.org/abs/2508.20072)) as a supporting
data point. Read 2026-08-07, timed to the #19 T-sensitivity rungs
scoring on the local GPU as this page is written. **Fed:** #19 — the
dT diagnostic's interpretive frame (what a temperature *should* do
to a mean-of-draws panel number, so the record-only table lands with
a prior instead of as bare digits) and a fresh caveat on the
selection rung; a theory anchor for the q-token AR trunk itself
(#5/#19).

## The theme

Our sampled-draws program treats the AR policy's decode temperature
as a fixed, registered constant: the primary is T=1.0, and tonight's
rungs re-score the frozen checkpoint at T ∈ {0.5, 0.7, 1.3} purely
to put an error bar on how sensitive the mean-of-10 numbers are to
that choice. The question this cluster answers: what does the field
actually know about *when* stochastic decoding of actions helps,
when it is inert, and when it hurts? The short version — stochastic
capacity matters only where the demonstrated action distribution is
genuinely multimodal, and most frames of most manipulation tasks are
not; where it is unimodal, deterministic (or sharpened) decoding is
at least as good and often better. That is exactly the regime our
draws10_t1 readout measured from the other side: Δ_AR = −0.145,
mean-of-10 barely beating greedy, the pre-registered mean-collapse
shape.

## 1. Multimodal failure in action-chunking BC (2605.22493)

The anchor read. The paper proves and then measures that the two
standard ways of making a chunking policy stochastic fail for
*different structural reasons*: latent-variable policies (CVAE-style)
lose the mode information to posterior–prior KL regularization — at
ACT's default β=10 the policy is provably driven toward unimodality
(their bound: preserving modes needs I(A;Z|S) above a Fano floor,
while KL strength caps it at C/β) — and action-space generators
(flow/diffusion heads) are geometrically capped: a smooth map from a
unimodal base can only represent about 1 + L/Δ well-separated modes,
so low-Lipschitz samplers either bridge between modes with invalid
trajectories (bridge fraction ~0% at 2 modes → ~51% at 16) or snap
discontinuously. The empirical kicker for us: on the nearly unimodal
UR3 BlockPush, the *deterministic* baseline beats every generative
variant (1.98 vs 0.82–1.78 goals), and on their 16-mode synthetic
task near-complete mode *coverage* still yields only 0.37–0.52
success — diversity is necessary for multimodal tasks but nowhere
near sufficient, and on unimodal ones it is pure overhead.

**Transfers:** this is the published mechanism behind our measured
mean-collapse (E2 of the draws10_t1 readout): where the conditional
action distribution is near-unimodal, sampling buys nothing that the
mean didn't already have, and pooling draws re-averages whatever
spread the sampler added. It gives tonight's dT table a directional
prior worth writing down *before* the numbers land: on a
mostly-unimodal panel, T<1 (sharpening toward the local mode) should
move pooled chunk/first MAE little or slightly down, and T=1.3
(flattening) should hurt more than T=0.7 helps — an asymmetry, not a
symmetric bump. Record-only either way; if the table comes out
inverted, that is evidence the q4 subset carries more multimodality
than the family baseline suggests, which would be worth knowing.
**Doesn't transfer:** their β/Lipschitz analysis targets CVAE and
flow heads; our AR softmax over q-tokens has neither bottleneck —
its capacity for multimodality is the full categorical per token,
which is exactly why the draws program tests sampling on it at all.

## 2. MARS — stochasticity only when it matters (2605.29766)

The engineering mirror of the same claim: a policy that *selectively*
activates stochastic multimodal generation in task phases with real
behavioral diversity and runs deterministic elsewhere, reporting
+16.7% real-world success with an 83% inference-latency cut over
always-stochastic baselines (8 sim + 4 real tasks). The
counterintuitive result they highlight: on near-deterministic tasks
MARS *trains* more efficiently than a purely deterministic policy —
modeling the small pockets of genuine diversity helps even when most
of the trajectory is prescribed.

**Transfers:** supports the per-frame view over the per-run view —
the interesting unit is the frame, not the policy. Our
selection-ceiling diagnostics (dispersion-vs-gain quartiles in
`selection_ceiling_results.py`) are already shaped to answer the
MARS question on our data: if best-of-K headroom concentrates in the
high-dispersion quartile, phase-adaptive stochasticity is the shape
of any follow-up; if it is flat, even that is not worth building.
**Doesn't transfer:** MARS is closed-loop success on live rollouts;
our panel is open-loop MAE against one demonstration — a
mode-*matching* metric that structurally cannot reward picking a
valid non-demonstrated mode.

## 3. Theory anchor — BC with action quantization (2603.20538)

First sample-complexity analysis of exactly our trunk recipe:
behavior cloning with quantized actions under log-loss. Result:
optimal sample complexity matching known lower bounds, with
quantization error compounding only *polynomially* along the horizon
(not exponentially) given stable dynamics plus a policy-smoothness
condition; they characterize which quantization schemes satisfy the
conditions and give a model-based augmentation that provably tightens
the bound. Abstract-level read (no experiments to audit).
**Transfers:** a clean citation that the q-token + CE-loss trunk is
not just an engineering convenience — it is the statistically
efficient estimator class for this problem. **Doesn't transfer:** no
decode-time content at all; it is about what training converges to.

## 4. BOKBO — temperature draws as a safety substrate (2605.30660)

A conformal abstention layer over VLA policies: sample K candidate
actions, score them with a learned violation predictor, abstain with
finite-sample distribution-free guarantees on executed-violation
rate (per-task Mondrian calibration lifts the worst per-task
conditional hold rate 0.71 → 0.93 on OpenVLA-OFT/LIBERO variants).
Relevant to us for two reasons. First, its candidate generators are
exactly our draws menu — token-level temperature sampling among them
— used not to improve the mean but to expose the policy's spread to
a downstream decision layer; a use of draws the panel-MAE lens never
sees. Second, its measured caveat: **policy-internal confidence
correlates poorly with actual violations under perturbation
sampling** — learned predictors were needed. That is the second
independent strike against cheap probe-style selectors (after the
test-time-selection page's rollout caveat), banked on the #19
selection rung: the oracle ceiling read stays the gate, and any
selector that survives it should expect to need *trained* scoring,
not a free confidence readout.

## 5. Supporting data point — decode-order temperature in DDVLA (2508.20072)

Discrete Diffusion VLA's Table 7 (LIBERO-Goal): hard argmax 96.2%,
fixed T=1 96.4%, linear decay T 1→0 **97.4%** — mild exploration
early, sharp commitment late. Weak transfer (it schedules temperature
over *refinement iterations* of a discrete-diffusion decode, not
over an AR chunk, and it is closed-loop success), but it is the one
published cell we found where a temperature *schedule* beats both
extremes — a reminder that "which fixed T" (our rung question) and
"T as a schedule" are different axes, and the second is unexplored
in AR-VLA decoding. Parked as a hook, not an arm: nothing opens
unless the dT table shows sensitivity worth chasing.

## Where this leaves #19

The dT rungs land tonight with a written prior: near-flat response
with mild asymmetry against T=1.3 says the panel is
unimodal-dominated and the T=1.0 registration was safe (the expected
outcome); a monotone win for T=0.7/0.5 says greedy-adjacent decoding
is simply better on this metric and the mean-collapse story deepens;
sensitivity concentrated in high-dispersion frames says MARS-shaped
phase-adaptivity is the only version of this worth ever building.
All three readings are record-only against the frozen T=1.0 primary
— the rung exists to put an error bar on the leaderboard's
mean-of-10 rows, and this cluster says that error bar should be
small.
