# Noise-space steering III: attribution, and a selector that needs no judge

*Read 2026-08-08 (standing lit slice, the GPU-busy window while
golden-ticket stage 3 and the molmo2 draws arm run to their ~08:1xZ
landings). Two sources, found by scanning for post-Golden-Ticket work
on initial-noise choice: Noise-Space Attribution and Control of the
Chunk-Boundary Artifact
([2603.11642](https://arxiv.org/abs/2603.11642)) shows *what kind of
variable* initial noise is — directional, context-dominated, and only
controllable when the sampler keeps the noise→action path intact —
and SDN — Self-Improving VLA Policies: Selected Diffusion Noise for
Spurious-Robust Action Smoothing
([2606.14084](https://arxiv.org/abs/2606.14084)) ships a
*verifier-free per-step noise selector* (12 candidates, no external
judge) worth +8–18 pp success on π₀/GR00T. Fed:
[#1 noise-draw ensembling](../ideas/01-noise-draw-ensembling.md)
(three priors written into the upcoming per-dataset-tickets pre-reg)
and [#19 AR sampled-draws](../ideas/19-ar-sampled-draws.md) (a
concrete, label-free selector to hold against the banked selection
ceiling — its smoothness half is computable on our already-banked
draw stacks for free).*

## Why these two, today

The ticket screen just adjudicated: one searched noise vector is REAL
(complement Δ −0.924), the effect is directional not norm, and the
noise-structure ladder's entry condition is met. The next rung —
per-dataset tickets — gets its pre-reg after R3 lands today. These
two papers are the freshest evidence on the two questions that
pre-reg has to take a stance on: *how task-local are noise effects*
(attribution paper: very — with numbers), and *what selects noise at
deployment when there is no panel to search against* (SDN: the model
itself, plus a smoothness prior).

## Attribution: noise is a mechanism variable, but only through an intact path (2603.11642)

The paper's object is the **chunk-boundary artifact** — the jerk
spike where one action chunk hands off to the next (jerk = second
difference of the action sequence; their boundary metrics contrast
jerk at chunk phases {0,1} against interior phases). The standard
story treats it as stochastic debris. Their claim: hold the
observation fixed, vary only the latent noise, and the artifact level
is a *systematic function of the noise draw* — attributable,
steerable, and causally linked to task success.

Three results carry the page:

**1. The information-path result.** Same Diffusion Policy checkpoint,
three samplers: DDPM's per-frame correlation between a noise
direction and boundary jerk is ~0.11/0.05; *zero-variance* DDPM —
determinism alone — is unchanged; DDIM (η=0) jumps to **0.96/0.93**.
Removing per-step randomness is not what restores control; DDPM's
Markovian reverse updates remix the signed direction of the initial
noise even when deterministic. Only when the sampler preserves the
noise→action map does directional structure in noise space survive to
the output. **Transfer to us:** our teacher decodes a deterministic
Heun-30 ODE — the flow analog of the intact-path regime, and their
OpenPI main experiments are flow-matching policies (π₀/π₀.5-LIBERO).
This is the published *why* behind our screen even finding structure:
a 64-ticket search would have read null through a path-breaking
sampler at the same checkpoint quality. It also sharpens the ladder's
future student rung: any decode change that breaks the path (or an
escalation to stochastic samplers) forfeits the channel, separately
from the draw-spread collapse we already measured on SnapFlow.

**2. The locality numbers.** 192 contexts × 16 shared noises: the
variance decomposition of boundary gap is **59.1% context main
effect, 1.4% noise main effect, 39.4% interaction**. The globally
best noise is optimal in only **3.1%** of contexts; per-context
selection removes 93.8% of the mean gap. This is the quantitative
form of Golden Ticket's shared-vs-per-task table, measured on a
different artifact with a different criterion — and it is the prior
the per-dataset-tickets pre-reg should encode: expect
interaction-dominated structure (a shared ticket captures the 1.4%
main effect; per-dataset search is aiming at the 39.4%). Our R4a
read (per-dataset per-ticket argmin disagreement, banked at stage 1)
is exactly this decomposition's cheap shadow on our own panel.

**3. Steering moves task outcome, in both directions.** Local
directional search (12 random unit directions, probe at α=±0.5, pick
the best separator) finds near-linear jerk response (r 0.90–0.97).
Steering along it at held-out matched continuations — same rollout
prefix, paired futures, only boundary noise differs — moves success
0.033→0.717 (+68 pp, n=60 pairs) in one context; and the *sign* of
the useful direction flips by context (some contexts want *higher*
artifact). No fixed direction transfers across contexts; their
adaptive online version is honest about being not-yet-stable.

**The caveat that matters for us:** their criterion is a rollout
property (boundary jerk, success under execution). Our panel is
offline chunk MAE and cannot see chunk hand-offs at all — a ticket
that wins our panel could carry arbitrary boundary artifact (the #16
offline-vs-rollout gap, in its sharpest form yet). Banked as a
named unknown on ticket 33: if the flow decode ever reaches a
rollout, measure boundary jerk ticket-vs-stable-key before trusting
the panel win.

## SDN: selection without a judge (2606.14084)

SDN is test-time-only, weights frozen, and needs no external
evaluator — the niche our #19 selection-ceiling read exists to bound.
Per inference step it draws **12 candidate noises**, decodes each,
and picks in two stages:

1. **Contrastive grounding.** Decode the same candidates under an
   *object-masked* observation (zero out the target object's pixels).
   Score each clean candidate by k-NN distance to the masked-decode
   action set minus distance to the clean set — keep the top 5 that
   look least like what the policy does when it cannot see the
   object. This is a self-diagnostic: π₀ drops 63.9%→24.4% under
   masking, so masked decodes are a live sample of the policy's
   ungrounded-shortcut behavior.
2. **Kinematic stability.** Among survivors, pick the minimum-jerk
   trajectory (RMS third difference over an *extended* chunk —
   decoded past the executed horizon to expose delayed oscillation).

Numbers: π₀ SimplerEnv 63.9→72.6 avg (+8.7 pp), GR00T-N1.6
49.5→54.5, real-robot ALOHA +18.3 pp avg; jerk down 5%. Cost: ~245
ms/step on a consumer GPU — the 12-candidate decode plus one masked
forward.

**The ablation that matters:** smoothness-only gets **+16.7 pp** of
the +18.3 real-robot gain; grounding-only +13.3; both +18.3. The
cheap, judge-free, *second-forward-free* half — pick the smoothest
of N draws — is most of the method.

**Transfer to us, concrete and free:** we hold banked `--dump-draws`
stacks (teacher drawsprobe draws-10, the 64-ticket stack, and the
molmo2 draws10_t1 full stack landing today). Min-jerk selection is a
pure function of the draw stack — no forwards, no labels. A
record-only CPU read can place "pick the smoothest draw" on the
selection-ceiling ladder (single draw → mean-of-N → jerk-pick →
oracle best-of-N) for both families, before anyone builds a
deployment selector. If jerk-pick recovers a nontrivial slice of the
oracle gap on our panel, #19's escalation has a published,
verifier-free candidate; if it recovers nothing, the SDN prior is
falsified *for our stacks* at table cost.

*Executed same session (`jerkpick_selector_results.py`, oracle-green;
results in `reports/analysis__jerkpick_selector.json`): a clean
two-family split. On the flow teacher's fresh-noise draws-10 stack
jerk-pick is **null on every diagnostic** — oracle agreement 10.5%
vs a 10% null, Spearman(jerk, MAE) +0.13, −2.3% of the oracle gap
recovered (i.e. slightly worse than an average single draw); the
ticket-64 stack reads the same. Heun-30 ODE draws are uniformly
smooth — the criterion has nothing to grip. On the AR q4 stacks the
prior is **real but small and temperature-monotone**: 5.6% / 7.5% /
20.9% of the oracle gap recovered at T = 0.5 / 0.7 / 1.3, Spearman
+0.36 — jerky sampled-token decodes really are bad decodes, and the
wilder the temperature the more a smoothness filter rescues. On
neither family does jerk-pick approach mean-of-N, so the family
decodes stand. SDN's smoothness prior: falsified for our flow
stacks, confirmed-in-miniature for AR — consistent with SDN's own
setting (stochastic diffusion policies, not deterministic ODE
decodes). The molmo2 draws10_t1 stack gets the same read when the
#19 arm lands.* The grounding half is heavier (one masked forward per frame,
and our observation masking would need object boxes we don't have) —
noted for [#11 visual grounding](../ideas/11-visual-grounding.md),
not queued.

**Caveats:** SDN's gains are success-rate under rollout on SimplerEnv
/ ALOHA — the same offline-vs-rollout gap applies in reverse (our
panel may under- or over-credit a smoothness pick); their k-NN
grounding is tuned (N=12, k≤10, M=5) with diminishing returns past
N=12; and jerk-as-criterion overlaps with what our chunk MAE already
partially rewards, so the honest expectation for the offline read is
modest.

## What moved where

- **#1 (noise ladder)** — three priors banked for the
  per-dataset-tickets pre-reg: (i) interaction-dominated locality is
  the published expectation (1.4% noise main effect vs 39.4%
  interaction); (ii) the channel exists *because* our decode is
  path-intact — any sampler change re-tests the whole ladder; (iii)
  boundary artifact is a named unknown of ticket 33 the panel cannot
  see (rollout-gated read banked).
- **#19 (selectors)** — jerk-pick joins the selector shortlist with
  published rollout numbers and a zero-cost offline evaluation path
  on banked stacks; new record-only analysis item queued.
- **#11 (grounding)** — SDN's masked-decode contrast noted as a
  self-diagnostic pattern (needs object masks we don't have; not
  queued).
