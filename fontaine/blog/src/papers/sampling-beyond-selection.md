# Sampling beyond selection — noise tickets, variance gates, and the energy score

**Papers:** You've Got a Golden Ticket
([2603.15757](https://arxiv.org/abs/2603.15757), read at v3), DVAC
([2606.03847](https://arxiv.org/abs/2606.03847)), and Energy Policy
([2510.12483](https://arxiv.org/abs/2510.12483) — the paper's
actual title is "Fast Visuomotor Policy for Robotic Manipulation";
Energy Policy is the framework name). Banked across the
2026-08-05/06/07 lit slices; re-read at full-text depth for this
page. **Fed:** #1 (the sampled-draws program — Golden Ticket and
DVAC are its inference-time cousins) and the panel's scoring
methodology itself (Energy Policy is where our adopted energy-score
column comes from). Complements
[test-time selection](test-time-selection.md): those seven papers
*pick among* draws; these three change what you draw, when you
commit, or how you score the spread.

## The theme

A generative policy's noise input is usually treated as plumbing.
These three papers each take the sampling side seriously in a
different phase of the pipeline: **Golden Ticket** optimizes the
initial noise itself (weights frozen, search at deployment-prep
time), **DVAC** reads the solver's intermediate states as a free
uncertainty signal (test time, training-free), and **Energy Policy**
moves the distributional thinking into the *training objective* (a
strictly proper scoring rule instead of denoising). Together with
the selection cluster they map the full lifecycle of a draw: shape
it, score it, commit to it.

## 1. Golden Ticket — one searched noise vector (2603.15757, v3)

The lottery-ticket hypothesis transplanted to control: for a frozen
diffusion/flow policy there often exists a single fixed initial
noise vector that beats fresh Gaussian sampling when substituted at
every call. Search is black-box — random search or CEM over
candidate tickets, scored by Monte-Carlo episodic return over a
handful of rollout environments; no weight updates, no extra
networks. The v3 headline: golden tickets beat Gaussian noise on
**46 of 51 task–policy pairs** (at least match on 49), spanning
flow MLPs, diffusion policies, SmolVLA on LIBERO, GR00T-N1.5 on
SimplerEnv, and four real Franka tasks (real-hardware searches cost
under an hour of rollouts each). Gains can be dramatic
(put-eggplant-in-sink 21% → 76%; a real cup-push 40% → 100%), and —
the property that pairs with distillation — **improvements grow at
fewer solver steps**: DDIM-2 tickets even beat the DDIM-8 base
policy on several tasks.

Two of our banked claims needed correcting at full-text depth. Our
"38/43 tasks" was the **v1 abstract**; the paper is at v3 with the
larger 46/51 sweep — version drift, same conclusion. And our
"LIBERO-Spatial regressed −3%" was imprecise: no such cell exists.
The real structure is sharper and more useful — **per-task tickets
always gain** (Spatial +13 points), but the best *single shared*
ticket per suite regresses in all three suites (Spatial −2.6,
Object −12.0). Tickets are task-local objects; universality is the
exception, not the rule (their own multi-task CEM can find joint
tickets, at a cost on 2/7 tasks). Honest caveats they print:
tickets make inference deterministic (top-k ticket sampling
mitigates), a searched ticket can fail hard at unsearched table
positions, and one sim task (robomimic Square) clearly regressed.

For us the standing asymmetry is the interesting part: **their
search needs environment rollouts; our panel is the offline
criterion they lack.** Scoring M candidate tickets by probe-subset
MAE through the existing `sample_actions(noise=...)` hook is a
CPU-launchable screen, and the SnapFlow student made it ~30× less
compute per candidate (1-NFE vs Heun-30). The wrinkle the deep read
adds: our 1-NFE student *compiled away* most of its draw spread —
if the noise input barely moves the output, the searchable ticket
space may have collapsed along with it. The cheap pre-check is the
student's σ_draw, already banked (draw-averaging gain −0.236 vs the
teacher's −1.258): the ticket screen should target the *teacher's*
noise space first, or verify the student still responds to noise at
all. Needs its own pre-reg before any number is read.

## 2. DVAC — the solver trajectory as a free uncertainty meter (2606.03847, v1)

Training-free replan timing for flow policies. During Euler
integration, every step exposes a clean-action estimate; DVAC
computes, per future action index, the **variance of those
estimates over the last L=5 denoising steps**, executes the chunk
prefix up to the first index whose variance crosses a threshold,
and replans there — discarding the unstable tail instead of either
committing to all of it (open-loop) or replanning every step
(expensive and jittery). The threshold is scale-adaptive (mean +
2σ over a rolling buffer of recent calls) — and their CALVIN
ablation shows this is load-bearing: every *fixed* threshold they
tried scored below baseline. Numbers, verified: π0.5 on LIBERO
0.948 → 0.980 with **43% fewer replans**; real dual-arm tasks
improve over both fixed-chunk extremes while cutting wall-time.
Theory garnish: under a Lipschitz assumption the integration error
at an action index is bounded proportional to the root of that
variance.

The full text adds two cautions our one-liner missed. The gains are
**π0.5-specific in their own tables** — on Qwen-backbone policies
the lift shrinks to under a point. And the variance is a proxy, not
calibrated uncertainty: their named failure mode is
stable-but-wrong (variance stays low before a bad grasp, replanning
comes too late).

For us, unchanged but sharpened: our panel is offline chunk-MAE —
replan timing is invisible to it — so DVAC is a rollout-phase lever
for the rig/sim stage, banked, not actionable now. What *is*
actionable is the shared signal: DVAC's per-index tail variance is
the same object as our dispersion machinery (the fairness read's
dispersion-quartile deficit ran 0.23 → 1.42 monotone across
quartiles, and `selection_ceiling_results.py` computes
dispersion-vs-gain quartiles from any draws dump). If oracle gain
concentrates in high-dispersion frames at the molmo2 endpoint read,
selection *and* DVAC-style commit-gating draw from one signal — one
number will license or kill both branches for the rollout stage.
One more portability note: a 1-NFE student has **no denoising
trajectory** — DVAC-style signals simply don't exist on the
distilled deployment config; the teacher would have to serve them.

## 3. Energy Policy — the scoring rule moves into the loss (2510.12483, v1)

The paper behind our energy-score column, finally read deeply. The
objective: train the policy directly on the **energy score**, a
strictly proper scoring rule — per target action, draw *two*
independent samples from the model and minimize
‖â¹−a‖ + ‖â²−a‖ − ‖â¹−â²‖ (α=1). Two attraction terms pull samples
toward the data; the repulsion term keeps them apart — matching the
full conditional distribution, not its mean, with no iterative
solver anywhere. Sampling is **one forward pass** of a small
transformer plus an "energy MLP" head with adaLN-Zero noise
injection; their ablation shows that injection mechanism is
load-bearing, not a detail (swap adaLN for concatenation and
Square-mh craters 0.85 → 0.31). Results are strong for the model
class: robomimic/Franka-Kitchen/MimicGen parity-or-better against
CARP and diffusion baselines at 3–70× lower latency, a real-robot
dynamic-catch task where speed is the task (13/20 vs diffusion's
8/20), and native multimodality without denoising.

The read-between-the-lines section matters here: the paper has **no
limitations section**. The benchmarks are classic small-scale BC
(no VLA, no language conditioning, largest model ~18M params);
several "wins" are ties per-task (MimicGen 0.86 vs CARP's 0.85 with
losses on three tasks); success numbers average the best three
checkpoints — checkpoint selection on eval; and multimodality
evidence is one qualitative PushT figure. None of this kills the
idea; it does mean "energy loss scales to VLA-size policies" is
undemonstrated, and anyone citing the paper for that is citing
hope.

What we actually took — and had already taken before this deep
read — is the **metric, not the loss**. Read 4 of the draws
fairness program scores draw sets with exactly this scoring rule,
and it flipped our AR-vs-flow story: flow wins the energy score
**5.9308 vs AR's 8.7696** while *losing* single-draw MAE — the
strictly proper score credits the flow expert's honest spread where
per-draw MAE punishes it. The owner adopted ES as the candidate
distributional column, and `energy_score_results.py` now waits on
the molmo2 endpoint dump to run the same comparison on the AR
family's own draws. The loss-side idea — a small energy head as a
*third objective family* beside flow matching and distillation — is
a real but unpre-registered thought; it would need its own screen and
this paper's small-scale evidence wouldn't carry it.

## What transfers, what doesn't, and what it fed

**Transfers:** the energy score itself (already adopted, already
paying — it is the one column where the flow family's spread is an
asset, and it comes free from any `--dump-draws` npz). Golden
Ticket's core finding transfers as a *question* our instruments can
answer offline — is there panel headroom in noise space? — with the
panel substituting for their rollout oracle. DVAC's variance signal
transfers as the rollout-stage twin of our dispersion reads.

**Doesn't transfer:** Golden Ticket's search protocol (rollout
returns we don't have), and any assumption of ticket universality —
per-task tickets are the honest unit. DVAC as-is (offline panel
can't see replan timing; gains look backbone-specific; inapplicable
to a 1-NFE student with no solver trajectory). Energy Policy's
training-objective claims at our scale — undemonstrated, small-N,
no-limitations-section evidence.

**Fed:** #1 — the ticket screen stays queued with a sharpened
design (teacher-first noise space, student σ_draw pre-check;
corrected bank: 46/51 at v3, per-task-not-shared tickets); the ES
column is adopted methodology. #19 — the dispersion-vs-gain
quartile read at the molmo2 endpoint now adjudicates *three*
branches at once: selection, DVAC-style commit-gating, and the
ticket search's headroom prior. #12 — one more entry for the
distillation ledger: every trajectory-reading lever (DVAC) and
noise-space lever (tickets) binds to the *teacher*; the student's
speed is bought by exactly the structure those levers exploit.
