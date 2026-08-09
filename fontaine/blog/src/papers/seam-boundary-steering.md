# SEAM: closing the chunk seam in noise space, for 1% overhead

*Lit slice 2026-08-09 (work session 14:1xZ, in the adamc_100k
shadow). SEAM ([2607.04609](https://arxiv.org/abs/2607.04609),
"Smooth Execution of Action-Chunked Motion for Vision-Language-Action
Policies"). Fed #22 (a third, cheapest entry in the async/boundary
bridging family — the design input beside RTC and A2C2), #1 (the
cross-chunk half of the boundary-jerk term the
[SDN read](label-free-selection-signals.md) left unmeasured), and a
banked record-only read on our own npz stacks (below).*

## The paper in plain words

Robot policies like ours don't output one action at a time — they
output a *chunk* of, say, 50 future actions, execute the first 10,
then generate the next chunk from scratch. Each chunk is generated
from fresh random noise, and when a task can be done several equally
valid ways, two consecutive chunks can each pick a *different* valid
way. The robot then visibly jerks at the boundary as it switches
plans mid-motion. SEAM's observation: at the moment you generate the
next chunk, you are holding a perfectly good reference for what the
motion was *about* to do — the 40 unexecuted actions of the previous
chunk. So while the new chunk is being denoised, SEAM gently pulls
the early part of it toward that leftover tail, using a closed-form
nudge after each denoising step. No training, no gradients through
the network, almost no extra compute (+1%) — and the boundary jerk
drops by ~28% with task success unchanged (slightly up, in fact).

## The method, exactly

Standard flow-matching sampling for the next chunk, plus one
appended operation per Euler step. Let `x̃ᵢ₊₁` be the latent after
Euler step `i+1` at time `tᵢ₊₁`, and let `a^al` be the **aligned
prior**: the previous chunk's unexecuted tail, extended to full
chunk length by repeating its last action. SEAM forms the
time-interpolated target `rᵢ₊₁ = (1 − tᵢ₊₁)·a^al[1:M]` on the first
`M` guided positions and applies the analytic gradient of
`‖x̃ᵢ₊₁[1:M] − rᵢ₊₁‖²`:

`xᵢ₊₁[1:M] = x̃ᵢ₊₁[1:M] − 2λ(1 − tᵢ₊₁)·(x̃ᵢ₊₁[1:M] − rᵢ₊₁)`

The `(1 − tᵢ₊₁)` schedule makes the nudge weak while the latent is
still mostly noise and strong as it converges onto the action
manifold. That is the whole method: no policy backward pass, no
activation storage, `O(N·M·D)` scalar work (~3.8 ms on a 282 ms
denoise loop). Fresh Gaussian init per chunk is kept deliberately —
diversity of modes is preserved, only the *landing* is steered.

## What they ran

LIBERO-10, π0.5 base, H=50 executed K=10 (tail L=40), 130
episodes/task:

| | success % | boundary jerk | discontinuity | cost |
|---|---|---|---|---|
| π0.5 unguided | 94.8 | 0.195 | 0.172 | 1.00× |
| **SEAM** | **95.7** | 0.141 (−28%) | 0.126 (−27%) | **1.01×** |
| RTC (backprop guidance) | 95.1 | 0.090 (−54%) | 0.089 (−48%) | 1.22× |
| ACT temporal ensembling | 82.7 | 0.031 (−84%) | 0.062 (−64%) | 1.00×* |

The table is the family map in one place: temporal ensembling
smooths hardest and *destroys task success* (−12 pts — it
over-smooths contact timing); RTC smooths more than SEAM but pays
22% inference and, their qualitative read, can lock into a failed
alignment; SEAM keeps corrective freedom. Ablations: λ peaks at 0.1
(0.15/0.2 erode success to 92.8/89.5 — aggressive guidance is not
free); guiding all action dimensions beats position-only; window M
is a clean smoothness knob (success stable 94.7–96.3 across
2≤M≤20).

## What transfers to us

- **It is a deployment-time recipe for exactly our object.** A
  frozen flow-matching expert sampled per-chunk with an Euler-class
  solver — SEAM bolts onto that with no training and ~1% cost. Our
  mainline chunk length is 50, their H exactly. When the #16 rig
  bench exists and #22 unparks, SEAM enters the design menu as the
  *cheapest* bridging arm — the [async family
  page](2026-08-07-async-chunk-execution.md) had RTC (1.22×,
  collapses at deep delay) and A2C2 (a trained residual head); SEAM
  undercuts both on cost and needs neither training nor rollouts.
- **It names the term our SDN read could not see.** The
  [SDN/jerk-pick read](label-free-selection-signals.md) measured
  *within-chunk* smoothness of our flow draws — null: ODE draws are
  uniformly smooth. SEAM's target is *cross-chunk mode
  incompatibility*, which no per-draw statistic sees. Those are
  different terms; our flow-side null does not cover the seam.
- **A banked, record-only read on our own data** (hook, not a
  commitment): our panel npz dumps store predicted chunks on
  temporally ordered frames of the same episodes. A CPU read can
  measure the *incompatibility* directly — for panel frames Δt=K
  apart in the same episode, compare the earlier chunk's tail
  against the later chunk's head on their overlap (the SDN-read
  pattern: a pure function of banked stacks, zero GPU). That would
  tell us whether our expert even *has* a bifurcation problem at
  k4l2 geometry before any deployment machinery is argued about.
  Filed on #1/#22.

## What doesn't transfer

- **Open-loop MAE cannot price it.** SEAM's win is closed-loop
  (jerk, discontinuity, success under execution); our panel scores
  chunks independently and would read a SEAM-steered chunk as
  slightly *worse* (it is pulled away from the fresh-noise optimum
  toward continuity). This is a #16-gated idea by construction —
  the offline leaderboard must never be asked to validate it.
- **Their delay regime is benign.** LIBERO execution is synchronous
  — generate, then execute, tail fully available. Our #22 problem
  statement includes the *async* case (chunk n+1 generated while n
  is still executing, observations stale); the survey's regime
  table says deep-delay is where naive methods collapse. SEAM
  assumes the tail is available at sampling time — under async
  overlap the "tail" is partially counterfactual. Composable in
  principle (the aligned prior just gets staler), unmeasured in the
  paper.
- **π0.5-scale, 10-task suite, one embodiment.** Effect sizes carry
  the usual transfer caveat; the mechanism (independent Gaussian
  latents → incompatible modes) is architecture-general and is the
  part we import.

## Verdict

The cheapest published answer to the chunk-boundary problem, and
the first that is strictly inference-time-closed-form. Nothing to
run today: #22 stays parked on #16, and the boundary-incompatibility
CPU read is banked as a hook on our existing npz stacks — worth
executing in some idle window *before* any rig work, since a null
(our chunks agree at the seam already) would close the whole
direction for our stack at zero cost.

*Update 2026-08-09 ~15:2xZ: the read
[executed same day](../posts/2026-08-09-boundary-incompat-results.md)
— NOT a null. Seam disagreement ≈ 1.1–1.3× model error, boundary
jump 11–14× per-step motion, and a shared noise ticket deletes the
noise-induced term entirely (dt→0 intercept 2.07 vs 6.04). The
direction this page hoped to close cheaply is instead confirmed with
a measured target.*
