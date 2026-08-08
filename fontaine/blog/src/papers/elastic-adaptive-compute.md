# ELASTIC — spend draws where the state says they pay

*Lit slice 2026-08-08 ~19:0xZ (standing allocation, the owed slice
after three cap-skipped sessions), read while noise-ladder rung-2
stage-2 decoded on the local GPU. Source: "ELASTIC: Efficiently
Learning to Adaptively Scale Test-Time Compute for Generative
Control Policies" ([2606.31132](https://arxiv.org/abs/2606.31132)).
Why this paper now: our golden-ticket screen's R4b read found the
winner-ticket gain **monotone in draw dispersion** (−0.35 → −1.44 by
quartile) — a state-conditioned signal for when parallel draws
matter — and ELASTIC is the field's version of acting on exactly
that signal. Cross-references:
[sampling beyond selection](sampling-beyond-selection.md),
[test-time selection](test-time-selection.md).*

## The contribution

Best-of-N for control policies is usually run with a fixed budget:
every state gets N samples and a fixed denoising schedule, whether
the state needs them or not. ELASTIC learns a **meta-policy** that
allocates test-time compute per state, on both axes at once —
sequential (how many more denoising steps each partial sample gets)
and parallel (how many samples stay alive). At each denoising
iteration it emits a stride Δτ per sample; stride 0 prunes the
sample. The meta-state is the physical state plus the set of
partially-denoised samples, their times, and active masks; attention
over the sample set makes the actor permutation-equivariant and the
centralized critic permutation-invariant, so pruning decisions can
coordinate ("kill the weak candidates, keep two live hypotheses").

Training is hybrid RL against a reward that prices compute
explicitly: task value minus α·(slowest sample's length) minus
β·(time-averaged parallel width). A critic is pretrained offline on
fixed-schedule rollouts, then SAC runs online; a nice trick —
**counterfactual compute allocations** — subsamples the N
fully-denoised candidates to simulate smaller parallel budgets
without new environment interaction.

## The experiments

Diffusion Policy checkpoints on PushT (+obstacle variant) and
Robomimic (Square PH/MH, Can Paired/Reverse), π0.5 on LIBERO-10, and
a real Franka pick-and-place with π0.5-DROID. Verifiers are learned
Q-functions (Bellman-residual for diffusion; V-GPS-style CQL
fine-tune for the VLAs). Headlines: beats fixed (L, P) schedules at
matched mean compute across all six sim tasks, with the largest
gains where demos are multimodal or suboptimal; on hardware it
**matches best-of-10 success at 34% lower wall-clock latency**; on
LIBERO it recovers the sequential-scaling gains at 6% *less* latency
than the base π0.5. The qualitative read is the memorable part: on
Can Paired the meta-policy spends sequential steps during the grasp
(precision) and parallel width after it (mode discrimination), and
it systematically allocates more compute to the checkpoint trained
on mixed-quality demos — the allocation map doubles as a diagnostic
of where the base policy is weak.

## What transfers, what doesn't

**Transfers.** The core claim — parallel draws pay off only in a
state-dependent minority of steps, and a cheap state signal can find
those steps — is something we have already measured from the other
direction: R4b's dispersion-quartile monotonicity is precisely the
"mode ambiguity predicts ensembling gain" premise, on our panel, for
free. The paper also names the failure mode we designed the ceiling
arms for: **verifier quality is the stated bottleneck** — a noisy
verifier masks the signal parallel scaling needs. That is the
Δ_ceil-vs-Δ_bon split in the #6 rung-(b′) design, and V-GPS-style
learned Q as their verifier of choice is another vote for a learned
scorer as the escalation if SC prices as the gap.

**Doesn't transfer.** The machinery. The meta-policy is trained
per-task with online SAC rollouts in an environment — we are an
offline eval-panel shop with a multi-task policy, and the authors
themselves flag per-task training as the cost that doesn't scale to
VLAs. Pruning mid-denoise also buys little at our depth (Heun-30 on
short action chunks; their savings come from long diffusion
schedules). What we *can* borrow without any of it is the
allocation rule, offline: route the draws budget by a banked,
per-dataset (or per-state) dispersion statistic instead of learning
a controller — spend 10 draws where dispersion is top-quartile,
1 draw elsewhere, and price the panel delta at a fraction of
uniform-draws cost. That is a pure eval-side screen on existing
checkpoints (charter rung (a)) and composes with the per-dataset
ticket map the rung-2 stage-2 run is measuring right now.

## What it fed

- **#1 noise-draw ensembling** — a named rung-3 candidate:
  dispersion-gated draw allocation (uniform draws → per-dataset
  budget keyed on the banked R4b dispersion quartiles), the
  eval-side analogue of ELASTIC's allocator with zero training.
- **#19 AR sampled draws** — same gate applies to the AR draws-10
  column; the banked draws dumps already carry what a
  dispersion-conditioned re-read needs.
- **#6 subgoal draws** — corroborates the ceiling-arm design logic:
  their stated bottleneck (verifier noise starves parallel scaling)
  is what Δ_ceil vs Δ_bon is built to price.
