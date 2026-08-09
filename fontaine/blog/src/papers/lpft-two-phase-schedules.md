# LP-FT: the two-phase schedule with a matched control and a theorem

*Read 2026-08-09 (lit slice, standing allocation — this one steered
directly by the owner's 10:38Z reframing of F-vs-K as schedule
curves a(t)·AR + b(t)·flow under a fixed compute budget). Papers:
[2202.10054](https://arxiv.org/abs/2202.10054) — "Fine-Tuning can
Distort Pretrained Features and Underperform Out-of-Distribution"
(Kumar, Raghunathan, Jones, Ma, Liang; ICLR 2022 oral), with the
mechanism follow-up
[2405.16747](https://arxiv.org/abs/2405.16747) — "Understanding
Linear Probing then Fine-tuning Language Models from NTK
Perspective" (NeurIPS 2024).*

**The paper in plain words.** When you adapt a big pretrained model
to a new task, you can either bolt a small new output layer on top
and train just that (cheap, but limited), or train everything at
once (expensive, and — this paper's point — subtly destructive: while
the random new layer is flailing early in training, its gradients
scramble the good features underneath). The paper proves, in a
simplified model, that the damage comes specifically from training
the backbone *while the head is still uninformed*, and shows a
two-step fix: first train only the head on the frozen backbone, then
unfreeze everything. That ordering gets the best of both — and it is
the cleanest published version of the schedule our F-then-joint
escalation rung proposes.

## What it contributes

The **feature-distortion theorem**: in overparameterized two-layer
linear networks, full fine-tuning from a random (or fixed, wrong)
head provably incurs high out-of-distribution error, because the
backbone moves to compensate for head error and distorts directions
the pretrained features already had right. The distortion is
front-loaded: it happens *while the head is misaligned*. Align the
head first (linear probe on frozen features) and the subsequent full
fine-tune starts with small head-error gradients, so the backbone
moves much less — LP-FT.

## The experiments it ran

Ten distribution-shift dataset pairs (Breeds-Living17/Entity30,
DomainNet, CIFAR→STL, CIFAR-10.1, FMoW, ImageNetV2, ImageNet-R,
ImageNet-A, ImageNet-Sketch), comparing linear probe (LP), full
fine-tune (FT), and LP-FT from the same pretrained encoders:

- FT vs LP: FT wins in-distribution (~+2%) and loses
  out-of-distribution (~−7%) — the trade the theorem predicts.
- **LP-FT beats FT on both sides: ~+1% ID, ~+10% OOD** — the
  matched-control comparison APT and ActionX never ran. Same
  encoder, same data, same objective; only the *schedule* differs.

The NTK follow-up (2405.16747) re-derives the effect in language
models and locates the mechanism the same way: LP-FT's benefit comes
from starting the joint phase with a near-optimal head, which
shrinks early backbone updates.

## What transfers to us

This is the third same-shape citation for the **#4 F-then-joint
rung** (after [APT](apt-expert-pretraining.md) and
[ActionX](actionx-rl-expert-pretraining.md)) — and the only one of
the three with (a) a *matched frozen-schedule control* and (b) a
mechanism theory. The mapping: our flow expert is the "head", the
trunk's residual taps are the "features", the F arm is the LP phase
(expert converges against a frozen trunk), and the escalation rung's
joint phase (unfrozen trunk, seam stop-grad lifted per the readout)
is the FT phase. The theory then says: **unfreezing the trunk under
a converged expert is categorically safer than joint-from-scratch**,
because flow gradients through an uninformed expert are exactly the
head-misalignment distortion channel — the same channel π0.5-KI
measured as VLM-knowledge damage and cured with the stop-grad.

It also speaks to the owner's a(t), b(t) compute framing in a way
the robotics citations don't. LP-FT's phase 1 is far cheaper per
step than its phase 2 (head-only backward — our F steps at 0.92
s/step vs K's 3.80 are the same asymmetry). So under a *fixed
compute budget*, "spend cheap a=0 steps until the head is aligned,
then pay for a>0" is not just distortion-safe, it is the
compute-Pareto move: the expensive steps are reserved for the phase
where they can help rather than harm. LP-FT is the published
existence proof that the a(t) step-function schedule beats the
constant-a schedule *at equal-or-less compute*.

## What doesn't transfer

- **K is not the paper's FT arm.** Our KI-joint trunk never sees
  expert gradients (the seam stop-grad blocks the distortion channel
  by construction); its trunk moves under its own CE loss. LP-FT
  therefore does NOT predict K loses to F — it is silent on tonight's
  Δ_seam read. It prices the rung *behind* that read: what to do if
  the a>0 region opens (or if F wins and we still want trunk
  adaptation later).
- The theorem is two-layer linear; our "head" is a ~300M flow expert
  reading 12 residual taps through adapters. The NTK paper closes
  some of that gap (deep nonlinear LMs) but not the cross-attention
  seam geometry.
- The +10% is an OOD number; ID gains were ~+1%. Our panel is
  near-ID (held-out episodes, same tasks) — so the honest expected
  effect size for F-then-joint on panel MAE is the *small* end,
  with the OOD-style gains only visible on the rig-transfer
  benchmark (#16) if anywhere.
- LP-FT tunes the switch point by convergence of the probe, not by
  a compute-optimal rule; nobody in this lineage solves for optimal
  a(t) under a budget — that remains an open (and possibly
  screen-worthy) question.

## What it fed

**#4 f-then-joint pre-reg draft** (queued, opens after Δ_seam): the
rung's citation set is now APT (+8..+26, their grid) + ActionX (+38
LIBERO-Long, supervised row) + **LP-FT (+1 ID / +10 OOD with a
matched control and the distortion theorem)** — and the draft should
inherit LP-FT's design point that the joint phase starts from the
*converged* F expert, not a fixed number of F steps. Also feeds the
post-Δ_seam compute-matched follow-up framing (owner steering
2026-08-09): the schedule family's published points all say the
step-function a(t) dominates constant a(t) in their regimes; none
of them measured it compute-matched — that comparison would be ours
to run.
