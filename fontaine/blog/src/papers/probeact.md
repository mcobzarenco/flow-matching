# ProbeAct: the probe is a position sensor, not a failure detector

*Read 2026-08-09 (lit slice `lit-radar-0818`, priority 2). Paper:
[2606.09740](https://arxiv.org/abs/2606.09740) — "ProbeAct:
Probe-Guided Training-Free Failure Recovery in Vision-Language-Action
Models" (Fan Zhang, Seongbin Park, Baharan Mirzasoleiman, Shahriar
Talebi, Nader Sehatbakhsh; UCLA; 8 Jun 2026; CC BY 4.0; under review;
no code released — verified 2026-08-09).*

**The paper in plain words.** When a robot policy fails in a slightly
unfamiliar scene, it often is not because the model cannot see —
it is because the part that turns seeing into moving keeps replaying
a memorized motion. The authors show that a small side-network can
read the robot model's internal activity and recover where the target
object actually is, even while the arm is reaching for the wrong
spot. They use that recovered position plus simple physical common
sense (did the gripper close on empty air? did the object actually
rise with the hand?) to notice failures, and a small math filter to
nudge repeated attempts away from the spot where the arm keeps
failing. No retraining of the robot model — but the side-network
itself is trained on 50,000 examples where a simulator provided the
true object positions, so the recipe is not label-free.

## What it contributes

- **A multi-target 3D position probe on frozen VLA hidden states.**
  4-layer MLP [2048,1024,512,256] on layer-8 activations of
  OpenVLA-OFT: 16×16 image tokens mean-pooled to a 4×4 grid of
  4096-d vectors, PCA to 1024-d; predicts up to K object positions
  + sigmoid confidences, Hungarian-matched to ground truth in
  training and Hungarian-matched across time at inference for
  identity tracking. Best config R² = 0.968.
- **An object-agnostic kinematic failure state machine.** Six phases
  (APPROACH/MONITOR/GRASPING/POST_GRASP/… + PLACED event) driven
  by gripper width q, EE pose, and probe tracks. Hard empty grasp:
  q ≤ eps_limit. Soft empty grasp: EE rises (dz_e > tau_lift) while
  object stays (dz_obj ≤ tau_noise). Drop: q snaps shut mid-motion.
  All thresholds are hardware tolerances, not learned.
- **A two-tier CBF correction.** First failure: stateless push-back.
  Repeat failure at the same spot: instantiate a spherical barrier
  h(x) = ||x−c||² − r_safe² and project the VLA's translational
  action via a closed-form minimal QP (identity mapping when the
  action is already safe). Zones flush on task progress. Needs no
  environment model — but it is inherently a closed-loop, online
  mechanism.
- **A clean perception-vs-action dissociation measurement.** Probe
  and action endpoint computed from the same forward pass: 6.9 cm vs
  23.6 cm mean error overall; 3.4 vs 7.8 cm on successes, 10.4 vs
  34.9 cm on failures (text inconsistently says 12.4). The trunk
  knows where the object is; the action head drifts.

## The experiments it ran

- **Benchmark:** LIBERO-plus (sim only; 7 perturbation categories).
  Backbone: OpenVLA-OFT only. Success = LIBERO goal predicate.
- **Main result (Table 1):** OpenVLA-OFT 69.6% → 74.1% (+4.5).
  Biggest gains are geometric: Camera 56.4→63.8 (+7.4), Robot
  Initial States 31.9→40.3 (+8.4), Layout 74.2→80.9 (+6.7);
  Noise +1.0, Background +0.2. Other rows (π-0 53.6, π-0-Fast
  61.6, RIPT-VLA 68.4, …) are comparison policies, NOT ProbeAct
  applied to them.
- **Fine-tuned baseline (Table 2):** on OpenVLA-OFT-mixdata
  (fine-tuned on LIBERO-plus-style perturbations), Robot Initial
  States gains persist: +2.0/+6.8/+4.9/+3.1 across the four LIBERO
  suites (28.0 → 32.2 avg). Runtime correction stacks on data-side
  fixes.
- **Probe training:** 50,000 (hidden-state, position) pairs from
  baseline-VLA rollouts, labels from the simulator's
  `obj_of_interest` oracle; 200 epochs, AdamW, batch 512. Layer ×
  pooling sweep (Table 4): img-spatial > img-mean > lang-mean >
  last-token at every layer; shallow-mid layers best (0.968 @L8,
  0.934 @L28).
- **Step efficiency (Table 5):** +6 steps (~5%) on the 1,643 tasks
  both succeed; 151 rescued tasks finish in 197 steps vs 600-step
  baseline timeout; 724 joint failures. Flag: subsets sum to 2,518
  yet the "All tasks" row says 2,591 — 73 tasks (~2.8%) are
  unaccounted, and the natural missing cell (baseline succeeds,
  ProbeAct fails) is never reported.
- **No detection-quality metrics at all** (no AUROC/precision/
  recall/detection latency), no real-robot runs, no code.

## What transfers to us

- **Probe the trunk, not just the action head.** Their R² = 0.968
  position decoding lives in the VLM trunk (layer 8), exactly the
  residual stream our flow expert taps. Read jointly with SAFECAST
  (flow cells at 0.38–0.45, below coin flip), the two results are
  consistent with a specific hypothesis: failure-relevant signal
  survives in trunks even when action-head states are hard to probe.
  Directly actionable for our Molmo2-4B taps.
- **Probe-input recipe:** preserve spatial token layout (their 4×4
  pooled grid; mean-pooling costs 0.04 R², last-token costs 0.15)
  and sweep shallow-mid layers rather than only deep ones.
- **Perception-action dissociation as a failure feature:** the gap
  between where the trunk says the object is and where the predicted
  action chunk ends (3.4/7.8 cm on success vs 10.4/34.9 cm on
  failure, N=300 Layout episodes) is a per-episode scalar we could
  compute from a single forward pass — a candidate probe *feature*,
  not just a probe *target*.

## What does NOT transfer

- **The "training-free" framing.** The probe needs dense 3D object
  position labels — theirs came from a sim oracle, 50k pairs. Our
  real SO-100/101 corpus and ArmnetBench rollouts have no object
  position ground truth. No label-free path here.
- **The failure detector.** It is a rule-based kinematic state
  machine needing live gripper width, EE pose, and object tracks;
  on real logged data without object positions it cannot run. It is
  not a hidden-state failure classifier and reports no detection
  metrics we could benchmark against.
- **The CBF correction.** Closed-loop by construction (online safe
  zones from observed failures, action projection at every step).
  Useless for our offline frozen-panel setting; only relevant once
  we have real rollouts — and even then unproven off-sim (authors'
  own limitation).
- **Policy-family evidence.** AR-only (OpenVLA-OFT). Nothing here
  tests whether flow/diffusion action experts are probeable —
  SAFECAST's below-coin-flip caution for flow heads stands.

## Which idea it feeds

**Idea #6 (failure attribution/detection).** Sharpens the go/no-go
probe-separability gate; does not unblock it and does not kill it:

1. Add a **trunk-tap arm** to the gate: probe Molmo2 trunk residual
   taps alongside flow-expert hidden states against ArmnetBench
   failure labels. Decision rule: if flow-expert states fail
   separability but trunk taps pass, the SAFECAST caution localizes
   to action heads and the gate still GOes on trunk features.
2. Gate probes should use **spatial-layout-preserving pooling** and
   sweep shallow-mid trunk layers (their deltas: −0.04 R² for mean
   pooling, −0.15 for last-token; −0.034 from L8 to L28).
3. Keep ArmnetBench labels as supervision. ProbeAct's "training-
   free" detection is a sim-oracle-fed position probe plus hand
   rules with zero reported detection metrics — it offers no
   label-free or offline detection recipe to borrow.
