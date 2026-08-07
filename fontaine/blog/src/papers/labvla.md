# LabVLA

**Paper:** LabVLA: Grounding Vision-Language-Action Models in
Scientific Laboratories
([arXiv:2606.13578](https://arxiv.org/abs/2606.13578), Zhejiang
University / Shanghai AI Lab / HIT, June 2026, work-in-progress).
Banked from the 2026-08-07 lit slices; re-read at full-text depth
for this page. **Fed:** #4 — the attachment-seam screen. LabVLA is
the third independent group shipping the KI-joint recipe, which is
what makes our frozen-vs-KI-joint screen a measurement of *the
field's incumbent*, not of an exotic alternative.

## Why we care

Our project trains an autoregressive trunk on FAST action tokens
first, then attaches a flow-matching expert. The live #4 question is
what to do at the seam: keep the trunk hard-frozen (our default) or
keep its discrete losses training under a stop-gradient
([π0.5/KI's answer](pi05-knowledge-insulation.md)). LabVLA is an
independent group, on a different trunk, in a different domain,
adopting exactly the second recipe — and that adoption pattern is
the signal, because they publish no ablation of it.

## What the paper contributes

Two artifacts aimed at lab-bench automation:

1. **RoboGenesis** — a programmable simulation data engine for
   laboratory manipulation: 2,947 annotated assets, 10,000 lab
   scenes, an agentic workflow generator, producing their
   "LabEmbodied-Data" demonstrations.
2. **LabVLA** — the policy: a Qwen3-VL-4B-Instruct trunk trained
   with **FAST action-token pretraining first** (masked next-token
   prediction over FAST-encoded action chunks, with VQA/annotation
   losses co-trained), **then flow-matching post-training under
   knowledge insulation** — an 18-layer DiT expert (width 1024)
   that cross-attends to a *linearly projected, detached* slice of
   the VLM prefix. The flow gradients update only the projection
   and the DiT; the trunk keeps learning from FAST + annotation CE
   throughout. Combined loss α·L_FM + L_FAST + Σλ·L_CE with
   **α = 10** (π0.5's tuned balance — notably not KI's α = 1).
   Inference: 10 Euler steps.

Two structural differences from π0.5 worth naming: the expert reads
a *projected prefix slice* through cross-attention, not the
per-layer full-KV-stack reads of a shared-transformer suffix
expert; and there is a third phase — benchmark fine-tuning on
LabUtopia — before evaluation.

## What they ran

- **Simulation (LabUtopia, 6 operations, 120 episodes/setting):**
  LabVLA averages **71.1% ID / 70.0% OOD**, beating the next-best
  baseline π0 (63.3/63.2) by ~7–8 points; π0.5 scores 52.4/52.1,
  SmolVLA 52.2/53.1, π0-FAST 16.9/19.7. Per-task spread is wide
  (Press Button ~100%, Pour Liquid 43.3/34.2).
- **Data transferability (their only quasi-ablation):** fine-tuning
  a third-party policy (X-VLA) on their LabEmbodied-Data lifts its
  5-task average 49.3 → 64.3% ID and 43.7 → 63.0% OOD — the data
  engine, not just the model, carries value.
- **Real robot:** one Franka, four composed benchtop tasks, 50
  rollouts per setting: 86.5% in-domain clean down to 74.0% OOD
  cluttered, above their DreamZero and π0.5 baselines.
- **What they did NOT run:** no ablation of the two-stage ordering
  (FAST-pretrain vs flow-from-scratch), no KI-on/off comparison, no
  α sweep, no Euler-step sweep. The recipe is adopted wholesale
  from π0.5/KI, not re-measured.

## What transfers to us, and what doesn't

**Transfers.** The staging is ours, independently reproduced at our
trunk's size class: action-aware-first AR pretraining on FAST, flow
expert attached second, trunk's CE kept alive under stop-grad
during attachment. That is now π0.5/KI *plus* LabVLA on the
KI-joint side of the #4 screen, versus our sequential-freeze
default. It is also a second 4B-scale data point (Qwen3-VL-4B) for
the Molmo2-4B port's size class — the recipe does not obviously
need 30B-scale trunks. Their projected-detached-prefix attachment
is closer to our exported-streams mechanism than π0.5's full-KV
reads, which keeps the depth-of-reads dial (#4 arm 1) genuinely
open — LabVLA is evidence you can win benchmarks *without*
all-layer reads.

**Doesn't transfer.** No component-level evidence: because they
ablated nothing at the seam, LabVLA cannot tell us whether KI-joint
*beats* frozen — only that a capable group bet on it. Success rates
on LabUtopia/Franka set no bands for our offline chunk-MAE panel.
And their domain (lab protocols, composed skills) plus their
benchmark fine-tuning phase make the absolute numbers
incommensurable with our setup.

## Which arm it fed

**#4, directly.** The
[attachment-seam screen](../posts/2026-08-07-prereg-molmo2-attach-screen.md)
(F frozen vs K KI-joint at the Molmo2 40k endpoint) was
pre-registered with LabVLA cited as the third independent adopter
of the K recipe. If Δ_seam favors K, the field's incumbent wins on
our stack too; if F holds, we have a paired, pre-registered result
the adoption literature doesn't have — either way the screen earns
its GPU hours precisely because everyone else is adopting without
measuring.
