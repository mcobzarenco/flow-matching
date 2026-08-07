# To unfreeze SigLIP or not: the vision-encoder freeze question has two right answers

*Read 2026-08-07 (same-session slice, prompted by the owner's 17:04Z
question: what does the literature say about unfreezing molmo2's
SigLIP encoder?). Sources: the OpenVLA ablation
([2406.09246](https://arxiv.org/abs/2406.09246)), LoRA-for-π₀ on
industrial assembly
([2607.10172](https://arxiv.org/abs/2607.10172)), MAPS
([2511.19878](https://arxiv.org/abs/2511.19878)), and the
dual-encoder representation-preservation paper
([2509.11417](https://arxiv.org/abs/2509.11417)). Companion to the
[VLA-initialization page](vla-initialization.md) (VLM4VLA's
frozen-encoder collapse) — this page is the *other* pole plus the
reconciliation.*

## The question

Our live molmo2 AR 40k run trains with the vision tower frozen (no
`--backbone-vision-lr`); the pre-reg names a vision-unfreeze rung as
a follow-on. VLM4VLA gave the loudest pro-unfreeze evidence —
freezing a general VLM's encoder during VLA adaptation collapses
performance (PaliGemma, a SigLIP encoder: Calvin 3.51 → 0.50). Is
that the whole story? No — the literature splits cleanly in two, and
the split is informative.

## The pro-unfreeze pole: adaptation regimes

**OpenVLA** (2406.09246) is the canonical result: against
prior practice of freezing VLM vision backbones, they fine-tuned all
7B parameters *including* the fused SigLIP+DINOv2 towers and report
fine-tuning the vision encoder as clearly necessary — frozen-encoder
VLAs underperform across tasks, which they attribute to the encoder
needing to adapt to the spatial detail the policy actually consumes.
Their regime: ~970k-episode OXE pretraining — a huge embodiment
corpus, i.e. structured gradients in APT's sense.

**LoRA-for-π₀ on real assembly** (2607.10172, already deep-read on
the [data-and-trunks page](data-and-trunks.md) — quoted here as the
small-data pole): UR5e precision assembly, 200 demos/task, Average
Task Progress on physical rollouts. The SigLIP treatment alone
swings the outcome — LoRA r=32 with trainable vision 0.74 (ties FFT
0.76), LoRA-restricting SigLIP 0.43, **frozen SigLIP 0.14**. Their
diagnosis: the visual domain shift to the target workspace has high
intrinsic rank (~52%), so low-rank or frozen vision cannot carry it.
Their recipe — **LoRA the language blocks, full-FT the vision tower**
— is almost exactly the "leash the LLM, let vision move" prior the
VLM4VLA page banked, now with rollout numbers. Notably this
*inverts* 2605.25802's LoRA-beats-full-FT finding when applied
naively to the vision pathway: LoRA is for the semantic blocks, not
the eyes.

## The pro-freeze pole: retention regimes

**MAPS** (2511.19878) runs the component study in the *downstream
fine-tuning* regime (MiniVLA/OpenVLA-OFT on SimplerEnv, CALVIN,
LIBERO): there, **freezing the vision encoder improves ID accuracy
by 7–17% and OOD by 7–25%** on SimplerEnv, and freezing DINOv2
beats freezing SigLIP (geometric priors matter most). Their method
generalizes the observation: schedule module-wise proximity
constraints — vision held close to its pretrained prior, action-side
language layers freest — for up to +30% with no new parameters.

**The dual-encoder paper** (2509.11417) explains *why* naive
unfreezing pays an OOD tax: fine-tuning collapses the encoder's
semantic structure (their t-SNE probe), so they keep a **frozen
encoder as an anchor** concatenated with a trainable one —
OpenVLA 35.0% → 55.6% on SimplerEnv visual matching from the dual
encoder alone, 78.5% with co-training, with the gains concentrated
under visual perturbation and paraphrase, at 0.5–1.3× extra
inference cost.

## The reconciliation (and it matters which regime we're in)

The poles do not contradict each other; they measure different
things. Unfreezing vision buys **in-distribution adaptation** —
essential when the visual domain shift from VLM pretraining to the
embodiment is large (OpenVLA's OXE, the assembly workspace,
VLM4VLA's Calvin adaptation). Freezing/anchoring vision buys
**out-of-distribution retention** — it protects the pretrained
semantic geometry that generalization to perturbed visuals rides on.
Every paper above fits this line, including APT's gradient-quality
frame: unfreezing hurts exactly when the gradients that move the
encoder are too narrow to preserve what it knew.

## What transfers to us

- **The #17 vision-unfreeze rung draft gains a sharpened shape.**
  Our molmo2 40k run *is* the embodiment-adaptation regime (18.6M
  frames, panel = held-out episodes of the same distribution) — the
  published prior says an unfreeze arm should help the panel number,
  and the failure cases live on an axis (OOD robustness) our panel
  barely measures. Recipe prior from 2607.10172: **full-FT the
  vision tower at low LR; if anything gets LoRA/leash treatment it's
  the language blocks** — not LoRA-on-SigLIP, which sits in an
  uncanny valley (0.43).
- **The OOD tax is a #16 concern, not a panel concern.** If the rig
  benchmark ever exists, MAPS/2509.11417 predict a specific failure
  signature for an unfrozen-vision checkpoint: degradation under
  visual perturbation, not ID regression. Worth one line in any
  unfreeze pre-reg: the panel read cannot see this cost.
- **A cheap middle path exists** if the unfreeze arm ever needs
  hedging: MAPS-style proximity (weight-space L2-to-init on the
  tower) is free to implement; the dual-anchor concat is not (vram,
  and our token budget is the binding constraint at 67.07/71 GiB).

## What doesn't transfer

MAPS/2509.11417 numbers come from OpenVLA-family models fine-tuned
on simulator benchmarks — absolute magnitudes don't map to our
panel MAE. The dual-encoder architecture change is out of scope for
a rung (it alters the interface the attach screen depends on). And
none of these papers touch AR-token action heads on a frozen-text
trunk — our freezing split (frozen `wte`/`lm_head`, trainable
decoder) has no published twin; the encoder evidence transfers, the
rest of the recipe stays ours.

*Correction to the 17:4xZ Discord reply: it said no banked case of
vision-unfreeze hurting action metrics existed — MAPS and 2509.11417
are such cases, in the downstream/OOD-retention regime. The
directional recommendation for our rung is unchanged (our regime is
the adaptation one), but the claim needed the regime qualifier.*

*Radar hooks banked unread: VEGA (2605.10485, visual-encoder
grounding alignment), HyperVLA (2510.04898).*
