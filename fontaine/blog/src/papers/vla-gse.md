# VLA-GSE: carving adapters out of the frozen trunk's own spectrum

*Read 2026-08-09 (lit slice `lit-radar-0816`, priority 4: the
#4/fjoint alternative). Paper:
[2605.06175](https://arxiv.org/abs/2605.06175) — "VLA-GSE: Boosting
Parameter-Efficient Fine-Tuning in VLA with Generalized and
Specialized Experts" (Jiang, Lu, Qin, Chen, Wang, Gao, Zhao; v2
2026-05-08, code released).*

**The paper in plain words.** When you adapt a big vision-language
model into a robot controller, you can retrain everything (slow,
and the model forgets what it knew) or bolt on small trainable
"adapter" modules and leave the original weights frozen. This paper
asks: instead of starting those adapters from random numbers, why
not start them from pieces of the frozen model itself? They split
each weight matrix's spectrum — its natural decomposition into
important and less-important directions — giving the strongest
directions to one always-on shared adapter and parceling the
weaker ones out to a handful of small routed "experts." Training
just these (2.5% of parameters) beats both ordinary adapters and
full retraining on a robustness benchmark, while forgetting far
less than full retraining. The striking ablation: the same
architecture started from random numbers is *worse than a plain
adapter* — the spectral initialization is the whole trick.

## What it contributes

- **Spectral-init adapter-MoE**: per adapted matrix, SVD the frozen
  weight; the leading singular components initialize an always-on
  "generalized expert" (PiSSA-style), disjoint blocks of the
  residual spectrum initialize 7 routed rank-2 "specialized
  experts" (top-2 learned gating), total rank 16 per block. The
  stored frozen weight is adjusted once at init so the expected
  function is unchanged. Trainable: 2.51% (114M/4.5B) — of which,
  worth noting, 57.5% is a fully-finetuned action head, not the
  spectral experts.
- Supporting machinery: load-balancing loss and a gradient-scale
  balancing rule so experts initialized from different spectral
  energies train at matched rates.
- Base model: Qwen3-VL-4B + OpenVLA-OFT-style continuous L1
  parallel-decoding head (not flow, not tokens); 80k steps, 8×A100.

## The experiments it ran

Trained on all four LIBERO suites, evaluated on **LIBERO-Plus** —
held-out *perturbations* (camera, lighting, layout, language…) of
the *trained* tasks: **81.2% average**, vs full finetune 74.9,
LoRA 69.2, and the best of eight matched-budget PEFT baselines
76.8. Full finetune also collapses VLM knowledge (MMMU 53.2→35.6)
while VLA-GSE retains LoRA-grade (51.1 vs LoRA's 51.8). The
ablation ladder on LIBERO-Plus-Long (full = 74.1): Gaussian init
instead of spectral **60.9** — below plain LoRA — no shared expert
67.2, no routed experts 63.1. Real robot: AgileX PiPER, 4 tasks × 4
distribution shifts, VLA-GSE 82.5 vs π0.5's 74.2 and FFT's 65.8.

## What transfers to us

- **Hook corrections first**: "zero-shot" means zero-shot to
  *perturbations only* — the tasks themselves are trained; and
  "knowledge-insulation-by-construction" oversells — backbone
  weights stay frozen, but the additive experts shift the effective
  function like any LoRA, and measured retention is LoRA-grade
  (marginally *below* LoRA), i.e. empirical, not architectural.
- **The real claim for the #4 ledger**: full finetuning loses to
  frozen-plus-spectral-adapters on both robustness (+6.3) and
  retention — an anti-unfreeze datum from the PEFT direction. But
  it does not test *our* fjoint design (converged frozen expert,
  then brief joint phase); every method here adapts from scratch.
  It argues for a **third arm**, not against the rung: our current
  F trains no trunk adapters at all, so spectral-init adapters on
  the tap layers are an upgrade path from pure-frozen that never
  risks the trunk.
- **The cheapest decisive probe is well-isolated by their own
  ablations**: since Gaussian-init MoE (60.9) < LoRA (69.2) <
  spectral single-expert PiSSA (74.5) < full GSE (81.2), most of
  the gain is the *init*, and the MoE plumbing is worth ~4–6
  points on top. For us: PiSSA-style spectral-init adapters vs
  vanilla LoRA vs nothing, on the trunk layers our taps read,
  frozen trunk, F recipe otherwise — an F-cost run, well under the
  fjoint rung's ~32 GPU-h, answering "can the expert get
  joint-like adaptation without the trunk ever moving?"

## What doesn't transfer

- Continuous-L1 parallel-decoding head, not a flow expert — the
  FFT-overfits result may shift under our objective; sim-heavy
  headline; single trunk family (Qwen3-VL), no evidence the
  spectral structure transfers across trunks; no compute accounting
  vs FFT.
- Real-robot cells are 15 trials each, no CIs.

## Which idea/arm it fed

#4 (`seam-screen`) — the attachment frontier gains a third pole
between frozen-F and fjoint: spectral-init trunk adapters,
priced at ~F cost, with the PiSSA-vs-LoRA-vs-nothing probe as its
cheapest falsification; ledger notes the anti-FFT robustness +
retention datum (with the caveat that sequential-then-brief-unfreeze
was never their comparison). No gate changes; the fjoint rung's
frozen reads are untouched.
