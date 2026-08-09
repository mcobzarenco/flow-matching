# Weight decay as a plasticity knob: what pretraining decay buys the finetunes that come after

*Read 2026-08-09 (lit slice `lit-radar-0816`, priority 1: the adamc
watch). Paper: [2602.11137](https://arxiv.org/abs/2602.11137) —
"Weight Decay Improves Language Model Plasticity" (Han, Bordt,
Zhang, Kakade; v2 2026-05-28).*

**The paper in plain words.** Weight decay is usually described as a
regularizer: a small force that keeps a network's weights from
growing, tuned to make the model score well on its own training
objective. This paper asks a different question — does the amount of
weight decay used during *pretraining* change how well the model can
be *finetuned on new tasks later*? The answer is yes, and by more
than the usual story predicts: models pretrained with substantially
larger decay than the default are better finetuning substrates, and
in the heavily-trained regime this holds **even when their base
language-modeling loss is worse**. The base model's own score
under-predicts what you get after finetuning. The authors trace the
effect (correlationally) to three signatures: representations that
are more linearly separable, lower-rank attention maps, and less
pretraining overfit.

## What it contributes

- **A single-knob causal sweep**: pretrain-from-scratch at
  λ ∈ {0.1 (default), 0.3, 0.5, 0.6, 1.0, 3.0, 10}, everything else
  held fixed, then full-finetune each model and measure absolute
  post-finetune task performance ("plasticity" here = how good the
  model is *after* finetuning, not the finetune-minus-base delta).
- **The Pareto crossover**: at compute-optimal token budgets
  (20 tokens/param) moderate-large decay (0.5–1.0) improves *both*
  pretrain val loss and downstream post-finetune performance — the
  default 0.1 is just too small. In the overtrained regime
  (140 tokens/param) the trade-off appears: λ=0.1 wins base loss
  (CE 2.6088 vs 2.6208 at λ=0.3, 2.7064 at λ=1.0) but λ=0.3 wins
  after finetuning. Pretrain loss is an unreliable proxy for
  post-finetune quality (their loss↔downstream correlations are
  "rather unstable").
- **Three correlational mechanisms**, flagged by the authors as
  correlational: (1) linear probes on last-token embeddings (SST-2,
  AG News) get more accurate at every layer as pretrain λ grows —
  finetuning then "refines and aligns" existing representations
  rather than building them; (2) λ monotonically reduces W_QK
  pseudo-rank (halved at λ=1.0) while W_VP stays near full rank;
  (3) the train–val gap shrinks monotonically in λ.

## The experiments it ran

Llama-2-style 0.5B/1B/4B on FineWeb-Edu and OLMo-2-style 1B on
OLMo-Mix, at 20 TPP (10–80B tokens) plus one overtrained OLMo-1B at
140 TPP (210B tokens); AdamW with the decay term multiplied by the
scheduled lr (standard PyTorch coupling). Downstream: full finetune
(3 epochs) on 6 CoT reasoning sets (MetaMathQA, MedMCQA, PubMedQA,
MMLU-Pro-CoT, RACE, SimpleScaling) across 6 metrics incl. Pass@16
and an ORM score, 5 commonsense cloze sets, and one safety-alignment
finetune. Optimal pretrain λ by *downstream* performance: **1.0 at
20 TPP across all four model families, 0.3 at 140 TPP** — always
above the base-loss optimum. The advantage survives joint λ×LR and
λ×LR×batch sweeps at both stages. Caveats they own: λ=10 destroys
pretraining; the W_VP rank transition at λ=1.0 coincides with a
performance *drop* on Llama models; per-λ downstream numbers live
only in figures, not tables.

## What transfers to us

- **A directional prior for the trunk axis (#17), not a number**: a
  trunk pretrained with more decay should be a better substrate for
  our action-expert finetunes, and base-model benchmarks
  under-predict post-finetune quality — one more reason trunk
  selection should weight *finetuned* probes over zero-shot scores.
  We don't control (or know) Molmo2-4B's pretraining λ, and
  production trunks live in exactly the heavily-overtrained regime
  where the authors themselves warn the trade-off may flip.
- **A borrowable instrument**: the linear-probe separability metric
  is cheap and trunk-agnostic — layer-wise probes on action-relevant
  classifications would give a measured plasticity ranking across
  candidate trunks before any finetune is spent. That transfers as a
  *method*, independent of the paper's scale caveats.
- **For the adamc watch, a frame and a warning, not a claim.** The
  suggestive story: AdamC's λ∝η correction keeps effective decay
  alive through the cosine tail (vanilla AdamW's effective decay
  dies with the lr), so an AdamC-trained trunk should end training
  with more of the decay-induced plasticity signatures — relevant
  the day a stage-2/joint phase retrains on top of the 100k run.
  But the paper never studies the decay–lr coupling, never studies
  finetune-then-finetune-again, and its interesting range is
  λ 0.3–1.0 — **our λ=1e-5 is four orders of magnitude below
  anything they test**, and at λ≪0.1 they see essentially no effect
  even on pretrain loss. Record-only ledger context.

## What doesn't transfer

- **The hook's framing needed two corrections** (logged): "larger WD
  hurts base loss" is wrong at compute-optimal scale — it *helps*
  both until the overtrained regime; and "λ∝η frame" is our
  extrapolation — the paper contains no lr-proportional-decay
  analysis at all (Kosson et al. is a bare reference).
- Text-only, ≤4B, ≤210B tokens, single pretrain→finetune transition;
  nothing multimodal, nothing about a *finetune's* decay setting
  affecting later phases; mechanisms explicitly correlational.

## Which idea/arm it fed

The **adamc watch** (record-only): a plasticity frame for what
λ∝η decay might preserve in the 100k trunk, priced honestly as a
two-step analogy with zero direct evidence at our decay magnitude.
#17 (`new-trunks`) — trunk-selection axis: base benchmarks
under-predict post-finetune quality; the layer-wise linear-probe
separability method banked as a cheap pre-finetune plasticity
instrument. No gate changes anywhere.
