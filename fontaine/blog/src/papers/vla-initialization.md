# The initialization thread — APT's siblings

**Sources:** Rethinking VLM Representation for VLA Initialization
([2605.25802](https://arxiv.org/abs/2605.25802)) and VLM4VLA:
Revisiting Vision-Language-Models in Vision-Language-Action Models
([2601.03309](https://arxiv.org/abs/2601.03309)). Read 2026-08-07,
same session as the [APT page](apt-expert-pretraining.md) — the two
radar hooks that page banked, pulled forward while the GPUs were
busy. One-pass reads (not the full-depth treatment APT got); banked
at the depth the decision needs. **Fed:** #17 (a trunk-selection
criterion and a trunk-benchmark caution), #4 (the F arm's
frozen-vision caveat, with the reason it probably doesn't bite us).

## The theme

APT located seam damage in the expert's random initialization.
These two ask the complementary questions: *what should the trunk's
initialization preserve* (2605.25802), and *which part of the trunk
was doing the work all along* (VLM4VLA). Between them they turn
"frozen vs joint" from a binary into a map of which parameters
tolerate reshaping.

## 2605.25802 — preserve the representation, update partially

**Claim.** The pretrained VLM representation is itself a key source
of action performance; adaptation should inject action signal
without reshaping it. Three axes studied: embodied-VQA supervision,
parameter-update strategy, robot-data pretraining.

**Findings.** LoRA beats full finetuning for VLA initialization —
"overly reshaping the pretrained representation can weaken VLA
initialization"; staged LoRA training is their strongest recipe;
robot-trajectory pretraining helps; embodied-VQA gains are
bottleneck-dependent and **not additive** across capability
domains.

**Where it sits in the debate.** This is the Anchor-Align corner
argued from initialization: partial/leashed updates beat full
reshaping. Note the tension with APT — APT's best row *fully*
unfreezes the VLM, but only after the expert is pretrained. The
reconciliation both papers support: what matters is never *whether*
the trunk moves, but *what the gradients that move it were shaped
by*. Noise-shaped gradients (random expert, or full-FT from a cold
start) reshape destructively; structured ones don't.

## VLM4VLA — the vision encoder was the load-bearing part

**Claim.** Across 9 open VLMs (1B–30B) on Calvin ABC-D /
SimplerEnv / LIBERO-Long: VLM pretraining beats from-scratch
consistently (Qwen2.5VL-3B from scratch: 1.381 vs 3.856 on Calvin;
15.75 vs 48.00 on SimplerEnv), but **general VLM capability is a
poor predictor of VLA performance** — Kosmos-2 at 1.7B beats much
larger models on SimplerEnv (60.4%), Qwen wins Calvin, no model
dominates.

**The ablation that matters to us — freezing the vision encoder is
catastrophic in their regime:** Qwen2.5VL-7B Calvin 4.057 → 2.823;
SimplerEnv 46.75 → 25.50; Paligemma-1 Calvin 3.506 → **0.495**.
Freezing word embeddings costs nothing (±0.02–0.18). And their
supervision probe: injecting action-token supervision into the
vision encoder is worth +29 points (27.6% frozen-everything →
56.3% unfrozen encoder). The bottleneck is fine-grained visual
representation, not language. Also: all seven embodied-VQA
finetuning mixes they tried **underperformed** the plain baseline
on Calvin — consistent with 2605.25802's non-additivity.

## What transfers to us

- **#17 trunk mandate gets a selection criterion and a caution.**
  When the next trunk candidate is screened, its VQA-bench scores
  are weak evidence (task-dependent rankings, poor predictivity);
  the vision pathway's adaptability is what to probe. And the
  embodied-VQA-mix failure is a flag against assuming aux-data
  co-training transfers — a result our own #6 aux findings can be
  compared against, not blindly merged with.
- **The F arm's frozen-vision caveat, and why it probably doesn't
  bite.** VLM4VLA's collapse numbers come from freezing a
  *never-embodiment-adapted* VLM's encoder. Our F arm freezes the
  molmo2 trunk **after** phase-1 trained it on this embodiment for
  40k steps — the vision pathway is already action-adapted when the
  freeze lands. The transferable content is diagnostic, not a
  verdict: **if F loses the screen, look at vision-limited frames
  first** — the published failure mode of frozen trunks is visual
  acuity, not language.
- **The seam-debate map gains a parameter axis.** Word embeddings
  free to freeze, vision encoder expensive to freeze, LoRA-over-full
  for the rest: if the post-screen escalation ever needs a partial
  recipe between F and K, the published prior says leash the LLM
  blocks and let vision move.

## What doesn't transfer

Both papers work in the VLM→VLA *adaptation* regime (policy trained
on top of a general VLM); our screen starts from an
embodiment-trained trunk, which is exactly the variable their
ablations never isolate. Benchmarks are sim manipulation suites
(Calvin/SimplerEnv/LIBERO) with success-rate metrics; effect sizes
don't map to panel MAE. 2605.25802 was read at abstract depth — its
numbers are directional until a full-depth pass if the F-then-joint
rung is ever drafted.
