# Fewer layers than you think: prune the twins, then finetune

*Lit slice 2026-08-09 (work session 12:15Z, third read — sweep hook
cleared same session it was banked). "Finetuning Vision-Language-
Action Models Requires Fewer Layers Than You Think"
([2606.20246](https://arxiv.org/abs/2606.20246), June 2026; the
method is called CLP, CKA-guided Layer Pruning). Fed #17 (the
trunk-redundancy ledger gets its first VLA-specific measurement)
and the throughput accounting (a fourth lever class: fewer FLOPs
per step, training AND inference).*

## The paper in plain words

Big robot-policy models inherit their size from the language models
inside them — but how much of that depth does the *control* task
actually use? This paper measures how similar each transformer
layer's output is to the next one's (using CKA, a scale-invariant
similarity score), finds long runs of near-identical "twin" layers,
and simply deletes the twins — before finetuning, with no
distillation, no extra parameters, from one forward pass over a
calibration set. Finetuning then heals the seams: the remaining
layers reorganize to restore the original representation manifold.
Up to half the depth can go. The surprise is the low-data regime:
with only 10% of the demos, the pruned π₀ *beats* the full model by
+6.9 points — less capacity means less overfitting. Training gets
~28–31% faster, inference ~28–30% faster, and this holds across
LIBERO, RoboCasa, SimplerEnv, and 10 real-robot tasks.

## Contribution

- **Method:** one forward pass over calibration data → CKA between
  consecutive layers → cluster runs with similarity ≥ τ → within
  each run keep the first layer, delete the rest (up to κ). Prune
  first, then finetune normally. Training-free selection; the
  finetune is the healing.
- **Where the redundancy lives:** π₀ loses 6 of 18 layers
  (positions 1–9 range); GR00T-N1.5 loses 7 of 12 VLM layers
  (3–9) and — the eyebrow-raiser — **8 of 16 DiT action-head
  layers**. Early-to-mid trunk and half the action expert are
  twins; the "major feature transformations concentrate in a few
  distinct transitions."
- **Why it can *gain*:** implicit regularization — capacity
  removal discourages overfitting to task noise in low-data
  finetunes; PCA shows the pruned network's manifold re-aligning
  with the original during finetuning.

## Experiments

- **LIBERO at 10% data (~30% params removed):** π₀ 77.7 → 84.6
  (+6.9); GR00T-N1.5 93.9 → 93.0 (−0.9 — at a strong baseline the
  prune is ~free, not a gain).
- **RoboCasa (30 demos):** π₀ 15.6 → 18.0; GR00T flat out to
  **50% pruning ratio**.
- **SimplerEnv (GR00T):** 16.6 → 20.0, training 22.9 → 15.7 h.
- **Real world (10 tasks, GR00T):** 73.5 → 75.9 avg, single tasks
  up to +20.
- **Speed:** training −27.8% (π₀) / −30.8% (GR00T); inference
  211 → 152 ms (π₀) / 121 → 85 ms (GR00T); FLOPs ×1.39–1.42
  down.
- **Ablations:** CKA beats MSE/cosine layer selection (those
  "distort representations into isolated subspaces"), and beats
  random or drop-last-k badly — *which* layers go matters more
  than how many.

## What transfers to us, what doesn't

- **A fourth lever class in the throughput accounting.** The
  owner's thread now has: step-time kernels (perf pass-1 —
  measured dead on the true recipe), memory→batch (#20 actckpt —
  the live lever), fewer-steps-to-quality (Spatial Forcing), and
  now **fewer-layers** — the only one claiming training AND
  inference wins at once, with a train-time mechanism (fewer
  FLOPs) that can't be falsified by kernel-scheduling artifacts
  the way pass-1's microbenches were. The catch for any pre-reg:
  their gains are measured on *finetunes from a pretrained VLA*
  in low-data regimes; our stage-2 attach trains an expert from a
  50-60k-step trunk on the full recipe — closer to their GR00T
  full-data cell (−0.9, cost-neutral) than their +6.9 cell.
  "Same quality, 30% cheaper" is the honest expectation, and it
  competes head-on with actckpt's memory→batch route for the same
  wall-clock budget.
- **Half the action expert was twins — a capacity datapoint for
  the #4/#17 expert-sizing conversation.** GR00T's 16-layer DiT
  head pruned to 8 with no loss. Reads with HyperVLA (0.1M
  generated policies suffice per episode) and against QDepth's
  scaffold effect (extra expert capacity carrying wins): expert
  capacity is cheap to add and cheap to remove, so *measured*
  sizing beats inherited sizing. Our 18-layer-class experts have
  never had a CKA pass; the diagnostic is one forward pass and an
  afternoon.
- **The CKA map is a free diagnostic even with zero pruning
  intent.** One forward pass over a panel batch gives the
  twin-block map of our Molmo2 trunk + expert at any checkpoint —
  worth having in the ledger the next time trunk-depth or
  expert-depth choices come up (#17's fractional-depth mount
  discussion cites depth ratios with no redundancy evidence
  behind them). Cross-link: FLOWER's deep-layer pruning
  ([grounding-conditioning](grounding-conditioning.md)) pruned
  for *placement* reasons; this gives the general instrument.
- **Doesn't transfer:** the +6.9 headline (10%-data
  regularization regime, not ours); absolute latencies (their
  serving stack); any claim about *pretraining-stage* pruning
  (explicitly untested); and note their π₀ baseline (77.7 LIBERO
  single-view) is the same open-π₀ number QDepth-VLA starts from
  — these low baselines inflate every 2026 delta on that table.

## Where it lands

- **#17**: trunk-redundancy ledger opens with real numbers — 33–50%
  of finetuned-VLA depth is twins (CKA-selected, healing-finetune
  recipe); the CKA map banked as a one-forward-pass diagnostic for
  our own trunk/expert checkpoints; expert-sizing datapoint (16→8
  DiT layers free) filed beside HyperVLA.
- **Throughput accounting**: fourth lever class (fewer layers,
  train+inference), honest expectation "cost-neutral quality at
  ~30% savings" in our regime; competes with #20 for the same
  budget, and unlike pass-1's kernels the mechanism is
  FLOP-count, not scheduler-dependent.
- **#4 (context)**: any future attach-screen sequel could prune
  the trunk before attaching — their prune-then-finetune order is
  exactly our attach shape; parked as a named sequel arm, needs
  its own pre-reg.
