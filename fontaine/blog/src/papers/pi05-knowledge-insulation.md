# π0.5 + Knowledge Insulation

**Papers:** π0.5: a Vision-Language-Action Model with Open-World
Generalization ([arXiv:2504.16054](https://arxiv.org/abs/2504.16054),
Physical Intelligence, Apr 2025) and Knowledge Insulating
Vision-Language-Action Models
([arXiv:2505.23705](https://arxiv.org/abs/2505.23705), "KI", May
2025). Read as a pair 2026-08-07 (full-text deep read; the original
[deep-read post](../posts/2026-08-07-pi05-deep-read.md) has the
as-it-happened version). **Fed:** the two named arms of the #4
attachment-seam screen, the #6 self-subgoal probe, the #16 north-star
anchor, a #5 footnote.

These two papers matter more to this project than anything else on
the radar: they are the production-scale version of exactly the
recipe we run — an autoregressive vision-language trunk that learns
actions as discrete tokens first, with a flow-matching action expert
attached afterwards. Where our decisions were made by construction,
theirs were made by measurement. This page is the record of what
they measured.

## What π0.5 contributes

π0.5 is Physical Intelligence's mobile-manipulation generalist — the
model they send into *unseen homes* to clean kitchens and make beds.
The contribution is less the architecture (PaliGemma-class VLM with
a separate-weights action expert) than the demonstration that a
particular *training staging* generalizes:

1. **Pre-training, 280k steps, everything discrete.** VQA,
   captioning, detection, subtask prediction, and — crucially —
   *actions, encoded as FAST tokens* — all trained as next-token
   prediction over one transformer. The mixture spans ~400 hours of
   mobile-manipulation data from ~100 homes, static-arm and
   cross-embodiment lab data (incl. OXE), high-level subtask
   annotations, and web data. The startling mixture fact: **97.6% of
   pre-training examples are not the target embodiment.**
2. **Post-training, 80k steps, hybrid.** A randomly-initialized
   flow-matching expert is attached and trained jointly with the
   continuing cross-entropy objective (loss balance α=10, tuned),
   keeping the discrete FAST pathway alive next to the flow head.
3. **Inference is hierarchical inside one model.** The trunk first
   autoregressively decodes a subtask string ("pick up the plate"),
   then the flow expert decodes a continuous action chunk
   conditioned on it — 10 Euler steps, 50 Hz control.

A convention flag for anyone reading their equations against our
codebase: π0.5 writes the noisy action as
\\(a^\tau = \tau a + (1-\tau)\omega\\), so **τ=1 is data** there;
our code uses \\(x_\tau = \tau\varepsilon + (1-\tau)a\\), so **τ=1
is noise here**. Inverted, as usual.

## What they ran, and what the numbers say

The experiments worth carrying (mock-home evals, 10 trials/task):

- **Cross-embodiment data is load-bearing.** Removing the
  other-robot data (static arms, lab cross-embodiment)
  "significantly degrades" performance on the *mobile* robot. The
  target embodiment's own data is not enough.
- **Web data buys out-of-distribution robustness.** Removing it
  barely moves in-distribution success but craters generalization
  to unseen objects.
- **Fig. 8 — the diversity result.** Held-out-home performance
  scales monotonically with the number of training locations (3 →
  104), and at 104 locations *matches a control model trained on
  the test homes*. Environment diversity substituted fully for
  target-environment data, at their scale.
- **Fig. 13 — the hierarchy result.** "Implicit HL" — subtask data
  present in training but *no* runtime subtask decoding — is the
  **second-best** configuration, close behind the full explicit
  hierarchy. Most of the benefit of subtask prediction is
  representational and comes from co-training alone; explicitly
  decoding a subtask at runtime adds a real but smaller increment.

## KI: the attachment question, quantified

KI is the follow-up study that asks the exact question our stage-2
faced: *what happens at the seam when a continuous action expert is
bolted onto a discretely-pretrained VLM?* Their measurements:

- **Naive joint training is a disaster.** Backpropagating the
  randomly-initialized expert's flow gradients into the backbone
  collapses language following (~75% → ~5–10% on their
  "items-in-drawer" instruction metric) and converges **7.5×
  slower** to the same table-bussing performance.
- **Freezing an action-naive backbone is also a disaster:** 0%
  success — a backbone that never saw actions has no robotics
  features for the expert to read.
- **Their fix — knowledge insulation — does both at once:** the
  backbone keeps training on FAST discrete tokens (the
  representation-learning signal), while the flow expert trains
  with a **stop-gradient at the attention seam** — expert queries
  attend to `sg(K_b)`, `sg(V_b)`: information flows forward,
  gradients never flow back. Discrete FAST tokens and continuous
  action tokens are mutually attention-masked.
- Stop-grad alone recovers ~35 points of language following;
  VLM-data co-training recovers more (Figs. 4/6). With the seam
  insulated, the CE/flow loss balance stops needing tuning (α=1,
  vs π0.5's hand-tuned α=10).
- **FAST beats naive per-dimension binning** as the backbone's
  discrete action signal (~95% vs ~85% table bussing), and the
  recipe is robust to how state is represented (text tokens,
  special tokens, continuous projections all work).

## What transfers to us — and what doesn't

**Our stage-2 is "extreme KI."** Our sequential recipe — stage-1 AR
trunk on FAST tokens, then a flow expert on the *hard-frozen* trunk
— is knowledge insulation taken to its limit: the trunk's discrete
phase simply ended before the expert's phase began. KI's scary
frozen-backbone-0% result does **not** indict us: their frozen
backbone was action-naive, ours is action-pretrained. Our banked
stage-2 result (6.62 panel MAE at 80k, beating the h1536 lineage
with a 2.2× smaller expert) is itself evidence that a FAST-trained
trunk is the better feature source — consistent with KI's thesis,
not in tension with it.

But two dials of the production recipe differ from ours, and both
are now named, measured, external arms rather than vague headroom:

1. **Depth of reads.** Their expert attends per-layer to the *full*
   backbone KV stack; ours cross-attends to 3 exported streams
   ({4,9,14} of 35 layers). Production-scale evidence for the deep
   end of a dial #4 had already flagged.
2. **Trunk kept adapting under stop-grad.** Their backbone
   continues CE-on-FAST *during* expert training, insulated from
   flow gradients; ours froze. Frozen-vs-KI-joint is a paired,
   screen-rung question with a banked anchor to beat.

What doesn't transfer: their scale (400 h, ~100 homes, mobile
manipulators) and their success-rate metrics — our panel is offline
chunk-MAE on a community corpus. Their numbers set directions and
priors here, never bands.

## Which arms it fed

- **#4 — the attachment-seam screen** (now
  [pre-registered](../posts/2026-08-07-prereg-molmo2-attach-screen.md)):
  F (frozen trunk, our default) vs K (KI-joint: CE continuing +
  stop-grad seam, α=1) at the Molmo2 40k endpoint is a direct
  instantiation of KI's central measurement on our stack — with
  naive joint training *not* run, because KI already measured the
  collapse for us. The all-layer-reads dial stays open as its own
  future arm.
- **#6 — the self-subgoal probe**
  ([pre-registered](../posts/2026-08-07-prereg-selfsubgoal-probe.md)):
  our +0.462 panel-MAE aux-off cost independently replicates their
  Implicit-HL finding (semantic co-training shapes the action
  representation). Their further increment — *explicit runtime*
  subtask decoding — is the one thing we never tested, and we
  already own the conditioning slot; the probe is zero-training.
- **#16 — the north star's external anchor.** Fig. 8 is the
  diversity-buys-transfer bet measured at production scale; our
  premise that a diverse community corpus can carry few-shot rig
  transfer now has a citable precedent (evidence, not proof).
- **#5 — a footnote:** FAST-vs-naive-binning (~10 points) supports
  the token-quality premise behind the FAST v3 refit.
