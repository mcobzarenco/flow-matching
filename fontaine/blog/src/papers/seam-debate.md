# The seam debate — AEGIS and Wall-OSS-0.5

**Papers:** AEGIS: Anchor-Enforced Gradient Isolation for
Knowledge-Preserving Vision-Language-Action Fine-Tuning
([arXiv:2604.16067](https://arxiv.org/abs/2604.16067), single-author
preprint, Apr 2026) and Wall-OSS-0.5 Technical Report
([arXiv:2605.30877](https://arxiv.org/abs/2605.30877), industrial
lab, ~27 authors, May 2026). Banked from the 2026-08-07 lit slice as
the two poles of the attachment-seam argument; re-read at skim depth
with number verification for this page. **Fed:** #4 — the named
escalation branches on either side of the frozen-vs-KI-joint screen.

## The debate these two papers bracket

When a continuous action expert trains against a pretrained VLM,
gradients want to cross the seam in both directions. Let them cross
freely and (per [KI's measurement](pi05-knowledge-insulation.md))
language following collapses; block them entirely and the backbone
can't adapt to the expert's needs. π0.5/KI's stop-gradient is one
answer. These two papers stake out the remaining positions: AEGIS
says *repair the gradients instead of blocking them*; Wall-OSS-0.5
says *let them flow, but route the backbone's share through a
discrete CE pathway it natively understands*. Between them, every
branch of our #4 screen's decision rule has a named citation.

## AEGIS — the trained-repair middle path

AEGIS gives the failure mode its best name so far: **cross-modal
gradient asymmetry** — low-rank, high-magnitude MSE/flow regression
gradients from the action expert bulldozing the high-dimensional
CE-sculpted semantic manifold. The fix is surgical: pre-compute
per-layer Gaussian anchors of VQA activations (a one-time ~5-minute
pass); each step, backward a Wasserstein-2 penalty toward those
anchors *separately* from the task loss; then, layer-wise, subtract
the task gradient's component that opposes the anchor-restoration
direction (Gram-Schmidt, only when the inner product is negative).

Their numbers: naive fine-tuning drops OK-VQA 60.15% → 57.36% while
AEGIS holds 60.23% — full preservation — while shedding a mean of
only **0.62% of gradient energy** (range up to ~3.4%; about half
the layers get throttled on a typical step). Cost: a second
backward, ~40% wall-clock overhead.

The catch, and it is a big one: **no closed-loop task success
anywhere** — not even simulation. Action-side evidence is
flow-matching loss curves only, and there are no formal ablations.
So AEGIS is a well-named mechanism and a plausible instrument, not
a validated recipe. That is exactly the weight our screen's
decision rule gives it: if the K arm wins *with a named cost*
(trunk drift out of band), AEGIS-style projection is the
pre-registered escalation direction — bank, don't build.

## Wall-OSS-0.5 — the gradient-bridge opposite corner

Wall-OSS-0.5 is a 4B production VLA (Qwen2.5-VL-3B backbone +
mixture-of-transformers action expert) that takes the opposite bet:
**no stop-gradients anywhere**. Their seam design: discrete action
tokens (a learned vision-aligned RVQ tokenizer, not FAST) route
through the *VL expert* and carry "VLM-native" CE gradients into
the backbone; continuous action tokens route through the action
expert under flow matching; the two token families are mutually
attention-masked; the discrete pathway is simply dropped at
inference. Loss weights 1.0 flow / 0.01 action-CE / 0.01
multimodal-CE, with the flow loss contributing only a residual ~5%
of backbone updates after early training. Structurally this is
*our* recipe's argument — a discrete CE bridge makes the backbone
action-aware, flow is the deployment interface — arrived at from
the multimodal-preservation side.

Results at scale: 51.1% average task progress zero-shot across 17
real-robot tasks; fine-tuned 60.5% vs π0.5's 43.0% on 15 tasks. The
VL-preservation claim is more nuanced than the abstract: embodied
grounding improves (+21.8) but general VQA drops (RealWorld VQA
−15.0).

**The ablation that matters most to us** — a 5-task, from-scratch,
70k-step comparison of seam designs: full co-training **57.0%**,
flow-only 36.6%, **stop-gradient 31.9%** (worst), stop-grad-then-
co-train 49.6%, with VQA tightly clustered across all four. In
their setting, stop-grad was the *losing* recipe. The setting
matters before anyone panics on KI's behalf: this is from-scratch
co-training on an action-naive backbone — precisely the regime
where KI also says pure insulation fails (their frozen-naive-
backbone 0%) — not KI's regime of insulating an already
action-trained trunk during expert attachment. But it is the
strongest published counter-evidence to treating stop-grad as a
free lunch, and it is why our screen's "frozen default stands"
branch cites Wall-OSS rather than declaring the question closed.

## What transfers, what doesn't, what they fed

**Transfers.** The three-way map itself: block (KI), repair
(AEGIS), bridge (Wall-OSS). Our F-vs-K screen measures the
block-vs-extreme-block axis on our stack; whichever way Δ_seam
lands, the escalation direction is now a citation, not a guess.
Wall-OSS's RVQ-beats-FAST ablation (48.1% vs 29.3% task progress)
is also a second data point for #5's learned-tokenizer rung, after
FASTer.

**Doesn't transfer.** AEGIS's numbers are preservation-only on a
PaliGemma2 VQA stack — nothing about action quality. Wall-OSS's
ablation regime (from-scratch, action-naive, 3B, their data
mixture) differs from our sequential setting in exactly the
variable under test; their scale disclaimers say as much
("gradient-bridge dynamics validated only at 3B"). Neither paper
sets bands for our reads.

**Fed:** #4's decision rule, verbatim — read 2 of the seam screen
codes "K-wins-with-named-cost → AEGIS escalation + owner steer" and
"frozen-default-stands + Wall-OSS reading" as frozen branches. The
screen's job is to find out which of these citations we get to use.
