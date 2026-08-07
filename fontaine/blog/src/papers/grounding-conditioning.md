# Grounding & conditioning placement — where the trunk should feed the expert

**Papers:** IVRA
([2601.16207](https://arxiv.org/abs/2601.16207), read at v2),
FLOWER ([2509.04996](https://arxiv.org/abs/2509.04996), CoRL 2025),
SCALE ([2602.04208](https://arxiv.org/abs/2602.04208), ICML 2026
spotlight, read at v2), and SmolVLA
([2506.01844](https://arxiv.org/abs/2506.01844)). Banked across the
2026-08-05/06 lit slices; re-read at full-text depth for this page.
**Fed:** #11 (the grounding front — all four triangulate the acuity
probe's story) and the architecture batch (arm B's full-residual
conditioning has its published head-to-head baseline here).

## The theme

Our acuity probe found position information sharpest at the
vision-tower output and degraded through the LM stack; our panel's
first_mae sits barely ahead of state-copy. These four papers are
the field's answers to the same diagnosis — *where* in the stack
the action head should read (FLOWER, SmolVLA), whether lost visual
geometry can be re-injected without training (IVRA), and whether
the vision encoder can be modulated at test time (SCALE). Three of
the four needed corrections against our skim-depth bank; the
corrected versions actually agree with each other better.

## 1. IVRA — re-injecting the tower's geometry, training-free (2601.16207)

**What it contributes.** VLAs flatten image patches into a 1-D
token sequence, and the LM gradually discards the 2-D structure.
IVRA computes a patch-affinity matrix (cosine similarities from an
*intermediate* layer of the frozen vision encoder) and uses it to
smooth the visual tokens at **one mid-LM layer**: each visual token
is replaced by a convex blend of itself and its affinity-weighted
average over the other patches (λ ≈ 0.2–0.3). No training, no new
parameters, +3% latency. Note the mechanism precisely — it is
token-*feature* mixing along tower affinities, not attention-score
editing (our bank's "injects affinity signals" was loose).

**What they ran.** The headline is the low-data regime: VIMA with
LLaRA at 12% data, average 53.9 → 58.1 (**+4.2**), shrinking to
+1.5 at full data. On LIBERO: OpenVLA +1.1, FLOWER +0.8 (from a
96.3 base). One banked correction: we had "consistent LIBERO gains
across LLaRA/OpenVLA/FLOWER" — **LLaRA is never evaluated on
LIBERO**; its results are VIMA + a small real-robot study. The
ablations are where the value is for us: injection at **layer 20 of
32 (~62% depth) is best**, any layer above 19 works, one layer
beats two or five — and injecting at the LM *input* (the projector)
is **catastrophic** (0.0–6.9% across tasks, vs 22.5–73.1 at layer
20). Early re-injection destroys the very computation that needs
the signal later.

**What transfers.** Direct support for the acuity story, plus a
zero-training A/B we could run on the frozen teacher: affinity-mix
our soft visual tokens at one Gemma layer near the kv9–kv14 zone
and read panel first_mae. Banked to #11 as the cheap rung behind
the img280 read — more tokens and better-used tokens are the same
front from opposite ends.

## 2. FLOWER — cut the top of the trunk, feed the middle (2509.04996)

**What it contributes.** A 947M-parameter flow VLA built on a
deliberately *truncated* VLM: prune the top of the LM and
cross-attend the flow transformer into the truncation-point hidden
states, reallocating the saved capacity to a bigger action head
(339M flow transformer vs 205M of kept LM). Plus Global-AdaLN —
one shared modulation across all flow layers (−20% head params at
identical performance, 4.43 vs 4.44 CALVIN).

**What they ran.** CALVIN ABC→D **4.53** average rollout length
(previous SoTA 4.28, π0 4.01), LIBERO-Long **94.9** (π0 85.2),
a 20-task real-kitchen suite at 61.0 vs OpenVLA's 31.0 — from 200
H100-hours of pretraining on ~250k trajectories. The
fusion-placement ablation is the sharpest published version of our
question: on CALVIN, early fusion 57.1, **intermediate 89.5**, late
71.2. Reading top-of-stack features is 18 points worse than
mid-stack; feeding the expert too early is 32 points worse. Honest
scope: SIMPLER Google-Robot loses to RT-1X (31.9 vs 42.4).

**The corrections.** Two precision fixes to our bank. The "prunes
up to 50%" headline is the *encoder-decoder* (Florence-2) config,
where 50% = deleting the decoder; for **decoder-only trunks (our
case) their own sweep says 30% pruning is the optimum and 50%
degrades** (72.1/70.7 → 66.4/62.5). And "conditions below
mid-stack" is wrong: the truncation point sits at ~50% depth for
Florence-2 but **~70% depth for decoder-only** — the accurate
phrase is *at or somewhat above mid-stack, never the top*.

**What transfers.** This is arm B's published head-to-head
baseline: a single truncation-point cross-attention feed at 60–70%
depth captures most of what a multi-stream design chases — if
full-residual res0..res14 nulls, the corrected FLOWER read says the
follow-on is a *single deep-but-not-top tap* (near our layer-14
stream), not more streams and not maximally-early ones. The
frozen-VLM row (3.42 vs 4.44 trained) is also worth carrying: at
sub-1B scale, freezing the trunk costs a full CALVIN point.

## 3. SCALE — temperature, not tokens (2602.04208)

**What it contributes.** Our bank said "token budget spent
adaptively rather than uniformly" — **there is no token budget in
this paper**; the correction is the content. SCALE is training-free
and verifier-free, for AR/FAST-token VLAs, at constant single-pass
compute. It computes a per-token "self-uncertainty" (a dual-KL
score against both a one-hot and a uniform reference — capturing
spread *and* decisiveness) and uses it twice: as the **action
sampling temperature** (near-greedy when confident, explorative
when not), and — through its step-to-step *change* — as the
**vision-encoder attention temperature** (attention flattens to
explore when uncertainty rises above its recent average, sharpens
to focus when it falls).

**What they ran.** OpenVLA on LIBERO 75.7 → **81.5** (+5.8, where
naive sampling and top-k/p buy nothing); π0-FAST 91.2 → 93.0 (+1.8,
where naive sampling *degrades* to 84–88); π0-FAST on
SIMPLER-WidowX +14.6; real UR10e +19.5/+13.9 in-distribution. It
beats the trained test-time-selection methods it compares against
(MG-Select by +10.7 average) at 1× compute — and sits within 1.3
points of a double-pass oracle that uses the exact current
uncertainty. Component ablation: decoding alone +5.3, visual
attention alone +3.3, both +10.6 (super-additive). The row that
matters most for us: **modulating the vision encoder beats
modulating cross-modal attention in the VLA** (63.3 vs 57.4).

**What transfers.** Directly pluggable on our AR/FAST decode path —
their π0-FAST config (2048-vocab, T0=0.3, κ=2) is the template, and
it stacks conceptually with the #19 selection program (it is a
*draw-shaping* method at N=1, orthogonal to best-of-N). The
vision-encoder-beats-cross-modal result is yet another independent
pointer that the tower is the highest-leverage intervention site.
Limitation for us: nothing here applies to the continuous flow
head — this is an AR-side tool only.

## 4. SmolVLA — the half-depth cut is cheap, not better (2506.01844)

**What it contributes.** The 450M community-data VLA: SmolVLM-2
(SigLIP + SmolLM2) with the flow expert cross-attending to features
from the **first 16 of 32 LM layers — the upper half discarded
entirely** — interleaved cross/self-attention in the expert, and an
async inference stack that decouples acting from predicting. Also
the closest published analogue of our data situation: pretrained on
**481 community datasets, 22.9k episodes** — an order of magnitude
below the big VLAs — with VLM-regenerated instructions.

**What they ran.** LIBERO average 87.3 (above π0-3.3B's 86.0 at 7×
the size), Meta-World 57.3 (π0 47.9), real SO100 78.3 vs π0 61.7;
async inference cuts task time ~30% and doubles fixed-window
throughput (19 vs 9 cycles/60s). Community pretraining is worth
+26.6 points on their real suite (51.7 → 78.3 multitask). Our
banked claim (L/2 conditioning, 450M, SigLIP) is **confirmed** —
with the nuance their own Table 8 adds: **the full 32-layer stack
is slightly *better* (80.3 vs 78.5)**; N=16 was chosen as a
compute/performance tradeoff. Several shipped choices are likewise
efficiency-driven rather than ablation-best (expert width 0.75×
ships, 1.0× ablates better; chunk 50 ships, 10 ablates better) —
quote them as tradeoffs, not optima.

**What transfers.** SmolVLA is the *softer* datapoint in the
mid-stack story: it says the top half of a small VLM is nearly
free to discard, not that discarding helps. Its real lessons for
us are elsewhere — external validation that community-corpus
pretraining transfers to a real rig (the north-star protocol, #16),
the interleaved-CA/SA expert ablation (85.5 vs 79.0 CA-only /
74.5 SA-only), and the async execution stack as deployment
reference for the rig.

## The triangulation

After corrections, the four papers plus our acuity probe agree on
a sharper story than the one we banked. Grounding signal lives in
the vision tower and decays up the LM stack — but the *optimum tap
is not maximally early*: IVRA's best injection is ~62% depth and
its projector-level injection is catastrophic; FLOWER's
decoder-only optimum keeps 70% of the stack and early fusion
collapses; SmolVLA's half-depth cut costs a little rather than
gaining. The trunk's early layers do work the expert needs done —
the failure is only at the *top*, where features over-specialize
for next-token prediction. For arm B that reframes the follow-on
before its data lands: if full-residual nulls, the literature's
bet is one tap at 60–70% depth, and if it wins, the early streams
are the ones to suspect of contributing least.
