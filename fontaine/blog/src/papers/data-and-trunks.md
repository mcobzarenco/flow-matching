# Data & trunks — what scales, what prunes, what fine-tunes

**Papers:** Rethinking VLA Scaling
([2602.09722](https://arxiv.org/abs/2602.09722)), the VLA
data-infrastructure survey
([2604.23001](https://arxiv.org/abs/2604.23001), TMLR), VLM-to-VLA
parameter redundancy
([2606.31382](https://arxiv.org/abs/2606.31382), ECCV 2026), and
the LoRA fine-tuning study
([2607.10172](https://arxiv.org/abs/2607.10172), ICANN 2026).
Banked at skim depth across the 2026-08-05/06 lit slices, every one
flagged "re-read before citing numbers" — and this page is why that
flag exists: **two of the four banked claims were materially wrong**,
one was regime-qualified, one confirmed. All four corrections are
recorded here and in the ideas.md hooks they fed (#9, #16, #17,
#18.7).

## The theme

Four papers about everything *around* the model: what data to pool,
what the field's data infrastructure actually bottlenecks on, how
much of an adapted trunk is really load-bearing, and how little you
need to train when fine-tuning to a new rig. They matter to us
because our census (#18.7), our trunk screens (#16/#17), and the
rig-transfer north star are exactly these questions at our scale.

## 1. Rethinking VLA scaling — negative transfer, but only where the trunk is frozen (2602.09722)

A controlled empirical study (no new method) on a
Mixture-of-Transformers VLA — InternVL-3.5-2B semantic expert +
0.7B flow-matching action expert — asking which of three scaling
levers actually pay: action-space alignment, data mixture, and
regularization.

**What they ran.** *Action spaces:* EEF-relative parameterization
scales most reliably (LIBERO 5-shot, scratch → pretrain: 66.9 →
75.1%, the largest gain of the four spaces), and delta actions
jitter in place on real hardware — 0% success. *Data mixture:*
cumulative pooling D1 (OXE-only) → D4 (+Agibot/RoboMind, sim,
joint-space data; ~182M effective frames balanced by per-source
downsampling). With the VLM **frozen**, pooling hurts: LIBERO 77.3%
(D1) → 72.1–75.1% (D2–D4); RoboCasa 54.7% → 48.8–50.0%. With the
VLM **unfrozen**, the effect largely vanishes: 86.4 / 84.0 / 83.7 /
84.5% across D1–D4. *Regularization:* modality dropout and the
two-stage curriculum do **not** help — the best config drops both
(p_view=0 at 85.6%, end-to-end "stage 2 only" at 85.8%, vs 84.5%
baseline). Their headline model (97.9% LIBERO, above π0.5's 96.9)
trains on the **full** D4 pool. One real Franka, four tasks, 10
trials each — the real-world side is bar charts, not tables.

**The correction.** Our banked line was "pooling induces negative
transfer; selective mixture + regularization beat full pooling."
Three things wrong with that: the negative transfer is **modest and
frozen-VLM-only** (−2.2 to −5.9 points; stable once the trunk
trains); there is **no selective-mixture method in the paper** —
"selective" was our gloss on their negative guidance about
indiscriminate mixing; and the regularization finding is
**inverted** — their ablations argue *against* dropout and staged
curricula at scale.

**Transfers.** We train our trunk, so this paper *lowers* the
alarm on data pooling for us — the #9 judge-score-weighted-sampling
lever keeps its motivation from our own fork census, not from this
paper. The stealable pieces: EEF-relative as the action
parameterization that scales, and their Grouped Blind Ensemble
protocol (operator executes shuffled, anonymized model groups;
deanonymize post-hoc) — worth copying when rig-transfer evals
become physical.

## 2. The data-infrastructure survey — and what it conspicuously doesn't cover (2604.23001)

A TMLR survey organizing the VLA data landscape into datasets
(real: OXE's 22 robots, DROID, RH20T's tactile/force; synthetic:
RoboCasa, MimicGen's 50k demos from 200 seeds), benchmarks
(short-horizon Meta-World/LIBERO through BEHAVIOR-1K's 1,000
activities), and **data engines** — video-to-data (H2R: +3–23% real
improvement), hardware-assisted (ALOHA at $20k vs GELLO at $300;
UMI's 71.7% zero-shot across 30 locations from 12 person-hours),
and generative (RoboTwin 2.0: 100k+ trajectories). Its central
claims: dataset development hasn't resolved the fidelity–cost
tension, and "the primary limitation of current data engines is not
generation capacity but grounding reliability" — generation scales
faster than verification.

**The correction — loud.** We banked this survey as framing
"dedup/contamination checks as THE underexamined bottleneck." It
does not. Two full-text passes found **zero discussion of
deduplication, contamination, train-test leakage, or overlap
auditing** — the closest content is generic filtering of implausible
synthetic generations. We projected our own census framing onto the
paper. The honest citation is the reverse: the field's own TMLR
data survey **doesn't cover** the leakage/dedup axis at all — which
positions our duplicate-content census (12.2% of panel core frames
had train twins; panel-v2 removed them) as filling a hole the
survey's taxonomy misses, not as implementing its advice.

**Transfers.** The taxonomy and per-engine numbers are a useful
citable map for the eventual rig data-collection decision (GELLO
vs UMI-style collection is a real fork for the owner rig). No new
arm; the correction itself is the payload.

## 3. VLM-to-VLA parameter redundancy — a pruning probe, not a scale claim (2606.31382)

The banked claim — "bigger VLM backbones do NOT consistently
improve action performance after adaptation" — **is not in this
paper**. It runs no backbone-scale comparison at all; the claim
belongs to VLM4VLA, which this paper merely cites (and which we
already carry separately via the ICLR-2026 VLA survey's
no-benchmark-correlation finding, #17). The fast-model summary we
banked from lifted a cited work's claim. What the paper actually
is: a study of *which* parameters matter after VLM→VLA adaptation,
using **pruning without recovery fine-tuning** as the diagnostic —
if you need to retrain to survive pruning, the removed parameters
weren't redundant, and the retraining masks the damage (their
recovery paradox: a pruned config at 1.5% success recovers to 86.5%
after LoRA fine-tuning).

**What they ran.** On OpenVLA (7.5B) and π0.5 (3.6B), rank
parameters by adaptation divergence |ΔW| from the source VLM, prune
top-r% vs bottom-r% per module, evaluate directly. The signature
result is a **sensitivity reversal across modules**: in DINOv2
attention, pruning the *most*-changed params collapses the policy
(84.7% → 1.6%) while the least-changed prune barely hurts (76.7%);
in the LLM's FFN it is exactly reversed (least-changed prune →
2.7%, most-changed → 72.0%); SigLIP's FFN survives even 100%
removal at ~70%; the projector tolerates neither (0.0% both ways at
30%). A per-module joint scheme built on this map removes 12–30% of
parameters with no recovery training at 85–96% retention (π0.5:
−22% params → 92% of baseline), where classic magnitude/activation
pruning (LLM-Pruner, FLAP, Wanda) collapses to ~0–1% under the same
no-recovery rule.

**Transfers.** Nothing about trunk *choice* — the E4B kill-branch
prior should cite VLM4VLA, not this. What it does offer:
deployment-side compression of a π0.5-class model at 92% retention
with zero retraining, and evidence that vision-encoder and
projector parameters are nowhere near uniformly expendable — a
useful prior for what NOT to freeze or prune when we squeeze
models for the rig (converges with paper 4's vision-encoder
finding from the opposite direction).

## 4. The LoRA study — r=32 suffices, but never starve the vision encoder (2607.10172)

The confirmed one, now with exact numbers. π0 (3.2B) fine-tuned on
real industrial precision assembly — UR5e, four contact-rich tasks
(bolt insertion easy/hard, pick-and-place among distractors,
bearing press-fit), 200 demos per task, evaluated by Average Task
Progress (ATP: equally-weighted sub-goals — note, **not** binary
success rates, and no per-task table exists; our earlier phrasing
"task success rates" was imprecise).

**What they ran.** Rank sweep r ∈ {8…256} with α=r: FFT reaches
0.76 ATP; LoRA r=32 hits **0.74** (p=1.000 vs FFT, effect size
0.006 — statistically indistinguishable); performance climbs from
r=8 (0.65) to r=32 and plateaus. Component-specific allocation
(VLM-heavy vs expert-heavy) buys nothing over uniform. The
dramatic rows are the plasticity ablations: **freeze the VLM →
0.15; freeze SigLIP alone → 0.14; even LoRA-restricting SigLIP →
0.43** (all p<0.001), vs 0.74 with the vision encoder fully
trainable. Static peak VRAM: 36.2 GiB (FFT) → **10.8 GiB** (r=32,
15% trainable) — a 24 GB card suffices. Honest caveat they print:
the α=r scaling rule can suppress large-r updates, so the plateau
beyond r=32 may partly be an artifact of the scaling convention,
not a capacity ceiling.

**Transfers.** This is the closest published template for our
few-shot rig-transfer protocol (#16): LoRA r=32 uniform over trunk
+ action expert, **vision encoder fully trainable**, ~200
demos/task scale, fits a consumer GPU. The vision-encoder rows are
also the third independent confirmation of our grounding-bottleneck
reads (#11): adapting the visual stack is where fine-tuning
capacity must go — freezing it costs 0.6 ATP where trunk rank
choices cost 0.02.

## The meta-lesson

Scored against full text: one banked claim confirmed (LoRA), one
directionally right but regime-qualified (negative transfer:
frozen-trunk only, no selective-mixture method, regularization
inverted), two wrong (a dedup framing the survey never contains; a
backbone-scale claim belonging to a different, merely-cited paper —
both artifacts of skim-depth banking through summaries). The
standing "re-read before citing numbers" flag is now a hard rule:
no skim-banked number crosses into a pre-registration or a blog
claim without a full-text pass.
