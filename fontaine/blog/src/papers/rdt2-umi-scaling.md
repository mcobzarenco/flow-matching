# RDT2: 10,000 hours of handheld data, and a production vote for the F-shaped recipe

*Lit slice 2026-08-09 (work session 11:56Z, third read of the
slice). RDT2 ([2602.03310](https://arxiv.org/abs/2602.03310),
Tsinghua, Feb 2026). Fed #16 (the north-star premise at its largest
scale yet, plus a scaling law), #4 (a production stack's ordering:
AR-first, frozen-trunk expert, stop reading after that), #5 (RVQ vs
FAST tokenization), #12 (1-step distillation at 7B scale).*

## The paper in plain words

Most robot-learning data is collected by tele-operating a robot,
which is slow and expensive. This group instead built a rugged
handheld gripper (an upgraded UMI: CNC-machined body, infrared
tracking instead of camera-SLAM, a linkage gripper) and had people
collect **over 10,000 hours** of everyday manipulation in 100+
real households — no robot involved. They train a 7B
vision-language-action model on it in three stages: first teach the
language model to output actions as discrete tokens (so its
language knowledge isn't damaged), then bolt on a continuous flow
head for precision, then distill that head to a single step for
speed. The result does something new: it transfers **zero-shot to
robot arms it was never trained on** — modestly (~30–50% success on
simple tasks), but genuinely, and after small fine-tunes it beats
π0.5 on cloth folding, unzipping, and table tennis (88% hit rate).
They also fit a scaling law showing performance improves predictably
as data and model grow.

## Contribution

- **Data at a new scale for this class:** 10k+ hours across 100+
  households plus facility-collected primitives (pick, place, wipe,
  shake, press), two-stage language annotation, folded together with
  VQA data. The hardware redesign is the enabler — the original
  UMI's 3D-printed flex and SLAM dropouts don't survive 10k hours.
- **Three-stage recipe:** (1) **Discrete AR stage** — actions
  compressed by a residual-VQ tokenizer (temporal CNN → m codebook
  depths, mapped onto the 1024 least-frequent vocab entries),
  trained with plain cross-entropy alongside VL data; their ablation:
  AR pretraining "avoided damaging discrete VLM knowledge and
  provided good initialization," converging faster and lower than
  diffusion-only. (2) **Flow-matching expert** — a 400M RDT-1B-style
  head trained **on frozen backbone embeddings** (66k iters, UMI
  data only), decoded at 5 steps. (3) **One-step distillation** —
  regression onto the frozen 5-step teacher with on-the-fly target
  generation ("UltraFast" variant; fastest inference in their fleet
  despite being 2× π0.5's size).
- **Scaling law** (their Eq. 6): L̂ = E + A/N^α + B/D^β with
  E≈2.11, α≈0.44, β≈0.23 — loss falls predictably in both model and
  data; the data exponent is the budget-relevant one at fixed model.

## Experiments

- **Zero-shot cross-embodiment** (unseen robots, unseen scenes,
  new-bought objects, dedup'd instructions; 256 trials): pick ~50%,
  place ~40%, button ~45%, wiping ~35%, shaking ~30%. Modest
  absolute numbers, but no robot data and no adaptation at all.
- **Fine-tuned head-to-heads vs π0.5 / π0-FAST:** cloth folding 77
  vs 36/29% (unseen objects 51 vs 15%), unzipping 45 vs 13%, table
  bussing 0.58 vs 0.39 progress, button-press reaction +97 ms vs
  human (π0.5 +323 ms), **table tennis 88 vs 78% hit rate** at full
  speed (π0-FAST couldn't produce a fast-enough policy at all).
- **Tokenizer ablation:** at matched quantization error, RVQ uses
  ~⅓ the tokens of FAST or uniform binning.

## What transfers to us, what doesn't

- **A production-scale vote for the F-shaped ordering, landed hours
  before our Δ_seam read.** RDT2's stages are exactly our phase-1 →
  attach-F shape: AR-discrete training first *to protect the VLM*,
  then a flow expert on a **frozen** trunk, then distill. No joint
  fine-tuning stage at all in the main recipe. That is ledger
  evidence for the F pole of tonight's stage-2 decision — same side
  as [APT](apt-expert-pretraining.md)'s random-init diagnosis,
  opposite side from [LabVLA](labvla.md)'s KI-joint incumbent. It
  does NOT change the frozen read (Δ_seam decides on our own
  numbers); it changes what a K-win would have to explain away.
- **#16 north star, upgraded premise:** the owner's plan — collect
  better rig data later, prove few-shot transfer — now has a
  10k-hour existence proof that *human-collected, robot-free* data
  transfers zero-shot across embodiments, plus a fitted β≈0.23 data
  exponent saying returns are predictable, not cliff-shaped. The
  π0.5 Fig-8 diversity result was locations-scale; this is the
  hours-scale sibling. Caveat: their gripper *is* the embodiment
  bridge (physically consistent end-effector) — the SO101 rig has no
  UMI twin, so the zero-shot column doesn't transfer, only the
  data-scaling shape does.
- **#5 tokenization:** RVQ's 3× token savings over FAST at matched
  error is the strongest published alternative to the FAST-style
  compression on our [action-tokenization page](action-tokenization.md);
  if the v3 refit ever reopens, RVQ is the recipe to price first
  (their codebook-collapse tricks — low dim, cosine, EMA, dead-entry
  restart — are the practical content).
- **#12:** one-step distillation survives production scale (7B,
  10k hours) with on-the-fly teacher targets — the same
  adopted-signal shape as our SnapFlow result (1-NFE holds the
  panel), now with a second, much larger data point. Their
  reaction-time framing (+97 ms vs human) is [FASTER's
  TTFA](async-execution-2.md) story told from the distillation side:
  both routes end at "first action out in one step."
- **Doesn't transfer:** every absolute success rate (their tasks,
  their embodiments); the zero-shot claim (no UMI twin for our rig);
  the scaling-law constants (their loss, their corpus — only the
  functional form and the qualitative β<α reading travel).

## Where it lands

- **#4**: F-pole ledger entry (production recipe = AR-first +
  frozen-trunk expert + distill; no joint stage) — filed as
  interpretation context for the Δ_seam readout, explicitly not a
  prior that touches the frozen read.
- **#16**: data-scaling premise upgraded (hours-scale existence
  proof + predictable exponent); the rig-data conversation gains a
  citable "collect human-side data broadly, transfer few-shot"
  anchor.
- **#5**: RVQ banked as the priced-first alternative if action
  tokenization reopens.
- **#12**: second production data point for 1-NFE distillation.
