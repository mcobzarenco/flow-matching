# Reflex: the trunk never sees the clock — so stop paying for it every denoising step

*Read 2026-08-09 (lit slice `lit-radar-0817`, priority 4: the #22
streaming slot, read as a cluster with [Legato](legato.md) —
inference-time vs train-time complements). Paper:
[2607.14695](https://arxiv.org/abs/2607.14695) — "Reflex: Real-Time
VLA Control through Streaming Inference" (Guo & Liu, 2026-07-16).
Code: [github.com/9yc/Reflex](https://github.com/9yc/Reflex).*

**The paper in plain words.** A flow-matching robot policy answers
"what should I do next" by running a big vision-language model over
the camera images, then running a small "action expert" ten-ish
times to iteratively refine an action out of noise. The naive
implementation re-runs everything, including the big model, at
every refinement step — even though the big model's output never
changes during refinement, because it never receives the
refinement clock as an input. Reflex formalizes that observation
(the encoder is "timestep-invariant"), caches the big model's
attention state once per control step, recomputes only the tiny
refinement-dependent tail, and runs perception and action
generation on separate threads so the arm never stalls waiting for
the camera pipeline. Result: 2.58× faster inference, ~half the
reaction latency, zero stalls, identical outputs to the exact
computation (MSE 0.00) — no retraining anywhere.

## What it contributes

- **The timestep-invariance observation, made precise** (§3.1,
  Prop A.1): in π0/π0.5-class VLAs the VLM trunk is functionally
  independent of the flow timestep — only the action expert sees
  t. Hence trunk KV computed once is *exactly* valid for every
  ODE/denoising step of the chunk; partitioned attention equals
  full-batch attention, verified at MSE 0.00. The converse
  ablation: naively caching the *expert's* side too fails
  catastrophically (action MSE 1.42, success collapses to ~12.5%).
- **A three-region KV partition**: static instruction prefix
  (computed once ever), sliding observation history (incremental,
  once per new frame), dynamic flow-state suffix (the only part
  recomputed per denoising step).
- **An async serving pipeline**: VLM thread at 10–30 Hz feeding
  KV, expert thread emitting actions at 50 Hz, with a
  future-conditional scheduling heuristic (assume the last
  commanded action describes the state Δ ahead) so chunks are
  generated against the state they'll land in.
- Systems garnish: fused Triton kernels (+15–20% wall clock),
  ring-buffer KV (zero allocations), −24–27% peak VRAM.

## The experiments it ran

π0.5 (2.3B) and π0 (3.1B) on LIBERO + Kinetix, RTX 4090, plus a
real AgileX PiPer arm. Inference 135.2→52.4 ms (**2.58×**; π0
2.73×); partitioned attention alone carries most of it
(135.2→61.5). Reaction latency −47% to −54% (the 54% headline is
the single best cell, π0-3.1B LIBERO-Long); stall rate 100%→0%.
Success never degrades and improves where stalls used to bite:
LIBERO-Long 68.8→72.4, Kinetix +7.4 pp, real-robot Pick-Place
65→76% / Dynamic Recovery 38→55% at a held 101–110 ms latency.
Caveats in the numbers: the "Standard" baseline appears to
recompute the *full* history each step — the paper never says
whether it reuses prefix KV within a chunk, which competent
implementations (openpi) already do, so 2.58× is against a soft
baseline; and their π0.5 LIBERO baselines (68.8–82.4) sit far
below commonly reported mid-90s, so success deltas are
internal-comparison only. No RTC comparison despite citing it.

## What transfers to us

- **Our stack satisfies timestep-invariance by construction** —
  the frozen Molmo2 trunk never sees the flow timestep. The
  actionable check, before banking anything: **does our inference
  path recompute trunk features per ODE step, or cache them per
  chunk?** If it reruns the trunk, Reflex says the fix is free and
  *exact* — not an approximation — and worth ~2× at deployment. If
  we already cache (likely, given the expert reads a fixed tap
  surface), most of the 2.58× is already banked and only the async
  thread split + fusion remain.
- **The #19 reframe is the sharpest cross-read**: timestep
  invariance means K sampled draws share ONE trunk prefill — the
  marginal cost of a draw is expert-only FLOPs. Combined with
  ActionCache's lesson (trunk unskippable, ~102 ms/decision
  end-to-end on a real SO-101, VLM ≈ 22 ms of it), the draws cost
  model splits cleanly: trunk cost is per-decision and fixed,
  draw count scales only the small expert. Draw economics are
  better than the ActionCache correction alone implied.
- **The stall-rate metric (100%→0%) is worth adopting** as a #22
  instrument — it separates "model too slow" from "pipeline
  blocks," which raw latency hides. Their 82–110 ms reaction
  latencies bracket our banked 102 ms anchor nicely.
- This does NOT contradict ActionCache's "trunk unskippable":
  Reflex never skips the trunk across control steps — it stops
  *re-running it within* a chunk's denoising loop and hides its
  latency on a second thread.

## What doesn't transfer

- The sliding 10-frame history region — we condition on the
  current frame; and how a no-retraining method feeds 10 frames to
  natively single-frame π0.5 is under-explained in the paper.
- AdaRMSNorm "added without training" is under-specified (π0's
  expert already has adaptive RMSNorm; an untrained MLP gate that
  helps is suspicious) — ignore that component.
- Kernel-level numbers are 4090-tuned; two-author paper, 4-star
  repo, excludes unified DiT-style VLAs where the timestep enters
  vision.

## Which idea/arm it fed

#22 (`async-staleness`) — the serving-layer decomposition under
RTC/Legato (boundary policy) is now explicit: trunk-KV reuse
within chunks (exact, free) + async thread split (the measured
latency lever); stall rate adopted as an instrument. #19
(`ar-sampled-draws`) — draws cost model refined: K draws amortize
one trunk prefill, marginal draw = expert FLOPs only. One infra
check queued informally: verify our rollout path caches trunk
features across ODE steps. No gate changes.
