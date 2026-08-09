# ActionCache: remembering old answers instead of re-deciding — and why it misses our bottleneck

*Read 2026-08-09 (lit slice `lit-radar-0816`, priority 5: the #22
speed lever and #19's draws cost model). Paper:
[2607.06370](https://arxiv.org/abs/2607.06370) — "ActionCache:
Training-Free Acceleration for Vision-Language-Action Models with
Action Caching and Refinement" (Oi, Otsuka, Matsushima, Ichikawa,
Motomura, Kaneko, Fujiki; v2 2026-08-03).*

**The paper in plain words.** A robot policy that thinks from
scratch at every tick wastes effort: many situations look like ones
it has already handled. ActionCache gives a flow-matching robot
policy a memory of its own past decisions. Each time the policy
fully works out an action, the half-denoised version is stored
under a cheap fingerprint of what the robot was seeing; when a
similar situation recurs, the stored action is pulled out and
either executed directly or polished with a couple of quick
refinement steps under the current camera view. Nothing is
trained — the fingerprint is a fixed random projection, and only
decisions from successful episodes are kept. The action-generation
step gets up to 40× faster; whether task success survives depends
on the model and benchmark.

## What it contributes

- **A retrieval cache over the flow decode**: cached items are
  intermediate noisy action chunks from past denoising runs, keyed
  by a sparse ternary random projection of the **VLM trunk's
  output embeddings** (d=500, ~0.3 ms overhead), retrieved by
  cosine top-1 above a threshold; a hit warm-starts the ODE for
  N_hit remaining steps (or executes directly at N_hit=0), a miss
  runs the full decode and populates the cache. Cache commits are
  gated on episode success; LFU eviction.
- Genuinely training-free, and the caching is across decisions and
  episodes — an external memory, not per-step reuse inside one
  generation.

## The experiments it ran

π0.5 (NFE 10) and GR00T-N1.6 (NFE 4) on VLABench, LIBERO, and —
usefully for us — a **real SO-101**. The headline 10.44×/40.17× are
**action-head-only latency ratios, in sim, at zero refinement**:
π0.5 18.8→1.8 ms with success actually held (38.8→40.9%), GR00T
24.1→0.6 ms with success **not** held (34.0→30.8). LIBERO: 97.1%
→ 92.1% at 8.2× head speedup. Real SO-101: success held
(90/88/100 → 88/90/100) but hit rates of 41.8–94.4% cap head
speedup at 1.62–6.26×, and **end-to-end is 1.66×** — the VLM
(22 ms) + embedding (24 ms) stages are untouched. Against
refinement-budget baselines at NFE=1 it dominates (41.0% vs
Falcon's warm-start-from-previous-step 7.6% — naive temporal
warm-starting collapses). Composes with trunk-side VLA-Cache at a
small SR cost.

## What transfers to us

- **The hook's cost-model clause dies on our stack — logged
  loudly.** Two structural reasons: (1) the cache key is computed
  *from* trunk output embeddings, so the trunk forward — which
  dominates our 143.8 ms/decision — runs on every tick regardless;
  ActionCache accelerates exactly the part that is already cheap
  for us. (2) Retrieval is top-1 and returns one deterministic
  chunk: it *collapses* the draw distribution rather than making N
  draws cheap. Our #19 draws economics are unchanged (and our
  1191 ms 10-draw figure is the AR decoder anyway, out of scope
  for a flow-head cache).
- **What's actually worth keeping**: the real-SO-101 latency
  breakdown — a π0.5-class VLA runs ~102 ms/decision end-to-end on
  our embodiment, VLM ≈ 22 ms + embedding ≈ 24 ms — banked as a
  budget reference for #22; the trunk-embedding random-projection
  fingerprint as a near-free state-similarity primitive (offline
  eval dedup, cache keys, retrieval experiments); and the Falcon
  negative result as a caution against naive warm-starting of
  chunk decodes across time.
- A top-k variant of the retrieval would hand back k diverse
  candidate chunks after one trunk forward — a plausible
  cheap-draws mechanism, but that is our extrapolation, not in the
  paper.

## What doesn't transfer

- Headline speedups are sim, head-only, cold-start-dependent
  (success-gated cache warm-up presumes a success signal), and
  task-dependent; the flattering SR rows coexist with −3.2 (GR00T)
  and −5.0 (LIBERO) degradations.
- No dynamic-scene or distribution-shift analysis at all —
  N_hit=0 executes stale cached actions verbatim, and the paper
  doesn't discuss when that's unsafe.

## Which idea/arm it fed

#22 (`async-staleness`) — the SO-101 end-to-end latency breakdown
banked as the trunk-budget anchor (the async/trunk-overlap thread
remains the lever that addresses our actual bottleneck); the
random-projection state fingerprint filed as a reusable primitive.
#19 (`ar-draws`) — the cheap-draws cost model is explicitly *not*
changed; hook corrected. No gate changes.
