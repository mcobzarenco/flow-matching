# Pre-registration DRAFT: activation-checkpointing lineage flip (#20 → the AR trunk)

*2026-08-09 ~05:0xZ — **DRAFT, not yet immutable.** Finalization
(immutability stamp + re-pinned baselines at then-HEAD) happens when
the target launch is actually scheduled; the design below is frozen
in shape now so the item doesn't rot. Entry condition: the
[perf/memory review](2026-08-08-molmo2-perf-review.md) named
`--activation-checkpointing` the single biggest memory lever
(~2.4–2.8 GiB/sample of prefix activations) and the flag is now
**proven on CUDA**: the [#20 sdpa-pin crash](../ideas/idea-20.md) is
fixed at `913fdc4` (GPU regression test green) and the K-smoke
ladder ran the flag live on the box true recipe — B12c6 full batch,
alloc peak 57.34 GiB, rc=0. What's left is the question the review
could not answer: on the lineage that does NOT carry the flag today,
is the flip throughput-positive or a memory-for-speed trade?*

## Question

The AR-trunk lineage (40k → 60k, `BATCH=12 BACKWARD_CHUNKS=6`,
zero1, eff-48) runs without activation checkpointing at
**vram_alloc_peak ~73.8 GiB of 80** — thin margin, and the reason
backward is chunked 6× (per-chunk size 2) with
`--chunk-grad-allreduce` overhead on every chunk. Checkpointing
collapses prefix activations to ~one layer's worth for ~30% trunk
recompute. The freed memory can pay for **removing backward
chunking** (fewer suffix passes, one allreduce). Net s/step effect
is unknown and the perf-pass1 box ladder falsified our last
local-microbench transfer — so this is measured on the box true
recipe or not at all. **Adoption question: should the next fresh
AR-trunk launch carry `--activation-checkpointing` (and what chunk
setting)?**

## Scope pin — perf only, optimization semantics frozen

- **In scope (rung ladder below): same eff-48, same B12, same
  optimizer/schedule.** Only the flag and `BACKWARD_CHUNKS` vary.
  Gradient semantics of chunked vs unchunked backward are already
  oracle-pinned equivalent (chunk-grad-allreduce keystones);
  checkpointing recompute is bitwise-pinned by the #20 oracles (CPU
  keystones + GPU regression on the crash shape).
- **Out of scope: spending the headroom on batch size** (B16–20/GPU
  ⇒ eff 64–80). That changes optimization (lr scaling, lineage-curve
  comparability) and needs its own science pre-reg. This ladder only
  RECORDS the memory map (what B would fit) to seed that decision;
  it licenses nothing.
- **Never a live run**: the flip applies to fresh launches only.
  Attach-screen arms are governed by their own pre-reg (K already
  carries the flag; F stays as pre-registered).

## Ladder (box, 4×DDP true recipe, 150 steps/rung, K-smoke pattern)

Warm start + data/topology identical to the target launch's recipe;
150 steps with one eval + one save exercised per rung; s/step =
median of the last 100 steps; vram from `vram_alloc_peak_gib`
(lifetime-monotone) + `vram_window_peak_gib` (landed 6a4b45e).

| rung | ckpt | chunks | role |
|------|------|--------|------|
| 0 | off | 6 | control at ladder-HEAD (same-code anchor; 60k ran ~2.2–2.5 s/step) |
| 1 | on | 6 | flag cost in isolation |
| 2 | on | 1 (no chunk args) | **candidate adoption config** |
| 3 | on | 1 | record-only memory map: max B that fits ≤ 71 GiB (bisect from 16; NO adoption license) |

Rung order fixed; a rung that OOMs records the fact and the ladder
continues (rung 3 bisects downward). Loss-trajectory guard: rungs
0–2 start from the same warm start with the same seed; the 150-step
loss curves must overlay within the rung-0 re-run band (an A/A
calibration rung is run ONLY if the overlay judgment is ambiguous —
the perf-pass1 miscalibrated-bound lesson: no post-hoc
re-tolerancing, calibrate before judging).

## Decision rule (frozen now)

Let `r0`, `r2` be median s/step of rungs 0 and 2.

- **ADOPT** (flag + no chunking on the next fresh AR-trunk launch)
  iff `r2 <= 1.02 × r0` AND rung-2 `vram_alloc_peak <= 63 GiB`
  (≥ ~10 GiB relief vs the 73.8 status quo). The memory relief is
  the point; we pay at most 2% step time (noise-scale) for it.
- **RECORD-ONLY** otherwise — the lineage keeps its current flags;
  rung-3's memory map still feeds any future batch-science pre-reg.
- No mixed outcomes: rung 1 informs diagnosis, never adoption
  (chunked+ckpt is strictly dominated if rung 2 passes).

## Cost + gates

4 rungs × 150 steps × ≤3.2 s/step × 4 GPUs ≈ 0.7 GPU-h + loads;
**gate ≤ 2 GPU-h** (babysit entry at launch; `vram_max_gib 78` smoke
tolerance, kill on first OOM-loop). Runs in a box-idle window
strictly AFTER the attach screen completes, and only once a fresh
AR-trunk launch (100k continuation, arch-batch arm, or #17
vision-unfreeze — whichever the owner green-lights first) is
actually scheduled; the ladder result rides that launch's pre-reg as
a named amendment.

## Out-of-scope list (unchanged from the perf review)

ViT SDPA path (M), valid-row CE + fused RMSNorm (parity re-gates),
any torch.compile work, P1 suffix-cuDNN (dead twice over:
loss-bound oracle fail + −10.8% measured on the true recipe).
