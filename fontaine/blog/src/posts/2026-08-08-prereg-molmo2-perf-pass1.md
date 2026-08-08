# Pre-registration: molmo2 perf pass 1 — the S-bundle

*2026-08-08 ~14:3xZ — **finalized; immutable from this commit.**
Entry condition met 2026-08-08: the
[perf/memory deep review](2026-08-08-molmo2-perf-review.md) (owner
ask 13:09Z) measured the gaps this bundle closes, with file:line
receipts and idle-local-GPU microbenchmarks. This pre-reg bundles
ONLY the review's S-effort items (its recommended-sequence rungs 1–3)
into one before/after benchmark; every heavier item is deliberately
out of scope (list below). **Zero training-run risk: nothing lands on
`main` or touches the box before the 60k run + its chained evals
finish (~23Z 08-08); the benchmark runs branch-only on the idle local
H100.** Execution cost ceiling **≤ 3 GPU-h** (bench ladder ≈ 1.7 +
re-run contingency + a 4-GPU box smoke ≈ 0.3 at adoption time).*

## Question

The review measured the loss-bearing suffix attention running the
MATH sdpa backend (13× vs cuDNN per layer), three per-step
host↔device sync families, a 60 MB/step embed clone, and a vram
"peak" metric that is a lifetime ratchet. Combined estimate from the
microbenches: **~8–15% of the 2.2 s step at S effort**. This pre-reg
asks: **does the S-bundle actually buy that on a pinned end-to-end
training run, with every item individually parity-gated — and is the
gain worth carrying into every future launch?**

## The bundle (exact change specs, pinned)

Each item ships with its own parity oracle; **an item that fails its
oracle is dropped from the bundle, never re-toleranced post hoc.**
Line numbers are HEAD at finalization (post-annotation pass,
commit `36495d9`); the landing diff may shift them but not the sites.

**P1 — suffix sdpa → cuDNN, TRAINING-ONLY scope.**
`bijou/decoders/ar_molmo2.py:216-222` pins
`[FLASH, EFFICIENT, MATH]`; the dense suffix mask rejects FLASH and
`enable_gqa` rejects EFFICIENT, so training runs MATH (measured
0.968 → 0.075 ms/layer under cuDNN, ×36 layers, fwd + more in bwd).
Change: prepend `SDPBackend.CUDNN_ATTENTION` to the list **when the
decoder is in training mode only**. The same `_transform` serves
decode — and every eval byte-anchor we own (banked panel npzs,
preflight byte-matches, state-copy rows) assumes decode numerics are
frozen. Decode keeps the exact HEAD dispatcher list. *Oracles:
(a) one-step parity — fixed micro-batch, fixed seed, MATH vs cuDNN:
loss abs diff ≤ 1e-3, grad-norm rel diff ≤ 1e-2 (bf16 kernel-family
noise, not bitwise by construction; bounds banked here, before the
run); (b) 50-step loss-curve overlay on the bench config — cuDNN
curve inside the MATH curve's step-to-step noise band, no NaN/inf
(this is also the pytorch#122695-family crash gate: the known
failure is the fused backward, so the gate must run backward);
(c) decode byte-match — a pinned small eval plan decodes
byte-identical on the branch vs HEAD (proves the training-only
scoping actually scoped).*

**P2 — windowed vram peak logging.** `bijou/train.py:4002-4005` logs
`torch.cuda.max_memory_allocated` and never resets — a lifetime
ratchet (the 41,780/42,940 "creep" we spent a tick investigating).
Change: per log window, emit new field `vram_window_peak_gib`, then
`reset_peak_memory_stats`; **`vram_alloc_peak_gib` keeps its exact
monotone lifetime semantics** as a Python-side running max, so every
existing scan/babysit consumer parses unchanged. *Oracle: unit test —
schema (both fields present, lifetime ≥ window, lifetime monotone
across windows); zero numerics touched.*

**P3 — per-step host-sync removals (three sites).**
- *(a)* `bijou/molmo2/model.py:114` — `int(is_patch.sum())`
  validation syncs every encode (×6 chunks/step). Change: keep the
  guard, make it device-side (`torch._assert_async` on the equality);
  abort-on-mismatch survives, the friendly message degrades to a
  device assert (this guard has never fired in any run — acceptable).
  Fallback if `_assert_async` misbehaves under the stack: run the
  host-sync check only when not in training mode.
- *(b)* `bijou/molmo2/text.py:101` — `bool(is_extension.any())`
  syncs on every wte call. Change: drop the branch — always compute
  the extension lookup + `torch.where`. Output is **bitwise
  identical** in both regimes (`where` with an all-False mask returns
  `embeds` rows exactly). Honest cost flagged in advance: the
  no-extension case (the ~1.1k-token prefix) now materializes a
  [B, S, hidden] extension lookup (~60 MB bf16) it used to skip —
  we trade bandwidth for CPU run-ahead; the bench adjudicates, and
  if the bundle read is flat this is the first ablation suspect.
- *(c)* `bijou/decoders/ar_backbone.py` sum-form losses
  (`ar_backbone_loss_sums`, the live chunked path — ×6 chunks/step):
  boolean advanced indexing (`elementwise[valid].sum()`,
  `elementwise[action_positions].sum()`, …) forces `nonzero` +
  host sync. Change: mask-multiply / straight sums —
  `F.cross_entropy(..., reduction="none")` already writes exact 0.0
  at IGNORE_INDEX positions, so the masked sums equal the indexed
  sums **up to fp reduction order only**. The mean-form path
  (`ar_backbone_losses`, byte-anchored by the loss oracles) is NOT
  touched. *Oracles: (a) device-assert still aborts on a planted
  mismatched batch; (b) bitwise wte equality on fixed batches with
  and without FAST tokens; (c) CPU loss-oracle re-pin — sum-form
  fp64 reference equality (reduction-order change is declared, so
  the re-pin is pre-registered here, not a post-hoc tweak; the
  mean-form anchors must pass UNCHANGED).*

**P4 — drop the per-step prefix-embedding clone.**
`bijou/molmo2/model.py:120` clones [B·S, hidden] (~60 MB bf16/step
×6 chunks) before the masked feature add. The source is the fresh
non-leaf `wte` output — nothing else aliases it, embedding backward
does not consume its output value, so the in-place masked add on the
view is autograd-safe and **bitwise identical**. Change: delete
`.clone()`. *Oracle: fixed micro-batch fwd+bwd — outputs and ALL
parameter grads bitwise vs HEAD; plus no autograd version-counter
error (torch raises loudly if the aliasing claim is wrong).*

## Explicitly out of scope (each needs its own pre-reg)

ViT SDPA path (M, parity contract); `F.rms_norm` + valid-row CE
(bf16-bitwise HF parity suites need re-gating);
`--activation-checkpointing` lineage flip (a launch decision with a
batch re-tune, not a code change); static shapes (review verdict:
keep dynamic); torch.compile (idea #2b — P3 is its prep work, not
its landing). The shape-annotation long tail (`processor.py`,
`cache.py`, `encoders/molmo2.py`, `decoders/ar_molmo2.py`) rides
whichever item touches each file first, comments only.

## Benchmark protocol (pinned)

- **Where/when:** branch `perf-pass1`, local idle H100 (1×), any
  quiet window — pre-23Z allowed (touches neither the box nor any
  eval artifact). Launch via `run_detached.sh`; babysit entry at
  launch; util+rate checked at first poll (standing rule).
- **Config:** single-process `bijou.train`, flags byte-matched to
  the 60k launcher's data/model recipe (same three datasets, fps 30,
  camera-counts 1 2, max-crops 1, FAST tokenizer v2, aux/condition
  fields, batch 12, `--backward-chunks 6` — the chunked path is
  where the sync multiplier lives) minus torchrun/ZeRO/allreduce;
  fresh init, `--seed 0`, `--steps 320`, `--log-every 20`,
  eval/save off. Kernel-level deltas are geometry-dependent, not
  weight-dependent — fresh init is representative.
- **Ladder, run sequentially on the same GPU:** A = HEAD,
  B = HEAD + P1, C = full bundle. 320 steps each; **discard the
  first 80 steps (warmup/allocator), read median `s_per_step` over
  the last 240** (12 log records), plus `vram_window_peak_gib` (C
  carries P2; A/B read the lifetime field).
- **Primary read:** (A − C)/A relative median step time.
  **Secondary:** (A − B)/A isolates P1. **Guard read:** C's steady
  vram vs A's — regression > 2% fails the bundle read (P3b's extra
  60 MB is the named suspect).
- **Transfer check at adoption time (not a decision gate for
  landing):** one 100-step 4-GPU box smoke (~0.3 GPU-h, post-evals)
  before the first lineage launch that carries the bundle — expect
  ≥ half the local relative gain; if not, record and investigate
  before any launch adopts it.

## Frozen decision rules

1. **Parity first:** any item failing its oracle is dropped; the
   bench ladder runs with the surviving bundle. Oracle bounds above
   are frozen — no post-hoc widening.
2. **C ≥ 5% faster than A** (primary read) → the bundle lands on
   `main` post-60k-close + chained evals, and every subsequent
   launch (noise-ladder stage 2 excluded — eval-only) carries it.
   The 5% bar is deliberately below the 8–15% estimate: the bundle
   also buys compile-prep (P3) and observability (P2), so it clears
   at half the estimate.
3. **C < 5%:** land only P2 (observability, zero numerics) and the
   strictly-free bitwise items (P4, P3b if vram-clean); P1/P3a
   park with their measured numbers in the review post's ledger —
   a real result either way.
4. **Any cuDNN instability** (crash, NaN, curve outside the noise
   band) → P1 drops, the pin's "cheap insurance" comment gets
   replaced by a receipt: this geometry reproduces the crash family,
   dated. The remaining bundle re-benches as C′.
5. `check.py` green + the full oracle list green before the landing
   commit; landing is one commit, revert is one revert.

## Numbered expectations (banked before the bench)

1. All four parity oracles pass on the first run — confidence
   medium-high (P3b/P4 are bitwise by construction; P1's bounds are
   the uncertain ones).
2. P1 alone (read B) buys ≥ 4% median step time — confidence
   medium (35 ms fwd measured; backward multiplier is the unknown).
3. Full bundle (read C) lands in the review's **8–15%** window —
   confidence medium-low (sync-removal gains via restored CPU
   run-ahead are the least microbench-predictable term); ≥ 5%
   (the decision bar) — confidence medium.
4. No cuDNN backward crash on standard head_dim-128 dense geometry —
   confidence medium-high (the Gemma crashes were ragged-geometry).
5. `vram_window_peak_gib` steady-state reads ≥ 1 GiB under the
   lifetime ratchet on the bench config — confidence medium — and
   turns the next box-run "creep" judgment into a direct read.

## Cost & scheduling

CPU: this document + the branch diff + oracles (any window). GPU:
bench ladder 3 × 320 steps × ~2.2 s ≈ 35 min/config ≈ **1.7 GPU-h**
local; ceiling ≤ 3 GPU-h including one contingency re-run and the
box transfer smoke. Sequencing: **bench may run in any quiet
local-GPU window including pre-23Z** (branch-only, box untouched);
**landing on `main` is gated on the 60k close + chained panel/fields
evals completing** (~23Z), because P1 changes training numerics and
nothing may perturb a live lineage mid-run (and the eval byte-anchors
must be banked against HEAD decode before any code moves). Adoption
is per-launch and recorded in each launcher's pre-reg from then on.

## Finalization record

Drafted and finalized in one session 2026-08-08 ~14:3xZ (the queue
item planned DRAFT → finalize, but nothing here waits on data: the
review's measurements are already banked, all change specs and
oracle bounds are pinned above from a re-audit of HEAD `36495d9`
this session). Execution gets its own queue entry
(`molmo2-perf-pass1-exec`, gpu-local) and babysit entries at launch.
Expectations 1–5 banked before any bench step runs.
