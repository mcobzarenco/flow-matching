# Molmo2 training perf & memory — deep review (owner ask 13:09Z)

*2026-08-08 ~14:2xZ — review-only: findings with file:line receipts,
effort (S/M/L) and risk tags, ranked by expected payoff. **No model
code changed** (the 60k run is live and runs as-launched; every
proposed change needs its own pre-reg + before/after benchmark —
review now, surgery with instruments). Owner scope: (a) unnecessary
copies vs in-place, (b) attention kernels, (c) static vs dynamic
tensor sizes, (d) memory — at low complexity cost. Method: three
parallel read lenses over `bijou/molmo2/*`, `bijou/encoders/molmo2.py`,
`bijou/decoders/ar_molmo2.py` + `ar_backbone.py`, and the
`bijou/train.py` path; the two headline kernel claims were
**measured on the idle local H100s** (microbenchmark, ~zero GPU-h)
and the code receipts spot-checked before posting.*

## The baseline (what's already right — don't re-fix)

The training path is not virgin territory. Already in place and
verified: **ZeRO-1** + **backward chunking** (`--backward-chunks 6
--chunk-grad-allreduce`, which removed DDP's ~14.6 GiB reducer
duplicate), **fused AdamW** (one kernel per group, not foreach),
**`expandable_segments:True`** in every launcher (the
dynamic-shapes fragmentation mitigation; zero `empty_cache()` calls
repo-wide, correctly), **async checkpoint saves** (CPU-side, no GPU
spike), **bf16 autocast on the suffix forward** (halves the 153k-vocab
logits tensor), batch-max (never global-max) padding everywhere with
per-chunk re-padding, a **fused QKV** matmul + fused-gate SwiGLU in
the text stack, one shared RoPE table per forward (cheap, correct),
and — the dominant FLOPs — **the text prefix attention already
dispatches to the cuDNN flash kernel** (profiler-verified, dense
additive mask + `enable_gqa` notwithstanding). The teacher-forced
suffix correctly reuses a one-shot prefix KV cache rather than
re-encoding ~1.1k prefix tokens per suffix token. Training-side
tensor plumbing that *looks* like waste but is load-bearing parity
discipline (ViT fp32 attention casts, the dual-lookup `wte` that
replaced a ~1.5 GB per-call cat, the Gemma in-place `_patched_logits`)
is catalogued so we don't re-litigate it.

## Top findings, ranked

**1. The suffix forward — the loss-bearing attention — silently runs
the MATH sdpa backend.** `bijou/decoders/ar_molmo2.py:216-222` pins
`[FLASH, EFFICIENT, MATH]` (cuDNN excluded, inherited from the
Gemma ragged-geometry crash guard, pytorch#122695 family). But the
suffix always carries a dense additive mask (`text.py:503-511` ⇒
`is_causal=False`) — **FLASH rejects any mask** — and we pass
`enable_gqa=True` — **EFFICIENT rejects GQA** — so the only eligible
backend is MATH. The comment calls the pin "cheap insurance"; it
actually costs the fused kernel entirely. Measured at B=8, T=60,
S=1100: **0.968 ms/layer (math) vs 0.075 ms (cuDNN) — 13×**, ×36
layers ≈ 35 ms/step forward, more in backward — order **5–10% of a
~2.2 s step**. Molmo2's suffix is standard head_dim-128 dense
geometry, not the Gemma case the pin guards. *Fix: re-admit
`SDPBackend.CUDNN_ATTENTION` for the molmo2 suffix (one-line list
change) behind a one-step parity gate. Effort S, risk low-med.*

**2. The ViT runs reference eager attention — einsum + materialized
scores — with no SDPA path at all.** `bijou/molmo2/vision.py:125-137`:
`einsum → masked_fill → fp32 softmax → einsum`, 25 blocks × [B·views,
16, 729, 729] scores materialized every step. Measured per block at
B·views=16: **1.98 ms eager fp32 vs 0.146 ms SDPA-flash bf16 (13×)**
≈ 45 ms/step forward at that scale — and it doubles+ if a vision LR
ever unfreezes the tower (backward through materialized scores).
Sharp edge found on the way: under the live-trunk bf16 autocast the
`float32_attention` upcast (`vision.py:120-121`) is **silently undone
for the score matmul** (einsum is autocast-eligible) — we pay the
naive materialization without fully getting the fp32 fidelity it
exists to provide. Self-attention here is mask-free, bidirectional,
head_dim 72 — fully flash-eligible. *Fix: SDPA path in `ViTAttention`
with eager kept as the parity baseline. Effort M, risk med (parity
contract).*

**3. Hand-rolled RMSNorm, ~10× slower than the fused op.**
`bijou/nn.py:128-140` does the fp32 round-trip in ~6 elementwise ops:
measured 0.389 ms vs 0.038 ms for `F.rms_norm` on [8, 1100, 2560].
It runs ~4×/layer ×36 layers ×(prefix+suffix) ≈ 30–50 ms/step
forward + similar backward, and its fp32 intermediates inflate
activation memory wherever checkpointing is off. *Fix:
`F.rms_norm`; but the parity suites pin bf16-bitwise equality to the
HF reference, so this lands as a runtime-optional path with a parity
re-gate. Effort S, risk med.*

**4. Activation checkpointing exists, is oracle-pinned — and the live
lineage doesn't use it.** `--activation-checkpointing`
(`bijou/train.py:2086`, non-reentrant per-block with a KV shim so the
prefix cache appends exactly once) collapses the dominant dynamic
memory term — prefix activations measured at **~2.4–2.8 GiB/sample**
— to ~one layer's worth, for ~30% recompute. Neither the 40k nor the
60k launcher passes the flag. At vram peak 73.8/80 GiB this is the
single biggest headroom lever: it buys batch size (12 → likely
16-20/GPU), which typically nets *positive* throughput despite the
recompute. *Fix: flip the flag on the next lineage launch, pre-reg
the batch/throughput read. Effort S, risk low (bitwise oracles
already landed).*

**5. Full-vocab CE chain materializes and fp32-upcasts pad rows.**
`bijou/decoders/ar_molmo2.py:238-244` assembles [B, W, 153,090] via
`torch.cat` (base head + gap + fast block — the cat duplicates the
bf16 logits), then `ar_backbone.py:1124-1126/:1169-1177` (and
`ar_fast.py:435-439`) do `logits.reshape(-1, V).float()` — a full
fp32 copy **including the ~40–60% of rows that are IGNORE_INDEX**,
plus the fp32 log-softmax saved for backward. *Cheap fix: select
`targets != IGNORE_INDEX` rows before the upcast — semantics
unchanged, removes the batch-max-length dependence of the CE-side
tensors (S-M, low-med; re-pin the CPU loss oracles for reduction
order). Full segmented/chunked-lse CE is L and not justified at 4B
scale.*

**6. Per-step host↔device syncs (also the torch.compile blockers).**
(a) `bijou/molmo2/model.py:121` `int(is_patch.sum())` — sync every
encode; (b) `bijou/molmo2/text.py:101` `bool(is_extension.any())` —
sync ≥2×/step; (c) `ar_backbone.py:1132-1139` boolean advanced
indexing in the losses (`elementwise[valid].sum()`) — `nonzero` +
sync 2-3×/loss where a mask-multiply is exact and free (ignored rows
are already 0). *Fixes all S/low; percent-level step time from
restored CPU run-ahead, and they're the graph breaks if compile
(idea #2b) is ever picked up.*

**7. A full prefix-embedding clone per step.**
`bijou/molmo2/model.py:127` clones [B, S, 2560] (~60 MB bf16/step)
just to scatter image features in; the source is a non-leaf fresh
embedding output, so in-place `index_put_` is autograd-safe. *S,
low-med (verify against the parity/grad oracle).*

**8. vram "creep" is partly the metric, not the model.**
`bijou/train.py:4002-4005` logs `torch.cuda.max_memory_allocated`
and **never resets it**: `vram_alloc_peak_gib` is a lifetime ratchet
— one long batch permanently raises it (exactly the 41,780/42,940
bumps we investigated at the last tick). *Fix: add a
`vram_window_peak_gib` field + `reset_peak_memory_stats()` per log
window, keep the lifetime max in Python. S/low — turns the next
"creep" judgment call into a direct read.*

**Smaller, all S/low:** crops collate fp32 and cast on-GPU
(`encoders/molmo2.py:279`, `vision.py:413`) — bf16 at the collator
seam halves H2D bytes; `encode` computes a graph-dead `ln_f` over the
full prefix every step (`encoders/molmo2.py:421` → `text.py:548`) —
skip-flag; per-call opener-ids H2D + double `suffix_targets` build
(`ar_backbone.py:1068-1072`); `torch.tile` where broadcast indexing
works (`vision.py:425-428`); collate-side per-image `Resize` module
construction and double-copy batch assembly
(`processor.py:145-149,295`, `encoders/molmo2.py:272-290`) — worker
CPU, only matters if input-starved.

## Static vs dynamic shapes — measured answer: keep dynamic

The owner's max-size + slice question has prior art with numbers
([idea #2](../ideas/02-throughput-bucketing-compile.md)):
`--bucket-by-length` already exists (`bijou/data.py:822-902`, default
OFF) and the measured padding inflation under `--camera-counts 1 2`
batch-max padding is only **+5.09% → ≤3.6% step-time ceiling** —
below the deprioritize line. A static global max would pad *more*
than batch-max does, and the FAST-token suffix has **no hard cap**
(`bijou/fast/codec.py:57-72`), so a static scheme needs a truncation
policy for an unbounded dimension — complexity for negative expected
return. Prefix shape variance is already tame (410 image tokens per
camera is constant at `--max-crops 1`; only camera count 1|2 and text
length move it). *Verdict: don't. Revisit only if torch.compile
(idea #2b) or a wider camera selection lands; the cheap intermediate
if shape churn ever matters is width-rounding to multiples of 64/16
at the three collate `max()` sites.*

## Recommended sequence (each gets its own pre-reg + benchmark)

1. **Suffix → cuDNN backend** (S; one line + parity gate) — measured
   13×/layer on the loss-bearing site.
2. **Windowed vram peak logging** (S) — observability, zero risk.
3. **Sync removal batch** (6a-c + 7; S) — percent-level, compile prep.
4. **Activation checkpointing on the next lineage** (S flag +
   batch-size re-tune) — the big memory lever, throughput-positive.
5. **ViT SDPA path** (M) — biggest single kernel win after #1.
6. **Valid-row CE + fused RMSNorm** (S-M each, parity re-gates).

A combined 1+3+6 pass plausibly buys **~8–15% step time** on the
current recipe at S risk; 4 converts ~2.5 GiB/sample of activations
into batch headroom. All of it is post-60k work: nothing touches the
live run.

## Shape annotations (second aim)

Started in the same pass — `bijou/molmo2/` forward-path signatures
get `[dims]` per-arg annotations (one arg per line), long tail
tracked as a queue item so annotations don't crowd out the findings.
Convention: `x: [B, S, H]` comment on the arg line, dims named from
one shared legend per module docstring.
