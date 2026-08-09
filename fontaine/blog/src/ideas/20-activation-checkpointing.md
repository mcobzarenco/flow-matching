# 20. Activation checkpointing for live-trunk training — `confirmed`/landed (wrap + keystone oracles 2026-08-07; CUDA sdpa-pin fix 2026-08-09; GPU ladder = the K smoke item)

*Tag: `activation-ckpt` · idea #20 · [index](../ideas.md)*

The Molmo2 AR smoke measured the wall: fp32 masters + DDP grad
buckets + Adam on a 3.7B trainable set ≈ 63 GiB static on an 80 GiB
card, and at ~2.4 GiB/sample of saved activations (820 image tokens ×
36 layers × 9,728-wide MLP) only ~2-sample chunks fit. Chunked
backward works (gradient-exact) but 6 passes/step taxes throughput.
torch.utils.checkpoint over the decoder blocks would cut saved
activations to ~1 layer's worth for ~30% recompute — the standard
trade at this scale. Scope: the Molmo2 transformer first (uniform
blocks make it trivial), Gemma later if a live-trunk E4B+ run recurs.
Gate: keystone oracle (checkpointed ≡ plain forward/backward, loss
bit-close) + a measured chunk-size ladder re-run.

**LANDED 2026-08-07 ~06:4xZ** — `--activation-checkpointing` in
`bijou.train`: non-reentrant `torch.utils.checkpoint` per decoder
block in `Molmo2Transformer`, with a single-layer KV shim so the live
cache is never mutated inside the checkpointed region (backward
recompute would double-append the layer's K/V and break the replay
against the [B,1,S,T] mask); the real append happens once, outside,
with the escaped graph-connected K/V — suffix CE gradients still
reach the prefix through the cache. Engages only under grad: no-grad
encodes, eval and generation take the plain path untouched (the F arm
is bitwise unaffected even with the flag on). 4 keystone oracles
(`tests/test_molmo2_activation_checkpointing.py`): the joint K-step
and a transformer-level prefill+cached-suffix pass are BITWISE the
plain step (loss + every param grad, cache contents included), with a
call spy pinning that checkpointing actually engaged (2×blocks calls
— no vacuous equality); no-grad and F-arm paths never enter
checkpoint. The K launcher carries the flag. The measured ladder's
SCRIPT landed 2026-08-07 ~06:5xZ (`smoke_attach_k_ddp4.sh`, B12c6 →
B8c4 → B6c3 vs the 71 GiB alloc-peak gate — see #4); still open: RUN
it on the box at the endpoint window.

**2026-08-08 ~15:0xZ — REAL BUG FOUND (perf pass-1 bench detour,
receipts in `outputs/train/perfpass1_parity_A.launch.log` round 3):**
the flag CRASHES on CUDA at the first backward through the AR suffix
— the non-reentrant checkpoint's recompute runs during backward,
**outside the `sdpa_kernel([...no cuDNN])` pin** that wrapped the
forward, so the recompute dispatches a different sdpa backend than
the saved forward (fp32 MATH score tensors saved; bf16 fused shapes
recomputed) and aborts on tensor-metadata mismatch. The 4 keystone
oracles are bitwise-green and never caught it because CPU/tiny runs
dispatch identically inside and outside the pin. Consequence: the
perf review's "flip the flag on the next lineage" recommendation has
a named PREREQUISITE FIX — move the backend pin inside the
checkpointed function (or wrap the backward too) — and the perf
pass-1 P1 change (training-mode cuDNN re-admit) incidentally
removes the divergence for the training path. Fix rides the
act-ckpt lineage-flip pre-reg, with a CUDA regression oracle.

**2026-08-09 ~04:2xZ — FIX LANDED (the bug fired live first):** the
K-smoke ladder's rung 1 (the first real CUDA consumer of the flag)
crashed at its first backward with exactly the predicted
`CheckpointError` — fp32 MATH tensors saved under the suffix pin,
bf16 fused shapes recomputed outside it. The named fix landed the
generic way: `_checkpointed_block` captures the ambient sdpa backend
set at forward time (`_ambient_sdpa_backends()` reads the four
global flags, so it reconstructs whatever pin is active — full
dispatcher for the prefix encode, non-cuDNN pin for the suffix) and
the checkpointed callable re-enters `sdpa_kernel` with that set, so
backward recompute always dispatches the saved forward's backends.
Prefix-encode backend selection is untouched (its ambient set is the
full dispatcher, and re-applying it is a no-op). Oracles: the 4 CPU
keystones stay bitwise-green; new capture unit test (pin
reconstruction, in `check.py`); new `@pytest.mark.gpu` regression on
the exact crash shape (MATH-pinned forward + bf16 autocast +
backward outside the pin), calibrated non-vacuous — the unfixed
block body raises the production `CheckpointError` on the same
scenario, the fixed path is bitwise the plain step on an H100.
