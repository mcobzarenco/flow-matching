# Stable-key re-bank executed: flow anchor 6.5997 — ADOPTED (#18.2)

*2026-08-06 ~08:3xZ. The pre-registered instrument break
([pre-reg](2026-08-05-noise-reseed-prereg.md), band amendment
[σ_draw finalization](2026-08-06-sigma-draw-finalization.md)) executed
at the first eval boundary after the box-batch reads, as registered.
One full-panel eval of flow-80k
(`bijou_flow_artrunk_h1024_40k_ddp2` @80k, Heun-30, N=1, seed 0,
`--noise-key stable`), launched 07:41Z, scored 08:30Z (25,800 frames,
~650 f/min — the fastest panel pass this box has run); reads by
the frozen protocol, in order:*

1. **Controls (hard gate): PASS.** state-copy and state-copy-norm
   summaries are **bitwise identical** to the banked index-keyed
   report across every cell (chunk_mae, chunk_mse, first_mae,
   per-motor, p50/p90) — the keying change touched nothing outside
   the noise path.
2. **Primary band: INSIDE — ADOPT.** Stable-key chunk_mae **6.5997**
   vs band [6.4882, 6.7582] (= 6.6232 ± 3·0.045, the floor binding
   over σ_draw_direct 0.02367). Shift vs the index-keyed same-file
   comparator: **−0.0242 ≈ 1σ_draw** — an ordinary fresh draw of the
   same noise distribution, exactly the pre-registered expectation.
3. **Secondary (quoted, not gated):** first_mae 1.9355 (index-keyed
   1.9335, +0.002); state-copy margin 5.185.

**Qualitative check (the shape of the draw, not just the scalar):**
per-motor MAE moved coherently and tiny — index
[4.43, 9.06, 9.27, 6.98, 5.55, 4.45] → stable
[4.39, 9.01, 9.22, 6.91, 5.55, 4.52] (five of six motors a hair
better, wrist-roll a hair worse); p50 4.91→4.96, p90 12.74→12.81.
No motor, percentile, or baseline shows anything but draw-level
jitter — the profile is the same model on the same frames under a
fresh noise draw, which is exactly what a keying change must look
like. Full trajectory renders: the eval's
[HTML report](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_stablekey_heun30.html)
(32 sampled frames).

## What changes

- **`stable` is now the quoted keying for all new flow numbers.** The
  re-banked deployment-class flow anchor on panel v1 k4l2:
  **chunk 6.5997 / first 1.9355** (Heun-30, N=1, noise-key stable).
  Index-keyed numbers stay valid as-labeled at frozen corpus; every
  banked npz remains its own paired-comparison reference.
- Ledger anchor row updated with the keyed pair; ideas #18.2 →
  banked/done.
- In-flight pre-registrations finish as registered (SnapFlow's
  endpoint reads run per its own pre-reg; its adopt band ≤ 6.7732 was
  σ-finalized before this eval opened and is untouched by the
  re-bank).

*Instrument note: this closes the noise-reseed amendment chain opened
by the owner deep-dive's finding 1 — corpus-index keying made every
flow number hostage to corpus mutations; stable keying decouples them.
Cost: one 49-min panel eval at a boundary the queue had to visit
anyway.*
