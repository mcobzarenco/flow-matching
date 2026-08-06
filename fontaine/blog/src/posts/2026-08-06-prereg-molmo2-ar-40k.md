# 2026-08-06 — Pre-registration: Molmo2-4B AR trunk, 40k × 4×DDP (`fontaine_molmo2_ar_40k_ddp4`)

*Status: PRE-REGISTERED before launch. Owner steering 2026-08-06
18:10Z "I want to run molmo2 tonight" + 18:12Z "Agreed" + 19:11Z
"Agreed on molmo2 on the 4x box … smoke test with ddp enabled" — the
[port plan](2026-08-06-molmo2-port-plan.md) §6 AR-first amendment's
phase-1 run. Code: the AR decoder arm landed this evening
(`c1119fb`+`f569f94`+`4ce9136`, check.py 337 green; keystone oracle:
prefill+continue ≡ monolithic multimodal forward under left padding).*

## 1. Question

Does a **video-grounded VLM trunk** (Molmo2-4B: spatio-temporal
pointing/tracking pretraining, Qwen3-4B decoder) beat our **text-first
Gemma-4 E2B** trunk as an AR VLA — FAST actions + aux text on the live
trunk — at matched data, matched steps, matched action tokenizer? The
grounding probes located E2B's error in frame-dependent level
mis-estimation and weak visual-token use (#11); Molmo2's pretraining
objective is the nearest open-weights match to "where is the gripper
and what is it doing". The untrained-gen probe (18:4xZ) showed the raw
trunk already reads our rig scenes accurately under our exact prompt —
night-and-day vs Gemma's refusals.

## 2. Recipe (one variable: the trunk; plus the scale-out it needs)

Mainline `arb_rcond` recipe verbatim where the trunk allows it:

- `--decoder ar_backbone --backbone allenai/Molmo2-4B --max-crops 1`
  (410 image tokens/camera — the smallest layout inside the shipped
  distribution; Amendment-2 pixel-math rationale)
- FAST tokenizer v2 (`vocab_total` 1,026), block anchored at
  **[152,064, 153,090)** — the second extension block
  (`fast_block_base`); trainable = `fast_embed` + fresh untied
  `fast_head` rows + decoder blocks + `ln_f` + `state_proj`;
  **frozen = `wte` (both matrices) + shipped `lm_head`** (the 18:1xZ
  freezing split; aux text reads the frozen head, grads flow through)
- aux fields `subgoal holding progress event visible`, aux-dropout 0,
  field-dropout 0.1; conditioning `subgoal outcome smoothness`
  (0.1/0.5 dropouts); instruction-augment 0.5; camera-kind-dropout 0.1
- `--decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100`
  (vision tower frozen — no `--backbone-vision-lr`; a vision-unfreeze
  rung is a follow-on, not this run)
- data: `community_curated_v0`, fps 30, camera-counts {1,2}, holdout
  0.1 @ split-seed 0 (E1 gate: **878 datasets / 38,571 episodes /
  18,636,749 frames / dims 6/6** — byte-identical to the arb_rcond
  mainline banner; any deviation aborts. Verified in the smoke ✓)
- **40k steps, 4×DDP, B12/rank (global 48)** — the e4b-screen scale-out
  rung; batch semantics FROZEN at launch. Declared confound vs the
  1×GPU B10 E2B mainline: global batch 48 vs 10 — the e4b screen
  carries the same one, and the comparison is a screen, not a paired
  ablation. Warmup 1000, seed 0, save 2500, eval 500.
- Memory plumbing (no semantic content): `--backward-chunks 6` (6×2,
  gradient exactly equivalent) + `--zero1` (ZeRO-1 optimizer sharding,
  exact) + `--chunk-grad-allreduce` (explicit in-place gradient
  allreduce instead of DDP's reducer, equal up to fp reduction order —
  §3 rung-6 amendment).

## 3. Gates before launch

- F1 memory smoke (DDP4, 150 steps of the exact recipe, owner-asked):
  rc=0 AND peak ≤ ~75 GiB/GPU. Ladder: B12 direct → B12 chunked 2×6 →
  B12 chunked 6×2. **Rung 1 (B12 direct): OOM at 77.5 GiB in the
  forward (MLP intermediates; measured 19:5xZ) — REJECTED. Rung 2
  (B12 chunked 2×6): OOM at step 2's forward once Adam materialized —
  REJECTED. Rung 3 (B12 chunked 6×2): OOM at a forward RMSNorm with
  77.46 GiB already allocated — the chunk ladder is EXHAUSTED, and the
  mechanism is fully measured: per-rank STATIC budget once Adam
  materializes = bf16 weight copy 9.7 + fp32 masters 19.4 + DDP fp32
  grad buckets 14.6 + Adam moments 29.1 ≈ 73 GiB + CUDA/NCCL context ≈
  76–77 GiB on a 79.18 GiB card (~2 GiB activation headroom) — no
  chunk size closes a static gap; REJECTED.**
- **Amendment (F1 rung 4): B12 chunked 2×6 + `--zero1` (ZeRO-1,
  `ZeroRedundancyOptimizer`, commit `a08db04`).** NOT a recipe change:
  update semantics are exact (each parameter's Adam state lives on one
  rank; updated shards broadcast per step), machine-checked by a
  2-process oracle — ZRO(AdamW) bit-equal to plain AdamW over this
  run's param-group shape with a stepping scheduler, checkpoint
  round-trips into both sharded and un-sharded resume
  (`tests/test_zero1.py`). Global 48 and every LR constant unchanged.
  **Measured (20:16–20:23Z): OOM at step 1's SECOND chunk forward,
  77.5 GiB allocated — REJECTED, and the vram traces across rungs
  rewrite the §3 mechanism: the "static ~77 GiB once Adam
  materializes" story was over-attributed. Measured components:
  init static 33.9 GiB (masters + bf16 weights + context); activations
  ~2.8 GiB/sample; autocast bf16 weight cache ~9.7 GiB live during
  each forward; DDP fp32 grads +14.6 GiB after the first chunk
  backward; Adam +29.1 GiB (unsharded) at the first optimizer step.
  So a 6-sample chunk's forward with grads resident (48.5 + 9.7 + ~17
  ≈ 75–77 GiB) OOMs in step 1 REGARDLESS of optimizer sharding —
  rung 2 died there too (its "step 2 once Adam materialized" reading
  was inferred from arithmetic, its vram sampler had died); rung 3
  (6×2) is the one that genuinely completed step 1 and died at step 2
  when unsharded Adam (+29.1) landed.**
- **Amendment (F1 rung 5, pre-declared before its smoke, 20:2xZ): B12
  chunked 6×2 + `--zero1` — the two fixes compose: 2-sample chunks
  keep every forward inside the budget (proven by rung 3's completed
  step 1), zero1 removes the Adam block that killed rung 3 (29.1 →
  ~7.3 GiB/rank). Predicted steady-state peak ≈ 55.8 static + 9.7
  bf16 cache + ~5.7 activations ≈ 71–73 GiB (~6 GiB margin, inside
  the ≤~75 GiB pass rule). Fallbacks if rejected: 12×1 chunks
  (~3 GiB more margin), then bf16 grad buckets (−7.3 GiB, composable).
  **Measured (20:28–20:3xZ): OOM in STEP 1's backward, 77.0–77.15 GiB
  allocated on ALL FOUR ranks — REJECTED, and the trace comparison
  closes the mechanism: rung 3's trace ALSO peaked 81 GiB (nvidia-smi)
  in step 1's final-chunk backward; its "completed step 1" was
  fragmentation luck at the wall, not margin. The block the arithmetic
  missed: under no_sync-first chunk accumulation, autograd allocates
  plain fp32 grad tensors (+14.6 GiB) — then the final SYNCED chunk
  materializes DDP's reducer bucket buffers (+14.6 GiB more, a full
  duplicate; `gradient_as_bucket_view=True` is already set but cannot
  help — the views only exist while the reducer owns the backward,
  and with `zero_grad(set_to_none=True)` the duplicate recurs every
  step). Step-1 sync-chunk backward ≈ 33.9 init + 14.6 grads + 14.6
  buckets + 9.7 saved bf16 casts + ~5 activations ≈ 78 GiB — the
  measured wall, zero1-independent.**
- **Amendment (F1 rung 6, pre-declared before its smoke, 20:5xZ): B12
  chunked 6×2 + `--zero1` + `--chunk-grad-allreduce` (commit at
  launch): every chunk's backward stays in no_sync and the accumulated
  fp32 grads allreduce IN-PLACE once per step — DDP's reducer buckets
  never materialize, removing the 14.6 GiB duplicate exactly.
  Semantics: sum/world, identical to DDP's average up to fp reduction
  order (the tolerance `--backward-chunks` already declares);
  machine-checked by a 2-process gloo oracle — flag path == DDP-sync
  path == single-process global-batch reference to 1e-12 over 3
  optimizer steps (`tests/test_chunk_grad_allreduce.py`; check.py 342
  green). Predicted peaks: step 1 ≈ 33.9 + 14.6 grads + 9.7 casts +
  ~5.7 act ≈ 64 GiB; steady state adds sharded Adam +7.3 ≈ 71 GiB
  worst (~6 GiB margin, inside the ≤~75 GiB rule). The declared 12×1
  fallback is SKIPPED with reason: it shrinks only activations
  (~−2.8 GiB → predicted ~75 GiB, inside the measured death band
  77±1 with batch-length variance over 40k steps — a smoke could pass
  and the run still die at a heavy batch); rung 6 removes the
  measured largest transient instead. Fallback if rung 6 is rejected:
  rung 6 + 12×1 (composable, −2.8 GiB).
  **Measured (20:49–20:56Z): OOM in step 1's backward AGAIN, 77.0
  GiB allocated — REJECTED. The new sync path verifiably ran (banner)
  and the peak did not move: three rungs have now each removed a real
  block (Adam sharded −21.8, reducer duplicate −14.6) without moving
  the ~77 GiB step-1 backward wall ⇒ the component arithmetic is
  missing ≥10 GiB common to all rungs, and further arithmetic-driven
  rungs are not admissible evidence.**
- **Amendment (F1 rung 7, pre-declared before its smoke, 21:0xZ):
  the declared rung-6 fallback — B12 chunked 12×1 + `--zero1` +
  `--chunk-grad-allreduce` — PLUS a measurement instrument:
  `BIJOU_MEM_SNAPSHOT` (env-gated
  `torch.cuda.memory._record_memory_history` from process start,
  per-rank snapshot pickle dumped at OOM; smoke-only, never set on a
  launch). Predicted peak ~74 GiB by the (now discredited)
  arithmetic — the PASS RULE stays rc=0 AND peak ≤ ~75 GiB, but
  whatever the outcome, the snapshot replaces arithmetic with
  measured allocation attribution for the next decision. If rung 7
  is rejected, the ladder PAUSES for the snapshot read: the next
  rung is chosen by the measured largest removable block
  (candidates: activation checkpointing #20, bf16 frozen params,
  loss-region chunking), posted before its smoke.
  **Measured (21:03–21:44Z): rc=0 — IT TRAINS. 150 steps, loss
  16.1→8.0, grad norms sane, step-100 eval + zero1-consolidated save
  both exercised (save ≈ 13 min — a 40k schedule needs a save-every
  rethink). Peak 78,057 MiB on the nvidia-smi sampler — FAILS the
  ≤~75,000 rule — but the sampler reads the RESERVED pool, which
  under expandable_segments never shrinks; the rule's metric is now
  known to be a shadow. Steady rate 3.83–3.88 s/step (the "last5"
  13.2 in the verdict line is save-window-skewed) → 40k ≈ 43 h,
  over the F2 30 h line ⇒ the 10k-screen branch is the admissible
  schedule for this config. REJECTED as pre-declared (peak rule);
  ladder paused for the snapshot read — forensics A (6×2 +
  instrument, OOM-moment attribution) and B (12×1, 30 steps,
  true-allocated peaks via the new per-step vram log fields) decide
  the next rung.**
- **Forensics verdict (21:5xZ, snapshot banked
  `reports/mem_forensics/`): the block three fixes never touched is
  DDP's reducer bucket buffers, allocated AT CONSTRUCTION
  (train.py DDP wrap → 13.56 GiB measured), not at first sync.
  Measured OOM-moment ledger (rank 0, live 71.0 / reserved 77.5 —
  ~6.5 GiB allocator fragmentation): fp32 params 18.1 + DDP buckets
  13.6 + accumulated fp32 grads 13.6 + autocast bf16 casts 9.7 +
  activations ~8.8 (2-sample chunk; the arithmetic said 5.7) +
  sharded Adam 6.8.**
- **Amendment (F1 rung 8, pre-declared before its smoke, 22:0xZ): B12
  chunked 6×2 + `--zero1` + `--chunk-grad-allreduce`, where the flag
  now skips the DDP wrapper ENTIRELY — `broadcast_module_states`
  replicates the constructor's rank-0 broadcast, plain autograd
  accumulates chunk gradients, the explicit in-place allreduce stays
  the whole per-step sync. Removes the measured 13.6 GiB bucket
  block. Oracle re-verified: bare-module path from deliberately
  rank-divergent init == DDP-sync path == single-process global-batch
  reference to 1e-12 (2-proc gloo). Predicted live peak ≈ 57.5 GiB,
  reserved ≈ 64–66 — and the pass rule is UPGRADED to the true
  metric now that it exists: rc=0 AND max `vram_alloc_peak_gib`
  (torch max_memory_allocated, logged per step) ≤ 71 GiB (≥8 GiB
  allocated headroom), nvidia-smi recorded as context only.
  **Measured (smoke 21:59–22:4xZ, 150 steps): rc=0; max
  `vram_alloc_peak_gib` 66.67 flat from step 20 on — PASSES the ≤71
  rule with 12.5 GiB allocated headroom (reserved 67.78; nvidia-smi
  sampler peak 71,345 MiB incl. context — vs ~78,000 on every dead
  rung); steady rate 2.517–2.55 s/step (last5 11.8 is
  save-window-skewed, as rung 7's was); loss 16.1 → 8.0, grad norms
  sane; step-100 eval + consolidated save AND the step-150 endpoint
  save both exercised on the no-DDP path (consolidation ≈ 15 min per
  save boundary — ~16 saves ≈ +3.5 h wall on a 40k, accepted).**
- F2 wall: 40k × rate ≤ 30 h ⇒ full 40k; else this pre-reg SHRINKS to
  a 10k screen (label changes to `_10k`), no mid-run change.
  **Projected: 40k × 2.55 = 28.3 h ≤ 30 h ⇒ FULL 40k stands** (+ ~3.5
  h of save-boundary consolidation on top, outside the F2 metric as
  declared at rung 7).
- Box pytest green at the launch commit (337 passed, run 19:4xZ ✓);
  the smoke also exercises eval/probe decode + the wandb metric path
  (offline) + the Molmo2 checkpoint writer at step 100.

## 4. In-run instruments and kill line

- Probe (eval-every 500): AR greedy decode chunk MAE on the probe set
  — the same instrument as every arb run. **New-trunk band: none
  declared** (no Molmo2 prior); anchors for READING (not killing):
  E2B arb family probes descend into ~10–11 @40k.
- **K1 kill line**: NaN/inf loss, OR the probe fails to descend below
  its own step-2500 value by step 10k, OR probe > 25 sustained for 3
  consecutive evals after step 5k. Kills wait for save boundaries.
- @10k, @20k, @30k: record-only milestone notes (Discord), no
  discretionary kills — the panel decides.

## 5. Endpoint reads (frozen before data, this section is the contract)

Chained after 40k, 4-GPU sharded, the family voice:
`panel_curated_v0_k4l2` (plans/holdout_curated_v0_k4l2.json, holdout
0.1 split-seed 0), stem
`eval__fontaine_molmo2_ar_40k_ddp4__step_040000__panel_curated_v0_k4l2`.

- **Read 1 (primary):** panel pooled chunk/first MAE vs the E2B AR
  anchors — A-s0 mainline **7.7966 / 3.9422** (same panel, same
  plan). Classification: BEATS if pooled chunk < 7.30 (−0.5, ~the
  family's seed spread); PARITY within ±0.5; WORSE beyond. Paired
  per-frame Δ + CI95 via the npz where row alignment holds (same
  plan ⇒ it should; a pairing failure is reported, not silently
  pooled).
- **Read 2:** state-copy fallback rows byte-match the banked panel
  values (11.7639/2.5851) — instrument integrity, not a result.
- **Read 3 (context):** vs e4b screen milestones (7.54@10k probe
  family) and arm C statedrop (10.50) — narrative only.
- Decision: BEATS ⇒ Molmo2 becomes the phase-2 flow-trunk candidate
  (frozen AR-adapted prefix — kills the −2.7 confound); PARITY ⇒
  grounding-probe follow-ups decide; WORSE ⇒ clean null banked, E2B
  stays mainline (VLM4VLA says nulls are modal — transferable either
  way).

## 6. Cost

~40k × 2.55 s/step ≈ 28.3 h (+ ~3.5 h saves) on 4×H100 + ~2 h panel
eval. Checkpoints: full-trunk snapshots (~9.7 GB bf16) every 2500 —
~160 GB transient, pruned to milestones at the boundary (disk 6.6 T
free).

## Finalization amendment (pre-launch)

Filled from the smoke before `torchrun` fires; the launch commit hash
and the filled numbers are the launch record.

**Finalized 2026-08-06 22:5xZ.** Rung 8 (B12 = 6×2 chunks + `--zero1`
+ `--chunk-grad-allreduce` with no DDP wrapper) is the launch config:
smoke rc=0, true allocated peak 66.67 GiB (≤71 rule, 12.5 GiB
headroom), 2.52–2.55 s/step ⇒ full 40k ≈ 28.3 h + ~3.5 h saves.
Launcher `fontaine/scripts/box/launch_box_fontaine_molmo2_ar_40k_ddp4.sh`
at its defaults (STEPS=40000, BATCH=12, BACKWARD_CHUNKS=6); run name
`fontaine_molmo2_ar_40k_ddp4`. Box pytest 337 green at `fd8bc0e`
(19:4xZ); the launcher-only delta to `4d530a1` touches no Python.
Launch commit: this post's finalization commit.
