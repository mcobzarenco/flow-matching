# Pre-registration DRAFT: Molmo2-ER 60k AR trunk run with rig data from step 0

*Drafted 2026-08-09 ~22:4xZ, from owner steering 22:14:00Z. Status:
**DRAFT — awaiting owner inputs** (rig dataset pointers + mixture
call + adamc kill go). Finalizes into an immutable pre-reg with a
param sheet posted in-channel for approval before any launch, per
the standing gate.*

## What and why

Owner proposal (22:14Z): a 60k-step AR trunk run initialized from
**`allenai/Molmo2-ER`** — MolmoAct2's embodied-reasoning
specialization of our exact Molmo2-4B trunk — with training
parameters matched to our AR 40k recipe, and the owner's rig
datasets included in the mix from step 0. The box frees by killing
`adamc_100k` (owner's call; its probe has risen three consecutive
evals off the 10.63@9500 run-best — our named watch agrees it is
not on a good trajectory).

External pricing for the init swap: MolmoAct2's own ablation —
Molmo2 → Molmo2-ER at fixed everything-else lifts LIBERO-Long
77.6 → 83.6 (+6.0), the largest single lever in their stack
([deep dive](2026-08-09-molmoact2-deep-dive.md)).

## Init: verified drop-in

- Config diff `allenai/Molmo2-ER` vs `allenai/Molmo2-4B`:
  `max_position_embeddings` 36864 → 16384 (RoPE metadata, no weight
  shapes; our sequences are far below both) + `transformers_version`
  stamp. Nothing else.
- Safetensors manifests: identical key sets, identical total size
  (19,403,476,800 bytes). Loader change is exactly
  `--backbone allenai/Molmo2-ER`.
- ER snapshot download to the box HF cache COMPLETE (verified
  22:35Z: 0 incomplete blobs, all 4 shards + processor/code files) —
  launch is not gated on weights.

## Recipe (re-pinned verbatim from the 40k launcher, deltas marked)

Matched from `launch_box_fontaine_molmo2_ar_40k_ddp4.sh` +
its pre-reg ([40k pre-reg](2026-08-06-prereg-molmo2-ar-40k.md)):
4×H100 DDP, eff-batch 48 (12/rank), ZeRO-1 + backward-chunks 6 +
chunk-grad-allreduce, decoder `ar_backbone`, FAST tokenizer v2,
`--max-crops 1`, fps 30, camera-counts 1 2, holdout 0.1 split-seed
0, aux fields `subgoal holding progress event visible` (dropout
0.0/field 0.1), condition fields `subgoal outcome smoothness`
(dropout 0.1, subgoal-dropout 0.5), instruction-augment 0.5,
camera-kind-dropout 0.1, decoder-lr 1e-4, backbone-text-lr 2e-5,
grad-clip 100, warmup 1000, eval-every 500, log-every 20, async
saves (default-on).

**Deltas vs 40k, each named:**

1. `--steps 60000` (owner spec).
2. `--backbone allenai/Molmo2-ER` (owner spec; init swap above).
3. `--train-data community_curated_v0 + <RIG DATASETS — TBD owner
   pointers>` (owner spec: from step 0).
4. `--save-every 5000` at 60k (keeps ~12 saves; the 40k cadence
   2500 at 60k would double checkpoint I/O; **owner may override to
   2500**).
5. Seed: `--seed 0` — owner override 22:46:40Z ("let's use the same
   seed too"): SAME shuffle seed as the 40k run; the fresh-seed
   standing rule was proposed (seed 2) in the sheet and explicitly
   overridden. Seed touches data order + head init only (init is ER
   weights). First launch 22:50Z at seed 2 stopped PRE-STEP-1
   22:52Z (~0 GPU-h), relaunched 22:53Z at seed 0.
6. `--num-workers 20 --prefetch-factor 4` kept (box has 1.4 TB host
   RAM headroom at batch 12/rank — the tiny10k OOM class was
   local-host-specific at batch 48×1; re-checked at first poll per
   the standing rule).

**NOT matched (named non-deltas):** no AdamC (matched = AdamW as in
the 40k run); vision tower stays frozen (the 40k recipe did not
pass vision LR; the vu5k screen owns that question separately —
mixing an untested unfreeze into this run would confound the ER
init read).

## Open inputs — updated 22:4xZ after owner go (22:36Z)

- **Rig datasets — RESOLVED**: `mcobzarenco/so101_pick_place_clean`
  (7 ep / 3,399 frames) + `mcobzarenco/so101_pick_place_v2` (50 ep /
  32,679 frames), ids confirmed by owner 22:40Z; both already on the
  box in `~/datasets/mcobzarenco/`, LeRobot v3.0, fps 30, action
  dims 6/6 — pipeline-compatible as-is.
- **Mixture — OPEN, arithmetic now pinned**: natural share =
  36,078 / 18.67M frames = **0.19%** → ~5.5k rig samples across the
  whole 60k×48 run ≈ each rig frame seen 0.15× in expectation — it
  cannot register. The loader dedups repeated roots (`bijou/data.py`
  path-dedup + cross-root duplicate-repo-id error), so there is no
  zero-code oversample. Recommended to owner in the param sheet
  (posted 22:43Z): a small per-root `--dataset-repeat` flag +
  oracle test, rig at ~5% (≈27× repeat ≈ 4 rig epochs) — inside the
  CL-triangle 2–20% replay evidence band
  ([page](../papers/cl-triangle.md)). Awaiting natural / 5% / other.
- **adamc_100k disposition — EXECUTED**: owner go 22:36Z; unit
  stopped 22:40Z at step ~11,840 (~35.7/310 GPU-h), all 4 GPUs
  verified 0 MiB. Final probe ladder ended 10.30@11500 = run-best
  (the 3-rise watch receded). step_010000 kept on box; weights-only
  upload to `fontaine-checkpoints` in flight (unit
  `hf-up-adamc10k`; optimizer.pt 32.6 GB stays box-local);
  `train_log.jsonl` banked box + local for the zero-GPU AdamC
  post-mortem chart.

## Frozen reads (finalize with numbers at param-sheet time)

- Probe ladder vs the 40k run's probe curve at matched steps
  (same eval cadence, same holdout) — the ER-init delta is the
  primary curve read.
- Endpoint: chained k4l2 panel_v2 eval in-unit (`--report` + npz per
  the standing rule), paired per-frame CI95 vs the banked 40k
  endpoint panel (6.0079/2.1871) and vs the 60k continuation panel
  (5.8602) — the latter is the steps-matched comparison.
- Rig-data effect: held-out rig episodes (if rig data ships a
  holdout-compatible split) scored at endpoint — record-only unless
  finalization pins a band.
- Kill lines: K1-style probe bars re-derived from the 40k curve at
  finalization; NaN/inf; vram near-OOM bar per first-poll actuals.
- Gate: ~**65 GPU-h** estimate (40k ran ~0.92 s/step-class recipe at
  eff-48; 60k steps ≈ 60/40 × the 40k cost + evals) — pinned at
  finalization from the 40k actuals.

## Cost note

Kills a live run ~11.4k/100k steps in (~35 GPU-h spent; owner
cost-call). The freed budget covers this run ~4× over at the gate
estimate.
