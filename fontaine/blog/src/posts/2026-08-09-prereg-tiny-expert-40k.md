# Pre-registration: tiny-expert capacity rung (h256, frozen 60k trunk, matched-F 10k, local H100)

*Posted 2026-08-09 ~20:0xZ, before launch. Owner-approved 19:59:04Z
("yes to T1") with two amendments: "Use the biggest batch that fits"
(19:59:34Z) and "Maybe do 40k steps" (20:00:08Z).*

***FINAL AMENDMENT 20:08:53Z, before any training step ran:** on
seeing the measured-arithmetic wall-clock estimate (~2.5–3 days at
b96×40k on one GPU), the owner chose "Let's do your original plan" —
the design reverts to the **matched-F rung: 10,000 steps at eff-batch
48** (single-GPU 48×1 vs F's 12×4, same LR schedule, saves every
1,250 = F's cadence), ~10–11 h overnight. Consequences: the fit
ladder shrinks to (48,12) → (48,24); the chained eval is the single
panel_v2 @10000; the primary read is the fully **step- AND
batch-matched Δ_capacity@10k** (the former "secondary"), and the @40k
read is void. Kill lines and bands unchanged; the 90 GPU-h gate
re-prices to 15. Everything below is otherwise as first posted; 40k
references should be read through this amendment.*

*Original header and design follow. Basis: the
[stage-2 attachment decision memo](2026-08-09-molmo2-stage2-attachment-decision.md)
(frozen default stands) + the
[Decoupled Action Expert](../papers/decoupled-action-expert.md)
capacity prior (a 5M MLP head matches a 244M U-Net when task
knowledge lives in the conditioning; our banked one-liner: "F is not
expert-starved"). This rung tests that prior on our stack in the
width direction.*

## Hypothesis

The frozen-trunk flow expert is over-provisioned at h1024/d12
(~367M-param class). If the Decoupled-Action-Expert result transfers,
a width-shrunk expert (h256, ~16× smaller transformer blocks, same
adapters and tap surface) attached to the same frozen 60k trunk
should land within noise of F's panel number — because the task
knowledge lives in the trunk's residual streams, not in the expert.
If instead capacity binds, the tiny expert should trail F by a clear
margin. Either answer re-prices every future expert (rig inference
cost, #16; fjoint sizing, #4).

## Arm

One run: `fontaine_molmo2_flow_tiny_h256_40k_1xh100`, local H100.

- **Identical to the F arm** (`launch_box_fontaine_molmo2_attach_F_10k_ddp4.sh`):
  frozen trunk init from `fontaine_molmo2_ar_60k_ddp4/step_060000`
  (pulled from the box, backbone sha-verified against the upload
  dedup record), `--decoder flow --conditioning-streams residual`
  (structural stride-3 12-tap surface, expert layer i reads tap i),
  bidirectional self-attention, adarms time conditioning, chunk 50,
  same condition fields/dropouts/augments, decoder-lr 1e-4, warmup
  500, weight-decay 1e-5, grad-clip 10, seed 0, eval-seed 0,
  eval-every 500 (256-sample probe), chunked backward. (*Amendment at
  first launch 20:03Z: `--zero1` and `--chunk-grad-allreduce` are
  DDP-only by explicit guard — both dropped for the single-process
  run; gradient/optimizer semantics unchanged, they only shard state
  and replace DDP's reducer sync, neither of which exists at world
  size 1.*)
- **Changed (the contrast):** `--decoder-hidden 256 --decoder-heads 4
  --decoder-intermediate 1024` (F: 1024/8/4096). Cross-heads stay 8
  and adapters are untouched — geometry is pinned by the trunk, so
  the *read surface is identical*; only the expert transformer's
  width shrinks. Trainable-param counts printed at launch banner and
  recorded.
- **Changed (owner amendments):** steps 40,000 (F: 10,000);
  batch = biggest that fits, found by a fit ladder (F: eff-48).
  Saves every 2,500.

## Fit ladder (pre-launch, ~10–20 min, part of this pre-reg)

150-step rungs at (batch, backward-chunks) in order **(96,24) →
(96,48) → (64,16) → (48,12)**; a rung is green iff rc=0 and measured
`vram_alloc_peak_gib ≤ 74.0` (5 GiB headroom under the measured
79.18 OOM line). First green rung launches the 40k run. All-red =
owner steer, no launch. Ladder logs retained.

## Cost + kill lines

- **Wall/cost estimate:** ~3–7 s/step depending on the winning batch
  → 40k ≈ 33–78 h wall ≈ same in GPU-h (single GPU). **Gate: 90
  GPU-h** (babysit `gpu_hours_max`); measured first-jsonl projection
  posted in-channel at first poll — the owner can steer the horizon
  down there (the "maybe" in 40k is honored as a live lever: any
  step-count cut is a plain truncation, reads move to the last save).
- **Kill lines (judged at save boundaries):** NaN/inf loss; probe
  (`eval_chunk_mae`) > 20 sustained ×3 consecutive evals after step
  5,000; vram alloc peak > 77 GiB.
- **Record-only watch:** probe vs F's phase-1 curve (F: 10.2595@5000,
  9.9391@7500, 9.3798@10000) — the tiny arm trailing early is
  expected and is NOT a kill signal (bigger batch, longer horizon,
  cosine tail lands at 40k).

## Frozen reads (after the chained evals; nothing read alone)

The endpoint chains two single-GPU panel_v2 evals in-unit
(`heun30 draws1 stable`, plan sha `2c98c3e1…`, `--report` + npz/json
dumps per the standing rule):

1. **Primary — Δ_capacity@40k:** paired per-frame chunk_mae, tiny@40000
   − F@10000, panel-v2 core frames, 10k-resample bootstrap CI95 (the
   `attach_seam_results.py` read-1 machinery pointed at explicit
   paths). This is capacity-at-its-best vs F-as-shipped —
   **explicitly not step- or batch-matched** (owner amendments);
   quoted with that caveat always attached.
2. **Secondary — Δ_capacity@10k (semi-matched):** same paired read at
   tiny step_010000 vs F step_010000 — step-matched, batch-unmatched.
   The closest controlled comparison this run design allows.
3. **Execution oracle:** state-copy rows from the tiny npz must
   reproduce the banked F-side state-copy values (same plan, same
   frames) — guards silent eval drift.
4. **Record-only:** first_mae mirror, per-step-index curves, probe
   ladder chart, param-count table.

Interpretation bands, pinned pre-data: |Δ@10k| ≤ 0.3 → capacity
prior CONFIRMED at this scale (tiny is a free ~16× shrink);
Δ@10k ≥ +1.0 → capacity binds, h1024 justified; between → gray zone,
the @40k read + curves adjudicate whether width buys convergence
speed vs asymptote.

## What this does NOT test

Depth/tap-count (structurally coupled — a depth rung is a different
surface, T2's business), the joint pole (closed by the memo), and
trunk quality (frozen, shared). A tiny-expert win here licenses a
cheaper fjoint expert but does not by itself re-open KI-joint.
