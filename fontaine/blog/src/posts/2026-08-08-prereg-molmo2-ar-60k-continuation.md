# Pre-registration: molmo2 AR +20k continuation (40k → 60k)

*2026-08-08 ~10:2xZ. Immutable once posted. Owner-steered: proposed
08:49Z ("train for longer: --resume the 4×DDP run for an additional
20k steps, --rewarmup-steps 1000 + new seed for fresh data shuffle"),
design discussed 09:00Z, **GO + prioritized over the attach screen
09:04Z** ("let's prio the 60k molmo2 run as you described it").
Parent run: [molmo2 AR 40k](2026-08-06-prereg-molmo2-ar-40k.md),
endpoint results [BEATS](2026-08-08-molmo2-endpoint-results.md).*

## Question

The 40k endpoint reads **6.0079/2.1871** greedy — far ahead of the
matched-steps AR baseline (A-s0 7.7966, paired −1.717) but **+0.205
behind AR-100k (5.8026)**, which trained 2.5× the steps. The probe
curve says the last third of the 40k run bought nothing (low
**5.91 @ 26,500**, endpoint **6.2075 @ 40,000** — a cosine-floor
tail). Does re-warm + re-decay for +20k steps on fresh-shuffled data
close the gap to (or pass) the AR-100k bar?

## Data arithmetic (banked before launch)

The corpus is 18,636,749 frames (the 40k run's E1 banner). 40k steps
× eff-48 = 1.92M samples ≈ **10% of one epoch** — the continuation's
2.88M cumulative samples still sit well under one epoch, so
additional steps buy genuinely unseen data, and the **new shuffle
seed** (owner rule, banked 2026-08-08; mechanized by
`check_resume_seed`, which hard-aborts a resume reusing the
checkpoint's seed) draws a fresh 10% rather than replaying the
consumed order.

## Design (frozen)

One run, box 4×DDP, launcher
`launch_box_fontaine_molmo2_ar_60k_resume_ddp4.sh`:

- `--resume outputs/train/fontaine_molmo2_ar_40k_ddp4/step_040000`
  (weights + optimizer + step; consolidated ZeRO-1 save, on-box),
- `--steps 60000` (total, so +20,000), `--rewarmup-steps 1000`,
- `--seed 1` (checkpoint trained on seed 0),
- **every other flag byte-identical to the 40k launcher** (B12/rank
  ×4 = eff-48, zero1 + 6-chunk backward + chunk-grad-allreduce,
  decoder-lr 1e-4 / text-lr 2e-5, same aux/condition dropout,
  save-every 2500 — async saves default-on),
- new run name/save-dir/wandb: `fontaine_molmo2_ar_60k_ddp4`.

**LR path, stated exactly:** the cosine recomputes over 60k total —
at step 40,000 the multiplier is 0.332 (decoder 3.32e-5), reached
through the 1,000-step rewarmup ramp, then decays to the 10% floor
(1e-5) at 60k. Re-warm + re-decay, the continued-pretraining recipe
`lr_lambda`'s docstring names.

- **E1 gate (unchanged):** dataset banner must read 878 datasets /
  38,571 episodes / 18,636,749 frames / dims 6/6 — any deviation
  aborts before step 1. The resume banner must show
  `resumed optimizer/scheduler at step 40000` and the
  `check_resume_seed` line.
- **K1 kill lines:** NaN/inf loss; probe (`eval_chunk_mae`) worse
  than **8.2075** (endpoint 6.2075 + 2.0) sustained ×3 evals any
  time after step 41,500 (rewarmup settled) — judged at save
  boundaries; vram gate 71 GiB (the 40k run's envelope).
- **Cost:** ~2.2 s/step × 20k ≈ 12.2 h wall ≈ **49 GPU-h; ceiling
  60 GPU-h**. Chained endpoint eval at `step_060000` rides the same
  launcher (greedy panel, dumps retained).
- First babysit validates the first async-save lines + util/rate per
  standing rule; babysit entry live at launch.

## Frozen reads (at the 60k endpoint)

Panel = `plans/holdout_curated_v0_k4l2.json`, greedy, 4-GPU sharded,
stem `eval__fontaine_molmo2_ar_60k_ddp4__step_060000__panel_curated_v0_k4l2`.

1. **Primary: paired per-frame Δ vs the banked 40k endpoint npz**
   (identical plan/rows), seeded bootstrap CI95 (seed 0, 10,000
   resamples), core rows. IMPROVED if CI95 entirely below 0; PARITY
   if CI spans 0; DAMAGED if entirely above. This is the "was +20k
   worth it" number.
2. **The owner bar:** pooled chunk/first vs AR-100k **5.8026 /
   2.1431** (cross-trunk, unpaired — quoted with that caveat).
   PASSES-100k if pooled chunk < 5.8026.
3. State-copy rows byte-match the banked panel values (instrument
   integrity).
4. Probe trajectory read (record): does the rewarmed segment set a
   new probe low vs 5.91@26.5k?
5. Decision, frozen: IMPROVED + PASSES-100k ⇒ the 60k endpoint
   replaces the 40k endpoint as the phase-2 flow-trunk candidate and
   the attach screen (queued behind this run) warm-starts from
   step_060000 (attach pre-reg gets a checkpoint-repoint amendment +
   K-smoke re-run before any arm launches). IMPROVED only ⇒ same
   repoint, bar noted honestly. PARITY/DAMAGED ⇒ 40k endpoint
   stands; the continuation result is banked as the "longer
   training" answer at this scale and the attach screen proceeds
   from step_040000 unchanged.

## Numbered expectations (banked before data)

1. No kill line fires; the run completes 20k steps — confidence
   high.
2. The probe sets a new low below 5.91 during the rewarmed segment —
   confidence medium.
3. Read 1 IMPROVED (CI below 0) — confidence medium-high (fresh
   data + restored LR on a curve that plateaued at the floor).
4. Read 2 PASSES-100k (chunk < 5.8026) — confidence **open**; this
   is the owner's question and the honest answer is we don't know:
   the remaining gap is 0.205 and the 40k→100k segment bought
   AR-100k its lead at 3× this continuation's length.

## Scheduling

Box is free now (both #19 obligations and the microbench closed);
launch immediately after this post via the box `run_detached.sh`
(unit `fontaine-molmo2-60k`). The #4 attach chain (K smoke ladder →
screen) queues strictly behind this run per the owner's 09:04Z
priority call; vu5k stays behind attach.
