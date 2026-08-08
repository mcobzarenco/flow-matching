# Golden tickets are real: one searched noise vector ≈ 75% of the mean-of-10 gain

*2026-08-08 05:2xZ. Stages 1–2 of the golden-ticket noise screen
([pre-reg](2026-08-07-prereg-golden-ticket-screen.md), #1), reads
frozen there and executed by `ticket_scores.py` (stage 1) and
`goldenticket_stage2_results.py` (stage 2, oracle-green before its
data). Stage 3 (mean-of-top-10, record-only R3) launched 05:16Z,
lands ~08:1xZ. Model: flow teacher @80k, Heun-30, k4l2 panel.*

## The question

Noise-draw ensembling (mean-of-10 random draws) buys the flow
family −1.24 chunk MAE. The golden-ticket hypothesis (the
test-time-scaling literature's "lucky noise" claim, transplanted to
our stack): is there a **fixed** noise vector — one ticket, reused at
every frame — that captures part of that gain at single-draw cost?

## Stage 1 — R1: CONFIRM, loudly

One batched draws-64 eval on the 2,458-frame probe, where draw *m*
IS ticket *m* (64 candidates from a sha-pinned N(0,I) bank), scored
per-ticket by pooled core chunk MAE:

- **sd of the 64 ticket scores = 0.82252** vs kill line 0.0785 (the
  upper 95% χ²₆₃ edge of the banked null σ_probe 0.0669) — **12×
  the null**. Smooth spread 5.71–9.37, no outlier artifact: tickets
  differ enormously, in both directions.
- **min = 5.70564** vs line 6.52401 (expected null min − 2 sd).
- Winner: **ticket 33** (5.7056/1.8701 on probe rows —
  selection-biased by construction; stage 2 exists because of that).

## Stage 2 — R2: REAL, bigger on unseen rows than on the rows that picked it

Winner ticket, full 25,800-frame panel, judged ONLY on the 14,746
complement core rows (panel core minus every probe frame-identity
triple):

- **Paired Δ (ticket 33 − banked stable-key single draw) = −0.924
  [CI95 −0.985, −0.866]** vs the REAL line −0.05. The
  selection-biased probe-row delta was −0.819 — the effect is
  *larger* where the ticket was never evaluated during selection.
  This is not probe-row luck surviving; it is a property of the
  ticket.
- Core-pooled, board convention: **5.6468 / 1.8963** — a single
  Heun-30 draw that lands within 0.005 of AR-100k's draws-10 row
  and captures **~75% of the mean-of-10 gain (−0.924 of −1.235) at
  1/10th the draws**.

**The mechanism is directional, not norm.** The obvious deflationary
story — a small-norm ticket mimicking the mean's noise-shrinkage —
is dead on the data: ticket 33's norm ranks 29/64 (17.15 vs bank
mean 17.26), and corr(norm, ticket score) = −0.05 across the bank.
Specific *directions* in the 300-dim noise space are systematically
better across thousands of held-out frames. That is the LAFM/DSRL
premise (structured noise carries mode information) showing up in
our own decoder, unprompted.

## What executes next (all pre-registered)

- **Stage 3 (running)**: mean of the top-10 tickets, draws-10 ticket
  noise — does searched noise beat random noise *inside* the
  ensembling regime? R3 = pooled Δ vs the banked mean-of-10 5.3645,
  tie band ±0.02, **record-only either way** (mean-of-10's row is
  not displaced without a paired follow-up). Screen budget after
  stage 3 ≈ 5.5 of the 6 GPU-h gate.
- **Leaderboard**: R2 REAL earns the ticket row (keying stated,
  sha-pinned): teacher single-draw ticket-33 at 5.6468/1.8963.
- **R4 record-only reads** (per-dataset argmin disagreement — the
  task-locality/LAFM question — plus dispersion-quartile geometry
  and horizon profile) come with the stage-3 write-up; the stage-1
  dumps already carry them.
- The noise-structure ladder above the screen (per-dataset tickets →
  LAFM learned priors → DSRL-style state-conditioned noise) was
  pre-mapped on the
  [noise-space-steering pages](../papers/noise-space-steering.md);
  R1+R2 landing this hard is the ladder's entry condition. Each rung
  needs its own pre-reg.

## Cost

Stage 1 ~1.7 GPU-h, stage 2 ~0.85, stage 3 ~2.9 projected — ~5.5 of
the pre-registered 6 GPU-h, one quiet local-GPU day, zero training.
