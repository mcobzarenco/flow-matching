# Utilization window roll — 2026-08-19 (receipts)

Queue item `util-window-roll`, work session 08:08–1x:xxZ 08-19. Rolls
the trailing-7-day footer forward from the 08-17 rebase (window
2026-08-10 00:00Z → 2026-08-17 19:45Z; receipts
`fontaine/notes/utilization-rebase-2026-08-17.md`). Method identical:
per-run figures from babysit registry prune records + archive session
notes; instrument `fontaine/scripts/util_ledger_extract.py` (page glob
widened this session to cover all `now-2026-08-*` archives).

**New window**: 2026-08-12 00:00Z → 2026-08-19 08:45Z. Local-only per
the queue item — the box was killed by the owner 08-17 ~15:xxZ; its
line retires from the footer (final full-history table stays in the
08-17 rebase note). For the record, ~106 box GPU-h fall inside this
window (every box row of the 08-17 table except er_60k, which
completed 12:36Z 08-11): demo-gen v1c 16.9, sft_v1 launch-1 3.7,
run-1b 14.7, run-2 31.0, sim100 eval ~5 (estimated), demo_gen_v2
17.8, v2_joint 2.6, demosonly 4.0, rigonly 10.5.

## Local H100 (~85.5 GPU-h total in-window)

Start from the 08-17 table's local total ~80.2, drop rows that
completed before 2026-08-12 00:00Z, add accruals after the 08-17
19:45Z stamp.

**Dropped (pre-08-12, ~22.75)**: tiny10k in-window part 5.4 (08-10),
er15k panel 2.5, molmoact2_sweep 1.3 (complete 14:23Z 08-10), rig_ft
preflight+r1 2.8 (complete 20:27Z 08-10), er35k aux+panel 2.2 (08-10
→ 00:41Z 08-11), port parity reads 0.7 (08-11), molmoact2_ae_ours 1.9
(complete 06:56Z 08-11), er55k_panel 2.2 (complete 12:00Z 08-11),
er60k_events_dump 1.55 (complete 15:5xZ 08-11), sim100b pre-window
part ~2.2 of 5.5 (phase 1 ~2.0 + phase-2 head ran pre-midnight 08-11;
phase 2 completed 03:16Z 08-12 — ~3.3 stays in-window).

**Retained (08-12 → 08-17 19:45Z stamp)**: 80.2 − 22.75 ≈ **57.45**
(every other row of the 08-17 local table, from the 08-12 sim
foreground probes through the discriminator's first ~1.0).

**Added (post-stamp → 08-19 08:45Z, ~28.1)**:

| run / spend | GPU-h | note |
|---|---|---|
| discriminator roll-in | 4.8 | total ~5.8 across both attempts (attempt-1 OOM ~1.25 + attempt-2 20:20:55Z → 00:42Z 08-18); ~1.0 already counted at the 08-17 stamp |
| stack-parity probe evals (08-18 00:49) | 0.1 | |
| sim100 baseline + k4l2 + report legs (08-18 02:04) | 2.8 | demosonly control 11/100 read |
| released-ckpt panel leg (08-18 07:53) | 0.45 | |
| pdnorm smoke + train head (08-18 10:37) | 2.3 | smoke ~0.1 + train 11:02→13:15Z |
| re-gate embeds (08-18 13:44) | 0.02 | |
| pdnorm train remainder | 10.7 | train total ~12.9 (13:15Z 08-18 → step 3000 complete ~00:42Z 08-19), minus the 2.2 above |
| pdnorm endpoint battery | 3.0 | sim100 leg ~2.55 + k4l2 panel tail ~0.45; CONVICT verdict battery |
| joint-probe leg 3 (token-unseen) | 2.4 | clean full run 04:11:18Z → 06:33Z 08-19 |
| joint-probe leg 4 (token-base, LIVE at stamp) | ~1.5 | relaunched 06:54:56Z, accruing at the ~1 GPU-h/wall-h class |

**Total ≈ 57.45 + 28.1 ≈ 85.5**; experiments ≈ **84.1** (ops/loss:
discriminator attempt-1 OOM burn ~1.25, smoke fractions ~0.15).
Precision class local ±2 (same as the 08-17 rebase; the one soft
edge is the sim100b midnight pro-rate ±0.3).

pdnorm screen-wide check (the queue item's named figure): train ~12.9
+ battery ~3.0 ≈ **~15.9** ✓ matches the item.
