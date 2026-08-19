# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 23:04–23:1xZ (tick) — **step-1000 drift guard
PASS: probe 5.83@1000, Δ −2.21 vs the ≤ +0.30 band, still improving;
owner 👍 on the boundary-launcher post recorded; disk drop explained
(step-1000 save, 42G).***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1050/3000 at
the 23:05Z poll, loss 0.503 (−0.047 vs 970, new low, falling).
**Step-1000 drift guard (this tick's registered read): PASS** —
eval_chunk_mae 12.85@250 → 8.04@500 → 6.73@750 → **5.83@1000**, Δ
vs the 8.04@500 anchor is **−2.21** against the ≤ +0.30 READ band;
no drift, precedent (pdnorm-mixed rose within band) not even needed.
Rate 15.301 s/step cumulative, window 3.8 steps/min (step-1000
save + probe eval in the interval — wash); 62.21/71 GiB, babysit
exit 0, no gate crossings. ~8.3 h to endpoint → ETA ~07:2x–07:3xZ
08-20.

**Steering**: **NEW — owner 👍×1 on the 20:35Z boundary-launcher
post** (first surfaced this tick's history check; the 22:2x tick saw
none, so it landed after ~22:3xZ): agreement with the R2
one-command-boundary instrument, recorded per the reaction-as-steering
rule, no reply owed (acknowledged in the 23:07Z drift post). Read +
inbox otherwise empty.

**Done**: babysit poll (healthy, exit 0) + drift read PASS. Disk
read: 129G free vs 171G last tick — the Δ is exactly the step-1000
save (42G apparent: 32G optimizer.pt + 10.7G weights, vision
hard-linked); pruner forward math: −42G per save then +32G back at
each optimizer prune → transient floor ~57G free at the step-3000
save — no risk. RAM available 48G, flat third tick running. Posted
drift-read PASS in-channel (id …404337). Queue validate green (depth
2, 15 open). No work-session chain: both queued items GPU-gated
post-onerig, no CPU items, depth at threshold.

**Next**: step-1500 save ~00:2xZ — confirm step_000500/optimizer.pt
pruned (watch item) + disk re-read against the projection; onerig
endpoint ~07:2x–07:3xZ 08-20 → `onerig-endpoint-close` (frozen-grid
sim100 ≥20 / ≤10 / 11–19 bands, anchors demosonly 11 and both
convicted cells 1), then the R2 parity read + relaunch in the freed
window (A5 gate, no GO ask); at the R2 endpoint the boundary is
`./launch_grpo_r2.sh boundary outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-19 22:22–22:3xZ (tick) — **onerig healthy at step
900, loss 0.5558 falling and the rate reads agreeing again; step
1000 lands ~22:48Z so the drift read slips one more tick; fully
quiet tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 900/3000 at
the 22:23Z poll, loss 0.5558 (−0.009 vs 810, falling), window 4.2
steps/min (~14.3 s/step) with trainer-line 15.422 s/step cumulative
— the two reads now agree with the smoke band, last tick's
disagreement was probe-eval wash; 62.21 GiB vs the 71 gate, babysit
exit 0, no gate crossings. ~9.0 h to endpoint → ETA ~07:2x–07:3xZ
08-20. Step 1000 lands ~22:48Z, after this tick's close → the drift
read is the NEXT tick's duty (READ not kill, Δ ≤ +0.30 raw vs the
8.04@500 probe; 6.73@750 makes a breach unlikely).

**Steering**: none — read + inbox empty, history clean.

**Done**: babysit poll (healthy, exit 0). RAM re-read: available 49G
vs 48G last tick — flat, steady state holds. Queue validate green
(depth 2, 15 open); disk 171G free, flat. No work-session chain:
both queued items GPU-gated post-onerig, no CPU items, depth at
threshold.

**Next**: step-1000 drift read next tick (~22:4xZ; step 1000 +
probe eval land ~22:48–22:5xZ, so the read may still be mid-eval at
that poll) + rate re-read; onerig endpoint ~07:2x–07:3xZ 08-20 →
`onerig-endpoint-close`, then the R2 parity read + relaunch in the
freed window (A5 gate, no GO ask); at the R2 endpoint the boundary
is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`. Watch item standing: confirm
step_000500/optimizer.pt pruned after the step-1500 save (~00:2xZ).*

*Updated 2026-08-19 22:01–22:1xZ (tick) — **onerig healthy at step
810; probe 6.73@750 — still improving ahead of the step-1000 drift
read; RAM baseline confirmed flat; fully quiet tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 810/3000 at
the 22:02Z poll, loss 0.5647 (−0.015 vs 740, falling), probe
eval_chunk_mae 12.85@250 → 8.04@500 → **6.73@750** — still improving
one probe ahead of the drift read; window 3.3 steps/min (~18.1
s/step — the step-750 probe eval sits in this interval, same wash as
the step-500 save), trainer-line 15.825 s/step; 62.21 GiB vs the 71
gate, babysit exit 0, no gate crossings. Wall-clock cumulative ~16.2
s/step → ETA ~07:4x–08:0xZ 08-20 (13.5 h total worst case, inside
the 17 GPU-h gate). Step-1000 lands ~22:5xZ at this rate → the drift
read is the NEXT tick's duty (READ not kill, Δ ≤ +0.30 raw vs the
8.04@500 probe; the 6.73@750 read makes a breach unlikely).

**Steering**: none — read + inbox empty, history clean.

**Done**: babysit poll (healthy, exit 0). RAM re-read vs the banked
baseline: available 48G flat vs last tick, trainer RSS 145.77M KB —
the SAME number as the 145.9G banked read (that stamp was decimal GB
≈ 139 GiB), i.e. flat, no leak. Queue validate green (depth 2, 15
open); disk 171G free, flat. No work-session chain: both queued
items GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-1000 drift read next tick (~22:2x–22:4xZ tick window;
step 1000 lands ~22:5xZ, so it may slip one more tick) + rate
re-read; onerig endpoint ~07:4x–08:0xZ 08-20 →
`onerig-endpoint-close`, then the R2 parity read + relaunch in the
freed window (A5 gate, no GO ask); at the R2 endpoint the boundary
is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`. Watch item standing: confirm
step_000500/optimizer.pt pruned after the step-1500 save (~00:5xZ).*

## Utilization footer

Session 2026-08-19 23:04–23:1xZ (tick; `onerig` riding, ~4.7 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step
1050/3000, loss 0.503 new low (−0.047 interval); step-1000 drift
guard PASS (probe 5.83@1000 vs anchor 8.04@500, Δ −2.21 vs the
≤+0.30 READ band, curve still improving) — posted in-channel; rate
15.301 s/step cumulative, 62.21 GiB, no gate crossings, ETA
~07:2x–07:3xZ 08-20; disk 129G free vs 171 last tick = exactly the
step-1000 save (42G: 32G optimizer.pt + weights), pruner math floors
at ~57G free at the step-3000 save — no risk; NEW owner 👍 on the
20:35Z boundary-launcher post recorded (landed after the 22:2x
tick); RAM flat (available 48G); no chain (both queued items
GPU-gated post-onerig, no CPU items)** — queue green depth 2 (15
open).

Session 2026-08-19 22:22–22:3xZ (tick; `onerig` riding, ~4.0 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 900/3000,
loss 0.5558 falling (−0.009 interval), window 4.2 steps/min (~14.3
s/step) with trainer-line 15.422 s/step cumulative — the rate reads
agree again (last tick's disagreement was probe-eval wash), 62.21
GiB, no gate crossings, ETA ~07:2x–07:3xZ 08-20; step 1000 lands
~22:48Z after this close → drift read slips to next tick (6.73@750
makes a ≤+0.30 breach unlikely); RAM re-read flat (available 49G vs
48G); Discord fully quiet (read + inbox empty, no new reactions); no
chain (both queued items GPU-gated post-onerig, no CPU items)** —
queue green depth 2 (15 open). Disk 171G free (94%), flat.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
