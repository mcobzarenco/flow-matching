# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 21:40–21:4xZ (tick) — **onerig healthy at step
740; the launch-to-now RAM available drop (91→48G) slope-checked —
steady state, not a leak; fully quiet tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 740/3000 at
the 21:41Z poll, loss 0.5801 (+0.02 vs 650 — noise on a falling
trend: 0.64@570 → 0.56@650 → 0.58@740), window 4.3 steps/min (~13.9
s/step, fastest window yet; the trainer-line 16.661 s/step read
disagrees with the wall clock, so the window governs — re-read next
tick); 62.21 GiB vs the 71 gate, babysit exit 0, no gate crossings.
ETA ~07:0x–07:1xZ 08-20. Step-1000 lands ~22:4xZ → the drift read is
the NEXT tick's duty (READ not kill, Δ ≤ +0.30 raw vs the 8.04@500
probe).

**Steering**: none — read + inbox empty, history clean.

**Done**: babysit poll (healthy, exit 0). RAM read: available 48–50G
vs 91G at the 18:29Z first poll → 4-min slope check showed
MemAvailable RISING (49.96 → 50.84G) — loader/cache steady state,
not an OOM trajectory; trainer RSS baseline 145.9G banked at 21:46Z
for next-tick comparison. Queue validate green (depth 2, 15 open);
disk 171G free, flat. No work-session chain: both queued items
GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-1000 drift read ~22:4xZ (tick) + rate re-read + RAM
re-read vs the 145.9G RSS baseline; onerig endpoint ~07:0x–07:1xZ
08-20 → `onerig-endpoint-close`, then the R2 parity read + relaunch
in the freed window (A5 gate, no GO ask); at the R2 endpoint the
boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`. Watch item standing: confirm
step_000500/optimizer.pt pruned after the step-1500 save (~00:4xZ).*

*Updated 2026-08-19 21:19–21:2xZ (tick) — **onerig healthy at step
650, loss through 0.56 and rate holding under band; fully quiet
tick, fast close. Drift read is next tick's duty (~22:4xZ).***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 650/3000 at
the 21:20Z poll, loss 0.5602 (falling, −0.081 over the interval),
14.851 s/step cumulative / 3.9 steps/min over the last window; 62.21
GiB vs the 71 gate, babysit exit 0, no gate crossings. ~9.7 h to
endpoint → ETA ~07:0xZ 08-20. Step-1000 lands ~22:4x–22:5xZ → the
drift read is the NEXT tick's duty (READ not kill, Δ ≤ +0.30 raw vs
the 8.04@500 probe).

**Steering**: none — read + inbox empty, history clean (the 👍 on the
20:35Z post was recorded two ticks ago, nothing new).

**Done**: babysit poll (healthy, exit 0); queue validate green (depth
2, 15 open); disk 171G free, flat — on the priced trajectory. No
work-session chain: both queued items GPU-gated post-onerig, no CPU
items, depth at threshold.

**Next**: step-1000 drift read ~22:4xZ (tick), onerig endpoint
~07:0xZ 08-20 → `onerig-endpoint-close`, then the R2 parity read +
relaunch in the freed window (A5 gate, no GO ask); at the R2 endpoint
the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`. Watch item standing: confirm
step_000500/optimizer.pt pruned after the step-1500 save (~00:4xZ).*

*Updated 2026-08-19 20:58–21:0xZ (tick) — **onerig healthy at step
570, rate back under band (14.66 s/step cumulative); fully quiet
tick, fast close.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 570/3000 at
the 20:59Z poll, loss 0.6413 (falling), 14.657 s/step cumulative —
now *under* the 15.1–15.4 band (the slow interval read, 3.4
steps/min, is the step-500 save stall washing through the window);
62.21 GiB vs the 71 gate, babysit exit 0, no gate crossings. ~9.9 h
to endpoint → ETA ~06:5x–07:0xZ 08-20 (back inside the registered
window). Step-1000 drift read ~22:4x–23:0xZ tonight (tick duty, READ
not kill, Δ ≤ +0.30 raw vs the 8.04@500 read).

**Steering**: none — read + inbox empty, history clean (the 👍 on
the 20:35Z post was recorded last tick, nothing new).

**Done**: babysit poll (healthy, exit 0); queue validate green (depth
2, 15 open); disk 171G free, flat since the step-500 save — matches
the priced trajectory. No work-session chain: both queued items
GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-1000 drift read ~22:4xZ (tick), onerig endpoint
~07:0xZ 08-20 → `onerig-endpoint-close`, then the R2 parity read +
relaunch in the freed window (A5 gate, no GO ask); at the R2 endpoint
the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`. Watch item standing: confirm
step_000500/optimizer.pt pruned after the step-1500 save (~00:4xZ).*

## Utilization footer

Session 2026-08-19 21:40–21:4xZ (tick; `onerig` riding, ~3.3 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 740/3000,
loss 0.5801 (+0.02 noise blip on a falling trend), window 4.3
steps/min (~13.9 s/step, fastest yet; the trainer-line 16.661 s/step
read disagrees with the wall clock — window governs, re-read next
tick), 62.21 GiB, no gate crossings, ETA ~07:0x–07:1xZ 08-20; RAM
available-drop since launch (91→48G) slope-checked: MemAvailable
RISING over 4 min — steady state not a leak, trainer RSS baseline
145.9G banked for next tick; step-1000 drift read next tick ~22:4xZ;
Discord fully quiet (read + inbox empty, no new reactions); no chain
(both queued items GPU-gated post-onerig, no CPU items)** — queue
green depth 2 (15 open). Disk 171G free (94%), flat.

Session 2026-08-19 21:19–21:2xZ (tick; `onerig` riding, ~3.0 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 650/3000,
loss 0.5602 falling (−0.081 interval), 14.851 s/step cumulative /
3.9 steps/min window, 62.21 GiB, no gate crossings, ETA ~07:0xZ
08-20; step-1000 drift read is next tick's duty (~22:4xZ); Discord
fully quiet (read + inbox empty, no new reactions); no chain (both
queued items GPU-gated post-onerig, no CPU items)** — queue green
depth 2 (15 open). Disk 171G free (94%), flat — on the priced
trajectory.

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
