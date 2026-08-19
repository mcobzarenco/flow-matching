# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-19 20:38–20:4xZ (tick) — **onerig healthy through the
step-500 save (probe 12.85→8.04); owner 👍 on the boundary-launcher
post; disk trajectory priced — the in-trainer pruner keeps the
endpoint reachable (~47G worst-case transient floor).***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 500/3000 at
the 20:39Z poll — first save boundary landed (step_000500, 44G) and
the probe improved eval_chunk_mae 12.85@250 → 8.04@500; 4.1 steps/min
over the last window (~14.6 s/step — back inside the 15.1–15.4 band),
65.1 GiB vs the 71 gate, babysit exit 0. Endpoint ETA ~07:0x–07:4xZ
08-20. Step-1000 drift read ~22:4x–23:0xZ tonight (tick duty, READ
not kill, Δ ≤ +0.30 raw vs the 8.04@500 read).

**Steering**: owner 👍 on the 20:35Z boundary-launcher post (surfaced
by the history check — agreement, recorded, no reply owed). Read +
inbox otherwise empty.

**Done**: babysit poll (healthy, exit 0); queue validate green (depth
2, 15 open); disk priced after the 45G drop at the step-500 save —
171G free, one full save is 44G (32G of it optimizer.pt), and the
live argv carries `--prune-superseded-optim` (in-trainer promotion
CLOSED 04:4xZ, keeps latest 2 full saves): worst-case transient
bottoms at ~47G free at the step-3000 save, endpoint reachable with
margin. Watch item for a later tick: confirm step_000500/optimizer.pt
is gone after the step-1500 save (~00:4xZ — this run's first
in-trainer pruning event). No work-session chain: both queued items
are GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-1000 drift read ~22:4xZ (tick), onerig endpoint ~07:xZ
08-20 → `onerig-endpoint-close`, then the R2 parity read + relaunch
in the freed window (A5 gate, no GO ask); at the R2 endpoint the
boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-19 21:19–21:2xZ (tick; `onerig` riding, ~3.0 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 650/3000,
loss 0.5602 falling (−0.081 interval), 14.851 s/step cumulative /
3.9 steps/min window, 62.21 GiB, no gate crossings, ETA ~07:0xZ
08-20; step-1000 drift read is next tick's duty (~22:4xZ); Discord
fully quiet (read + inbox empty, no new reactions); no chain (both
queued items GPU-gated post-onerig, no CPU items)** — queue green
depth 2 (15 open). Disk 171G free (94%), flat — on the priced
trajectory.

Session 2026-08-19 20:58–21:0xZ (tick; `onerig` riding, ~2.6 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 570/3000,
loss 0.6413 falling, 14.657 s/step cumulative (under band; the slow
interval read is the step-500 save stall), 62.21 GiB, no gate
crossings, ETA back to ~06:5x–07:0xZ 08-20; Discord fully quiet (read
+ inbox empty, no new reactions); no chain (both queued items
GPU-gated post-onerig, no CPU items)** — queue green depth 2 (15
open). Disk 171G free (94%), flat since the save — on the priced
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
