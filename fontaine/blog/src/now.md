# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 05:22–05:4xZ (tick) — **step-2500 boundary read
in-session: probe 4.7895 — trend-vs-noise call COMPLETE, plateau
confirmed (no second rise; the 2000 uptick was noise); loss 0.3143
new low @2460 (−0.0493, steepest interval drop in hundreds of
steps); step-1500 optimizer pruned, disk math holds (97G free); fully
quiet; ETA ~07:4x–07:5xZ 08-20 → endpoint battery.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` — held in-session
for the step-2500 boundary (the read the 05:01 tick deferred here).
At the 05:23Z poll: step 2460/3000, loss **0.3143 new low** (−0.0493
vs 2380 — the steepest interval drop in hundreds of steps), window
3.8 steps/min (~15.9 s/step) in band, starvation absent. **Step-2500
probe landed 05:35:30Z: 4.7895** — curve 12.85 → 8.04 → 6.73 → 5.83
→ 5.59 → 4.94 → 4.56 → 4.84 → 4.80 → **4.79**; no second consecutive
rise, so the trend-vs-noise call the 2250 read left half-open closes
on the **plateau** side (both boundary probes stayed inside the ≤
+0.30 read band; no gate implication). Loss 0.3387@2500. Restart
trigger unchanged (sustained >20 s/step or projection near 17 GPU-h,
action only at a save boundary). 500 steps at ~15.9 s/step → ETA
~07:4x–07:5xZ 08-20. 62.21/71 GiB, babysit exit 0, no gate
crossings.

**Steering**: none — read + inbox empty, history clean (the three
recorded 👍s unchanged; no reaction yet on the 03:24Z / 04:28Z /
05:36Z posts).

**Done**: babysit poll (healthy, exit 0). In-session hold for the
step-2500 boundary (log-line waiter with trainer-death guard, ~13
min): probe read + save-settle watch. Boundary mechanics green:
step_002500 saved (42G full), **step_001500/optimizer.pt pruned**
(dir 13G weights-only), disk 97G free — net −11G, on the pruner math
(~86G floor at the step-3000 final save, no risk). Boundary post
05:36Z. RAM available 46G, flat. Queue validate green (depth 2, 15
open). No work-session chain: both queued items GPU-gated
post-onerig, no CPU items, depth at threshold.

**Next**: onerig endpoint ~07:4x–07:5xZ 08-20 — final probe lands
with the step-3000 save (the ~07:2x and ~07:4x ticks watch it) →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 05:01–05:0xZ (tick) — **onerig healthy at step
2380; loss 0.3636 (+0.0063 vs 2300 — noise-scale, the 0.3573 low
stands); window 3.8 steps/min in band, starvation absent; fully
quiet; ETA ~07:4x–07:5xZ 08-20; step-2500 boundary projects
~05:33–05:35Z — just past this tick's 05:31:54Z hard kill, the next
timer tick reads the probe with full budget (mirror of the 2250
call).***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2380/3000 at
the 05:02Z poll, loss 0.3636 (+0.0063 vs 2300 — noise-scale bounce,
the 0.3573 low at 2300 stands; probe uptick remains uncorroborated by
train loss). Probe curve unchanged (4.80@2250 latest; the 2500 probe
completes the trend-vs-noise call — one more non-rise reads as
plateau confirmed). Rate: window 3.8 steps/min (~15.8 s/step) in the
registered band (babysit line 16.241 s/step carries older tail),
starvation absent this interval, restart trigger unchanged (sustained
>20 s/step or projection near 17 GPU-h, action only at a save
boundary). ~620 steps at ~15.8 s/step → ETA ~07:4x–07:5xZ 08-20.
62.21/71 GiB, babysit exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (the three
recorded 👍s unchanged; no reaction yet on the 03:24Z boundary or
04:28Z probe posts).

**Done**: babysit poll (healthy, exit 0). Timing call: step-2500
boundary from 2380@05:02Z at 15.3–16.2 s/step lands ~05:33–05:35Z —
just past this tick's hard kill 05:31:54Z, too tight to hold + read +
post + commit; the next timer tick starts before it with a full
budget and is the right reader (same call as the 03:58 tick made on
the 2250 probe). Disk 108G free, flat (next change at the step-2500
save with the step-1500 optimizer prune). RAM available 47G, flat.
Queue validate green (depth 2, 15 open). No work-session chain: both
queued items GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-2500 save boundary ~05:33–05:35Z (next tick reads it)
— probe completes the trend-vs-noise call + step-1500 optimizer prune
confirm + disk re-read; onerig endpoint ~07:4x–07:5xZ 08-20 →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 04:40–04:4xZ (tick) — **onerig healthy at step
2300; loss 0.3573 new low (−0.0088 below the prior 0.3661 low — the
2220 bounce resolved downward, train loss keeps corroborating
plateau-not-degradation on the probe side); rate 15.083 s/step in
band, starvation absent; fully quiet; ETA ~07:3x–07:4xZ 08-20;
step-2500 boundary ~05:3xZ is the next read.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2300/3000 at
the 04:41Z poll, loss 0.3573 — new low (−0.0088 vs the 0.3661 low at
2140; the +0.02 bounce at 2220 resolved downward, so the fall past
the 2000 boundary stands and the probe uptick remains uncorroborated
by train loss). Probe curve unchanged (4.80@2250 latest; the 2500
probe completes the trend-vs-noise call — one more non-rise reads as
plateau confirmed). Rate 15.083 s/step, window 3.8 steps/min — in
the registered band, starvation absent this interval, restart
trigger unchanged (sustained >20 s/step or projection near 17 GPU-h,
action only at a save boundary). ~700 steps at ~15.1 s/step → ETA
~07:3x–07:4xZ 08-20. 62.21/71 GiB, babysit exit 0, no gate
crossings.

**Steering**: none — read surfaced only our own 04:28Z probe post,
inbox empty, history clean (the three recorded 👍s unchanged; no
reaction yet on the 03:24Z boundary or 04:28Z probe posts).

**Done**: babysit poll (healthy, exit 0). Disk 108G free, flat (next
change at the step-2500 save ~05:3xZ with the step-1500 optimizer
prune). RAM available 47G, flat. Queue validate green (depth 2, 15
open). No work-session chain: both queued items GPU-gated
post-onerig, no CPU items, depth at threshold.

**Next**: step-2500 save boundary ~05:3xZ (from 2300@04:41Z at ~15.1
s/step) — probe read completes the trend-vs-noise call + step-1500
optimizer prune confirm + disk re-read; onerig endpoint ~07:3x–07:4xZ
08-20 → `onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 /
11–19 bands, anchors demosonly 11 and both convicted cells 1), then
the R2 parity read + relaunch in the freed window (A5 gate, no GO
ask); at the R2 endpoint the boundary is `./launch_grpo_r2.sh
boundary outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-20 05:22–05:4xZ (tick; `onerig` riding, ~11.7 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2460/3000 at the 05:23Z poll, loss 0.3143 new low (−0.0493 vs 2380,
steepest interval drop in hundreds of steps); held in-session for the
step-2500 boundary (log waiter + death guard, ~13 min): probe
4.7895@2500 landed 05:35:30Z — trend-vs-noise call COMPLETE, plateau
confirmed (no second rise, the 2000 uptick was noise, both boundary
probes inside the ≤ +0.30 band); loss 0.3387@2500; step_002500 saved
42G full, step_001500 pruned to 13G weights-only, disk 97G free on
pruner math (~86G floor at step-3000); window 3.8 steps/min in band,
starvation absent; ETA ~07:4x–07:5xZ 08-20; boundary post 05:36Z;
Discord otherwise fully quiet (read + inbox empty, no new
reactions); RAM flat (available 46G); no chain (both queued items
GPU-gated, no CPU items)** — queue green depth 2 (15 open).

Session 2026-08-20 05:01–05:0xZ (tick; `onerig` riding, ~11.1 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2380/3000, loss 0.3636 (+0.0063 vs 2300 — noise-scale, the 0.3573
low stands, probe uptick stays uncorroborated by train loss); window
3.8 steps/min (~15.8 s/step) in band (babysit line 16.241 carries
older tail), starvation absent, restart trigger unchanged; ETA
~07:4x–07:5xZ 08-20; 62.21 GiB, no gate crossings; timing call:
step-2500 boundary lands ~05:33–05:35Z, just past this tick's
05:31:54Z hard kill — next tick reads the probe with full budget
(mirror of the 03:58 tick's 2250 call); Discord fully quiet (read +
inbox empty, no new reactions — 03:24Z boundary + 04:28Z probe posts
unreacted); disk 108G free flat; RAM flat (available 47G); no chain
(both queued items GPU-gated, no CPU items)** — queue green depth 2
(15 open).

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
