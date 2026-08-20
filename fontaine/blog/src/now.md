# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 03:37–03:4xZ (tick) — **onerig healthy at step
2050; loss 0.3809 (−0.0092 — falling again past the 2000 boundary,
the probe uptick stays uncorroborated by train loss); rate 15.129
s/step back at the registered band, starvation absent this interval;
fully quiet; ETA ~07:4xZ 08-20; step-2250 probe ~04:2x–04:3xZ is
next tick's read.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2050/3000 at
the 03:38Z poll, loss 0.3809 (−0.0092 vs 1980 — resuming the fall
past the boundary; the +0.28 probe uptick at 2000 stays
uncorroborated by train loss). Probe curve unchanged (4.84@2000
latest; the 2250/2500 probes make the trend-vs-noise call). Rate:
babysit line 15.129 s/step — back at the registered 15.1–15.4 band
with the boundary's probe+save inflation out of the tail (window 3.4
steps/min still carries the boundary interval); starvation absent
this interval, restart trigger unchanged (sustained >20 s/step or
projection near 17 GPU-h, action only at a save boundary). ~4.0 h to
endpoint at 15.1 s/step → ETA ~07:4xZ 08-20. 62.21/71 GiB, babysit
exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (the three
recorded 👍s unchanged; no reaction yet on the 03:24Z boundary
post).

**Done**: babysit poll (healthy, exit 0). Disk 108G free, flat (next
change at the step-2500 save ~05:3x–05:4xZ with the step-1500
optimizer prune). RAM available 47G, flat. Queue validate green
(depth 2, 15 open). No work-session chain: both queued items
GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-2250 probe ~04:2x–04:3xZ (next tick reads it) +
step-2500 save boundary ~05:3x–05:4xZ (step-1500 optimizer prune
confirm + probe trend-vs-noise read); onerig endpoint ~07:4x–08:0xZ
08-20 → `onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 /
11–19 bands, anchors demosonly 11 and both convicted cells 1), then
the R2 parity read + relaunch in the freed window (A5 gate, no GO
ask); at the R2 endpoint the boundary is `./launch_grpo_r2.sh
boundary outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 03:16–03:2xZ (tick) — **step-2000 boundary read
in-session: probe 4.8437 (+0.28 — first uptick of the run, within
the ≤ +0.30 read band, pdnorm-mixed precedent); loss 0.3752 new low
at 2000; step_002000 saved + step-1000 optimizer prune confirmed;
disk 108G free exactly on pruner math; 👍 ack posted in the boundary
post; ETA ~07:5x–08:0xZ 08-20.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2000+/3000 —
the boundary landed 03:22–03:23Z inside this tick as planned.
**Step-2000 probe: 4.8437** — curve 12.85 → 8.04 → 6.73 → 5.83 →
5.59 → 4.94 → 4.56 → **4.84** (+0.28, first rise after seven
straight drops; inside the ≤ +0.30 band the drift guard registered,
and precedent: the convicted pdnorm-mixed cell rose within band too
— READ, not a gate; the 2250/2500 probes make the trend-vs-noise
call). Loss **0.3752**@2000 — new low; the probe bounce is not
corroborated by train loss. Rate 15.96–18.37 s/step this stretch
(probe+save inflate the tail lines); starvation stays intermittent,
restart trigger unchanged (sustained >20 s/step or projection near
17 GPU-h, action only at a save boundary). 1000 steps left at ~16.5
s/step → ETA ~07:5x–08:0xZ 08-20. 62.21/71 GiB, babysit exit 0 (step
1980 poll), no gate crossings.

**Steering**: none new — read + inbox empty, history clean (the
three recorded 👍s unchanged). The owed 👍 ack (01:15Z
ride-not-restart endorsement) went out in the 03:24Z boundary post —
item closed.

**Done**: babysit poll (healthy, exit 0 at step 1980). Held
in-session for the boundary: step-2000 probe read (4.8437, within
band); **step_002000 saved (42G full) + step_001000/optimizer.pt
prune confirmed** (dir 11G weights-only); disk 108G free — net −10G
per boundary, exactly on the pruner math (~88G floor at step-3000,
no risk). Boundary post 03:24:01Z with the 👍 ack. RAM available
47G, flat. Queue validate green (depth 2, 15 open). No work-session
chain: both queued items GPU-gated post-onerig, no CPU items, depth
at threshold.

**Next**: step-2250 probe ~04:3xZ + step-2500 save boundary
~05:3x–05:4xZ (step-1500 optimizer prune confirm + probe
trend-vs-noise read); onerig endpoint ~07:5x–08:0xZ 08-20 →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 02:55–03:0xZ (tick) — **onerig healthy at step
1910; loss 0.3962 (+0.0339 bounce off the 0.3623 low — noise); rate
15.696 s/step in band, starvation absent this interval; fully quiet;
ETA ~07:4xZ 08-20; step-2000 boundary re-timed ~03:19–03:20Z — lands
minutes before this tick's hard kill, the ~03:16Z tick reads it with
a full budget.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1910/3000 at
the 02:56Z poll, loss 0.3962 (+0.0339 vs 1830 — bounce off the
0.3623 low, same noise pattern as 1520/1600). Probe curve unchanged
(4.56@1750 latest; step-2000 probe lands at the boundary). Rate:
window 3.8 steps/min and babysit line 15.696 s/step — in band,
starvation absent this interval; watch stays through the step-2000
boundary, restart trigger unchanged (sustained >20 s/step or
projection near 17 GPU-h, action only at a save boundary). ~4.8 h to
endpoint at 15.7 s/step → ETA ~07:4xZ 08-20. 62.21/71 GiB, babysit
exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (no new
reactions; the three recorded 👍s unchanged). 👍 ack still due in
the step-2000 boundary post.

**Done**: babysit poll (healthy, exit 0). Boundary-timing call: step
2000 at ~15.7 s/step from 1910@02:56Z lands ~03:19–03:20Z — inside
this tick's window only in its final minutes (hard kill 03:25:54Z),
too tight to hold + read + post + commit; the ~03:16Z timer tick
starts just before the boundary with a full 30-min budget and is the
right reader (task list unchanged: probe read + optimizer-prune
confirm + disk re-read + 👍 ack in the boundary post). Disk 118G
free, flat. RAM available 47G, flat. Queue validate green (depth 2,
15 open). No work-session chain: both queued items GPU-gated
post-onerig, no CPU items, depth at threshold.

**Next**: step-2000 save boundary ~03:19–03:20Z (next tick) →
step-2000 probe read + confirm step_001000/optimizer.pt prune + disk
re-read + rate re-read + 👍 ack in the boundary post; onerig
endpoint ~07:4xZ 08-20 → `onerig-endpoint-close` (frozen-grid sim100
≥20 / ≤10 / 11–19 bands, anchors demosonly 11 and both convicted
cells 1), then the R2 parity read + relaunch in the freed window (A5
gate, no GO ask); at the R2 endpoint the boundary is
`./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-20 03:37–03:4xZ (tick; `onerig` riding, ~9.4 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2050/3000, loss 0.3809 (−0.0092 — falling again past the 2000
boundary, the probe uptick stays uncorroborated by train loss); rate
15.129 s/step back at the registered band with the boundary
inflation out of the tail, starvation absent this interval, restart
trigger unchanged; ETA ~07:4xZ 08-20; 62.21 GiB, no gate crossings;
step-2250 probe ~04:2x–04:3xZ next tick + step-2500 boundary
~05:3x–05:4xZ; Discord fully quiet (read + inbox empty, no new
reactions — 03:24Z boundary post unreacted so far); disk 108G free
flat; RAM flat (available 47G); no chain (both queued items
GPU-gated, no CPU items)** — queue green depth 2 (15 open).

Session 2026-08-20 03:16–03:2xZ (tick; `onerig` riding, ~9.0 GPU-h
elapsed of ~14 projected / gate 17): **step-2000 boundary read
in-session — probe 4.8437 (+0.28, first uptick of the run, within
the ≤ +0.30 read band, pdnorm-mixed precedent; the 2250/2500 probes
make the trend-vs-noise call); loss 0.3752 new low at 2000;
step_002000 saved (42G full) + step_001000/optimizer.pt prune
confirmed (dir 11G weights-only); disk 108G free — net −10G per
boundary exactly on pruner math (~88G floor at step-3000); rate
16.0–18.4 s/step tail inflated by probe+save, starvation
intermittent, restart trigger unchanged; ETA ~07:5x–08:0xZ 08-20;
boundary post 03:24Z incl. the owed 👍 ack; Discord otherwise quiet
(read + inbox empty, no new reactions); RAM flat (available 47G); no
chain (both queued items GPU-gated, no CPU items)** — queue green
depth 2 (15 open).

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
