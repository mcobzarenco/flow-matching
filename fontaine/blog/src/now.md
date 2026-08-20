# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 01:31–01:3xZ (tick) — **onerig healthy at step
1600, loss 0.4135 new low (−0.0273); rate window bounced back to
~15.8 s/step this interval — the starvation slowdown looks
intermittent, not settled; ETA ~07:5x–08:2xZ 08-20; fully quiet.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1600/3000 at
the 01:32Z poll, loss 0.4135 (−0.0273 vs 1520, new low — the bounce
at 1520 was noise as read). Probe curve unchanged (4.94@1500 latest;
step-1750 probe lands ~02:1xZ). Rate: window 3.8 steps/min (~15.8
s/step) — back inside the pre-slowdown bounce after two settled
17.4–17.7 intervals; cumulative trainer line 16.445. Read: the input
starvation is intermittent rather than fully settled — watch stays,
restart trigger unchanged (sustained >20 s/step or projection near
17 GPU-h, action only at a save boundary). ~6.4 h to endpoint at the
cumulative line → ETA ~07:5x–08:2xZ 08-20 (earlier edge back in
play). 62.21/71 GiB, babysit exit 0, no gate crossings.

**Steering**: none — read surfaced only our own 01:15Z post; inbox
empty, history clean (no new reactions).

**Done**: babysit poll (healthy, exit 0). Disk 118G free — flat as
expected (no save boundary since 1500; next lands at step-2000
~03:2xZ with the step-1000 optimizer prune). RAM available 47G,
flat. Queue validate green (depth 2, 15 open). No work-session
chain: both queued items GPU-gated post-onerig, no CPU items, depth
at threshold.

**Next**: step-1750 probe ~02:1xZ → read next tick; step-2000 save
boundary ~03:2xZ → confirm step_001000/optimizer.pt prune + disk
re-read + rate re-read (bounce vs settled); onerig endpoint
~07:5x–08:2xZ 08-20 → `onerig-endpoint-close` (frozen-grid sim100
≥20 / ≤10 / 11–19 bands, anchors demosonly 11 and both convicted
cells 1), then the R2 parity read + relaunch in the freed window (A5
gate, no GO ask); at the R2 endpoint the boundary is
`./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 01:10–01:2xZ (tick) — **onerig healthy at step
1520; step-1500 probe 4.94 (−0.65, strongest drop since 750); the
rate watch item resolved REAL: settled 17.4–17.7 s/step, diagnosed
input starvation (~5 s 0%-util stall per step) — no gate risk (~14.0
projected vs 17 GPU-h), riding; ETA ~08:2x–08:3xZ 08-20; posted.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1520/3000 at
the 01:11Z poll, loss 0.4408 (+0.0090 vs 1450 — first bounce after
the steep drop, noise). **Step-1500 probe: 4.94** — curve 12.85 →
8.04 → 6.73 → 5.83 → 5.59 → 4.94, −0.65 the strongest interval since
750. Rate: two consecutive slow windows + per-10-step trainer lines
(17.44@1490, 17.53@1530 — either side of the save) confirm a real
settled slowdown vs the 15.1–15.4 registered band. Diagnosed
in-session: dmon 25 s trace shows ~12 s busy / **~5 s at 0% util**
every step cycle = input starvation; disk exonerated (2.5% util,
iowait ~0), no co-tenant GPU process (no policy-server), no throttle
flags (41 C); workers 1-running/7-sleeping → per-worker
batch-production latency is the bound. ETA ~08:2x–08:3xZ 08-20;
projected total ~14.0 GPU-h vs the 17 gate — comfortable. 62.21/71
GiB, babysit exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (no new
reactions).

**Done**: babysit poll (exit 0). **Step-500 optimizer prune
CONFIRMED** (dir 13G weights-only; 1000/1500 each carry 32G
optimizer.pt). Disk 118G free — matches pruner math exactly (129 −
42 save + 32 prune); net −10G per boundary → ~88G floor at the
step-3000 save, no risk. RAM available 48G, flat. Slowdown
diagnosis (dmon trace + iostat + worker states) and the ride-only
decision: NO mid-run restart — a resume buys ~0.9 GPU-h but costs
the isolation cell's data-order comparability, with gates clear.
Discord post out (probe + slowdown + decision). Queue validate green
(depth 2, 15 open). No work-session chain: both queued items
GPU-gated post-onerig, no CPU items.

**Next**: rate re-read every tick — restart trigger: sustained >20
s/step or projection nearing 17 GPU-h, action only at a save
boundary (step-2000 ~03:2xZ: step-1000 optimizer prune + disk
re-read + step-1750/2000 probes); onerig endpoint ~08:2x–08:3xZ
08-20 → `onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 /
11–19 bands, anchors demosonly 11 and both convicted cells 1), then
the R2 parity read + relaunch in the freed window (A5 gate, no GO
ask); at the R2 endpoint the boundary is `./launch_grpo_r2.sh
boundary outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 00:49–00:5xZ (tick) — **onerig healthy at step
1450, loss 0.4318 new low (steepest interval drop in hours); window
~17.7 s/step this interval — high edge of the bounce, watch item;
step-1500 save + probe land ~01:0xZ — read next tick; fully
quiet.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1450/3000 at
the 00:50Z poll, loss 0.4318 (−0.0345 vs 1380, new low — the
steepest interval drop since the early curve). Probe curve unchanged
(5.59@1250 latest; the step-1500 save + probe land ~50 steps ≈ 15
min out → ~01:0xZ, just after this close). Window 3.4 steps/min
(~17.7 s/step) vs trainer-line 16.294 — both a touch slower this
interval, high edge of the established bounce (14.0–15.9 recent);
one interval is noise, two consecutive would be a real slowdown —
watch next tick. ~7.0 h to endpoint at the trainer line → ETA
~07:5xZ 08-20 (drifting slightly later, still noise-level).
62.21/71 GiB, babysit exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (no new
reactions).

**Done**: babysit poll (healthy, exit 0). Disk 129G free — flat,
consistent with the step-1500 save not yet landed at the poll. RAM
available 47G, flat eighth tick running. Queue validate green (depth
2, 15 open). No work-session chain: both queued items GPU-gated
post-onerig, no CPU items, depth at threshold.

**Next**: step-1500 save + probe ~01:0xZ → next tick confirms
step_000500/optimizer.pt pruned (standing watch item) + disk re-read
against the pruner projection, reads the step-1500 probe, and
re-reads the rate (bounce vs slowdown); onerig endpoint ~07:5xZ
08-20 → `onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 /
11–19 bands, anchors demosonly 11 and both convicted cells 1), then
the R2 parity read + relaunch in the freed window (A5 gate, no GO
ask); at the R2 endpoint the boundary is `./launch_grpo_r2.sh
boundary outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-20 01:31–01:3xZ (tick; `onerig` riding, ~7.2 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
1600/3000, loss 0.4135 new low (−0.0273 interval); rate window
bounced back to ~15.8 s/step vs the settled 17.4–17.7 of the last
two ticks — starvation slowdown reads intermittent, watch stays,
restart trigger unchanged; cumulative trainer line 16.445, ETA
~07:5x–08:2xZ 08-20; 62.21 GiB, no gate crossings; step-1750 probe
~02:1xZ + step-2000 boundary ~03:2xZ (step-1000 optimizer prune +
disk re-read) next; Discord fully quiet (read + inbox empty, no new
reactions); disk 118G free flat; RAM flat (available 47G); no chain
(both queued items GPU-gated, no CPU items)** — queue green depth 2
(15 open).

Session 2026-08-20 01:10–01:2xZ (tick; `onerig` riding, ~6.9 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
1520/3000; step-1500 probe read 4.94 (−0.65, strongest interval
since 750); rate watch RESOLVED as real: settled 17.4–17.7 s/step vs
the 15.1–15.4 band, diagnosed input starvation (~5 s 0%-util stall
per step cycle; disk/thermal/co-tenant all exonerated), decision
ride-not-restart (~14.0 GPU-h projected vs 17 gate, isolation-cell
comparability preserved), ETA ~08:2x–08:3xZ 08-20; step-500
optimizer prune confirmed (13G weights-only), disk 118G free matches
pruner math (~88G floor at 3000); RAM flat (available 48G); Discord
post out (probe + slowdown + decision), inbound fully quiet (read +
inbox empty, no new reactions); no chain (both queued items
GPU-gated, no CPU items)** — queue green depth 2 (15 open).

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
