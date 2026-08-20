# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 00:07–00:1xZ (tick) — **onerig healthy at step
1300, loss 0.4756 new low; probe read landed: 5.59@1250, curve still
improving; fully quiet.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1300/3000 at
the 00:08Z poll, loss 0.4756 (−0.0095 vs 1220, new low, falling).
**Step-1250 probe (last tick's pending read): 5.59** — curve 12.85 →
8.04 → 6.73 → 5.83 → 5.59, still improving, −0.24 on the interval;
the drift guard passed at 1000 and the trajectory keeps confirming
it. Window 3.8 steps/min (~15.9 s/step) vs trainer-line 14.926
cumulative — window slower this interval, the mirror of last tick's
bounce; ~7.0–7.5 h to endpoint → ETA ~07:1x–07:4xZ 08-20 (holding).
62.21/71 GiB, babysit exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (no new
reactions).

**Done**: babysit poll (healthy, exit 0). Disk 129G free — flat as
expected (step-1500 save pending, now ~200 steps ≈ 50 min out →
lands ~00:5xZ, before next tick). RAM available 47G, flat sixth tick
running. Queue validate green (depth 2, 15 open). No work-session
chain: both queued items GPU-gated post-onerig, no CPU items, depth
at threshold.

**Next**: step-1500 save ~00:5xZ → next tick confirms
step_000500/optimizer.pt pruned (standing watch item) + disk re-read
against the pruner projection, and reads the step-1500 probe; onerig
endpoint ~07:1x–07:4xZ 08-20 → `onerig-endpoint-close` (frozen-grid
sim100 ≥20 / ≤10 / 11–19 bands, anchors demosonly 11 and both
convicted cells 1), then the R2 parity read + relaunch in the freed
window (A5 gate, no GO ask); at the R2 endpoint the boundary is
`./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-19 23:46–23:5xZ (tick) — **onerig healthy at step
1220, loss 0.4851 new low; probe 1250 lands right at this close —
read next tick; fully quiet.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1220/3000 at
the 23:47Z poll, loss 0.4851 (−0.0075 vs 1130, new low, falling);
probe curve unchanged (5.83@1000 latest; the step-1250 probe lands
~23:5xZ, right at this tick's close — read next tick). Window 4.3
steps/min (~14.0 s/step) vs trainer-line 15.761 cumulative — window
faster this interval, normal bounce; ~7.8 h to endpoint → ETA
~07:3x–07:4xZ 08-20 (holding). 62.21/71 GiB, babysit exit 0, no gate
crossings.

**Steering**: none — read + inbox empty, history clean (no new
reactions; the 👍 on the drift-PASS post was recorded last tick).

**Done**: babysit poll (healthy, exit 0). Disk 129G free — flat, as
expected (step-1500 save pending; at the current rate it lands
~01:0xZ, 280 steps out from the poll). RAM available 48G, flat fifth
tick running. Queue validate green (depth 2, 15 open). No
work-session chain: both queued items GPU-gated post-onerig, no CPU
items, depth at threshold.

**Next**: probe 5.83→?@1250 read next tick (~00:1xZ); step-1500 save
~01:0xZ → confirm step_000500/optimizer.pt pruned (standing watch
item) + disk re-read against the pruner projection; onerig endpoint
~07:3x–07:4xZ 08-20 → `onerig-endpoint-close` (frozen-grid sim100
≥20 / ≤10 / 11–19 bands, anchors demosonly 11 and both convicted
cells 1), then the R2 parity read + relaunch in the freed window (A5
gate, no GO ask); at the R2 endpoint the boundary is
`./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-19 23:26–23:3xZ (tick) — **onerig healthy at step
1130, loss 0.4926 new low; owner 👍 on the drift-PASS post recorded;
fully quiet otherwise.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1130/3000 at
the 23:26Z poll, loss 0.4926 (−0.0104 vs 1050, new low, falling);
probe curve unchanged since the drift PASS (5.83@1000, next probe
lands at 1250). Window 3.8 steps/min (~15.8 s/step) and trainer-line
15.909 s/step agree on a clean interval (no save/probe in it) — a
touch above the 15.2–15.4 smoke band but within the tick-to-tick
bounce; ~8.3 h to endpoint → ETA ~07:4xZ 08-20 (drifted ~15 min later
vs last tick's read). 62.21/71 GiB, babysit exit 0, no gate
crossings.

**Steering**: **NEW — owner 👍×1 on the 23:07Z drift-PASS post** (id
…404337; wasn't there when posted last tick): agreement with the
step-1000 drift verdict, recorded per the reaction-as-steering rule,
no reply owed (a result-post ack). Read + inbox otherwise empty.

**Done**: babysit poll (healthy, exit 0). Disk 129G free — flat vs
last tick, as projected (the step-1500 save hasn't landed; at the
current rate it lands ~00:5x–01:0xZ, later than the earlier ~00:2xZ
estimate). RAM available 48G, flat fourth tick running. Queue
validate green (depth 2, 15 open). No work-session chain: both
queued items GPU-gated post-onerig, no CPU items, depth at
threshold.

**Next**: step-1500 save ~00:5x–01:0xZ → confirm
step_000500/optimizer.pt pruned (standing watch item) + disk re-read
against the pruner projection; onerig endpoint ~07:4xZ 08-20 →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-20 00:07–00:1xZ (tick; `onerig` riding, ~5.8 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step
1300/3000, loss 0.4756 new low (−0.0095 interval); step-1250 probe
read landed: 5.59 (curve 12.85→8.04→6.73→5.83→5.59, still
improving); window 3.8 steps/min (~15.9 s/step) vs trainer-line
14.926 cumulative — mirror of last tick's bounce, ETA ~07:1x–07:4xZ
08-20; 62.21 GiB, no gate crossings; Discord fully quiet (read +
inbox empty, no new reactions); disk 129G free flat (step-1500 save
~00:5xZ carries the optimizer-prune watch item to next tick); RAM
flat (available 47G); no chain (both queued items GPU-gated
post-onerig, no CPU items)** — queue green depth 2 (15 open).

Session 2026-08-19 23:46–23:5xZ (tick; `onerig` riding, ~5.4 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step
1220/3000, loss 0.4851 new low (−0.0075 interval); window 4.3
steps/min (~14.0 s/step) vs trainer-line 15.761 cumulative — normal
bounce, ETA ~07:3x–07:4xZ 08-20; 62.21 GiB, no gate crossings;
step-1250 probe lands right at this close — read next tick; Discord
fully quiet (read + inbox empty, no new reactions); disk 129G free
flat (step-1500 save ~01:0xZ carries the optimizer-prune watch
item); RAM flat (available 48G); no chain (both queued items
GPU-gated post-onerig, no CPU items)** — queue green depth 2 (15
open).

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
