# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-20 00:29–00:3xZ (tick) — **onerig healthy at step
1380, loss 0.4663 new low; window and trainer line agree ~15.9
s/step; step-1500 save + probe land ~01:0xZ — read next tick; fully
quiet.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 1380/3000 at
the 00:29Z poll, loss 0.4663 (−0.0093 vs 1300, new low, falling).
Probe curve unchanged (5.59@1250 latest; the step-1500 probe and save
land together ~01:0xZ — 120 steps out). Window 3.8
steps/min (~15.8 s/step) and trainer-line 15.934 agree on a clean
interval — a touch above the smoke band, within the established
bounce; ~7.2 h to endpoint → ETA ~07:4xZ 08-20. 62.21/71 GiB, babysit
exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (no new
reactions).

**Done**: babysit poll (healthy, exit 0). Disk 129G free — flat as
expected (step-1500 save now ~120 steps ≈ 32 min out → lands ~01:0xZ,
before next tick). RAM available 47G, flat seventh tick running.
Queue validate green (depth 2, 15 open). No work-session chain: both
queued items GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-1500 save ~01:0xZ → next tick confirms
step_000500/optimizer.pt pruned (standing watch item) + disk re-read
against the pruner projection, and reads the step-1500 probe; onerig
endpoint ~07:4xZ 08-20 → `onerig-endpoint-close` (frozen-grid sim100
≥20 / ≤10 / 11–19 bands, anchors demosonly 11 and both convicted
cells 1), then the R2 parity read + relaunch in the freed window (A5
gate, no GO ask); at the R2 endpoint the boundary is
`./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

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

## Utilization footer

Session 2026-08-20 00:49–00:5xZ (tick; `onerig` riding, ~6.5 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step
1450/3000, loss 0.4318 new low (−0.0345 interval, steepest drop in
hours); window 3.4 steps/min (~17.7 s/step) vs trainer-line 16.294 —
both slower this interval, high edge of the bounce (watch: two
consecutive slow intervals = real slowdown), ETA ~07:5xZ 08-20;
62.21 GiB, no gate crossings; step-1500 save + probe land ~01:0xZ —
optimizer-prune confirm + disk re-read + probe read + rate re-read
next tick; Discord fully quiet (read + inbox empty, no new
reactions); disk 129G free flat (save not yet landed at the poll);
RAM flat (available 47G); no chain (both queued items GPU-gated
post-onerig, no CPU items)** — queue green depth 2 (15 open).

Session 2026-08-20 00:29–00:3xZ (tick; `onerig` riding, ~6.1 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step
1380/3000, loss 0.4663 new low (−0.0093 interval); window 3.8
steps/min (~15.8 s/step) and trainer-line 15.934 agree on a clean
interval — within bounce, ETA ~07:4xZ 08-20; 62.21 GiB, no gate
crossings; step-1500 save + probe land ~01:0xZ — optimizer-prune
confirm + disk re-read + probe read next tick; Discord fully quiet
(read + inbox empty, no new reactions); disk 129G free flat; RAM
flat (available 47G); no chain (both queued items GPU-gated
post-onerig, no CPU items)** — queue green depth 2 (15 open).

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
