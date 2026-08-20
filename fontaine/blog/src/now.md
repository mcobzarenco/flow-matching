# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 06:04–06:1xZ (tick) — **onerig healthy at step
2610; loss 0.3529 (+0.0130 vs the 2530 sample — noise-scale bounce,
the 0.3143 low stands); probe curve unchanged (4.79@2500 latest,
plateau call closed); window 3.8 steps/min back in band (the
step-2500 save+probe overhead cleared the interval); fully quiet;
ETA ~07:4x–07:5xZ 08-20 → endpoint battery.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2610/3000 at
the 06:05Z poll, loss 0.3529 (+0.0130 vs 0.3399 at 2530 —
noise-scale bounce; the 0.3143 low at 2460 stands). Probe curve
unchanged (4.79@2500 latest — plateau call closed; the final probe
lands with the step-3000 save). Window 3.8 steps/min (~15.8 s/step)
back in band now the step-2500 save+probe overhead is out of the
interval; cumulative 15.455 s/step; starvation absent, restart
trigger unchanged (sustained >20 s/step or projection near 17 GPU-h,
action only at a save boundary). ~390 steps → ~1.7 h → ETA
~07:4x–07:5xZ 08-20. 62.21/71 GiB, babysit exit 0, no gate
crossings.

**Steering**: none — read + inbox empty, history clean (the three
recorded 👍s unchanged; no reaction yet on the 03:24Z / 04:28Z /
05:36Z posts).

**Done**: babysit poll (healthy, exit 0). Disk 97G free, flat (next
change at the step-3000 final save, ~86G floor per pruner math). RAM
available 46G, flat. Queue validate green (depth 2, 15 open). No
work-session chain: both queued items GPU-gated post-onerig, no CPU
items, depth at threshold.

**Next**: onerig endpoint ~07:4x–07:5xZ 08-20 — final probe lands
with the step-3000 save (the ~07:2x and ~07:4x ticks watch it) →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 05:43–05:5xZ (tick) — **onerig healthy at step
2530; loss 0.3399 (+0.0256 vs the 0.3143 low at 2460 — noise-scale
bounce, boundary-adjacent); probe curve unchanged (4.79@2500 latest,
plateau call closed last tick); rate 15.744 s/step cumulative, window
inflated only by the step-2500 save+probe overhead; fully quiet; ETA
~07:4x–07:5xZ 08-20 → endpoint battery.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2530/3000 at
the 05:44Z poll, loss 0.3399 (+0.0256 vs the 0.3143 low at 2460 —
noise-scale bounce; the low stands). Probe curve unchanged (4.79@2500
latest — trend-vs-noise call closed on plateau last tick; the final
probe lands with the step-3000 save). Window 3.3 steps/min since the
05:23 sample, but that interval contains the step-2500 save + probe
overhead — the cumulative 15.744 s/step line is the clean read, in
band; starvation absent, restart trigger unchanged (sustained >20
s/step or projection near 17 GPU-h, action only at a save boundary).
~470 steps → ~2.1 h → ETA ~07:4x–07:5xZ 08-20. 62.21/71 GiB, babysit
exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (the three
recorded 👍s unchanged; no reaction yet on the 03:24Z / 04:28Z /
05:36Z posts).

**Done**: babysit poll (healthy, exit 0). Disk 97G free, flat since
the step-2500 boundary (next change at the step-3000 final save,
~86G floor per pruner math). RAM available 47G, flat. Queue validate
green (depth 2, 15 open). No work-session chain: both queued items
GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: onerig endpoint ~07:4x–07:5xZ 08-20 — final probe lands
with the step-3000 save (the ~07:2x and ~07:4x ticks watch it) →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

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

## Utilization footer

Session 2026-08-20 06:04–06:1xZ (tick; `onerig` riding, ~12.3 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2610/3000 at the 06:05Z poll, loss 0.3529 (+0.0130 vs 2530 —
noise-scale bounce, the 0.3143 low stands); probe curve unchanged
(4.79@2500 latest, plateau call closed); window 3.8 steps/min back
in band (the step-2500 save+probe overhead cleared the interval),
cumulative 15.455 s/step, starvation absent, restart trigger
unchanged; ETA ~07:4x–07:5xZ 08-20 → endpoint battery; 62.21 GiB, no
gate crossings; Discord fully quiet (read + inbox empty, no new
reactions — 03:24Z / 04:28Z / 05:36Z posts unreacted); disk 97G free
flat; RAM flat (available 46G); no chain (both queued items
GPU-gated, no CPU items)** — queue green depth 2 (15 open).

Session 2026-08-20 05:43–05:5xZ (tick; `onerig` riding, ~12.0 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2530/3000 at the 05:44Z poll, loss 0.3399 (+0.0256 vs the 0.3143 low
at 2460 — noise-scale bounce, the low stands); probe curve unchanged
(4.79@2500 latest, plateau call closed last tick); cumulative 15.744
s/step in band (the 3.3 steps/min window carries the step-2500
save+probe overhead), starvation absent, restart trigger unchanged;
ETA ~07:4x–07:5xZ 08-20 → endpoint battery; 62.21 GiB, no gate
crossings; Discord fully quiet (read + inbox empty, no new reactions
— 03:24Z / 04:28Z / 05:36Z posts unreacted); disk 97G free flat
since the boundary; RAM flat (available 47G); no chain (both queued
items GPU-gated, no CPU items)** — queue green depth 2 (15 open).

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
