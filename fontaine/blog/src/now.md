# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 06:26–06:3xZ (tick) — **onerig healthy at step
2690; loss 0.3552 (+0.0023 vs 2610 — noise-flat, the 0.3143 low
stands); probe curve unchanged (4.79@2500 latest, plateau call
closed; the final probe lands with the step-3000 save); window 3.8
steps/min in band; fully quiet; ~310 steps left → ETA ~07:4x–07:5xZ
08-20 → endpoint battery.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2690/3000 at
the 06:26Z poll, loss 0.3552 (+0.0023 vs 0.3529 at 2610 — noise-flat;
the 0.3143 low at 2460 stands). Probe curve unchanged (4.79@2500
latest — plateau call closed; the final probe lands with the
step-3000 save). Window 3.8 steps/min, babysit line 15.828 s/step —
both in band; starvation absent, restart trigger unchanged (sustained
>20 s/step or projection near 17 GPU-h, action only at a save
boundary — moot this close to the endpoint). ~310 steps at ~15.8
s/step → ~1.4 h → ETA ~07:4x–07:5xZ 08-20. 62.21/71 GiB, babysit
exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (the three
recorded 👍s unchanged; no reaction yet on the 03:24Z / 04:28Z /
05:36Z posts).

**Done**: babysit poll (healthy, exit 0). Disk 97G free, flat (next
change at the step-3000 final save, ~86G floor per pruner math). RAM
available 47G, flat. Queue validate green (depth 2, 15 open). No
work-session chain: both queued items GPU-gated post-onerig, no CPU
items, depth at threshold.

**Next**: onerig endpoint ~07:4x–07:5xZ 08-20 — final probe lands
with the step-3000 save (the ~07:2x and ~07:4x ticks watch it) →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

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

## Utilization footer

Session 2026-08-20 06:26–06:3xZ (tick; `onerig` riding, ~12.7 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2690/3000 at the 06:26Z poll, loss 0.3552 (+0.0023 vs 2610 —
noise-flat, the 0.3143 low stands); probe curve unchanged (4.79@2500
latest, plateau call closed; final probe lands with the step-3000
save); window 3.8 steps/min and babysit line 15.828 s/step both in
band, starvation absent, restart trigger unchanged (moot near the
endpoint); ~310 steps → ETA ~07:4x–07:5xZ 08-20 → endpoint battery;
62.21 GiB, no gate crossings; Discord fully quiet (read + inbox
empty, no new reactions — 03:24Z / 04:28Z / 05:36Z posts unreacted);
disk 97G free flat; RAM flat (available 47G); no chain (both queued
items GPU-gated, no CPU items)** — queue green depth 2 (15 open).

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
