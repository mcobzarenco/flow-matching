# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 07:30–11:0xZ (work session) — **ONERIG VERDICT:
sim100 28/100 — MIX-EXONERATED through the frozen grid. The
two-dataset mix (demos + v2 ×4, clean dropped) beats its demos-only
control +17 (CI95 [8, 26], McNemar p = 0.0009) and the convicted
three-way cell +27 (CI95 [19, 36], p = 1.5e-8). One recipe delta —
dropping 13.6k clean frames, 0.65% of the corpus — turned 1/100 into
28/100. Rig data at ~6% share HELPS grasping once clean is out.***

**Status**: battery leg 2 (k4l2 panel npz) live on the H100 (started
~10:51Z, ~0.5 GPU-h, unit `fontaine-onerig-endpoint-battery`);
checkpoint bank upload live (unit `fontaine-onerig-ckpt-bank`,
weights-only ~12 GiB → `fontaine-checkpoints/`
`grasp_sft_v2_joint_pdnorm_onerig_step3000`). Train COMPLETE 3000/3000
(~13.4 GPU-h, loss 0.3299, vram 62.21/71, zero crossings); final
probe 4.5266@3000 held the run low — the curve ended improving, the
opposite shape of the convicted cell. Honest cell total ~17.0 vs the
17 gate: a ~48-min idle gap (my own watch-loop cmdline matched the
run pgrep and deadlocked the battery wait — the 08-19 class incident
reproduced, note sharpened in the registry) ate the margin; no extra
legs.

**Steering**: none — inbox empty all session (reads 07:30/08:38/
09:11/09:40/10:09Z); the three recorded 👍s unchanged.

**Done**: onerig-endpoint-close primary verdict banked (battery
script staged + armed pre-endpoint 028c94c; registry rolled
train→battery 5e73979; verdict + paired reads + queue/now/blog this
commit). Posts: train-complete 08:38Z, mid-leg signal 09:11Z (7/25
early-terms), VERDICT 10:51Z (id 1539950050740801616). Paired jsons
banked (`analysis__sim100_paired_onerig3000_vs_{disc1000,pdnorm3000}
.json`). Queue: verdict noted on onerig-endpoint-close (leg-2 CPU
tail remains), refill `prereg-draft-demos-plus-clean` queued (the
poison-pinning cell) — validate green, depth 3.

**Next**: leg 2 lands ~11:2xZ → chained session takes the CPU tail
(panel guard vs disc-1000 npz, truthfit rewear, ladder restamp,
onerig HTML report, bank verify) + the demos+clean pre-reg draft;
`run_work_next` armed. Then `queue_cli.py next` →
grpo-r2-parity-read-and-relaunch owns the first free GPU window
(`./launch_grpo_r2.sh parity`, A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 07:07–07:1xZ (tick) — **onerig healthy at step
2850; loss 0.3338 (−0.0098 vs 2770 — falling again, the 0.3143 low
stands); probe curve unchanged (4.50@2750 new-low latest; the final
probe lands with the step-3000 save); window 3.8 steps/min in band;
fully quiet; ~150 steps → ~0.7 h → ETA ~07:5xZ 08-20 — the endpoint
lands after this tick's cap, the next tick catches the battery.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2850/3000 at
the 07:08Z poll, loss 0.3338 (−0.0098 vs 0.3436 at 2770 — falling
again; the 0.3143 low at 2460 stands). Probe curve unchanged
(4.50@2750 latest — the new run low; the final probe lands with the
step-3000 save and is the read to beat). Window 3.8 steps/min,
cumulative 15.734 s/step — in band; starvation absent, restart
trigger moot this close to the endpoint. ~150 steps → ~0.7 h → ETA
~07:5xZ 08-20; the endpoint lands just past this tick's 07:38Z cap,
so the ~07:2x–07:4x tick takes the final probe + step-3000 save and
opens the endpoint battery (chaining a work session — the sim100
battery exceeds a tick cap). 62.21/71 GiB, babysit exit 0, no gate
crossings.

**Steering**: none — read surfaced only our own 06:47Z probe-low
post (cursor advance), inbox empty, history clean (the three
recorded 👍s unchanged; no reaction on the 03:24Z / 04:28Z / 05:36Z
/ 06:47Z posts).

**Done**: babysit poll (healthy, exit 0). Disk 97G free, flat (next
change at the step-3000 final save, ~86G floor per pruner math). RAM
available 46G, flat. Queue validate green (depth 2, 15 open). No
work-session chain this tick: both queued items GPU-gated
post-onerig, no CPU items, depth at threshold.

**Next**: onerig endpoint ~07:5xZ 08-20 — final probe + step-3000
save (the next tick catches it; the 4.50@2750 low makes the final
read the one to beat) → `onerig-endpoint-close` (frozen-grid sim100
≥20 / ≤10 / 11–19 bands, anchors demosonly 11 and both convicted
cells 1), then the R2 parity read + relaunch in the freed window (A5
gate, no GO ask); at the R2 endpoint the boundary is
`./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 06:46–06:5xZ (tick) — **onerig step-2750 probe:
4.50 — NEW RUN LOW (below the 4.56@1750 low); the plateau call gets
amended: the 2000–2500 stretch was a shoulder, the curve is improving
again into the endpoint; loss 0.3436 (−0.0116, corroborating); window
3.8 steps/min in band; fully quiet; ~230 steps → ETA ~07:4x–07:5xZ
08-20 → endpoint battery.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 2770/3000 at
the 06:47Z poll. **Probe 4.50@2750 — new run low**: curve 12.85 →
8.04 → 6.73 → 5.83 → 5.59 → 4.94 → 4.56 → 4.84 → 4.80 → 4.79 →
**4.50** (−0.29, strongest drop since 1500, undercuts the 4.56@1750
low). The plateau call closed at 2500 amends to: the 2000–2500
stretch was a shoulder, not the ceiling — the run is still buying
probe improvement at the end. Loss 0.3436 (−0.0116 vs 2690; the
0.3143 low at 2460 stands). Window 3.8 steps/min, 15.92 s/step —
in band; starvation absent, restart trigger moot this close to the
endpoint. ~230 steps → ~1.0 h → ETA ~07:4x–07:5xZ 08-20. 62.21/71
GiB, babysit exit 0, no gate crossings.

**Steering**: none — read + inbox empty, history clean (the three
recorded 👍s unchanged; no reaction on the 03:24Z / 04:28Z / 05:36Z
posts).

**Done**: babysit poll (healthy, exit 0). Probe-low post to Discord
06:47Z. Disk 97G free, flat (next change at the step-3000 final
save, ~86G floor per pruner math). RAM available 47G, flat. Queue
validate green (depth 2, 15 open). No work-session chain: both
queued items GPU-gated post-onerig, no CPU items, depth at
threshold.

**Next**: onerig endpoint ~07:4x–07:5xZ 08-20 — final probe lands
with the step-3000 save (the ~07:2x and ~07:4x ticks watch it; the
4.50@2750 low makes the final read the one to beat) →
`onerig-endpoint-close` (frozen-grid sim100 ≥20 / ≤10 / 11–19 bands,
anchors demosonly 11 and both convicted cells 1), then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-20 07:30–11:0xZ (work session; exploit; onerig cell
~17.0 GPU-h total vs gate 17 — train ~13.4 + battery ~2.8 + idle-gap
incident ~0.8): **onerig-endpoint-close primary verdict: sim100
28/100 MIX-EXONERATED (control 11, convicted cells 1; paired +17
CI95 [8, 26] p = 0.0009 vs control, +27 CI95 [19, 36] p = 1.5e-8 vs
convicted); battery armed pre-endpoint (no idle gap by design —
then a ~48-min gap anyway: my watch-loop cmdline matched the run
pgrep, the 08-19 deadlock class reproduced, registry note
sharpened); ckpt bank firing (weights-only); queue refilled with the
demos+clean poison-pinning draft; leg-2 CPU tail chained via
run_work_next** — queue green depth 3 (16 open).

Session 2026-08-20 07:07–07:1xZ (tick; `onerig` riding, ~13.3 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2850/3000 at the 07:08Z poll, loss 0.3338 (−0.0098 vs 2770 — falling
again, the 0.3143 low stands); probe curve unchanged (4.50@2750
new-low latest, final probe lands with the step-3000 save); window
3.8 steps/min / cumulative 15.734 s/step in band, starvation absent;
~150 steps → ~0.7 h → ETA ~07:5xZ 08-20 — endpoint lands past this
tick's cap, the next tick takes the final probe + endpoint battery;
62.21 GiB, no gate crossings; Discord fully quiet (read surfaced
only our own 06:47Z post, inbox empty, no new reactions); disk 97G
free flat; RAM flat (available 46G); no chain (both queued items
GPU-gated, no CPU items)** — queue green depth 2 (15 open).

Session 2026-08-20 06:46–06:5xZ (tick; `onerig` riding, ~13.0 GPU-h
elapsed of ~14 projected / gate 17): **babysit exit 0 — step
2770/3000 at the 06:47Z poll; probe 4.50@2750 NEW RUN LOW (−0.29,
strongest drop since 1500, below the 4.56@1750 low — the 2000–2500
plateau amends to a shoulder, curve improving into the endpoint);
loss 0.3436 (−0.0116, corroborating; the 0.3143 low stands); window
3.8 steps/min / 15.92 s/step in band, starvation absent; ~230 steps
→ ETA ~07:4x–07:5xZ 08-20 → endpoint battery; 62.21 GiB, no gate
crossings; probe-low post 06:47Z; Discord otherwise fully quiet
(read + inbox empty, no new reactions); disk 97G free flat; RAM flat
(available 47G); no chain (both queued items GPU-gated, no CPU
items)** — queue green depth 2 (15 open).

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
