# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-23 11:36–12:2xZ (work session) — **RETIREMENT:
owner message 11:50:09Z — the machine is being retired and all work
is PAUSED; this session pivoted from the planned verdict block to
evacuation-only and the box is ready to retire. ch0fix train
COMPLETED 3000/3000 ~12:02Z (loss 0.2845, ~13.4 GPU-h vs the 16
gate) and the endpoint is banked weights-only on HF; the sim100
battery was stopped pre-verdict — the rung-2 verdict is UNBANKED and
waits for a future box.***

**Status**: no live jobs. `fontaine-v2-joint-pdnorm-ch0fix` train
COMPLETE (endpoint verified 44G complete, published ~12:02Z; probe
closed 8.34@3000, record-only; vram peak 62.24 ≤ 71). The endpoint
battery (armed 11:38Z, started 12:03:09Z) was STOPPED ~12:04Z at the
owner pause before any seed completed. GPU free, babysit registry 0
live runs.

**Steering**: owner 11:50:09Z (id 1541051920825716736): machine
retiring, pause work, save all valuable artifacts to GitHub/HF,
remove all uploaded optimizers, upload no more, report when ready.
Disposition: executed same-session (evacuation receipt below);
replied 12:04Z + acked; receipt posted at close.

**Done** (this session): [pre-pause] battery unit cloned + armed
(self-waiting), BOTH rung-3 branch materializers written + run with
all oracles green (`clean_ch0fix_act_j` action-only affine;
`clean_ep015_c` bisection subset via the rig_fewshot machinery),
both launchers + upload script staged, check.py 1112 green, commit
`0adb8f09`. [post-pause] endpoint publish verified + optim@2500
pruned; battery stopped; **evacuation**: ch0fix endpoint →
`fontaine-checkpoints/grasp_sft_v2_joint_pdnorm_ch0fix_step3000`
(weights-only + train_log); gitignored eval record → 
`fontaine-checkpoints/h100_evac_2026-08-23/` (reports.tar 5.3G,
outputs_sim_grpo_r2.tar 17G, outputs_sim_rest.tar 6.5G,
outputs_squint_screen.tar 1.1G, outputs_misc.tar 14G incl. the
snapdistill_ftrig_4k endpoint + rig_r1_step500 conversion + all
launch logs + norm_stats); `squint_twin_demos_v1` (123M labeled
corpus) → `fontaine-sim`; **optimizer purge**: ALL optimizer.pt
files AND dangling LFS blobs permanently deleted from
fontaine-checkpoints (1 file, 31.4G) and bijou-checkpoints (3 files
+ 16 dangling blobs, ~94G freed); wandb verified synced; demos
v1/v2 + grasp_sft_demos_v0 verified already on HF file-for-file;
queue → retirement pause (validate green, depth-0 stated reason);
babysit registry pruned; tick timer disabled.

**Next**: NOTHING on this box — program paused. On a future box:
(1) re-run `launch_ch0fix_endpoint_battery.sh` from the HF endpoint
→ bank the rung-2 verdict vs democlean 8/100 (frozen grid,
posts/2026-08-22-prereg-clean-ch0-affine.md); (2) the verdict
mechanically selects carrier-hunt rung 3 — both branches committed
+ oracle-verified (posts/2026-08-22-prereg-carrier-hunt-rung3.md);
datasets rebuild deterministically from the committed materializers.
`queue.json` carries the full pause record.*

*Updated 2026-08-23 11:33–11:4xZ (tick) — **pre-completion handoff
tick: healthy — step 2900/3000, loss 0.2846 (−0.0027 since 2740,
monotone continues), vram 62.24/71, rate 14.958 s/step window (3.9
steps/min). No new probe row (@2750 = 8.70 latest; the endpoint row
lands with the save). Train hits step 3000 ~11:59Z with the ENDPOINT
publish ~12:0xZ — right at this session's 12:03Z hard kill, far too
tight to verify + launch the battery in-tick — so **`run_work_next`
ARMED**: the chained 4-h work session catches completion, verifies
the endpoint publish, runs the sim100 battery (~3 GPU-h), banks the
rung-2 verdict, and executes the rung-3 branch select. Disk 73G
flat (endpoint trough ~31G safe).***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
2900/3000 at poll, babysit exit 0 (liveness 5 procs, vram 66641MiB
on gpu0, 3.9 steps/min since last sample). Loss 0.2873@2740 →
0.2846@2900 — monotone fall continues into the endpoint. Rate 14.958
s/step window, inside the judged 14.7–24.5 band. No new probe row
(series stands at …8.63@2250 → 6.01@2500 → 8.70@2750, band ~6–8.7,
within-lineage color only; the decision read is the endpoint sim100
battery vs democlean 8/100). **Disk**: 73G free, flat vs 10:52 —
the step-3000 ENDPOINT staging trough ~31G is safe, no load-bearing
prunes remain (post-endpoint optim@2500 prune stays routine). Host
RAM available 47G — sixteenth stable-plateau read. **Handoff**: 100
steps ≈ 25 min to go → step 3000 ~11:59Z, endpoint publish ~12:0xZ,
this session's hard kill 12:03:54Z — holding to verify would leave
no comfortable commit margin and the battery + verdict + rung3
select exceed a tick anyway, so the marker is armed and the chained
work session (4-h budget) executes the completion block per the
pre-reg: verify endpoint publish → sim100 battery → rung-2 verdict
→ `carrier-hunt-rung3-exec` branch select + launch (fit smoke →
launch, ONE dataset delta, seed 0; 11–19 fires neither branch,
owner escalation) → routine optim@2500 prune.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), loss/rate/probe reads
judged (all in-band, no new probe row), disk read (73G flat,
endpoint trough ~31G safe), RAM read (47G, sixteenth), Discord read
+ history (quiet), queue validate (depth 1 — below depth-2 AND the
completion block is now imminent, so the standing stated-reason
lapses: **`run_work_next` ARMED** for the completion work), now.md
keep-3 + footer rolls (09:31 entry → archive); no blog rebuild
(routine poll precedent, reader-visible content unchanged).

**Next**: the chained work session owns the completion block —
verify the step-3000 ENDPOINT publish (weights + optimizer.pt +
metadata + tokenizer, no staging temp), read the endpoint probe row,
launch the sim100 battery (~3 GPU-h) via systemd-run, bank the
rung-2 verdict vs democlean 8/100, then rung3-exec selects and
launches the branch same session; post the completion + verdict to
Discord (the quiet-tick precedent ends at completion). Routine
optim@2500 prune after the endpoint publish verifies.*

*Updated 2026-08-23 10:52–10:5xZ (tick) — **routine ch0fix poll +
short probe hold: healthy — step 2740/3000, loss 0.2873 (−0.0022
since 2580; 0.2746@2750 during the hold — monotone continues), vram
62.24/71, rate 15.093 s/step window (3.9 steps/min). Held ~3 min
in-session to catch the @2750 probe row: **8.70 — a NEW series high**
(…7.41 → 8.63 → 6.01 → 8.70) — the oscillation band widens to
~6–8.7; train_mae co-moves (7.94@2750 vs 5.59@2500), so the swings
track eval-batch composition, within-lineage color only. Disk 73G
free (−4G drift, endpoint trough ~31G still safe). ETA ~1.0 h →
done ~12:0xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
2740/3000 at poll, babysit exit 0 (liveness 5 procs, vram 66641MiB
on gpu0, 3.9 steps/min since last sample). Loss 0.2895@2580 →
0.2873@2740 → 0.2746@2750 — monotone fall continues. Rate 15.093
s/step window, inside the judged 14.7–24.5 band. Probe
eval_chunk_mae @2750 = **8.70, a new series high** (4.61 → 5.24 →
5.97 → 6.84 → 6.62 → 5.55 → 6.07 → 7.41 → 8.63 → 6.01 → 8.70): the
band is now ~6–8.7 with no monotone trend; the 10:1x "oscillation
restored" read stands but with a wider envelope. New observation
this tick: train_mae co-moves with the probe (7.94@2750 / 5.59@2500
/ 7.87@2250), so the swings look like eval-batch composition, not
model drift. Still within-lineage-only per the banked
pdnorm-rescale confound — decision read stays the endpoint sim100
battery vs democlean 8/100; carry the full series + the
train_mae-co-movement note as verdict-session color. **Disk**: 73G
free (77G at 10:1x — ~4G drift from logs/wandb, routine). Checkpoint
dir verified by file listing: step_000500/1000/1500/2000
weights-only, step_002500 full with the single optimizer.pt (32G),
no staging temp — exact keep-1 anchor. Endpoint staging trough from
73G ≈ ~31G — safe, no further load-bearing prunes (post-endpoint
optim@2500 prune stays routine). Host RAM available 47G — fifteenth
stable-plateau read. ETA 250 steps at ~15 s/step ≈ 1.0 h → done
~12:0xZ 08-23 → sim100 endpoint battery; its verdict mechanically
selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), short in-session hold
(~3 min) to catch the @2750 probe row, probe read judged (new
series high 8.70, band widened, train_mae co-movement noted —
within-lineage, non-actionable, verdict-session color), disk read
(73G, −4G routine drift, endpoint trough re-projected ~31G safe) +
checkpoint dir verification (exact keep-1, single optimizer.pt at
step_002500), RAM read (47G, fifteenth), Discord read + history
(quiet), queue validate green (depth-1 stated reason — rung3-exec
verdict-gated, no CPU items, so `run_work_next` stays unarmed),
now.md keep-3 + footer rolls (08:49 entry → archive); no blog
rebuild (routine poll precedent).

**Next**: nothing load-bearing before the boundary — the step-3000
ENDPOINT save stages ~12:0xZ into a ~31G trough (safe). Train done
~12:0xZ 08-23 → the completion tick runs the sim100 battery (~3
GPU-h) → rung-2 verdict banks → `carrier-hunt-rung3-exec` selects
and launches the branch same session (fit smoke → launch, ONE
dataset delta, seed 0; 11–19 fires neither branch, owner
escalation). Post-endpoint tick also verifies the endpoint publish
and prunes optim@2500 (routine class). The ~11:3x/12:0x tick is
likely the completion tick — budget for battery launch + verdict
write-up.*

## Utilization footer

Session 2026-08-23 11:36–12:2xZ (work session; ~0.05 marginal GPU-h
— the battery's ~1 min of seed-0 rollouts before the stop): **exploit
(the planned verdict block) pivoted to RETIREMENT EVACUATION at the
owner's 11:50:09Z pause. ch0fix train completed 3000/3000 (~13.4
GPU-h total vs the 16 gate, all in earlier notes' wall-clock);
endpoint banked weights-only to HF; battery stopped pre-verdict.
Evacuated: 44G of gitignored eval record + the ftrig endpoint +
squint corpus to HF; ALL uploaded optimizers + dangling LFS blobs
permanently purged (~125G freed across two repos). Rung-2 verdict +
rung-3 launch wait for a future box — both branches committed +
oracle-verified. FINAL session note on this machine.**

Session 2026-08-23 11:33–11:4xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **pre-completion handoff tick, healthy — step
2900/3000, loss 0.2846 monotone, vram 62.24/71, rate 14.958 s/step
(3.9 steps/min). No new probe row (@2750 = 8.70 latest). Disk 73G
flat (endpoint trough ~31G safe); RAM 47G sixteenth read. Step 3000
lands ~11:59Z with the ENDPOINT publish ~12:0xZ — at this session's
hard kill — so `run_work_next` ARMED: the chained 4-h work session
verifies the endpoint publish, runs the sim100 battery (~3 GPU-h),
banks the rung-2 verdict, and executes the rung3-exec branch select
+ launch; routine optim@2500 prune rides along.**

Session 2026-08-23 10:52–10:5xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll + short probe hold, healthy — step
2740/3000, loss 0.2873 → 0.2746@2750 monotone, vram 62.24/71, rate
15.093 s/step (3.9 steps/min). Held ~3 min to catch probe @2750 =
**8.70 NEW series high** (…8.63 → 6.01 → 8.70) — band widens to
~6–8.7, train_mae co-moves → eval-batch composition, within-lineage
color only. Disk 73G (−4G routine drift, endpoint trough ~31G safe),
checkpoint dir verified = exact keep-1 state; RAM 47G fifteenth
read. ETA ~1.0 h → ~12:0xZ 08-23 → sim100 battery → rung3-exec
branch select; queue depth-1 stated reason (rung3-exec
verdict-gated), no CPU items → `run_work_next` stays unarmed.**

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
