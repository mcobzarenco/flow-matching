# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-23 08:49–08:5xZ (tick) — **routine ch0fix poll:
healthy — step 2290/3000, loss 0.3074 (−0.0201 since 2140, monotone
continues), vram 62.24/71, rate 15.58 s/step window (3.7 steps/min,
util sampled 100% this poll). Probe @2250 written: **8.63 — a SECOND
consecutive new series high** (…5.55 → 6.07 → 7.41 → 8.63); the last
four reads are now a clean upward run, but the read stays
within-lineage-only per the banked pdnorm-rescale confound —
decision is the endpoint sim100 battery. Disk 87G flat, checkpoint
dir verified exact keep-1 state. RAM 47G stable. ETA ~3.1 h →
~12:0xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
2290/3000, babysit exit 0 (liveness 5 procs, vram 66641MiB on gpu0,
util sampled 100% this poll). Loss 0.3275@2140 → 0.3074@2290 —
monotone fall continues. Rate 15.58 s/step window, inside the judged
14.7–24.5 band. Probe eval_chunk_mae @2250 = **8.63, second
consecutive new high** (4.61 → 5.24 → 5.97 → 6.84 → 6.62 → 5.55 →
6.07 → 7.41 → 8.63) — what read as oscillation ~6 at 06:4x now reads
as an upward drift over the last four rows; still within-lineage
record only per the banked pdnorm-rescale confound (the probe is
computed in the rescaled pdnorm space, not comparable across
lineages), decision read stays the endpoint sim100 battery vs
democlean 8/100. Worth carrying forward: if @2500/@2750 continue the
run, note it in the verdict-session write-up as lineage color — it
changes no gate. **Disk**: 87G free, flat since the 07:3x prune.
Checkpoint dir verified by file listing: step_000500/1000/1500
weights-only, step_002000 full with optimizer.pt, no staging temp —
exactly the keep-1 anchor. Next save step 2500 lands ~09:45Z (210
steps at the window rate) → trough ~45G from the 87G floor, safe;
the post-save tick (~10:0x–10:1xZ window, after this session's
09:20Z hard kill) verifies it and prunes optim@2000 — that prune is
LOAD-BEARING (missed → the step-3000 endpoint staging trough is
~3G, ENOSPC class). Host RAM available 47G — plateau holding
(twelfth read). ETA ~3.1 h at the window rate → done ~12:0xZ 08-23
→ sim100 endpoint battery; its verdict mechanically selects the
rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), probe @2250 second
new-high read judged (upward run, within-lineage, non-actionable —
flagged as verdict-session color), disk read (87G flat) +
checkpoint dir verification (keep-1 state, single optimizer.pt at
step_002000), RAM read (47G, twelfth), Discord read + history,
queue validate green (depth-1 stated reason — rung3-exec
verdict-gated, no CPU items, so `run_work_next` stays unarmed),
now.md keep-3 + footer rolls (06:46 entry → archive).

**Next**: the step-2500 save lands ~09:45Z, after this session's
hard kill — the ~10:0x tick verifies it and prunes optim@2000, and
the prune is LOAD-BEARING (missed → endpoint staging trough ~3G
ENOSPC). Probes @2500/@2750 readable en route — watch whether the
upward run extends. Otherwise nothing fires before the ch0fix
boundary — train done ~12:0xZ 08-23 → battery ~3 GPU-h → rung-2
verdict banks → `carrier-hunt-rung3-exec` selects and launches the
branch same session (fit smoke → launch, ONE dataset delta, seed 0;
11–19 fires neither branch, owner escalation).*

*Updated 2026-08-23 08:08–08:1xZ (tick) — **routine ch0fix poll:
healthy — step 2140/3000, loss 0.3275 (−0.0081 since 2000, monotone
continues), vram 62.24/71, rate 16.535 s/step window (3.4
steps/min). No new probe row (@2000 = 7.41 latest; @2250 lands
~08:5xZ). Disk 87G flat since the post-save prune; checkpoint dir
verified — exactly the keep-1 state (only step_002000 carries
optimizer.pt). RAM back to 47G (the 45G tenth read resolved as
noise, plateau restored). ETA ~4.0 h → ~12:1xZ 08-23. The step-2500
save lands ~09:5xZ; its post-save prune stays LOAD-BEARING.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
2140/3000, babysit exit 0 (liveness 5 procs, vram 66641MiB on gpu0;
instant util sampled 0% this poll but steps advance at 3.4/min with
the window rate in-band — the recurring sampling artifact, not
starvation). Loss 0.3356@2000 → 0.3275@2140 — monotone fall
continues. Rate 16.535 s/step window, inside the judged 14.7–24.5
band. Probe eval_chunk_mae unchanged: …5.55@1500 → 6.07@1750 →
7.41@2000 (@2250 lands ~08:5xZ) — the @2000 new-high stands as the
latest read; within-lineage record only per the banked
pdnorm-rescale confound, decision read stays the endpoint sim100
battery vs democlean 8/100. **Disk**: 87G free, flat since the
07:3x prune (ambient drift paused). Checkpoint dir verified by file
listing: step_000500/1000/1500 weights-only, step_002000 full with
optimizer.pt — exactly the keep-1 anchor, no staging temp. Next
save step 2500 lands ~09:5xZ (360 steps at the window rate) →
trough ~45G from the 87G floor, safe; the post-save tick verifies
it and prunes optim@2000 — that prune is LOAD-BEARING (missed → the
step-3000 endpoint staging trough is ~3G, ENOSPC class, aimed at
the one save the rung needs). Host RAM available 47G — the 07:3x
45G read resolved as noise, plateau restored (eleventh read). ETA
~4.0 h at the window rate → done ~12:1xZ 08-23 → sim100 endpoint
battery; its verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), disk read (87G flat) +
checkpoint dir verification (keep-1 state, single optimizer.pt at
step_002000), RAM read (47G, plateau restored), Discord read +
history, queue validate green (depth-1 stated reason — rung3-exec
verdict-gated, no CPU items, so `run_work_next` stays unarmed),
now.md keep-3 + footer rolls (06:05 entry → archive).

**Next**: the step-2500 save lands ~09:5xZ — that post-save tick
verifies it and prunes optim@2000, and the prune is LOAD-BEARING
(missed → endpoint staging trough ~3G ENOSPC). Probes @2250/@2500
readable en route. Otherwise nothing fires before the ch0fix
boundary — train done ~12:1xZ 08-23 → battery ~3 GPU-h → rung-2
verdict banks → `carrier-hunt-rung3-exec` selects and launches the
branch same session (fit smoke → launch, ONE dataset delta, seed 0;
11–19 fires neither branch, owner escalation).*

*Updated 2026-08-23 07:27–07:4xZ (tick) — **post-save tick executed:
step-2000 save VERIFIED (published 07:29:18Z, 42G complete — staging
watched live: 78G mid-stage, 56G trough = the projection exactly) and
the keep-1 prune fired — optim@1500 deleted → 87G free. Run healthy:
step 2000/3000, loss 0.3356 (the 06:4x flat window was noise —
monotone resumed), vram 62.24/71, ~16.8 s/step. Probe @2000 = 7.41 —
NEW series high. ⚠ The step-2500 post-save prune is now
LOAD-BEARING: a missed optim@2000 prune puts the endpoint staging
trough at ~3G (ENOSPC class). ETA ~4.7 h → ~12:1xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
2000/3000 (two-thirds), babysit exit 0 (liveness 5 procs, vram
66641MiB on gpu0, util sampled 100% this poll). Loss 0.3594@1850 →
0.3356@2000 — down again, confirming the 06:4x flat window as noise.
Rate ~16.8 s/step from the jsonl window (3.7 steps/min), inside the
judged 14.7–24.5 band; babysit's None loss/rate fields were the
trailing eval row at the save boundary, not a fault. Probe
eval_chunk_mae @2000 = **7.41, a new series high** (4.61 → 5.24 →
5.97 → 6.84 → 6.62 → 5.55 → 6.07 → 7.41) — the oscillation now reads
as drifting upward; still within-lineage record only per the banked
pdnorm-rescale confound, decision read stays the endpoint sim100
battery vs democlean 8/100. **Disk**: this tick caught the staging
live — 78G free with 21.6G of the optimizer written, then published
07:29:18Z and verified complete (weights + 33.7G optimizer.pt +
metadata + tokenizer, no staging temp), trough 56G exactly as
projected → optim@1500 pruned per the keep-1 anchor → **87G free**
(dir 13/11/11/42G du, hardlink-adjusted). Remaining troughs from the
87G floor: ~45G at the step-2500 staging (~09:5xZ), then the prune of
optim@2000 → ~76G, ~34G at the step-3000 ENDPOINT — but ONLY IF that
post-2500-save tick executes its prune: missed, the endpoint trough
is ~3G (ENOSPC class, the leg-B failure aimed at the one save the
rung needs). Registry DISK anchor rewritten with the executed state
and the load-bearing flag. Host RAM available 45G — tenth read,
−2G vs the stable 47G plateau, trend-watch only. ETA ~4.7 h at the
window rate → done ~12:1xZ 08-23 → sim100 endpoint battery; its
verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): step-2000 save verified on disk (staging
watched live to publish, 07:29:18Z) + keep-1 optim prune executed
(optim@1500, +31G → 87G free), registry DISK anchor updated (troughs
~45/76/34G, step-2500 prune flagged load-bearing), babysit poll
(exit 0), probe @2000 new-high read judged (within-lineage,
non-actionable), loss flat-window resolved as noise (monotone
resumed), RAM read (45G tenth), Discord read + history, queue
validate green (depth-1 stated reason — rung3-exec verdict-gated, no
CPU items, so `run_work_next` stays unarmed), now.md keep-3 + footer
rolls (05:24 entry → archive), blog rebuild + Space upload
(post-save precedent).

**Next**: the step-2500 save lands ~09:5xZ — that post-save tick
verifies it and prunes optim@2000, and the prune is LOAD-BEARING
(missed → endpoint staging trough ~3G ENOSPC). Probes @2250/@2500
readable en route. Otherwise nothing fires before the ch0fix
boundary — train done ~12:1xZ 08-23 → battery ~3 GPU-h → rung-2
verdict banks → `carrier-hunt-rung3-exec` selects and launches the
branch same session (fit smoke → launch, ONE dataset delta, seed 0;
11–19 fires neither branch, owner escalation).*

## Utilization footer

Session 2026-08-23 08:49–08:5xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 2290/3000, loss 0.3074
monotone, vram 62.24/71, rate 15.58 s/step (util 100% this poll).
Probe @2250 = 8.63 — SECOND consecutive new high (…6.07 → 7.41 →
8.63), last four reads an upward run; within-lineage only, flagged
as verdict-session color. Disk 87G flat, checkpoint dir verified =
exact keep-1 state; RAM 47G twelfth read. Next save step 2500
~09:45Z (post this session's kill) → trough ~45G safe; the ~10:0x
tick's prune of optim@2000 stays LOAD-BEARING (missed → endpoint
~3G ENOSPC). ETA ~12:0xZ 08-23; queue depth-1 stated reason
(rung3-exec verdict-gated), no CPU items → `run_work_next` stays
unarmed.**

Session 2026-08-23 08:08–08:1xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 2140/3000, loss 0.3275
monotone, vram 62.24/71, rate 16.535 s/step. No new probe row
(@2000 = 7.41 latest, @2250 ~08:5xZ). Disk 87G flat since the
post-save prune; checkpoint dir verified = exact keep-1 state
(single optimizer.pt at step_002000). RAM 47G — plateau restored
(the 45G read was noise). Next save step 2500 ~09:5xZ → trough ~45G
safe; the post-save prune of optim@2000 stays LOAD-BEARING (missed
→ endpoint ~3G ENOSPC). ETA ~12:1xZ 08-23; queue depth-1 stated
reason (rung3-exec verdict-gated), no CPU items → `run_work_next`
stays unarmed.**

Session 2026-08-23 07:27–07:4xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **post-save tick executed — step-2000 save verified
(published 07:29:18Z, 42G complete; staging watched live, 56G trough
= projection exact) and keep-1 prune fired: optim@1500 deleted → 87G
free. Run healthy: step 2000/3000, loss 0.3356 (flat window resolved
as noise, monotone resumed), vram 62.24/71, ~16.8 s/step. Probe
@2000 = 7.41 — NEW series high (…5.55 → 6.07 → 7.41), within-lineage
only. RAM 45G tenth read (−2G, trend-watch). Remaining troughs
~45/76/34G with the step-2500 post-save prune now LOAD-BEARING
(missed → endpoint ~3G ENOSPC); registry anchor updated. ETA ~12:1xZ
08-23; queue depth-1 stated reason (rung3-exec verdict-gated), no
CPU items → `run_work_next` stays unarmed.**

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
