# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-23 09:31–09:5xZ (tick) — **routine ch0fix poll +
save-boundary hold: healthy — step 2440/3000, loss 0.3122 (+0.0048
since 2290 — first window uptick after the long monotone fall,
noise-class), vram 62.24/71, rate 16.489 s/step window (3.6
steps/min, util 87% sampled). No new probe row (@2250 = 8.63 latest;
@2500 lands with the save). Disk 87G flat, checkpoint dir verified
exact keep-1 state. RAM 47G thirteenth stable read. The step-2500
save lands ~09:48Z — INSIDE this session — so this tick holds at the
boundary to attempt the LOAD-BEARING optim@2000 prune itself;
fallback stays the ~10:0x tick. ETA ~2.6 h → ~12:1xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
2440/3000, babysit exit 0 (liveness 5 procs, vram 66641MiB on gpu0,
util 87% sampled). Loss 0.3074@2290 → 0.3122@2440 — +0.0048, the
first window uptick since the 06:4x flat window; same noise class,
watch next read. Rate 16.489 s/step window, inside the judged
14.7–24.5 band. Probe eval_chunk_mae unchanged: …6.07@1750 →
7.41@2000 → 8.63@2250 (the @2500 row lands with the save) — the
upward run stands at four reads, within-lineage-only per the banked
pdnorm-rescale confound, decision read stays the endpoint sim100
battery vs democlean 8/100. **Disk**: 87G free, flat. Checkpoint dir
verified by file listing: step_000500/1000/1500 weights-only,
step_002000 full with optimizer.pt (33.7G), no staging temp, no
step_002500 dir yet — exactly the keep-1 anchor. **Save-boundary
hold**: step 2500 lands ~09:48Z at the window rate, publish ~09:5xZ
— this session holds with sleep-polls to verify the publish and
execute the optim@2000 prune (LOAD-BEARING: missed → step-3000
endpoint staging trough ~3G, ENOSPC class). Prune fires ONLY on a
verified-complete publish (weights + optimizer.pt + metadata +
tokenizer, no staging temp); if staging is still mid-flight at
~09:55Z the session exits clean and the ~10:0x tick executes it per
the standing registry anchor. Host RAM available 47G — thirteenth
stable read. ETA ~2.6 h → done ~12:1xZ 08-23 → sim100 endpoint
battery; its verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), loss uptick judged
noise-class (first after long fall, next-read watch), disk read (87G
flat) + checkpoint dir verification (keep-1 state, single
optimizer.pt at step_002000, no staging temp), RAM read (47G,
thirteenth), Discord read + history, queue validate green (depth-1
stated reason — rung3-exec verdict-gated, no CPU items, so
`run_work_next` stays unarmed), now.md keep-3 + footer rolls (07:27
entry → archive), save-boundary hold attempted (outcome recorded in
the headline above).

**Next**: if this session's hold caught the publish, optim@2000 is
pruned and the ~10:0x tick reverts to a routine poll (verify only);
if not, the ~10:0x tick verifies + prunes — LOAD-BEARING either way
until executed (missed → endpoint staging trough ~3G ENOSPC). Probe
@2500 readable at the save, @2750 en route — watch whether the
upward run extends. Otherwise nothing fires before the ch0fix
boundary — train done ~12:1xZ 08-23 → battery ~3 GPU-h → rung-2
verdict banks → `carrier-hunt-rung3-exec` selects and launches the
branch same session (fit smoke → launch, ONE dataset delta, seed 0;
11–19 fires neither branch, owner escalation).*

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

## Utilization footer

Session 2026-08-23 09:31–09:5xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll + save-boundary hold, healthy — step
2440/3000, loss 0.3122 (+0.0048 first uptick after long fall,
noise-class), vram 62.24/71, rate 16.489 s/step. No new probe row
(@2250 = 8.63 latest, @2500 lands with the save). Disk 87G flat,
checkpoint dir verified = exact keep-1 state; RAM 47G thirteenth
read. Step-2500 save lands ~09:48Z INSIDE this session — held at the
boundary to attempt the LOAD-BEARING optim@2000 prune (verified
publish required; fallback ~10:0x tick). ETA ~12:1xZ 08-23; queue
depth-1 stated reason (rung3-exec verdict-gated), no CPU items →
`run_work_next` stays unarmed.**

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
