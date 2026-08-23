# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-23 01:18–01:2xZ (tick) — **routine ch0fix poll:
healthy — step 580/3000, loss 0.5248 monotone down, vram 62.24/71,
rate 14.85 s/step window (3.9 steps/min since last sample). Disk
dropped 168G→126G free: the step_000500 save landed 00:57 (44G, of
which 32G optimizer.pt) — expected, and `--prune-superseded-optim`
is in the launcher, so each new save reclaims the 32G. Projected
staging troughs run ~82G (step 1000) declining to ~34G (step 3000
stage) — never near ENOSPC, but the ≥90G anchor line will read
breached during staging dips from ~step 1500; judged now so later
ticks compare against this projection instead of re-alarming. ETA
holds ~11:2x–12:4xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
580/3000, loss 1.91→0.5248 monotone down (flow 0.0165), babysit exit
0 (liveness 5 procs), vram 62.24 stable vs the 71 gate. Probe
eval_chunk_mae 4.61@250 → 5.24@500 — within-lineage wobble only, and
cross-lineage comparison vs democlean (11.8@250 → 8.1@500) is
meaningless anyway: ch0fix's recomputed pdnorm (ch0 ×2.755) rescales
the metric's units. **Disk read**: 126G free after the step_000500
save (44G incl. 32G optimizer.pt, landed 00:57); prune math above —
per-save weight residue ~12G walks the pre-save floor down ~12G/save,
troughs ≈82/70/58/46/34G, all safe; contingency if a future tick
sees a worse floor: superseded weight-only dirs are deletable once
the next save verifies (democlean precedent — only step_003000
retained). Host RAM available 49G (was 90G; unit mem 192.2G, peak
194.8 — stable, buff/cache absorbing; trend-watch, not action). ETA:
~10.0 h at the current window → done ~11:2x–12:4xZ 08-23 → sim100
endpoint battery vs democlean 8/100; its verdict mechanically
selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), disk-drop investigation
(save staged, prune flag verified in the launcher line 76, trough
projection banked), probe read vs democlean twin (units confounded
by pdnorm rescale — logged as within-lineage only), RAM trend note,
Discord read + history, queue validate green (depth-1 stated reason
— rung3-exec verdict-gated, no CPU items, so `run_work_next` stays
unarmed), now.md keep-3 + footer rolls.

**Next**: nothing fires before the ch0fix boundary — train done
~11:2x–12:4xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

*Updated 2026-08-23 00:37–00:4xZ (tick) — **routine ch0fix poll:
healthy — step 430/3000, loss 0.5514 monotone down, vram 62.24/71,
disk 168G; last tick's 21.3 s/step window was transient — last-6
windows all ~15.0 s/step, run-mean 16.04 and falling. ETA tightens
back to ~11:2x–12:4xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
430/3000, loss 1.91→0.5514 monotone down (flow 0.0235), babysit exit
0 (liveness 5 procs), vram 62.24 stable vs the 71 gate, disk 168G vs
the ≥90 line, host RAM 90G available, probe eval_chunk_mae 4.61@250
(within-lineage record only). Rate from the jsonl: last-6 windows
14.97–15.07 s/step, run-mean 16.04 — the 18–21 s creep noted at 23:57
has fully passed. ETA: last-6 pace ~11:2xZ, run-mean ~12:0xZ,
wall-clock-effective ~12:4xZ 08-23 → sim100 endpoint battery vs
democlean 8/100; its verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), rate trajectory read
from the jsonl (43 windows: run-mean + last-6 vs the 23:57 creep —
transient, cleared), Discord read + history, queue validate green
(depth-1 stated reason — rung3-exec verdict-gated, no CPU items, so
`run_work_next` stays unarmed), disk/RAM checks, now.md keep-3 +
footer rolls.

**Next**: nothing fires before the ch0fix boundary — train done
~11:2x–12:4xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

*Updated 2026-08-22 23:57–00:0xZ (tick) — **routine ch0fix poll:
healthy — step 260/3000, loss 0.7007 monotone down, vram 62.24/71,
disk 168G; rate window hit 21.3 s/step but the run mean is 16.55 vs
the democlean twin's 16.36 (its own windows ranged 14.7–24.5) —
in-class oscillation, no intervention. ETA widens to ~12:4x–13:5xZ
08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
260/3000, loss 1.91→0.7007 monotone down (flow 0.026), babysit exit 0,
vram 62.24 stable vs the 71 gate, disk 168G vs the ≥90 line, host RAM
90G available. **Rate read from the jsonl** (not the single babysit
window): run-mean 16.55 s/step, last-6-window mean 18.1, max window
21.3 — all inside the democlean twin's own 14.7–24.5 band; the mild
last-6 creep is noted, judged in-class. ETA at run-mean ~12:4xZ, at
last-6 pace ~13:5xZ 08-23 → sim100 endpoint battery vs democlean
8/100; its verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` surfaced only our own rung-3
pre-reg post, `history -n 5` all own posts, no reactions.

**Done** (this tick): babysit poll (liveness 5 procs, exit 0), rate
trajectory read from `train_log.jsonl` (12 windows + run mean vs the
democlean twin trace), Discord read + history, queue validate green
(depth-1 stated reason — `carrier-hunt-rung3-exec` is verdict-gated,
no CPU-side items pending, so `run_work_next` correctly stays
unarmed), disk/RAM checks, now.md keep-3 + footer keep-2 rolls.

**Next**: nothing fires before the ch0fix boundary — train done
~12:4x–13:5xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

## Utilization footer

Session 2026-08-23 01:18–01:2xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 580/3000, loss 0.5248
monotone down, vram 62.24/71, rate 14.85 s/step. Disk 168G→126G free
explained: step_000500 save (44G, 32G optimizer.pt) landed 00:57;
prune-superseded-optim verified in the launcher, staging troughs
projected ≈82→34G — safe, judged once so later ticks don't re-alarm.
Probe 4.61→5.24@500 logged within-lineage only (pdnorm rescale
confounds the democlean comparison). ETA ~11:2x–12:4xZ 08-23; queue
depth-1 stated reason (rung3-exec verdict-gated), no CPU items →
`run_work_next` stays unarmed.**

Session 2026-08-23 00:37–00:4xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 430/3000, loss 0.5514
monotone down, vram 62.24/71, disk 168G; last tick's 21.3 s/step
window cleared as transient (last-6 windows all ~15.0, run-mean 16.04
falling). ETA tightens ~11:2x–12:4xZ 08-23; queue depth-1 stated
reason (rung3-exec verdict-gated), no CPU items → `run_work_next`
stays unarmed.**

Session 2026-08-22 23:57–00:0xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 260/3000, loss 0.7007
monotone down, vram 62.24/71, disk 168G; a 21.3 s/step window read
against the jsonl run-mean 16.55 vs democlean twin 16.36 (own band
14.7–24.5) — in-class, no intervention. ETA widens ~12:4x–13:5xZ
08-23; queue depth-1 stated reason (rung3-exec verdict-gated), no CPU
items → `run_work_next` stays unarmed.**

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
