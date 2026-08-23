# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-22 23:14–00:0xZ (work) — **carrier-hunt rung 3
pre-registered with BOTH contingent branches frozen ahead of the
ch0fix verdict — the verdict session now executes instead of
drafting. Dataset names pinned by holdout-draw search; measured
basis: the ch0 compression is episode-UNIFORM, so branch B is a
frame-balanced bisection, not suspects-first LOO.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
240/3000, loss 1.91→0.707 monotone down, window rate 15.2–17.7 s/step
(twin class: democlean own-trace mean 16.36), vram 62.24 stable vs
the 71 gate, GPU 100% util at poll, babysit exit 0 both polls (23:19,
23:49). ETA unchanged ~12:0x–12:4xZ 08-23 → sim100 endpoint battery
vs democlean 8/100; its verdict mechanically selects the rung-3
branch.

**Steering**: none — inbox empty at boot and both babysit polls,
history all own posts.

**Done** (this session): **rung-3 pre-reg draft landed**
([pre-reg](posts/2026-08-22-prereg-carrier-hunt-rung3.md)) — branch A
(ch0fix ≥20): action-only ch0 affine cell (`clean_ch0fix_act_j`,
draw `(2,)`, train split episode-identical to democlean;
shift-vs-scale closed at zero cost by the banked constant-freeze
read, action-vs-state is the open axis, honesty clause on the
manufactured action/state inconsistency); branch B (≤10):
content-bisection cell training {0,1,5} (1504 frames, frame-balanced
vs {3,4,6} 1522) + decoy ep2 (`clean_ep015_c`, draw `(2,)` lands on
the decoy so the full subset trains; complement `clean_ep346_a` draw
`(0,)` = registered follow-up on EITHER verdict, dose confound
registered). Measured basis banked
(`carrier_rung3_basis_read.py` →
`reports/analysis__carrier_rung3_basis.json`): per-episode ch0 std
5.7–11.4 vs demos 28.0, KS 0.32–0.47 — uniform, no outlier episode;
also proves per-episode LOO could never have found a ch0-class
carrier. Queue: `carrier-hunt-rung3-prereg` done,
`carrier-hunt-rung3-exec` queued (verdict-gated, selection
mechanical); validate green (depth-1 stated reason). check.py PASSED
(1112).

**Next**: `queue_cli.py next` → `carrier-hunt-rung3-exec`, gated on
the ch0fix boundary: train done ~12:0x–12:4xZ 08-23 → battery ~3
GPU-h → rung-2 verdict banks → rung-3 branch fires same session (fit
smoke → launch, democlean launcher verbatim, ONE dataset delta, seed
0). The 11–19 band fires neither branch (owner escalation).*

*Updated 2026-08-22 23:10–23:2xZ (tick) — **routine ch0fix poll:
healthy — step 100/3000, loss 0.953 monotone down, vram 62.19/71,
disk 171G; the 0%-util duty cycle I caught mid-poll is a shared
recipe characteristic (democlean's own trace runs 16.36 s/step mean),
not starvation — no intervention, comparability preserved. ETA
refines to ~12:0x–12:3xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
100/3000, loss 1.91→0.953 monotone down (flow 0.033), per-window rate
oscillating 14.9–16.7 s/step, vram 62.19 stable vs the 71 gate, disk
171G vs the ≥90 line, host RAM 90G available. Babysit exit 0, both
gates green. **Rate/util judgment call**: nvidia-smi sampling showed a
~6 s-at-0% / ~9 s-at-100% duty cycle per step — checked against the
democlean twin's own train log before judging: democlean averaged
**16.36 s/step** (windows 14.7–24.5) under the identical launcher, so
ch0fix at 14.9–16.7 is running *slightly faster than its twin*; the
stall phase is the recipe's CPU-side step section, not input
starvation, and a mid-run dataloader change would break
recipe-verbatim comparability anyway (no-resume lineage → full
retrain). ETA at twin-class rate: done ~12:0x–12:3xZ 08-23 (a shade
later than the 11:2xZ first estimate) → sim100 battery vs democlean
8/100.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (liveness 5 procs, exit 0), util
duty-cycle investigation (multi-sample nvidia-smi + jsonl window
rates + democlean twin trace comparison → healthy verdict), Discord
read + history, queue validate green (depth 1 stated-reason, 14
open), RAM/disk checks, `run_work_next` confirmed armed (23:09),
now.md keep-3 + footer keep-2 rolls.

**Next**: chained work session owns `carrier-hunt-rung3-prereg`
(CPU, both contingent branches drafted ahead of the verdict).
Boundary: ch0fix done ~12:0x–12:3xZ 08-23 → sim100 endpoint battery
vs democlean 8/100; its verdict picks the rung-3 branch.*

## Utilization footer

Session 2026-08-22 23:57–00:0xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 260/3000, loss 0.7007
monotone down, vram 62.24/71, disk 168G; a 21.3 s/step window read
against the jsonl run-mean 16.55 vs democlean twin 16.36 (own band
14.7–24.5) — in-class, no intervention. ETA widens ~12:4x–13:5xZ
08-23; queue depth-1 stated reason (rung3-exec verdict-gated), no CPU
items → `run_work_next` stays unarmed.**

Session 2026-08-22 23:14–00:0xZ (work; exploit/pre-reg, 0 marginal
GPU-h — ch0fix riding gpu0 the whole session): **carrier-hunt rung-3
pre-reg landed with both contingent branches frozen ahead of the
ch0fix verdict (action-only affine cell / frame-balanced bisection
cell, dataset names pinned by holdout-draw search, dose confound +
honesty clauses registered); measured basis banked — ch0 compression
episode-uniform (std 5.7–11.4 vs demos 28.0). Two babysit polls
green (step 240/3000, loss 0.707, vram 62.24/71). Next boundary:
ch0fix done ~12:0x–12:4xZ 08-23 → battery → verdict selects the
branch.**

Session 2026-08-22 23:10–23:2xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 100/3000, loss 0.953
monotone down, vram 62.19/71, disk 171G; investigated a 0%-util duty
cycle mid-step and cleared it against the democlean twin's own trace
(16.36 s/step mean — ch0fix at 14.9–16.7 windows is the faster twin),
no intervention. ETA refined ~12:0x–12:3xZ 08-23; `run_work_next`
armed for the rung-3 pre-reg draft.**

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
