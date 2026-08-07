# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 01:56–02:0xZ (real `date -u`) — tick (babysit).*

**Status** (babysit 01:56Z, both green, exit 0):
- box molmo2 AR 40k — 4120/40k, loss 4.07 (−0.056 this window), probe
  **10.47@4000** (holds the low; descent intact, K1 gate @10k with
  margin), 2.18 s/step, vram 67.07 ≤ 71, 4 ranks + 4 GPUs ~71.6 GiB;
  **@5000 save ~02:2x–02:3xZ → next tick's duty**, endpoint ~08-08.
- local draws10_t1 — 4192/25800, window 59.1 f/min, cumulative
  30.3 f/min → **~14.2 h total, INSIDE the 24 GPU-h gate**; boundary
  pulled in to ~13:5x–14:2xZ.

**Steering**: none (`read` clean, `history -n 5` no new reactions;
owner asleep since 00:58Z).

**Done**: tick only — babysit CLI exit 0 on both runs;
`queue_cli.py validate` green (depth 4, 9 open); GPUs busy ×5 +
owner-signed CPU queue → `run_work_next` armed.

**Next** (`queue_cli.py next`): p5-deadline-stamp (chained work
session), then p6 gpu markers, p7 tee-to-logs, #6 rung-(a) pre-reg
draft; molmo2 @5000 save ~02:3xZ next-tick duty; draws10_t1 boundary
~13:5x–14:2xZ → frozen reads; arm A img280 HELD.

*Updated 2026-08-07 01:47–02:0xZ (real `date -u`) — work session (chained, bounded):
**#21 P4 LANDED — this entry is the new contract** (commit `40e782f`).*

**Status** (babysit 01:47Z, both green, exit 0):
- box molmo2 AR 40k — 3900/40k, loss 4.11, probe **10.49@3500**
  (re-descended below the 12.09 low; K1 gate: below 12.0944 by 10k),
  2.18 s/step, vram 67.07 ≤ 71, 4 ranks + 4 GPUs 71.6 GiB; @5000
  save ~02:3xZ (tick duty), endpoint ~08-08.
- local draws10_t1 — 3872/25800, window 108 f/min (fast content
  stretch), cumulative 29.9 f/min → **~14.4 h total, INSIDE the 24
  GPU-h gate**; boundary pulled in to ~14:0x–14:3xZ.

**Steering**: none (owner asleep since 00:58Z; `read` + `history`
clean at boot and close).

**Done**: #21 P4 — the now.md head-entry skeleton, applied to the
file that defines it: entries are now four labeled blocks
(Status / Steering / Done / Next; contract in work.md §4, pointer in
tick.md §7, charter now.md bullet amended), the utilization footer
slimmed to trailing-7-day figure + last 2 session notes (286 stale
lines rolled verbatim to
[the archive](archive/now-2026-08-07.md)), and `archive_now.py
--keep 3` codified at every work-session close (was habit-only).
Queue hygiene in the same commit: p4 closed, molmo2 watch-title
cleared, draws10 boundary refreshed.

**Next** (`queue_cli.py next`): p5-deadline-stamp (driver stamps a
deadline the prompts can read, minutes), then p6 gpu markers, p7
tee-to-logs, #6 self-subgoal rung-(a) pre-reg draft; draws10_t1
boundary ~14:0x–14:3xZ → frozen reads (Δ_AR vs 5.8026, fairness vs
−1.258, family vs 5.365); molmo2 @5000 save ~02:3xZ tick duty; arm A
img280 HELD (fresh owner go required).

*Updated 2026-08-07 01:19–01:4xZ (real `date -u`) — work session (chained, bounded):
**#21 P2 LANDED — the queue is data now** (commit `19f3d71`).
`fontaine/queue.json` is the CANONICAL queue (now.md narrates it —
charter §3 bullet 1 amended per the signed diff);
`fontaine/scripts/queue_cli.py` `list`/`next`/`depth`/`validate`
machine-gates what ticks used to eyeball: depth ≥ 2 or a stated
`depth_reason`, every gpu-* item must name a pre-reg post that EXISTS
on disk, `owner_hold` forces `blocked` (a held item can never be
silently pickable), unique ids + schema. 8 oracles in
`tests/test_queue.py` incl. the real queue validating green; the
signed prompt diffs applied (tick §5 runs `validate`; work boot reads
`queue.json`, end gate requires validate green). Migration: 10 items
(2 live runs, 5 queued CPU: P4→P5→P6→P7→#6 pre-reg draft, 3 blocked:
frozen reads @ draws10 boundary, stage-2 decision @ molmo2 endpoint,
arm A img280 HELD). **One stated deviation from the signed spec: the
CLI is `queue_cli.py`, not `queue.py`** — sibling scripts
`sys.path.insert` the scripts dir, so a module named `queue` shadows
the stdlib; torch spawn children (`test_zero1`,
`test_chunk_grad_allreduce`) died on `from queue import Queue`
inside check.py — the gate caught it pre-commit, root cause traced
(not vibed as flaky), class fix = never stdlib-shadow in a
path-inserted dir. check.py 372 passed. BABYSIT (CLI, boot + close):
**molmo2 probe 10.49@3500 — RE-DESCENDED below the 12.09 low; the
@3000 wiggle (12.60) is resolved as noise, watch item closed**; step
3780/40k, loss 4.14 (−0.13 this window), 2.18 s/step, vram 67.07
(≤71), 4 ranks + 4 GPUs 71.5–71.7 GiB, ~21.9 h to 40k, @5000 save
~02:3xZ (tick duty). Local draws10_t1 3712/25800, window 35.1 f/min,
**cumulative 29.7 → projected ~14.5 h total, INSIDE the 24 GPU-h
gate; boundary pulled in to ~14:0x–14:2xZ**. Discord: no inbound at
boot, mid, or close (owner asleep). Queue (per `queue_cli.py next`):
**next (chained work session) → P4 head-entry skeleton, then P5
deadline stamp, P6 gpu markers, P7 tee-to-logs, #6 self-subgoal
rung-(a) pre-reg draft; draws10_t1 boundary ~14:0x–14:2xZ → frozen
reads (Δ_AR vs 5.8026, fairness vs −1.258, family vs 5.365) +
T-sensitivity rung; molmo2 @5000 save ~02:3xZ tick duty, K1 gate
@10k now with margin (10.49 < 12.09); arm A img280 HELD (fresh
owner go required).** GPUs busy ×5 + owner-signed CPU queue →
`run_work_next` armed.*

*Updated 2026-08-07 01:02–01:2xZ (real `date -u`) — work session (chained, bounded):
**#21 P3+P1 LANDED — the owner-signed queue head, both live-tested**
(commit `4c4fea8`). **P3**: repo pre-commit hook
(`fontaine/harness/hooks/pre-commit`, installed via `core.hooksPath`
in the driver) — code commits run `check.py` and its exit status IS
the gate (the 9f26f13 piped-exit-code class is closed); `*.md` /
`harness/state/` / `blog/book/` commits stay instant;
`FONTAINE_SKIP_CHECKS=1` escape hatch prints loudly. Live-tested all
three paths: a lint-failing commit BLOCKED, escape hatch lands,
md-only commit 0.01 s. **P1**: `fontaine/scripts/babysit.py` — one
command per checkpoint, built to the owner's three constraints:
liveness by pgrep + GPU-mem floor (never a log tail; exit 1 on a dead
rank), **trajectories not verdicts** (last-k probe values, loss delta
vs previous cached sample, window rate vs cumulative, anchors printed
alongside; an oracle asserts no verdict language in gate lines), gate
crossings SURFACED (exit 3) never acted on; the Discord poll runs
last and unconditionally — a checkpoint cannot skip it. Registry
`fontaine/harness/babysit.toml` (one entry per live run, updated at
launch); 13 oracles anchored to the hand-verified 00:59Z window
(37.2 f/min, 27.8 cumulative, 15.46 h projection); tick/work prompts
now point at the CLI. **The second live call earned its keep
immediately: probe 12.5951@3000 — the FIRST non-descending anchor**
(30.84@500 → 25.72 → 15.25 → 13.21 → 12.09@2500 → 12.60@3000).
Judgment (charter §6): single-anchor wiggle after a 60% descent,
loss delta +0.025 (log noise), K1 gate is @10k — watch @3500, no
action. BABYSIT (via the new CLI, twice): box molmo2 step 3060/40k,
loss 4.31, 2.18 s/step, vram 67.07 (≤71), 4 ranks + 4 GPUs at
71.4–71.7 GiB, ~22.4 h to 40k (endpoint ~08-08, @5000 save ~02:3xZ
tick duty). Local draws10_t1 2752/25800, window 49.5 f/min,
**cumulative 28.1 f/min → projected total ~15.3 h, INSIDE the 24
GPU-h gate; boundary ~14:5x–15:2xZ**. Discord: no inbound at boot,
mid, or close (owner asleep since 00:58Z); history check clean.
Queue: **next (chained work session) → P2 queue-as-data
(`fontaine/queue.json` + `queue.py validate`, ~1 session), then
P4–P7 in order (P4 head-entry skeleton, P5 deadline stamp, P6 gpu
markers, P7 tee-to-logs); #6 self-subgoal rung-(a) pre-reg draft
still banked (CPU; probe wants a quiet GPU ≥ draws10 boundary);
draws10_t1 boundary ~14:5x–15:2xZ → frozen reads (Δ_AR vs 5.8026,
fairness vs −1.258, family vs 5.365) + T-sensitivity rung after;
molmo2 probe @3500 is the watch item (first re-descend check), @5000
save ~02:3xZ; arm A img280 HELD (fresh owner go required).** GPUs
busy ×5 + owner-signed CPU queue → `run_work_next` armed.*

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 01:19–01:4xZ: all-CPU, 0 GPU-h — #21 P2 (owner-signed infra,
exploit-side): the queue became data (queue.json + queue_cli.py
validate), and the new check.py commit gate caught a real stdlib
shadowing bug in the first version before it landed. Lit slice
skipped — owner-signed P-block in progress, slice taken two sessions
ago as the work item (π0.5); balance on cadence.

Session 01:47–02:0xZ: all-CPU, 0 GPU-h — #21 P4 (owner-signed infra,
exploit-side): the now.md contract itself — head entries became the
four-block Status/Steering/Done/Next skeleton (this entry is the
exemplar), the footer slimmed to figure + last-2 session notes with
the stale mass rolled verbatim to the archive; archive_now.py
--keep 3 codified at every close. Lit slice skipped — owner-signed
P-block in progress (slice taken three sessions ago as the work
item, π0.5); balance on cadence.
