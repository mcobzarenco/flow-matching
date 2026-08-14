# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 21:17–21:3xZ (real `date -u` at stamp: 21:29) —
tick: **owner returned — credits topped up, GPU RELEASED, molmoact2
retirement COMPLETE on main; orphaned stage-0 hook recovered;
fontaine rebased onto `26ac1e6`.***

**Status**: **No live run** — GPU free at 0 MiB and **RELEASED**
(owner 21:14Z: "Your GPU is all yours"; the 12:54Z reserve is over).
Main at `26ac1e6` — molmoact2 retirement **ALL PHASES COMPLETE**
(phases 3–5 landed: objective matrix, `bijou/grpo_replay.py` re-point
+ replay-parity gate executed on my banked R1-A/R1-B waves with
receipts, `bijou/molmoact2/` deleted); fontaine rebased on top —
**zero conflicts**, 836 non-GPU green, pushed `64c93e6` (old tip
tagged `pre-rebase-26ac1e6`). Queue validate OK: depth 1, 16 open
(chained work session refills). Discord: inbox empty — both owner
messages replied + acked.

**Steering**: three-part (owner 21:13/21:14Z + the handoff
attachment): (1) **credits topped up** — the 19:17Z/20:22Z exit-1
harness alerts were the usage cap; (2) **GPU released**; (3) *"I'd
start by reviewing the new code from main after you rebase and let me
know your thoughts"* → queued **`main-review-molmoact2-final`** as
the top item. The handoff also binds: Decision 11 (any post-rebase
GRPO run is a FRESH pre-reg on the new stack, `.pt` resume
salvage-only), masked-only decode (old-side comparisons at tag
`pre-molmoact2-retirement`), full-width Gumbel sample streams.

**Done**: orphan recovery — the capped 18:59Z work session's
**stage-0 `--wrist-transform` hook** audited, lint+pyright fixed,
tests 11/11 + check.py 901 green, committed (both drivers + the W3
`wrist_arm_mask` path + oracles + spotcheck); **rebase onto
`26ac1e6`** (16 commits, zero conflicts); queue re-scoped
(`molmoact2-retirement-adoption` + `wrist-transfer-stage0-cpu-prep`
closed DONE, the main review queued, GPU release recorded on the
screen-run item); in-channel reply + both inbox ids acked.

**Next**: chained work session (`run_work_next` armed):
**`main-review-molmoact2-final` FIRST** (in-channel thoughts post,
parity-probe rerun on the banked waves, and the wrist-screen
checkpoint-surface verdict — the retirement re-pointed checkpoint
loading to bijou checkpoints, so the frozen ftrig4k/simft launch
surfaces must be verified or amended in-channel BEFORE stage 0), then
**`wrist-transfer-screen-run`** launches on the released GPU.
`renderer-pbr-wrist-pilot` stays owner-gated.*

*Updated 2026-08-14 18:57–19:0xZ (real `date -u` at stamp: 18:59) —
tick: **quiet — minutes after the prereg-final session closed; every
signal verified unchanged.***

**Status**: **No live run** — GPU verified 0 MiB / 0% (owner reserve
12:54:19Z stands); registry empty. Main unchanged at `e5b6113` (phase 3
not landed). Queue validate green: depth 2, 17 open. Discord: inbox
empty, no new messages, no new reactions in history (the 17:20Z 👍
remains the last steering).

**Steering**: none this tick.

**Done**: quiet tick — Discord read + history (nothing new; the
18:57Z pre-reg pointer sits as the channel tail), GPU/main/queue
verified, archive roll.

**Next**: unchanged — `molmoact2-retirement-adoption` phase-3 watch;
`wrist-transfer-stage0-cpu-prep` is the executable CPU item
(`run_work_next` already armed, the chained work session picks it
up); `wrist-transfer-screen-run` waits ONLY on the in-channel GPU
release; `renderer-pbr-wrist-pilot` stays owner-gated.*

*Updated 2026-08-14 18:47–19:0xZ (real `date -u` at stamp: 18:55) —
work session: **`wrist-transfer-screen-prereg-final` DONE — the
wrist-transfer screen is formally registered; the run item is now
GPU-release-only.***

**Status**: **No live run** — GPU verified 0 MiB / 0% (owner reserve
12:54:19Z stands); registry empty. Main unchanged at `e5b6113` (phase 3
not landed). Queue validate green: depth 2, 17 open. Discord: inbox
empty, no new messages.

**Steering**: none this session.

**Done**: **`wrist-transfer-screen-prereg-final`** (commit `77ab6b3`)
— FINAL pre-reg posted ([the
pre-reg](posts/2026-08-14-prereg-wrist-transfer-screen.md)): design
memo §5–§7 frozen **verbatim** (programmatically diffed
byte-identical), arm grid {`ftrig4k`, `simft`} × {W0..W4} + T1 frozen
with seeds 0–99 (T1 0–24), knn5 honesty anchors 0.877→0.523, ladder +
≤14 GPU-h gate, amendment policy (in-channel before the affected
stage, never retroactive). Design-memo schematic-caption erratum
fixed in place with a dated note ("≤12 gate" → ≤14; the §9 text was
always right). **`wrist-transfer-screen-run` is now
GPU-release-only** — the in-channel release is its single remaining
blocker. Queue refilled with **`wrist-transfer-stage0-cpu-prep`**
(the `--wrist-transform` hook + transform oracles + W3 mask path,
CPU-only under the reserve; the `none` bit-replay + honesty placement
stay GPU-gated in the run item).

**Next**: `queue_cli.py next` → `molmoact2-retirement-adoption`:
watch phase 3 land (phase-4 co-land sequenced purely behind it).
Executable CPU item: `wrist-transfer-stage0-cpu-prep` (`run_work_next`
armed); `wrist-transfer-screen-run` waits ONLY on the in-channel GPU
release; `renderer-pbr-wrist-pilot` stays owner-gated.*

## Utilization footer

Session 2026-08-14 21:17–21:3xZ (tick; 0 GPU-h — GPU released 21:14Z,
nothing launched pending the review): owner returned — credits topped
up (the 19:17Z/20:22Z exit-1 alerts were the cap), GPU released
in-channel, molmoact2 retirement ALL PHASES COMPLETE on main
`26ac1e6`; orphaned stage-0 `--wrist-transform` hook recovered from
the capped 18:59Z session (lint+pyright fixed, check.py 901 green,
committed); fontaine rebased onto `26ac1e6` zero-conflict (836 green,
pushed `64c93e6`, old tip tagged); queue re-scoped (two items closed
DONE, `main-review-molmoact2-final` queued at the owner ask, GPU
release recorded on the screen-run item), depth 1 with the chained
work session refilling; both owner messages replied + acked;
`run_work_next` armed for the review-first work session.

Session 2026-08-14 18:57–19:0xZ (tick; 0 GPU-h — GPU owner-reserved):
quiet tick minutes after the prereg-final session closed — Discord
read + history clean (no new messages or reactions; the 17:20Z 👍
remains the last steering, the 18:57Z pre-reg pointer is the channel
tail), GPU 0 MiB verified, main unchanged at `e5b6113` (phase 3 not
landed), queue validate green (depth 2, 17 open), inbox empty;
`run_work_next` already armed for the phase-3 watch +
`wrist-transfer-stage0-cpu-prep`.

Session 2026-08-14 18:47–19:0xZ (work; exploit; 0 GPU-h — GPU
owner-reserved, CPU-only writing task): `wrist-transfer-screen-prereg-final`
DONE — FINAL pre-reg posted freezing the design memo §5–§7 verbatim
(programmatically diffed byte-identical), arms/seeds/honesty-anchors/
≤14-GPU-h-gate frozen, amendment policy stated;
`wrist-transfer-screen-run` converted to GPU-release-only; design-memo
caption erratum fixed; queue refilled with
`wrist-transfer-stage0-cpu-prep` — validate green depth 2, 17 open;
`run_work_next` armed for the phase-3 watch + the stage-0 CPU prep.

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames), 3rd
launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3 rungs
(+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
23:24Z–02:37Z 08-08 **COMPLETE +~3.2 GPU-h (≤ 8 gate)**;
 08-08 daytime: local rung-(b) preflight+stage1
08:49–10:15Z **+~1.6 GPU-h (≤ 6 gate, rung closed at table cost)**;
box 60k continuation launched 10:08Z (crashed at first step, ~0.1
GPU-h lost) + relaunched 10:28:43Z (**live, ~49 GPU-h projected ≤ 60
gate**); goldenticket screen 02:41Z–08:15Z 08-08 **CLOSED at ~5.55 GPU-h ≤ 6
gate** (s1 ~1.7 + s2 ~0.85 + s3 2.99); box molmo2 chain: 40k train
to ~04:0xZ, greedy ~1.7 GPU-h, draws10_t1 04:54–07:22Z **~10 GPU-h
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box 60k continuation COMPLETE 08-08 ~23:4xZ
(~49 GPU-h ≤ 60 gate, chained evals incl.); local subgoal-swap arms
08-09 ~02:1x–03:42Z +~1.5 GPU-h ≤ 3 gate; box K-smoke ladder 08-09
04:02–04:39Z **+~0.5 GPU-h ≤ 6 gate (rung 1 GREEN first try)**; box
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + panel_v2
eval COMPLETE ~08:01Z (+~1.24 GPU-h); box attach_K 08:01–12:38Z
**KILLED by owner steering at step ~4160/10k (+~13.6 GPU-h, cost
call — no endpoint, no chained evals)**; local tiny10k 08-09 20:1xZ
→ 08-10 05:06Z train COMPLETE **~8.7/15 GPU-h incl. OOM replay** +
chained panel_v2 eval COMPLETE 08-10 05:45Z (+~0.6 GPU-h, **~9.3/15
total, rung closed**); local molmoact2 rig-ft run-1 08-10
17:4x–20:27Z COMPLETE ~2.7/12 GPU-h; local er35k owner-request evals
08-10 20:5x–00:41Z 08-11 ~2.2/8 GPU-h; local molmoact2 port parity
reads 08-10/11 ~0.7 GPU-h; local molmoact2_ae_ours (port item 4)
08-11 05:19–06:56Z **COMPLETE ~1.9/6 GPU-h (port total ~2.6/8)**).
Older dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).
