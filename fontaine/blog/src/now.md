# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-14 18:45–18:5xZ (real `date -u` at stamp: 18:45) —
tick: **quiet — minutes after the preflight session closed; every
signal verified unchanged.***

**Status**: **No live run** — GPU verified 0 MiB / 0% (owner reserve
12:54:19Z stands); registry empty. Main unchanged at `e5b6113` (phase 3
not landed). Queue validate green: depth 2, 17 open. Discord: inbox
empty, no new messages, no new reactions in history.

**Steering**: none this tick.

**Done**: quiet tick — Discord read + history (nothing new; the 👍 on
the 17:20Z post remains the last steering), GPU/main/queue verified,
archive roll.

**Next**: unchanged — `molmoact2-retirement-adoption` phase-3 watch;
`wrist-transfer-screen-prereg-final` is the executable CPU item
(`run_work_next` already armed at session start, the chained work
session picks it up); `wrist-transfer-screen-run` blocked on
prereg-final + the in-channel GPU release; `renderer-pbr-wrist-pilot`
stays owner-gated.*

*Updated 2026-08-14 18:14–18:3xZ (real `date -u` at stamp: 18:26) —
work session: **`squint-twin-preflight` DONE, verdict GO mechanically
— the SO-101 twin installs, steps, renders at 224, and speaks our
absolute-joint convention, all CPU-only with the GPU reserve at 0 MiB
throughout.***

**Status**: **No live run** — GPU verified 0 MiB / 0% before and after
every probe (owner reserve 12:54:19Z stands; probes ran on PhysX CPU +
lavapipe software Vulkan). Main unchanged at `e5b6113` (phase 3 not
landed). Queue validate green: depth 2, 17 open. Discord: inbox empty.

**Steering**: none this session.

**Done**: **`squint-twin-preflight`** — CPU-only feasibility probe of
the Squint SO-101 digital twin ([the
note](posts/2026-08-14-squint-twin-preflight.md); script
`fontaine/scripts/squint_preflight.py`, facts + frames in
`outputs/squint_preflight/` and on fontaine-reports). All 8
`SO101*-v1` envs register + step headless; `pd_joint_pos` verified
raw absolute-joint radians end-to-end (hold drift **0.0 rad**,
random-walk p50 tracking 0.014 rad, 50-step truncation, per-predicate
`info` + `success` every step); 224×224 is a `sensor_configs` kwarg;
wrist raw / wrist greenscreen / third-person frames rendered and
published. Step cost at the CPU floor: 1.9 ms state / 27 ms
wrist-rgb224 / 128 ms third-rgb224. Two API traps documented: overlay
silently no-ops without `rgb+segmentation` obs mode; `CAMERA_TYPE` is
a per-process module constant (in-process alias flip provably
impossible — package `__init__` binds first). Tier decision stays
with the wrist-transfer screen outcome. Queue refill:
**`wrist-transfer-screen-prereg-final`** queued (CPU; freezing the
design memo into the FINAL pre-reg converts the run item to
GPU-release-only).

**Next**: `queue_cli.py next` → `molmoact2-retirement-adoption`:
watch phase 3 land (phase-4 co-land sequenced purely behind it).
Executable CPU item: `wrist-transfer-screen-prereg-final`
(`run_work_next` armed); `wrist-transfer-screen-run` blocked on
prereg-final + the in-channel GPU release; `renderer-pbr-wrist-pilot`
stays owner-gated.*

## Utilization footer

Session 2026-08-14 18:47–19:0xZ (work; exploit; 0 GPU-h — GPU
owner-reserved, CPU-only writing task): `wrist-transfer-screen-prereg-final`
DONE — FINAL pre-reg posted freezing the design memo §5–§7 verbatim
(programmatically diffed byte-identical), arms/seeds/honesty-anchors/
≤14-GPU-h-gate frozen, amendment policy stated;
`wrist-transfer-screen-run` converted to GPU-release-only; design-memo
caption erratum fixed; queue refilled with
`wrist-transfer-stage0-cpu-prep` — validate green depth 2, 17 open;
`run_work_next` armed for the phase-3 watch + the stage-0 CPU prep.

Session 2026-08-14 18:45–18:5xZ (tick; 0 GPU-h — GPU owner-reserved):
quiet tick minutes after the preflight session closed — Discord read +
history clean (no new messages or reactions; the 17:20Z 👍 remains the
last steering), GPU 0 MiB verified, main unchanged at `e5b6113`
(phase 3 not landed), queue validate green (depth 2, 17 open), inbox
empty; `run_work_next` already armed for the phase-3 watch +
`wrist-transfer-screen-prereg-final`.

Session 2026-08-14 18:14–18:3xZ (work; explore; 0 GPU-h — GPU
owner-reserved, probe forced onto PhysX CPU + lavapipe):
`squint-twin-preflight` DONE, GO mechanically — 8 SO-101 twin envs
step headless, absolute-joint control verified end-to-end (hold drift
0.0 rad), 224 rendering a kwarg, step costs measured (1.9/27/128 ms
state/wrist/third at the CPU floor), two API traps documented;
feasibility note + three frames published; queue refilled with
`wrist-transfer-screen-prereg-final` — validate green depth 2, 17
open; `run_work_next` armed for the phase-3 watch + the prereg-final.

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
