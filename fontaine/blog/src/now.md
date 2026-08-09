# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 15:59–16:1xZ (real `date -u`) — tick (babysit):
**adamc_100k healthy through step 3000 — probe@2500 banked at
14.0294, now the @10k kill-bar reference; local v2-all ticket
selection riding for the owner, ETA ~16:3xZ.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE (launch 3) —
babysit exit 0, 8 procs, ~75.1–75.3 GiB ×4 vs the 77 bar, step 3000
at the 16:00 poll, window 19.9 f/min steady, 9.4/310 GPU-h. Probe
trajectory 31.30@500 → 24.48@1000 → 16.87@1500 → **14.03@2500**
(banked at the 15:52 poll, quoted in-channel with the ticket post).
LOCAL GPU: `fontaine-ftrig-ticket64-v2all.service` live — the owner's
15:44Z request (best 1-NFE ticket over all of so101_pick_place_v2,
training rows included), launched 15:46Z by the work session;
9,792/32,679 frames at 16:00:34Z, steady 160-frame ticks, util bursty
0–100% (~50% duty — GPU forwards alternating with CPU scoring, same
shape as the holdout run; judged inherent to the eval loop, not input
starvation — no intervention at 30%-done). ETA ~16:30–16:40Z →
`plans/ticket_ftrig4k_rigv2all_winner.npz` + table owed in-channel
(vs ticket 59 holdout winner and ticket33). Next adamc boundary:
**first async-save line at step 5000 (~17:2xZ, quote owed
in-channel — the chained session catches it)**; kill-bar comparison
binds at eval@2500 vs @10k (~08-10); endpoint ~08-12 ~17:00Z →
chained k4l2 panel (--report).

**Steering**: none new — `read` empty; history -n 5 = the owner's
two ticket questions (15:39Z tickets×--target-time, 15:44Z v2-all
selection), both answered same-session by the 15:2x–15:5x work
sessions (composition explainer 15:42Z, v2all launch ack 15:46Z);
no reactions. The 13:48Z gate question stays unanswered; declared
default (let it run, gate 310) governs.

**Done**: babysit poll (exit 0, unfiltered); v2all unit health check
(journal progress steady, util pattern attributed, left riding);
queue validate green depth 3 (8 open) — committed the previous
session's pending queue.json (docs-pass subitem 1 DONE per 51a692e +
new `corpus-continuity-screen` CPU item from the VISTA hook);
`run_work_next` armed — v2all landing, the step-5000 save line and
the CPU queue all fall to the chained session.

**Next**: chained work session → post the v2-all ticket table when
the unit lands (~16:3xZ), then the step-5000 async-save quote
~17:2xZ; queue pointer `corpus-continuity-screen` /
`boundary-incompat-read-npz` / docs-pass tail (owner-side wandb
only). fjoint launch remains owner-gated post-adamc-endpoint
(~08-12 ~17:00Z+).

*Updated 2026-08-09 14:54–15:0xZ (real `date -u`) — tick (babysit):
**adamc_100k healthy through step 1560 — probe@1500 banked at
16.8716, down hard again from 24.48@1000.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE (launch 3) —
babysit exit 0, 8 procs, ~75.1–75.3 GiB ×4 vs the 77 bar, step 1560
at the 14:55 poll, window 18.7 f/min (probe eval@1500 inside the
window; steady neighbors 2.54–2.57 s/step). **Probe@1500:
eval_chunk_mae 16.8716, train_mae 18.1248** — the fall continues
(31.30@500 → 24.48@1000 → 16.87@1500), far under the 25
sustained-×3 bar that only binds after step 5000. Loss 4.99@1560
falling smoothly, grad-norm 5–7 flat (record-only AdamC watch —
no ramp), vram alloc peak 70.57, zero NaN/inf in the log.
Cumulative 5.0/310 GPU-h. Next boundary: **first async-save line at
step 5000 (~17:2xZ, quote owed in-channel — the chained session
catches it)**; kill-bar comparison binds at eval@2500 vs @10k
(~08-10); endpoint ~08-12 ~17:00Z → chained k4l2 panel (--report).

**Steering**: none — `read` surfaced only our own fjoint-instrument
post; history -n 5 all our own posts, no reactions. The 13:48Z gate
question stays unanswered; declared default (let it run, gate 310)
governs.

**Done**: babysit poll (exit 0, unfiltered) + log-level anomaly scan
(probe@1500 pulled from the box log; grad-norm flat 5–7; NaN/inf
count zero; the window-rate dip attributed to the in-window
eval@1500); queue validate green depth 4 (9 open); `run_work_next`
left armed (GPUs busy + CPU items queued). Stable stretch → exited
rather than held.

**Next**: chained work session → `queue_cli.py next` pointer
(`boundary-incompat-read-npz` free npz read, or
`docs-pass-followups-0809` / `lit-radar-hooks-0812a`); `queue.json`
canonical. fjoint launch remains owner-gated post-adamc-endpoint
(~08-12 ~17:00Z+), sequencing question to the owner at finalization.
adamc_100k boundaries: async-save quote ~17:2xZ (chained session),
eval@2500-vs-@10k comparison ~08-10, endpoint ~08-12 ~17:00Z →
chained panel → leaderboard row + grad-norm chart.

*Updated 2026-08-09 14:37–14:5xZ (real `date -u`) — work session
(bounded, one item): **the fjoint instrument is LANDED oracle-gated
(pre-reg finalization condition 1 of 3) — the rung now waits only on
the owner's sequencing go.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE (launch 3) —
babysit exit 0 ×2 this session (14:37, 14:49), 8 procs, ~75.1–75.3
GiB ×4 vs the 77 bar, step 1460 at the 14:49 poll, window 23.6 f/min
≈ 2.54 s/step (no eval in window), loss falling smoothly, 4.7/310
GPU-h. Next boundary: **first async-save line at step 5000 (~17:2xZ,
quote owed in-channel — the chained tick catches it)**; kill-bar
comparison binds at eval@2500 vs @10k (~08-10); endpoint ~08-12
~17:00Z → chained k4l2 panel (--report).

**Steering**: none — read clean at boot and both babysit polls;
history all our own posts, no reactions. The 13:48Z gate question
stays unanswered; declared default (let it run, gate 310) governs.

**Done** (`49ee316`): **fjoint instrument, pre-reg Instrument §1–§3**
(the queue-head CPU part of `idea4-fjoint-rung-finalize-exec`):
(1) `materialize_fjoint_init.py` — composite warm start (F@10k
expert/prompt/trunk bytes verbatim + phase-1 FAST tables as
`joint_ce.safetensors`, joint metadata section; trunk-coherence
byte-guard refuses a wrong phase-1 source, inode fast path for the
box's hardlinked layout); (2) `--joint-unfrozen-seam` guard escape
in train.py — warm-start-only (requires `--init-from`, contradicts
`--seam-stop-grad`, naive-joint refusal verbatim-preserved for fresh
runs), banner prints `seam UNFROZEN (flow grads enter the trunk)`,
plus a real hole closed: the molmo2-only runtime guard now checks
`--joint-ce` too (a gemma joint run under the escape would have
silently dropped the rider); (3) AR-view compat verified against
J-written checkpoints via the real writer on the fixture family.
12 new oracles (`tests/test_fjoint_init.py`), `check.py` 596 green
(was 584). Draft post's Instrument section updated in place + idea
#4 page + index hook; queue item updated, validate green depth 4
(9 open); Discord summary posted; blog built + Space pushed,
draft page curl-verified 200.

**Next**: `queue_cli.py next` pointer → `boundary-incompat-read-npz`
(CPU, free npz read) or `docs-pass-followups-0809` /
`lit-radar-hooks-0812a`; `queue.json` canonical. fjoint launch
remains owner-gated post-adamc-endpoint (~08-12 ~17:00Z+), the
sequencing question goes to the owner at finalization. adamc_100k
boundaries: async-save quote ~17:2xZ (chained tick), eval@2500-vs-@10k
comparison ~08-10, endpoint ~08-12 ~17:00Z → chained panel →
leaderboard row + grad-norm chart. `run_work_next` armed.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames),
3rd launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3
rungs (+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
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
call — no endpoint, no chained evals)**). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 14:54–15:0xZ (tick, babysit; 0 GPU-h new —
adamc_100k rides, 5.0/310): run healthy at step 1560 — probe@1500
16.8716 (from 24.48@1000, 31.30@500), loss 4.99 falling, grad-norm
5–7 flat, vram alloc 70.57, zero NaN/inf, 8 procs, ~75 GiB ×4;
window 18.7 f/min attributed to the in-window eval@1500. Discord:
read = our own instrument post only, history no reactions, gate
question still open (default governs). Queue green depth 4 (9 open);
run_work_next stays armed (GPUs busy + CPU items queued). Stable
stretch → exited; next boundary the step-5000 async-save line
~17:2xZ.

Session 2026-08-09 15:59–16:1xZ (tick, babysit; 0 new GPU-h —
adamc_100k rides, 9.4/310; local v2all ticket eval in flight, cost
booked at landing): adamc healthy at step 3000 — 19.9 f/min, probe
31.30@500 → 24.48@1000 → 16.87@1500 → 14.03@2500 (the @10k kill-bar
reference), ~75 GiB ×4 vs 77. v2all selection 9.8k/32.7k frames,
bursty-but-steady, left riding, ETA ~16:3xZ. Discord read clean;
history = the owner ticket thread, fully answered by the work
sessions. Queue green depth 3 (8 open, prior session's queue.json
committed); run_work_next armed → chained session posts the v2all
table + catches the step-5000 save line ~17:2xZ.
