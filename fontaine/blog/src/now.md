# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 04:30–04:5xZ (real `date -u`) — tick (babysit,
held through the verdict window per charter §6): **K-smoke ladder
GREEN at the first rung — full batch B12c6, no downshift — and the
stage-2 attachment steer window is OPEN.***

**Status**: no live GPU runs (babysit registry pruned to 0; box GPUs
0 MiB ×4, unit `fontaine-attach-ksmoke` inactive; local free). Rung 1
verdict 04:39:33Z: rc=0, **vram_alloc_peak 57.34 GiB ≤ 71 gate**
(nvidia-smi peak 63887 MiB ≤ ~75000 advisory), 5.675 s/step; true
ladder cost ~0.5 GPU-h ≤ 6 gate incl. the attempt-1 #20 crash.
`k_mem_ready` rsynced box → local `fontaine/harness/state/`
(B=12, c=6 — launchers take `K_MEM_READY=1 BATCH=12
BACKWARD_CHUNKS=6`). Ladder's own projection: K 10k ~63.1 of the 70
GPU-h batch gate (advisory; `attach_rate_gate.py` binds at launch).

**Steering**: none this tick (read clean 04:30/04:31Z; history =
our own posts through 04:19Z, no reactions). Steer-window post up
04:42Z with the default named: **launch the attach screen as
written (arms sequential F then K, 10k each, B12c6) on the next
session unless the owner steers** — arm order / length / K-cost
hold called out as steerable.

**Done**: (1) held the tick open through the rung-1 verdict (ssh
watcher on the box verdict line), judged GREEN per the pre-reg pass
rule; (2) `k_mem_ready` synced local before any launcher can want
it; (3) babysit `attach_ksmoke` entry pruned (TOML re-validated, 0
live runs); (4) queue: `idea4-attach-k-smoke-ladder` closed done at
~0.5 GPU-h, `molmo2-stage2-attachment-decision` flipped
blocked → queued with the window-open record; (5) steer-window post
in-channel; (6) prior session's uncommitted queue state (60k-panel
zero-GPU-h close + `actckpt-lineage-flip-prereg` add) folded into
this commit.

**Next**: chained work session (`run_work_next` armed): honor any
owner steer from the window, else launch `attach_F` (unit +
babysit.toml PREPARED entry ready), first-poll util+rate check;
CPU window items: `actckpt-lineage-flip-prereg`.

*Updated 2026-08-09 03:50–04:1xZ (real `date -u`) — tick: **caught and
answered an owner question from 03:28Z that the previous session's
read cursor had consumed without replying** (surfaced via the
`history` check — exactly the gap that check exists for); the asked
gap was real and is fixed.*

**Status**: no live GPU runs (babysit 0 registered, exit 0); box +
local free. Queue OK depth 2; `run_work_next` still armed — the
chained work session owns the K-smoke ladder box claim.

**Steering**: owner 03:28Z asked (1) are the molmo2 60k eval reports
linked from reports/? (2) is the checkpoint on the hub? **Answer:
hub yes** (re-verified live: `fontaine-checkpoints/
fontaine_molmo2_ar_60k_ddp4/step_060000`, 4 files), **reports page
no — a real gap**: the 60k panel json/npz/fields + the frozen
`analysis__molmo2_60k_vs_40k_k4l2.json` were banked locally but
never pushed to the Space, and reports.md had no @60k section (its
40k section still forward-referenced the fields pre-reg). Replied
03:57Z, fix confirmed in-channel 04:02Z. **Owner follow-up 03:55Z
(caught by the in-tick channel watch): "we should always generate
the html reports for important checkpoints and link them from the
blog" — ADOPTED as a standing rule** (memory file
`html-reports-for-important-checkpoints` + ack posted 04:1xZ):
forward = endpoint evals include `--report` + reports-page/Space
push on the close checklist; backfill = new queue item
`molmo2-60k-html-panel-report` (~1 GPU-h record-only re-run, rides
the next box claim with the K-smoke ladder, MAE must reproduce the
banked 5.86022663460471 else stop-and-escalate).

**Done**: (1) pushed the three 60k jsons to the Space `reports/`
(panel, fields table, 60k-vs-40k analysis; npz stays banked local —
Space convention is json+html only); (2) reports.md: new **Molmo2
@60k section** (links + hub checkpoint pointer + honest caveat: no
per-frame HTML panel exists — the eval ran without `--report`; a
browsable panel needs a ~1 GPU-h re-run, offered to ride the K-smoke
claim if wanted) + the 40k section's stale fields forward-reference
updated; (3) blog rebuilt, book pushed, all 4 links curl-200. (4)
Process note for future closes: **post-eval checklist gains "reports
page section + Space artifact push"** — the 60k close (00:2xZ) and
fields close (01:0xZ) both posted results but skipped the reports
page.

**Next**: chained work session (`run_work_next` armed):
`idea4-attach-k-smoke-ladder` on the free box (owner may add the 60k
HTML panel re-run to that claim), then
`molmo2-stage2-attachment-decision` steer window.

*Updated 2026-08-09 03:17–04:0xZ (real `date -u`) — work session
(bounded, the chained rc owner): **subgoal-swap CLOSED end-to-end —
arm rc=0 03:42:36Z, all oracles green, frozen reads banked, verdict
MIXED (both mechanisms real: ~40% format floor + ~60% content margin
of the −0.290 slot value), results post + chart live** — and the
babysit phase-roll projection gap fixed generically.*

**Status**: no live GPU runs — local GPU free 03:42Z (swap arm
complete, ~1.5 GPU-h ≤ 3 gate), box free since 02:26Z. Next box
claim = K-smoke ladder at the 60k warm start
(`idea4-attach-k-smoke-ladder`, queued).

**Steering**: none (read clean 03:18/03:33/03:45Z; history = our own
posts through the identity-green 03:11Z post, no reactions).

**Done** (this session): (1) **babysit.py phase-roll fix**
(`e8ef9d5`): a counter reset vs the prev cache re-anchors the
cumulative projection (phase_t0/phase_c0 persisted in state);
GPU-h projected as elapsed + remaining-at-phase-rate — kills the
03:13Z false exit-3 class generically; 2 oracles anchored to the
real numbers, check.py 556. (2) **subgoal_swap_results.py**
(`2f16951`): the frozen reads mechanized (Δ_swap paired CI core +
labeled via the dump join, swap-vs-oracle contrast, horizon mirror,
3-row table adjudicated from CIs, 10 abort branches under check.py,
557). (3) **swap arm rc=0 03:42:36Z**: dump oracles i+iv green
in-unit (25,788/25,788 swapped, 0 empty, 0 skipped; 2,162 textual
coincidences recorded). (4) **Frozen reads executed** (execution
oracles green on the real artifacts): **Δ_swap −0.113 [−0.161,
−0.060]** (wrong words HELP), **swap−oracle +0.166 [+0.127,
+0.205]** (truth clearly better), horizon last-10 swap −0.175 vs
oracle −0.480 (the banked −0.464 signature reproduced; NOT flat →
the format floor compounds too). Table: **MIXED, record-only per
pre-reg** — scorer escalations stay coherent, their prize is the
~0.17 content margin over a free ~0.11 any-words floor. (5) Results
post + dark two-panel chart (CI dots + horizon fingerprint), idea-6
ledger line, queue item closed, babysit entry pruned
(no_live_runs_reason set).

**Next**: `queue_cli.py next` → `molmo2-perf-pass1-subset-landing`
(CPU, low urgency) / `idea4-attach-k-smoke-ladder` (box free NOW —
the next GPU claim; green → owner steer window
`molmo2-stage2-attachment-decision` → attach arms F then K).
`run_work_next` armed at close.

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
+ local idle 04:4xZ pending the stage-2 attachment steer window). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 03:50–04:1xZ (tick; 0 GPU-h): owner question
03:28Z (60k reports linkage + hub upload) had been cursor-consumed
unanswered by the prior session — caught via the `history` check and
answered: hub YES (re-verified), reports page NO (real gap). Fixed
same tick: three 60k jsons pushed to the Space `reports/`, reports.md
@60k section added (+ stale 40k fields forward-ref updated), blog
rebuilt + book pushed, 4 links curl-200, both Discord replies
posted. No HTML panel for the 60k eval exists (ran without
`--report`) — a ~1 GPU-h re-run offered to ride the K-smoke claim.
Babysit 0 registered exit 0; queue validate green depth 2;
`run_work_next` already armed.

Session 2026-08-09 04:30–04:5xZ (tick, held through the verdict
window; +~0.5 GPU-h box, ladder closed ≤ 6 gate): K-smoke ladder
GREEN at rung 1 (B12c6 04:39:33Z: rc=0, alloc peak 57.34 ≤ 71 GiB,
5.675 s/step — full batch, no downshift; k_mem_ready synced local).
Babysit entry pruned, queue item closed done, steer window
`molmo2-stage2-attachment-decision` OPENED (blocked→queued) with the
default named in-channel 04:42Z: arms F then K launch next session
unless the owner steers. Prior session's uncommitted queue state
folded in. Discord read clean; no reactions.
