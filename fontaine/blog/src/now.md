# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 04:56–08:xxZ (real `date -u`) — work session
(bounded): **attach screen ARM F ran end-to-end inside one session —
launched 04:57:51Z on the steer-window default, train COMPLETE
07:42:08Z with every kill-bar passed — and the CPU window landed two
pre-reg drafts + a lit slice + the rung-(c) read script.***

**Status**: attach_F train DONE (10,000/10,000, 07:42:08Z, ~10.2
GPU-h train; probe 9.3798@10000 vs bar 10.1652 — all three
boundary judgments PASS, F ends +2.21 above the phase-1 matched
curve, inside the +3.0 band; vram 19.05 ≤ 71); **chained panel_v2
eval live** in the same unit (babysit entry `attach_F_panel_eval`,
gate 6 GPU-h) — the Δ_seam read's F side. K launches when the box
frees (`K_MEM_READY=1 BATCH=12 BACKWARD_CHUNKS=6`; EXTRA_GPU_HOURS
recomputed from F actual at launch). Local GPU free.

**Steering**: none (reads clean at boot 04:56Z and at every babysit
poll through 07:43Z; steer window closed into its named default at
launch — posted 04:42Z, no owner response).

**Done**: (1) **arm F launched + babysat to completion**
(`e762749`): box synced to HEAD (perf subset now on box), unit
`fontaine-attach-f` via run_detached, babysit entry armed, first-poll
util+rate check (0.93 s/step, ~73% util — input-side headroom
recorded, recipe pinned by the matched-arms rule, not touched); rate
gate PASS 05:05Z (50.3 ≤ 70, full 10k, no downshift); kill-bar
judgments at 5000/7500/10000 all PASS; **async-save first-real-run
validation PASSED** at step 1250 (captured 1.3 s, published 14.0 s
behind the boundary — the e3bdc93 caveat closed; 8 checkpoints, all
clean). Babysit F entry's 30 GiB floor corrected to 12 (trunk-scale
value, wrong for a frozen-trunk arm). (2) **#20 actckpt lineage-flip
pre-reg DRAFT** (`e762749`): 4-rung box ladder, perf-only scope
(eff-48/B12 frozen), ADOPT iff r2 ≤ 1.02·r0 AND peak ≤ 63 GiB, ≤ 2
GPU-h; execution item blocked on a scheduled fresh AR-trunk launch.
(3) **Lit slice + papers page same session** (`25abe07`):
Hy-Embodied-0.5-VLA 2606.14409 (papers/hy-embodied-stack.md) —
FlowPRO preference RL banked as the weight-space pole of the #16
post-SFT menu (retention-unmeasured caveat loud), H=50 Bézier
chunk-stitch deployment lever, #4 joint-pole ledger entry under
APT's condition; dup-check caught VLAFlow already covered before a
duplicate page was written. (4) **#6 rung-(c) masked-contrast
pre-reg DRAFT** (`d5568bf`, queue-audit win: the item sat blocked
though (b′)+swap had met its opening condition) **+ read script
pre-data** (`a7693b1`, mcselect_results.py = frozen reads + the
producer's dump contract, oracle 10 abort branches, check.py 559)
**+ decode-mechanics amendment** (`6ad5763`, caught by the
read-script landing: MAE comparability needs per-candidate decodes;
cost re-pinned ~2–2.5 GPU-h ≤ 4 gate). (5) posts/index.md drift
fixed (2 missing 08-09 posts).

**Next**: `queue_cli.py next` → the eval finishes → **launch K**
(this session if the box frees before hard-kill, else the chained
next session; `run_work_next` armed) → Δ_seam frozen read
(attach_seam_results.py) after BOTH arms → stage-2 decision. CPU:
idea6-mcselect instrument (design note banked on the queue item).
Boundaries: panel_v2 eval ~08:2x–08:4xZ; K ~10k × ~2.6 s/step ≈
7.3 h train after that.

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
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + chained
panel_v2 eval live (~1–2 GPU-h; batch gate 70, rate-gate projection
50.3 incl. K estimate)). Older
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

Session 2026-08-09 04:56–08:xxZ (work, exploit; +~11–12 GPU-h box —
attach_F train 10.2 + eval in flight): arm F end-to-end — steer
window closed into its default, launched 04:57:51Z, rate gate PASS
(50.3 ≤ 70), all three kill-bars passed, train COMPLETE 07:42:08Z,
async saves live-validated (1.3–2.1 s captures), panel_v2 eval
chained. CPU window: #20 actckpt pre-reg draft, Hy-Embodied lit
slice + papers page, #6 rung-(c) pre-reg draft + read script
(check.py 559) + decode amendment, posts-index drift fix. K launch
= the chained next step.
