# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 02:42–03:4xZ (real `date -u` at write: 03:39) —
work session: **serving-norm audit DONE (the queue's gating item) —
sim100's token 0/100 was OUR serving bug (found + fixed, `b779ba4`);
flow 5/100 verified REAL model regression. 20-seed local proof:
token-with-fix 3/20 vs box 0/100; flow 0/20 replication.***

**Status**: no training run live; local GPU idle again after the two
20-seed audit legs (units norm-audit-{token,flow}, 02:42–03:27Z, ~1.1
GPU-h, strikes 0); box idle. Audit verdict: (1) TOKEN leg — inference
collator couldn't carry the merged action table (codec-required
guard), AR decode fell back to per-item quantiles = real-v2 row in
the sim harness while training tokenized under the recomputed merged
row; merged lift pair descending (+44.26→−124.8) vs v2 ascending ⇒
every token lift command decoded **sign-inverted**. Fixed
(`molmoact2_action_table` pinned family-gated in BijouPolicy, guard
removed, test added; checks green). (2) FLOW leg — table path audited
clean end-to-end (decoder-owned baked row empirically == metadata
merged after load; state clamp affine-consistent; box code
byte-identical to HEAD): **5/100 stands as a model result**.

**Steering**: none new this session (inbox empty at boot and at every
babysit poll; owner 👍 on sequencing already recorded 02:39Z).

**Done**: (a) box forensics — sim100 shard configs + code hashes
(both legs ran `stats_repo_id=so101_pick_place_v2` at 07f6de5, files
== local HEAD); (b) end-to-end table trace + empirical load check of
the banked endpoint (Hub download → local); (c) the bijou fix +
regression test, commit `b779ba4`; (d) 20-seed × 2-leg local re-run
(seeds 100–119, disjoint from box 0–99): **token 3/20 with the fix,
flow 0/20** — seam confirmed for token, parity confirmed for flow
(median final 8.9 vs box 8.7 cm); (e) queue: audit item DONE,
`sft-v1-flow-regression-isolation` queued (named suspect: pooled
table dilutes wrist_flex flow-MSE weight; discriminator = sim20 of
run-1b remap-only saves, no training); registry pruned; verdict
posted in-channel (1538754170457428018).

**Next**: `queue_cli.py next` → `wrist-cam-pose-refit`
(position-offset fit; on the regen's critical path), then boundary
results page + HTML with the corrected verdict, then
`grasp-demos-v2-regen` (pre-reg first) → `grasp-sft-v2-joint-run`
(recipe waits on the flow-isolation read). Owner-pending unchanged:
disk composite exemption 👍, approach redesign go, v2.1 bands,
ckpt-format, morning-veto items.*

*Updated 2026-08-17 02:39–02:4xZ (real `date -u` at write: 02:40) —
tick: **quiet close-out — no live runs (run 2 complete + banked,
sim100 verdict merged 02:3xZ last session), inbox clear; one NEW
signal: owner 👍 on the 01:35Z pipeline-sequencing post — sequencing
confirmed.***

**Status**: no training run live (registry `no_live_runs_reason`
02:0xZ stands); box idle after sim100, local GPU idle — both
idle-by-design pending the serving-norm audit. Boundary remainder
(results page + HTML report + consolidated post) and
`sft-v1-serving-norm-audit` (gates the regen→SFT-v2 pipeline) wait on
the chained work session — `run_work_next` armed.

**Steering**: owner 👍 (new since the 02:38Z close, caught via the
history check) on the 01:35Z post that laid out sim100-on-box + the
refit → 5k regen → SFT v2 sequencing — read as agreement with the
sequencing and the future-evals-run-local split; applied as-is, no
reply warranted for a bare agreement react. Inbox empty, no messages.

**Done**: routine tick — Discord read + history (reaction caught),
queue validate (OK depth 4, 25 open), registry/state check confirmed
no live runs, this entry + roll of the 08-16 entries/notes to
[archive](archive/now-2026-08-16.md).

**Next**: chained work session leads with `sft-v1-serving-norm-audit`
(decode-table provenance end-to-end + 20-seed local re-run with the
verified table — cheap, decisive; gates regen→SFT-v2), then boundary
page/HTML finalize, then `wrist-cam-pose-refit` position-offset fit.
Owner-pending unchanged: disk composite exemption 👍, approach
redesign go, v2.1 bands, ckpt-format, morning-veto items.*

*Updated 2026-08-16 22:43 → 08-17 02:4xZ (real `date -u` at first
write: 01:56) — work session: **run 2 COMPLETE at step 3000 (01:07Z,
~31/40 GPU-h, zero tracebacks) — endpoint banked to the Hub, sim100
sharded on the box; three owner steering threads executed live (wrist
refit prioritized + stages 1–3 first pass DONE on local, pipeline
order queued: 5k regen → SFT v2); `outputs/` audit freed 434 GB.***

**Status**: `grasp_sft_v1_joint_8xa100` run 2 **COMPLETE** 01:07:43Z
at 3000/3000 (~31 GPU-h wall×8 vs the 40 gate; final eval 5.41,
train 5.27). Checkpoint **banked + byte-verified**:
`fontaine-checkpoints/grasp_sft_v1_joint_step3000` (weights-only +
train_log). Eval curve oscillated all run: 4.05 → 4.54 → 3.74 → 5.00
→ 5.09 → **3.62 @1500** → **6.64 @1750** → 5.49 → 6.36 → 5.48 → 5.52
→ 5.41 @3000; per-dataset @3000: sim 5.06/4.99 (eval/train), **real
v2 15.77/8.15 — a 2× train/eval gap on real data** (the boundary
caveat, sharper than run-1b's rise). **sim100 VERDICT (merged 02:3xZ):
FLOW 5/100, TOKEN 0/100** vs anchors probe 44 / base 9 / corrupt 28 +
token bar ≥20 → pre-reg band flow <25 = **seam/serving investigation
FIRST, no bank**. Not read as raw model failure: flow moved the boat
51/100 (median final 8.7 cm from ~11 spawns) — reaches, cannot grasp;
prime suspect a serving-path norm mismatch on exactly the recomputed
wrist/lift channels → `sft-v1-serving-norm-audit` queued, GATES the
v2 pipeline (same flag). Page/HTML finalize on the next tick. Local
GPU idle after refit renders.

**Steering** (3 threads, all replied + acked, inbox clear): (1)
23:06Z wrist-cam status ask → full status reply; (2) 23:56Z
**"prioritise this work, use the local machine"** → refit stages 1–3
first pass executed same-session (below); (3) 00:45Z **pipeline
order**: after the run, regen 5k demos with all improvements + wrist
angle "definitely" in the new demos, then SFT v2 same hyperparameters;
sim100s on local going forward → queued `grasp-demos-v2-regen` +
`grasp-sft-v2-joint-run` (sequenced behind the refit), flagged that
this sim100 was already mid-flight on the box (finishes faster there,
box idle after for the regen).

**Done** (commits `bba830b`…`ee7f789` + close, checks green): (a)
**`outputs/` audit + prune 486 → 52 GB** (owner cleanup thread
closed: GRPO full-state .pts with Hub-verified weights-only endpoints,
measurement-run step dirs, optimizer.pt + intermediates of banked
runs; disk 239 → 673 GB free, report in-channel); (b) **wrist-cam
refit stages 1–3 first pass**: matched-pairs instrument
(`wrist_cam_matched_pairs.py`, 312 pairs replaying rig-v2 states into
the sim at identical kinematics, composite posted) + measurements
(per-pair jaw-angle discrepancy 63° mean, sim bottom-band occupancy
0.050 vs real 0.115) + rotation-only grid fit **measured insufficient**
(held-out 7.56 → 7.41; degenerate jaw-hiding first winner fixed by a
dominating visibility penalty) — registered next: position offsets +
glare-robust angle metric; (c) run 2 ridden end-to-end with trend
posts at 1000/1250-1500/1750; (d) boundary executed: final eval +
per-dataset table, Hub upload verified, sim100 launched (2 unit
relaunches — systemd-run cwd defaults to $HOME, cd/abs-path fix),
results page rewritten for run 2 (sim100 numbers pending); (e) queue
+2 owner-pipeline items, wrist refit item updated with stage-1 done.

**Next**: `queue_cli.py next` → finish the boundary (merge shards →
reads → sim100 verdict into the page + HTML report + consolidated
post) as soon as the shards land; then `wrist-cam-pose-refit`
(position-offset fit, leads next work session, on the regen's
critical path) → `grasp-demos-v2-regen` (pre-reg first) →
`grasp-sft-v2-joint-run`. `run_work_next` re-armed at close.
Owner-pending unchanged: disk composite exemption 👍, approach
redesign go, v2.1 bands, ckpt-format, morning-veto items.*

## Utilization footer

Session 2026-08-17 02:42–03:4xZ (work, exploit; local ~1.1 GPU-h —
two parallel 20-seed rollout legs 02:42–03:27Z on the shared H100;
box idle): **serving-norm audit closed same-session — token-leg
decode bug found/fixed/proven (0/100 → 3/20), flow regression
verified real (0/20 replication), fix + test + registry landed
`b779ba4`, isolation item queued** — queue depth 4, inbox clear,
`run_work_next` armed for the wrist refit.

Session 2026-08-17 02:39–02:4xZ (tick; GPUs idle by design, box +
local — no live runs): **quiet close-out — inbox clear, owner 👍 on
the 01:35Z sequencing post recorded (refit → 5k regen → SFT v2
confirmed, future sim100s local)** — queue depth 4, `run_work_next`
stays armed for the serving-norm audit + boundary page/HTML finalize.

Session 2026-08-16 22:43 → 08-17 02:4xZ (work, exploit; box: run-2
ride 22:43→01:07 complete ≈ +19.2 GPU-h in-window (run-2 total ~31/40
gate) + sim100 sharded eval ≈ +4–5 GPU-h to its ~02:2x end; local
~0.2 GPU-h refit renders/fit): **run 2 COMPLETE + endpoint banked
(final eval 5.41, real-v2 2× train/eval gap the headline caveat),
sim100 verdict in the tail or next tick; owner pipeline queued (5k
regen with fitted wrist pose → SFT v2 same hparams); wrist refit
stages 1–3 first pass on local (312 matched pairs, defect measured,
rotation-only fit an honest negative); outputs/ prune freed 434 GB**
— queue depth 4, inbox clear, `run_work_next` re-armed.

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
