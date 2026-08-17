# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 03:43–03:4xZ (real `date -u` at write: 03:45) —
tick: **quiet tick — no live runs (local + box idle by design; the
13 GiB on the local H100 is the owner's policy-server process, not
ours), inbox clear, no new messages or reactions since the 03:39Z
audit-verdict post.***

**Status**: no training run live; local GPU idle (owner policy-server
holds ~13 GiB at 0% util — left alone), box idle awaiting the regen.
Serving-norm audit closed last session (`b779ba4`): token 0/100 was
our decode bug (fixed + proven 3/20), flow 5/100 verified real —
`sft-v1-flow-regression-isolation` queued as the cheap discriminator
before SFT-v2 recipes lock.

**Steering**: none new (inbox empty, `read` empty; history check —
no reactions yet on the 02:35/02:38/03:39Z posts; the 01:35Z 👍
already recorded).

**Done**: routine tick — Discord read + history, queue validate (OK
depth 4, 25 open, updated 03:38Z), GPU/unit check (no fontaine units,
policy-server identified as the memory holder), `run_work_next`
confirmed armed, 08-16 entry + 02:39Z tick entry rolled to the
archive ([08-16](archive/now-2026-08-16.md),
[08-17](archive/now-2026-08-17.md)).

**Next**: chained work session per queue order —
`wrist-cam-pose-refit` (position-offset fit; on the regen's critical
path), boundary results page + HTML with the corrected sim100
verdict, `sft-v1-flow-regression-isolation` (run-1b remap-only sim20
discriminator), then `grasp-demos-v2-regen` pre-reg →
`grasp-sft-v2-joint-run`. Owner-pending unchanged: disk composite
exemption 👍, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

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

## Utilization footer

Session 2026-08-17 03:43–03:4xZ (tick; GPUs idle by design, box +
local — no live runs; local 13 GiB = owner policy-server, not ours):
**quiet tick — inbox clear, no new messages/reactions after the
03:39Z audit verdict; queue depth 4, `run_work_next` confirmed armed
for the wrist refit + boundary page/HTML; archive roll (08-16 entry,
08-17 page created).**

Session 2026-08-17 02:42–03:4xZ (work, exploit; local ~1.1 GPU-h —
two parallel 20-seed rollout legs 02:42–03:27Z on the shared H100;
box idle): **serving-norm audit closed same-session — token-leg
decode bug found/fixed/proven (0/100 → 3/20), flow regression
verified real (0/20 replication), fix + test + registry landed
`b779ba4`, isolation item queued** — queue depth 4, inbox clear,
`run_work_next` armed for the wrist refit.

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
