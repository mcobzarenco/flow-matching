# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 18:41–19:0xZ (real `date -u` at write: 18:51) —
tick: **the discriminator is LIVE. Owner GO landed 18:40:56Z ("You
can do whatever you want", 24 s after the GO-gap post; ask open since
15:14Z) and the tick executed the full ON-GO checklist inside the
session: pre-reg dated + published
(`posts/2026-08-17-prereg-sft-drift-discriminator.md`, SUMMARY +
Space pushed + 200-verified, in-channel 1538981787479449671),
`systemd-run --user` unit `fontaine-demosonly-1gpu-disc` launched
18:44:15Z on the local H100 (preflight guard passed — GPU was clean),
babysit entry active, launch commit `b02cfed` pushed.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` (demosonly
recipe on ONE GPU, single delta = distributed machinery removed;
eff-96 = micro-12 × 8 chunks, seed 0). Verdict read AT STEP 1000, not
mid-run: Δeval(1000 vs 500) ≤ +0.30 → HEALTHY (distributed CONVICTED);
≥ +1.0158 → same-drift (EXONERATED); else AMBIGUOUS. No probe-kill
bars by design — drift is the expected-interesting outcome. Gates:
vram 78 GiB, GPU-h 12. Startup verified: 4500/500 episode split as
pre-registered, weights on GPU 18:49Z, wandb run `oc2zc46t`.

**Steering**: the GO itself — recorded, replied 18:42:31Z, acked
(inbox empty). Read as a delegation on the pending ask; per the
standing rules (idle GPU is the failure, GO-gap staged to minutes)
the call was launch-now.

**Done**: ON-GO checklist end-to-end as above; queue item
`sft-drift-discriminator-run` → live (prereg field repointed to the
dated post); check.py 992 green on the launch commit; first-poll
held in-session to 18:59Z: **GPU util 95% at 66.5 GiB** — the first
eff-96 step computing (jsonl lands at its completion; no starvation);
host RAM 92 GB available with the batch-96 loader buffers filled (the
flagged watch item is real but headroom is fine — next poll re-checks
`free -g`).

**Next**: babysit cadence owns the run (~7–9 h to step 1000, probes
every 250, saves 500/1000). On completion: `sft_drift_saga_charts.py
--discriminator` verdict → drift-saga finalize slot + in-channel.
CPU queue: `local-dataset-mirrors-restore` is the executable item
(`prereg-draft-per-dataset-flow-norm-rerun` stays gated on this
run's verdict); `run_work_next` armed. Owner-pending: G1-miss ride 👍,
augment-report reaction, disk composite exemption, approach redesign
go, v2.1 bands, ckpt-format, morning-veto items.*

*Updated 2026-08-17 18:23–18:3xZ (real `date -u` at write: 18:33) —
work session: **discriminator GO-gap collapsed to minutes. The queue
head (`sft-drift-discriminator-prereg-post-draft`) is DONE and
over-delivered: the formal pre-reg DRAFT is cut
(`posts/2026-08-xx-prereg-sft-drift-discriminator.md`, deliberately
NOT in SUMMARY.md — drafting is not posting), the launcher is
re-platformed to the local H100
(`fontaine/scripts/launch_local_grasp_sft_v2_demosonly_1gpu_disc_h100.sh`,
command block byte-identical to the frozen box script by diff,
full-parse green vs the merged CLI: `molmoact2_joint`,
`per_dataset_flow_norm=False`, seed 0, plus a GPU-busy abort guard for
the owner policy-server), and the v2 corpus is BACK ON LOCAL DISK
(35 GiB snapshot of `mcobzarenco/fontaine-grasp-demos-v2` →
`~/datasets/fontaine/grasp_demos_v2/merged` — it was HF-only after
the box kill). Frozen bounds quoted verbatim in the draft: healthy
≤ +0.30 / drift ≥ +1.0158 (= 0.5 × demosonly +2.0317), fixture
rigonly +0.6929 → AMBIGUOUS agrees.***

**Status**: NO live runs (babysit: 0 registered, exit 0). Local H100
free (0 MiB, no compute apps) and idle-by-design: the 1-GPU
discriminator stays OWNER-GATED (ask 15:14Z, open ~3.5h). Queue
validated, depth 2 (both CPU).

**Steering**: none this session — `read` empty, inbox empty at boot
and at close.

**Done**: queue head `sft-drift-discriminator-prereg-post-draft`
DONE (this commit): draft + local launcher + dataset pull as above;
check.py 992 green; `sft-drift-discriminator-run` re-classed
gpu-local with the ON-GO checklist in its boundary (date post → 
SUMMARY → blog push → in-channel → systemd-run → babysit entry →
first-poll util + `free -g`, loader workers 8 × prefetch 4 at
batch-96 flagged as the host-RAM watch item, GPU-h gate 12). Queue
refill: `local-dataset-mirrors-restore` (CPU — v1 corpus is HF-only
since the box kill; audit which held gpu-local arms need it, then
pull). Queue page regenerated; posted in-channel.

**Next**: `queue_cli.py next` = `prereg-draft-per-dataset-flow-norm-rerun`
— but it is GATED behind the discriminator verdict (its baseline arm
depends on it), so the executable item is
`local-dataset-mirrors-restore`; `run_work_next` armed. On
discriminator GO: the run item's boundary carries the full minutes-
scale checklist. Owner-pending: discriminator go (head item), G1-miss
ride 👍, augment-report reaction, disk composite exemption, approach
redesign go, v2.1 bands, ckpt-format, morning-veto items.*

## Utilization footer

Session 2026-08-17 18:41–19:0xZ (tick; GPU-h accruing — discriminator
launched): **owner GO 18:40:56Z → full ON-GO checklist in-session:
pre-reg published + `grasp_sft_v2_demosonly_1gpu_disc` LIVE on the
local H100 from 18:44:15Z (unit fontaine-demosonly-1gpu-disc, ~7–9 h
to step 1000, GPU-h gate 12), babysit entry active, launch commit
`b02cfed`** — `run_work_next` armed for the CPU queue
(v1-mirror-restore) while the run rides.

Session 2026-08-17 18:23–18:3xZ (work, exploit; zero GPU-h — local
H100 free and idle-by-design behind the owner-gated discriminator):
**discriminator GO-gap collapsed to minutes — formal pre-reg draft
cut (frozen kit bounds quoted verbatim), launcher re-platformed to
local H100 (command block byte-identical to the frozen box script,
full-parse green, policy-server abort guard), v2 corpus re-pulled
local (35 GiB HF snapshot); check.py 992 green; queue refilled with
the v1-mirror-restore infra item** — `run_work_next` armed, next
executable CPU item is the v1 mirror restore.

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
