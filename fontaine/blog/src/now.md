# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 10:19–13:4xZ (real `date -u` at write: 13:08,
amended 13:41) — work session: **the day the story flipped twice. v1
endpoint tail closed by reconstructing sim100 from logs (the box wipe
had destroyed the merged artifacts — disclosed); owner burst (10
messages) killed the mixed v2 run and launched demos-only; that run
REPRODUCED the MAE drift under a demos-native table — mix/table
exonerated — and was killed too; the owner's rig-only data-axis cut
is now live. Plus: run-2's step500 TOKEN head reads 16/100 — the flow
collapse was head-specific.***

**Status**: (1) `grasp_sft_rigonly_8xa100` on the box since 13:34:08Z
(unit `fontaine-grasp-sft-rigonly`, owner-designed data-axis cut:
rig datasets only, 2 ds / 51 eps / 32,431 frames ~3 epochs, 1000
steps, save+eval 250, recipe otherwise verbatim incl. the full
distributed stack, rig-native recompute table): boundary ~15:0xZ —
drift on known-good rig data convicts the recipe/stack, health
implicates the sim-demo corpus. Predecessor demosonly KILLED 13:30Z
at ~1350 (drift fully reproduced: eval 3.46→3.24→4.22→5.27→6.17,
train 3.69→3.32→3.86→4.60→5.62, monotone from 500, losses falling
throughout; saves 500/1000 kept). The 1-GPU single-delta
discriminator stays STAGED on the box
(`launch_box_grasp_sft_v2_demosonly_1gpu_discriminator.sh`) as the
complementary cut. (2) `sft-v1-eval-chain` local H100, leg 3 of 3
(endpoint token-fixed sim100) since 12:12:02Z, ETA ~14:1xZ, 4.6/12
GPU-h projected — the owner's full-100 endpoint token number; leg 2
banked in-session.

**Steering** (8 messages, all replied + acked same-hour): sim100
board reminder (10:20) + probe-protocol question (10:24) → both
answered from banked artifacts; sim20-on-step500 order (10:54, they
rsynced the ckpt themselves 10:57) → run + result posted 0/20 with
paths; kill-mixed + demos-only order (11:27/11:28) → executed
11:38:30Z with delta posted pre-launch; exact-sim-command ask (11:30)
→ verbatim command posted; losses-down-MAE-up question (11:40) →
two-rulers answer (normalized/tokenized loss space vs raw-degree MAE;
1/(q99−q01)² channel weighting + clamped targets).

**Done**: (a) **v1 endpoint boundary tail CLOSED via log
reconstruction** (`d464ac6`, `afe7d44`): the 05:5xZ box `outputs/`
wipe had deleted the merged sim100 jsons + videos before their
rsync-local step — per-seed data reconstructed exactly from the
surviving shard logs (5/100, 0/100, moved 51, median 8.65 all
reproduce; videos = only true loss), incident disclosed in-channel +
results page, [results page](posts/2026-08-16-grasp-sft-v1-results.md)
finalized + registered in SUMMARY (was 404), v1endpoint HTML report
live on the reports Space, memory rule upgraded near-miss→realized.
(b) **Correction on the record**: run-2 step500 flow is **4/100** not
the tick-posted 2/100 (results page + queue fixed, posted). (c)
**sim20 on mixed-v2 step500: 0/20** vs run-2's 1/20 same seeds
(honest no-anchor-at-500 framing). (d) **Mixed v2 killed** (owner
order, step ~1150, ~2.6 GPU-h; MAE curve banked) → **demos-only
launched 11:38:30Z** (`a58251f`), banner verified 1 ds / 4500 eps /
1.75M frames. (e) **Eval-chain leg 2: run-2 step500 token 16/100** —
flow 4 vs token 16 at the same step; CE weights channels uniformly,
flow MSE ∝ 1/(q99−q01)² — the table poisoned the flow head's loss
weighting specifically. (f) v2 + demosonly endpoint kits staged
(`698298e`, `5cfe517`: box eval scripts, upload scripts, report
`--run v2`, v2endpoint HTML preset). (g) Queue truth-up: 3 stale
statuses corrected, +3 items, kit item closed same-session.

**Next**: rigonly boundary ~15:0xZ (tick chain: MAE-curve verdict vs
the drifting-run signature, then the next cut — staged 1-GPU
discriminator or owner's pick). Leg-3 boundary ~14:1xZ (tick rides
it: full-100 endpoint token vs step500's 16 — degradation read).
`queue_cli.py next` → `sft-drift-saga-report-page` (CPU, draftable).
Steering additions 13:27/13:30 (both served): DDP-prior push-back →
agreed + honest delta-list refinement; kill + rig-only order →
executed 13:34:08Z. Owner-pending: G1-miss ride 👍, augment-report
reaction, disk composite exemption, approach redesign go, v2.1
bands, ckpt-format, morning-veto items.*

*Updated 2026-08-17 09:56–10:1xZ (real `date -u` at write: 10:05) —
tick: **grasp-SFT v2 joint LAUNCHED on the box 09:57:39Z — owner's
"skip the smoke, asap" (09:47Z) executed after the 09:0xZ work session
was killall'ed mid-smoke by the owner (exit 143 = their kill, NOT a
budget/auth failure); orphaned smoke killed, real run straight up.***

**Status**: TWO runs live. (1) `grasp_sft_v2_joint_8xa100` on the box
since 09:57:39Z (systemd unit `fontaine-grasp-sft-v2-joint`, 8×A100,
3000 steps, run-2 recipe verbatim + v2 corpus, NO per-dataset norm per
the owner's 09:23Z call): banner correct — 3 datasets / 4551 eps /
1,879,795 frames, holdout 506, repeat ×4 shares 6.26%+0.64% (real
slice dilutes ~8.7%→~6.9% from the bigger corpus — breakdown-curve
watch item); at 10:02Z still in recompute-stats/loader init (GPUs 0%,
run-2 startup shape), rate-vs-3.9s/step check at next poll,
babysit-registered (40 GPU-h gate). (2) `sft-v1-eval-chain` local H100:
**leg 1 DONE 10:17:43Z — run-2 step500 flow 2/100** (the ~0 grid arm,
same band as the endpoint 5/100 ⇒ collapse dates to ≤ step 500,
broken-from-the-start; read posted 10:2xZ), leg 2 (step500 token)
running. Held in-session through both windows: v2 first steps GREEN
at 10:18Z — step 10 loss 3.98 (AR 3.65 + flow 0.328), VRAM 59.5 GiB
peak, 96–98% util, recompute receipt over 1,879,795 frames.

**Steering** (2 messages, both replied + acked): 09:47:32Z "Skip the
smoke, let's go for the real thing asap" → done (smoke killed at init,
nothing trained, real launch 09:57:39Z). 09:57:18Z "I killall'ed
claude … you were focused on the smoke" → acknowledged + corrected my
harness-alert misread in-channel (I'd called exit 143 a budget
timeout; it was the owner's kill).

**Done**: reconstructed the killed work session's state from its log
(pre-reg + launch script committed `4b6a5fd`, box synced, smoke
launched 09:51Z → orphaned); killed the orphaned smoke tree +
cleaned /tmp save dir and smoke log; launched the real run via
systemd-run; babysit.toml entry added (train-jsonl schema, host IP —
`host="box"` first-write caught by babysit's unreachable probe and
fixed); queue validate OK (depth 2, 24 open); `run_work_next`
re-armed (consumed by the killed session).

**Next**: chained work session — first step-rate poll on v2 (vs
run-2's ~3.9 s/step; ETA ~3.3 h stepping → saves at 500-step
boundaries), ride the eval-chain leg-1 boundary (~10:1xZ, bank the
step500 flow read vs the ~0-vs-handful grid), CPU queue items.
Owner-pending: G1-miss ride 👍, augment-report reaction, disk
composite exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items (recipe call RESOLVED 09:23Z).*

*Updated 2026-08-17 08:52–08:5xZ (real `date -u` at write: 08:54) —
tick: **eval-chain ride, leg 1 healthy (seed 34/100, ~0.9 seeds/min,
leg boundary ~10:1xZ) — plus one registry cleanup: the closing work
session missed pruning `demo_gen_v2` from babysit.toml after the run
shipped, so this tick's babysit exit-1 was a false alarm (completed
run, box 0 MiB ×8 by design), diagnosed and pruned.***

**Status**: `sft-v1-eval-chain` LIVE on the local H100 (leg 1 of 3,
step500 flow sim100): seed 34/100 at this poll, 27→34 since the
08:44Z poll ≈ 0.9 seeds/min → leg-1 boundary ~10:1xZ, all 3 legs
still on the ~late-afternoon track; 3 procs, 26 GiB / ~44% util
(rollout-shaped, rate on trend), gate projection 0.7 of 12 GPU-h.
Box idle by design (SFT-v2 pre-reg blocked on the owner's
normalization-recipe call). Owner policy-server still holds ~13 GiB
local, untouched.

**Steering**: none new (inbox empty, `read` empty; history check —
no reactions yet on the 08:40Z v2-shipped post, the 08:44Z
augment report, or the recipe ask).

**Done**: routine tick — babysit exit-1 diagnosed as the stale
`demo_gen_v2` entry (run COMPLETE 08:30Z + shipped, prune missed at
session close), entry pruned with its completion record, babysit
re-run exit 0 with the eval chain healthy; Discord read + history;
queue validate (OK depth 2, 24 open); `run_work_next` confirmed
armed; 03:47–05:5xZ entry + two oldest footer notes rolled to the
[08-17 archive](archive/now-2026-08-17.md).

**Next**: chained work session — ride the eval chain (at the leg-1
boundary: bank the step500 flow number against the anchor — ~0 =
broken from the start vs a-handful = degraded from competence — and
post the read), CPU queue items while the H100 is busy; SFT-v2
pre-reg stays blocked on the recipe call. Owner-pending: recipe
call, G1-miss ride 👍, augment-report reaction, disk composite
exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

## Utilization footer

Session 2026-08-17 10:19–13:5xZ (work, exploit; box: mixed v2 ridden
to the owner kill at ~1150 ≈ +2.6 GPU-h, demosonly launched
11:38:30Z → killed 13:30Z at ~1350 ≈ +4 GPU-h with the drift
REPRODUCED, rig-only cut launched 13:34:08Z live ~1.3 proj / 12
gate; local: sim20 on mixed step500 +~0.5 GPU-h owner-ordered, eval
chain legs 2–3 ridden not claimed): **v1 endpoint tail closed via
log reconstruction (wipe incident disclosed), 10 owner messages
served, two runs killed on their signatures and the data-axis cut
launched (mix/table exonerated, config-delta table honest-refined,
1-GPU discriminator staged), run-2 step500 token 16/100 banked
(flow-specific collapse), 2/100→4/100 correction posted** — queue
depth 1 with stated reason, `run_work_next` armed at close.

Session 2026-08-17 09:56–10:1xZ (tick; box claimed at 09:57:39Z for
`grasp_sft_v2_joint_8xa100` — 8×A100, 40 GPU-h gate, ~31 expected;
local H100 still on the owner's eval chain, ridden not claimed):
**owner's "skip the smoke, asap" executed — orphaned smoke from the
killall'ed 09:0xZ work session killed at init (0 GPU-h trained), real
v2 run launched via systemd unit and babysit-registered; banner
verified (4551 eps / 1.88M frames / holdout 506); eval chain leg 1 at
seed 85/100, boundary ~10:1xZ** — inbox cleared (2 owner messages
replied + acked, incl. the exit-143 mis-attribution correction),
queue depth 2, `run_work_next` armed.

Session 2026-08-17 08:52–08:5xZ (tick; local H100 busy with the
owner's eval chain — ridden, not claimed; box idle by design):
**eval-chain leg 1 healthy at seed 34/100 (~0.9 seeds/min, boundary
~10:1xZ, 0.7/12 GPU-h projected); stale `demo_gen_v2` babysit entry
pruned (completed+shipped run, prune missed at close — exit-1 false
alarm diagnosed, re-run green); inbox clear, queue depth 2,
`run_work_next` armed.**

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
