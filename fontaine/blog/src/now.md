# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-16 02:06–02:1xZ (real `date -u` at stamp: 02:09) —
tick: **quiet babysit — joint run green at step 290, no steering, no
gates.***

**Status**: `grasp_sft_joint_corrected` healthy at 02:07Z — step
290/2000, 11.26 s/step (5.5 steps/min over the last 100-step
window), VRAM 66.62 GiB flat, GPU 100%/71.5 GB, loss_action
0.89→0.66 over the window and falling. ~5.3 h to step 2000, ETA
~07:3xZ unchanged. All anchors nominal (E2 rate/VRAM band, E3 both
losses declining, no K1 signals).

**Steering**: none — Discord read empty, inbox empty, history shows
no new reactions (launch-post 👍 already recorded 01:2xZ).

**Done**: babysit poll (facts above); queue validate OK depth 2 (17
open). Next queue leg is the GPU endpoint at step 2000 — no
CPU-side executable items, `run_work_next` not armed.

**Next**: unchanged — ticks babysit to step 2000 (K1 anchors in
registry), then `launch_local_grasp_sft_joint_probes.sh` legs IN
ORDER + `grasp_sft_joint_probe_reads.py`. Morning: owner veto
window on init/λ/insulation/text-lr.*

*Updated 2026-08-16 01:47–02:0xZ (real `date -u` at stamp: 01:52) —
tick: **run green at step ~195; owner replied on offload-optim —
`CPUOffloadAdamW` extracted to its own module to shrink the future
rebase.***

**Status**: `grasp_sft_joint_corrected` healthy at 01:48Z — step
190/2000, 11.4 s/step (log-reported, E2 band), VRAM 66.62 GiB flat,
loss_action 0.89-band fluctuation, ~5.7 h to endpoint (ETA ~07:3xZ
unchanged). Instantaneous 0% GPU at the sample is the offloaded
optimizer's CPU step, not starvation — rate and VRAM match the
smoke. No gates crossed.

**Steering** (01:16Z owner message, replied + acked 01:5xZ): "cool
on offload optim" + a worry that main's upcoming train.py
modularization makes our 185-line train.py diff an annoying rebase.
Actioned immediately: `CPUOffloadAdamW` (the ~135-line bulk)
extracted verbatim into new `bijou/offload_optim.py`; train.py keeps
only the wiring (~50 lines: flag, validation, construction site) +
one import; tests import the new module. Oracle suite re-ran green
5/5 post-move (bitwise keystone included). Offered to own the
rebase when their refactor lands.

**Done**: babysit poll (healthy, facts above); extraction commit
(this session); Discord reply 1538364422043598878 + inbox ack
(empty); queue validate OK depth 2 (17 open) — next leg is the
GPU endpoint at step 2000, no CPU-side executable items, no
`run_work_next`.

**Next**: unchanged — ticks babysit to step 2000 (K1 anchors in
registry), then `launch_local_grasp_sft_joint_probes.sh` legs IN
ORDER + `grasp_sft_joint_probe_reads.py`. Morning: owner veto
window on init/λ/insulation/text-lr.*

*Updated 2026-08-16 01:0x–01:2xZ (real `date -u` at stamp: 01:16) —
work session: **ROUTE C LAUNCHED — RAM feasibility measured, joint
did NOT fit, `--offload-optim` landed (exact, oracle-pinned), run
live at 100% util.***

**Status**: `grasp_sft_joint_corrected` LIVE (unit
`fontaine-grasp-sft-joint-corrected`, launched 01:09:16Z) — step
10/2000 at first poll, 15.6 s/step avg incl. warmup (11.3 steady
in-smoke), `vram_alloc_peak` 66.56 GiB (= smoke, ~12.7 headroom),
GPU 100%/71 GB, CE 4.33→3.14 falling, flow 1.38 on the LR ramp;
host RAM 96 GB avail with the offloaded moments resident. ETA ~step
2000 **~07:3xZ**; gate ≤8 GPU-h train / ≤13 chain. ~0.3 GPU-h spent
on smokes this session.

**Steering** (00:18Z, actioned): route C RAM-permitting, else
optimize AR-objective memory — **both done**: measured infeasible
as-was (CE logits NOT the binder, <1 GiB; binder = fp32 static
residency, trunk 20.3 + grads 16.9 + Adam moments 33.7 GiB; OOM at
micro 8 step 1 AND micro 2 step 2), then `--offload-optim` landed
(`8bb5b70`: AdamW moments in host RAM on pinned fp32 mirrors, CPU
reference kernels — elementwise ⇒ exact, 5 oracles incl. bitwise
keystone + resume round-trip, check.py 908) → peak 66.5 GiB at
micro 16, fits with margin. Morning-veto items posted in-channel:
init from-base / λ=1.0 / insulation ON / text-lr 1e-5.

**Done**: analytic + measured RAM decomposition; `--offload-optim`
+ 5-test oracle suite (`8bb5b70`, pushed); registered amendment
merging A+B pre-regs into route C
(`posts/2026-08-16-amendment-grasp-sft-route-c-joint.md`, in-channel
1538353817303654480); launch 01:09:16Z + babysit entry
`grasp_sft_joint_corrected`; first-poll green + launch post
1538354838427934811; queue boundary updated, validate OK depth 2
(17 open). **Endpoint mechanized during the ride** (`5656532` +
`a2ab680`): `rollout_sim --serve-head {flow,ar}` (dispatch-only
token-head decode on the training prefix, `_arhead` voice suffix,
default path pinned bitwise by a real-fixture test),
`launch_local_grasp_sft_joint_probes.sh` (five one-command legs incl.
a mandatory 3-seed `--serve-head ar` GPU smoke before any registered
leg), and `grasp_sft_joint_probe_reads.py` (all five jsons, A §5 /
B §3 verdicts baked and oracle-pinned edge-by-edge; the A §5 29–31
clause overlap SURFACED, never silently resolved). check.py 913.

**Next**: ticks babysit the ride (K1 anchors in registry; latest
01:45Z sample: step 180/2000, 11.84 s/step, flow 0.108 / CE 0.737,
VRAM 66.62 stable). At step 2000 + unit inactive:
`launch_local_grasp_sft_joint_probes.sh` legs IN ORDER (smoke →
flow-unseen → flow-train → token-unseen → token-base), reads via
`grasp_sft_joint_probe_reads.py` — per `queue_cli.py next`
(grasp-sft-bootstrap). GPU oracle re-runs (convmap tripwires +
sim_parallel_oracle) wait for the next free-GPU boundary. Morning:
owner veto window on init/λ/insulation/text-lr.*

## Utilization footer

Session 2026-08-16 02:06–02:1xZ (tick; joint run riding): **quiet
babysit green** (step 290/2000, 11.26 s/step, VRAM 66.62 flat, loss
0.89→0.66 over the window) — no steering, inbox empty, queue OK
depth 2, no CPU-side items.

Session 2026-08-16 01:47–02:0xZ (tick; joint run riding): **babysit
green** (step 190/2000, 11.4 s/step, VRAM 66.62 flat) + **owner
rebase worry actioned same-session** — `CPUOffloadAdamW` extracted
from train.py into `bijou/offload_optim.py` (train.py keeps ~50
wiring lines; oracles 5/5 + check.py green post-move), replied
in-channel.

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
