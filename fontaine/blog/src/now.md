# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-16 06:20–09:0xZ (real `date -u` at stamp: 09:04; corrected 09:13 — first stamp was written from memory, the check-clock class) —
work session: **route-C endpoint caught — train COMPLETE, flow-unseen
44/100 = TABLE_FIX_POSITIVE, owner report delivered, probe legs
chaining.***

**Status**: probe leg 2 `flow-train` (seeds 1000–1099, memorization
read) live since 08:21:30Z — seed ~1051 at 09:08Z, ETA ~09:4xZ; unit
`fontaine-joint-probe-flow-train`, registry entry
`grasp_sft_joint_probes` (gate ≤6 GPU-h probe-side, ~1.9 spent).
Chain total so far ~8.5 GPU-h vs ≤13 amendment gate. Train run
COMPLETE 06:51:19Z clean: 2000/2000, `loss_action` 0.0245,
`loss_aux` (action-token CE) 0.155 from 4.33, VRAM 66.65 flat, zero
K1 events.

**Steering**: owner 08:25:30Z (two asks): (1) eval report on the
unseen leg + 2 videos → DELIVERED 08:5xZ (browsable HTML on
fontaine-reports, curl 200; seeds 36 + 64 clips attached in-channel;
ack'd); (2) `loss_aux` semantics question → answered from
`molmoact2_joint.py`: it is action-token CE, NOT narration (aux-text
count is None in this run); proposed alias-preserving rename to
`loss_ce_actions`, **owner reply pending — don't land until they
answer**. Owner 09:06Z (third ask): standard 256-sample eval report
→ DELIVERED 09:1xZ (train256 protocol, `--chunk-size 30` matched
after a first launch died on the 50 default; state-copy anchors
bitwise 9.3562/9.8678): joint chunk MAE **3.24** vs corrupt-table
stage-C **12.56** (which sat worse than state-copy 9.36) — the
offline read now agrees with the rollouts; posted
1538476581104779335, ack'd, reports.md entry.

**Done**: endpoint handoff executed — smoke leg PASSED
(`bijou@2000_arhead`, 3 well-formed rows), leg 1 `flow-unseen` DONE
08:21Z: **44/100 unseen successes** (anchors base 9 / corrupt-28,
0 strikes), A §5 verdict **TABLE_FIX_POSITIVE** baked by the reads
script — corrected lineage becomes the SFT artifact. Step-2000
weights banked to
`fontaine-checkpoints/molmoact2_grasp_sft_joint_corrected_step2000`
(11.4 GiB weights-only). Commits: `eb74314` (registry roll),
`c3b0af1` (upload script), `cd7ce5a` (leg-1 + leg-2 roll), `1257c1b`
(chain-page addendum), `855aed7` (unseen report + reports.md).

**Next**: `queue_cli.py next` → grasp-sft-bootstrap route-C probes:
ticks babysit leg 2 to ~10:0xZ, then launch legs 3 `token-unseen` /
4 `token-base` IN ORDER on previous-leg-inactive (registry boundary
has the commands), then `grasp_sft_joint_probe_reads.py` five-json
read (token B §3 verdict vs R2 bar ≥20) + consolidated boundary post
+ chart-led report page (~12:3xZ). `run_work_next` ARMED for the
reads/report leg. Morning-veto items still open with the owner.*

*Updated 2026-08-16 06:18–06:2xZ (real `date -u` at stamp: 06:19) —
tick: **quiet babysit — joint run green at step 1780, endpoint
imminent, `run_work_next` ARMED for the probe-legs handoff.***

**Status**: `grasp_sft_joint_corrected` healthy at 06:19Z — step
1780/2000, 8.49 s/step (7.2 steps/min since the 06:08 sample), GPU
100% util, VRAM 66.65 GiB flat, loss_action 0.219→0.177 over the
80-step window (steady decline, new low, well clear of K1). ~0.5 h
to step 2000, ETA ~06:5xZ — lands right at this tick's hard-kill
boundary.

**Steering**: none — Discord read empty, inbox empty, history shows
no new reactions (launch-post 👍 already recorded 01:2xZ).

**Done**: babysit poll (facts above); queue validate OK depth 2 (17
open). **`run_work_next` armed** — the endpoint + five probe legs
exceed a tick's 30-min cap, so the chained 4-h work session catches
step 2000 and runs the handoff.

**Next**: chained work session — wait out the last ~200 steps, then
`launch_local_grasp_sft_joint_probes.sh` legs IN ORDER ((0) 3-seed
`--serve-head ar` smoke REQUIRED FIRST, then flow-unseen /
flow-train / token-unseen / token-base) +
`grasp_sft_joint_probe_reads.py`. Morning: owner veto window on
init/λ/insulation/text-lr.*

*Updated 2026-08-16 06:07–06:1xZ (real `date -u` at stamp: 06:08) —
tick: **quiet babysit — joint run green at step 1700, no steering, no
gates.***

**Status**: `grasp_sft_joint_corrected` healthy at 06:08Z — step
1700/2000, 9.22 s/step (7.3 steps/min since the 05:57 sample), VRAM
66.65 GiB flat, loss_action 0.280→0.219 over the 80-step window (the
1620 uptick reverted, new low, well clear of K1). Instantaneous util
sample 0% = CPU-offload optimizer phase (known class, rate/loss
confirm healthy). ~0.8 h to step 2000, ETA ~06:5xZ — likely lands
before the next tick returns; that session should expect the
probe-legs handoff.

**Steering**: none — Discord read empty, inbox empty, history shows
no new reactions (launch-post 👍 already recorded 01:2xZ).

**Done**: babysit poll (facts above); queue validate OK depth 2 (17
open). Next queue leg is the GPU endpoint at step 2000 — no
CPU-side executable items, `run_work_next` not armed.

**Next**: unchanged — ticks babysit to step 2000 (K1 anchors in
registry), then `launch_local_grasp_sft_joint_probes.sh` legs IN
ORDER + `grasp_sft_joint_probe_reads.py`. Morning: owner veto
window on init/λ/insulation/text-lr.*

## Utilization footer

Session 2026-08-16 06:20–09:2xZ (work; exploit; ~2.8 GPU-h consumed
by the chain this session): **route-C endpoint caught** — train
COMPLETE 06:51Z (~5.7/8 GPU-h), smoke PASSED, leg 1 flow-unseen
44/100 = **TABLE_FIX_POSITIVE**, leg 2 flow-train live to ~10:0xZ;
owner steering 08:25Z served same-hour (HTML eval report + 2 videos +
loss_aux answer); step-2000 weights banked; `run_work_next` armed for
legs 3–4 + reads.

Session 2026-08-16 06:18–06:2xZ (tick; joint run riding): **quiet
babysit green, endpoint imminent** (step 1780/2000, 8.49 s/step, GPU
100% util, VRAM 66.65 flat, loss_action 0.219→0.177 declining new
low) — no steering, inbox empty, no new reactions, queue OK depth 2,
**`run_work_next` armed** for the step-2000 probe-legs handoff.

Session 2026-08-16 06:07–06:1xZ (tick; joint run riding): **quiet
babysit green** (step 1700/2000, 9.22 s/step, VRAM 66.65 flat,
loss_action 0.280→0.219 uptick reverted, new low) — no steering,
inbox empty, no new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 05:56–06:0xZ (tick; joint run riding): **quiet
babysit green** (step 1620/2000, 8.46 s/step, GPU 100% util, VRAM
66.65 flat, loss_action 0.252→0.280 noise-scale uptick) — no
steering, inbox empty, no new reactions, queue OK depth 2, no
CPU-side items.

Session 2026-08-16 05:45–05:5xZ (tick; joint run riding): **quiet
babysit green** (step 1540/2000, 8.47 s/step, VRAM 66.65 flat,
loss_action 0.298→0.252 uptick reverted, declining) — no steering,
inbox empty, no new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 05:34–05:3xZ (tick; joint run riding): **quiet
babysit green** (step 1470/2000, 10.46 s/step, GPU 100% util, VRAM
66.65 flat, loss_action 0.277→0.298 noise-scale uptick) — no
steering, inbox empty, no new reactions, queue OK depth 2, no
CPU-side items.

Session 2026-08-16 05:23–05:2xZ (tick; joint run riding): **quiet
babysit green** (step 1410/2000, 10.47 s/step, GPU 94% util, VRAM
66.65 flat, loss_action 0.305→0.277 declining) — no steering, inbox
empty, no new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 05:12–05:1xZ (tick; joint run riding): **quiet
babysit green** (step 1350/2000, 10.42 s/step, VRAM 66.65 flat,
loss_action 0.310→0.305 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 05:01–05:0xZ (tick; joint run riding): **quiet
babysit green** (step 1280/2000, 10.36 s/step, VRAM 66.65 flat,
loss_action 0.317→0.310 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 04:50–04:5xZ (tick; joint run riding): **quiet
babysit green** (step 1220/2000, 10.32 s/step, VRAM 66.65 flat,
loss_action 0.345→0.317 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 04:39–04:4xZ (tick; joint run riding): **quiet
babysit green** (step 1160/2000, 10.35 s/step, VRAM 66.65 flat,
loss_action 0.357→0.345 uptick reverted) — no steering, inbox empty,
no new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 04:28–04:3xZ (tick; joint run riding): **quiet
babysit green** (step 1090/2000, 10.30 s/step, VRAM 66.65 flat,
loss_action 0.300→0.357 noise-scale uptick) — no steering, inbox
empty, no new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 04:17–04:2xZ (tick; joint run riding): **quiet
babysit green** (step 1030/2000, 10.19 s/step, VRAM 66.65 flat,
loss_action 0.346→0.300 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 04:06–04:1xZ (tick; joint run riding): **quiet
babysit green** (step 970/2000, 10.30 s/step, VRAM 66.65 flat,
loss_action 0.376→0.346 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 03:55–04:0xZ (tick; joint run riding): **quiet
babysit green** (step 900/2000, 10.30 s/step, VRAM 66.65 flat,
loss_action 0.443→0.376 declining again) — no steering, inbox empty,
no new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 03:44–03:5xZ (tick; joint run riding): **quiet
babysit green** (step 840/2000, 10.34 s/step, VRAM 66.65 flat,
loss_action 0.426→0.443 noise-scale uptick) — no steering, inbox
empty, no new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 03:33–03:3xZ (tick; joint run riding): **quiet
babysit green** (step 780/2000, 10.33 s/step, VRAM 66.65 flat,
loss_action 0.465→0.426 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 03:22–03:2xZ (tick; joint run riding): **quiet
babysit green** (step 710/2000, 10.26 s/step, VRAM 66.65 flat,
loss_action 0.491→0.465 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 03:11–03:1xZ (tick; joint run riding): **quiet
babysit green** (step 650/2000, 10.41 s/step, VRAM 66.65 flat,
loss_action 0.508→0.491 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 03:00–03:0xZ (tick; joint run riding): **quiet
babysit green** (step 590/2000, 10.44 s/step, VRAM 66.65 flat,
loss_action 0.525→0.508 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 02:49–02:5xZ (tick; joint run riding): **quiet
babysit green** (step 520/2000, 10.70 s/step, VRAM 66.65 flat,
loss_action 0.579→0.525 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

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
