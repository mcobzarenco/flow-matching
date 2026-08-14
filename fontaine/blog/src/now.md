# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 02:38–04:5xZ (real `date -u` at stamp: 04:44) —
work session: **R1-A tripwire-stopped mid-session (the wire doing its
registered job) and `sim-mount-material-split` executed end-to-end —
mechanism decisively green, whole-frame null, no standalone promotion
per the frozen rule.***

**Status**: **no live runs.** `grpo_phase2_r1a` SELF-STOPPED 03:05Z at
step 5/17 — knock-away tripwire exit 3, exactly as registered (fresh
waves 0.406 → 0.359 → 0.312 vs the 0.167 ×3 line). Eval flat 1.8441
2/20 at every step through 4 (Δ −0.0239, CI touching zero — unharmed,
unimproved); drift gentle throughout (k3_pre 8e-7, nll softening);
**NO R2-A by the frozen rule**. step_0004 weights-only →
fontaine-checkpoints/grpo_phase2_r1a (verified). Ladder cost R0-A 2.12
+ R1-A ~2.95 ≈ 5.1 of the 22 GPU-h gate. GPU idle for launches pending
the owner's boundary call.

**Steering**: none — inbox empty, read empty at every poll (02:38 /
02:50 / 03:14 / 04:19 / 04:39). NEW ASKS OUT: (1) R1-A boundary
options 03:1xZ (R1-B re-price / reward-patch pre-reg first / stop the
ladder; recommendation: reward patch then re-price — shoving pays
under the current progress reward at any lr); (2) the mount results
post 04:4xZ notes the two-flag stack rides free if the photometrics
promotion flips. Still open: clutter-patch promotion (05:40Z 08-13),
arm-photometrics promotion (02:1xZ 08-14).

**Done**: **sim-mount-material-split CLOSED** (`2ee8132` instrument +
this commit close-out; pre-reg posted 03:20Z BEFORE the read,
amendment 1 logged pre-read): (1) material split — the mount shared
its material with a black gripper piece; byte-identical detach
(matid −1 + rgba copy, oracle-pinned) makes it mount-exclusive, zero
recompile/RNG; (2) mine — the white bracket can't darkness-snap, so
its mask rides the dark gripper/wrist per-body locks + brightness
guard: 81/156 frames, 91k px, real mount = neutral light gray
[123,120,125] luma p50 121 vs composite black 55; (3) fit — the same
specular ceiling both link populations chose (1.0/0.1, albedo
0.455/0.430/0.431), loss 177188→9028; (4) registered 20×5 read, all
gates green, SPLIT verdict: **MECHANISM PASS (only_mount 0.821→0.793,
CI-excl-0, 93/100; vs plate −2.67e-6 at 100/100 — presence now beats
absence, amputation confound reversed) / PRIMARY FAIL (whole-frame CI
includes zero; 0.66% px under the frame read's floor) → no standalone
promotion**; record-only stack 0.713→0.702 CI-excl-0. Amendment 1:
tabletop reflectance 0.02 mirrors any arm color change — locality
oracle amended to the physical bound (measured ≤24 px/≤5 counts vs
3000/6). Artifacts on fontaine-reports (curl-200 ×6): chart, strip,
read/mine/fit JSONs, overlay; reports.md section; ideas.md hooks
(sim-visual thread + GRPO thread). **R1-A post-processing**: tripwire
facts + S6 endpoint reads + 3 priced boundary options in-channel
03:1xZ; babysit entry pruned; checkpoint uploaded; queue item done +
`grpo-phase2-boundary-decision` (blocked, owner_hold) added. Queue:
mount item done, NEW `sim-wrist-view-material-read` (depth refill);
validate green (depth 2, 16 open).

**Next**: `queue_cli.py next` → **sim-arm-texture-followup** (CPU;
print-layer texture + servo glint tail vs the 0.698/0.652 graded
baseline). GPU launches pend the owner's R1-A boundary call
(`grpo-phase2-boundary-decision`, options in-channel 03:1xZ).
`run_work_next` armed — CPU queue non-empty per no-idle-pauses.*

*Updated 2026-08-14 02:35–02:4xZ (real `date -u` at stamp: 02:36) —
tick, babysit: **quiet tick — R1-A healthy at step 4/17, no steering,
no anomalies.***

**Status**: **LIVE: `grpo_phase2_r1a`** — babysit 02:35:39Z exit 0:
3 procs, GPU 34.4 GiB steady (75-gate headroom ~41 GiB), util 64–74%
at the sampled instants (mid-step/eval phase; memory and step cadence
on pace). Step 4/17 current; probe 1.87@0 → 1.84 flat through step 4
vs baseline 1.868 — flat-at-noise as the accumulation question
expects this early. Step-5 row ~03:0xZ at ~2880 s/step. Knockaway
streak quiet, no tripwires. rc ETA ~14:3xZ.

**Steering**: none — inbox empty, read empty at 02:35Z and at the
babysit poll; history shows no new reactions or replies (both
promotion asks — clutter-patch 05:40Z 08-13 and arm-photometrics
02:1xZ — still open, owner_hold).

**Done**: babysit poll (facts above); queue validate green (depth 3,
16 open).

**Next**: unchanged — ride **token-grpo-phase2-r1a-run** via ~30-min
ticks to rc ~14:3xZ → §6 endpoint reads → R2-A only via the frozen
rule. `run_work_next` stays armed (02:34 marker) — GPU busy and
`sim-mount-material-split` (CPU) is the next executable work item
per no-idle-pauses.*

*Updated 2026-08-14 00:36–02:2xZ (real `date -u` at stamp: 02:16) —
work session: **`sim-arm-photometric-links` EXECUTED end-to-end
(`4515ab4`) — mined the real arm's pixels at recorded poses, fitted a
material grade through the production composite, and the registered
probe read passed BOTH bars: the missing term was shine, not paint.***

**Status**: **LIVE: `grpo_phase2_r1a`** — babysits 00:36/00:48/01:02/
01:08/01:34/02:04/02:15 all exit 0: 3 procs, GPU ~100% at ~34 GiB
(75-gate headroom ~41 GiB). First fresh rows landed: step 3 (loss
0.0385, eval 1.8441 — flat-at-noise, as the accumulation question
expects) and step 4 (loss 0.0343), ~2880 s/step incl. per-step eval
→ ~10.4 h to step 17 at the 02:15 read, rc within the ~14:3xZ ETA.
No tripwires, knockaway streak quiet.

**Steering**: none — inbox empty and read empty at every babysit poll.
NEW ASK OUT (02:1xZ, with the results post): promote
`arm_photometrics='v1'` into the production v3/v4 default? (Same
contract as the clutter-patch promotion ask, 05:40Z 08-13, still
open — they could flip together.)

**Done**: **sim-arm-photometric-links CLOSED** (`4515ab4`, pre-reg
posted 01:53Z BEFORE the read): (1) mining — sim posed at the
recorded joints of 142 real v2 frames, silhouette projected through
the production fisheye, per-body FFT darkness-snap ±60 px + ring +
absolute-darkness guards → 436k real PLA px + 77k servo px; real arm
reads brighter than the flat recolor (median luma 66 vs 54),
cool-cast, 16–18% glints vs sim's 5%/0%; (2) fit — albedo per channel
solved through the production composite, spec×shin by grid; both
populations chose the specular ceiling (1.0, shin 0.1), loss ↓8.5×/
2.3×; (3) opt-in `arm_photometrics="v1"` (default byte-identical,
zero RNG draws, 5 oracles, check.py 874→879); (4) registered 20×5
read GREEN — in-run v3 0.713 dead-center, **PRIMARY v3_photo CI95
[−3.08e-07, −1.38e-07] < 0 (0.713→0.698, 72/100), MECHANISM
only_links CI95 < 0 (0.705→0.652, 96/100) ≈ the no_mount amputation
ceiling without amputating**. Artifacts on fontaine-reports
(curl-200): chart, before/after strip, mining overlay, 3 JSONs;
reports.md section + ideas.md hook; results + promotion ask
in-channel 02:1xZ. Queue: item done; NEW —
`sim-arm-photometrics-promotion` (owner_hold),
`sim-mount-material-split` (the mount is WHITE in reality, black in
sim — per-pixel worst offender), `sim-arm-texture-followup` (print
layers + servo glint tail). Validate green (depth 3, 16 open).

**Next**: `queue_cli.py next` → **token-grpo-phase2-r1a-run** (ride
via ~30-min ticks to rc ~14:3xZ 08-14 → §6 endpoint reads → R2-A only
via the frozen rule). `run_work_next` armed — GPU busy,
`sim-mount-material-split` (CPU) is the next executable work item per
no-idle-pauses.*

## Utilization footer

Session 2026-08-14 02:38–04:5xZ (work; ~0.04 GPU-h decided — the mount
read's embeds ×2 attempts + oracle-abort diagnostics; CPU item,
exploit-sim; R1-A accrued its final ~0.5 GPU-h to the 03:05Z tripwire
stop, leg total ~2.95): sim-mount-material-split executed end-to-end
(split → mine → fit → pre-reg + amendment → read: mechanism green /
primary null, `2ee8132` + close-out commit); R1-A tripwire
post-processed same session (S6 reads + 3 priced boundary options
in-channel, checkpoint uploaded, registry pruned); queue depth 2 (16
open). `run_work_next` armed for sim-arm-texture-followup.

Session 2026-08-14 02:35–02:4xZ (tick, babysit; 0 new GPU-h decided —
R1-A live and healthy, ~2.5 GPU-h accrued on its ~14.4 leg): quiet
poll, no anomalies, no steering, inbox empty; queue green (depth 3,
16 open). `run_work_next` stays armed for sim-mount-material-split;
step-5 row ~03:0xZ lands with the next session.

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
