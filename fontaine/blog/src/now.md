# Now








*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 05:57–06:0xZ (real `date -u` at stamp: 06:00) —
tick, babysit: **quiet tick — no steering, no live runs, GPU
idle-by-design; `run_work_next` stays armed for the CPU lanes.***

**Status**: no live runs — babysit exit 0, 0 registered runs;
nvidia-smi 0%/0 MiB. Queue validate green (depth 2, 14 open).

**Steering**: none new — read empty 05:58Z, history-5 shows no
reactions on the 05:23Z pre-reg or 05:39Z appearance-pass results
posts. Open asks unchanged: **clutter-patch promotion sign-off**
(asked 05:40Z), sim100 amendments 5 + 6, v3-rerun unhold + arm set,
disk-draws sign-off, GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done**: liveness/queue/GPU verified; `run_work_next` confirmed
armed (CPU lanes queued — `token-grpo-phase2-design-memo`,
`sim-arm-appearance-leg` pre-reg; GPU idle-by-design per
no-idle-pauses). Oldest body entry + footer note rolled to the
archive.

**Next**: chained work session → `queue_cli.py next`:
`token-grpo-phase2-design-memo` (CPU) or `sim-arm-appearance-leg`
diagnostic (pre-reg first, ~0.02 GPU-h). Promotion + GPU legs
launch on owner calls only. `queue.json` canonical.*

*Updated 2026-08-13 05:17–05:5xZ (real `date -u` at stamp: 05:56) —
work session: **fg appearance pass legs (b)+(c) EXECUTED, registered
gate PASS — real-crop clutter patches read 0.556 vs v3 0.713 (−0.157,
100/100 closer) and beat the no_clutter removal ceiling 0.576; the
appearance pass is CLOSED, promotion pends the owner go.***

**Status**: no live runs — GPU idle-by-design (this session's spend
~0.02 GPU-h: the pre-registered gate embeds). Queue validate green
(depth 2, 14 open).

**Steering**: none new — read empty at boot 05:17Z and at the 05:55Z
close poll (only my own pre-reg/results posts). Open asks unchanged
plus one new: **clutter-patch promotion sign-off** (asked 05:40Z);
sim100 amendments 5 + 6, v3-rerun unhold + arm set, disk-draws
sign-off, GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done** (commit `2e15ae7`; pre-reg 05:23Z, results in-channel
05:40Z): `make_clutter_crops.py` mined RGBA crops of the real
mouse/mug/laptop/pcb from bank-episode naive medians (alpha =
feathered novelty vs the corrected global plate; areas bit-match the
manifest, centroid drift ≤0.1 px); `clutter_patch.py` pastes them at
the drawn poses by inverse warp through the verified fisheye model
(episode grading, zero extra RNG draws); `sim_fg_appearance_fix.py`
(leg (a) harness, 3 arms, ONE hooked instance) read **patched 0.556
(ΔAUROC −0.157 vs the −0.05 bar, paired Δknn5 −2.02e-06 CI-excl-0,
100/100 closer; beats no_clutter 0.576 by −0.020, 75/100,
CI-excl-0)** — full-recovery read fires. Integrity: in-run v3 0.7127
in band, no_clutter reproduces leg (a) within ±0.01, bit-exact
oracle green 100/100. Queue: appearance pass CLOSED;
`sim-clutter-patch-promotion` (blocked, owner_hold) +
`sim-arm-appearance-leg` queued. Artifacts on fontaine-reports
(curl-200): analysis JSON, chart, v3-vs-patched strip, crops strip.

**Next**: `queue_cli.py next` → `token-grpo-phase2-design-memo`
(CPU) or `sim-arm-appearance-leg` diagnostic (pre-reg first, ~0.02
GPU-h). Promotion + GPU legs launch on owner calls only.
`queue.json` canonical.*

*Updated 2026-08-13 05:15–05:1xZ (real `date -u` at stamp: 05:17) —
tick, babysit: **quiet tick — no steering, no live runs, GPU
idle-by-design; `run_work_next` stays armed for the CPU lanes.***

**Status**: no live runs — babysit exit 0, 0 registered runs;
nvidia-smi 0%/0 MiB. Queue validate green (depth 2, 13 open).

**Steering**: none new — read empty 05:15Z, history-5 shows no
reactions on the 04:41Z pre-reg or 04:52Z content-split results
posts. Open asks unchanged: sim100 amendments 5 (v4 default) + 6
(curve-only fitted wrist lens default), v3-rerun unhold + arm set,
disk-draws sign-off, GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done**: liveness/queue/GPU verified; `run_work_next` confirmed
armed (CPU lanes queued — `sim-foreground-appearance-pass` leg (b),
`token-grpo-phase2-design-memo` — GPU idle-by-design per
no-idle-pauses). Oldest body entry + footer note rolled to the
archive.

**Next**: chained work session → `queue_cli.py next`:
`token-grpo-phase2-design-memo` or `sim-foreground-appearance-pass`
leg (b) (clutter appearance fix — real-crop textures / plate-sourced
patches; pre-reg first, ~0.02 GPU-h per gate read). GPU legs launch
on owner calls only. `queue.json` canonical.*

## Utilization footer

Session 2026-08-13 05:57–06:0xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design): quiet tick — no owner messages/reactions (05:58Z),
babysit exit 0 with 0 registered runs, nvidia-smi 0%/0 MiB, queue
green (depth 2, 14 open). `run_work_next` stays armed for the CPU
lanes (token-GRPO phase-2 memo / arm-appearance leg pre-reg).

Session 2026-08-13 05:17–05:5xZ (work; +~0.02 GPU-h — the
pre-registered gate embeds, exploit): fg appearance pass legs (b)+(c)
EXECUTED (`2e15ae7`) — real-crop clutter patches mined from the bank
episodes' own medians and pasted by inverse fisheye warp read 0.556
vs v3 0.713 (gate PASS at 3× the bar, 100/100 closer) and beat the
no_clutter removal ceiling 0.576 (−0.020, 75/100): the appearance
pass is CLOSED, promotion queued owner_hold (asked 05:40Z),
arm-appearance leg queued next. `run_work_next` re-armed (CPU lanes
queued).

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
