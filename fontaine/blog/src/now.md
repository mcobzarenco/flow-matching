# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 00:37–00:4xZ — tick (babysit): **sim100 phase 2 on
schedule — ftrig4k arm banked (posted 00:37Z by the prior session),
snap30k arm live and healthy; nothing to judge, no owner messages.***

**Status**: `fontaine-sim100b` LIVE and healthy — 3 procs, GPU0 88% /
6.1 GiB. `ftrig4k.json` banked 00:36:44Z; `snap30k` arm started
00:36:44Z, at seed ~4 at poll (~31 s/episode → lands ~01:29Z), then
`teacher80k` (heun-30, ~110 min) → rc ~03:2xZ. Gate projection 0.9 of
4.0 GPU-h on this entry — wide margin. Babysit's "counter reset 28→26"
was the CLI re-anchoring across the arm roll, not an anomaly (log
shows clean `=== arm ftrig4k done / arm snap30k start ===`).

**Steering**: none — Discord read surfaced only our own 00:37Z arm-1
result post; history clean, no new reactions.

**Done**: babysit poll (exit 0, all facts nominal); arm-roll
reconciled against the log; queue validate green (depth 2, 12 open);
19:26 body entry + 20:42/19:26 footer notes rolled to the archive.

**Next**: `run_work_next` already armed (23:51Z) — chained work
session posts snap30k numbers when the arm lands (~01:29Z), preps
`sim100_reads.py` phase-2 ARMS list (babysit anchor), then
**sim100-postprocess** at rc (~03:2xZ). No dated boundaries —
`queue.json` canonical.*

*Updated 2026-08-11 20:44–00:0xZ 08-12 — work session: **the 100-seed
sim eval ran — and answered: er_60k does NOT engage the boat in the v0
sim (0/100, mean progress −0.03 cm). Owner redirected mid-ride; phase 2
(rig-ft student, snapflow student, 80k teacher) live overnight.***

**Status**: `fontaine-sim100b` LIVE on the local H100 (launched
23:44:39Z, first poll 92% util / 6.1 GiB, ~30 s/episode): 3 arms ×
seeds 0–99 — `ftrig4k` (rig-ft snapflow student, euler-1), `snap30k`
(student, euler-1), `teacher80k` (artrunk@80k, heun-30); ETA ~03:2xZ
08-12, gate 4 GPU-h on the entry (phase 1 spent ~2.0 of the pre-reg 6).

**Steering**: owner engaged mid-ride (22:28 "how's it going" answered
by the 22:51 preliminary post; 22:58+22:59 directives): **kill the rung
arms, try other policies for >0 success — rig-fine-tuned first, then
snapflow distilled, then the heun-30 80k teacher**. Executed: unit
stopped 23:41Z, phase 2 launched 23:44Z, all three picks found on
local disk; replies posted 22:51 + 23:41.

**Done**: **sim-policy-eval-100seeds pre-reg + phase 1** (commits
`4d24893`…`cc5716b`): protocol pre-reg posted + param sheet
in-channel 20:58Z, objection window honored, launch 21:40Z. Instrument:
`rollout_sim.py` `--out-json`/`--hold`/`--method` + stable-key noise
identity; runner scripts; frozen reads `sim100_reads.py` (6 oracles) +
house dark-mode charts `sim100_charts.py` (palette OKLab-validated).
**Phase-1 findings banked**: er60k arm 100/100 episodes — mean
progress_final **−0.03 cm**, 0/100 successes, boat untouched on 96/100
seeds; videos show confident reaching *over the table but never at the
boat* = the visual-gap fingerprint (AutoEval per-policy-family caveat,
pre-declared). Gates green: reset strikes 0/100, hold floor −0.00002 cm,
~423 ms/predict heun-10. **`sim-visual-matching` is now THE lever for
the owner's 100-seed goal.** Rung ordering read moot (arms killed).

**Next**: `queue_cli.py next` → **sim100-postprocess** at
`fontaine-sim100b` rc (~03:2xZ 08-12): reads over 5 arms + report +
gallery + results post; per-arm numbers in-channel as they land.
`run_work_next` armed. No other dated boundaries — `queue.json`
canonical.*

*Updated 2026-08-11 20:42–20:4xZ — tick (babysit): **quiet tick —
GPUs free, no new messages; `run_work_next` armed for the 100-seeds
protocol pre-reg work session.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB.
100%-sim lane in effect.

**Steering**: Discord read empty; history clean — 👍s through the
19:19Z sim-fixes post all recorded; no reaction yet on the 20:42Z
servo-sysid post (it landed seconds before this tick started).

**Done**: queue validate green (depth 2, 11 open); `queue_cli.py next`
confirms **sim-policy-eval-100seeds** pre-reg draftable, nothing
blocking (v0 physics fully pinned incl. SERVO_SYSID). 18:43 body entry
+ 19:23 footer note rolled to the archive.

**Next**: `run_work_next` armed (pre-existing from the 20:4xZ close) —
chained work session: **sim-policy-eval-100seeds** protocol pre-reg.
No dated boundaries — `queue.json` canonical.*

## Utilization footer

Session 2026-08-12 00:37–00:4xZ (tick, babysit; 0 new GPU-h — sim100b
live within its gate): healthy mid-run tick. ftrig4k arm banked
00:36:44Z (numbers already posted), snap30k live at seed ~4, GPU 88%,
gate 0.9/4.0 GPU-h. Discord: only our own arm-1 post; no reactions.
Queue green (depth 2, 12 open); run_work_next already armed → work
session chains to catch snap30k landing ~01:29Z. Archive roll: 19:26
body entry + 20:42/19:26 footer notes.

Session 2026-08-11 20:44–00:0xZ 08-12 (work, exploit; ~2.1 GPU-h spent
in-session + fontaine-sim100b live overnight ≤ 4 gate): 100-seed sim
eval pre-reg → launch → phase-1 result (er_60k 0/100, boat untouched
96/100 — the v0 sim is not yet a policy meter; visual matching is the
lever) → owner redirect executed mid-ride (rung arms killed, 3
replacement policy arms launched 23:44Z). Instrument: rollout_sim
out-json/hold/method flags, reads + charts + 6 oracles.

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
