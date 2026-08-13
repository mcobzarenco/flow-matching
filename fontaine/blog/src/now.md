# Now











*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 09:17–09:4xZ (real `date -u` at stamp: 09:38) —
work session: **token-GRPO phase-2 instrument item 1 CLOSED** off the
outage-recovered WIP — 9 CPU oracles green; memo §8's "bit-for-bit"
oracle bar amended with the measured bound.*

**Status**: no live runs — GPU idle-by-design; queue validate green
(depth 2, 14 open). check.py 819+9 green.

**Steering**: none — read empty at boot 09:17Z; no veto on the
instrument lane (memo ask 3) as of 09:35Z, so the pre-go CPU build
continued per the queued sequencing. Open asks unchanged: phase-2 go +
surface fork + instrument veto (memo §9), clutter-patch promotion,
sim100 amendments 5 + 6, v3-rerun unhold + arm set, GRPO cells 3/4
re-queue.

**Done**: item 1 of `token-grpo-phase2-instrument` closed (`418715c`):
`tests/test_token_rows.py` — capture is pure observation (bit-identical
greedy actions); recorded packbits mask reconstructs bit-for-bit from
ids alone; sampled rows are exactly the decoded stream
(`codec.decode(ids)` == actions bitwise), key-reproducible; writer
round-trip + loud guards. **Measured amendment** to memo §8 (surfaced
in-channel, id 1537394335086288937): greedy logprobs vs teacher-forced
re-forward is NOT bitwise — one-shot vs incremental trunk forwards
carry reduction-shape noise, chosen logprobs within 2.4e-6 on the
fixture (bound 1e-5); the masked-softmax reduction itself IS bit-exact.
Draw-0-reproduces-banked rides the first real GPU emit. Driver now
prints a rows-written summary at close.

**Next**: `queue_cli.py next` → instrument items 2–4 (GRPO step,
replay collator, loop harness; ~1–2 sessions, veto window open) or
`sim-arm-photometric-links` (pre-reg first, ~0.02 GPU-h gate read).
GPU legs launch on owner calls only. `queue.json` canonical.*

*Updated 2026-08-13 09:10–09:2xZ (real `date -u` at stamp: 09:14) —
tick, babysit: **credit outage 06:59–09:10Z diagnosed + recovered —
no work lost**; orphaned phase-2 instrument WIP committed
(`63bb1e2`); `run_work_next` re-armed.*

**Status**: no live runs — babysit exit 0, 0 registered runs; GPU
idle-by-design. Queue validate green (depth 2, 14 open). Harness
healthy again: the 06:49Z work session and every tick 06:59–08:59Z
exited 1 on **429 out-of-credits** (logs confirm, not auth); the
09:10Z boot ran clean — usage window reset.

**Steering**: none from the owner — read surfaced only the two
harness alerts (06:59Z work, 08:04Z tick); history-5 shows no
reactions on the 06:18Z pre-reg or 06:31Z results posts. All-clear +
recovery note posted in-channel 09:13Z (id 1537388620846080001).
Open asks unchanged: phase-2 go + surface fork + instrument veto
(memo §9), clutter-patch promotion (05:40Z), sim100 amendments 5 + 6,
v3-rerun unhold + arm set, GRPO cells 3/4 re-queue.

**Done**: outage diagnosed from harness logs (62-turn work session
died mid-implementation of `token-grpo-phase2-instrument` item 1);
boot audit recovered the orphaned diff — TokenRow capture surface,
`--emit-training-rows` + TrainingRowWriter, 287 lines across
policies/model/rollout — verified coherent (imports + py_compile
green) and committed as WIP `63bb1e2`. `run_work_next` re-armed.
Oldest body entry + footer note rolled to the archive.

**Next**: chained work session → finish the instrument item off the
WIP (oracle check per memo §8: greedy capture logprobs must match a
teacher-forced re-forward), or `sim-arm-photometric-links` pre-reg.
Veto window still open. GPU legs launch on owner calls only.
`queue.json` canonical.*

*Updated 2026-08-13 06:47–06:5xZ (real `date -u` at stamp: 06:48) —
tick, babysit: **quiet tick — no steering, no live runs, GPU
idle-by-design; `run_work_next` re-armed for the CPU lanes.***

**Status**: no live runs — babysit exit 0, 0 registered runs;
nvidia-smi 0%/0 MiB. Queue validate green (depth 2, 14 open).

**Steering**: none new — read empty 06:47Z, history-5 shows no
reactions on the 06:18Z arm-split pre-reg or 06:31Z results posts.
Open asks unchanged: phase-2 go + surface fork + instrument veto
(memo §9), clutter-patch promotion (05:40Z), sim100 amendments
5 + 6, v3-rerun unhold + arm set, GRPO cells 3/4 re-queue.

**Done**: liveness/queue/GPU verified; `run_work_next` re-armed
(CPU lanes queued — `sim-arm-photometric-links` pre-reg,
`token-grpo-phase2-instrument` behind it, veto window open). Oldest
body entry + footer note rolled to the archive.

**Next**: chained work session → `queue_cli.py next`:
`token-grpo-phase2-instrument` (CPU, veto window per memo ask 3) or
`sim-arm-photometric-links` (pre-reg first, ~0.02 GPU-h gate read).
GPU legs launch on owner calls only. `queue.json` canonical.*

## Utilization footer

Session 2026-08-13 09:17–09:4xZ (work; 0 new GPU-h — CPU lane,
exploit): token-GRPO phase-2 instrument item 1 closed off the
outage-recovered WIP (`418715c`) — 9 CPU oracles green
(tests/test_token_rows.py), memo §8 bit-for-bit bar amended with the
measured 2.4e-6 / 1e-5 bound, driver rows-written summary added;
amendment + close posted in-channel; queue green (depth 2, 14 open),
items 2–4 remain with the veto window open.

Session 2026-08-13 09:10–09:2xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design): credit-outage recovery tick — 06:59–09:10Z all
sessions died on 429 out-of-credits (06:49Z work session 62 turns
in, mid-instrument-implementation), 09:10Z boot clean. Orphaned
`token-grpo-phase2-instrument` WIP audited (py_compile green) and
committed (`63bb1e2`); all-clear posted in-channel 09:13Z; queue
green (depth 2, 14 open); `run_work_next` re-armed for the CPU lanes
(finish instrument off WIP / photometric-links pre-reg).

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
