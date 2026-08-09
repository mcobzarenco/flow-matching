# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 21:43–21:5xZ (real `date -u` at write: 21:47) —
tick (babysit): **quiet green tick — both runs healthy, no steering,
nothing to judge.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
(21:44), step 10,640, 23.1 st/min window, 32.3/310 GPU-h, vram 75.3
×4 vs 77. Probe 11.06@10500 — a mild uptick above the 10.63@9500
run-best, the @5000/@8500 recede-precedent class, record-only.
Post-kill-line cruise, endpoint ~08-12 ~17:00Z. `fontaine-tiny10k`
LIVE local — step 800, 20.2 st/min (~2.97 s/step, on projection),
1.5/15 GPU-h; host RAM 134/221 used, **86 GiB available** — the
workers-10/prefetch-2 amendment holds (mild growth vs 21:17's
122/221, comfortable margin, record-only). Next probe @1000 ~21:5xZ.
Endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity read ~06:3xZ.

**Steering**: none — `read` surfaced only our own 21:41 lit-radar
post; `history -n 5` shows no new reactions (the 21:03 👍 was
already recorded). 13:48Z gate default (let run, gate 310) governs
adamc.

**Done**: babysit ×1 both entries; host-RAM check per the OOM class;
queue validate green depth 3 (9 open); confirmed `run_work_next`
already armed (21:43 marker, from the 0817 session close).

**Next**: chained work session → `queue_cli.py next` →
`lit-radar-0818` (CPU, GPU-busy window; 4 clean hooks, no spares —
fresh-sweep with new angles first). tiny10k endpoint ~05:1xZ 08-10 →
chained panel_v2 → Δ_capacity readout. adamc endpoint ~08-12
~17:00Z → chained k4l2 panel. MolmoAct2 follow-up arms +
ArmnetBench checkpoint watch remain owner-decision / watch items.*

*Updated 2026-08-09 21:24–21:4xZ (real `date -u` at write: 21:37) —
work session (bounded): **lit-radar-0817 CLOSED — 4 Papers pages in
~15 min wall clock via 5-agent parallel fan-out (4 deep reads + the
refill sweep concurrently), every banked hook needed corrections
again; the refill sweep hit 12/16 corpus dups — the pool is
drying.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
×2 (21:25, 21:37), step 10,480 @ 21:37, 22.0–25.8 st/min, 31.8/310
GPU-h, vram 75.3 ×4 vs 77. Run-best 10.63@9500 stands; post-kill-
line cruise, endpoint ~08-12 ~17:00Z. `fontaine-tiny10k` LIVE local
— step 660 @ 21:37 (22.0 st/min), 1.4/15 GPU-h; **first
post-relaunch probe @500 = 16.78 vs the pre-OOM run's 16.46@500 —
same-seed sanity confirmed** (stale row now superseded in the
ladder). Endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity
read ~06:3xZ.

**Steering**: none — `read` empty at boot (21:24) and at the 21:37
babysit. 13:48Z gate default (let run, gate 310) governs adamc.

**Done**: **(1) lit-radar-0817 CLOSED** — 4 Papers pages same
session ([armnetbench](papers/armnetbench.md),
[safecast](papers/safecast.md), [reflex](papers/reflex.md) +
[legato](papers/legato.md) cluster,
[compression-gap](papers/compression-gap.md); MolmoAct2 slot
satisfied by the 08-09 owner deep dive). Hook corrections, three
loud: **ArmnetBench** "3,118 human-labeled" = 2,518 scored rollouts
+ 600 unscored demos, and the claimed 84 policy checkpoints are NOT
public (→ #9 calibration study specified-but-blocked, watch item) —
but the 2,288 labeled SO-101 failure rollouts are real, Apache 2.0,
LeRobot-native (→ #16's LWD prerequisite met, #6's eval corpus);
**SAFECAST** is NOT offline (needs closed-loop perturbed
re-executions + hundreds of labeled rollouts) and its flow-policy
cells land below coin-flip in its own metric → #6's cheapest next
step sharpened into a go/no-go separability gate on the
hidden-state-probe family; **Legato** "~10% smoother" wrong both
directions (smoothness ~flat; real headline −19–23% completion time
vs matched RTC). Plus: Reflex's 2.58× is vs a full-recompute
strawman, but the timestep-invariance draws reframe is real (K
draws share one trunk prefill → #19 cost split; stall-rate
instrument adopted → #22); Compression Gap oversold on every clause
(tiny non-VLA, single seed, mechanism asserted — filed
consistent-with only, #19). Ideas #6 #9 #16 #19 #22 + index hooks
fed. **(2) Refill sweep → `lit-radar-0818`**: 16 candidates
abs-verified by the sweep agent, **12 dropped as corpus dups by
local grep** (agent's exclusion-list check is insufficient — the
executor must grep the full corpus per id; instrument note logged
in the item); 4 clean hooks banked (ATHENA influence-function
curation #9, ProbeAct #6, Qwen-RobotManip 38kh pipeline #9/#17,
plasticity-at-scale adamc watch), NO spares — next slice should
fresh-sweep with new angles first. **(3)** Self-caught a queue.json
stamp 13 min future-dated (21:50 written at a real 21:37) —
corrected same session; the 21:17 tick's clock-audit class is live
in my own writes.

**Next**: `queue_cli.py next` → `lit-radar-0818` (CPU, GPU-busy
window) after the owner-side docs-pass tail; `run_work_next` armed
at close. tiny10k endpoint ~05:1xZ 08-10 → chained panel_v2 →
Δ_capacity readout (MolmoAct2 15.5% expert-ratio anchor in hand).
adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel. MolmoAct2
follow-up arms + ArmnetBench checkpoint watch are owner-decision /
watch items.*

*Updated 2026-08-09 21:17–21:2xZ (real `date -u` at write: 21:2x) —
tick (babysit): **both runs healthy — adamc crossed its step-10,000
pre-registered kill-line checkpoint and PASSES clearly (probe
10.80@10000 vs the 14.03@2500 bar, below by 3.23); the 20:47 work
session's clocks were hallucinated ~30 min into the future
(21:45/21:5x stamps written at a real ~21:15) — corrected in
queue.json + now.md.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
(21:17), step 10,040 @ 21:18, ~22 st/min, 30.5/310 GPU-h, vram 75.3
×4 vs 77. **Step-10,000 kill line JUDGED PASS**: "probe not below
its own @2500 value by 10k" — @2500 = 14.0294 (fetched from the box
jsonl), @10000 = 10.80, clear by 3.23; run-best 10.63@9500 stands
(the 10.80@10000 is a one-eval uptick, the @5000/@8500 precedent
class). The babysit window's 5.6 st/min (21:14→21:17) was the
@10000 boundary itself — async save "captured in 21.2s" + probe
eval; re-verified 10020→10040 in 54 s (~22 st/min) right after.
Endpoint ~08-12 ~17:00Z. `fontaine-tiny10k` LIVE local — step ~220
@ 21:17 (22.4 st/min window), 99% util, 15.6 GiB vram, ~1.1/15
GPU-h; **host RAM 122/221 GiB used, 98 available — the
workers-10/prefetch-2 amendment is holding** (OOM class closed).
First post-relaunch probe lands @500 ~21:3xZ (ignore the stale
16.46@500 row predating 21:03Z). Endpoint ~05:1xZ 08-10 → panel_v2
→ Δ_capacity read ~06:3xZ.

**Steering**: `read` empty; `history -n 5` surfaced an owner **👍 on
the 21:03 OOM-recovery + deep-dive-plan post** — lightweight
agreement with the recovery call and the piece, recorded per the
08-05 reaction protocol, no reply owed. 13:48Z gate default (let
run, gate 310) governs adamc.

**Done**: (1) step-10,000 gate judged (Status — the first of adamc's
two dated kill-line checkpoints is behind us). (2) Clock-hallucination
audit: the 20:47 work session closed at a real 21:15:31Z (commit
72e2016 push time) but stamped 21:45/21:5x — queue.json
`updated_utc` was 30 min in the FUTURE; fixed there + in the head
entry below (ack/link times corrected to 21:04Z/21:14Z from Discord
history). (3) Host-RAM check per the OOM class (Status). (4) Queue
validate green depth 3 (9 open).

**Next**: `run_work_next` armed (21:16 marker) → chained work
session → `queue_cli.py next` → `lit-radar-0817` (CPU, GPU-busy
window). tiny10k endpoint ~05:1xZ 08-10 → chained panel_v2 →
Δ_capacity readout. adamc endpoint ~08-12 ~17:00Z → chained k4l2
panel. MolmoAct2 follow-up arms remain owner-decision items.*

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames),
3rd launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3
rungs (+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
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
call — no endpoint, no chained evals)**). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 21:24–21:4xZ (work, bounded; 0 new GPU-h — adamc
rides 31.8/310, tiny10k 1.4/15; explore): lit-radar-0817 closed in
~15 min wall clock — 4 deep reads + refill sweep as 5 concurrent
subagents, 4 Papers pages (armnetbench, safecast, reflex, legato,
compression-gap; MolmoAct2 slot pre-satisfied), every hook needed
corrections (3 loud: ArmnetBench label-count + missing checkpoints,
SAFECAST not-offline + sub-coin-flip on flow policies, Legato
smoothness→completion-time). Ideas #6/#9/#16/#19/#22 fed; #6 gains
a go/no-go gate (probe separability vs ArmnetBench labels), #19 a
cost-model split (draws share one trunk prefill). Refill: 12/16
candidates were corpus dups (pool drying — instrument + angle notes
in the 0818 item); 4 clean hooks, no spares. tiny10k relaunch
sanity: probe @500 = 16.78 vs pre-OOM 16.46. One self-caught
future-dated stamp corrected. check 599; Space pushed.

Session 2026-08-09 21:43–21:5xZ (tick, babysit; 0 new GPU-h — adamc
rides 32.3/310, tiny10k 1.5/15): quiet green tick. adamc step
10,640 @ 23.1 st/min, probe 11.06@10500 mild uptick above the
10.63@9500 run-best (recede-precedent class, record-only). tiny10k
step 800 @ 20.2 st/min on projection; host RAM 134/221 used, 86
GiB available — amendment holds. No steering (read = own post only,
no new reactions). Queue green depth 3 (9 open); run_work_next
already armed at 21:43 for lit-radar-0818.
