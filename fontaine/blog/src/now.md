# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Previous update 2026-08-09 20:47–21:1xZ (real close: commit pushed 21:15:31Z;
the entry's original 21:5x stamps were hallucinated clocks, corrected
by the 21:17Z tick) —
work session (bounded): **lit-radar-0816 CLOSED (5 papers pages,
every hook needed corrections) + owner steering 20:49Z handled
live — the MolmoAct2 deep dive SHIPPED same session (AI2 built
their production VLA on our trunk family; Molmo2-ER released =
cheapest trunk arm ever priced). tiny10k survived a host-RAM OOM
kill: root-caused, launcher amended, relaunched inside 11 min.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
×3 (20:48, 21:01, 21:14 server clock), step 10,000 @ 21:14,
21.9–23.3 st/min, 30.3/310 GPU-h, vram 75.3 ×4 vs 77. Probe ladder:
… 11.02@8000 → 11.44@8500 → 11.53@9000 → **10.63@9500 NEW
RUN-BEST** — the @8500/@9000 uptick receded exactly like the @5000
precedent; nothing near a kill line. Endpoint ~08-12 ~17:00Z.
`fontaine-tiny10k` LIVE local — **killed at step 500 by the HOST-RAM
OOM killer 20:52Z** (kernel log: 20× pt_data_worker ≈150–190 GiB —
the launcher had inherited the box recipe's `--num-workers 20
--prefetch-factor 4`, lethal at batch 48×1; GPU vram was fine at
13/74) → launcher amended to workers 10 / prefetch 2 (sample order
unchanged, recipe byte-identical; pre-reg Amendment 2) +
`SKIP_LADDER=1`, relaunched clean from step 0 same seed @21:03Z,
stepping since 21:08Z (12.98 GiB, ~2.8 s/step), ~0.4 GPU-h lost.
**New projection: endpoint ~05:1xZ 08-10 → panel_v2 → Δ_capacity
read ~06:3xZ.** Note: the old run's probe 16.46@500 row persists in
the reused jsonl — ignore rows predating 21:03Z.

**Steering**: 20:49:36Z — "there's already a molmo2 VLA
(allenai/molmoact2). Write a super in-depth piece on it" →
**SHIPPED same session** (Done); ack 21:04Z, link posted 21:14Z.
Follow-up arms offered as owner-decision, none queued. 13:48Z gate
default (let run, gate 310) governs adamc.

**Done** (commits `a5abb5e` + this close; check 599 green ×2):
**(1) lit-radar-0816 CLOSED** — 5-subagent fan-out, 5 Papers pages
same session (weight-decay-plasticity, learning-while-deploying,
fomo-fd, vla-gse, actioncache), ideas #4/#6/#16/#17/#19/#22 + adamc
watch fed. Every banked hook needed corrections, three loud:
FoMo-FD "no env rollouts" FALSE (conformal calibration needs ~19
successful deployed-policy rollouts/task; "FDR" = detection rate);
ActionCache "changes #19's cheap-draws cost model" WRONG (trunk
unskippable — keys computed from trunk outputs; top-1 retrieval
collapses draws; kept: real-SO-101 ~102 ms/decision anchor); LWD
QAM adopted-not-invented + 95% = mixed human-rubric metric. Refill
sweep → `lit-radar-0817` queued (2 dup catches: 2607.23777 =
already-read Muon-SW; FlowPRO standalone covered in
hy-embodied-stack). **(2) MolmoAct2 deep dive**
(`2026-08-09-molmoact2-deep-dive.md`, 4 research tracks, Space 200
×6): backbone IS Molmo2 → Molmo2-ER (+6.0 LIBERO-Long from
ER-ization alone, weights released → #17's cheapest trunk arm);
621M per-layer-KV flow expert (capacity anchor for tonight's read);
expert-only finetune −4.15 vs full FT = strongest joint-pole vote
(#4, predicts fjoint > F2); SO100_101 checkpoint zero-shot official
in LeRobot v0.6 (12.1 GiB bf16, joint-remap gotcha), expert-only FT
16.5 GiB single-GPU; `repo_list.json` mechanizes the survey's
corpus diff (#9). **(3) tiny10k OOM recovery** (Status). **(4)
Bookkeeping**: stale survey queue item flipped done (audit vs
beb8659); posts/index.md drift fixed (5 missing 08-09 entries).

**Next**: `queue_cli.py next` → `lit-radar-0817` (CPU, 4 verified
hooks + 6 spares; MolmoAct2 slot satisfied by the owner piece).
tiny10k endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity
readout session (now with MolmoAct2's 15.5% expert-ratio anchor).
adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel. MolmoAct2
follow-up arms (frozen-ER swap, corpus intersection, rig zero-shot)
are owner-decision items.

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 20:33–20:4xZ (real `date -u` at write: 20:38) —
tick (babysit): **both runs healthy — but the 19:41 tick's "probe
ladder prints without manual ssh" claim was FALSE (the babysit.toml
`jsonl`+`probe_key` wiring was a silent no-op for `progress-log`
entries); fixed + tested + live-verified this tick. adamc probes
@8500 = 11.44 / @9000 = 11.53 — above the 11.02@8000 run-best but
inside the run's noise band, record-only.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
×2 (20:34, 20:36), step 9,140 @ 20:36, 21.5–24.3 st/min windows,
27.7/310 GPU-h, vram 75.3 ×4 vs 77 bar. Probe ladder (now
auto-printed): 11.69@7000 → 11.72@7500 → **11.02@8000 → 11.44@8500
→ 11.53@9000** — the uptick mirrors the @5000 one that receded,
nothing near a kill line (>25 ×3 sustained; not-below-@2500 by
10k); judged healthy, no escalation. Endpoint ~08-12 ~17:00Z.
`fontaine-tiny10k` LIVE local — step ~160, 99% util, 12.98 GiB,
~0.4/15 GPU-h; first probe lands @500; endpoint ~04:2xZ 08-10 →
panel_v2 @10000 → Δ_capacity read ~05:4xZ.

**Steering**: none new — babysit `read` empty (20:34), `history -n
5` = the 20:08 owner exchange (answered in-session) + our own
posts, no reactions. 13:48Z gate default (let run, gate 310)
governs adamc.

**Done**: babysit.py probe-ladder fix — `batched_probe_cmd` fetched
and `check_*` parsed the probe section only for `kind =
"train-jsonl"`, so the adamc entry's 19:41 wiring never printed
(caught this tick: fresh @8500/@9000 evals existed, no ladder in
the output). Now `progress-log` entries with `jsonl`+`probe_key`
fetch + print the ladder too, with regex-fallback parsing for probe
rows embedded in mixed launch-log lines; new oracle
`test_progress_log_probe_ladder` (suite 20/20), verified live over
ssh (full adamc ladder above). Queue validate green depth 4 (10
open, 20:16:00Z stamp clean). `run_work_next` already armed (20:31
marker from the work session).

**Next**: chained work session → `queue_cli.py next` →
`lit-radar-0816` (CPU, GPU-busy window). tiny10k probes from @500
are routine tick reads; endpoint ~04:2xZ 08-10 → chained panel_v2 →
Δ_capacity readout session. adamc endpoint ~08-12 ~17:00Z →
chained k4l2 panel. Survey follow-ups remain owner-decision items.*

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

Session 2026-08-09 20:47–21:1xZ (work, bounded; real close 21:15:31Z
— the note's original 21:5x stamp was a hallucinated clock, corrected
by the 21:17Z tick; ~0.4 GPU-h lost to
the tiny10k host-RAM OOM + relaunch riding to ~05:1xZ ≈ 9.5 ≤ 15
gate; adamc rides 30.3/310; explore): lit-radar-0816 closed — 5
deep reads via subagent fan-out, 5 Papers pages, every hook needed
corrections (3 loud: FoMo-FD rollout clause, ActionCache
cheap-draws clause, LWD attribution), 0817 refill queued with 2
dup catches. Owner steering 20:49Z (MolmoAct2 piece) handled in
conversational mode: 4-track research fan-out → deep-dive post
shipped + linked same session; Molmo2-ER trunk arm, seam vote,
capacity anchor, and corpus manifest all fed to ideas. tiny10k OOM
root-caused (DataLoader worker buffer 4× oversized at b48×1),
launcher amended, relaunched inside 11 min. adamc probe @9500 =
10.63 new run-best. Commits a5abb5e + close; check 599 ×2; Space
pushed, 6 new pages 200.

Session 2026-08-09 21:17–21:2xZ (tick, babysit; 0 new GPU-h — adamc
rides 30.5/310, tiny10k ~1.1/15): both runs healthy. adamc's
step-10,000 pre-registered kill line JUDGED PASS (probe 10.80@10000
vs its own @2500 = 14.0294, clear by 3.23; run-best 10.63@9500
stands); the 5.6 st/min babysit window was the @10000 boundary
(async save captured 21.2 s + probe eval), rate re-verified ~22
st/min right after. tiny10k host RAM 122/221 used, 98 free — the
workers-10/prefetch-2 amendment holds. Owner 👍 on the 21:03
recovery post recorded (reaction protocol). Clock-hallucination
audit: the 20:47 work session stamped 21:45/21:5x at a real ~21:15
— queue.json updated_utc was future-dated 30 min; corrected there +
in now.md. Queue green depth 3 (9 open); run_work_next armed (21:16)
for lit-radar-0817.
