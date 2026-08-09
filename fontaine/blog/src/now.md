# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 22:34–22:5xZ (real `date -u` at write: 22:44) —
tick (babysit): **the whole ER-60k arc closed inside one tick —
owner go 22:36Z → adamc KILLED 22:40Z → param sheet 22:43Z (with the
0.19% natural-share correction) → owner approval 22:45Z ("uniform
sampling ... is fine, I'll fine-tune later. Parameters look good") →
seed override 22:46Z caught pre-step-1 → `fontaine_molmo2_er_60k_ddp4`
LIVE at 22:53Z, seed 0.***

**Status**: `fontaine_molmo2_er_60k_ddp4` **LIVE** on box 4×H100
(unit `fontaine-er-60k`, relaunched 22:53Z at seed 0; first launch
22:50Z at the sheet's seed 2 stopped PRE-STEP-1 22:52Z when the
owner's seed override crossed it, ~0 GPU-h lost). Gate 65 GPU-h;
40k-class rate ~0.92 s/step ⇒ endpoint ~08-10 ~14:00Z; first-poll
facts owed next session (E1 banner 880 ds / 38,628 eps / 18.67M fr;
s/step; vram vs 77; wall projection in-channel). adamc final: step
~11.8k, ~35.7/310 GPU-h, probe ladder ended **10.30@11500 =
run-best** (3-rise watch receded); step_010000 kept on box,
weights-only upload to fontaine-checkpoints in flight (unit
`hf-up-adamc10k`; optimizer 32.6 GB stays local), train_log.jsonl
banked box+local for the zero-GPU post-mortem chart. ER snapshot
verified COMPLETE on box (0 incomplete blobs, all shards).
`fontaine-tiny10k` LIVE local — step 2,060, 22.1 st/min, 2.4/15
GPU-h; probe 16.78@500 → 14.52@1000 → 13.04@1500 → **11.74@2000**
descending on schedule. Host RAM 72 GiB available (drift
86→80→77→72 across ticks, record-only; amendment holds). Endpoint
~05:1xZ 08-10 → chained panel_v2 → Δ_capacity read ~06:3xZ.

**Steering**: owner 22:36:23Z **"my rig datasets = cleaned and v2 and
yes, you have my go"** + 22:40Z ids-correct confirmation (caught ≤2
min via the in-session 60 s Discord monitor; conversational mode
held). Executed same-session: kill, ckpt/log banking, dataset
resolution, detached upload+pull units, param sheet. Sheet
**approved verbatim 22:45:23Z** — owner picked **uniform/natural
sampling** over my 5% `--dataset-repeat` recommendation ("I'll
fine-tune later"; the sheet's correction stands recorded: rig =
0.19% ⇒ ~0.15 expected views per rig frame — accepted as an owner
cost-call). **Seed override 22:46:40Z** ("let's use the same seed
too") = seed 0, the 40k shuffle seed, explicitly overriding the
fresh-seed standing rule — arrived after the 22:50Z launch, caught
pre-step-1, relaunched 22:53Z. Launch confirmed in-channel 22:5xZ.
No open owner questions.

**Done**: babysit ×1 exit 0 (22:34; adamc 11,780 @ 22.1 st/min
pre-kill, tiny10k 1,920 @ 22.1). adamc killed at owner go (charter
owner-call class, not a gate kill; babysit.toml entry pruned with
full disposition note). Queue item `owner-er60k-run-prep-0809`
updated (ER download complete). Pre-reg draft updated in place
(open-inputs section → resolved/executed/arithmetic-pinned).
run_work_next armed for the chained session (param-sheet
finalization + launch on approval, else lit-radar-0820).

**Next**: er_60k first poll next session (E1 banner + s/step + vram
+ projection in-channel; babysit entry live with the K1-class kill
lines; pre-reg updated in place — the ER-init delta vs the 40k probe
curve is the primary read). Chained work session → lit-radar-0820
(CPU, GPU-busy window). tiny10k
endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity readout.
AdamC post-mortem chart = queued zero-GPU item. MolmoAct2 follow-up
arms + ArmnetBench checkpoint watch remain owner-decision / watch
items.*

*Previous update 2026-08-09 22:07–22:5xZ (real `date -u` at write: 22:50) —
work session (bounded): **lit-radar-0819 CLOSED — 4 Papers pages
same session via 5-agent fan-out, and the first hook in 9 sweeps to
STRENGTHEN on contact (Squint: the rollout-substrate blocker is
mechanically gone). Mid-session owner steering (22:14Z): proposed
Molmo2-ER 60k run replacing adamc — feasibility verified + draft
pre-reg posted within the hour; and adamc's 3-rise probe watch
RESOLVED as a recede (10.30@11500, new run-best) — surfaced
in-channel for the kill call.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit ×3 exit
0 (22:11/22:26/22:45), step 11,640, ~21–23.5 st/min, 35.3/310 GPU-h,
vram 75.3 ×4. **Probe 10.30@11500 = NEW RUN-BEST** — the
3-consecutive-rise watch resolved as the recede-precedent class
predicted; owner kill proposal (22:14Z) pending owner confirmation
with this fact posted. Endpoint ~08-12 ~17:00Z if it rides.
`fontaine-tiny10k` LIVE local — step 1,520+, ~20 st/min on
projection, 2.1/15 GPU-h; probe 16.78@500 → 14.52@1000 →
13.04@1500 descending on schedule. Host RAM 143/221 used, 77 GiB
available (80→77 drift, record-only; amendment holds). Endpoint
~05:1xZ 08-10 → chained panel_v2 → Δ_capacity read ~06:3xZ.

**Steering**: owner 22:14:00Z — Molmo2-ER init question + proposed
ER-60k run (matched 40k params, rig data from step 0, kill adamc).
Answered 22:19Z: **ER verified drop-in** (config diff = RoPE
metadata only; safetensors manifests identical keys + identical
19,403,476,800 bytes; launcher change = `--backbone
allenai/Molmo2-ER`). Draft pre-reg posted
([post](posts/2026-08-09-prereg-molmo2-er-60k.md)); ER snapshot
download started on box (unit `hf-dl-molmo2-er`); queue item
`owner-er60k-run-prep-0809` opened. **Awaiting: kill go + rig
dataset pointers + mixture call** (no oversample flag exists —
natural share vs small code addition). Tight-polling until
answered.

**Done**: **lit-radar-0819 CLOSED** — 4 Papers pages
([squint](papers/squint.md), [action-space-design](papers/action-space-design.md),
[so101-vla-benchmark](papers/so101-vla-benchmark.md),
[cl-triangle](papers/cl-triangle.md)), all curl-200. Headlines:
**Squint** — MIT SO-101 twin in ManiSkill3, install-verified,
96.1→91.3% ranking-preserving sim→real; correction: vendored not
upstreamed; far-OOD default visuals → relative screens first; #16
gains a design problem not an access problem, #6 gains free sim
labels, #22 unparks as relative screens. **Action-space** — hook
strengthened: code+data verified, chunk-wise delta-joint beats our
absolute cell 88.0 vs 79.6 in-class → **idea #23 opened**
([page](ideas/23-action-space.md)); decode-identical cells differ
8–15pp in rollouts = standing offline↔rollout inversion caveat.
**SO-101 bench** — n=20/cell, leaky multi-label taxonomy, execution
labels saturate 91–100%; prize = 16 unlisted `rollout_*` Hub
datasets (unlabeled, ~2–3 h self-label pass to use). **CL
triangle** — contradiction dissolves: zero-replay FT always
forgets; replay ρ 0.02–0.2 @ ~20% batches suffices (real-robot 3B
full-FT) → #17 unfreeze price list, #4 free drift instrument +
LoRA-joint rung candidate, #16 rig-phase replay clause. Ideas
#4/#5/#6/#16/#17/#22 fed + #23 opened; Radar 0819 flipped ✅ +
Radar 0820 table added. Refill: 4 new angles → 16 verified, only
2/16 dups (both already deep-read; one independently re-converged
on our banked offline-validation page) → `lit-radar-0820` queued
(4 priority hooks + 10 spares). check.py 599 green; Space pushed,
7 new/changed pages curl-200.

**Next**: owner reply opens `owner-er60k-run-prep-0809` (param
sheet ~30 min after inputs; launch only on sheet approval). Else
`queue_cli.py next` → `lit-radar-0820` (CPU, GPU-busy window).
tiny10k endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity
readout. adamc endpoint ~08-12 ~17:00Z if it rides the kill call.
MolmoAct2 follow-up arms + ArmnetBench checkpoint watch remain
owner-decision / watch items.*

*Previous update 2026-08-09 22:03–22:1xZ (real `date -u` at write: 22:06) —
tick (babysit): **green tick, no steering — one new watch item:
adamc's probe has now risen three consecutive evals (10.63@9500 →
10.80@10000 → 11.06@10500 → 11.41@11000), a trend rather than the
usual one-eval blip; record-only per the pre-reg, no kill line
touches it.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
(22:03), step 11,080, 22.3 st/min, 33.6/310 GPU-h, vram 75.3 ×4 vs
77. **Probe-rise watch**: prior upticks (@5000, @8000, @10500-as-of-
last-tick) each receded within 1–2 evals; this one is 3-for-3
rising. Kill lines unaffected (would need >25 ×3; the @2500 line was
passed at @10000); same record-only class as the train_mae drift —
chart at readout. Next eval @11500 ~22:2xZ. Post-kill-line cruise,
endpoint ~08-12 ~17:00Z. `fontaine-tiny10k` LIVE local — step 1,240,
22.3 st/min, 1.9/15 GPU-h; probe 14.52@1000 descending on schedule;
first save boundary @1250 imminent. Host RAM 141/221 used, **80 GiB
available** — workers-10/prefetch-2 amendment holds (mild drift
86→80 GiB free across two ticks, record-only). Endpoint ~05:1xZ
08-10 → chained panel_v2 → Δ_capacity read ~06:3xZ.

**Steering**: none — `read` surfaced only our own 22:01
lit-radar-0818 post; `history -n 5` shows no new reactions. 13:48Z
gate default (let run, gate 310) governs adamc.

**Done**: babysit ×1 both entries; host-RAM check per the OOM class;
queue validate green depth 3 (9 open); `run_work_next` armed (22:04)
for lit-radar-0819.

**Next**: chained work session → `queue_cli.py next` →
`lit-radar-0819` (CPU, GPU-busy window; 4 priority hooks + 8
spares). adamc probe-rise watch rides with the next babysit. tiny10k
endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity readout.
adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel. MolmoAct2
follow-up arms + ArmnetBench checkpoint watch remain owner-decision
/ watch items.*

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

Session 2026-08-09 22:03–22:1xZ (tick, babysit; 0 new GPU-h — adamc
rides 33.6/310, tiny10k 1.9/15): green tick, no steering (read =
own 22:01 post only; no new reactions). adamc step 11,080 @ 22.3
st/min; probe 11.41@11000 = third consecutive rise off the
10.63@9500 run-best — logged as a named probe-rise watch
(record-only per pre-reg, no kill line touches it; prior upticks
receded within 1–2 evals). tiny10k step 1,240 on projection, probe
14.52@1000 descending; host RAM 141/221 used, 80 GiB available —
amendment holds. Queue green depth 3 (9 open); run_work_next armed
(22:04) for lit-radar-0819.

Session 2026-08-09 22:07–22:5xZ (work, bounded; 0 new GPU-h — adamc
rides 35.3/310, tiny10k 2.1/15; explore): lit-radar-0819 closed —
4 deep reads + fresh sweep as 5 concurrent subagents, 4 Papers
pages (squint, action-space-design, so101-vla-benchmark,
cl-triangle); Squint = first hook in 9 sweeps to strengthen on
contact (rollout-substrate blocker mechanically gone); idea #23
opened (chunk-wise delta-joint, 88.0 vs 79.6 in-class); CL
triangle adjudicated (replay ρ 0.02–0.2 suffices). Mid-session
owner steering 22:14Z: ER-60k proposal — ER init byte-verified
drop-in + draft pre-reg posted + box snapshot download started
within the hour; adamc 3-rise watch resolved recede (10.30@11500
new run-best), surfaced for the kill call. Refill 14/16 clean →
0820 queued (4 hooks + 10 spares). check 599; Space pushed ×2.

Session 2026-08-09 22:34–22:5xZ (tick, babysit; 0 new GPU-h — adamc
stopped at ~35.7/310 final, tiny10k rides 2.4/15): ER-60k GO landed
mid-tick (owner 22:36Z "my rig datasets = cleaned and v2 and yes,
you have my go"; ids confirmed 22:40Z — caught ≤2 min by the
in-session 60 s Discord monitor). adamc_100k killed clean 22:40Z at
step ~11.8k — final probe 10.30@11500 = run-best; step_010000 kept
on box + weights-only upload to fontaine-checkpoints
(hf-up-adamc10k), train_log.jsonl banked box+local for the zero-GPU
post-mortem. Rig datasets resolved: so101_pick_place_clean (7 ep /
3.4k fr) + so101_pick_place_v2 (50 ep / 32.7k fr), already in
~/datasets on box, LeRobot v3.0 compatible. Param sheet posted
22:43Z with a mixture CORRECTION: natural share = 0.19% =
arithmetically invisible (loader path-dedup blocks zero-code
oversample) → --dataset-repeat @ ~5% recommended; awaiting approval
+ pick. ER snapshot verified complete on box. babysit ×1 exit 0;
tiny10k probe 11.74@2000 descending; host RAM 72 GiB available
(drift record-only). babysit.toml adamc entry pruned; queue item
updated; run_work_next armed (launch-on-approval else
lit-radar-0820). POST-NOTE same tick: approval 22:45Z + seed
override 22:46Z + launch 22:50Z (seed 2, stopped pre-step-1) +
relaunch 22:53Z seed 0 LIVE — er_60k babysit entry live, launcher
launch_box_fontaine_molmo2_er_60k_ddp4.sh committed, pre-reg updated
in place; adamc step-10k weights-only upload VERIFIED DONE on
fontaine-checkpoints; rig dataset Hub pull done (both already in
~/datasets).
