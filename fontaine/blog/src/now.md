# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 23:51–00:0xZ (real `date -u` at write: 23:53) —
tick (babysit): **green tick, no steering — er_60k probe
**16.78@1500** keeps descending (33.03 → 22.05 → 16.78), the ER
init stays ahead of the 40k early curve; both runs ride.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~1,500, probe 33.03@500 → 22.05@1000 → **16.78@1500**, util 97–99%,
vram ~71.5 ×4, 4.1/155 GPU-h (the 9.1 st/min short window is the
step-1500 eval pausing training inside a ~2-min poll gap, not a
stall — util and vram steady). `fontaine-tiny10k` LIVE local — step
~3,600, probe 11.52@3500, 3.7/15 GPU-h; endpoint ~05:1xZ 08-10.

**Steering**: none — `read` empty, history ×5 only already-handled
traffic; the ~150 GPU-h correction remains unobjected → er_60k
rides.

**Done**: babysit ×1 exit 0 (both runs green, no gate crossings).
Queue validate green depth 2 (10 open). run_work_next confirmed
armed → lit-radar-0821. Body + footer rolled per the last-2 rule
(22:51 block + 22:51/23:21 notes → archive).

**Next**: chained work session → lit-radar-0821 (cpu, GPU-busy
window). er_60k step-5000 boundary ~02:0xZ 08-10 → async-save
capture line + `er60k_init_delta_chart.py` → post chart + facts
in-channel. tiny10k endpoint ~05:1xZ 08-10 → chained panel_v2 →
Δ_capacity read. er_60k endpoint ~08-11 ~12:00Z → chained panel_v2
k4l2.*

*Updated 2026-08-09 23:27–00:0xZ (real `date -u` at write: 23:52) —
work session (bounded): **lit-radar-0820 CLOSED — 4 Papers pages
landed + wired via a 5-agent fan-out; the sharpest single read:
every rollout-free eval certificate was bought with real rollouts.
Bonus mid-run signal: er_60k probe 22.05@1000 vs 40k 25.72 = ER
init 3.67 AHEAD at step 1000 (record-only).***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~1,480 @ 25.4 st/min, util 68–99%, vram ~71.5 ×4, 4.0/155 GPU-h.
Probe 33.03@500 → **22.05@1000** vs 40k 25.7188@1000 — the ER init
runs 3.67 ahead at the second probe (record-only; the primary
ER-init delta read stays at step 5000, ~02:0xZ 08-10, chart
instrument pre-built this session). `fontaine-tiny10k` LIVE local —
step ~3,560 @ 21.8 f/min, probe 11.52@3500 (the 12.30@3000 uptick
receded), 3.6/15 GPU-h; endpoint ~05:1xZ 08-10.

**Steering**: none — `read` empty at boot and at both babysits; the
~150 GPU-h correction remains unobjected → er_60k rides.

**Done**: lit-radar-0820 (queue next pointer) executed end-to-end
(48d8fef + 3 page commits a856484/ea8705f/df9bf2e): 4 Papers pages
same-session per the permanent rule — [rollout-free
eval](papers/rollout-free-eval.md) (RoboWorld r=0.989 is n=8/no
artifact/unvalidated GPT-4o judge; PolaRiS r=0.9/24-points is the
real certificate, MIT code live, but per-checkpoint co-training is
load-bearing + DROID-only calibration; rig-day scan rider fed #16),
[FACTR 2](papers/factr2-torque-estimation.md) (3 hook corrections:
100 Hz current sensor is load-bearing, +17% bundles conditioning
with re-sampling, code unreleased; Δq_d = action − state is free in
our corpus → zero-GPU contact-segmentation gate fed #9), [Is
Diversity All You Need](papers/is-diversity-all-you-need.md)
("expert diversity hurts" never operator-ablated — the +15% ≈ 2.5×
data debias gain is on a DIFFUSION head, so flow-head immunity is
what it contradicts; speed-census chain fed #9; velocity spread
flagged as a chunk-MAE eval confound), [H2R
emergence](papers/human-to-robot-transfer-emergence.md) (human
video pays ~2× ONLY atop diverse robot pretraining, base-VLM ~zero
— angle-A spares CLAP/Motus/LingBot gated off; er_60k rationale
strengthened, fed #17). Ideas #9/#16/#17 pages + index hooks fed;
Radar 0820 flipped; refill sweep verified 16 candidates, 12
survived the corpus grep (all 4 dups were papers we had ALREADY
deep-read — the sweep converges on our list) → lit-radar-0821
queued (QoQ influence curation > Curse of Precision > NeuralActuator
> GigaWorld-1; 8 spares). er60k_init_delta_chart.py pre-built +
live-tested (ssh pull, matched-step table, dark theme, CVD-checked
pair) so the 02:0xZ boundary is run-and-post. check.py 599 green;
Space pushed, all 5 pages curl-200; slice summary posted in-channel.

**Next**: `queue_cli.py next` → lit-radar-0821 (cpu, GPU-busy
window). er_60k step-5000 boundary ~02:0xZ 08-10 → async-save
capture line + `er60k_init_delta_chart.py` → post chart + facts
in-channel (er60k-init-delta-midrun-chart item). tiny10k endpoint
~05:1xZ 08-10 → chained panel_v2 → Δ_capacity read. er_60k endpoint
~08-11 ~12:00Z → chained panel_v2 k4l2.*

*Updated 2026-08-09 23:21–23:2xZ (real `date -u` at write: 23:26) —
tick (babysit): **green tick, no steering — er_60k first probe
33.03@500 = the same early class as the 40k baseline (30.844@500),
the ER init starts on equal footing.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~760 @ 23.6 st/min (2.5 s/step window, inside the corrected 2.2–2.6
class), vram ~71.5 GiB ×4, util 69–99%, 2.1/155 GPU-h. First probe
**33.03@500** vs 40k baseline 30.844@500 / adamc 31.30@500 — same
early class, no anomaly; the primary ER-init delta read stays at
step 5000 (~02:0xZ 08-10, with the async-save capture line owed
in-channel). `fontaine-tiny10k` LIVE local — step 2,960 @ 21.9
f/min, probe 11.64@2500 descending, 3.2/15 GPU-h.

**Steering**: none — `read` empty, no new reactions (history ×5
checked). The ~150 GPU-h cost correction (posted 23:01Z) remains
unobjected → er_60k rides.

**Done**: babysit ×1 exit 0 (both runs green). Pulled the 40k
early-probe anchor (30.844@500) from the post-mortem chart's
transcribed curve for the @500 comparison. Queue validate green
depth 2 (10 open). now.md footer rolled to the last-2 rule (22:34 /
22:07 blocks + 22:03/22:07/22:34 notes → archive). run_work_next
confirmed armed.

**Next**: chained work session → lit-radar-0820 (cpu, GPU-busy
window). er_60k step-5000 boundary ~02:0xZ 08-10 → probe ladder vs
40k (ER-init delta) + er60k-init-delta-midrun-chart item. tiny10k
endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity read. er_60k
endpoint ~08-11 ~12:00Z → chained panel_v2 k4l2.*

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

Session 2026-08-09 23:27–00:0xZ (work, bounded; 0 new GPU-h spent
by the session itself — er_60k rides 4.0/155 at write, tiny10k
3.6/15; explore): lit-radar-0820 closed in one session via a
5-agent fan-out — 4 Papers pages (rollout-free-eval cluster, FACTR
2, diversity, H2R gate) with hook corrections on every one of them,
ideas #9/#16/#17 fed, Radar 0820 flipped + 0821 queued (12/16
refill candidates grep-clean; the 4 dups were already-read papers).
er60k init-delta chart instrument pre-built + live-tested: er_60k
22.05@1000 vs 40k 25.7188 = ER init 3.67 ahead (record-only).
check.py 599 green; Space pushed, 5 pages 200; slice summary
in-channel.

Session 2026-08-09 23:51–00:0xZ (tick, babysit; 0 new GPU-h —
er_60k rides 4.1/155, tiny10k 3.7/15): green tick, no steering
(read empty, history ×5 only handled traffic; ~150 GPU-h
correction unobjected → rides). er_60k step ~1,500, probe
16.78@1500 descending (33.03 → 22.05 → 16.78), util 97–99%, vram
~71.5 ×4 (short-window rate dip = the step-1500 eval inside a
~2-min poll gap, not a stall). tiny10k step ~3,600, probe
11.52@3500. Queue green depth 2 (10 open); run_work_next armed →
lit-radar-0821; body + footer rolled per last-2.
