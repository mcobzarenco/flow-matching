# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 19:54–19:5xZ (tick) — **onerig healthy at step
330; fully quiet tick, fast close to the chained work session.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 330/3000,
loss 0.6878 (falling, −0.03 over the last 10 steps), 15.604 s/step
cumulative — a hair above the 15.1–15.4 band with warmup still in
the average; 62.21 GiB vs the 71 gate, 99% util, 5 procs, babysit
exit 0, no gate crossings. At the current rate ~11.6 h to endpoint →
ETA ~07:3xZ 08-20 (vs ~07:0x–07:1x registered — noise-level, no
re-registration). Step-1000 drift read ~22:3x–22:4xZ tonight (tick
duty, READ not kill, Δ ≤ +0.30 raw).

**Steering**: none — read + inbox empty, history clean (no
reactions; last 5 messages all ours).

**Done**: babysit poll (healthy, exit 0); queue validate OK (depth
3, 16 open); `run_work_next` confirmed armed (19:53Z at the work
close) — GPU busy + CPU item queued
(`grpo-r2-boundary-legs-launcher`).

**Next**: chained work session takes `grpo-r2-boundary-legs-launcher`
(CPU, unblocked by ad70476). Tick duties: 22:3xZ drift read, endpoint
~07:0x–07:3xZ 08-20 → `onerig-endpoint-close`, then the R2 parity
read + relaunch in the freed window (A5 gate, no GO ask).*

*Updated 2026-08-19 19:0x–19:5xZ (work session, chained on the 19:04
tick) — **R2 serving-parity fix EXECUTED + CLOSED (ad70476): the
18:06Z kill's root cause was the phase-4 re-point carrying the
port-era v30→v21 joint-frame shim unconditionally — every bijou-format
table is v3.0-frame, so lift/elbow state bins clamped at +1 every
frame and the inverted chunk map drove the arm out of range. The
mismatch class is now UNREPRESENTABLE at the seam (fingerprint + loud
refusal), parity is oracle-pinned on CPU, and the GPU parity read is
wired into the launcher as the A5 launch gate.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 250/3000 at
the 19:35Z poll, ~15.4 s/step (3.9 steps/min), first probe row
eval_chunk_mae 12.85@250, 67.2 GiB vs the 71 gate, 83% util —
healthy. Endpoint ETA holds ~07:0x–07:1xZ 08-20; step-1000 drift read
~22:3x–22:4xZ tonight (tick duty, READ not kill, Δ ≤ +0.30 raw).

**Steering**: none — read + inbox empty at boot and both babysit
polls (19:07, 19:35).

**Done**: `grpo-r2-serving-parity-fix` EXECUTED + CLOSED (ad70476,
check.py 1093 green): (1) root cause pinned in code AND quantified —
the shim maps sim lift [−103,+29]° to [61,193]° against the v2 table
row [−110,+12]° → normalized state clamps at +1.0 every frame (elbow
likewise), and chunks return through the inverse (lift 90−a, elbow
a−90) → poses outside the trained range every replan; exact kill
telemetry both times. R1-B is no counterexample: it ran the port-era
HF predictor on a genuine v2.1 table where the shim is correct. (2)
The kill post's suspect seam (`_batch` action-quantiles-as-
state_stats) EXONERATED — predict_ar detokenizes under the family
table; batch stats never reach a decode. (3) Fix: `--joint-frame
{auto,rig,v30-to-v21}` on both sim discrete drivers through ONE
resolver (JointFrameTransform literals, test-pinned); auto
fingerprints the state table (conventions doc §4), refuses
unclassifiable tables and explicit-vs-classified mismatches; frame
recorded in meta/rows/out-json. (4) CPU parity oracle
(`tests/test_joint_frame_parity.py`): classifier on the three real
table shapes + refusal semantics + shim math both directions +
loop-vs-BijouPolicy prompt parity BIT-EQUAL on the tiny fixture. (5)
GPU parity read wired as REQUIRED launch gate: `launch_grpo_r2.sh
parity` → `grpo_r2_parity_verdict.py` (registered rule: PASS iff
|Δsucc| ≤ 2 AND |Δinteracted| ≤ 0.30); `launch` refuses without PASS;
frozen argv pins `--joint-frame rig`. (6) A5 on the pre-reg page;
budget: lane ~4.0 spent + 0.7 parity + ~14.9 relaunch ≈ 19.6 vs gate
≤20 — zero slack, any further abort ends the lane. Queue:
`grpo-r2-boundary-legs-launcher` UNBLOCKED (was gated on this fix),
`grpo-r2-parity-read-and-relaunch` refilled (gpu-local, post-onerig
window); depth 3.

**Next**: `queue_cli.py next` → `grpo-r2-boundary-legs-launcher`
(CPU, stageable any window — chained work session). Boundaries:
onerig step-1000 drift read ~22:3xZ 08-19 (tick), onerig endpoint
~07:0xZ 08-20 → `onerig-endpoint-close`, then the R2 parity read +
relaunch in the freed window (A5 gate, no GO ask).*

*Updated 2026-08-19 19:04Z (tick) — **onerig healthy at first
post-warmup read; work session chained for the R2 parity fix.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 130/3000,
15.127 s/step — warmup pace fully resolved into the smoke/mixed-cell
band (15.1–15.4); loss 1.0273, 62.19 GiB vs the 71 gate, 84% util, 5
procs. ~12.1 h to endpoint → ETA holds ~07:1xZ 08-20; step-1000
drift read ~22:3xZ remains the next tick duty (READ not kill, Δ ≤
+0.30 raw).

**Steering**: none — read + inbox empty, history clean (no
reactions).

**Done**: babysit poll (healthy, no gate crossings); queue validate
OK (depth 2, 16 open); `run_work_next` touched 19:04:42Z — GPU busy
+ CPU item queued (`grpo-r2-serving-parity-fix`, the R2 launch
gate).

**Next**: chained work session takes `grpo-r2-serving-parity-fix`
(path diff + parity oracle; the cheap GPU parity read waits for the
post-onerig window). Tick duties: 22:3xZ drift read, endpoint
~07:0xZ 08-20.*

## Utilization footer

Session 2026-08-19 19:54–19:5xZ (tick; `onerig` riding, ~1.5 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 330/3000,
loss 0.6878 falling, 15.6 s/step cumulative (band-adjacent, warmup
washing out of the average), 62.2 GiB / 99% util, no gate crossings;
Discord fully quiet (read + inbox empty, no reactions);
`run_work_next` already armed at the 19:53 work close — fast close to
hand off** — queue green depth 3 (16 open). Disk 216 GB free (93%).

Session 2026-08-19 17:5x–18:4xZ (work, same session cont.; ~0.33
GPU-h banked — killed R2 relaunch — + `onerig` live from 18:22:47Z,
~13 expected / gate 17): **R2 relaunch step-0 eval read 0/20 ALL
scenes frozen under verified standins (P≈2e-8) → loop serving stack
convicted on v2 checkpoints (R1-B/released interacted through it) →
KILLED 18:06:48Z, lane parked on `grpo-r2-serving-parity-fix` (launch
gate); demos+one-rig fired on the banked GO (smoke green, preamble
verified, 66 GiB / 83–100%)** — exploit (registered lane + integrity);
queue green depth 2 (16 open). Disk 214 GB free (93%).

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
