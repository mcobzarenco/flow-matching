# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-19 17:5x–18:4xZ (real `date -u` at write: 18:35, same
work session continued) — **the R2 relaunch exposed a DEEPER break and
was KILLED 18:06:48Z: the loop's serving stack (MolmoAct2DiscreteStack
+ hardcoded official shim, er60k-era) is inert on v2 corrected-table
checkpoints — the A4 substrate fix was necessary but not sufficient.
R2 lane PARKED on a serving-parity fix (now a launch gate).
Demos+one-rig took the GPU 18:22:47Z on the banked owner GO.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` LIVE (unit
`fontaine-v2-joint-pdnorm-onerig`, launched 18:22:47Z after a green
fit smoke) — step 10+ at warmup pace (20.8 s/step; smoke measured
~15.4, mixed-cell precedent ~15.1–15.3), 66.4 GiB / 83–100% util, no
starvation, RAM 91G, disk 214G free. Endpoint ETA ~07:0xZ 08-20;
step-1000 drift read ~22:3xZ tonight (READ not kill). Gates vram 71 /
17 GPU-h. R2 lane: killed relaunch burned ~0.33 GPU-h (lane total
~4.0); artifacts banked (`loop_wave0abort_patched/`, killed `loop/`,
`wave0_diag/`).

**Steering**: none — read + inbox empty at every poll (one babysit
read was piped through head against the never-truncate rule;
recovered immediately: inbox empty, history clean, nothing lost).
Kill + root-cause post 18:08Z, onerig launch post 18:36Z — decide +
announce, no GO asks.

**Done (this half)**: (1) R2 relaunch ridden to the step-0 eval row
(+17 min, on the re-measured gap): 0/20 with ALL 20 scenes bit-frozen
under VERIFIED standins (meta + worker seam) — vs the greedy anchor
leg's 59/100 visible displacement, P≈2e-8 → loop path inert
independent of substrate; my wave0_diag probe had confounded
driver-with-substrate (no sequential-patched control). (2) Class
pinned: R1-B on the released ckpt through the SAME loop stack
interacted (knockaway 0.33–0.45, wave successes 3–4) — the break is
v2-checkpoint-specific; suspicious seam spotted
(`grpo_replay._batch` reuses ACTION quantiles as `state_stats`).
(3) Run killed 18:06:48Z on that evidence (saved ~1 GPU-h to the
wave-0 re-fire); registry pruned with the full postmortem. (4)
`grpo-r2-serving-parity-fix` queued as the R2 launch gate;
`grpo-r2-boundary-legs-launcher` blocked on it. (5)
`demos-plus-one-rig-exec` EXECUTED (closed superseded-by-execution
per the pdnorm precedent): smoke green, unit live, preamble verified
(2 datasets, clean dropped, v2 ×4 = 6.30% share), babysit entry
live; `onerig-endpoint-close` refilled (frozen grid ≥20 / ≤10 /
11–19; anchors demosonly 11, mixed 1).

**Next**: `queue_cli.py next` → CPU item
`grpo-r2-serving-parity-fix` (diff the two serving paths on v2,
parity oracle, launcher-gated); onerig boundaries: step-1000 drift
read ~22:3xZ 08-19 (tick duty), endpoint ~07:0xZ 08-20 →
`onerig-endpoint-close`. R2 relaunch only on parity green +
re-registration (A5).*

## Utilization footer

Session 2026-08-19 17:5x–18:4xZ (work, same session cont.; ~0.33
GPU-h banked — killed R2 relaunch — + `onerig` live from 18:22:47Z,
~13 expected / gate 17): **R2 relaunch step-0 eval read 0/20 ALL
scenes frozen under verified standins (P≈2e-8) → loop serving stack
convicted on v2 checkpoints (R1-B/released interacted through it) →
KILLED 18:06:48Z, lane parked on `grpo-r2-serving-parity-fix` (launch
gate); demos+one-rig fired on the banked GO (smoke green, preamble
verified, 66 GiB / 83–100%)** — exploit (registered lane + integrity);
queue green depth 2 (16 open). Disk 214 GB free (93%).

Session 2026-08-19 16:21–17:5xZ+ (work, chained; ~1.44 GPU-h banked
this session — aborted patched wave ~1.2 + diagnosis probe 0.24 —
plus `grpo_r2` relaunched 17:46:56Z riding, ~18.5 lane total
expected / gate ≤20 per A4): **boundary-reads instrument landed
(0a405a2) → wave-0 gate FIRED 17:19Z (mixed 0.0) → substrate bug
convicted (loop rendered patched, anchors standins; probe A/B on the
same seeds: 6/8 interact under standins) → fix + A4 + RELAUNCH
17:46:56Z (4914f80)** — exploit (registered lane + integrity fix);
queue green depth 2 (15 open: instrument closed,
`grpo-r2-boundary-legs-launcher` refilled). Disk 227 GB free (92%).

Session 2026-08-19 16:17–16:2xZ (tick; `grpo_r2` riding, ~0.2 GPU-h
elapsed of ~14 expected): **first poll of live R2 healthy in the
declared startup window (procs+GPU liveness, babysit exit 1 = known
train.jsonl gap until ~17:1xZ); Discord quiet, no gate crossings;
`run_work_next` already armed — fast close** — queue green depth 2
(15 open). Disk 227 GB free (92%).

Session 2026-08-19 13:07–16:1xZ (work, chained; ~2.25 GPU-h banked —
preflight leg 0 — + `grpo_r2` live from 16:10Z, ~14 expected / gate
≤15): **owner agree-with-recs mid-session → A3 ACTIVATED end-to-end:
launch kit landed (570e53e) → preflight PASS (sampled 8/100 vs greedy
7) → R2 FIRED 16:10:02Z; demos+one-rig GO banked, fires at the next
free GPU boundary; disk sweep executed in the ride window (~41G
freed, 16/16 bitwise audit)** — exploit (registered activation +
infra debt); queue green depth 2 (15 open: `grpo-r2-launch-kit` +
`disk-retirement-sweep-banked-sources` closed,
`grpo-r2-boundary-reads-instrument` refilled). Disk 231 GB free
(92%).


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
