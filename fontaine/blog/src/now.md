# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 20:0x–20:3xZ (work session, chained on the 19:54
tick) — **R2 boundary-legs launcher EXECUTED + CLOSED (982cecd): the R2
endpoint is now one command end-to-end — `./launch_grpo_r2.sh boundary
<overlay.pt>` materializes the servable endpoint dir, fires the three
A3.4 legs sequentially as one detached unit, and chains the banked
verdict instrument. The endpoint read needs zero new code.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 440/3000 at
the 20:24Z poll, loss 0.6515 (falling, −0.06 over the interval), ~15.7
s/step effective over the last 90 steps (eval pauses included; a hair
above the 15.1–15.4 band), 62.21 GiB vs the 71 gate, babysit exit 0.
ETA drifting toward ~07:4x–08:0xZ 08-20 (vs ~07:0x–07:1x registered —
watch at the drift read; still noise-level, no re-registration).
Step-1000 drift read ~22:4x–23:0xZ tonight (tick duty, READ not kill,
Δ ≤ +0.30 raw).

**Steering**: none — read + inbox empty at boot and both babysit polls
(20:00, 20:24).

**Done**: `grpo-r2-boundary-legs-launcher` EXECUTED + CLOSED (982cecd,
check.py 1099 green): (1) `boundary` subcommand — three legs (greedy
token sim100 / sampled T=1.0 sim100 / flow unseen100 euler-10, seeds
0–99, anchors' exact driver + substrate pins) sequential in ONE
detached unit chaining `grpo_r2_boundary_verdict`; refuses while unit
`grpo-r2` is alive, without a PASS preflight verdict, and on the
pinned base dir. (2) The missing seam found by the git audit:
the loop banks trainable-only `step_NNNN.pt` overlays but the anchor
serving path loads self-contained VLA dirs — new
`grpo_r2_materialize_endpoint.py` applies the text-surface overlay
onto the base's backbone_text via `write_checkpoint` (atomic,
validated, hard-linked untouched parts; 6 oracles on the tiny VLA
fixture). (3) parse-check extended: the legs' exact argv through the
driver's own parser + the verdict's provenance guards on synthesized
configs — launcher and verdict cannot drift apart. (4) Registered-pin
correction (git-audited, recorded in the launcher + queue): NO
`--stats-repo-id` on the boundary legs — the spelled
so101_pick_place_v2 row exists only on the retired step_002000 dir;
on the v2 base the explicit pin would be REFUSED at load, and the
default lookup is the lane's registered serving convention (the
preflight PASS wore `<merged-table>`).

**Next**: `queue_cli.py next` → `grpo-r2-parity-read-and-relaunch`
(gpu-local, post-onerig window). Boundaries: onerig step-1000 drift
read ~22:4xZ 08-19 (tick), onerig endpoint ~07:4xZ 08-20 →
`onerig-endpoint-close`, then the R2 parity read + relaunch in the
freed window (A5 gate, no GO ask); at the R2 endpoint, the boundary is
`./launch_grpo_r2.sh boundary outputs/sim/grpo_r2/loop/step_0010.pt`.*

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

## Utilization footer

Session 2026-08-19 20:0x–20:3xZ (work, chained; `onerig` riding ~2.5
GPU-h elapsed of ~13 expected / gate 17, CPU item in the GPU-busy
window): **`grpo-r2-boundary-legs-launcher` EXECUTED (982cecd, check.py
1099 green) — boundary subcommand (3 legs, one detached unit, chained
verdict, triple refusal ladder) + the endpoint materializer the item
implied but git audit showed missing + parse-check oracle wired to the
verdict's own guards + stats-pin drift corrected against the live
metadata** — exploit (registered lane instrument); queue green depth 2
(15 open). Onerig healthy both polls (step 440, loss falling, 62.2
GiB). Disk 216 GB free.

Session 2026-08-19 19:54–19:5xZ (tick; `onerig` riding, ~1.5 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 330/3000,
loss 0.6878 falling, 15.6 s/step cumulative (band-adjacent, warmup
washing out of the average), 62.2 GiB / 99% util, no gate crossings;
Discord fully quiet (read + inbox empty, no reactions);
`run_work_next` already armed at the 19:53 work close — fast close to
hand off** — queue green depth 3 (16 open). Disk 216 GB free (93%).

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
