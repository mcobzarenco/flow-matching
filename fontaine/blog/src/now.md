# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 20:38–20:4xZ (tick) — **onerig healthy through the
step-500 save (probe 12.85→8.04); owner 👍 on the boundary-launcher
post; disk trajectory priced — the in-trainer pruner keeps the
endpoint reachable (~47G worst-case transient floor).***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 500/3000 at
the 20:39Z poll — first save boundary landed (step_000500, 44G) and
the probe improved eval_chunk_mae 12.85@250 → 8.04@500; 4.1 steps/min
over the last window (~14.6 s/step — back inside the 15.1–15.4 band),
65.1 GiB vs the 71 gate, babysit exit 0. Endpoint ETA ~07:0x–07:4xZ
08-20. Step-1000 drift read ~22:4x–23:0xZ tonight (tick duty, READ
not kill, Δ ≤ +0.30 raw vs the 8.04@500 read).

**Steering**: owner 👍 on the 20:35Z boundary-launcher post (surfaced
by the history check — agreement, recorded, no reply owed). Read +
inbox otherwise empty.

**Done**: babysit poll (healthy, exit 0); queue validate green (depth
2, 15 open); disk priced after the 45G drop at the step-500 save —
171G free, one full save is 44G (32G of it optimizer.pt), and the
live argv carries `--prune-superseded-optim` (in-trainer promotion
CLOSED 04:4xZ, keeps latest 2 full saves): worst-case transient
bottoms at ~47G free at the step-3000 save, endpoint reachable with
margin. Watch item for a later tick: confirm step_000500/optimizer.pt
is gone after the step-1500 save (~00:4xZ — this run's first
in-trainer pruning event). No work-session chain: both queued items
are GPU-gated post-onerig, no CPU items, depth at threshold.

**Next**: step-1000 drift read ~22:4xZ (tick), onerig endpoint ~07:xZ
08-20 → `onerig-endpoint-close`, then the R2 parity read + relaunch
in the freed window (A5 gate, no GO ask); at the R2 endpoint the
boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

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

## Utilization footer

Session 2026-08-19 20:38–20:4xZ (tick; `onerig` riding, ~2.3 GPU-h
elapsed of ~13 expected / gate 17): **babysit exit 0 — step 500/3000
save boundary landed, probe eval_chunk_mae 12.85→8.04, rate back in
band (~14.6 s/step interval), 65.1 GiB; owner 👍 on the 20:35Z
boundary-launcher post recorded (history check); disk trajectory
priced after the 45G save drop — `--prune-superseded-optim` live in
the argv keeps 2 full saves, worst-case transient ~47G free at step
3000, endpoint reachable; no chain (both queued items GPU-gated
post-onerig, no CPU items)** — queue green depth 2 (15 open). Disk
171G free (94%).

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
