# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*


*Updated 2026-08-13 01:44–01:5xZ (real `date -u` at stamp: 01:47) —
tick, babysit: **quiet tick — no steering, GPU idle as declared; one
housekeeping kill: the stale boxsync loop (polling the dead 08-05
box for 6 days) found and stopped.***

**Status**: no live runs — registry carries the declared reason
(next GPU legs pend owner calls); nvidia-smi 0%/0 MiB, no stray
compute procs. Queue validate green (depth 2, 12 open).

**Steering**: none new — read empty 01:45Z, history-5 shows no
reactions on the 01:10Z probe results post or the 01:40Z lens post.
Open asks unchanged: v3-rerun unhold + arm set, disk-draws sign-off,
GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done**: process sweep surfaced `boxsync_loop.sh` still running
since 08-06 23:44Z in a tmux pane — ssh-polling the retired 4xH100
box (192.222.55.210) every 20 min; box confirmed unreachable
(connection timeout), all its registry entries historical → loop +
hung ssh killed. `run_work_next` re-armed (CPU lanes queued, GPU
idle-by-design). Footer notes >2 rolled to the archive.

**Next**: chained work session → `sim-composite-contact-shadows`
(queue head) or lens leg (b) render path; phase-2 token-GRPO design
memo open. GPU legs launch on owner calls only.*

*Updated 2026-08-13 01:18–01:5xZ (real `date -u` at stamp: 01:41) —
work session (chained by the 01:14 tick): **wrist lens fit leg (a)
DONE — the real lens is measurably not ideal-equidistant.** Plumb-line
θ→r fit on the 150 pinned real wrist frames (pure CPU, no rig time):
optical center 22 px left / 14 px below the image midpoint (~5σ), and
the curve compresses the periphery −12.8 px at the frame corner vs
the deployed equidistant assumption (CI95 [−17.2, −10.0], excludes
0). Results + house chart posted in-channel 01:40Z.*

**Status**: no live runs — GPU idle-by-design (registry carries the
declared reason; next GPU legs pend owner calls). Queue validate
green (depth 2, 12 open).

**Steering**: none new — read empty at boot 01:18Z and at close; no
reactions yet on the 01:10Z probe results post or the 01:40Z lens
post. Open asks unchanged: v3-rerun unhold + arm set, disk-draws
sign-off, GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done** (commit `5581d6d`): `sim-fit-real-lens-model` leg (a) —
plumb-line fit instrument (`fontaine/scripts/fit_lens_plumbline.py`:
Canny → PCA/quadratic-filtered seam chains, 382 chains from 132/150
frames; Nelder-Mead over (cx, cy, k₂, k₄) with center-only /
curve-only decompositions + frame bootstrap), synthetic-recovery
oracles (`tests/test_lens_plumbline.py`, 4 tests), house dark chart
(`lens_fit_chart.py`). Plank straightness RMS 1.07 → 0.90 px;
fitted params are the stage-2 resampler spec for leg (b).
check.py 801 green. Queue item updated with the leg-(a) record.

**Next**: `queue_cli.py next` → `sim-composite-contact-shadows`
(queue head, CPU) or lens leg (b) cubemap→equirect→fitted-lens
render path (same item, CPU-side first); phase-2 token-GRPO design
memo also open. GPU legs launch on owner calls only. `queue.json`
canonical.*

*Updated 2026-08-13 01:14–01:2xZ (real `date -u` at stamp: 01:14) —
tick, babysit: **quiet tick after the probe close** — no owner
messages or reactions, GPU idle as declared, work session chained
for the CPU lanes.*

**Status**: no live runs — registry carries the declared reason
(grpo_signal_probe COMPLETE, unit stopped 01:08:05Z at the cell-5
boundary); nvidia-smi 0%/0 MiB, no stray procs. Queue validate green
(depth 2, 12 open).

**Steering**: none new — read empty, history-5 checked 01:15Z; no
reactions yet on the 01:10Z probe results post. Open asks unchanged:
v3-rerun unhold + arm set (15:13Z 08-12), disk-draws sign-off, cells
3/4 re-queue on owner call.

**Done**: boot audit clean (no orphaned diffs); GPU/registry/queue
verified; `run_work_next` armed — GPU is idle-by-design (next GPU
legs pend owner calls) and CPU lanes are queued, so the tick chains
a work session per no-idle-pauses. 08-12 body entries + older footer
notes rolled to the archive.

**Next**: chained work session → `sim-fit-real-lens-model`
(owner-adopted), `sim-composite-contact-shadows` (queue head), and
the phase-2 token-GRPO design memo (CPU-side first per the frozen
decision rule). GPU legs launch on owner calls only.*

*Updated 2026-08-12 21:39Z–2026-08-13 01:1xZ (real `date -u` at stamp:
01:10 08-13) — work session (the chained session riding the probe):
**GRPO probe: AR signal is REAL and cheap at t=1.0 (0.771 cm vs the
0.25 bar, cost CI includes 0); t=1.6 clears 10× but pays −1.08 cm.
Tripwire fired at cell-1 (measured ~1.13 GPU-h/cell vs ~0.6
assumed) — re-scoped in-channel to cells 1/2/5, cells 3/4 parked.**
Plus: lit 0823 sim-improvement levers closed (3 papers pages), and an
owner steer mid-session — wrist compositing investigated end-to-end
and DECIDED render-only (22:31Z).*

**Status**: GRPO probe cell 5 (SDE a=0.5, the Flow-GRPO trainability
cell) finishing ~01:0xZ; unit stopped at its boundary per the
re-scope; cumulative 3.57 vs the 3.5 gate (announced overage, actuals posted). Cells 1/2 read out
at their boundaries (in-channel 23:0x/00:07Z). Cell 5 (SDE a=0.5): **1.860 cm CLEARS 7.4×**, cost −0.734 CI [−2.240, +0.294], 5b hedge not triggered. **Decision rule: BOTH families clear → token-GRPO (AR, t=1.0) first, Flow-GRPO SDE second, GRPO-on-sim does NOT park.** Unit stopped 01:08:05Z at the cell-5 boundary (0 GPU procs), entry pruned.

**Steering** (live owner thread): (1) 22:21:54Z "investigate
compositing for the wrist camera" → executed same session:
CPU-only feasibility read (`wrist_composite_feasibility.py`,
`d177c0d`) — plate poses spread 20.8 mm/5.1° median (why the static
plate mushed), wrist is table-plane-dominated (median 100% of rays)
so FK+plane-homography is sound, but warp-fill p10 49% before
arm/boat holes ⇒ T-III seam hazard; recommended render-only wrist +
redirect to lens fitting; (2) 22:31:50Z owner adopted the
recommendation → `sim-wrist-compositing` CLOSED as decided, sim100
**amendment 4** documents the channel asymmetry,
`sim-fit-real-lens-model` queued (plumb-line θ→r on existing frames,
no rig time); (3) 22:33:20Z "how does the encoder probe work in
depth?" → two-part in-depth reply 22:41Z. Probe re-scope default
posted 21:58Z, no objection at any boundary.

**Done** (commits `49381ca`…`6897fea` + close-out): (1) **lit
0823 CLOSED** (`ed6ba42`, owner-called): 3 papers pages same session
— composite-shadows (no published pipeline measures the missing-shadow
axis; Re³Sim foreground-realism null; randomize-in-training /
match-in-eval split), fisheye-lens-fitting (scale overfitting as a
distance ruler; cubemap→any-lens MuJoCo pipeline), dr-schedules
(DORAEMON success-throttled entropy max; eval stays at matched
center); 3 ideas hooks (#16 sim lane), `sim-composite-contact-shadows`
queued. (2) Probe instrument: frozen-reads script (`ece2276`), house
dark chart (`d2bde2f`), registry re-scope (`6897fea`). (3) Wrist
compositing decision artifacts (`a5e5784`). (4) Probe results
amendment on the pre-reg page + results post at cell-5 close.

**Next**: `queue_cli.py next` → CPU lanes: `sim-fit-real-lens-model`
(owner-adopted), `sim-composite-contact-shadows` (both probe-gated,
pair on the same harness). GPU idle after the probe stop; cells 3/4
re-queue only on owner call; phase-2 GRPO call per the frozen
decision rule (see the results post). v3-rerun unhold + disk-draws
sign-off still open. `queue.json` canonical.*

## Utilization footer

Session 2026-08-13 01:44–01:5xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design): quiet tick — no owner messages/reactions (01:45Z),
registry reason stands, nvidia-smi 0%/0 MiB, queue green (depth 2,
12 open). Housekeeping: killed the stale `boxsync_loop.sh` (up since
08-06 23:44Z, ssh-polling the dead 4xH100 box every 20 min;
unreachable, connection timeout). `run_work_next` re-armed for the
CPU lanes.

Session 2026-08-13 01:18–01:5xZ (work, chained; 0 new GPU-h — CPU
lane, exploit): `sim-fit-real-lens-model` leg (a) closed — plumb-line
θ→r fit on the pinned real wrist frames (center 22 px off, corner
ray placement −12.8 px vs equidistant, CI-excl-0), instrument +
oracles + chart `5581d6d`, results in-channel 01:40Z. check.py 801
green. GPU legs still pend owner calls.

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
