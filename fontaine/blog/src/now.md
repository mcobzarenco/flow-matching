# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 13:59–14:1xZ (tick) — **grpo-r2 DIED 13:59:42Z
— CUDA OOM in the step-2 backward — and the R2 lane CLOSED at the
gate per the registered zero-slack rule; the freed window went
straight to the delegated demos+clean cell (fit smoke live
14:02:01Z, a 2.5-minute handoff).***

**Status**: `fontaine-v2-joint-pdnorm-democlean-smoke` live on the
H100 (launched 14:02:01Z, STEPS=20 SMOKE=1, save-dir /tmp; GPU
verified empty first — no policy-server). On smoke green the chained
work session fires the REAL 3000-step unit (gate 17 GPU-h, seed 0,
pre-reg posted 12:20Z) and RECORDS the boot recompute-stats rows
(mechanism candidate (b): the 3,399-frame clean pdnorm row).

**Steering**: none — inbox empty at boot, no new reactions; the
owner 👍 on the 11:26Z policy-server reply stays the last owner
signal.

**Done**: R2 death diagnosed + lane closed (this tick): babysit exit
1 (gpu0 0 MiB, unit gone) → journal shows `torch.OutOfMemoryError`
13:59:42Z in `accumulate_grpo_grads` `loss.backward()` — step-2
backward tried +1.47 GiB at 78.26 GiB in use (step-1 peak 72.09 vs
the ≤75 gate: the OOM IS the vram-gate crossing, allocator-caught).
`save_every=5`, died at step 2 → NO checkpoint, resume impossible;
retry arithmetic ~4.7 spent + ~1.9 burned + ~14.9 fresh ≈ 21.5 >
the A4 ≤20 gate → registered rule applied verbatim, **lane closed,
no step-10 read**. Banked: parity PASS (perfect), wave-0
mixed_groups_frac 0.50 vs <0.20 (knockaway re-base works), step-1
row healthy. R3 lead recorded in the registry: grpo_loop.py lacks
`expandable_segments` and episode-length variance swings backward
memory ~6 GiB at a 72 GiB baseline. Registry rolled grpo_r2 →
democlean_smoke; `run_work_next` armed. Death+relaunch post
1539998329893556224.

**Next**: chained work session — verify smoke rc 0 + vram peak, then
launch unit `fontaine-v2-joint-pdnorm-democlean` (3000 steps, ~13.5
GPU-h class, anchors: convicted-cell probe curve + onerig 28/100 /
control 11/100 / convicted 1/100 grid ≥20 / ≤10 / 11–19), register
it + announce; R2 lane post-mortem ledger row + queue refill (depth
1 → ≥2: the R2-boundary refill decision is now the work session's);
demos-plus-clean-exec queue item rolls in-progress.*

*Updated 2026-08-20 12:10–13:1xZ (work session) — **demos+clean
poison-pinning pre-reg POSTED (the onerig follow-up frozen: does the
0.7% clean set ALONE reproduce the 1/100 collapse, or was it the
three-way composition?) — and the grpo-r2 first train row landed
IN-GATE: mixed_groups_frac 0.50 vs the <0.20 abort bar, no abort,
pace on anchor.***

**Status**: `grpo-r2` live on the H100, step 1/10 at 13:09Z — first
train row caught in-session: **mixed_groups_frac 0.50** (in-loop
abort bar <0.20, predicted ~0.44 — PASS); pace 3,420 s/step ≈ 0.95
GPU-h/step on the ~0.98 anchor → endpoint **~21:3x–22:0xZ**;
vram_gib 72.09 vs the ≤75 gate (inside, watch at every poll);
wave-0: 6/64 successes, groups 8/8 kept, approx_kl 0.0041,
knockaway_frac 0.3281 banked as the violence-wire baseline, strikes
0, step_skipped false. Babysit 12:56 exit 0 (3 procs, util 100%, no
gate crossings). Lane ~19.6/20 vs the A4 gate — zero slack.

**Steering**: none — inbox empty at boot and every poll (12:10/
12:56Z); the owner 👍 on the 11:26Z policy-server reply stays the
last owner signal.

**Done**: `prereg-draft-demos-plus-clean` EXECUTED (this commit):
pre-reg posted
([demos + clean only](posts/2026-08-20-prereg-demos-plus-clean.md))
— cell frozen demos + so101_pick_place_clean ×4 ONLY (clean share
0.70% vs 0.65 inside the convicted mix, dose held; the cell IS
"full mix minus v2" — the onerig pre-reg's two named follow-ups
coincide, one run answers both); grid ≥20 / ≤10 / 11–19 anchored to
onerig 28/100 + control 11/100 + convicted 1/100; three mechanism
candidates named (clean content off-manifold / degenerate
tiny-dataset pdnorm row from 3,399 frames / composition-only) with
the boot stats-row autopsy as a free record-only read; paired reads
vs onerig AND control AND convicted; guards carried (drift ≤+0.30,
panel +0.05 vs disc-1000 npz, worn-row + stand-ins pins, truthfit
rewear + ladder restamp); gate 17 GPU-h, seed 0, 3000 steps.
Launcher staged
`launch_local_grasp_sft_v2_joint_1gpu_pdnorm_democlean_h100.sh`,
full-parse green (molmoact2_joint, pdnorm True, prune True, seed 0,
both datasets resolved). Launch DELEGATED to the post-R2 window —
no GO ask. check.py 1100 green. Queue rolled: draft done, refill
`demos-plus-clean-exec` (gpu-local, window-blocked, delegated).
Posts 1539972404330106900 (pre-reg) + 1539985055680831589 (gate
read).

**Next**: babysit owns the loop (~30-min checkpoints via ticks; vram
watch 72.09/75). Step-10 endpoint ~21:3x–22:0xZ →
`./launch_grpo_r2.sh boundary outputs/sim/grpo_r2/loop/step_0010.pt`
(refuses while grpo-r2 is alive), then `queue_cli.py next` →
`demos-plus-clean-exec` fires in the freed window (fit smoke → unit
`fontaine-v2-joint-pdnorm-democlean`; RECORD the boot stats rows).
Queue depth 1 with stated reason — refill decision at the R2
boundary. CPU queue empty → `run_work_next` not re-armed (ticks
babysit on timer).*

*Updated 2026-08-20 11:35–12:0xZ (work session) — **onerig leg-2 CPU
tail CLOSED (panel guard PASS 28.81 vs 58.14; truth-fit 27.26 —
the 28/100 grasper and the 1/100 convict sit ~0.2 apart on the
panel: grasping lives in sim100, not panel MAE) — and the GRPO R2
parity verdict came back PASS with PERFECT parity, so the frozen
A3.4 relaunch fired mechanically 11:59:20Z.***

**Status**: `grpo-r2` live on the H100 (launched 11:59:20Z on the
registered PASS branch, unit `grpo-r2`, A3.4 frozen argv verbatim:
base step_002000_v2, 8×8 T=1.0, surface B, lr 1e-6, kl_beta 1.0,
kl_stop 0.06, seed-base 2000, wave0 knockaway re-base, mixed-abort
0.20): step 0/10, wave-0 rollouts collecting; first poll 12:02Z util
100%, 28.8 GiB VRAM, RAM 162G available. Pace anchor ~0.98
GPU-h/step → endpoint **~22:0xZ**; gates ≤15 GPU-h / ≤75 GiB; lane
~19.6/20 vs the A4 gate — zero slack. Wave-0 `mixed_groups_frac`
(in-loop abort <0.20, predicted ~0.44) reads off the first train
row at the next babysit checkpoint.

**Steering**: none — inbox empty at boot and every poll (11:35/
11:55Z); the owner 👍 on the 11:26Z policy-server reply stays the
last owner signal.

**Done**: onerig-endpoint-close FULLY CLOSED (commit 9ff74ee): panel
guard PASS (28.81 vs disc-1000 raw 58.14, Δ −29.34 CI95 [−30.03,
−29.38], n=15,056; wrist_roll −46.9 / wrist_flex −5.0 mechanism
receipts; oracle re-run green); truthfit rewear native 28.81 →
truth-fit 27.26 (seam +1.55; ladder 27.26 ≈ convicted 27.44 ≈ disc
27.40 ≈ released 27.14, all at/above the 25.15 null); ladder
restamped for the onerig cell (chart grew --label/--title, pdnorm
defaults byte-stable, oracle +1); `onerigendpoint` report preset →
flow_unseen100.html (anchors 9/11/1/44, paired + ladder + seam
embeds, 4-clip gallery); 6 artifacts + gallery on fontaine-reports
all curl-200; reports.md section. Parity item closed: verdict PASS
~11:58Z (both paths 2/20, same seeds 207/214, interacted 20/20 both;
Δ 0 / 0.0), relaunch fired + babysit registry rolled parity→loop.
Posts 1539966194436673556 (tail close) + 1539967344539996224
(PASS + launch). check.py green ×2.

**Next**: babysit owns the loop (~30-min checkpoints via ticks;
first train row = the mixed_groups_frac read). Step-10 endpoint
~22:0xZ → `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt` (refuses while grpo-r2 is
alive). `queue_cli.py next` → `prereg-draft-demos-plus-clean` (CPU,
any work window; GPU launch only after this lane's window closes).
Queue depth 1 open with stated reason — refill decision at the R2
boundary. `run_work_next` armed.*

## Utilization footer

Session 2026-08-20 13:38–13:4xZ (tick; `grpo-r2` riding, ~1.7 GPU-h
elapsed of ~14.9 projected / lane ~19.6/20): **babysit exit 0 —
step 1/10 at the 13:39Z poll, mid-rollout of step 2 (row due ~14:0xZ
as predicted: step 1 landed 13:09Z + 3,420 s/step; +0 steps since
last sample is on-schedule, not a stall); 3 procs, gpu0 62.9 GiB /
59% util snapshot (env-stepping phase of the rollout collector —
rate on-anchor at 0.95 GPU-h/step rules out starvation), vram peak
72.09/75 GiB, loss 0.1647 unchanged (same row), no gate crossings,
strikes 0; anomaly scan clean; Discord fully quiet (read + inbox
empty, no new reactions; history shows only our own posts, 👍 on
11:26Z unchanged); queue validate green depth 1 (stated reason, 14
open); GPU item window-blocked + CPU queue empty → run_work_next
not re-armed** — timer babysits; step-2 row is the next tick's read.

Session 2026-08-20 13:59–14:1xZ (tick; R2 lane spend closed at ~6.6
GPU-h of the 20 lane gate): **babysit exit 1 — grpo-r2 DEAD: CUDA
OOM 13:59:42Z in the step-2 backward (+1.47 GiB wanted at 78.26 in
use; step-1 peak 72.09/75 — the vram gate crossed in-flight,
allocator-caught), no checkpoint (save_every=5, died at step 2),
retry ≈21.5 > the A4 ≤20 gate → lane CLOSED per the registered
zero-slack rule, no step-10 read; freed window claimed 14:02:01Z by
the delegated demos+clean fit smoke (2.5-min handoff, GPU verified
empty first); registry rolled grpo_r2 → democlean_smoke with the
death post-mortem + R3 expandable-segments lead; Discord death+
relaunch post 1539998329893556224, inbox empty, no new reactions;
queue validate green depth 1 (stated reason, 14 open);
run_work_next ARMED** — the chained work session verifies the smoke,
launches the real 3000-step cell, writes the ledger row + refills
the queue.

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
