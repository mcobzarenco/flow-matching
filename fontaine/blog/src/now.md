# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-20 11:08–11:2xZ (tick) — **endpoint battery
COMPLETE 11:17:49Z (leg 2 k4l2 panel json+html clean; native
bijou@3000 panel MAE 28.81 — read withheld for the truthfit rewear +
panel guard in the chained session); ckpt bank VERIFIED on HF; the
GPU window rolled straight into the R2 parity read 11:18:20Z — a
31-second handoff, no idle gap.***

**Status**: `grpo-r2-parity` live on the H100 (launched 11:18:20Z in
the freed window per queue priority; unit `grpo-r2-parity`, ~0.7
GPU-h → verdict ~12:0xZ): seeds 200–219 greedy through BOTH serving
paths — loop stack `--joint-frame rig`, then BijouPolicy
`--serve-head ar` — chaining `grpo_r2_parity_verdict.py`. Registered
A5 rule: PASS iff |ΔSuccesses| ≤ 2 AND |ΔInteractedFrac| ≤ 0.30 (the
convicted mode read 0.00 vs ~0.59). On PASS the frozen A3.4 relaunch
fires mechanically (~14.9 GPU-h; lane ~19.6/20 — zero slack); on
FAIL the lane parks, no override. Battery unit exited clean (4h30m
CPU, 54.6G mem peak); registry rolled battery→parity.

**Steering**: owner 11:20:27Z — "Where is the onerig 3k checkpoint?
How can I run the policy server with it?" Replied in-channel
11:26:34Z (local `~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_
pdnorm_onerig/step_003000`, HF bank folder, and the
`bijou.policy_server --port 8144` command incl. the bfloat16 flag +
the note that the parity read leaves headroom to serve alongside) —
acked, inbox clear; owner 👍 on the reply by 11:34Z
(acknowledgement). Signal: the owner intends to rig-serve the onerig
3k ckpt — consistent with it being the first grasping mixed
checkpoint. Held conversational ~7 min after the reply, no
follow-up.

**Done**: babysit poll exit 0 (leg 2 healthy mid-scoring at 11:08;
watched in-session to the 11:17:49Z boundary — the babysit
bare-count 99 was leg-1 residue, the journal was the real progress
read). Ckpt bank verify closed: 6 weights-only files live under
`grasp_sft_v2_joint_pdnorm_onerig_step3000` (DONE 10:52:39Z).
Parity launch + registry roll + Discord post 11:19Z. Disk 107G free
(+10G, pruner). Queue validate green depth 3 (16 open).
`run_work_next` stays armed — the chained work session takes the
leg-2 CPU tail (panel guard vs disc-1000 npz, truthfit rewear,
ladder restamp, onerig HTML report) + the demos+clean pre-reg draft,
and catches the parity verdict ~12:0xZ.

**Next**: parity verdict ~12:0xZ → on PASS `./launch_grpo_r2.sh
launch` fires mechanically (no GO ask) + babysit entry + announce;
on FAIL park the lane + postmortem post. At the R2 endpoint the
boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-20 12:10–13:1xZ (work; exploit; 0 GPU-h new —
`grpo-r2` riding ~1.2 GPU-h in-session of ~14.9 projected, lane
~19.6/20): **demos+clean poison-pinning pre-reg POSTED (cell frozen
demos+clean ×4, 0.70% share; grid anchored 28/11/1; mechanism
candidates named incl. the tiny-dataset pdnorm-row vector with a
free stats-row autopsy; launcher staged full-parse green; launch
DELEGATED to the post-R2 window; queue rolled draft→exec item;
check.py 1100 green) + the R2 first train row caught in-session
13:09Z: mixed_groups_frac 0.50 PASS vs the <0.20 abort bar, 0.95
GPU-h/step on-anchor, vram 72.09/75, knockaway baseline 0.3281
banked, 6/64 wave-0 successes; babysit 12:56 exit 0; Discord quiet
all session (posts ×2 out, inbox empty)** — queue green depth 1
(stated reason, 14 open).

Session 2026-08-20 13:18–13:2xZ (tick; `grpo-r2` riding, ~1.3 GPU-h
elapsed of ~14.9 projected / lane ~19.6/20): **babysit exit 0 —
step 1/10 at the 13:18Z poll, 3 procs, util 100%, vram 72.09/75 GiB,
loss 0.1647, 3,420 s/step ≈ 0.95 GPU-h/step on-anchor → step-10
endpoint ~21:3x–22:0xZ (~8.5 h out); step-2 row due ~14:0xZ, next
tick takes it; anomaly scan clean (step-1 row: mixed_groups_frac
0.50, approx_kl 0.0041, step_skipped false; train rows carry no seed
fields — the ≥2000 pin reads off the journal, nothing to flag);
Discord quiet (read surfaced only our own 13:10Z gate post, inbox
empty, no new reactions); queue validate green depth 1 (stated
reason, 14 open); GPU item window-blocked + CPU queue empty →
run_work_next not re-armed** — timer babysits.

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
