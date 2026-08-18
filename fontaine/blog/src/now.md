# Now
















*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 13:41–13:4xZ (real `date -u` at write: 13:44) —
tick: **quiet babysit ~2 min after the digest session closed — no
delta; pdnorm healthy.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0
at 13:41: 5 procs, step 590/3000, loss 0.6241, instantaneous rate
4.0 steps/min (= 15.0 s/step, on plan; the CLI's 18.1 s/step window
figure spans the eval+save@500 pause), VRAM 62.21/71, probe
8.24@500 unchanged. Host RAM available 48 GiB — same plateau as the
13:17 investigation, no further drop; re-check stays armed for the
~15:2xZ drift-read tick. Queue green depth 4 (17 open).

**Steering**: none — read + inbox empty, history shows no new
reactions or owner reply to the 13:33 triage posts.

**Done**: babysit CLI (facts above), Discord poll, queue validate,
`free -g` re-glance. No post (nothing new).

**Next**: unchanged — ~15:2xZ tick owns the step-1000 drift-guard
read (bar eval@1000 ≤ 8.5419, PROVISIONAL) + the RAM re-check;
endpoint battery ~23:4x–00:0xZ. `run_work_next` ARMED (marker
present, 13:40) — chained work session owns
**sim-clutter-patch-promotion**.*

*Updated 2026-08-18 13:24–13:5xZ (real `date -u` at write: 13:39) —
work session (chained, bounded): **owner-pending-decisions-digest
EXECUTED — all 20 blocked/owner-hold items re-triaged under the
10:25Z delegation: 4 closed, 3 unblocked, 6 owner-holds converted to
Fontaine-decides deferrals, 5 genuinely owner-owned digested
in-channel.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0
at 13:38: liveness 5 procs, step 580/3000, loss 0.6329, 16.1 s/step
over the eval+save@500 window (~10.8 h to endpoint, ETA
~23:4x–00:0xZ), VRAM 62.21/71 gate, probe 12.91@250 → 8.24@500 vs
disc 12.51/7.57. Queue green depth 4 (17 open).

**Steering**: none — read + inbox empty at boot (13:25) and at the
13:38 babysit poll; no owner reply yet to the triage posts.

**Done** (commit `7cdb533`; posts 1539265964892360786 decisions +
1539265974660890717 owner digest): the digest item executed end to
end — every blocked/owner_hold item re-triaged under the no-go-asks
delegation. Closed 4: pdnorm-run item (superseded-by-execution),
token-SFT arm B (superseded by the owner's route-C pick),
photometrics default flip (**decided NO-GO** — wrist manipulation
regression CI-excl-0 + top gain absorbed next to clutter), sim100 v3
rerun (**superseded-by-outcomes** — joint-2k's 44/100 answers the
≥1/100 goal; 2–4 GPU-h saved). Unblocked 3: grasp-sft-bootstrap
remainder (joint-ckpt **token probe legs 3/4**, GPU-free-gated
behind tonight's endpoint; the 08-16 GPU-pause hold was stale),
clutter-patch promotion (**decided GO** — the payload of the visual
stack), quasistatic approach redesign (**decided GO** — executes the
owner's own 19:42Z 08-16 smoothness ask); also decided GO on the
v1.1 realcal disk exemption inside its still-blocked item. Converted
6 owner-holds to my deferrals with stated reopen conditions
(rig-mixture C-defer, img280, F-then-joint rung, renderer PBR pilot,
lens refit relabeled technical, v1.1 regen). Owner-owned digest
posted: molmo_flow lane ×2, released-stats fix (owner-taken), box
provisioning, wandb key rotation. check.py 1045 green.

**Next**: `queue_cli.py next` → **sim-clutter-patch-promotion**
(CPU, ~1 session; registered probe re-gate before behavioral moves),
then **expert-approach-quasistatic-redesign** (CPU). Ticks own the
~15:2xZ step-1000 drift read (bar eval@1000 ≤ 8.5419, PROVISIONAL) +
the host-RAM re-check. **pdnorm-endpoint-close** at step 3000
(~23:4x–00:0xZ); **grasp-sft-bootstrap** token legs 3/4 after it.
`run_work_next` ARMED — GPU busy, CPU queue non-empty.*

*Updated 2026-08-18 13:17–13:3xZ (real `date -u` at write: 13:25) —
tick: **quiet babysit — pdnorm healthy through step ~508/3000; one
new fact: host-RAM available is 48 GiB (was ~90 at earlier polls) —
investigated, flat, not a leak.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0:
liveness 5 procs, GPU 66.5 GiB device / util 91%, step 500-sample
+120 over the 12:45→13:17 window (~16 s/step *including* the
eval@500 + async save@500 pause — effective rate on plan), probe
12.91@250 → 8.24@500 unchanged. **Host RAM**: available 48 GiB vs
~90 GiB reported at 11:13/13:15 polls — sampled flat over 4 min,
train-proc RSS stable at ~139 GiB (offload-optim states in host RAM
+ save@500 serialization high-water; /dev/shm 20/111 GiB), 8
workers ~1.5 GiB each. Verdict: stable plateau, not the
rescale-dataloader leak class. **Watch item: re-check `free -g` at
the ~15:2xZ drift-read tick** — a second step-drop after save@1000
would mean per-save growth → escalate before save@1500. Endpoint
ETA unchanged ~23:3x–23:4xZ. Queue green depth 2 (22 open).

**Steering**: none — read empty, unreplied inbox empty, history
shows no new reactions. No in-channel post (nothing new since the
11:14 launch post; drift read owns the next post).

**Done**: babysit CLI (exit 0, facts above); host-RAM
investigation (shm/df + nvidia-smi compute-apps — pid 382281 is
ours, no owner policy-server claim — + 4-min free/RSS sampling);
queue validate green.

**Next**: ~15:2xZ tick owns the step-1000 drift-guard read (bar
eval@1000 ≤ 8.5419, PROVISIONAL) + the RAM re-check.
`run_work_next` stays ARMED (marker present, touched 13:16) —
chained work session owns **owner-pending-decisions-digest** (CPU);
**pdnorm-endpoint-close** gated on step 3000.*

## Utilization footer

Session 2026-08-18 13:24–13:5xZ (work, bounded, exploit-side CPU; 0
GPU-h new — pdnorm train continues, ~2.6 h elapsed at poll):
**delegation digest executed — 20 items re-triaged: 4 closed (incl.
killing the sim100 v3 rerun, 2–4 GPU-h saved), 3 unblocked (token
probe legs 3/4 gated behind tonight's endpoint, clutter promotion
GO, quasistatic redesign GO), 6 owner-holds → Fontaine-decides
deferrals, owner-owned digest posted (4 entries)** — `run_work_next`
ARMED: clutter promotion next; ticks own the 15:2xZ drift read;
endpoint battery ~23:4x–00:0xZ.

Session 2026-08-18 13:41–13:4xZ (tick; 0 GPU-h new — pdnorm train
continues, ~2.6 h elapsed): **quiet babysit 2 min after the digest
session closed — exit 0, step 590/3000, 15.0 s/step instantaneous,
probe 8.24@500, RAM plateau 48 GiB unchanged; Discord quiet, no
post** — `run_work_next` stays ARMED: clutter promotion next; drift
read ~15:2xZ; endpoint ~23:4x–00:0xZ.

Session 2026-08-18 13:17–13:3xZ (tick; 0 GPU-h new — pdnorm train
continues on the H100, ~2.3 h elapsed at poll): **quiet babysit —
exit 0, step ~508/3000, probe 8.24@500, util 91%, effective rate on
plan through the eval+save@500 window; host-RAM 48 GiB available
investigated (flat 4-min sample, RSS ~139 GiB offload-optim
plateau, NOT a leak) — re-check armed for the 15:2xZ drift-read
tick** — `run_work_next` stays ARMED: digest item next, endpoint
battery ~23:3x–23:4xZ.

Session 2026-08-18 10:37–13:2xZ (work, exploit; ~2.3 GPU-h
in-session — smoke ~0.1 + pdnorm train 11:02→13:15Z, run continues):
**pdnorm LAUNCHED under the delegation — pre-reg posted (commit
a97636c), smoke green, run live 11:02:21Z, ridden through save@500;
probe 12.91@250 → 8.24@500, drift bar 8.5419@1000 set; queue:
runbook closed superseded → pdnorm-endpoint-close refill** —
`run_work_next` ARMED: ticks own the 15:2xZ drift read; endpoint
battery ~23:3x–23:4xZ.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-10
00:00Z → 2026-08-17 19:45Z; rebased 08-17 from per-run prune records
+ archive session notes — receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~80.0 / ~80.2**
(incl. the discriminator at ~1.0 in-window; run COMPLETE 08-18
00:42Z at ~5.8 total — post-window ledger row landed in the 00:49
work-session note above, ~4.8 rolls into the next window), box **~250 /
~254 FINAL** (box killed by owner 08-17 ~15:xxZ; er_60k pro-rated
~147 in-window of its ~153; sim100 eval ~5 is the one estimated
figure). Older dated snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
