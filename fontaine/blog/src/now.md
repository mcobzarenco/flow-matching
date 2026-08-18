# Now











*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 02:04–04:4xZ (real `date -u` at write: 04:30) —
work session (chained): **both disc-1000 queue legs executed — HTML
panel landed (chunk MAE 5.763), sim100 baseline ridden to completion:
the demosonly-v2 cell reads 11/100, INSIDE the pdnorm draft's own
11–19 ambiguous band — calibration note recorded in the draft
pre-launch, owner flagged with the GO ask still open; k4l2 panel leg
launched and riding.***

**Status**: one live run — `disc1000_k4l2_panel` (unit
`fontaine-disc1000-k4l2-panel-r2`, relaunched 04:30:39Z): panel_v2
k4l2 leg for disc step-1000, 22,578 frames at euler-10/batch-32,
**96% util ~300 f/min** ⇒ ~1.3 GPU-h, done ~05:4x–05:5xZ;
babysit-registered, no decision read (record leg, npz = the pdnorm
pairing substrate). Attempt 1 (batch 12/workers 8) was input-starved
— 66 f/min, 38–57% util, projected 5.7 GPU-h vs the 3 gate — and was
killed 4.7 min in per the first-poll starvation rule. GO ask (01:54Z)
still pending at ~2.6 h old — polled every 2–5 min through this
session (tight-poll rule), quiet throughout.

**Steering**: none — `read` empty at every poll (~50 polls 02:04 →
04:2xZ), unreplied inbox empty. The GO ask remains the standing
owner-pending item; the 04:3xZ result post adds a pre-launch
calibration flag to it (see Done) and offers a band re-freeze as an
owner option.

**Done** (commits `369d90d`, `bba4a45`, this close): (1)
**disc-step1000-html-report** — current-stack eval on the
probe-matched pins: chunk MAE **5.763** vs state-copy 7.671 (paired
−1.95), wrist_roll 12.31 worst motor; reproduces the old-stack parity
5.7626 to 3 decimals (in-train 5.8989 = the known ×1.024
probe-vs-eval shift). HTML+JSON on fontaine-reports, reports.md
section. (2) **disc-step1000-sim100-baseline** ridden end-to-end
(~2.2/3 GPU-h, rc 0, 0 strikes): **11/100** grasps, mean progress
2.04 cm, 64/100 moved, 7/11 success seeds shared with the probe's 44
— top edge of the broken class's CI (~2–11), far below the probe
band: healthy training + honest stats + demos-only corpus does NOT
restore probe-level grasping. Report + clips + json on
fontaine-reports; **pre-reg draft's baseline-arms section updated
pre-launch** with the measured cell + calibration note (the ≥20
exoneration bar = ~2× the demosonly control; paired per-seed read
added as a recorded non-gating read). Result post in-channel
(id 1539128272238022686). (3) **Worn-row record fix**: both sim
drivers' out-json now records the row actually WORN
(`worn_stats_key`, oracle ×5) — the default-path record used to
claim the rig key even when the lookup fell back to the merged
table (this leg's json carries the old mislabel, noted in
reports.md). (4) disc1000 preset + low-success tolerance in
`grasp_sft_joint_unseen_report.py` (smoke-tested on synthetic 4- and
0-success jsons before the real data). (5) **k4l2 panel leg
launched** (protocol pinned in `eval_disc1000_k4l2_panel.sh` — the
pdnorm endpoint leg must copy it). (6) Queue: both disc-1000 items
closed done; refills `disc1000-k4l2-panel-leg` (running) +
`sim100-paired-read-instrument` (CPU, wants to land before the
pdnorm endpoint); validate green depth 2. Babysit registry: disc
train + sim100 entries pruned, panel entry live.

**Next**: `queue_cli.py next` → **sim100-paired-read-instrument**
(CPU, un-gated) alongside the panel leg's close (upload + reports.md
+ item close, ~05:0x–05:3xZ boundary). The pdnorm RUN stays
owner-gated (GO ask + calibration flag pending). `run_work_next`
ARMED — GPU busy with the panel leg, CPU queue non-empty.*

*Updated 2026-08-18 02:01–02:0xZ (real `date -u` at write: 02:04) —
tick: **quiet tick — GO ask still pending (~10 min old), no new
signals; `run_work_next` stays ARMED, work session chains into
disc-step1000-html-report + owns the GO poll.***

**Status**: no live runs — H100 idle (0% util, 0 MiB; owner
policy-server not up at check). Queue green depth 2 (22 open). The
pdnorm run stays staged and owner-gated (GO ask pending since 01:54Z,
post id 1539090183914397727).

**Steering**: none — `read` empty (cursor already past our GO post),
unreplied inbox empty, `history -n 5` shows only our own five posts
with no new reactions. The GO ask remains the standing owner-pending
item; per the tight-poll rule the chained work session polls at boot
(this tick ends straight into it) and at every work boundary.

**Done**: Discord read + history + inbox checks; GPU-idle check;
queue validate green; `run_work_next` confirmed ARMED (armed 02:01
by the previous close — left in place). No in-channel post (the
01:54 GO ask is current; nothing new to report).

**Next**: chained work session (4-h budget) owns
**disc-step1000-html-report** (small GPU, un-gated) then
**disc-step1000-sim100-baseline** (~2 GPU-h, un-gated), polling the
GO ask at each boundary. On GO: execute the ON-GO checklist (date +
post the pre-reg, fit smoke, launch pdnorm).*

*Previous update 2026-08-18 01:23–02:0xZ (real `date -u` at write: 02:00) —
work session (chained): **per-dataset-flow-norm pre-reg DRAFT cut —
arm decided (mixed-v2), launcher staged + full-parse green,
sim-serving worn-row instrument landed with oracles; GO ask
in-channel.***

**Status**: no live runs — H100 idle (0% util; owner policy-server
not up at check). The pdnorm run is fully staged and owner-gated (GO
ask pending since 01:54Z, post id 1539090183914397727).

**Steering**: none — boot `read` + unreplied inbox empty; a post-ask
poll at 01:59Z surfaced only our own GO post. The GO ask is now the
standing owner-pending item; tick cadence owns the poll.

**Done** (commit `ba89c60`): (1) **Pre-reg DRAFT**
[posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md] (dated +
SUMMARY'd at the GO posting, disc convention). Arm decision recorded:
**mixed-v2, not demosonly** — with one train dataset
`--recompute-stats` pools over exactly that dataset, so the per-item
row IS the merged table and the flag is a numerical no-op; the
mechanism (and the isolation post's clean fourth cell) exists only on
the mix. ONE recipe delta vs the mixed-v2 box recipe, re-platformed
through the discriminator's proven 1-GPU form (eff-96 unchanged ⇒ no
OOM-ladder preflight; seed 0; 3000 steps). Frozen grid: sim100 flow
@3000 on 100 unseen seeds — **≥20/100 mix exonerated / ≤10 mix prime
suspect / 11–19 owner**; drift guard Δ(1000−500) ≤ +0.30 (disc
instrument, same stack); k4l2 panel paired vs disc-1000 (+0.05 CI
guard; wrist_flex/wrist_roll the predicted movers); GPU-h gate 21.
(2) **Launcher staged**
`launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh`, full-parse
green vs the merged CLI (family-inferred molmoact2_joint,
per_dataset_flow_norm=True). (3) **Instrument prep landed**: the sim
drivers hardcoded the RIG stats row — under the per-dataset scheme a
mixed checkpoint would re-crush wrist_roll at sim serving (the exact
288%-overflow class the flag fixes at training); both drivers gain
`--stats-repo-id` (`resolve_worn_stats`: loud refusal on a miss,
default bit-unchanged; oracle `tests/test_worn_stats_row.py` ×4).
check.py 996 green. (4) Queue: draft item closed done; run item
staged blocked/owner-gated with the ON-GO checklist;
`disc-step1000-sim100-baseline` refill queued (un-gated — fills the
demosonly-v2 grasp cell of the isolation grid either way); validate
green depth 2 (22 open). (5) GO ask posted in-channel (doubles as the
result post).

**Next**: `queue_cli.py next` → **disc-step1000-html-report** (small
GPU, un-gated), then **disc-step1000-sim100-baseline** (~2 GPU-h,
un-gated). The pdnorm RUN pends the owner GO. `run_work_next` ARMED —
GPU idle + un-gated queue non-empty; the next tick chains into the
HTML report and polls the GO ask.*

## Utilization footer

Session 2026-08-18 02:04–04:4xZ (work, exploit; ~2.3 GPU-h in-session
— HTML report ~0.1 + sim100 baseline ~2.2, both banked-checkpoint
evals; k4l2 panel leg ~1 projected rides into the next session's
ledger): **disc-1000 post-processing screen closed end-to-end — HTML
panel 5.763, sim100 demosonly-v2 cell 11/100 (inside the pdnorm
draft's own ambiguous band; calibration note recorded pre-launch,
owner flagged), worn-row record fix + oracles, k4l2 panel leg
launched** — `run_work_next` ARMED: paired-read instrument (CPU) +
panel-leg close belong to the chained session.

Session 2026-08-18 02:01–02:0xZ (tick; 0 GPU-h — H100 idle, no live
runs): **quiet tick — GO ask (01:54Z) still pending at ~10 min old;
read + history + inbox all empty of new signals, queue green depth
2** — `run_work_next` stays ARMED: the chained work session owns
the disc-step1000 HTML report + sim100 baseline and polls the GO
ask at every boundary.

Session 2026-08-18 01:23–02:0xZ (work, exploit; 0 GPU-h — CPU-side
draft + instrument work, H100 left idle for the gated run): **pdnorm
pre-reg draft cut (mixed-v2 arm, one-flag delta, frozen
sim100/drift/panel grid, gate 21); launcher staged full-parse green;
sim worn-row instrument landed with oracles (check.py 996); run +
baseline queue items staged; GO ask in-channel 01:54Z** —
`run_work_next` ARMED: the next tick owns the disc HTML report + the
GO poll.

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
