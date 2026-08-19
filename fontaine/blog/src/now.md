# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 04:19–08:0xZ (real `date -u` at write: 07:30) —
work session (chained): **demos+one-rig pre-reg DRAFTED + posted
(owner call flagged); `--prune-superseded-optim` landed first-class
in bijou.train; joint-probe leg 3 COMPLETE — token-unseen 7/100,
BELOW the R2 bar; leg 4 launched after an AR-surface fix.***

**Status**: `grasp_sft_joint_probes` leg 4 (token-base anchor) LIVE —
unit `fontaine-joint-probe-token-base`, relaunched **06:54:56Z** from
the pinned worktree after the 06:34Z first attempt died at load (base
conversion records family `molmoact2_flow`, no AR surface; fix =
derived ckpt `…_vla_jointsurface`, hardlinked weights + family →
`molmoact2_joint` + the joint ckpt's parameter-free ar_decoder config,
corrected table worn; 1-seed AR smoke green pre-relaunch). Babysit
07:26 exit 0: 3 procs, 12.8 GiB / 42%, 24 seeds by 07:26 (~0.75/min),
RAM 190 GiB, disk 290 GB. rc **~08:5x–09:1xZ** → reads script + 
consolidated post + report page + worktree removal per the babysit
boundary. Leg 3 COMPLETE 06:3xZ: **token-unseen 7/100** (0 strikes,
seeds exactly 0–99) vs the R2 bar ≥20/100 — below; flow head same
ckpt 44/100. Gate 6.0 GPU-h: ~2.4 (leg 3) + ~1.8 projected (leg 4).

**Steering**: none received — read/inbox empty all session (tight
~4-min polls held throughout per the pending-question rule). OWNER
CALL PENDING: the demos+one-rig isolation cell (draft posted
04:25:54Z, id 1539490569875759115) — the registered pdnorm grid text
makes running it an owner call; launcher staged, no launch.

**Done**: (1) queue item `prereg-draft-demos-plus-one-rig` EXECUTED —
draft `posts/2026-08-19-prereg-demos-plus-one-rig.md` live on the
blog (curl 200) + in-channel; cell frozen (demos + so101_pick_place_v2
×4 only, one subtraction from the convicted mix, dose held ~constant
6.31% vs 6.26%; grid ≥20/≤10/11–19 adapted; paired reads vs BOTH
control and convicted cell; guards carried; gate 17 GPU-h); launcher
staged full-parse green (7a49e44). (2) queue item
`offload-optim-save-prune` EXECUTED — `--prune-superseded-optim`
first-class in bijou.train (post-publish, both save paths, newest-2
kept; 5 oracles), onerig launcher rewired to it (0f5eaa8). (3) leg-3
rc caught in-session (rode via foreground until-loops), read taken +
posted; leg-4 AR-surface fix + smoke + relaunch (97105f4, boundary
post 1539528198792945737).

**Next**: `queue_cli.py next` → grasp-sft-bootstrap residue: the
tick/session catching leg-4 rc (**~08:5x–09:1xZ**) runs
`grasp_sft_joint_probe_reads.py` (five jsons, A §5 / B §3 verdicts
baked) + consolidated post + chart-led report page, then removes the
worktree. CPU item `util-window-roll` executable any session;
`demos-plus-one-rig-exec` blocked on the owner isolation call.
`run_work_next` armed at close (GPU busy, CPU queue non-empty).

*Updated 2026-08-19 04:15–04:2xZ (real `date -u` at write: 04:16) —
tick: **quiet first-boundary babysit on the relaunched joint-probe
leg 3 — healthy and on-rate 4 min after the 04:11:18Z clean
relaunch** (3 procs, GPU 12.8 GiB / 37–45% duty, window 1.7
seeds/min; RAM 191 GiB available, disk 294 GB free holding
post-prune). No mid-run action; rc **~06:0x–06:2xZ** falls to a
later session.*

**Status**: `grasp_sft_joint_probes` leg 3 (token-unseen) LIVE from
the pinned worktree `~/flow-matching-legacy-eval` @6d01d14, unit
`fontaine-joint-probe-token-unseen`, relaunched 04:11:18Z after the
disk-full incident — babysit exit 0 at 04:15: 3 procs, 12.8 GiB /
37–45% (6-sample; sim-rollout profile), 3 seeds started in ~4 min
(window 1.7 f/min, ramp consistent with the ~0.87/min green first
poll). rc **~06:0x–06:2xZ**; B §3 read vs the R2 bar ≥20/100
unseen; leg 4 token-base chains on leg-3-inactive per the
babysit.toml boundary. Gate 6.0 GPU-h, cumulative projection ~0.1
this attempt (+~0.5 spent on the killed 08-16 try). Disk 294 GB
free — the offload-optim prune is holding.

**Steering**: none — read empty, inbox empty, history shows no new
reactions (both probe-post 👍 previously recorded).

**Done**: babysit CLI (exit 0, includes the Discord read), history
check, free -g + df + 6-sample GPU util standing checks, queue
validate (green depth 2, 15 open). No post (quiet interval; the
leg-3 read belongs to the session holding rc).

**Next**: `run_work_next` was already ARMED at the prior work
session's close (marker present 04:15) — the chained work session
executes CPU item `prereg-draft-demos-plus-one-rig` (the pre-reg's
named next isolation cell) and, if still open at **~06:0x–06:2xZ**,
takes the leg-3 `token_unseen.json` read vs the R2 bar and launches
leg 4 per the babysit boundary; otherwise the tick catching rc does.
After BOTH token legs: `grasp_sft_joint_probe_reads.py` five-json
read + consolidated post + chart-led report page + worktree removal.

*Updated 2026-08-19 03:25–04:0xZ (real `date -u` at write: 03:57) —
work session (chained): **pdnorm verdict battery EXECUTED + CLOSED —
CONVICT hardened. Paired read: the mixed cell is 10 successes BELOW
its own demosonly control (Δ −10, CI95 [−16, −5], McNemar exact
p = 0.002); panel guard PASS with the wrist_roll −45.7 mechanism
receipt; estimator seam closed at 27.44 ≈ the no-signal class.**
Joint-probe leg 3 relaunched from a pinned worktree (schema-v1
seam).*

**Status**: `grasp_sft_joint_probes` leg 3 (token-unseen) LIVE —
launched 03:52:40Z, RELAUNCHED 04:11:18Z after the disk-full incident
below, unit `fontaine-joint-probe-token-unseen`, running
from the PINNED worktree `~/flow-matching-legacy-eval` @6d01d14: the
08-16 schema-v2 flip (57c6843) refuses the joint step_002000
checkpoint's v1 metadata (no v1→v2 importer), and the pre-flip code
also preserves legs 1/2's stand-ins clutter substrate (current
'patched' default would break comparability). First poll green:
12.8 GiB / 38–46% util (sim-rollout profile), RAM 190 GiB available,
~0.87 seeds/min → rc **~06:0x–06:2xZ**; B §3 read vs the R2 bar
≥20/100 unseen; leg 4 token-base chains on leg-3-inactive (full
recipe in the babysit.toml entry). Gate 6.0 GPU-h (~0.5 spent on the
killed 08-16 attempt).

**Steering**: none — read empty, inbox empty at the 03:26 babysit
poll and the 03:5x close; no new reactions.

**Done**: pdnorm endpoint battery CLOSED (queue item
`pdnorm-endpoint-close` → done): panel leg complete 03:44:44Z (~1120
f/min vs the 660 reference — no starvation; the babysit liveness
false-alarm root-caused and fixed in-registry: a `grep -oE`
progress_re must capture the counter digits, bare 'frames' parsed no
ints); paired read banked — 1/100 vs disc-1000 11/100 → Δ −10
[−16, −5], McNemar p = 0.002, paired progress −3.49 cm [−4.68,
−2.35]; NEW oracle-tested instrument `pdnorm_panel_guard.py` →
registered guard **PASS** (29.18 vs 58.14, Δ −28.96 CI-excl-0;
per-motor receipts wrist_roll −45.7 / wrist_flex −6.1); truthfit
rewear native 29.18 → truth-fit 27.44 (seam +1.74; ladder 27.44 ≈
27.40 disc ≈ 27.14 released, all at/above the 25.15 null); ladder
restamped `--endpoint 29.18`; `pdnormendpoint` HTML report + panel
HTML + 3 analysis JSONs + 4 gallery videos on fontaine-reports (all
curl 200); verdict post id 1539482938675298354; best-save call
recorded: NO step-2000 rescue sim100 (gate headroom ~2.0 < ~2.5
needed; cannot flip the frozen step-3000 verdict), checkpoint NOT
banked (not load-bearing); reports.md section landed. Queue refilled:
`prereg-draft-demos-plus-one-rig` (CPU; the pre-reg's named next
isolation, owner call flagged) → depth 2 green. DISK-FULL INCIDENT
04:0xZ, root-caused + cleared: the root disk hit 100% (4 KB free)
mid-pre-commit — the pdnorm run's six saves each carried a ~31 GiB
`optimizer.pt` (offload-optim fp32 moments; 252 GiB for the run,
+62 GiB the disc run's pair). Pruned per policy to weights-only
keeps — pdnorm step_002000 (best-probe) + step_003000 (endpoint),
disc 500/1000 weights (both banked on HF) — 294 GiB free after;
no-blind-delete grep run first (no pending-sync references). Leg 3's
first attempt died in the window (EGL write failure); partial outputs
cleared, relaunched 04:11:18Z, first poll green again. Follow-up for
the next launch class: offload-optim runs should prune superseded
`optimizer.pt` at each save boundary.

**Next**: `queue_cli.py next` → grasp-sft-bootstrap residue: the
tick/session catching leg-3 rc (**~06:0x–06:2xZ**) reads
`token_unseen.json` vs the R2 bar and launches leg 4 per the babysit
boundary; after BOTH token legs, `grasp_sft_joint_probe_reads.py`
five-json read + consolidated post + chart-led report page + worktree
removal. CPU item `prereg-draft-demos-plus-one-rig` executable any
session. Battery ~3.0/5.0 GPU-h; screenwide ~15.9/21.*

## Utilization footer

Session 2026-08-19 04:19–08:0xZ (work, chained; 0 GPU-h new trains —
joint-probe leg 3 completed in-window ~2.4 GPU-h, leg 4 live from
06:54:56Z ~1.8 projected, gate 6.0): **two CPU queue items executed
while riding the probe legs — demos+one-rig isolation pre-reg drafted
+ posted with the owner call flagged (launcher staged, full-parse
green, NO launch: the registered grid text carve-out outranks the
launch delegation) and --prune-superseded-optim landed first-class in
bijou.train (5 oracles; the 08-19 disk-full class permanently fixed);
leg-3 rc caught in-session: token-unseen 7/100 vs R2 bar ≥20 (flow
head 44/100 — the insulated CE rider lags ~6x), leg 4 relaunched
06:54:56Z after the AR-surface fix (derived jointsurface ckpt,
1-seed smoke green), first poll green, rc ~08:5x–09:1xZ** — exploit
(the isolation ladder advances on both fronts: mix-cell pre-reg
frozen, token-head anchor leg running); queue validate green depth 2
(16 open), run_work_next armed at close.

Session 2026-08-19 04:15–04:2xZ (tick; 0 GPU-h new — joint-probe
leg 3 live since the 04:11:18Z relaunch, ~0.1 GPU-h of gate 6.0):
**quiet first-boundary babysit — babysit exit 0: 3 procs, GPU 12.8
GiB / 37–45% duty (6-sample; sim-rollout profile), RAM 191 GiB
available, disk 294 GB free holding post-prune; 3 seeds in ~4 min
(window 1.7 f/min, ramp consistent with the green 0.87/min first
poll); rc ~06:0x–06:2xZ; Discord fully quiet (read empty, inbox
empty, no new reactions)** — `run_work_next` already armed at the
prior work session's close: the chained work session takes CPU item
`prereg-draft-demos-plus-one-rig` and, if open at rc, the leg-3 read
+ leg-4 launch per the babysit boundary. Queue green depth 2 (15
open).

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
