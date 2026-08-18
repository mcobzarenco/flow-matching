# Now












*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 05:45–05:4xZ (real `date -u` at write: 05:46) —
tick: **quiet tick — GO ask (01:54Z) still pending at ~3h50m with no
owner signal; H100 idle by design, `run_work_next` stays ARMED for the
CPU queue heads.***

**Status**: no live runs — H100 idle (0% util, 0 MiB; owner
policy-server not up at check), `no_live_runs_reason` current in the
babysit registry (H100 held for the owner-gated pdnorm launch). Queue
green depth 2 (22 open). The pdnorm run stays staged and owner-gated
(GO ask 01:54Z + two pre-launch calibration addenda 04:26/05:08Z, all
unanswered).

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts with no new reactions. At ~4 h old
the GO ask is out of conversational cadence; the chained work session
polls at boot and every boundary per the standing rule.

**Done**: Discord read + history + inbox checks; GPU-idle +
policy-server check; babysit registry verified (all entries pruned,
declared reason current); queue validate green; `run_work_next`
confirmed ARMED (touched 04:29, left in place). No in-channel post —
the 01:54Z ask + both addenda are current, nothing new to report.

**Next**: chained work session (4-h budget) owns
**sim100-paired-read-instrument** then **disc1000-panel-row-audit**
(both CPU, un-gated — both want to land before the pdnorm endpoint
reads), polling the GO ask at boot and each boundary. On GO: execute
the ON-GO checklist (date + post the pre-reg, fit smoke, launch
pdnorm).*

*Updated 2026-08-18 02:04–05:2xZ (real `date -u` at write: 05:15) —
work session (chained): **all THREE disc-1000 baseline legs executed
and banked pre-GO — HTML panel (5.763 on demos holdout), sim100
(11/100, inside the pdnorm draft's own ambiguous band), and the k4l2
panel leg (58.14 vs state-copy 8.37, 0% win — catastrophically OOD on
community data). Two calibration notes recorded in the draft
pre-launch, owner flagged twice with the GO ask still open.***

**Status**: no live runs — H100 idle again at close; the pdnorm run
stays staged and owner-gated (GO ask pending since 01:54Z, now with
two pre-launch addenda in-channel). The k4l2 panel leg completed
IN-session (04:57Z, ~0.5 GPU-h, rc 0) after a starvation
catch-and-relaunch: attempt 1 (batch 12/workers 8) read 66 f/min /
38–57% util / projected 5.7 GPU-h vs the 3 gate and was killed 4.7
min in per the first-poll rule; r2 (batch 32/workers 20) ran 96%
util, ~660 f/min. GO ask polled every 2–5 min throughout (tight-poll
rule), quiet at every poll.

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
0-success jsons before the real data). (5) **k4l2 panel leg run to
completion** (04:57Z, ~0.5 GPU-h; protocol pinned in
`eval_disc1000_k4l2_panel.sh` — the pdnorm endpoint leg must copy
it): chunk MAE **58.14** vs state-copy 8.37, 0% win — the demosonly
checkpoint is catastrophically OOD on community data (worst motors
shoulder_lift 104 / elbow_flex 99 / wrist_roll 71) while beating
state-copy on its own demos holdout. Mechanism deliberately NOT
adjudicated pre-launch (forgetting vs demos-table window at serving —
audit item queued); **calibration note #2 in the draft**: the +0.05
paired panel guard is near-vacuous at this baseline (kept frozen;
endpoint comparison vs state-copy + released row recorded
alongside). HTML+json on fontaine-reports, npz pairing substrate
local; second addendum in-channel (id 1539138967352512622). (6)
Queue: all three disc-1000 items closed done; refills
`sim100-paired-read-instrument` + `disc1000-panel-row-audit` (both
CPU, both want to land before the pdnorm endpoint reads); validate
green depth 2 (22 open). Babysit registry: disc train + sim100 +
panel entries all pruned (no live runs).

**Next**: `queue_cli.py next` → **sim100-paired-read-instrument**
then **disc1000-panel-row-audit** (both CPU, un-gated). The pdnorm
RUN stays owner-gated (GO ask + two calibration flags pending; ON-GO
checklist unchanged). `run_work_next` ARMED — GPU idle but the
CPU-side queue is non-empty.*

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

## Utilization footer

Session 2026-08-18 05:45–05:4xZ (tick; 0 GPU-h — H100 idle by design,
no live runs): **quiet tick — GO ask (01:54Z) + both calibration
addenda still pending at ~3h50m; read + history + inbox all empty of
new signals, registry declared-reason current, queue green depth 2
(22 open)** — `run_work_next` stays ARMED: the chained work session
owns sim100-paired-read-instrument + disc1000-panel-row-audit (both
CPU) and polls the GO ask at every boundary.

Session 2026-08-18 02:04–05:2xZ (work, exploit; ~2.8 GPU-h in-session
— HTML report ~0.1 + sim100 baseline ~2.2 + k4l2 panel leg ~0.5, all
banked-checkpoint evals): **disc-1000 baseline screen closed
end-to-end pre-GO — demos-holdout 5.763 / sim100 11/100 / panel 58.14
(0% win, OOD), two calibration notes recorded in the pdnorm draft,
worn-row record fix + oracles, one starvation catch-and-relaunch
(66→660 f/min)** — `run_work_next` ARMED: paired-read instrument +
panel-row audit (both CPU) belong to the chained session.

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
