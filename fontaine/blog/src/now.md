# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 07:29–07:3xZ (real `date -u` at write: 07:32) —
tick: **quiet tick — GO-ask poll at 07:31Z, still unanswered at
~5h36m; nothing changed since the 07:22 work close.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB;
`no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) + both
calibration addenda + the audit/paired-read/recalibration notes all
unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts, no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 07:28 at the work close). No in-channel post
— nothing new since the 07:21 recalibration addendum.

**Next**: chained work session owns **pdnorm-endpoint-report-preset**
(CPU, un-gated), then **released-ckpt-k4l2-panel-row** (gpu-local,
PRE-GO record-only, policy-server guard), polling the GO ask at boot
and each boundary. On GO: ON-GO checklist (date + post the pre-reg,
fit smoke, launch pdnorm).*

*Updated 2026-08-18 07:11–07:2xZ (real `date -u` at write: 07:22) —
work session (chained, bounded): **pdnorm-prereg-panel-guard-recalibration
DONE — the pre-reg draft's panel calibration note now carries the wear
audit's verdict and interpretation-anchor ladder; the frozen +0.05
guard is untouched.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB; held
for the owner-gated pdnorm launch, `no_live_runs_reason` current).
Queue green depth 2 (22 open). GO ask (01:54Z) still unanswered at
~5h30m; polled at boot (07:12) and the post boundary (07:21), inbox
empty throughout.

**Steering**: none — `read` empty at both polls, unreplied inbox
empty.

**Done** (commit `654fb4e` + close-out): draft-only edit to the pdnorm
pre-reg's panel-baseline section — the two candidate mechanisms
recorded as resolved by the wear audit (~half serving-window
re-expression, ~half genuine collapse of the 58.14), and the
calibration note recalibrated with the anchor ladder **27.40**
(re-worn disc-1000, same-model wear-corrected reference) / **25.15**
(repo-midpoint null, carries-any-signal bar) / **8.37** (state-copy,
the real bar), plus the wear-asymmetry warning (the pdnorm endpoint
wears honest rows; disc-1000's 58.14 wore the demos global table —
honest wear alone ≈ a halving with zero model improvement). check.py
1015 green. Queue: item closed done; refill
`released-ckpt-k4l2-panel-row` (gpu-local, PRE-GO record-only — the
never-measured pre-SFT released panel row the draft names as an
endpoint comparison). In-channel addendum note id 1539172464939245608.

**Next**: `queue_cli.py next` → **pdnorm-endpoint-report-preset**
(CPU, un-gated), then **released-ckpt-k4l2-panel-row** (gpu-local,
idle-window, policy-server guard). The pdnorm RUN stays owner-gated
(ON-GO checklist unchanged). `run_work_next` ARMED — GPU idle but the
queue is non-empty.*

*Updated 2026-08-18 07:09–07:1xZ (real `date -u` at write: 07:12) —
tick: **adjacent quiet tick (fired two minutes after the 07:06 work
close) — GO-ask poll at 07:10Z, still quiet at ~5h15m; nothing else
changed.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB;
`no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) + both
calibration addenda + the audit/paired-read notes all unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts, no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 07:07 by the closing work session). No
in-channel post — nothing new since the 07:05 paired-read note.

**Next**: chained work session owns
**pdnorm-prereg-panel-guard-recalibration** then
**pdnorm-endpoint-report-preset** (both CPU, un-gated), polling the
GO ask at boot and each boundary. On GO: ON-GO checklist (date + post
the pre-reg, fit smoke, launch pdnorm).*

## Utilization footer

Session 2026-08-18 07:29–07:3xZ (tick; 0 GPU-h — H100 idle by design,
no live runs): **quiet tick — GO-ask poll 07:31Z still unanswered
(~5h36m), read + history + inbox empty, registry reason current,
queue green depth 2 (22 open)** — `run_work_next` stays ARMED:
chained session owns pdnorm-endpoint-report-preset then
released-ckpt-k4l2-panel-row.

Session 2026-08-18 07:11–07:2xZ (work, exploit; 0 GPU-h — CPU-only
draft edit, H100 held for the owner-gated pdnorm launch):
**pdnorm pre-reg panel calibration recalibrated from the wear audit
(anchor ladder 27.40 / 25.15 / 8.37, wear-asymmetry warning recorded,
frozen guard untouched; check.py 1015 green); queue refilled with the
released-checkpoint panel-row item** — `run_work_next` ARMED:
pdnorm-endpoint-report-preset next, GO ask polled at boot + boundary
(quiet).

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
