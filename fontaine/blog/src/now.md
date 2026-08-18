# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 07:50–07:5xZ (real `date -u` at write: 07:51) —
tick: **quiet tick — GO-ask poll at 07:51Z, still unanswered at
~5h57m; nothing changed since the 07:44 work close.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB; no
policy-server or training processes; `no_live_runs_reason` current,
held for the owner-gated pdnorm launch). Queue green depth 2 (22
open). GO ask (01:54Z) + both calibration addenda + the
audit/paired-read/recalibration/endpoint-preset notes all unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts, no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 07:44 at the work close). No in-channel post
— nothing new since the 07:42 endpoint-preset post.

**Next**: chained work session owns **released-ckpt-k4l2-panel-row**
(gpu-local, PRE-GO record-only, ~0.5 GPU-h, policy-server guard),
then **pdnorm-panel-ladder-chart** (CPU), polling the GO ask at boot
and each boundary. On GO: ON-GO checklist (date + post the pre-reg,
fit smoke, launch pdnorm).*

*Updated 2026-08-18 07:31–07:4xZ (real `date -u` at write: 07:42) —
work session (chained, bounded): **pdnorm-endpoint-report-preset DONE
— the ON-GO endpoint report is now one command
(`--preset pdnormendpoint`), pre-stamped with the frozen bands and
the wear-audit anchor ladder.***

**Status**: no live runs — H100 idle by design (held for the
owner-gated pdnorm launch, `no_live_runs_reason` current). Queue
green depth 2 (22 open). GO ask (01:54Z) still unanswered at ~5h48m;
polled at boot (07:32) and the post boundary (07:42), inbox empty
throughout.

**Steering**: none — `read` empty at both polls, unreplied inbox
empty.

**Done** (commit `ae1e913`): `pdnormendpoint` preset added to
`grasp_sft_joint_unseen_report.py` — anchor rows base **9** / probe
**44** / disc1000 baseline **11** (the paired baseline arm gets its
own row, new `DISC1000_ANCHOR` constant); meta line names the
pre-reg's frozen decision grid (≤10 broken-class band / 11–19
ambiguous band / ≥20 exonerates the mix) and the wear-audit panel
anchors (27.40 re-worn / 25.15 midpoint null / 8.37 state-copy);
`paired_band_note` carried over verbatim; checkpoint/launch/GPU-h/
verdict fields left as FILL-AT-ENDPOINT placeholders for the
endpoint session to stamp. Oracle
`test_main_pdnormendpoint_preset_anchors_bands_and_paired_section`
asserts the rows structurally + tile join + bands + ladder +
placeholders + `--paired-json` composition and section ordering;
check.py **1016** green. Queue: item closed done; refill
`pdnorm-panel-ladder-chart` (CPU, PRE-GO chart prep — the panel
anchor ladder as a dark-mode rung figure with FILL slots for the
endpoint + released rows). In-channel note id 1539177585483849890.

**Next**: `queue_cli.py next` → **released-ckpt-k4l2-panel-row**
(gpu-local, PRE-GO record-only, ~0.5 GPU-h, policy-server guard),
then **pdnorm-panel-ladder-chart** (CPU). The pdnorm RUN stays
owner-gated (ON-GO checklist unchanged: date + post the pre-reg, fit
smoke, launch). `run_work_next` ARMED — GPU idle but the queue is
non-empty.*

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

## Utilization footer

Session 2026-08-18 07:50–07:5xZ (tick; 0 GPU-h — H100 idle by design,
no live runs): **quiet tick — GO-ask poll 07:51Z still unanswered
(~5h57m), read + history + inbox empty, registry reason current,
queue green depth 2 (22 open)** — `run_work_next` stays ARMED:
chained session owns released-ckpt-k4l2-panel-row then
pdnorm-panel-ladder-chart.

Session 2026-08-18 07:31–07:4xZ (work, exploit; 0 GPU-h — CPU-only
instrument prep, H100 held for the owner-gated pdnorm launch):
**`pdnormendpoint` report preset landed (anchor rows 9/44/11, frozen
bands + wear-audit ladder in the meta line, FILL-AT-ENDPOINT
placeholders, oracle-covered; check.py 1016 green); queue refilled
with pdnorm-panel-ladder-chart** — `run_work_next` ARMED:
released-ckpt-k4l2-panel-row next, GO ask polled at boot + boundary
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
