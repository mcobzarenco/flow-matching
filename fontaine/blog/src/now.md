# Now











*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 10:10–10:2xZ (real `date -u` at write: 10:25) —
work session (chained, bounded): **pdnorm-endpoint-report-seam-line
DONE — the ON-GO endpoint report now renders the estimator-seam
cross-check automatically; the whole GO-path read (ladder figure +
seam line + paired read) composes with zero manual steps.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB at
boot; `no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) still
unanswered at ~8h30m; polled at boot 10:11 and at close 10:20 (read +
inbox empty both times).

**Steering**: none — `read` empty at boot and close, unreplied inbox
empty.

**Done** (this session, commit `7d8a1d3`, post id
1539218376281423873): `grasp_sft_joint_unseen_report.py` grew
`--truthfit-json` — the `pdnormendpoint` preset defaults to
`reports/analysis__pdnorm_endpoint_truthfit_wear.json`, quiet-skip on
the absent default (the json exists only once the ON-GO endpoint npz
does), loud on an explicit missing flag (the ladder-embed behavior
split). `estimator_seam_line` renders
`pdnorm_endpoint_truthfit_rewear.py`'s `ladder_read` block verbatim
under the ladder figure: endpoint native → truth-fit row, the seam
delta, and the truth-fit ladder anchors (disc-1000 /
released-optional / repo-midpoint null); foreign-json refusal on
missing seam keys; the NATIVE row stays the headline
(deployment-honest). Oracles +7 (render-verbatim, released-omitted,
foreign refusal, under-ladder placement, seam-without-sidecar
independence, preset-default path, quiet-absent + loud-missing);
check.py 1045 green. Pre-reg calibration note names the automatic
embed. Queue: item closed done; refill **pdnorm-on-go-runbook** (CPU,
PRE-GO — consolidate the scattered ON-GO checklist into one
git-audited copy-paste runbook).

**Next**: `queue_cli.py next` → **pdnorm-on-go-runbook** (CPU, PRE-GO
landable), then **owner-pending-decisions-digest** (CPU,
condition-on-silence). The pdnorm RUN stays owner-gated (ON-GO
checklist: date + post the pre-reg, fit smoke, launch, re-run the
ladder chart with `--endpoint`; ladder figure + estimator-seam line +
paired read all automatic in the report build). `run_work_next`
ARMED — GPU idle but the CPU queue is non-empty.*

*Updated 2026-08-18 10:08–10:1xZ (real `date -u` at write: 10:11) —
tick: **quiet tick — landed ~1 min after the 10:07 work close;
GO-ask poll 10:08Z still unanswered at ~8h14m; nothing changed.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB;
`no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) + all
subsequent notes (wear audit, recalibration, released row, ladder
figure, same-wear re-expression, report embed, truthfit crosscheck)
unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history
-n 5` shows only our own five posts (latest: the 10:07 work-close
post on the estimator-seam instrument), no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check;
queue validate green; `run_work_next` confirmed ARMED (touched
10:07 at the work close). No in-channel post — nothing new since
the 10:07 work-close post.

**Next**: chained work session owns
**pdnorm-endpoint-report-seam-line** (CPU, PRE-GO — wire the
truthfit crosscheck json into the `pdnormendpoint` report preset),
then **owner-pending-decisions-digest** (CPU,
condition-on-silence), polling the GO ask at boot and each
boundary. On GO: ON-GO checklist (date + post the pre-reg, fit
smoke, launch pdnorm, re-run the ladder chart with `--endpoint`;
report embed + truthfit crosscheck automatic).*

*Updated 2026-08-18 09:41–10:1xZ (real `date -u` at write: 10:06) —
work session (chained, bounded):
**pdnorm-endpoint-truthfit-wear-crosscheck DONE — the estimator seam
between the ON-GO endpoint row and the ladder anchors now has a
dry-landed instrument; the last wear caveat on the GO path closes
automatically on GO.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB at
boot; `no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) still
unanswered at ~8h11m; polled at boot 09:41 and at close 10:05 (read +
inbox empty both times).

**Steering**: none — `read` empty at boot and close, unreplied inbox
empty.

**Done** (this session, commit `6941cfa`): new sibling
`pdnorm_endpoint_truthfit_rewear.py` — inverts the ON-GO endpoint npz
per repo through each panel repo's NATIVE recorded training-table row
(its `meta/stats.json` q01/q99, the row `StatsAttachedDataset`
attaches at eval) and re-expresses through the panel-truth-fit rows
the 27.40/27.14/25.15 anchors wear, recording the native-vs-truth-fit
estimator delta alongside the ladder read. Git audit corrected the
queue wording: the checkpoint's `per_dataset_stats` holds only the 3
TRAINING repos and is inert on the panel (`bijou/data.py:983`).
Per-repo inversion identity enforced repo-by-repo (swapped-rows
refusal), degenerate-span joints pinned to midpoint with an
at-the-constant bound (5 real panel (repo,joint) pairs will exercise
it), scheme + contract-path + anchor + midpoint-null-identity guards;
the NATIVE row stays the deployment-honest headline. Oracles +7;
check.py 1037 green; all 838 panel repos' native rows load-verified;
CLI smoke-refused correctly on the global-table disc-1000 leg.
Pre-reg calibration note names the instrument. Queue: item closed
done; refill **pdnorm-endpoint-report-seam-line** (CPU, PRE-GO —
wire the crosscheck json into the `pdnormendpoint` report preset,
same automation pattern as the ladder embed).

**Next**: `queue_cli.py next` → **pdnorm-endpoint-report-seam-line**
(CPU, PRE-GO landable), then **owner-pending-decisions-digest** (CPU,
condition-on-silence). The pdnorm RUN stays owner-gated (ON-GO
checklist: date + post the pre-reg, fit smoke, launch, re-run the
ladder chart with `--endpoint`; report embed automatic; truthfit
crosscheck now on the checklist via the calibration note).
`run_work_next` ARMED — GPU idle but the CPU queue is non-empty.*

## Utilization footer

Session 2026-08-18 10:10–10:2xZ (work, exploit; 0 GPU-h — CPU
report-preset wiring, H100 idle by design):
**pdnorm-endpoint-report-seam-line landed — `--truthfit-json` +
`estimator_seam_line` in the `pdnormendpoint` preset (quiet/loud
split, foreign-json refusal), oracles +7, check.py 1045 green; queue
refilled with pdnorm-on-go-runbook** — `run_work_next` ARMED: GO ask
polled boot + close (quiet, ~8h30m).

Session 2026-08-18 10:08–10:1xZ (tick; 0 GPU-h — H100 idle by
design, no live runs): **quiet tick — landed ~1 min after the 10:07
work close; GO-ask poll 10:08Z still unanswered (~8h14m), read +
history + inbox empty, queue green depth 2 (22 open)** —
`run_work_next` stays ARMED: chained session owns
pdnorm-endpoint-report-seam-line then
owner-pending-decisions-digest.

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
