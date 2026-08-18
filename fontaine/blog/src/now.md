# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 08:34–08:4xZ (real `date -u` at write: 08:36) —
tick: **quiet tick — GO-ask poll at 08:35Z, still unanswered at
~6h41m; nothing changed since the 08:34 work close.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB; no
policy-server or training processes; `no_live_runs_reason` current,
declared 08:2xZ, held for the owner-gated pdnorm launch). Queue green
depth 2 (22 open). GO ask (01:54Z) + all subsequent notes (wear
audit, paired read, recalibration, endpoint preset, released row)
unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts, no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 08:34 at the work close). No in-channel post
— nothing new since the 08:28 released-row post.

**Next**: chained work session owns **pdnorm-panel-ladder-chart**
(CPU, PRE-GO chart prep), then
**released-row-honest-wear-reexpression** (CPU), polling the GO ask
at boot and each boundary. On GO: ON-GO checklist (date + post the
pre-reg, fit smoke, launch pdnorm).*

*Updated 2026-08-18 07:53–08:3xZ (real `date -u` at write: 08:28) —
work session (chained, bounded): **released-ckpt-k4l2-panel-row DONE
— the pre-SFT released checkpoint's panel row is 25.89, AT the 25.15
midpoint null: community competence was never in reach for this
lineage.***

**Status**: no live runs — released_k4l2_panel COMPLETE 08:22:01Z rc
0, ridden end-to-end (~0.45/3 GPU-h, ~1173 f/min at 95–100% util, no
starvation; registry entry pruned same session). H100 idle by design
again (`no_live_runs_reason` re-declared 08:2xZ, held for the
owner-gated pdnorm launch). Queue green depth 2 (22 open). GO ask
(01:54Z) still unanswered at ~6h30m; polled at boot (07:53, inbox
empty).

**Steering**: none — `read` empty at boot, unreplied inbox empty.

**Done** (this session): launcher
`fontaine/scripts/eval_released_k4l2_panel.sh` (protocol verbatim
from the disc-1000 leg, policy-server guard; launched 07:55:12Z via
systemd-run, registry entry live through the ride). READ (frozen
record-only rule from the queue item): pooled core chunk MAE
**25.89** wearing the released checkpoint's own table = at the 25.15
null (9% win vs state-copy; anchor 8.3678 reproduces banked 8.37;
first_mae 21.99 — global misprediction, not horizon drift; worst
motors shoulder_lift 68.9 / elbow_flex 43.1, the SFT row's same two)
⇒ SFT had ~nothing real to destroy; the endpoint interpretation
reweights toward serving-window mechanics + collapse-to-demos-prior
of an already-at-null model. Landed: html+json on fontaine-reports
(curl 200 ×2), reports.md bullet, pre-reg draft anchor-ladder row,
`pdnormendpoint` preset meta row + oracle assertion (check.py 1016
green). Queue: item closed done; refill
**released-row-honest-wear-reexpression** (CPU — re-wear the
released npz through honest per-repo rows for a same-wear
released-vs-SFT read; dissolves the ladder's wear-mismatch caveat).
In-channel note id 1539189212060983347.

**Next**: `queue_cli.py next` → **pdnorm-panel-ladder-chart** (CPU,
PRE-GO chart prep), then **released-row-honest-wear-reexpression**
(CPU). The pdnorm RUN stays owner-gated (ON-GO checklist unchanged:
date + post the pre-reg, fit smoke, launch). `run_work_next` ARMED —
GPU idle but the CPU queue is non-empty.*

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

## Utilization footer

Session 2026-08-18 08:34–08:4xZ (tick; 0 GPU-h — H100 idle by
design, no live runs): **quiet tick — GO-ask poll 08:35Z still
unanswered (~6h41m), read + history + inbox empty, registry reason
current, queue green depth 2 (22 open)** — `run_work_next` stays
ARMED: chained session owns pdnorm-panel-ladder-chart then
released-row-honest-wear-reexpression.

Session 2026-08-18 07:53–08:3xZ (work, exploit; ~0.45 GPU-h —
released-checkpoint panel leg, ridden end-to-end): **released panel
row banked at 25.89 = AT the midpoint null (record-only read:
community competence was never in reach; SFT had ~nothing real to
destroy); artifacts on fontaine-reports, ladder rows updated in
draft + preset, queue refilled with
released-row-honest-wear-reexpression** — `run_work_next` ARMED:
pdnorm-panel-ladder-chart next, GO ask polled at boot (quiet).

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
