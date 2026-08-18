# Now

















*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 06:46–07:0xZ (real `date -u` at write: 07:06) —
work session (chained, bounded): **pdnorm-endpoint-report-paired-section
DONE — the frozen paired read is now a rendered section on the
canonical disc-1000 flow-unseen report; the pdnorm endpoint gets the
same section from one `--paired-json` flag.***

**Status**: no live runs — H100 idle by design (held for the
owner-gated pdnorm launch; `no_live_runs_reason` current). Queue green
depth 2 (22 open). GO ask (01:54Z) still unanswered at ~5h10m; polled
at boot (06:46) and the post boundary (07:05), inbox empty throughout.

**Steering**: none — `read` empty at every poll, unreplied inbox
empty.

**Done** (commit `4cfefae` + close-out):
**pdnorm-endpoint-report-paired-section** —
`grasp_sft_joint_unseen_report.py` grows `--paired-json`: a frozen
`sim100_paired_read.py` output renders as a "Paired read" section
(delta tiles with CI wording + a McNemar discordant-seed chart,
house dark scheme; the disc1000 preset carries the 11–19
ambiguous-band note — recorded, never gating). Oracles
`tests/test_grasp_sft_joint_unseen_report.py` ×4 green, check.py 1015
green. Smoked on the banked probe-vs-disc1000 pair (44 vs 11, **+33**
CI95 [22, 44], McNemar p ≈ 1.0e-07, +3.57 cm progress, 80% win
rate), then the CANONICAL disc-1000 flow_unseen100 report
regenerated in place and re-pushed to fontaine-reports (curl 200,
section verified live); reports.md paired-read bullet extended.
In-channel note id 1539168266105131028. Queue: item closed done;
refill `pdnorm-endpoint-report-preset` (CPU, PRE-GO prep — the ON-GO
endpoint report as one command).

**Next**: `queue_cli.py next` →
**pdnorm-prereg-panel-guard-recalibration** then
**pdnorm-endpoint-report-preset** (both CPU, un-gated). The pdnorm
RUN stays owner-gated (ON-GO checklist unchanged). `run_work_next`
ARMED — GPU idle but the CPU-side queue is non-empty.*

*Updated 2026-08-18 06:44–06:4xZ (real `date -u` at write: 06:45) —
tick: **adjacent quiet tick (fired one minute after the 06:43 work
close) — GO-ask poll at 06:44Z, still quiet at ~4h50m; nothing else
changed.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB;
`no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) + both
calibration addenda + the paired-read note + the panel-row audit note
all unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts, no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 06:43). No in-channel post — nothing new
since the 06:43 audit note.

**Next**: chained work session owns
**pdnorm-endpoint-report-paired-section** then
**pdnorm-prereg-panel-guard-recalibration** (both CPU, un-gated),
polling the GO ask at boot and each boundary. On GO: ON-GO checklist
(date + post the pre-reg, fit smoke, launch pdnorm).*

*Updated 2026-08-18 06:19–06:3xZ (real `date -u` at write: 06:33) —
work session (chained, bounded): **disc1000-panel-row-audit DONE —
the 58.14 panel row is adjudicated: ~half serving-window
re-expression, ~half genuine collapse to the demos prior. The wear
fact is cleaner than the queue item feared: the checkpoint records
the MERGED scheme, so no per-dataset lookup ever ran.***

**Status**: no live runs — H100 idle by design (held for the
owner-gated pdnorm launch; `no_live_runs_reason` current). Queue
green depth 2 (22 open). GO ask (01:54Z) + both calibration addenda +
the paired-read note still unanswered at ~4h40m; polled at boot and
at the work boundary, inbox empty throughout.

**Steering**: none — `read` empty at every poll, unreplied inbox
empty.

**Done** (commit `00965c8`): **disc1000-panel-row-audit** —
`fontaine/scripts/disc1000_row_audit.py` (anchor-refusing wear audit
on the leg npz: box floor, edge-saturation, exact-inversion re-wear
through per-repo/released rows, repo-midpoint null, demos-prior
collapse probe), oracle `tests/test_disc1000_row_audit.py` ×7 green.
Findings: wear fact — `normalization: "q01q99"` +
`per_dataset_flow_norm: false` ⇒ every panel item wore the
recomputed demos-only global table (per-dataset rows never
consulted; "missing community rows" never arises). Decomposition —
85.8% of core truth elements outside the worn box but the box FLOOR
is only 14.40 of the 58.14 and predictions are NOT edge-saturated:
the wear hurts via affine re-expression, not the clamp. Re-wearing
the same normalized predictions through honest per-repo rows (838)
halves the row to **27.40** — but that is WORSE than a constant
repo-box-midpoint null (**25.15**), and raw predictions sit 22.6
from the constant demos action mean while truth sits 58.2 away:
output-wear-corrected, the checkpoint carries no usable signal on
community data. Analysis json on fontaine-reports (curl 200),
reports.md disc-1000 section extended + the panel-leg hedge
resolved. In-channel note id 1539162750654218351. Queue: item
closed done; refill
`pdnorm-prereg-panel-guard-recalibration` (CPU, draft-only —
fold 27.40/25.15 into the pdnorm draft's interpretation anchors).

**Next**: `queue_cli.py next` → **pdnorm-endpoint-report-paired-section**
then **pdnorm-prereg-panel-guard-recalibration** (both CPU,
un-gated). The pdnorm RUN stays owner-gated (ON-GO checklist
unchanged); its panel read now has wear-corrected reference points
(27.40 re-worn baseline / 25.15 midpoint null; real bar state-copy
8.37). `run_work_next` stays ARMED — GPU idle but the CPU-side queue
is non-empty.*

## Utilization footer

Session 2026-08-18 06:46–07:0xZ (work, exploit; 0 GPU-h — CPU-only
report item, H100 held for the owner-gated pdnorm launch):
**paired-read section landed on the eval report (`--paired-json`,
oracles ×4, check.py 1015 green); canonical disc-1000 flow-unseen
report regenerated + re-pushed with the +33 / p≈1e-7 read rendered;
queue refilled with the pdnorm endpoint-report preset item** —
`run_work_next` ARMED: pdnorm-prereg-panel-guard-recalibration next,
GO ask polled at boot + boundary (quiet).

Session 2026-08-18 06:44–06:4xZ (tick; 0 GPU-h — H100 idle by design,
no live runs): **adjacent quiet tick one minute after the 06:43 work
close — GO-ask poll 06:44Z still quiet (~4h50m), read + history +
inbox empty, registry reason current, queue green depth 2 (22
open)** — `run_work_next` stays ARMED: chained session owns
pdnorm-endpoint-report-paired-section +
pdnorm-prereg-panel-guard-recalibration.

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
