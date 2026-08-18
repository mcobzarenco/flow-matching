# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 09:21–09:4xZ (real `date -u` at write: 09:38) —
work session (chained, bounded): **endpoint-report-ladder-embed DONE —
the ON-GO endpoint report now embeds the stamped anchor-ladder figure
automatically; one manual composition step deleted from the GO path.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB at
boot; `no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) still
unanswered at ~7h43m; polled at boot 09:21 and at close 09:37 (read +
inbox empty both times).

**Steering**: none — `read` empty at boot and close, unreplied inbox
empty.

**Done** (this session, commit `353f6db`):
`grasp_sft_joint_unseen_report.py` grew `--ladder-b64` +
`ladder_section()` — embeds the `pdnorm_panel_ladder_chart.py` b64
sidecar as a "Panel anchor ladder" figure directly below the meta
line's textual ladder; the `pdnormendpoint` preset defaults to the
chart script's sidecar path `reports/pdnorm_panel_ladder.b64`, so on
GO the endpoint session just re-runs the chart with `--endpoint <row>`
and builds the report — zero manual embed. Behavior split: explicit
flag is loud on a missing file, preset default quiet-skips (reports/
is gitignored, sidecar regenerable); payload asserted base64-PNG.
Real 08-18 sidecar smoke-rendered through the section. Oracles +6 in
`tests/test_grasp_sft_joint_unseen_report.py`; check.py 1030 green.
Pre-reg chart note updated (no manual figure step on GO). Queue: item
closed done; refill **owner-pending-decisions-digest** (CPU,
condition-on-silence — ~20 of 22 open items pend an owner call
scattered across days of history; one digest page + pointer post
answers "what do you need from me" in a single read, posted only if
the owner is still silent then).

**Next**: `queue_cli.py next` →
**pdnorm-endpoint-truthfit-wear-crosscheck** (CPU, ON-GO rider — the
per-repo inversion extension can land dry PRE-GO), then
**owner-pending-decisions-digest** (CPU, condition-on-silence). The
pdnorm RUN stays owner-gated (ON-GO checklist unchanged: date + post
the pre-reg, fit smoke, launch, re-run the ladder chart with
`--endpoint`; the report embed step is now automatic). `run_work_next`
ARMED — GPU idle but the CPU queue is non-empty.*

*Updated 2026-08-18 09:18–09:2xZ (real `date -u` at write: 09:19) —
tick: **quiet tick — landed ~2 min after the 09:16 work close;
GO-ask poll 09:18Z still unanswered at ~7h24m; nothing changed.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB;
`no_live_runs_reason` current, declared 08:2xZ, held for the
owner-gated pdnorm launch). Queue green depth 2 (22 open). GO ask
(01:54Z) + all subsequent notes (wear audit, paired read,
recalibration, endpoint preset, released row, ladder figure,
same-wear re-expression) unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history
-n 5` shows only our own five posts (latest: the 09:16 same-wear
re-expression post with the updated ladder figure), no new
reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 09:17 at the work close). No in-channel
post — nothing new since the 09:16 re-expression post.

**Next**: chained work session owns **endpoint-report-ladder-embed**
(CPU, wire the b64 sidecar into the `pdnormendpoint` preset), then
**pdnorm-endpoint-truthfit-wear-crosscheck** (CPU, instrument can
land dry PRE-GO), polling the GO ask at boot and each boundary. On
GO: ON-GO checklist (date + post the pre-reg, fit smoke, launch
pdnorm, re-run the ladder chart with `--endpoint`).*

*Updated 2026-08-18 09:01–09:2xZ (real `date -u` at write: 09:16) —
work session (chained, bounded):
**released-row-honest-wear-reexpression DONE — the anchor ladder is
wear-consistent end to end; same-wear read: SFT ended within noise of
where it started, and both rows sit slightly worse than the null.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB at
boot; `no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) still
unanswered at ~7h20m; polled at boot 09:01 and at close (read +
inbox empty both times).

**Steering**: none — `read` empty at boot and close, unreplied inbox
empty.

**Done** (this session, commit `ca083cf`):
`fontaine/scripts/released_row_rewear.py` — sibling of the wear
audit; re-expresses the released checkpoint's banked panel
predictions through the SAME honest per-repo rows the disc-1000
27.40 reference wears (output-side only, no model re-run). Integrity:
anchors reproduced (25.8924/8.3678), inversion round-trip worst
1.5e-05°, midpoint-null identity anchor PASSED (panels
element-identical → honest rows byte-identical, null 25.154476 both
sides). **Same-wear read: released honest-wear 27.14 vs SFT@1000
27.40 (Δ +0.26)** — wear held fixed, SFT neither destroyed nor built
community competence, and both rows are slightly WORSE than the
25.15 repo-midpoint null; per-joint shoulder_lift 68.9→66.1 /
elbow_flex 43.1→36.2, same two dominant motors. Landed: analysis
json on fontaine-reports (curl 200), reports.md re-expression bullet
(caveat marked dissolved), pre-reg calibration ladder rewritten
(released rung = same-wear 27.14, own-table 25.89 in the note),
ladder chart + oracle updated and PNG/b64 re-rendered, oracles
`tests/test_released_row_rewear.py` ×5; check.py 1024 green.
In-channel post 1539201268197756948 (with the updated figure).
Queue: item closed done; refill
**pdnorm-endpoint-truthfit-wear-crosscheck** (CPU, ON-GO rider — the
truth-fit-vs-native-table estimator seam is the one wear caveat
left).

**Next**: `queue_cli.py next` → **endpoint-report-ladder-embed**
(CPU, wire the b64 sidecar into the `pdnormendpoint` report preset),
then **pdnorm-endpoint-truthfit-wear-crosscheck** (CPU, instrument
can land dry PRE-GO). The pdnorm RUN stays owner-gated (ON-GO
checklist unchanged: date + post the pre-reg, fit smoke, launch,
re-run the ladder chart with `--endpoint` — rungs now same-wear).
`run_work_next` ARMED — GPU idle but the CPU queue is non-empty.*

## Utilization footer

Session 2026-08-18 09:21–09:4xZ (work, exploit; 0 GPU-h — CPU
report-preset wiring, H100 idle by design):
**endpoint-report-ladder-embed landed — `--ladder-b64` +
preset-default sidecar embed, oracles +6, check.py 1030 green, one
manual step off the ON-GO path; queue refilled with
owner-pending-decisions-digest** — `run_work_next` ARMED:
pdnorm-endpoint-truthfit-wear-crosscheck next, GO ask polled boot +
close (quiet, ~7h43m).

Session 2026-08-18 09:18–09:2xZ (tick; 0 GPU-h — H100 idle by
design, no live runs): **quiet tick — landed ~2 min after the 09:16
work close; GO-ask poll 09:18Z still unanswered (~7h24m), read +
history + inbox empty, registry reason current, queue green depth 2
(22 open)** — `run_work_next` stays ARMED: chained session owns
endpoint-report-ladder-embed then
pdnorm-endpoint-truthfit-wear-crosscheck.

Session 2026-08-18 09:01–09:2xZ (work, exploit; 0 GPU-h — CPU
re-expression from the banked npz, H100 idle by design):
**released-row-honest-wear-reexpression landed — same-wear read
released 27.14 vs SFT 27.40 (Δ +0.26, both slightly worse than the
25.15 null), identity anchors green, ladder re-rendered
wear-consistent, queue refilled with the ON-GO estimator
cross-check** — `run_work_next` ARMED: endpoint-report-ladder-embed
next, GO ask polled boot + close (quiet).

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
