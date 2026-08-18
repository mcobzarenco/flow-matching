# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-18 08:55–09:0xZ (real `date -u` at write: 08:57) —
tick: **quiet tick — GO-ask poll at 08:56Z, still unanswered at
~7h02m; nothing changed since the 08:54 work close.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB;
`no_live_runs_reason` current, declared 08:2xZ, held for the
owner-gated pdnorm launch). Queue green depth 2 (22 open). GO ask
(01:54Z) + all subsequent notes (wear audit, paired read,
recalibration, endpoint preset, released row, ladder figure)
unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts (latest: the 08:54 ladder-figure
post with attachment), no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 08:54 at the work close). No in-channel post
— nothing new since the 08:54 ladder-figure post.

**Next**: chained work session owns
**released-row-honest-wear-reexpression** (CPU, feeds a same-wear
rung into the ladder), then **endpoint-report-ladder-embed** (CPU),
polling the GO ask at boot and each boundary. On GO: ON-GO checklist
(date + post the pre-reg, fit smoke, launch pdnorm, re-run the ladder
chart with `--endpoint`).*

*Updated 2026-08-18 08:37–08:5xZ (real `date -u` at write: 08:49) —
work session (chained, bounded): **pdnorm-panel-ladder-chart DONE —
the wear-audit anchor ladder is now a figure; the endpoint slot stamps
on GO.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB at
boot; `no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) still
unanswered at ~6h55m; polled at boot 08:37 (read + inbox empty).

**Steering**: none — `read` empty at boot, unreplied inbox empty.

**Done** (this session): `fontaine/scripts/pdnorm_panel_ladder_chart.py`
— house dark-scheme horizontal-rung figure of the pre-reg's
wear-corrected ladder (raw 58.14 / re-worn 27.40 / released 25.89 /
midpoint null 25.15 / clamp floor 14.40 / state-copy 8.37), the
pending pdnorm-endpoint slot rendered as a dashed full-width outline
(deliberately not a bar), `--endpoint <row>` stamps it magenta on GO.
Queue-vs-git drift resolved per the audit rule: the released row
(measured 25.89 last session) renders as a real rung, not a FILL
slot. Outputs: PNG `img/pdnorm/panel_ladder.png` + b64 sidecar
`reports/pdnorm_panel_ladder.b64`; figure embedded in the pre-reg
draft; oracle `tests/test_pdnorm_panel_ladder_chart.py` (rungs +
labels + placeholder + PNG/b64 roundtrip); check.py 1020 green.
Queue: item closed done; refill **endpoint-report-ladder-embed** (CPU
— wire the b64 sidecar into the `pdnormendpoint` report preset so the
ON-GO report embeds the stamped figure automatically).

**Next**: `queue_cli.py next` →
**released-row-honest-wear-reexpression** (CPU, dissolves the
ladder's wear-mismatch caveat — its output feeds a rung, so it stays
ahead of the embed item), then **endpoint-report-ladder-embed**
(CPU). The pdnorm RUN stays owner-gated (ON-GO checklist unchanged:
date + post the pre-reg, fit smoke, launch — now also re-run the
ladder chart with `--endpoint`). `run_work_next` ARMED — GPU idle but
the CPU queue is non-empty.*

## Utilization footer

Session 2026-08-18 09:01–09:2xZ (work, exploit; 0 GPU-h — CPU
re-expression from the banked npz, H100 idle by design):
**released-row-honest-wear-reexpression landed — same-wear read
released 27.14 vs SFT 27.40 (Δ +0.26, both slightly worse than the
25.15 null), identity anchors green, ladder re-rendered
wear-consistent, queue refilled with the ON-GO estimator
cross-check** — `run_work_next` ARMED: endpoint-report-ladder-embed
next, GO ask polled boot + close (quiet).

Session 2026-08-18 08:55–09:0xZ (tick; 0 GPU-h — H100 idle by
design, no live runs): **quiet tick — GO-ask poll 08:56Z still
unanswered (~7h02m), read + history + inbox empty, registry reason
current, queue green depth 2 (22 open)** — `run_work_next` stays
ARMED: chained session owns released-row-honest-wear-reexpression
then endpoint-report-ladder-embed.

Session 2026-08-18 08:37–08:5xZ (work, exploit; 0 GPU-h — CPU chart
prep, H100 idle by design): **pdnorm-panel-ladder-chart landed — the
wear-audit ladder is a stampable figure (PNG + b64 sidecar, oracle
green), released row rendered as a real rung per the git-audit rule,
queue refilled with endpoint-report-ladder-embed** — `run_work_next`
ARMED: released-row-honest-wear-reexpression next, GO ask polled at
boot (quiet).

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
