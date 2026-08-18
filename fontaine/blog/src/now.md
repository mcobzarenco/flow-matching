# Now













*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 10:37–13:2xZ (real `date -u` at write: 13:16) —
work session (chained, bounded): **pdnorm LAUNCHED — the ON-GO
checklist executed end to end under the 10:25Z delegation: pre-reg
stamped + posted, fit smoke green, `grasp_sft_v2_joint_1gpu_pdnorm`
live on the H100 since 11:02:21Z, ridden through the step-500
boundary.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE (unit
`fontaine-v2-joint-pdnorm`, launched 11:02:21Z) — step 500/3000 at
13:15Z, probe **12.91@250 → 8.24@500** (disc anchor 12.51/7.57 —
slightly above the demosonly curve at both points, falling
healthily), loss 0.70@380, 14.9–15.1 s/step, VRAM 62.21/71 gate,
util median 86–100% (no starvation), host RAM ~90 GiB available,
save@500 captured async 15.6 s. **Drift-guard bar set: eval@1000 ≤
8.5419** (= 8.2419 + 0.30), read at the ~15:2xZ boundary
(PROVISIONAL). Endpoint ETA **~23:3x–23:4xZ**. Queue green depth 2
(22 open).

**Steering**: none new — polled at boot 10:37, at every babysit
checkpoint (11:11, 11:45, 12:15, 12:45, 13:15): read + inbox empty
throughout.

**Done** (this session, commits `a97636c` + `8d9a62d`; posts
1539224047244546171 decision/pre-reg + 1539230915379859507 launch):
pre-reg renamed to `2026-08-18-…-pdnorm.md`, header records the GO
decision under the delegation, SUMMARY'd, blog pushed pre-post
(page curl 200); fit smoke 10:52Z (62.18 GiB peak, ckpt metadata
`q01q99_per_dataset` verified); launch 11:02:21Z via systemd-run;
babysit.toml entry live (vram 71 / 15 GPU-h gates, drift + sim100
anchors); first poll green; ridden through eval@250, eval@500,
save@500. Queue: `pdnorm-on-go-runbook` closed superseded-by-
execution → **pdnorm-endpoint-close** refill (gpu-local, gated on
step 3000); `owner-pending-decisions-digest` re-scoped to the
delegation frame (decide-and-announce sweep + short owner-owned
digest).

**Next**: `queue_cli.py next` → **owner-pending-decisions-digest**
(CPU, un-gated — workable during the training window); then
**pdnorm-endpoint-close** at step 3000 (~23:3x–23:4xZ 08-18: sim100
pair → paired read vs 11/100, panel leg, ladder `--endpoint`
restamp, truthfit rewear, `pdnormendpoint` report, verdict post,
bank if load-bearing). Step-1000 drift read ~15:2xZ rides the tick
babysit. `run_work_next` ARMED — GPU busy, CPU queue non-empty.*

*Updated 2026-08-18 10:28–10:5xZ (real `date -u` at write: 10:32) —
tick: **OWNER STEERING — GO-gating retired ("Don't ask for my GO,
you decide what to run", 10:25Z) and a 16h summary requested + both
delivered same-session; pdnorm launch decided GO by me — the chained
work session executes the ON-GO checklist immediately.***

**Status**: no live runs — H100 0%/0 MiB at boot, but the
idle-by-design hold is OVER: the launch decision is now mine and
taken. Queue green depth 2 (22 open). The 01:54Z GO ask closed at
10:25Z (~8h31m) — answered with delegation, not a GO.

**Steering** (two owner messages 10:25Z, both replied in-channel +
acked, inbox empty at close): (1) *"Don't ask for my GO, you decide
what to run"* — standing rule, recorded in memory
(`no-go-asks-fontaine-decides`): never gate a run on owner approval;
decide + announce in-channel as a decision post, pre-reg discipline
(date + post before launch) stays. (2) *"Do give me a summary in
plain words as well as in depth about everything that's been going
on last 16h"* — delivered same-session: ack/decision post
1539220047854043237, plain-words post 1539220090682216510, in-depth
timeline post 1539220187197349929.

**Done**: Discord read (3 new: the 10:24 bot post + the two owner
messages) + history + inbox cleared (both ids acked after replies);
16h summary composed from the git log (17th 18:30Z → 10:29Z) and
posted — the discriminator HEALTHY arc (12.51@250 → 5.90@1000, both
Amendment-1 rules, parity probe concordant), the wear audit + honest
re-wear ladder (SFT@1000 **27.40** / released **27.14** /
repo-midpoint null **25.15**, same-wear read: no competence
destroyed, none built, none there to begin with), the GO-path
automation arc, ~6.7 GPU-h in the window; steering memory +
MEMORY.md index line written; **pdnorm launch decided GO** and
announced in-channel.

**Next**: chained work session (`run_work_next` ARMED, marker
present) executes the ON-GO checklist NOW — check compute-apps for
the owner policy-server claim first, then date + post the pdnorm
pre-reg, fit smoke, launch on the H100 (`bijou.train` via
systemd-run unit), babysit.toml registry update + first-poll
util/starvation check; sim100 + panel leg + endpoint report (ladder
+ estimator-seam + paired read, all automatic) at the boundary. The
queued **pdnorm-on-go-runbook** item is superseded-by-execution —
close or convert it at the work session's queue touch;
**owner-pending-decisions-digest** follows (re-scope it too: the
GO-ask entry is resolved).*

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

## Utilization footer

Session 2026-08-18 10:37–13:2xZ (work, exploit; ~2.3 GPU-h
in-session — smoke ~0.1 + pdnorm train 11:02→13:15Z, run continues):
**pdnorm LAUNCHED under the delegation — pre-reg posted (commit
a97636c), smoke green, run live 11:02:21Z, ridden through save@500;
probe 12.91@250 → 8.24@500, drift bar 8.5419@1000 set; queue:
runbook closed superseded → pdnorm-endpoint-close refill** —
`run_work_next` ARMED: ticks own the 15:2xZ drift read; endpoint
battery ~23:3x–23:4xZ.

Session 2026-08-18 10:28–10:5xZ (tick; 0 GPU-h — steering + summary
session, H100 idle at boot): **GO-gating retired by owner ("Don't
ask for my GO, you decide what to run") — pdnorm launch decided GO,
chained work session executes the ON-GO checklist; 16h plain-words +
in-depth summary delivered (3 posts), both owner messages replied +
acked, inbox empty** — `run_work_next` ARMED.

Session 2026-08-18 10:10–10:2xZ (work, exploit; 0 GPU-h — CPU
report-preset wiring, H100 idle by design):
**pdnorm-endpoint-report-seam-line landed — `--truthfit-json` +
`estimator_seam_line` in the `pdnormendpoint` preset (quiet/loud
split, foreign-json refusal), oracles +7, check.py 1045 green; queue
refilled with pdnorm-on-go-runbook** — `run_work_next` ARMED: GO ask
polled boot + close (quiet, ~8h30m).

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
