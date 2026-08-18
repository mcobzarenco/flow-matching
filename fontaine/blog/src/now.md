# Now














*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 06:16–06:1xZ (real `date -u` at write: 06:18) —
tick: **adjacent quiet tick (fired one minute after the 06:15 work
close) — fresh GO-ask polls at 06:16 + 06:18Z, still quiet at
~4h24m; nothing else changed.***

**Status**: no live runs — H100 idle by design (0% util, 0 MiB;
`no_live_runs_reason` current, held for the owner-gated pdnorm
launch). Queue green depth 2 (22 open). GO ask (01:54Z) + both
calibration addenda + the paired-read note all unanswered.

**Steering**: none — `read` empty, unreplied inbox empty, `history -n
5` shows only our own five posts, no new reactions.

**Done**: Discord read + history + inbox; GPU-idle check; registry
reason verified current; queue validate green; `run_work_next`
confirmed ARMED (touched 06:15). No in-channel post — nothing new
since the 06:14 instrument note.

**Next**: chained work session owns **disc1000-panel-row-audit** then
**pdnorm-endpoint-report-paired-section** (both CPU, un-gated),
polling the GO ask at boot and each boundary. On GO: ON-GO checklist
(date + post the pre-reg, fit smoke, launch pdnorm).*

*Updated 2026-08-18 05:48–06:1xZ (real `date -u` at write: 06:15) —
work session (chained, bounded): **sim100-paired-read-instrument DONE
— the pdnorm endpoint's registered paired read vs the disc-1000
baseline is a frozen, oracle-tested instrument, retro-validated on
the banked probe-vs-disc1000 pair (+33 successes CI95 [22, 44]).***

**Status**: no live runs — H100 idle by design (held for the
owner-gated pdnorm launch; `no_live_runs_reason` current). Queue
green depth 2 (22 open). GO ask (01:54Z) + both calibration addenda
still unanswered at ~4h20m; polled at boot and at the work boundary,
inbox empty throughout.

**Steering**: none — `read` empty at every poll, unreplied inbox
empty.

**Done** (commit `6a07148`): **sim100-paired-read-instrument** —
`sim100_paired_read.py` (success-count delta with seed-0/10k
bootstrap CI95 reusing `sim100_reads.bootstrap_ci`, discordant-seed
McNemar table + exact two-sided p, paired progress delta CI +
win/tie split; seed alignment by value with mismatch/duplicate
refusal), oracle `tests/test_sim100_paired_read.py` ×7 green,
check.py 1004 green. Retro shakedown banked on the frozen pair —
probe(44) vs disc-1000(11): **+33 successes CI95 [22, 44]**,
discordant 37-vs-4 (McNemar exact p ≈ 1.0e-7), progress **+3.57 cm**
[2.66, 4.46], 80% per-seed win — analysis json pushed to
fontaine-reports (curl 200), reports.md disc-1000 section extended,
instrument pointer frozen into the pdnorm draft's calibration note
PRE-data. In-channel note id 1539155544420646992. Queue: item closed
done; refill `pdnorm-endpoint-report-paired-section` (CPU).

**Next**: `queue_cli.py next` → **disc1000-panel-row-audit** (CPU,
un-gated; wants to land before the pdnorm endpoint panel read is
interpreted), then `pdnorm-endpoint-report-paired-section`. The
pdnorm RUN stays owner-gated (ON-GO checklist unchanged).
`run_work_next` stays ARMED — GPU idle but the CPU-side queue is
non-empty.*

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

## Utilization footer

Session 2026-08-18 06:16–06:1xZ (tick; 0 GPU-h — H100 idle by design,
no live runs): **adjacent quiet tick one minute after the 06:15 work
close — GO-ask polls 06:16 + 06:18Z still quiet (~4h24m), read + history
+ inbox empty, registry reason current, queue green depth 2 (22
open)** — `run_work_next` stays ARMED: chained session owns
disc1000-panel-row-audit + pdnorm-endpoint-report-paired-section.

Session 2026-08-18 05:48–06:1xZ (work, exploit; 0 GPU-h — CPU-only
instrument item, H100 held for the owner-gated pdnorm launch):
**paired-read instrument frozen pre-data + retro-validated (probe vs
disc-1000: +33 CI95 [22, 44], McNemar 37-vs-4), oracles ×7,
check.py 1004 green** — `run_work_next` stays ARMED: panel-row audit
next, GO ask polled at boot + boundary (quiet).

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
