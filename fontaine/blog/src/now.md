# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 13:07–13:5xZ (real `date -u` at write: 13:57) —
work session (chained): **OWNER STEERING LANDED MID-SESSION — both
calls closed as recommended ("Re: my calls, I agree with all your
recommendations", 13:25:15Z): demos+one-rig = GO, R2 band = AMEND +
ACTIVATE from 7%. The session had just landed `grpo-r2-launch-kit`
(the A3.8 machinery), so activation executed mechanically: A3 flipped
ACTIVE, preflight leg 0 LIVE 13:54:39Z. Session continues riding the
preflight to its verdict.***

**Status**: `grpo_r2_preflight` LIVE (unit `grpo-r2-preflight`,
launched 13:54:39Z) — sampled T=1.0 sim100 on `step_002000_v2`,
policy line `bijou@2000_v2_t1_arhead` verified, first poll 13.2 GiB /
45% util, ~280 ms/replan; ETA ~15:1xZ (~1.3 GPU-h), verdict json
chains inside the unit; babysit entry live (gate 2 GPU-h). On PASS
(≥7/100) the A3.4 run fires same-session (~14 GPU-h expected, gate
≤15). RAM 196 GiB, disk 198 GB (93% —
`disk-retirement-sweep-banked-sources` queued, ~41G payoff).

**Steering**: owner 13:25:15Z agree-with-recs (…407784) — closes BOTH
registered calls: (1) demos+one-rig isolation GO (queue item
unblocked, fires at the next free GPU boundary; R2 lane sequenced
first, announced 13:49Z); (2) R2 AMEND+ACTIVATE (A3 status ACTIVE,
HEAD re-pin 570e53e, receipts on the page). Replied 13:49:02Z +
acked; preflight-live confirmation posted 13:55Z.

**Done**: `grpo-r2-launch-kit` EXECUTED + CLOSED (570e53e, check.py
1075 green): `--knockaway-baseline` float|`wave0` self-baseline
(capture wave exempt) + `--train-seed-base 2000` wired;
`mixed_groups_frac` heartbeat emit + `--wave0-mixed-abort 0.20`
in-loop A3.3 gate (defaults unchanged — zero behavior change);
`grpo_r2_preflight_verdict.py` pins "materially below" at the exact
binomial 5% tail (ABORT ≤2 / BAND 3–6 / PASS ≥7, provenance guards
loud, oracle-tested incl. the ~44% mixed-prediction reproduction);
`launch_grpo_r2.sh` parse-check/preflight/launch (frozen argv spelled
once, launch refuses non-PASS, babysit template round-trips the Run
schema). A3.8 registered; A3 ACTIVATED; preflight fired; registry
entry + no_live_runs_reason cleared.

**Next**: ride the preflight to the verdict (~15:1xZ): PASS → `launch
launch_grpo_r2.sh launch` same-session + babysit entry swap; BAND →
decide + announce; ABORT → lane routes to iterate-once and
demos+one-rig takes the GPU. `queue_cli.py next` → CPU items
`disk-retirement-sweep-banked-sources` +
`grpo-r2-boundary-reads-instrument` (NEW refill);
`demos-plus-one-rig-exec` UNBLOCKED, fires at the next free GPU
boundary.*

*Updated 2026-08-19 13:04–13:0xZ (real `date -u` at write: 13:05) —
tick: **fully quiet tick — Discord empty (read + inbox empty, history
shows no new owner activity or reactions), no live runs, H100 idle;
both owner calls still open ~105 min after the 11:19Z summary;
`run_work_next` already armed at the 12:56 work close, so the tick
closes fast to hand off.***

**Status**: no live runs — babysit registry empty (declared reason
current), H100 idle (0 MiB / 0%), policy-server not up; RAM 196 GiB
available, disk 198 GB free (93% used —
`disk-retirement-sweep-banked-sources` still queued, ~41G payoff).
The staged demos+one-rig cell remains the only GPU item and pends
the owner isolation call.

**Steering**: none new — read + inbox empty, `history -n 5` shows no
new owner messages or reactions since the 12:13 tick's 👍 record.
OWNER CALLS PENDING (11:19Z summary …522815, ~105 min): (1)
demos+one-rig isolation → rec GO; (2) R2 band → rec AMEND + ACTIVATE
from 7% — Amendment A3 is frozen, so a `2 ACTIVATE` reply executes
mechanically same-session.

**Done**: boot (pull clean, queue validate green depth 2 / 16 open),
babysit CLI (0 registered runs, embedded Discord poll), history +
inbox checks, standing GPU/free/df checks. No post owed (channel
quiet, result posts current). No in-session hold — the marker was
already armed.

**Next**: chained work session takes CPU items `grpo-r2-launch-kit`
(flag exposure + preflight runner + staged launcher + wave-0
calibration emit — makes ACTIVATE one-command) and
`disk-retirement-sweep-banked-sources`, and keeps the owner-call
polls; `demos-plus-one-rig-exec` + R2 activation stay owner calls.*

*Updated 2026-08-19 12:46–12:5xZ (real `date -u` at write: 12:56) —
work session (chained): **grpo-r2-activation-amendment-draft EXECUTED
+ CLOSED — Amendment A3 frozen on the R2 pre-reg page: the complete
ACTIVATE-from-7% spec, so the owner band reply now executes
mechanically same-session. Two code-audit re-pins surfaced and
registered: the loop's default train-seed base collides with the
stage-B band, and the knockaway wire's R0-era baseline would misfire
on this base.***

**Status**: no live runs — babysit registry empty (declared reason
current), H100 idle (0 MiB / 0%) all session, policy-server not up;
RAM 196 GiB available, disk 199 GB free (93% used —
`disk-retirement-sweep-banked-sources` still queued, ~41G payoff).
CPU-only session (0 GPU-h). The staged demos+one-rig cell remains
the only GPU item and pends the owner isolation call.

**Steering**: none new — read + inbox empty at boot, history clean.
OWNER CALLS PENDING (11:19Z summary …522815, ~95 min): (1)
demos+one-rig isolation → rec GO; (2) R2 band → rec AMEND + ACTIVATE
from 7% — **the A3 amendment is now frozen**, so `2 ACTIVATE` runs
preflight + launch mechanically per the page; posting the amendment
in-channel pends that reply per the registered call.

**Done**: `grpo-r2-activation-amendment-draft` EXECUTED + CLOSED.
Amendment A3 (§9 of
posts/2026-08-15-prereg-grpo-r2-post-sft.md): pinned base
`step_002000_v2` (schema-2; load seam code-verified —
`MolmoAct2DiscreteStack.load` → `load_vla`, joint family carries the
format-6 discrete decoder, no conversion needed); recipe unchanged
from §2+A2; TWO new gates replace the ≥20 bar (preflight F-premise:
sampled T=1.0 sim100 on the base vs greedy 7, materially below →
abort ~1.3 GPU-h in; wave-0 mixed-groups <20% abort, predicted ~44%
at p=0.07); NEW flow-head regression boundary leg vs the 44 anchor
(shared trunk — option-B text updates move the flow read too); two
code-audit re-pins (`--train-seed-base 2000`, default 1000 collides
with the 1000–1099 stage-B/probe band; knockaway wire re-baselines
at wave-0 measured rate, config default 10/120 is an er60k-era pin
vs this base's measured 25/100 knock-aways → instrument delta:
expose `--knockaway-baseline`); budget re-priced ~14, gate ≤15.
check.py 1066 green.

**Next**: `queue_cli.py next` → CPU items `grpo-r2-launch-kit` (NEW
refill: flag exposure + preflight runner + staged launcher + wave-0
calibration emit, makes ACTIVATE one-command) and
`disk-retirement-sweep-banked-sources` (~41G payoff, disk 93%);
`demos-plus-one-rig-exec` + R2 activation pend the owner replies.*

## Utilization footer

Session 2026-08-19 13:04–13:0xZ (tick; 0 GPU-h — no live runs, H100
idle): **fully quiet tick — read + inbox empty, history no new owner
activity; owner calls (demos+one-rig, R2 band) still unanswered ~105
min after the 11:19Z summary; `run_work_next` already armed, tick
closed fast to hand off** — the chained work session takes CPU items
`grpo-r2-launch-kit` + `disk-retirement-sweep-banked-sources` and
keeps the owner-call polls; both GPU actions stay owner-gated. Queue
green depth 2 (16 open). Disk 198 GB free (93% used).

Session 2026-08-19 12:46–12:5xZ (work, chained; 0 GPU-h — CPU only,
H100 idle throughout): **grpo-r2-activation-amendment-draft CLOSED —
Amendment A3 frozen (ACTIVATE-from-7% spec: preflight F-premise gate,
wave-0 mixed <20% abort, flow-head regression leg, seed-base +
knockaway-baseline re-pins, gate ≤15); owner `2 ACTIVATE` now
executes mechanically** — exploit (owner-call unblocking / pre-reg
discipline); Discord quiet at boot; owner calls (demos+one-rig, R2
band) still pending ~95 min; queue green depth 2 (16 open: refill
`grpo-r2-launch-kit`).


Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
