# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 16:17–16:2xZ (real `date -u` at write: 16:20) —
tick: **first babysit poll of live `grpo_r2` — healthy in the
pre-registered startup window. Liveness by procs+GPU (3 procs,
28.8 GiB, 62→100% util); babysit exit 1 "no parseable rows" is the
KNOWN startup read (first train.jsonl row ~17:1xZ). Discord fully
quiet; `run_work_next` already armed at 16:13, tick closes fast.***

**Status**: `grpo_r2` LIVE and healthy 7 min post-launch — 3 procs,
28.8 GiB / 100% util (62% momentarily mid-poll: step-0 eval + wave-0
rollout phase, replan/env cycles). No gate crossing (exit 3 did not
fire); nothing to judge yet — wave-0 gates (mixed ≥0.20 predicted
0.487, knockaway self-baseline) read at the first heartbeat row
~17:1xZ. RAM 162 GiB available, disk 227 GB free (92%).

**Steering**: none — read + inbox empty, history shows nothing after
our 16:11:50Z launch post, no reactions. No owner calls pending
(both closed 13:25Z).

**Done**: boot (pull clean), babysit CLI (exit 1 = the registry's
declared startup gap, procs+GPU confirm liveness), history + inbox
checks, queue validate green (depth 2, 15 open), standing
GPU/RAM/disk checks. No post owed (launch post 16:11Z is current;
next post-worthy event is the first heartbeat read).

**Next**: chained work session (marker armed 16:13) takes CPU item
`grpo-r2-boundary-reads-instrument` and reads the first train.jsonl
row ~17:1xZ (wave-0 gate judgment); ride cadence ~30-min babysit;
boundary ~step 10 (~0x:xxZ 08-20) → boundary legs per A3.4/A3.5,
then `demos-plus-one-rig-exec` takes the GPU.*

*Updated 2026-08-19 13:07–16:1xZ (real `date -u` at write: 16:12) —
work session (chained): **GRPO R2 IS LIVE. Owner steering landed
mid-session (13:25:15Z agree-with-recs → demos+one-rig GO, R2 AMEND +
ACTIVATE from 7%); the session had just landed `grpo-r2-launch-kit`,
so activation ran mechanically end-to-end: A3 flipped ACTIVE,
preflight leg 0 rode to its verdict — F-premise PASS, sampled T=1.0
8/100 vs greedy 7 — and the A3.4 run fired on the PASS at 16:10:02Z.
Disk sweep executed inside the ride window: ~41G freed.***

**Status**: `grpo_r2` LIVE (unit `grpo-r2`, launched 16:10:02Z) — 10
steps × 8×8 T=1.0 from `step_002000_v2`, lr 1e-6, kl_beta 1.0,
kl_stop 0.06; first poll 3 procs, 28.8 GiB / 100% util, step-0
baseline eval rolling (seed band 200+ correct). Budget ~14 GPU-h
expected / gate ≤15 incl. the ~2.25 preflight; boundary ETA ~step 10
(~0x:xxZ 08-20 at ~1 GPU-h/step). KNOWN STARTUP GAP noted in the
registry: first train.jsonl row ~17:1xZ — babysit exit 1 "no
parseable rows" before then is the startup read, procs+GPU are the
liveness truth. RAM fine, disk 231 GB free (92%).

**Steering**: owner 13:25:15Z agree-with-recs (…407784) — closed BOTH
registered calls: (1) demos+one-rig isolation GO
(`demos-plus-one-rig-exec` unblocked, fires at the next free GPU
boundary — R2 lane first, sequencing announced 13:49Z); (2) R2
AMEND+ACTIVATE (A3 ACTIVE, HEAD re-pin 570e53e). Replied 13:49:02Z +
acked; preflight-live 13:55Z, sweep result 15:0xZ, launch post
16:11Z.

**Done**: (1) `grpo-r2-launch-kit` EXECUTED + CLOSED (570e53e,
check.py 1075 green): `--knockaway-baseline` float|`wave0`
self-baseline + `--train-seed-base 2000` wired; `mixed_groups_frac`
heartbeat emit + `--wave0-mixed-abort 0.20` in-loop gate (defaults
unchanged); `grpo_r2_preflight_verdict.py` pins "materially below" at
the exact binomial 5% tail (ABORT ≤2 / BAND 3–6 / PASS ≥7);
`launch_grpo_r2.sh` parse-check/preflight/launch, launch refuses
non-PASS. A3.8 registered. (2) A3 ACTIVATED (ec87114) + preflight
ridden to verdict: **PASS 8/100 sampled vs greedy 7** (P=0.734;
success-seed overlap with greedy only 1/8 — sampling completes
different scenes; predicted mixed 0.487 vs bar 0.20); receipts
`outputs/sim/grpo_r2/preflight/preflight_verdict.json`. (3) R2
LAUNCHED on the PASS per A3.7/A3.8, registry entry live. (4)
`disk-retirement-sweep-banked-sources` EXECUTED + CLOSED (2e85b1c):
er_60k trainer dir 16/16 bitwise on HF (receipt
`reports/analysis__er60k_trainer_dir_sha_audit.json`) → 37G retired +
the two audit-proven legacy dirs (2.2G each); disk 93%→92%.

**Next**: `queue_cli.py next` → CPU item
`grpo-r2-boundary-reads-instrument` (mechanize the three endpoint
legs before the boundary); ride cadence: babysit every ~30 min, first
heartbeat row read ~17:1xZ (wave-0 gates: mixed ≥0.20 predicted
0.487, knockaway self-baseline capture); at the R2 boundary (~0x:xxZ
08-20): boundary legs per A3.4/A3.5, then `demos-plus-one-rig-exec`
takes the GPU (owner GO banked). `run_work_next` armed at close.*

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

## Utilization footer

Session 2026-08-19 16:17–16:2xZ (tick; `grpo_r2` riding, ~0.2 GPU-h
elapsed of ~14 expected): **first poll of live R2 healthy in the
declared startup window (procs+GPU liveness, babysit exit 1 = known
train.jsonl gap until ~17:1xZ); Discord quiet, no gate crossings;
`run_work_next` already armed — fast close** — queue green depth 2
(15 open). Disk 227 GB free (92%).

Session 2026-08-19 13:07–16:1xZ (work, chained; ~2.25 GPU-h banked —
preflight leg 0 — + `grpo_r2` live from 16:10Z, ~14 expected / gate
≤15): **owner agree-with-recs mid-session → A3 ACTIVATED end-to-end:
launch kit landed (570e53e) → preflight PASS (sampled 8/100 vs greedy
7) → R2 FIRED 16:10:02Z; demos+one-rig GO banked, fires at the next
free GPU boundary; disk sweep executed in the ride window (~41G
freed, 16/16 bitwise audit)** — exploit (registered activation +
infra debt); queue green depth 2 (15 open: `grpo-r2-launch-kit` +
`disk-retirement-sweep-banked-sources` closed,
`grpo-r2-boundary-reads-instrument` refilled). Disk 231 GB free
(92%).


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
