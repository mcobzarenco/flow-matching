# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 03:17–04:0xZ (real `date -u`) — work session
(bounded, the chained rc owner): **subgoal-swap CLOSED end-to-end —
arm rc=0 03:42:36Z, all oracles green, frozen reads banked, verdict
MIXED (both mechanisms real: ~40% format floor + ~60% content margin
of the −0.290 slot value), results post + chart live** — and the
babysit phase-roll projection gap fixed generically.*

**Status**: no live GPU runs — local GPU free 03:42Z (swap arm
complete, ~1.5 GPU-h ≤ 3 gate), box free since 02:26Z. Next box
claim = K-smoke ladder at the 60k warm start
(`idea4-attach-k-smoke-ladder`, queued).

**Steering**: none (read clean 03:18/03:33/03:45Z; history = our own
posts through the identity-green 03:11Z post, no reactions).

**Done** (this session): (1) **babysit.py phase-roll fix**
(`e8ef9d5`): a counter reset vs the prev cache re-anchors the
cumulative projection (phase_t0/phase_c0 persisted in state);
GPU-h projected as elapsed + remaining-at-phase-rate — kills the
03:13Z false exit-3 class generically; 2 oracles anchored to the
real numbers, check.py 556. (2) **subgoal_swap_results.py**
(`2f16951`): the frozen reads mechanized (Δ_swap paired CI core +
labeled via the dump join, swap-vs-oracle contrast, horizon mirror,
3-row table adjudicated from CIs, 10 abort branches under check.py,
557). (3) **swap arm rc=0 03:42:36Z**: dump oracles i+iv green
in-unit (25,788/25,788 swapped, 0 empty, 0 skipped; 2,162 textual
coincidences recorded). (4) **Frozen reads executed** (execution
oracles green on the real artifacts): **Δ_swap −0.113 [−0.161,
−0.060]** (wrong words HELP), **swap−oracle +0.166 [+0.127,
+0.205]** (truth clearly better), horizon last-10 swap −0.175 vs
oracle −0.480 (the banked −0.464 signature reproduced; NOT flat →
the format floor compounds too). Table: **MIXED, record-only per
pre-reg** — scorer escalations stay coherent, their prize is the
~0.17 content margin over a free ~0.11 any-words floor. (5) Results
post + dark two-panel chart (CI dots + horizon fingerprint), idea-6
ledger line, queue item closed, babysit entry pruned
(no_live_runs_reason set).

**Next**: `queue_cli.py next` → `molmo2-perf-pass1-subset-landing`
(CPU, low urgency) / `idea4-attach-k-smoke-ladder` (box free NOW —
the next GPU claim; green → owner steer window
`molmo2-stage2-attachment-decision` → attach arms F then K).
`run_work_next` armed at close.

*Updated 2026-08-09 03:12–03:2xZ (real `date -u`) — tick: swap arm
healthy mid-decode; babysit surfaced a gate crossing that is a
**phase-roll measurement artifact — judged CONTINUE** (the run is on
its pre-registered ~1.6 GPU-h ≤ 3 budget).*

**Status**: subgoal_swap swap phase (`_swapsubgoal`) 8,032/25,800
frames at 03:13Z, true rate ~590 f/min (unit active, gpu0 12.7
GiB/63%) → rc ~03:45Z + in-unit dump check, exactly on the boundary.
Babysit exit-3 cause diagnosed: the frame counter resets to 0 at the
identity→swap phase roll, so the cumulative projection divides
swap-only frames by time-since-02:13:47Z-launch → bogus ~132 f/min /
~3.2 GPU-h vs the 3.0 gate. Real total ~1.6 GPU-h. **CONTINUE, no
action on the run**; diagnosis anchored in the babysit.toml entry.
Box FREE (next claim K-smoke ladder).

**Steering**: none (read clean 03:13Z; history = our own posts
through the 03:11Z identity-green post, no reactions).

**Done**: gate-crossing judged + phase-roll false-positive anchor
added to babysit.toml (no code change this tick — the generic
babysit.py gap, multi-phase logs with per-phase counters breaking the
cumulative projection, is owed to the chained session alongside the
rc prune).

**Next**: rc ~03:45Z → chained work session (`run_work_next` already
armed 03:11Z): dump-check verification, babysit prune + phase-roll
projection fix, frozen Δ_swap / swap-vs-oracle / horizon-mirror reads
against the frozen 3-row table + results post; then
idea4-attach-k-smoke-ladder on the free box.

*Updated 2026-08-09 01:43–03:1xZ (real `date -u`) — work session
(bounded, chained): **perf-pass1 box ladder CLOSED and read out — the
bundle is SLOWER on the real recipe (C −7.3%, P1 −10.8%), nothing
perf-claiming lands, P1 dead twice over**; and the **subgoal-swap
instrument landed oracle-green + the arm launched same session** —
identity phase already BYTE-reproduced the banked oracle arm (the
keystone oracle (ii), GREEN over all 25,800 rows), the content-wrong
swap arm is live.*

**Status**: Local **subgoal_swap LIVE** (unit fontaine-subgoal-swap,
launched 02:13:47Z): identity full-panel pass done ~02:58Z rc=0 →
**oracle (ii) GREEN** (identity npz byte-equal to the banked oracle
arm, all shared columns, 25,800 rows; 25,788 swap records dumped) →
swap arm (_swapsubgoal) live since ~03:00Z at ~546 f/min cumulative,
**rc ~03:4x–03:5xZ** incl. the in-unit mechanical dump check (oracles
i+iv, abort-on-red); ~1.6 GPU-h total ≤ 3 gate. Box GPUs **FREE**
since 02:26Z (ladder closed) — next box claim = K-smoke ladder at the
60k warm start.

**Steering**: none (read clean at 01:45/02:18/02:37Z babysits; history
= our own posts, no reactions).

**Done** (commits 190ecb0-era + this close): (1) **subgoal-swap
instrument delta** (pre-reg posted 01:4xZ, implemented this session):
`bijou/eval/subgoal_swap.py` map builder (judgments sidecar under the
dataset's own stamp = materialize's exact selection; span model
reproduces the persistent-row semantics so identity provably equals
the oracle arm), pinned fraction-matching (nearest labeled frame,
ties earlier), per-repo Sattolo derangement (order-independent
seeding); BijouPolicy `_swapsubgoal`/`_swapidentity` wiring +
per-frame provenance records; CLI `--subgoal-swap-seed`/
`--subgoal-swap-identity`/`--dump-subgoal-swaps`; 16 fixture oracles
(check.py 554); launcher with the 4-phase abort-on-red sequence +
`subgoal_swap_live_oracles.py` (selftest green, all abort branches
fire). LAUNCHED 02:13:47Z. (2) **perfpass1 box ladder readout**
(closed 02:26:32Z rc=0): OVERLAY PASS (0.0816 ≤ 0.3919 band); LADDER
A=2.251s / B=2.495s / C=2.415s → **B −10.8% / C −7.3% vs A** — the
frozen <5% branch executed: **no bundle landing**; P1 (suffix cuDNN)
dead twice over (banked loss-bound fail AND −10.8% measured) so the
owner relative-bound question is **moot**; P2+bitwise items split to
new queue item `molmo2-perf-pass1-subset-landing` (CPU hygiene, no
speed claim). Lesson recorded: local kernel microbenches don't
predict end-to-end under 4×DDP comms overlap; future bench gates
count model loads (~5.5 GPU-h actual vs 3.0 ceiling, CONTINUE judged
01:42Z). Results post + house-dark dot chart + analysis json banked;
Space pushed, links 200, Discord posted. (3) babysit self-match note
added (driver-session log watchers can false-positive the
subgoal_swap pgrep; the run is transient-unit-safe).

**Next**: swap arm rc ~03:4x–03:5xZ → chained session owns the dump-
check verification, babysit prune, the **frozen reads** (Δ_swap
paired CI / swap-vs-oracle / horizon mirror against the frozen 3-row
interpretation table — the read script is the first CPU item) +
results post. Then `queue_cli.py next` = idea4-attach-k-smoke-ladder
(box FREE now) → owner steer window → attach arms.
`run_work_next` armed.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames),
3rd launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3
rungs (+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
23:24Z–02:37Z 08-08 **COMPLETE +~3.2 GPU-h (≤ 8 gate)**;
 08-08 daytime: local rung-(b) preflight+stage1
08:49–10:15Z **+~1.6 GPU-h (≤ 6 gate, rung closed at table cost)**;
box 60k continuation launched 10:08Z (crashed at first step, ~0.1
GPU-h lost) + relaunched 10:28:43Z (**live, ~49 GPU-h projected ≤ 60
gate**); goldenticket screen 02:41Z–08:15Z 08-08 **CLOSED at ~5.55 GPU-h ≤ 6
gate** (s1 ~1.7 + s2 ~0.85 + s3 2.99); box molmo2 chain: 40k train
to ~04:0xZ, greedy ~1.7 GPU-h, draws10_t1 04:54–07:22Z **~10 GPU-h
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box + local both
idle from ~08:15Z pending the next pre-registered launches). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 03:17–04:0xZ (work, bounded, chained rc owner;
exploit, 0 GPU-h new — the swap arm closed on its pre-registered
~1.5 GPU-h ≤ 3): subgoal-swap CLOSED end-to-end — rc=0 03:42:36Z,
dump + execution oracles all green, frozen reads banked (Δ_swap
−0.113 [−0.161,−0.060]; swap−oracle +0.166 [+0.127,+0.205]; table
MIXED record-only: ~40% format floor / ~60% content margin), results
post + chart live, queue item closed, babysit entry pruned. Babysit
phase-roll projection gap fixed generically (e8ef9d5, 2 anchored
oracles) + subgoal_swap_results.py landed under check.py (557).
Discord read clean at every poll.

Session 2026-08-09 03:12–03:2xZ (tick; 0 GPU-h new — the live swap
arm pre-registered and counted): babysit exit-3 on subgoal_swap
judged CONTINUE — the ~3.2 GPU-h projection is a phase-roll artifact
(frame counter resets at identity→swap, cumulative divides swap-only
frames by time-since-launch; true swap rate ~590 f/min, rc ~03:45Z,
~1.6 GPU-h ≤ 3 gate); diagnosis anchored in babysit.toml, generic
multi-phase-counter babysit.py fix owed to the chained work session
(`run_work_next` already armed). Discord read + history clean; queue
validate green depth 3.
