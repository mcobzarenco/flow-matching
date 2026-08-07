# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 16:06–17:3xZ (real `date -u`) — work session:
**driver-background-task-guard LANDED** (`96522b9`, the item that
killed 3 GPU runs in one day) — four live-verified defense layers:
`run_detached.sh` required launch wrapper, KillMode=process on the
tick service, babysit DRIVER-CGROUP surfacing at every poll,
post-session cgroup guard with Discord alert; the kill signature is
now reproduced in tests with real transient units. Plus the standing
lit slice with same-session papers page
([decode-temperature](papers/decode-temperature.md)) — a written
directional prior for tonight's dT read. Both runs green.*

**Status** (babysit 17:20Z):
- box molmo2 AR 40k — 24180/40k, loss 3.009, 2.16 s/step, vram
  67.07 ≤ 71, 25.1 steps/min window. Probe 6.86@24000 (in-band,
  no ≥7.5 pair). Gate margin 4.93. ~9.5 h to 40k → endpoint ~08-08
  morning.
- local **ar100k_tsens_q4 rung t0.5** — 832/4301 @ 44.6 f/min
  window, cumulative 25.1 f/min → ~2.9 h/rung, ~2.3 h remaining on
  t0.5. The 16:06 boot-poll "18.5 h" gate crossing was the startup
  artifact again (model-load contaminating a 2-min cumulative) —
  adjudicated CLEAN, projection now 2.9 ≤ 12. Rung roll t0.5 → t0.7
  ~19:4xZ (repoint the babysit `log` stem); all rungs ~00–01Z 08-08.

**Steering**: none new (polls 16:06 / 16:44 / 17:20Z all clean).

**Done**: this session —
(1) **driver-background-task-guard** (`96522b9`, owner 13:05Z item,
3 incidents' evidence consumed): `fontaine/scripts/run_detached.sh`
= the codified REQUIRED wrapper for any job that must outlive a
session (systemd-run --user + PATH/HOME setenv + a grace-window
launch-death check that surfaces the exit-127 class);
`KillMode=process` on fontaine-tick.service (installed symlink =
repo file, daemon-reload applied — noncompliant launches survive
unit stop as stragglers instead of dying silently); babysit now
surfaces **DRIVER-CGROUP** at every poll when a registered run's
processes sit inside the driver cgroup — fires BEFORE the kill; two
self-match false-positive classes were found live and excluded
(probe ancestor chain; the `| sort -u` pipeline fork inheriting the
pattern-bearing cmdline); `driver_guard.py` post-session cgroup
scan wired into the driver with a 1-h-cooldown Discord alert.
Driver test: `tests/test_driver_guard.py` reproduces the incident-3
kill live (default KillMode kills a setsid child; KillMode=process
spares it; a run_detached job survives parent-unit teardown), plus
fake-/proc scan oracles + unit-file regression guard; babysit
oracles extended and both directions verified live on the running
tsens run (decoy straggler → SURFACED; compliant unit → clean).
check.py 460 green. Charter harness section, memory file, and 6
local launcher headers codified.
(2) **Lit slice + papers page**
([decode-temperature](papers/decode-temperature.md), 5 sources):
the dT read now has a pre-written directional prior (near-flat
table with asymmetry against T=1.3 on a unimodal-dominated panel —
2605.22493's deterministic-beats-generative-on-unimodal result +
MARS); BOKBO banked as the second independent strike on cheap
probe selectors (#19 selection rung); the q-token+CE trunk gains
its sample-complexity-optimality citation (2603.20538); DDVLA's
temperature-schedule hook parked (verified at source: 97.4 decay
vs 96.4/96.2 fixed/argmax — the search digest misquoted it).
(3) Queue: driver guard + lit slice → done; refill
`attach-launch-save-cadence-prep` (the #18.9 hooks become the
attach launchers' save-every call); validate green depth 2.

**Next**: `queue_cli.py next` → **idea19-tsens-dt-read-execution**
(opens at rungs completion ~00–01Z 08-08; the read now lands
against the papers page's written prior). Dated boundaries: tsens
rung roll ~19:4xZ (babysit stem repoint t0.5 → t0.7) → rungs
complete ~00–01Z 08-08 (dT read, record-only); molmo2 endpoint
~08-08 morning → #19 box obligations → K smoke ladder →
attach-screen window (first save validates async ckpt in
production; save-cadence prep item now queued for that launch).
**Every GPU launch from here goes through `run_detached.sh`.**

*Updated 2026-08-07 15:56–16:2xZ (real `date -u`) — tick (babysit +
incident + owner q): **tsens q4 DEAD AGAIN at poll — THIRD
driver-background-task-guard incident, ROOT CAUSE UPGRADED**: the
15:13:44Z setsid relaunch was killed ~15:54–15:56Z when
`fontaine-tick.service` finished (journalctl: unit stopped 15:56:18Z
→ systemd killed its whole **cgroup**; setsid escapes the terminal
session, NOT the cgroup). **Relaunched 15:58:26Z via `systemd-run
--user --unit=fontaine-tsens-q4`** — its own transient unit, actually
outside the driver's cgroup. Owner question 15:48Z ("what is tsens
t0.5?") answered in-channel 15:57Z. molmo2 green.*

**Status** (babysit 15:56Z):
- box molmo2 AR 40k — 23240/40k, loss 3.0747, 2.229 s/step, vram
  67.07 ≤ 71, 26.3 steps/min window. Probe 5.97@22500 → 6.05@23000.
  Gate margin 4.93. ~10.4 h stepping + saves → endpoint ~08-08
  morning.
- local **ar100k_tsens_q4 3rd launch** — rung t0.5 restarted from
  frame 0 (992 frames = ~40 min lost from the 2nd kill). Launch
  sequence: draws10_t1 registry entry temp-restored from `85cdc0a` →
  primary gate re-passed 12.7 ≤ 24 → entry re-pruned; first
  `systemd-run` attempt died exit 127 (`uv` not on the clean unit's
  PATH — fixed with `--setenv=PATH/HOME`); gate + rung T=0.5 start
  confirmed in `journalctl --user -u fontaine-tsens-q4`. babysit
  started_utc repointed 15:58:26Z. Rung roll t0.5 → t0.7 now ~19:1xZ
  (repoint the babysit `log` stem); all rungs ~01:3xZ 08-08.

**Steering**: owner 15:48Z asked what tsens t0.5 is — answered
15:57Z (T-sensitivity rung definition + record-only framing) in the
same post as the third-incident report; no further reply by close.

**Done**: tick — babysit (molmo2 green; tsens dead-run diagnosed to
the CGROUP mechanism via journalctl, not a compliance failure of the
setsid rule); tsens relaunched in a transient unit + gate re-passed +
registry dance executed + started_utc repointed; queue item
`driver-background-task-guard` gained third-incident evidence + the
systemd-run codification ask; memory file
no-end-turn-waiting-on-notifications REWRITTEN (setsid insufficient
by mechanism; systemd-run pattern + PATH gotcha); owner q answered;
`queue_cli.py validate` green (depth 2, 12 open); run_work_next
already armed.

**Next**: chained work session → **driver-background-task-guard**
(now with the true mechanism in hand: codify systemd-run as the
required GPU-launch wrapper, consider KillMode=process for the tick
service, driver test). Boundaries: tsens rung roll ~19:1xZ (babysit
stem repoint) → rungs complete ~01:3xZ 08-08 (dT read, record-only);
molmo2 endpoint ~08-08 morning → #19 box obligations → K smoke
ladder → attach-screen window (first save validates async ckpt in
production).

*Updated 2026-08-07 15:22–16:1xZ (real `date -u`) — work session:
**async checkpoint saves LANDED** (owner HIGH 13:58Z; `e3bdc93`,
oracle-gated BYTE-identical, default-on for every future train run) +
the checkpointing-systems lit slice with its same-session papers
page; tsens q4's first-poll gate scare adjudicated to a startup
artifact (measured ~3.3 h/rung, well inside the 12 GPU-h gate);
molmo2 green.*

**Status** (babysit 15:52Z):
- box molmo2 AR 40k — 23140/40k, loss 3.0727, 2.165 s/step, vram
  67.07 ≤ 71, 25.5 steps/min window. Probe **5.97@22500 (NEW LOW)** →
  6.05@23000. Gate margin 4.93. ~10.1 h stepping + saves → endpoint
  ~08-08 morning.
- local **ar100k_tsens_q4 rung t0.5** — 832/4301 @ 21–27 f/min
  (four timestamped inter-batch measurements 15:20→15:44 + babysit
  windows). The 15:22Z babysit surfaced a 19.3 h > 12 GPU-h gate
  crossing — **adjudicated startup artifact** (cumulative rate was
  contaminated by the ~6-min model-load before the first progress
  line); measured projection ~3.3 h/rung → ~10 GPU-h for all three
  rungs, gate PASS. Rung roll t0.5 → t0.7 ~18:3xZ (repoint the
  babysit `log` stem at the first tick after the roll); all rungs
  complete ~01:0xZ 08-08 → the queued dT-read item opens.

**Steering**: none new (polls 15:22 / 15:45 / 15:52Z all clean;
15:46Z landing post + this close post are ours).

**Done**: this session —
(1) **async-checkpoint-saves** (`e3bdc93`, the queue's owner-HIGH
item): `bijou/async_save.py` + train.py refactor. Root cause
measured-then-fixed: ~14 of the ~15.5 min/save was
`consolidate_state_dict` serially pickling whole optimizer shards
over the TRAINING NCCL group; now device→CPU capture at the boundary
(seconds), background `gather_object` over a dedicated gloo group,
exact `ZRO.state_dict()` merge replica, atomic `.tmp`-dir rename,
final save joined before teardown. Default ON (`--sync-save`
escape). Oracles (check.py 446 green): 2-rank BYTE-identity vs the
consolidate path at consecutive boundaries with the gather
overlapping main-thread collectives — two byte-level subtleties
pinned (pickle memoization of the shared betas tuple → identity-
memoized snapshot copies; `gather_object` de-interning rank 0's own
dict keys → keep the local capture object) — plus dir-level
byte-identity, `weights_only` resume round-trip, crash atomicity,
loud background-failure surfacing. Sync path is now atomic too.
(2) **Lit slice + papers page**
([checkpointing-systems](papers/checkpointing-systems.md), 6
sources): design corroborated (the CheckFreq/DataStates two-phase
shape); transfers banked as #18.9 hooks (pinned-buffer reuse,
save-frequency retune now saves are ~free, the data-iterator-state
resume gap named); non-transfers stated honestly (memory tiers,
multi-step spreading, sharded formats). ideas.md #18 item 9 + hook,
papers index + SUMMARY rows.
(3) Queue maintenance: async item + lit item → done;
`idea4-f-then-joint-prereg-draft` corrected queued→blocked (its
boundary needs Δ_seam); **driver-background-task-guard pulled
forward = next CPU item** (2 kills today); refills:
`idea19-tsens-dt-read-execution` (opens at rungs completion),
validate green depth 2.

**Next**: `queue_cli.py next` → **driver-background-task-guard**
(mechanize the turn-completion teardown fix — 2 GPU runs killed by
it today; run_work_next armed, next tick chains into it). Dated
boundaries: tsens rung roll ~18:3xZ (babysit stem repoint) → rungs
complete ~01:0xZ 08-08 (dT read, record-only); molmo2 endpoint
~08-08 morning → #19 box obligations → K smoke ladder →
attach-screen window — **first save of that launch validates the
async path in production: look for the `captured in Xs` +
`saved ... (async, Xs behind the boundary)` lines at first babysit**.

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
accruing from the 15:58:26Z systemd-run 3rd launch, ≤12 GPU-h gate). Older dated
snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 16:06–17:3xZ: all-CPU work session, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — exploit/infra + sanctioned
lit: driver-background-task-guard landed (`96522b9`, 4 defense
layers, kill signature reproduced in tests with live transient
units; the 3-incidents-in-one-day class is mechanized away) + the
decode-temperature lit slice with same-session papers page (5
sources; dT directional prior + 2nd probe-selector strike banked to
#19); tsens boot-poll gate scare adjudicated startup artifact
(measured 2.9 h projection ≤ 12); queue: 2 done, 1 refilled.

Session 15:22–16:2xZ: all-CPU work session, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — exploit/infra + sanctioned
lit: async checkpoint saves landed oracle-gated (owner HIGH,
`e3bdc93`, byte-identical keystone on a live 2-rank group; ~14%
wall-time payoff targeted at the attach screen) + the
checkpointing-systems lit slice with same-session papers page
(6 sources; pinned-buffer + save-frequency hooks banked to #18.9);
tsens first-poll gate scare adjudicated to startup artifact
(measured ~3.3 h/rung, PASS); queue: 2 done, 2 refilled, driver
guard pulled forward.

Session 13:04–15:2xZ: work session, ~2 GPU-h local (microbench redo +
post-merge reruns) + tsens launch — exploit/infra + owner-comms
heavy: merge chain end-to-end (pre-merge baseline banked, merge
85cdc0a with review fixes, 9.1×/2.5× single-stream speedups
measured, leaderboard measured-⏱ rewrite + row 5, review post
live), Ideas refactor + tags + archive sort (owner 13:02/13:26Z),
charter codification, async-ckpt queued HIGH (owner 13:58Z), tsens
q4 launched at the freed GPU (gate PASS 12.7≤24).

Session 09:49–10:3xZ: all-CPU, 0 GPU-h — exploit/instrument +
owner-steered comms: #19 dT-table read script landed
(tsens_dt_results.py, record-only per the pre-reg sensitivity
clause; oracle PASS pre-data incl. exact T=1.0 re-pool reproduction
+ 11 guard aborts); then owner steering 10:04Z executed live —
Ledger → Leaderboard (evergreen scoreboard incl. the mean-of-10
teacher/student rows + measured compute column) and the
slow-molmo2-saves question answered with on-box facts (37 GB/save →
save-pause-aware ETA). Refills: attachment-frontier lit slice +
decode-cost micro-benchmark prep (check.py 437).

Session 10:1x–10:5xZ: all-CPU, 0 GPU-h — instrument/lit-side
(chained): endpoint-runbook git-audit executed CLEAN at HEAD
`3d9e2a2` (zero mismatches/fix items across the whole blocked
endpoint chain — stems, flags, gates, pgrep patterns all byte-match
landed code); leaderboard decode micro-benchmark PREP landed
(`leaderboard_decode_microbench.py`, 7 configs × batched/single,
`--selftest` oracle PASS + posted pre-reg; GPU run executes at the
draws10_t1 boundary); APT 2606.12366 deep-read + init-thread
siblings (VLM4VLA 2601.03309, 2605.25802) — two papers pages live
same-session, #4 gains the named F-then-joint escalation rung + the
F-loses vision-first diagnostic, #17 gains a trunk-screening
criterion (check.py 437).
