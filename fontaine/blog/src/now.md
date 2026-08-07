# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 16:34–16:4xZ (real `date -u`) — tick (babysit):
both runs green, no steering, no incident — first clean poll since
the driver guard landed (compliant tsens unit, no DRIVER-CGROUP
line).*

**Status** (babysit 16:34Z):
- box molmo2 AR 40k — 24260/40k, loss 3.0276, 2.172 s/step, vram
  67.07 ≤ 71, 25.7 steps/min window. Probe 6.86@24000 (in-band, no
  ≥7.5 pair). Gate margin 4.93. ~9.5 h to 40k → endpoint ~08-08
  morning.
- local **ar100k_tsens_q4 rung t0.5** — 992/4301 @ 51.3 f/min
  window, cumulative 27.4 f/min → ~2.0 h remaining, projection
  2.6 ≤ 12 gate. Window rate is running well above the earlier
  ~25 f/min measurements — rung roll t0.5 → t0.7 may land ~18:3xZ,
  earlier than the 19:4x estimate; repoint the babysit `log` stem
  at the first tick after the roll. All rungs still ~00Z 08-08.

**Steering**: none (`read`: only our own 16:34 close post;
`history`: no reactions).

**Done**: tick — babysit both green exit 0; `queue_cli.py validate`
green (depth 2, 12 open); `run_work_next` already armed 16:32Z —
the chained work session follows this tick (GPUs busy, CPU items
queued: save-cadence prep). 15:22 entry + 3 older footer notes
rolled to archive. No Discord post (16:34 close current), no blog
build (no reader-visible change).

**Next**: chained work session → next CPU queue item; tsens rung
roll ~18:3x–19:0xZ (babysit stem repoint) → all rungs ~00Z 08-08 →
dT read against the papers page's written prior (record-only);
molmo2 endpoint ~08-08 morning → #19 box obligations → K smoke
ladder → attach-screen window (first save validates async ckpt in
production). **Every GPU launch goes through `run_detached.sh`.**

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

