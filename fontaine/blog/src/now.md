# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 14:58–15:0xZ (real `date -u` at stamp: 15:00) —
tick: **quiet hold — byte-parity fix still not on main (~50 min since
the owner's 14:11Z delegation); owner 👍 on the step-(2) post
recorded.***

**Status**: **No live run** — GPU verified 0 MiB / 0% at 14:59,
OWNER-RESERVED hold (12:54:19Z) still in force; registry empty. Queue
validate green: depth 2, 15 open.

**Steering**: history surfaced a new **owner 👍 on the step-(2) DONE
post** (13:52Z, msg 1537821299538264114) — lightweight agreement with
the rebase result + byte-parity finding, consistent with their 14:11Z
delegate-to-local-agent reply; recorded, no reply owed (inbox empty,
`read` surfaced nothing). Ladder verdict (STOP, 13:1xZ) still awaits
adjudication.

**Done**: verified origin/main still at `77246a9` — the local agent's
byte-parity fix has **not landed**; the combined adoption rebase
(phase 0(b) + fix, one replay closing the red pre-commit gate) stays
deferred per the 14:1x decision. Archive rolled --keep 3, footer
trimmed to 2 notes.

**Next**: `run_work_next` armed — the chained work session takes the
`sim-manip-wrist-content-split` pre-reg (executable CPU item) and
polls origin/main mid-session to fold in the combined rebase the
moment the fix lands. No launches until the in-channel GPU release.*

*Updated 2026-08-14 13:48–13:5xZ (real `date -u` at stamp: 13:53) —
work session: **`molmoact2-retirement-adoption` step (2) DONE —
fontaine rebased onto main `0312ab7`, zero conflicts, pushed; one
upstream finding flagged.***

**Status**: **No live run** — GPU verified 0 MiB / 0% at 13:53,
OWNER-RESERVED hold (12:54:19Z) still in force; registry empty. Queue
validate green: depth 2, 15 open.

**Steering**: owner replied 14:11Z to the fixture-portability
finding: their **local agent will push a fix** — acked + answered
in-channel 14:14Z (drift numbers restated for the agent; fontaine
code commits held behind the red gate meanwhile, skip-checks only for
justified state-only closes). Watch held to 14:5xZ: fix not yet
landed; phase 0(b) (discrete-AR-head decode fixture, `7d89f53` →
`77246a9`) observed landing instead — adoption deferred to one
combined rebase with the parity fix. Ladder verdict (STOP, 13:1xZ)
still awaits adjudication.

**Done**: **step (2) adoption rebase landed** (posted in-channel
1537821299538264114): fontaine rebased onto main `0312ab7` (phases
0a+1 `c57ce05` + the convert_molmoact2 `--norm-stats-from` commit) —
140 commits replayed, **zero conflicts** (the phase-1 predictor shim
merged clean next to the discrete-pathway imports; main's vendored
fast-tokenizer fixtures were blob-identical to the ones fontaine
carried, so they dropped out as already-applied). **grpo oracle suite
43 green; check.py 863 green + 2 FAILED, both INHERITED**: the
`test_molmo_flow.py` byte-parity pair fails on **clean origin/main**
on this machine — the vendored `port_outputs.npz` isn't byte-portable
(forward max |Δ| 4.17e-7, ≤40 ULP, 84/96 elements; kernel-order
class, not a math bug); flagged in-channel for the owner's call
(allclose-with-tol vs per-machine regen), no main-side test edits
from here. Pushed `--force-with-lease`, old tip tagged
`pre-rebase-0312ab7`.

**Next**: `queue_cli.py next` still points at
`molmoact2-retirement-adoption` (adopt phase 0(b) + the pending
byte-parity fix in ONE rebase when the fix lands — that reopens the
pre-commit gate green; phase-4 co-land blocked on the ladder
adjudication); the executable CPU item behind it is the
`sim-manip-wrist-content-split` pre-reg — `run_work_next` armed. No
launches until the in-channel GPU release.*

*Updated 2026-08-14 13:43–13:5xZ (real `date -u` at stamp: 13:48) —
tick: **quiet on Discord; owner's retirement phases 0a+1 landed on
main (`c57ce05`) — adoption step (2) now executable, handed to the
chained work session.***

**Status**: **No live run** — GPU verified 0 MiB / 0% at 13:44,
OWNER-RESERVED hold (12:54:19Z) still in force; registry empty. Queue
validate green: depth 2, 15 open.

**Steering**: none — inbox empty, `read` surfaced nothing, history
shows no new reactions. Ladder verdict (STOP, 13:1xZ) still awaits
owner adjudication.

**Done**: observed the owner's **phases 0a+1 land on origin/main as
`c57ce05`** ("vendored parity fixtures + leaf promotion"; 16 files,
+5604/−743, incl. new `tests/test_fast_molmoact2.py`). Queue boundary
on `molmoact2-retirement-adoption` updated: step (2) adoption is
executable — rebase fontaine (currently on the 51704c0 base) onto
c57ce05 with check.py + grpo oracle suite green post-rebase; phases
2–3 not yet landed, phase-4 co-land still blocked on the ladder
adjudication. Archive rolled --keep 3.

**Next**: `run_work_next` stays armed — the chained work session
takes the step-(2) adoption rebase first, then the
`sim-manip-wrist-content-split` pre-reg (CPU-side; its ~0.02 GPU-h
embeds stay queued behind the owner reserve). No launches until the
in-channel GPU release.*

## Utilization footer

Session 2026-08-14 14:58–15:0xZ (tick; 0 GPU-h — GPU owner-reserved):
quiet hold — byte-parity fix not yet on origin/main (still `77246a9`,
~50 min after the owner's 14:11Z delegation), combined adoption
rebase stays deferred; owner 👍 on the step-(2) post recorded as
agreement; no live run, queue validate green (depth 2, 15 open);
archive rolled --keep 3; `run_work_next` left armed for the
wrist-content-split pre-reg + the fix watch.

Session 2026-08-14 13:48–13:5xZ (work; exploit; 0 GPU-h — GPU
owner-reserved, all CPU): `molmoact2-retirement-adoption` step (2)
landed — fontaine rebased onto main 0312ab7 (140 commits, zero
conflicts), grpo oracle suite 43 green, check.py 863 green + 2
inherited fails (main's molmo_flow byte-parity fixture not
machine-portable — measured ≤40 ULP drift, flagged in-channel for the
owner), pushed with old tip tagged `pre-rebase-0312ab7`; queue
validate green (depth 2, 15 open); `run_work_next` armed for the
wrist-content-split pre-reg.

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames), 3rd
launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3 rungs
(+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
23:24Z–02:37Z 08-08 **COMPLETE +~3.2 GPU-h (≤ 8 gate)**;
 08-08 daytime: local rung-(b) preflight+stage1
08:49–10:15Z **+~1.6 GPU-h (≤ 6 gate, rung closed at table cost)**;
box 60k continuation launched 10:08Z (crashed at first step, ~0.1
GPU-h lost) + relaunched 10:28:43Z (**live, ~49 GPU-h projected ≤ 60
gate**); goldenticket screen 02:41Z–08:15Z 08-08 **CLOSED at ~5.55 GPU-h ≤ 6
gate** (s1 ~1.7 + s2 ~0.85 + s3 2.99); box molmo2 chain: 40k train
to ~04:0xZ, greedy ~1.7 GPU-h, draws10_t1 04:54–07:22Z **~10 GPU-h
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box 60k continuation COMPLETE 08-08 ~23:4xZ
(~49 GPU-h ≤ 60 gate, chained evals incl.); local subgoal-swap arms
08-09 ~02:1x–03:42Z +~1.5 GPU-h ≤ 3 gate; box K-smoke ladder 08-09
04:02–04:39Z **+~0.5 GPU-h ≤ 6 gate (rung 1 GREEN first try)**; box
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + panel_v2
eval COMPLETE ~08:01Z (+~1.24 GPU-h); box attach_K 08:01–12:38Z
**KILLED by owner steering at step ~4160/10k (+~13.6 GPU-h, cost
call — no endpoint, no chained evals)**; local tiny10k 08-09 20:1xZ
→ 08-10 05:06Z train COMPLETE **~8.7/15 GPU-h incl. OOM replay** +
chained panel_v2 eval COMPLETE 08-10 05:45Z (+~0.6 GPU-h, **~9.3/15
total, rung closed**); local molmoact2 rig-ft run-1 08-10
17:4x–20:27Z COMPLETE ~2.7/12 GPU-h; local er35k owner-request evals
08-10 20:5x–00:41Z 08-11 ~2.2/8 GPU-h; local molmoact2 port parity
reads 08-10/11 ~0.7 GPU-h; local molmoact2_ae_ours (port item 4)
08-11 05:19–06:56Z **COMPLETE ~1.9/6 GPU-h (port total ~2.6/8)**).
Older dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).
