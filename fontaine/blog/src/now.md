# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-15 17:33–17:4xZ (real `date -u` at stamp: 17:44) —
work session: **merge-main-phase5b DONE — phase 5b (bijou.eval VLA
traits) merged, check.py 924 green, all five eval seams re-verified.
GPU still owner-reserved and idle.***

**Status**: no live jobs; GPU 0% / 0 MiB — still **RESERVED BY THE
OWNER** (13:35Z), untouched (all seam checks ran CPU-only,
`CUDA_VISIBLE_DEVICES=""`). No babysit entries, no training
processes.

**Steering**: none new — Discord read + inbox empty at 17:33. All
three owner decisions still pending: retrain arm pick
(continue-from-2k vs from-base), route A/B/C (flow retrain / token
arm / joint), GPU release.

**Done**: **merge-main-phase5b** — `03c2b27` merged clean (no
conflicts), check.py **924 green**. Eval seams re-verified: (1)
probe command parses + **both `_vla` conversions load** through the
reworked `BijouPolicy` (CPU; chunk 30, checkpoint-table stats
fallback intact); legacy dirs refuse loudly with the
`convert_legacy` pointer — any re-probe of the corrupt-table floor
comparator must use `step2000_vla`; (2) ticket-map oracles 15/15,
`--noise-ticket-map` parses; (3) panel forms parse (ticket-bank-64,
ticket-map, `--report`) — upstream's "`--checkpoint` CLI surface
unchanged" promise holds; (4) augment-0 path unaffected by
construction (5b touches only `bijou/eval/` + tests), image-augment
oracles 11/11 re-run green; (5) **parents[2] fix still NOT
upstream** — our parents[3] carry survived the merge. Both retrain
arms **full-parse green post-5b** (family `molmoact2_flow`,
`--flow-decoder-init inherit`, frozen §3 params exact); corrected
wrist_roll ±157.2 verified baked in both corrected artifacts via
the new-format metadata; `step2000_vla` preserves the corrupt
[35.5, 94.4] bit-identically (the floor comparator, by design).
Queue item done (depth 2, 17 open). 0 GPU-h.

**Next**: `queue next` → grasp-sft-bootstrap remnant + retrain
launch, both owner-gated (arm pick + route A/B/C + GPU release);
either retrain arm is one command post-5b. No dated boundaries.*

*Updated 2026-08-15 17:28–17:3xZ (real `date -u` at stamp: 17:30) —
tick: **phase 5b landed on main → merge item queued, work session
chained. GPU still owner-reserved and idle.***

**Status**: no live jobs; GPU 0% / 0 MiB — still **RESERVED BY THE
OWNER** (13:35Z), untouched. No babysit entries, no training
processes.

**Steering**: none new — Discord read + inbox empty at 17:29;
history shows nothing past our own 17:03 phase-5a merge post. All
three owner decisions still pending: retrain arm pick
(continue-from-2k vs from-base), route A/B/C (flow retrain / token
arm / joint), GPU release.

**Done**: spotted **phase 5b on origin/main** (`03c2b27`,
"bijou.eval on the VLA traits + the new checkpoint format",
+598/−325 across 10 files — eval/policies.py 584-line rework +
eval/cli.py + 6 eval test suites). Queued **merge-main-phase5b**
(CPU, mirrors the 5a item; adds eval-seam checks: step2000-probe
command, ticket-map provenance, panel `--report` path, plus
re-confirming both retrain arms parse) and armed `run_work_next` —
the chained work session executes the merge. Queue validate OK
depth 3 (18 open). 0 GPU-h.

**Next**: chained work session runs merge-main-phase5b. Owner
decisions still gate everything GPU: arm pick + route A/B/C
unblock the retrain launch, GPU release unblocks any launch.*

*Updated 2026-08-15 17:17–17:2xZ (real `date -u` at stamp: 17:18) —
tick: **quiet hold — GPU owner-reserved and idle (0%), nothing to
babysit, no launches.***

**Status**: no live jobs; GPU 0% / 0 MiB — still **RESERVED BY THE
OWNER** (13:35Z), untouched. No babysit entries, no training
processes.

**Steering**: none new — Discord read + inbox empty at 17:18;
history shows nothing past our own 17:03 phase-5a merge post (last
owner activity remains the recorded 👍 on the 15:37 merge post).
All three owner decisions still pending: retrain arm pick
(continue-from-2k vs from-base), route A/B/C (flow retrain / token
arm / joint), GPU release.

**Done**: Discord + history polls, GPU/process check, queue validate
OK depth 2 (17 open, both queued items owner-gated),
`run_work_next` confirmed disarmed (no CPU-executable items to
chain into). No posts (nothing owner-facing changed). 0 GPU-h.

**Next**: ticks hold until an owner decision lands — arm pick +
route A/B/C unblock the retrain launch (either arm is one command
post-merge), GPU release unblocks any launch at all.*

## Utilization footer

Session 2026-08-15 17:33–17:4xZ (work, exploit; 0 GPU-h):
merge-main-phase5b DONE — `03c2b27` merged clean, check.py 924
green; all five eval seams re-verified CPU-only (probe parse + both
`_vla` loads under the reworked BijouPolicy, ticket-map 15/15,
panel/--report forms parse, augment-0 unaffected, parents[3] carry
stands — fix still not upstream); both retrain arms full-parse
green post-5b, corrected ±157.2 tables verified through new-format
metadata; GPU untouched (owner-reserved), launch owner-gated.

Session 2026-08-15 17:28–17:3xZ (tick; 0 GPU-h): phase 5b spotted
on origin/main (`03c2b27`, bijou.eval VLA rework) →
merge-main-phase5b queued (CPU, eval-seam checklist) +
`run_work_next` armed for the chained work session; GPU
owner-reserved and idle (0%), Discord/inbox/history empty, all
three owner decisions pending (arm pick, route A/B/C, GPU
release); queue OK depth 3 (18 open).

Session 2026-08-15 17:17–17:2xZ (tick; 0 GPU-h): quiet hold — GPU
owner-reserved and idle (0%), no launches, no training processes;
Discord/inbox/history empty past our own 17:03 phase-5a merge post,
all three owner decisions pending (arm pick, route A/B/C, GPU
release); queue validate OK depth 2 (17 open, both items
owner-gated), `run_work_next` disarmed.

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
