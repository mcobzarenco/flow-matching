# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 19:20–20:1xZ (real `date -u` at stamp: 20:07) —
work session, bounded: **GRPO signal probe is LAUNCH-READY (pre-reg
finalized + posted, instrument complete: SDE decode wired end-to-end,
parallel driver runs stochastic groups) — and a mid-session owner
steer landed a headline: seed 6 at a 30-second budget GRABS the boat
and carries it to 1.0 cm of the disk. The failure mode on that seed
is place/release, not grasp.***

**Status**: no live jobs, GPU idle (seed-6 rerun ran 19:54–19:56Z,
~0.02 GPU-h, entry pruned). Queue validate green (depth 3, 13 open,
stamp 20:05:49Z).

**Steering**: owner 19:25:39Z ("seed 6 is a clear grab and lift —
rerun just seed 6 with a longer time horizon; the idea was 30
seconds") — replied 19:47Z, executed end-to-end same session: the
eval-20 runs gave 15 s vs the sim100 protocol's 30 s; new
`--episode-seconds` states the budget in time (`c26a99e`); the 30 s
rerun confirms **grab + carry** (boat in the gripper jaws on the wrist
cam ~4 s, benchy→disk 1.04 cm at t=15.9 s, never releases, sets the
boat back at spawn). Result + video posted 19:58Z, amendment 2 +
results on the convmap pre-reg page. Open asks: GRPO probe
launch-on-handback go (20:0xZ post), seed-16/all-20 30 s extension
offer (19:58Z), v3-rerun unhold + arm set (15:13Z), disk-draws
sign-off.

**Done** (commits `8b6d034`, `c26a99e`, `a06c33d`, close-out): (1)
**GRPO probe prep COMPLETE** per the owner's 13:16Z "get everything
ready" — SDE sampler wired end-to-end (`--sde-noise-level` on both
drivers, per-item keyed step noise batch-composition-invariant, new
`SDE_STEP_DOMAIN` stream), parallel driver work units are now
(seed, draw) with `--draws`/`--ar-temperature` (the ≤3 GPU-h probe
path), oracles for all of it (check.py 797 green); **finalized
pre-reg posted** (frozen: seeds 0–14, 30 s episodes, signal bar
median group std ≥ 0.25 cm, 5 cells + anchors + a=0.3 hedge cell,
within-driver paired-only discipline honoring the parallel-oracle
FAIL; the 13:36Z sequence predecessors are both done ⇒ launches on
GPU handback, no v3-rerun hard dependency). (2) seed6-30s owner call
closed (above). (3) Hygiene: convmap post was missing from
posts/index.md — indexed; queue item updated (prereg pointer → the
finalized page, "15 replans" drift corrected).

**Next**: `queue_cli.py next` → CPU lanes: `lit-sim-improvement-levers`
(owner-called lit slice), `sim-wrist-compositing`. GPU on handback:
GRPO probe (launch-ready), v3 rerun (pends unhold). `queue.json`
canonical.*

*Updated 2026-08-12 19:15–19:2xZ (real `date -u` at stamp: 19:22) —
tick, babysit: **caught and closed a dropped owner ask — 19:01:42Z
"Can you link a video?" was consumed mid-run by the prior session but
never directly answered (the 19:11Z results post linked only the
chart). Replied 19:16Z with three direct mp4 links.***

**Status**: no live jobs, GPU idle (0%, 0 MiB). Queue validate green
(depth 3, 13 open). `run_work_next` already armed (19:13, prior
session) — the work session chains after this tick. pgrep straggler
(ssh box checkpoint-ls, 25 s old) = background probe, benign.

**Steering**: owner 19:01:42Z video ask answered 19:16Z — arm A
seed 6 (directed reach to 1.4 cm), arm A seed 16 (the knock-away —
first boat contact), arm B seed 10 (knock-away); all three
curl-verified 200 through the Space redirect before posting, plus the
per-arm directory pattern for the other 17×2. Conversational hold
kept in-session ~4 min after the reply (30 s Discord poll loop, no
follow-up, no reactions at 19:20Z), then handed to the chained work
session — it boots only when this tick ends, and an idle-tick hold
delaying a 4-h work session is the banned idle pause; it rejoins the
thread at boot.
Lesson: an owner ask landing mid-run must get its own direct reply —
the results post didn't count (per the standing rule), and the ask
sat 14 min. Open asks unchanged: v3-rerun unhold + arm set (15:13Z),
GRPO memo review, disk-draws sign-off.

**Done**: video-links reply (19:16Z); hygiene — queue validate green,
GPU-idle confirm, marker check, straggler dispositioned.

**Next**: chained work session → CPU lanes
(`lit-sim-improvement-levers` owner-called, `sim-wrist-compositing`).
GPU idle pending the v3-rerun unhold. `queue.json` canonical.*

*Updated 2026-08-12 18:39–19:1xZ (real `date -u` at stamp: 19:12) —
work session, bounded: **release-eval20-officialmap DONE — the INERT
read is PARTIALLY OVERTURNED: under the official lift sign the release
engages the scene (a 1.4 cm near-touch, a knock-away — the boat was
touched, which never happened in the parent's 20 episodes) but still
0/20 pickups on both arms; grounding, not units, remains the blocker.
Canonical shim going forward = the snippet map exactly.***

**Status**: no live jobs, GPU idle again — both arms ran ridden
end-to-end 18:55–19:06:43Z (~0.25/0.4 GPU-h; first-poll 100% util /
20.8 GB; babysit entry pruned same session). Queue validate green
(depth 3, 13 open).

**Steering**: no new owner messages this session (polled at boot,
pre-launch, at both arm boundaries, and close). Executed the standing
18:19/18:34/18:36Z steering: official-map rerun, snippet map EXACTLY as
arm A, arm B = +wrist_roll −90, per-episode in-channel updates (40
episode lines streamed as rows landed). Open asks unchanged: v3-rerun
unhold + arm set (15:13Z), GRPO memo review, disk-draws sign-off.

**Done** (commits `45a41b6`, `fb76a96`, close-out): (1) instrument —
sign-carrying `--convmap-override` (`JOINT=[SIGN,]OFFSET`, oracles,
checks 793 green) + `--rows-jsonl` per-episode stream on the parallel
driver + the Discord watcher. (2) pre-reg amendment 1 posted BEFORE
launch; tripwires under the official map recorded (lift mirror covers
7.5% uncovered vs +180's 27.9%; arm B first-action 2.62° vs anchor
6.31°; arm A wrist-identity 34.0° = the known clamp signature, run per
the owner's call). (3) the read: arm A (snippet exact) mean −0.11,
seed 6 directed reach to 1.4 cm (+4.61), seed 16 knock-away −5.26;
arm B mean −0.09, 2 approaches, 1 knock-away; **A vs B NULL** (−0.02
[−0.75,+0.66], 11/20 exact ties) — snippet identity wrist stays
canonical; vs ftrig arms all CI-incl-0 (bracket claim softened: ft
steps buy *more frequent* engagement, 7/20 vs 1–2/20 approaches, not
engagement from zero). (4) INERT explicitly re-dispositioned on the
pre-reg page (amendment 1 results); MIRROR_MARGIN estimator lesson
flagged to the box (a real, documented mirror rejected despite winning
coverage — wants a coverage tiebreak / external-doc override).
Artifacts: dark per-seed chart + rows + 20 videos per arm on the
reports Space (curl 200); Discord launch + per-episode + results posts
18:58–19:0xZ.

**Next**: `queue_cli.py next` → CPU lanes: `lit-sim-improvement-levers`
(owner-called lit slice), `sim-wrist-compositing`. GPU idle pending the
v3-rerun unhold (15:13Z ask). `run_work_next` armed. `queue.json`
canonical.*

## Utilization footer

Session 2026-08-12 19:20–20:1xZ (work, bounded; **+~0.02 GPU-h**
seed-6 30 s rerun ridden in-turn; exploit + owner steering): GRPO
signal probe prep COMPLETE (SDE decode end-to-end, parallel stochastic
groups, finalized pre-reg posted — launch-ready on GPU handback, owner
sequence predecessors both done); owner 19:25Z seed-6 steer executed
end-to-end same session (`--episode-seconds` fix + 30 s rerun: GRAB
CONFIRMED, carried to 1.04 cm of the disk, failure mode = place/
release; video + amendment 2 results posted). check.py 797 green ×3
commits; queue drift fixed ("15 replans" → the sim100 30-replan
protocol); convmap post indexed into posts/index.md.

Session 2026-08-12 19:15–19:2xZ (tick, babysit; 0 new GPU-h — GPU
idle): caught a dropped owner ask — 19:01:42Z “Can you link a video?”
was consumed mid-run by the prior session and answered only by the
results post (chart link — doesn’t count per the standing rule) →
direct reply 19:16Z with 3 curl-verified mp4 links (arm A seeds 6/16,
arm B seed 10) + the per-arm directory pattern; ~4-min in-session
conversational hold (no follow-up, no reactions), handed to the
chained work session which rejoins the thread at boot. Hygiene green (queue depth 3, run_work_next
armed 19:13, ssh straggler benign). Archive roll: 1 main entry (17:29
work), 2 footer notes (18:19 tick, 17:29 work).

Session 2026-08-12 18:39–19:1xZ (work, bounded; **+~0.25 GPU-h** — 2×
tripwire probes + two 20-seed parallel arms, ridden end-to-end;
exploit, owner steering): release-eval20-officialmap DONE same session
— sign-carrying override instrument landed, pre-reg amendment posted
pre-launch, both arms streamed per-episode to Discord per the owner
ask, INERT re-dispositioned (PARTIALLY OVERTURNED: official lift sign
unlocks scene contact, still 0/20 pickups; grounding remains the
blocker), canonical shim = snippet map exactly, A-vs-B wrist arms null.
No steering traffic; launch + ~6 per-episode + results posts.

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
