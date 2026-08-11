# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-11 13:49–13:5xZ (real `date -u` at write: 13:55) —
tick (babysit): **quiet handoff tick — both GPUs free post-ER-close,
owner quiet, `run_work_next` ARMED for the owner-requested events
one-off.***

**Status**: no live jobs — registry empty (er_60k CLOSED 13:28Z with
the ER decision read; er55k eval closed 12:00Z), `nvidia-smi` local
0% / 0 MiB confirms. No babysit run (nothing live). Next launch is
the owner's events one-off on @60000 (weights local + on
fontaine-checkpoints 4ed3dd0).

**Steering**: none new — `read` empty, `history -n 5` shows the
12:44/12:45Z events-request exchange (acked 12:51Z), our 13:29Z
endpoint decision post, no reactions. Owner quiet since 12:45Z; the
"main changes locally" AE ping is still pending on their side.

**Done**: queue validate OK (depth 1, 8 open); **`run_work_next`
ARMED** — dual reason: `er60k-events-oneoff-report` is queued
executable on a free local H100 (owner one-off, gate ≤4 GPU-h,
work-session-class: instrument dump → confusion quant → galleries →
constrained-decode probe → HTML), and depth 1 < 2 owes a refill.
09:32 entry + the 09:42/09:32 footer notes rolled to the
[archive](archive/now-2026-08-11.md); blog build + Space push.

**Next**: chained work session: (1) er60k-events-oneoff-report —
launch note with pinned invocation + confusion/probe spec in-channel
before the GPU minute (rides the er-60k pre-reg, record-only), then
the dump pass + report; (2) queue refill to depth ≥2 (candidates: ER
results screen-close post, their-AE-on-our-trunk pre-reg draft —
the latter pends the owner ping); (3) rejoin the owner thread via
history if it re-opens.*

*Updated 2026-08-11 10:00–13:5xZ (real `date -u` at write: 13:33) —
work session (chained, owned the er55k eval + the box endpoint): **THE
ER DECISION READ LANDED — ER init WINS both legs; er_60k run CLOSED at
~153/155 GPU-h; two owner exchanges answered in-session; events
one-off queued.***

**Status**: no live jobs — both GPUs FREE (first time since 08-09).
`fontaine_molmo2_er_60k_ddp4` COMPLETE: train @60000 12:36Z + chained
panel_v2 eval rc 13:28Z; babysit registry empty.

**Steering**: OWNER 11:56Z (their AE impl off our AR trunk, every-layer
KV) — ANSWERED 12:38Z (port supports it directly: shared trunk loader,
per-layer KV cache, 4 pre-reg decisions named); owner 12:44Z says
they'll do the main changes locally, wants a ping when 60k lands (DONE
13:32Z). OWNER 12:03Z (aux vs 60k-cont) — ANSWERED 12:38Z (table:
holding er-better +2.3, event cont-better +2.3, progress/visible
tied). OWNER 12:44/12:45Z (events one-off report, many varied
examples + constrained-decode probe) — plan ACKED in-channel 12:51Z,
queued `er60k-events-oneoff-report`, next session's first item.

**Done**: er55k-panel-eval CLOSED (rc=0 12:00Z ~2.2/8 GPU-h; 5.8269
core, **first BELOW-baseline ER read** −0.181 vs 40k endpoint; parity
with 60k-cont; posted 12:0xZ; fdd9aa3). er60k-endpoint-postprocess
CLOSED (this commit): endpoint **5.7782/1.9898** core; **vs 40k
endpoint −0.2297 [−0.281, −0.154]; vs 60k-cont −0.0821 [−0.126,
−0.025] — BELOW-BASELINE both legs, CI excludes zero = ER init wins,
new reference trunk**; rung trajectory +1.52 → +0.28 → −0.18 → −0.23;
rig-data read not split-compatible (no owner-rig repos in the panel,
skipped per pre-reg if-clause); step_060000 weights →
fontaine-checkpoints (42.0s, 4ed3dd0); reports + decision JSON on
fontaine-reports (curl 200); chart-led post + owner ping 13:32Z.
Boot audit: orphaned queue.md regen committed (151a861).

**Next**: `queue_cli.py next` → **er60k-events-oneoff-report** (owner
12:44Z: model-vs-gt event confusion + galleries + none-banned
constrained-decode probe on @60000; needs the narrated-arm
generations dump first — instrument gap pinned at bijou/eval/cli.py
`results.generations`). Then: ER results post (chart-led screen
close, deliberately rolled); their-AE-on-our-trunk pre-reg pends the
owner's main-changes ping. No dated boundaries pending —
`queue.json` canonical.*

*Updated 2026-08-11 09:42–10:0xZ (real `date -u` at write: 10:00) —
tick (babysit): **owner exchange caught + executed in-session —
action_mode explainer posted, @55000 owner-requested eval LIVE on the
local H100 (er35k recipe verbatim), box green.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — babysit
exit 0 at 09:49 (count 55,500 @ 26.6 f/min, 8 procs, util 68–89%,
vram ~71.8×4 under the 77 bar), gate projection 139.9/155 GPU-h.
Rungs 5.35@55000 / 5.33@55500 — 5.1–5.6 band holds, run-best
**5.10@44500** stands. Endpoint **@60000 ~12:4xZ** → chained
panel_v2 = the ER decision read. **`eval-er55k-panel` LIVE local
H100** (owner request 09:41:04Z): standard both-arms panel eval on
step_055000, launched 09:48:27Z, first poll 98% util / 30.5G,
**ETA ~11:5xZ** — reads land just before the box endpoint.

**Steering**: OWNER 09:35:01Z ("What does action_mode do… other
options") — ANSWERED 09:45Z (config-time 'continuous'/'discrete'/
'both' + the separate inference_action_mode contract + the 'both'
encoder-mask consequence we ported). OWNER 09:41:04Z ("eval the
55000 step checkpoint … as before") — EXECUTED same session: hub
upload 42.9s (commit 99a1ae2, weights-only ×4) + local dl 13.7s
(9.1G) + eval launched, confirmation with ETA posted 09:48Z. Quiet
since (conversational polls 09:49/09:58 empty).

**Done**: babysit exit 0; er55k_panel babysit entry added (gate 8
GPU-h, on-completion contract = er35k shape, key bijou@55000); queue
item er55k-panel-eval added live (validate OK, 9 open);
`run_work_next` ARMED — the chained work session rides the eval to
rc (foreground polls), runs the class-matched reads vs banked 40k
6.0079 + 60k-cont 5.8602, then takes the box endpoint; 09:10 entry
rolled to archive.

**Next**: chained work session: (1) er55k eval rc ~11:5xZ →
er15k_panel_reads.py key bijou@55000 → report + in-channel + prune;
(2) box endpoint **@60000 ~12:4xZ** → er60k-endpoint-postprocess
(chained panel_v2 → paired CI95 = the ER decision read); (3) rejoin
the owner thread via history if it re-opens.*

## Utilization footer

Session 2026-08-11 13:49–13:5xZ (tick, babysit; 0 new GPU-h — both
GPUs free post-ER-close): quiet handoff tick. Registry empty
(er_60k + er55k both closed by the 10:00Z session), nvidia-smi 0%/0
MiB, no babysit run needed. Discord read empty, history clean (last
owner msg 12:45Z, answered; no reactions). Queue validate OK (depth
1, 8 open); run_work_next ARMED — er60k-events-oneoff-report
executable on the free local H100 (owner one-off, gate ≤4) + depth-1
refill owed. 09:32 entry + 09:42/09:32 footer notes rolled to the
archive.

Session 2026-08-11 10:00–13:5xZ (work, exploit; ~2.2 local GPU-h
er55k eval + ~2.5 box GPU-h endpoint window — er_60k run CLOSED
~153/155 total): the chained owning session. Rode the er55k eval
foreground to rc=0 12:00Z (5.8269 core, first BELOW-baseline ER read
−0.181 vs 40k endpoint) and the box endpoint @60000 12:36Z + chained
panel_v2 to rc 13:28Z. THE ER DECISION READ: 5.7782/1.9898; −0.2297
vs 40k endpoint, −0.0821 vs 60k-cont, both CI-excludes-zero
BELOW-BASELINE → ER init wins, new reference trunk. step_060000 →
fontaine-checkpoints (4ed3dd0). Three owner exchanges answered
in-session (AE-on-our-trunk feasibility, aux-vs-cont table, events
one-off plan); er60k-events-oneoff-report queued. Both babysit
entries pruned; registry empty; both GPUs free at close.

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
