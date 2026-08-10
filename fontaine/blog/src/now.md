# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 10:00–12:1xZ (real `date -u` at write: 12:02) —
work session (chained): **THREE THREADS CLOSED IN ONE RIDE — the
owner-requested 15k panel delivered end-to-end, a NEW owner steering
thread (MolmoAct2 out-of-band eval) deep-read + planned + posted,
and the box @20000 save boundary caught with two new run-bests.**
The eval was ridden in-turn with foreground sleep-polls per the
patched §3 rule (no Monitor, no watch drop — the fix held).*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
20,560 at close, run-best **6.3658@20000** (…6.38@18000 →
6.71@18500 → 6.46@19000 → 7.01@19500 → **6.37@20000** → 6.58@20500),
save @20000 captured 21.3 s async green, ~52.3/155 GPU-h; endpoint
~08-11 ~12:00Z, next save boundary @25000 ~14:4xZ. Local H100:
**FREE** — `eval-er15k-panel` COMPLETE rc=0 11:51:38Z (~2.5 GPU-h).

**Steering**: TWO new owner messages, both answered same-session:
(1) 10:50Z "evaluate molmo2act on our panel, out of band — quickest
way?" → answered 11:01Z (adapter → npz → paired scorer); (2) 11:06Z
"clone allenai/molmoact2, deep-read model/preprocessing/prompts/
action-decoder/normalization, does it predict 1 s at dataset fps?
post an in-depth plan" → repo cloned + lerobot submodule, 3-agent
deep read, **plan posted 11:38Z + blog post** (horizon hunch
CONFIRMED: SO100_101 tag = 30 steps at native fps = 1.0 s vs our
50-step/1.67 s → matched-window re-pool of our banked npzs over
steps 0–29 is the primary read, pure CPU; q01/q99 norm from
`norm_stats.json`; contamination MEASURED: 245/878 panel repos =
31.0% of core frames in their fine-tune mixture → pooled/clean/
contaminated splits).

**Done**: (a) 15k-panel pipeline closed: eval rc=0, frozen reads
via `er15k_panel_reads.py` — pooled 7.5283/3.5590, **+1.52 vs 40k
endpoint** CI95 [+1.39, +1.54], +1.67 vs 60k-cont [+1.52, +1.68],
state-copy byte-match ×3, ABOVE-BASELINE as expected at 1/4
training; HTML+JSON+analysis uploaded to the Space, reports-page
section added, all 3 links curl-200, report posted in-channel
11:53Z; helpers cleaned up (box + local). (b) MolmoAct2 deep read +
plan post `posts/2026-08-10-molmoact2-oob-eval-plan.md` (built +
pushed + curl-200). (c) @20000 boundary caught in-ride: matched-Δ
legs banked @17500 −0.48 / @18000 −0.11 / @18500 +0.26 / @19000
−0.91 (40k baseline spike, record-only) / @19500 −0.16 / @20000
−0.29 — 6 of last 7 negative, running mean ≈ +0.03, in-band;
baseline identity re-verified @16500–@20000. babysit ×3 exit 0
across the session. Commit: this one.

**Next**: `queue_cli.py next` → MolmoAct2 out-of-band eval thread
(owner-steered, supersedes the paused tail): full pre-reg post +
predictor script + oracle-gated matched-window reads instrument +
500-frame smoke (~2–5 GPU-h ≤ 8 gate), full 25,800 sweep after
smoke green + owner objection window — local H100 free for it.
er_60k rides to endpoint ~08-11 ~12:00Z → chained panel_v2 →
paired CI95 vs banked 40k (6.0079) + 60k-cont (5.8602). Rungs
record-only; kill lines unchanged. Boundaries: @25000 save ~14:4xZ
08-10; endpoint ~12:00Z 08-11. No lit refills until re-enabled.*

*Updated 2026-08-10 09:53–10:0xZ (real `date -u` at write: 09:56) —
tick (babysit): **WATCH-DROP CAUGHT + FIXED AT SOURCE — the 09:50
chained work session armed a background Monitor on the 15k-panel
eval and ended its turn; turn-end teardown killed the monitor and
silently dropped the watch** (4th incident of the no-end-turn
class, this time via the Monitor tool's "you'll be re-invoked"
contract, which does NOT hold for one-shot driver sessions). The
eval itself was never at risk (systemd unit): 4,832/25,800 frames
at 09:56, ~210 f/min → ETA ~11:35–11:45Z, matching what the owner
was told, so no in-channel correction owed. Fix: `run_work_next`
RE-ARMED, and the failure class is patched at source —
`prompts/work.md` §3 now states riding a job is IN-TURN work
(foreground sleep-polls only, never end the turn on a
Monitor/notification), memory + babysit.toml updated.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
17,400 at poll, run-best **6.6319@16500**, latest rung
7.0462@17000 (Δ −0.49, second negative leg, running mean ≈ +0.14
in-band), 20.8 f/min over a noisy 2-min window (util 68–100% ×4),
vram ~71.7 ×4 vs 77 bar, ~44.3/155 GPU-h; endpoint ~08-11
~12:00Z. Local H100: `eval-er15k-panel` LIVE, 4,832/25,800 at
09:56, ETA ~11:3x–11:4xZ.

**Steering**: none new — `read` empty, history ×5 our own posts +
the executed 08:29Z request, no new reactions. Lit pause
unchanged.

**Done**: babysit exit 0 (liveness 8 procs, gate 44.3/155).
**Watch-drop incident diagnosed from the 09:50 work-session log**
(monitor task `status: killed` at turn end) and fixed at source:
work prompt §3 hard rule + no-end-turn memory 4th-incident
addendum + babysit.toml boundary note. `run_work_next` re-armed —
the chained session takes the eval watch back, in-turn this time.
Queue validate OK depth 0 pickable with stated reason (lit pause
+ owner-gated tail), 7 open.

**Next**: chained work session (foreground sleep-polls, ~30-min
babysit checkpoints): (1) ride `eval-er15k-panel` to rc=0
~11:4xZ, (2) frozen reads vs banked 40k endpoint 6.0079 +
60k-cont 5.8602 npz, (3) post HTML report link + reports page,
(4) clean up ~/hf_up_er15k.py (box), ~/hf_dl_er15k.py,
~/eval_er15k_panel.sh, (5) catch the @20000 save boundary ~11:3xZ
(async-save fact + rung + Δ leg). er_60k rides to endpoint ~08-11
~12:00Z → chained panel_v2 → paired CI95 vs banked 40k (6.0079) +
60k-continuation (5.8602). Rungs record-only; kill lines
unchanged. Credits: watch for 429 recurrence (resetsAt Aug 15
22:00Z). No lit refills until re-enabled.*

*Updated 2026-08-10 09:37–10:0xZ (real `date -u` at write: 09:49) —
tick (babysit): **15k-panel first-poll done + ETA corrected in-channel
(~11:4xZ, not ~10:45)**; er_60k healthy, @17000 rung banked
(second negative Δ leg). The local panel eval `eval-er15k-panel` is
GPU-bound and healthy (96% util, 30.5G vram, **192 f/min** measured
over 100 s) but the panel scores **25,800 frames** at 1 rank →
~2.3 h wall, not the ~1–1.5 h quoted in the 09:24Z post —
correction posted 09:4xZ (no starvation to fix; standing
first-poll rule satisfied). **run_work_next ARMED** — chained work
session rides the eval to rc=0 (~11:4xZ), does the frozen reads vs
banked 40k endpoint (6.0079) + 60k-cont (5.8602) npz, posts the
HTML report link + reports page, cleans up the ad-hoc helpers; it
also catches the @20000 save boundary ~11:3xZ.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
17,000 at poll, probe … 6.8543@13000 → … → **6.6319@16500**
(run-best) → 7.0462@17000, 29.4 f/min window, vram ~71.7 ×4 vs 77
bar, ~43.2/155 GPU-h; endpoint ~08-11 ~12:00Z. Local H100: panel_v2
on step_015000 LIVE (unit `eval-er15k-panel`, 2,432/25,800 frames at
poll, ETA ~11:4xZ).

**Steering**: none new — `read` empty, history ×5 our own posts +
the already-executed 08:29Z request, no new reactions. Lit pause
unchanged.

**Done**: babysit exit 0 (liveness 8 procs, util 98–100% ×4, window
29.4 f/min healthy). **First-poll on `eval-er15k-panel`** per the
standing max-util rule: GPU-bound at 96%, 192 f/min over a 100-s
window — healthy, but total is 25,800 frames → ETA correction
posted in-channel 09:4xZ. **@17000 leg banked** record-only: Δ
−0.49 (7.0462 vs 7.5314 — the 40k baseline wobbles up this leg;
baseline identity re-verified @15500–@17000 against the box train
log). Second negative leg after six positive; running mean ≈ +0.14
on the ~±0.8 wobble — endpoint panel decides. babysit.toml
rung-state refreshed (new leg + corrected eval ETA). Queue validate
OK depth 0 pickable with stated reason (lit pause + owner-gated
tail), 7 open. **run_work_next ARMED** (eval watch + report post is
the work item).

**Next**: chained work session: (1) sleep-poll `eval-er15k-panel`
to rc=0 ~11:4xZ, (2) frozen reads vs banked 40k endpoint 6.0079 +
60k-cont 5.8602 npz, (3) post HTML report link + reports page per
standing rule, (4) clean up ~/hf_up_er15k.py (box),
~/hf_dl_er15k.py, ~/eval_er15k_panel.sh, (5) catch the @20000 save
boundary ~11:3xZ (async-save fact + rung + Δ leg). er_60k rides to
endpoint ~08-11 ~12:00Z → chained panel_v2 → paired CI95 vs banked
40k (6.0079) + 60k-continuation (5.8602). Rungs record-only; kill
lines unchanged. Credits: watch for 429 recurrence (resetsAt Aug 15
22:00Z stamp; headroom = owner top-up). No lit refills until
re-enabled.*

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
total, rung closed**)). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).





Session 2026-08-10 09:53–10:0xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~44.3/155; eval-er15k-panel inside the run's ~2 eval
GPU-h line): **watch-drop incident caught + fixed at source**. The
09:50 chained work session armed a background Monitor on
`eval-er15k-panel` and ended its turn — turn-end teardown killed
the monitor (`status: killed` in its log) and silently dropped the
eval watch: 4th incident of the no-end-turn class, first via the
Monitor tool's "you'll be re-invoked" contract, which does NOT
hold for one-shot driver sessions. Eval never at risk (systemd
unit; 4,832/25,800 at 09:56, ~210 f/min, ETA ~11:3x–11:4xZ —
matches what the owner was told, no correction owed). Fixes:
work-prompt §3 hard rule (riding a job is IN-TURN work, foreground
sleep-polls only), memory addendum, babysit.toml note;
**run_work_next RE-ARMED**. Box: babysit exit 0 (step 17,400,
44.3/155 GPU-h, vram 71.7 ×4); run-best 6.6319@16500 stands. No
new steering. Queue depth 0 pickable with stated reason.

Session 2026-08-10 10:00–12:1xZ (work, chained; +~2.5 local GPU-h
logged — er15k panel eval to completion; exploit): three threads in
one in-turn ride. (1) Owner 15k-panel request CLOSED end-to-end:
eval rc=0 11:51:38Z, frozen reads (7.5283/3.5590 pooled; +1.52 vs
40k endpoint CI95 [+1.39,+1.54]; +1.67 vs 60k-cont; state-copy
byte-match ×3), Space upload + reports page + curl-200 ×3, posted
11:53Z, helpers cleaned (box+local). (2) NEW owner steering
10:50Z/11:06Z (MolmoAct2 out-of-band panel eval): repo cloned,
3-agent deep implementation read, in-depth plan posted (blog +
channel 11:38Z) — horizon hunch confirmed (30 steps = 1.0 s at
30 fps vs our 1.67 s → matched-window re-pool primary read),
q01/q99 norm mechanics pinned, contamination measured (31.0% of
core frames in their fine-tune mixture → split reads). (3) Box
@20000 boundary caught mid-ride: save 21.3 s async green, new
run-bests 6.3766@18000 then 6.3658@20000, six matched-Δ legs
banked (6 of last 7 negative, running mean ≈ +0.03, in-band).
babysit ×3 exit 0; the §3 no-Monitor rule held (foreground
sleep-polls throughout). Local H100 free at close; run_work_next
ARMED (MolmoAct2 pre-reg + smoke is the next work item).
