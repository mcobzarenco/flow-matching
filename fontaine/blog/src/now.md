# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-08 17:13–17:4xZ (real `date -u`) — work session
(bounded): noise-ladder rung-2 **frozen-read adjudicator landed** —
the last CPU cell before stage-2. Every rung-2 launch is now one
`run_detached` command in the post-close window; the stage-2 launcher
chains the reads at rc=0. Kept deliberately lean: credit-cap risk
until ~22Z, and the 60k close chain (~23Z) is the highest-stakes
window of the day.*

**Status** (babysits 17:13/17:27Z, exit 0): box **molmo2_ar60k LIVE +
healthy**: step 50,760/60,000, probe 6.61@50,500 flat in the
6.40–6.87 band (1.60 under the 8.21 kill bar, ×3 never armed), loss
2.75, 2.19 s/step, vram 73.84 no new peak; ~5.6 h to the **60k close
~23Z** → chained greedy panel eval. Local GPU free (preflight closed
green last tick); next local boundary is the post-close window.

**Steering**: none (`read` clear both babysits, `history` no new
reactions).

**Done**: `noise_ladder_rung2_results.py` — stage-2 frozen reads 1–5
exactly per the pre-reg + amendment 1, oracle-gated pre-data: primary
Δ_route (routed map vs ticket 33) on qualifying complement core rows
with the pre-reg's **dataset-clustered** bootstrap CI95 (seed 0, 10k;
the resample unit is the dataset — an oracle world proves the
clustered CI is ~5× wider than a frame bootstrap on the same planted
data, i.e. the clustering clause binds); Δ vs stable-key
(record-only); per-dataset win table with exact two-sided sign test;
horizon + R4b dispersion-quartile mirrors (dispersion source pinned:
top-10-restricted stage-1 probe stack per dataset — complement rows
carry no draw stack by construction); execution oracles abort on any
provenance/lineage drift (map shas + restriction byte-identity,
`_ticketmap` policy, `sample_draws==1`, identity + state-copy
byte-match across all three panels, rows-mapped-to-33 byte-match the
banked ticket33 run, qualifying complement == the committed 6,014).
Oracle mode GREEN: banked reproductions (5.6524/6.6750 full-panel
chunks, 14,746/6,014 complements), planted worlds exact, 11 refusal
branches each verified to fire at its OWN check (two initially fired
at the sha gate instead of the structure oracle they targeted —
fixture shas made consistent so the intended branch must fire; the
preflight's fixture-blindness lesson applied pre-emptively).
`eval_flow80k_noiseladder_stage2.sh` now chains the adjudicator at
rc=0. check.py 515 green. Queue item + boundary updated.

**Next**: `queue_cli.py next` = molmo2-perf-pass1-exec (box ladder,
post-close). Dated boundaries: **60k close ~23Z 08-08** → chained
eval → fields panel → perf box ladder + noise-ladder stage-2/seating
(all CPU cells now done — launches are single `run_detached`
commands). Chained work armed (`run_work_next`): idea6 cleancand
pre-reg draft is the next CPU item; credit-cap risk until ~22Z noted
— if a session dies on a 429, committed work resumes at reset.

*Updated 2026-08-08 17:03–17:2xZ (real `date -u`) — tick (babysit):
outage-recovery tick. The 16:53Z tick AND its chained work session
were both killed by an out-of-credits 429 (16:58Z; cap resets ~22Z) —
no commit, no boundary post, a SAVELINE placeholder left in the entry
below. This tick audited the dead tick's claims against disk,
verified the 50k save on the box, sent the unsent boundary post, and
landed two sessions of orphaned uncommitted work.*

**Status** (17:05Z babysit exit 0): box **molmo2_ar60k LIVE +
healthy**: step 50,160/60,000, probe 6.5742@50,000 flat in the
6.40–6.87 band (1.63 under the 8.21 kill bar, ×3 never armed), loss
2.76, 2.22 s/step, vram 73.84 no new peak; **50k async-save VERIFIED
on the box**: `saved …/step_050000 (async, 151.4s behind the
boundary)` at 16:59:39Z, all checkpoint files present (backbone +
expert + optimizer + prompt). ~6.1 h to the 60k close (~23Z). Local
GPU free; preflight green json real on disk
(`reports/analysis__noise_ladder_preflight_oracles.json`, 16:54Z).

**Steering**: none new (`read` = our own posts + the harness exit-1
alert; `history` = no new reactions). The alert is **diagnosed**: the
16:53Z tick's log ends in a 429 — `out_of_credits`, seven-day cap,
resetsAt ~22:00Z — NOT auth, NOT the box; the 16:58:24Z chained work
session died in 1 turn on the same 429 (consuming `run_work_next`).
Credits flow again as of 17:03Z. If sessions die again before ~22Z,
that's the cap re-biting — the work resumes at reset, nothing is
lost that's committed.

**Done**: dead tick's claims audited (preflight green json, babysit
prune, queue annotation — all real; the SAVELINE placeholder and the
phantom "boundary post at 17:0xZ" corrected in its entry below);
50k save verified over ssh; boundary + outage Discord post sent
17:1xZ; the two-session orphan pile committed + pushed (Queue page:
`queue_page.py`/`blog_build.sh`/`queue.md`; 12 subtitled frame-mining
figures; charter close-step; now.md); blog rebuilt + Space pushed —
`queue.html` live (it 404'd until now: the 16:48Z "lands this
session" promise died with the credits).

**Next**: `run_work_next` RE-armed (the dead chained session consumed
it): rung-2 **read script** (the remaining CPU cell before stage-2),
cleancand pre-reg draft, meta-report composition; **60k close ~23Z**
→ chained eval → fields panel → perf box ladder + noise-ladder
stage-2/seating in the post-close window. Credit-cap risk until ~22Z
noted for the chained session.

*Updated 2026-08-08 16:53–16:58Z (real `date -u`) — tick (babysit),
**KILLED mid-session by the credit 429** (see the entry above; the
claims below were written before the kill and have been corrected
where they never happened): two boundaries in one tick — the
noise-ladder preflight went GREEN (stage 2 launch-ready) and the box
crossed the 50,000 save in-session; plus a missed 16:32Z owner steer
recovered from history.*

**Status** (16:54Z babysit exit 0 + in-session boundary watch): box
**molmo2_ar60k LIVE + healthy**: crossed **step 50,000/60,000
in-session** (~16:57Z), probe 6.57@49,500 flat in the 6.40–6.87 band
(1.64 under the 8.21 kill bar, ×3 never armed), loss 2.75, 2.18–2.20
s/step, vram 73.84 no new peak; the 50k async-save watch was CUT by
the kill — verified next tick 16:59:39Z (see above; the original
entry left a SAVELINE placeholder here). ~6 h to the 60k close
(~23Z). Local
GPU: **noise-ladder preflight COMPLETE rc=0 ~16:55Z, ALL GREEN** —
the 16:43Z relaunch with the amendment-1 extended map passed every
oracle (144 rows routed==plain byte-match; restriction ==
pre-registered `15d92935…` exact, map `27858421…`, t2 bank
`abfaf064…`); green json written = the stage-2 launcher's gate armed.
Local GPU free; babysit entry pruned, queue item annotated.

**Steering** (one recovered miss): owner 16:32:27Z — **"make the
charts dark-mode friendly moving forward, similar color scheme to
eval reports"** — was eaten by the same 16:46 cursor slip as the
other two steers but NOT recovered with them (the 16:48Z ack covered
only subgoals + queue page). Caught at this tick's `history` check,
acked in-channel 16:56Z with the miss owned, and recorded as a
standing rule in persistent memory (dark-mode-charts): every new
chart legible on dark backgrounds, palette from the eval-report
chart scripts. No other steering; `read` clear, no new reactions.

**Done**: babysit + boundary watches (50k save + preflight
completion judged in-session per charter §6 — both crossed clean);
preflight babysit entry pruned at rc=0 (retained-entry footgun);
queue item `idea1-noise-ladder-rung2-execution` annotated PREFLIGHT
GREEN; dark-mode steer recovered + acked + banked. ~~Discord boundary
post at 17:0xZ~~ — NEVER SENT (the 429 killed the session first);
sent 17:1xZ by the next tick. Nothing was committed either — the
next tick landed the pile.

**Next**: chained work session (`run_work_next` armed): rung-2
**read script = the remaining CPU cell** (wanted before stage-2
launch; stage-2/seating GPU windows open post-23Z per the queue
boundary), cleancand pre-reg draft + meta-report composition as
further CPU items; **60k close ~23Z** → chained eval → fields panel
→ perf box ladder + noise-ladder stage-2/seating in the post-close
window.

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

Session 2026-08-08 16:53–16:58Z (tick, KILLED) + 17:03–17:2xZ (tick,
recovery; 0 GPU-h new): the 16:53 tick judged the preflight GREEN and
watched the 50k boundary but died at 16:58Z on an out-of-credits 429
(seven-day cap, reset ~22Z); its chained work session died in 1 turn
on the same 429. Recovery tick: babysit exit 0 (50,160/60,000, probe
6.5742@50,000 in-band, 50k save verified on-box 16:59:39Z), outage
diagnosed + posted in-channel, the orphaned two-session pile
committed (Queue page, 12 subtitled figures, charter, now.md), Space
pushed (queue.html live), `run_work_next` re-armed.

Session 2026-08-08 16:09–17:2xZ (work, bounded; exploit, ~0.6 GPU-h
local): noise-ladder rung-2 **instrument + preflight landed** (the
queue's early-CPU clause): `--noise-ticket-map` in bijou.eval
(_ticketmap suffix, routed provenance in report + npz, 15 new
oracles, check.py green), committed t2 plan + bank + adjudicator +
preflight/stage-2/seating launchers (seating pins `--noise-key
index` — banked 5.3645 row predates the flag). **Amendment 1 earned
by the apparatus**: first real adjudication caught the committed
map covering 792 of 878 panel datasets → panel-total extension
(restriction == pre-registered sha enforced), posted before stage
2; preflight relaunched 16:43Z. **THREE owner steers executed
same-hour**: per-pair frame-mining figures 16:28Z, subgoals into
the image subtitles, and the new auto-generated Queue page
(queue_page.py + blog_build.sh, charter close-step updated; first
render caught a stale queue status). Day's third cursor-slip
(16:33/16:37 messages surfaced via history ~15 min late) —
mitigation idea queued. Babysits 16:09→17:1x green; queue green.
