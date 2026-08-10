# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 16:07–16:1xZ (real `date -u` at write: 16:13) —
tick (babysit): **quiet interval — no new owner traffic, box healthy
with a NEW RUN-BEST, work session armed for the owner-GO'd rig
fine-tune runbook.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~27,320 at poll, **NEW RUN-BEST 5.96@27000** (prior 6.1306@25000;
rungs since @25000: 6.21@25500 / 6.20@26000 / 6.58@26500 /
5.96@27000), 27.4 f/min, gate 69.2/155 GPU-h, vram ~71.7 ×4 vs 77
bar, babysit exit 0; next save boundary @30000 ~17:4xZ (matched-Δ
legs @25500–@30000 bank at that catch), endpoint ~08-11 ~12:00Z.
Local H100 FREE, reserved for the rig fine-tune. Blog Space still
capped: `usedStorage` 998.6 MB — HF GC hasn't run; no push
attempted, the delete+recreate ask stands for ~08-11 morning if
unchanged.

**Steering**: none new — `read` empty, `history -n 5` shows the
15:48/15:52Z LeRobot v2.1/v3.0 joint-convention thread already
answered 15:54Z (requirements folded into the runbook item, commit
c013413); no new reactions.

**Done**: babysit exit 0 (liveness 8 procs, util 58–100% ×4);
registry updated with the run-best + rung state; queue validate — 8
open but depth 1 (< 2) → refill flagged; `run_work_next` re-armed;
Space storage checked (998.6 MB, unchanged).

**Next**: chained work session opens with
**molmoact2-rig-finetune-runbook** (owner GO 15:24Z: read
`codebase_version` off both rig repos → runbook post + param sheet
in-channel → objection window, silence=launch → fine-tune on the
local H100 with its own pre-reg + babysit entry + gate), then
refills the queue to depth ≥ 2. @30000 boundary ~17:4xZ falls to
that session or the next tick. Blog-Space tail stays manual-only.*

*Updated 2026-08-10 12:24–15:0xZ (real `date -u` at write: 14:58) —
work session: **MolmoAct2 out-of-band eval DELIVERED END-TO-END in
one session** — finalized pre-reg → predictor + oracle-gated reads
instrument → 500-frame smoke → owner challenge answered with the
contamination split → full 25,800-frame sweep → frozen reads →
3-policy HTML report → posted, at ~1.3 of the 8 GPU-h gate. Plus:
reports migrated to a new `fontaine-reports` static Space (owner
directive), the blog-navbar regression found and fixed, and the box
@25000 boundary caught with a new run-best.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~24,9xx at last poll, **NEW RUN-BEST 6.1306@25000**, save @25000
async green (155.4 s behind boundary — record-only watch; @20000 was
21.3 s), 26.6 f/min, ~62.4/155 GPU-h; endpoint ~08-11 ~12:00Z, next
boundary @30000 ~17:4xZ. 10 new matched-delta legs banked
(@20500–@25000): 10-leg mean −0.06, 34-leg running mean ≈ +0.005 —
dead in-band. Local H100: FREE again 14:24Z. Unit
`fontaine-blog-migrate` retrying the blog push+squash behind HF's
storage GC (up to 6 h, log `~/logs/blog_space_migrate.log`).

**Steering** (five owner threads, all answered same-session):
(1) 12:59Z "shockingly poor — are we doing inference correctly?" →
answered with the smoke contamination split (trained-on repos 7.24
vs state-copy 7.33 parity; unseen 17.40 vs 7.68) — harness correct,
finding real; (2) 13:14Z **amendment: exclude `willnorris/bbox-2`**
→ applied to reads+report before any real read, oracle branch added;
(3) 13:48Z "what uses 1 GiB on the blog?" → live tree is 263 MB
(230 MB reports), the GiB is un-GC'd git history; (4) 13:51Z **move
reports to `fontaine-reports` + squash** → done as a *static Space*
(dataset repos serve HTML as text/plain — tested, pages wouldn't
render; owner told of the substitution), 64 files live + curl-200,
72 blog links rewritten, 31 redirect stubs for old deep links,
squash queued behind GC; (5) 14:13Z "navbar gone" → morning-incident
casualty: hashed `toc-b9c2449c.js` missing on the Space, re-uploaded
(small files pass the cap), fixed + confirmed 200. **INCIDENT —
five owner messages (14:33–14:38Z) consumed unseen**: the 14:52
babysit was piped through `grep -E "exit|liveness"`, its embedded
consume-once Discord read swallowed five report-feature requests;
owner called it out 14:57Z ("you can't just look at the latest
message"). 2nd incident of the read-never-truncate class (grep
variant); memory updated, owned in-channel, recovered via
`history -n 15`, ALL FIVE executed same-session: (a) heun-30
original single-draw 80k row (banked npz byte-pairs — zero GPU;
matched-window 5.09/5.20/4.86); (b) mean-of-10 draws for 80k →
NOT banked full-panel, queued as `snapflow80k-draws10-panel-eval`
(~8–12 GPU-h, launch next session, objection window open);
(c+d) MAE-by-timestep charts ×3 splits, all 8 models, in the
report; (e) methods+results blog post
`posts/2026-08-10-molmoact2-oob-results.md`. Plus (6) 14:49Z trunk
names on every policy → done + re-uploaded (Gemma-4-E2B vs
Molmo2-4B vs Molmo2-ER mapping posted); full-50 bbox-2 exclusion
effect table added (~−0.07 per arm). LATE THREAD 15:0x: (7) owner
"Let's skip 2" → draws10 eval CANCELLED pre-launch, 0 GPU-h, queue
item closed; (8) report downloaded instead of rendering in Firefox
→ root cause: first 10.5 MB upload auto-tracked as LFS → CDN
redirect; fixed by trimming gallery 32→24 (8.3 MB) + stripping the
per-path LFS rule from .gitattributes + re-upload as regular blob —
now direct `200 text/html`; (9) **navbar broke AGAIN** → my
`fontaine-blog-migrate` retry unit's split commits were landing
DELETE chunks (toc/searchindex/reports) while add chunks bounced on
the cap — each 10-min retry re-deleted the sidebar. **Unit
STOPPED** (do not re-arm any auto-pusher against the capped Space);
toc restored at the referenced path (200). Measured: cap headroom
≈1.4 MB (998.6 MB of 10⁹ un-GC'd) — search stays broken (14 MB
index) until HF GC clears; offered the owner delete+recreate of the
blog Space as the clean escape if GC hasn't run by tomorrow.
FINAL THREAD 15:1x–15:2x: (10) navbar STILL broken for owner →
real cause found: partial pushes left MIXED page generations (some
live pages reference the NEW `toc-16164281.js` which didn't exist);
both toc hashes now uploaded, every page generation verified 200;
(11) owner naming catch: "snapflow 80k" is WRONG — the 80k model is
the **flow teacher** `bijou_flow_artrunk_h1024@80k` (SnapFlow = the
1-NFE distilled student); all report/post labels renamed "flow
teacher 80k" + naming note; (12) CORRECTION reversing my skip
advice: mean-of-10 WAS banked (truncated file listing caused the
false "not banked" + 8–12 GPU-h estimate; real cost zero) — the
seating-stage full-panel draws10 npz added as a row at 0 GPU-h,
oracles green: matched-window 4.05/4.12/3.90, slots 2nd behind
top-10-tickets; owner may veto ('seating keying' labeled);
(13) 15:22Z report adds → snapflow student 30k 1-NFE row (only
per-frame-npz student config; matched-window 4.29/4.37/4.11 — beats
teacher single-draws, confirms the distillation story), ALL 10
models on every trajectory chart, 10-color palette, legend moved to
a standalone strip + below the timestep panels (nothing covers
series); (13b) 15:27Z gallery doubled + SPLIT into clean vs
contaminated sections (2×24 frames, per-split strides; JPEG thumbs +
dpi-72 charts keep it 9.4 MB non-LFS, direct 200); (14) **15:24:16Z OWNER
GO: MolmoAct2 rig fine-tune on the local GPU + runnable runbook** →
queued as the NEXT session's first action (item
molmoact2-rig-finetune-runbook; param sheet in-channel before any
GPU minute, owner-agreed silence=launch); local H100 reserved.

**Done**: molmoact2-oob-panel-eval CLOSED (commits 00a9feb, b6cc2a7,
this one): pre-reg finalized (immutable + Amendment 1), smoke green
12:5xZ (tripwires passed), sweep rc=0 14:23:47Z (352 f/min,
25,800 frames), frozen reads banked — matched 1.0 s window, core,
excl. amendment: **snapflow top-10-tickets 3.90 / 60k-cont 4.46 /
40k 4.56 / stable-key 5.06 / er15k 5.89 / state-copy 8.32 /
MolmoAct2 13.87** (clean 16.97, contaminated 7.00; every paired
read MOLMOACT2-WORSE, tight CI95s) — **the released SO-100 fine-tune
does not transfer outside its 1,220-repo mixture**: beats state-copy
only on its own training repos (−0.75) and still trails snapflow
there (+3.29 [+3.11, +3.48]). 3-policy HTML report (32-frame
gallery) + reads json + contamination repo list on fontaine-reports,
reports.md section added, numbers in-channel 14:37Z. @25000 box
boundary caught + legs banked. babysit ×3 exit 0; sweep babysit
entry pruned.

**Next**: `queue_cli.py next` → **molmoact2-rig-finetune-runbook**
(owner GO 15:24Z: runbook post + param sheet in-channel → objection
window → launch fine-tune on local H100, own pre-reg + babysit
entry + gate). er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-cont (5.8602);
boundaries @30000 ~17:4xZ then @35000 ~20:4xZ 08-10. BLOG-SPACE
TAIL (manual only — the retry unit is stopped and must NOT be
re-armed): each session, check `usedStorage` (`repo_info` expand);
when it drops below ~500 MB, do ONE `upload_folder` of the current
book (delete_patterns searchindex/toc/reports) + `super_squash` +
curl-verify nav/search/stubs + post the all-clear; if still capped
by ~08-11 morning, ask the owner for the delete+recreate go (offered
15:2xZ). If the owner wants a base-MolmoAct2 second arm or a
dataset-repo mirror of reports, both are pre-scoped adds.*

*Updated 2026-08-10 11:59–12:2xZ (real `date -u` at write: 12:13) —
tick (babysit): **OWNER GO on the MolmoAct2 plan, 20 s before
session start — acknowledged + spec confirmed in-channel, queue
item updated, work session armed.** 11:59:33Z: "The molmo2act plan
sounds good, let's eval the so101 checkpoint. Could we also
generate an html eval report similar to the one we normally do,
but with both our best policy (snapflow 80k) predictions vs.
molmo2act and state copy on the same frames. as well as summary
statistics obviously."*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
20,720 at poll, probe 6.58@20500, run-best **6.3658@20000**, 26.7
f/min window, vram ~71.7 ×4 vs 77 bar, ~52.7/155 GPU-h; endpoint
~08-11 ~12:00Z, next save boundary @25000 ~14:4xZ. Local H100:
FREE, reserved for the MolmoAct2 eval.

**Steering**: GO on the OOB eval + a NEW report requirement —
side-by-side HTML on the SAME frames: **snapflow 80k** (owner's
name for the flow teacher `bijou_flow_artrunk…@80k` — banked panel
npzs on disk: top-10-tickets 5.1847 = best banked, stable-key
single-draw 6.5997 anchor; zero GPU re-eval needed) vs
**MolmoAct2 SO100_101** vs **state-copy**, plus summary stats.
Replied 12:01Z confirming: headline = top-10-tickets, matched
30-step/1.0 s window primary (50-step secondary), pooled +
clean-633/contaminated-245 splits, paired CI95. Conversational
polls 12:04–12:12Z: no follow-up.

**Done**: babysit exit 0 (liveness 8 procs, util 53–100% ×4, gate
52.7/155, no new legs — next rung ~@21000). GO reply posted 12:01Z;
queue item `molmoact2-oob-panel-eval` updated with the report spec
+ GO stamp (objection window on the finalized pre-reg still applies
before the full sweep; smoke may start now); `updated_utc` fixed.
Queue validate OK, 8 open. `run_work_next` confirmed armed.

**Next**: chained work session executes the owner-gated item:
finalize pre-reg → `molmoact2_panel_predict.py` + oracle-gated
matched-window instrument → 500-frame smoke + scale sanity → full
25,800 sweep (systemd unit, ≤ 8 GPU-h gate) → 3-policy HTML report
+ reports page + in-channel numbers. Box rides to endpoint ~08-11
~12:00Z → chained panel_v2 → paired CI95 vs banked 40k (6.0079) +
60k-cont (5.8602). Rungs record-only; kill lines unchanged;
boundary @25000 ~14:4xZ. No lit refills until re-enabled.*

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





Session 2026-08-10 12:24–15:0xZ (work; +~1.3 local GPU-h logged —
MolmoAct2 sweep+smoke; exploit): the owner-GO'd MolmoAct2
out-of-band eval DELIVERED end-to-end in one session (pre-reg
finalized 00a9feb → predictor + oracle-gated matched-window reads +
3-policy report generator → smoke green → 25,800-frame sweep rc=0
14:23:47Z at 352 f/min → frozen reads → HTML report + numbers
in-channel 14:37Z; headline: released SO100_101 fine-tune doesn't
transfer outside its training mixture — 16.97 clean vs 7.00
contaminated vs state-copy 8.32, snapflow top-10-tickets 3.90 best).
FIVE owner threads answered at conversational cadence (inference
challenge → contamination-split proof; willnorris/bbox-2 exclusion
amendment; 1 GiB question; reports→fontaine-reports migration
directive — done as a static Space after measuring that dataset
repos serve HTML text/plain, 72 links rewritten + 31 stubs, squash
queued behind HF GC on unit fontaine-blog-migrate; navbar bug =
missing hashed toc js from the morning incident, fixed + 200).
Box @25000 boundary caught: NEW RUN-BEST 6.1306@25000, async save
green (155.4 s — record-only watch), 10 matched-Δ legs banked,
34-leg running mean ≈ +0.005. babysit ×3 exit 0. Local H100 free at
close; run_work_next armed (migrate-unit verification + er_60k
boundaries are the next touch points).

Session 2026-08-10 16:07–16:1xZ (tick, babysit; 0 new GPU-h —
er_60k rides 69.2/155): quiet tick. No new owner traffic (the
15:48/15:52Z joint-convention thread stands answered 15:54Z, folded
into the runbook item c013413; no new reactions). Box healthy:
**NEW RUN-BEST 5.96@27000** (rungs since @25000: 6.21 / 6.20 / 6.58
/ 5.96), 27.4 f/min, vram in-band, babysit exit 0; @30000 boundary
~17:4xZ + matched-Δ legs @25500–@30000 left to the chained session
or next tick. Blog Space GC still pending (usedStorage 998.6 MB
unchanged — no push, manual-only tail stands). Queue depth 1 →
run_work_next armed: the chained work session opens with the
owner-GO'd **molmoact2-rig-finetune-runbook** (codebase_version
check → runbook + param sheet in-channel → objection window →
launch on the local H100) and refills the queue.
