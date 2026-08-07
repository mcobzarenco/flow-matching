# Now

*Updated 2026-08-07 13:04–15:2xZ (real `date -u`) — work session:
**merge chain executed end-to-end** (pre-merge baseline banked →
origin/main MERGED `85cdc0a` → post-merge speedup measured 9.1× →
leaderboard measured-⏱ rewrite + review post live) + owner steering
×4 executed same-session (Ideas refactor + tags, archive sort,
async-ckpt queued HIGH, SigLIP answered); **tsens q4 rungs LAUNCHED
15:01Z**; molmo2 green.*

**Status** (babysit 15:0xZ):
- box molmo2 AR 40k — 21640/40k, loss 3.1046, 2.183 s/step, vram
  67.07 ≤ 71, 26.2 steps/min window. Probe 6.22@20500 (NEW LOW) →
  6.55@21000 → 7.18@21500 (bouncy, no ≥7.5 pair, watch not
  tripped). Gate margin 4.92. ~11.1 h stepping + ~7 saves → endpoint
  ~08-08 morning.
- local **ar100k_tsens_q4 LIVE** (launched 15:01:40Z, primary gate
  PASS mechanized: 12.7 ≤ 24 GPU-h): rung T=0.5 scoring (verified
  live 15:1xZ, first progress line + GPU fed), then T=0.7, T=1.3
  sequential; ≤12 GPU-h gate; RECORD-ONLY dT diagnostic. Babysit
  entry ACTIVE; draws10_t1 entry pruned (footgun order honored:
  launcher consumed started_utc first). Repoint the babysit `log`
  stem as rungs roll (t0.5 → t0.7 → t1.3).
- **Decode microbench COMPLETE + merge landed.** Pre-merge
  sequential baseline: all 7 singles + students-batched + the redo
  of the killed cell (teacher_heun30_draws10 batched **747.3**
  ms/frame). The 12:56Z incident cost 4 batched cells their timing
  (rates lived in the killed parent; logs carry no timestamps) —
  only that one had a pre/post claim, hence the redo. **Merge
  `85cdc0a`**: zero conflicts; test_batched_draws.py + 5e-4
  tolerance + GIT_* scrub committed WITH it; the lost tile_memory
  residual guard was CAUGHT by its own surviving oracle at the
  pre-commit gate and restored. **Post-merge measured: mean-of-N at
  single-draw latency** — teacher draws10 single-stream 11,283.6 →
  1,245.0 ms/frame (**9.1×**), student 277.9 → 111.2 (**2.5×**);
  batched-throughput teacher 747.3 → 409.6 (1.8×); draws=1 controls
  reproduce ≤0.3%.

**Steering** (owner active 13:02–13:58Z, all executed in-session):
(1) 13:02Z blog improvements → **Ideas refactor DONE** (22 per-idea
pages + hot/ice index at the old path; details audit repaired 2
git-history corruptions — the lost `## 5` heading, #9's consumed
bullet — and refreshed 4 stale pages) + **Now-archive sorted**
most-recent-first (archive_now.py now rebuilds sorted every roll);
(2) 13:05Z codify + tooling → charter §5 permanent rules (ideas
structure + same-session index maintenance; sorted archive) +
`driver-background-task-guard` queued; (3) 13:10Z SigLIP q →
answered in-channel (frozen, no --backbone-vision-lr; VLM4VLA
vision-unfreeze rung noted); (4) 13:26Z naming → two-word tags
landed (`noise-draws` … `async-staleness`); (5) 13:58Z **async
checkpoint saves → queued HIGH** (`async-checkpoint-saves`, molmo2
measures ~14% wall in saves; target: lands before the attach-screen
launch). Owner 👍 "Great stuff" 13:35Z.

**Done**: this session — merge chain complete (baseline → redo →
merge `85cdc0a` → post-merge reruns → leaderboard measured-⏱
columns + AR draws10_t1 row 5 + main-sync review post filled with
both speedup tables → blog + Space + report JSONs live); Ideas
refactor + tags + archive sort (`4f18582`, `b6b5ff0`); charter
codification (`bd1aea8`); tsens q4 launched + babysit entry
activated + draws10_t1 entry pruned; queue: 5 items closed, 2 added
(driver guard, async ckpt HIGH), tsens live item added.

**Next**: `queue_cli.py next` → **async-checkpoint-saves** (owner
HIGH, CPU, target before the attach screen). Boundaries: tsens rungs
roll (repoint babysit log stem; reads via `tsens_dt_results.py` at
completion, record-only); molmo2 endpoint ~08-08 morning → #19 box
obligations → K smoke ladder → attachment steer window.

















*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 12:56–13:1xZ (real `date -u`) — tick (babysit +
incident): **the 12:30Z chained work session ended prematurely at
12:56Z** (26 min into its 4-h budget) — post-mortemed to a measured
verdict, its in-flight artifacts inherited, the decode microbench it
took down **relaunched detached 12:59Z**; molmo2 green;
`run_work_next` RE-ARMED.*

**Status** (babysit 12:57Z; molmo2 green; draws10_t1 liveness fail =
the retained-entry signature, expected):
- **Work-session post-mortem** (`20260807T123009Z_work.log`): ended
  with `terminal_reason: completed` — its final turn said "Waiting on
  bench notifications now — next action fires on the completion
  event". The driver treats a completed turn as session end; no
  notification re-invoke exists, and the harness killed its 3
  background tasks at 12:56:07Z, **taking down the decode microbench
  mid-run 5/14** (a child of a session bash task, not
  process-detached). New footgun — memory file
  `no-end-turn-waiting-on-notifications` written: sleep-poll in
  foreground, setsid-detach GPU jobs.
- **Decode microbench**: 4/14 banked pre-merge (ar_greedy,
  ar_draws10_t1, teacher_heun30_draws1, teacher_heun30_draws10 — all
  batched; JSONs in `reports/`); run 5 (student_1nfe_draws1 batched)
  killed mid-run. **RELAUNCHED 12:59Z** `setsid nohup` (survives
  session end): remaining 3 batched then all 7 single, sequentially,
  same pre-reg harness →
  `~/leaderboard_decode_microbench_20260807_resume.log`. Verified
  live 13:00Z (backbone loaded, sampling-frames phase). Still
  **pre-merge code** — the sequential-baseline sequencing the owner
  👍'd is intact.
- **Merge origin/main: deliberately NOT done this tick** — the
  baseline is still accruing in this working tree; merging mid-bench
  would contaminate the remaining pre-merge runs. It stays item 1 of
  the re-armed work session, gated on bench completion.
- **Inherited work-session artifacts, reviewed**: (a) md committed —
  ideas.md **#22 async staleness bridging** (parked, waits on #16),
  papers page RTC 2506.07339 + async-methods 2605.08168,
  main-sync-review post **DRAFT** (contains
  `PLACEHOLDER_RESULTS_TABLE` and anticipatory merge language — **do
  NOT blog-build until filled post-merge**); (b) test changes left
  uncommitted ON PURPOSE: `test_batched_draws.py` imports
  `tile_memory`/`tile_stats` which land only with the merge — pytest
  collects it from disk, so check.py fails until then; the
  chunked-backward tolerance adjudication (1e-5 → **5e-4**,
  cross-hardware calibrated, guarded failure mode ≫1e-2 so still
  sharp) and the **GIT_* scrub fix** (real incident: a linked-worktree
  pre-commit hook exports absolute GIT_DIR → a test's throwaway `git
  init` re-initialized the real repo; both harness tests now scrub
  GIT_*) commit together with/after the merge.
- box molmo2 AR 40k — 19280/40k, loss 3.1669, 2.202 s/step, vram
  67.07 ≤ 71, window 34.7 steps/min. Probes 6.49@18000 →
  6.44@18500 → **7.37@19000** (bouncy again; single reading above
  the band, no ≥7.5 pair — watch rule NOT tripped, next read at
  19500). Gate margin 4.72. ~12.7 h stepping + saves → endpoint
  ~08-08 morning.

**Steering**: none new (`read` empty). `history -n 5`: owner
12:29:47Z "deeply review, feel free to modify" was acked 12:30:50Z
and executed by the work session (the review IS the inherited
artifact set above); **👍×1 on the boundary post and 👍×1 on the
sequencing ack** — both recorded, plans unchanged.

**Done**: tick — babysit (molmo2 green; draws10_t1 fail adjudicated
as the expected retained-entry signature); work-session post-mortem
to a measured verdict; microbench relaunched detached + verified
live; inherited md artifacts committed, test changes documented as
merge-gated; memory file written; `queue_cli.py validate` green
(depth 2, 12 open); **`run_work_next` RE-TOUCHED**. 11:48 + 11:37
tick entries rolled to archive.

**Next**: chained work session (4-h budget), in order: (1)
**sleep-poll the bench to completion in foreground** (never
end-turn-waiting — see footgun), (2) **merge origin/main** per the
12:26Z steering (tolerance adjudication already staged in tests;
commit the test changes with the merge), (3) post-merge draws-config
rerun → leaderboard ⏱ rows incl. the batched-vs-sequential delta,
fill the draft post's placeholder → blog build + ledger, (4) **tsens
q4 launch** (prune the draws10_t1 registry entry only AFTER —
started_utc footgun). molmo2 endpoint ~08-08 → #19 box obligations →
K smoke ladder → attachment steer window.

*Updated 2026-08-07 12:21–12:3xZ (real `date -u`) — tick (babysit →
boundary): **draws10_t1 COMPLETED at its boundary — frozen reads run
in-tick: ALL PRE-REG EXPECTATIONS MET, falsifier NOT tripped**; decode
microbench launched 12:26Z on the freed GPU; molmo2 green with the
18000 watch point **CLEARED** (probe new low). `run_work_next` touched
→ work session chains (microbench reads + leaderboard rows + tsens
launch).*

**Status** (babysit 12:21Z; exit 1 = draws10_t1 liveness fail = the
expected completion signature, verified on-disk):
- local draws10_t1 — **DONE ~12:1x–12:2xZ**: 25,800 frames scored,
  reports/html/npz written, clean final table, process gone, GPU 0
  freed. Cumulative 33.8 f/min → **~12.7 GPU-h, inside the 24 GPU-h
  gate by ~2×** — the q4-fallback question stays closed. **Frozen
  reads** (`draws10_t1_results.py` →
  `reports/analysis__draws10_t1_ar100k_k4l2.json`): **E1 MET** Δ_AR
  (draws10 − greedy) = **−0.14505**, CI95 [−0.182, −0.109], excludes
  zero; **E2 MET** |Δ_AR| ≪ flow draws gain 1.258 (~9× smaller — the
  pre-registered mean-collapse shape: greedy AR decode already sits
  near the predictive mean); **E3 MET** draws10_t1 5.6515 does not
  overtake the flow draws10 band 5.365; **falsifier (Δ_AR > +0.1) NOT
  tripped**; oracles clean (row pairing full byte-match, T=1.0,
  draws=10, both report arms reproduced |d| < 5e-3). Babysit registry
  entry RETAINED on purpose (started_utc footgun — prune only AFTER
  the tsens launch); babysit reports liveness fail on it until then —
  expected, not an alarm.
- **decode microbench LAUNCHED 12:26Z** detached
  (`leaderboard_decode_microbench.py` full pass →
  `/home/ubuntu/leaderboard_decode_microbench_20260807.log`): pre-reg
  7 configs × {batched b32/w20, batch=1 single-stream}, ≤1.5 GPU-h,
  30-min/run watchdog. The chained session reads it and writes the
  leaderboard ⏱ rows.
- box molmo2 AR 40k — 18320/40k, loss 3.221, 2.192 s/step, vram
  67.07 ≤ 71, window 27.5 steps/min. **18000 watch point CLEARED:
  probe 6.49@18000 — new low** (7.53@17000 → 7.41@17500 → 6.49): the
  descending envelope resumed, watch item closed. Gate margin 5.60.
  ~13.2 h stepping + 9 saves (~15.5 min each) → endpoint ~08-08
  morning.

**Steering**: **NEW — owner 12:26:40Z** (caught on the end-of-tick
poll, acknowledged in-channel 12:4xZ): merge the missing main changes
into fontaine — main was **rebased onto our snapshot `42a202a`**
(our work through mem-snapshot/vram-peaks is now mainline) + 3
commits on top; read `docs/notes/2026-08-06-main-sync-for-fontaine.md`
first (done, from origin/main). Contents: (1) `2ee2be5` batched
noise-draw ensembling — `sample_draws` via one solver call at
draws×B, **5.6× bf16** (576 ms mean-of-10 on the rig), draws-major so
`collapse_draws`/`--dump-draws` layouts stay byte-compatible; fp32
seq-vs-batched max Δ 9.2e-5°; (2) `36570c0` `--return-home` cosine
glide via our `rollout_safety.home_trajectory`; (3) known:
`test_chunked_backward` aux rel-err 1.0004e-4 vs 1e-4 — OUR tolerance
call (passes on this box; pin down before touching the bound); (4)
`bijou/train.py` import reorder only. **Sequencing** (posted): the
in-flight microbench finishes pre-merge as the sequential baseline
(matches the banked evals the ≈ rows measured) → merge origin/main
(normal merge, not ff) → rerun the draws configs post-merge → the
batched-vs-sequential speedup lands on the leaderboard as a measured
delta. Merge = FIRST item of the chained work session. **Owner 👍 on
the ack post** (seen 12:4xZ) — sequencing plan agreed, no further
reply needed.

**Done**: tick — boundary adjudicated (completion verified on-disk,
never off the liveness line alone); frozen reads executed in-tick and
posted (Discord 12:2xZ, id …398); microbench launched on the freed
GPU; **`run_work_next` TOUCHED** → chained work session: microbench
reads → leaderboard/ledger rows → blog build → tsens q4 launch →
prune the draws10_t1 registry entry. Inherited and committed the
12:1x session's staged babysit.toml completion note (that session
evidently hit its hard kill before committing — the staged note was
its only surviving artifact). `queue_cli.py validate` green (depth 2,
12 open). 11:26Z tick entry rolled to archive. No blog build (reader
content lands with the leaderboard rows in the chained session).

**Next**: chained work session (4-h budget), in order: (1) **merge
origin/main per the owner's 12:26Z steering + sync note** (after the
in-flight microbench completes its pre-merge sequential baseline;
adjudicate the test_chunked_backward tolerance call); (2) microbench
reads + post-merge draws-config rerun → leaderboard ⏱ rows incl. the
batched-draws speedup delta + ledger/blog; (3) **tsens q4 launch**
(`eval_ar100k_tsens_q4_draws10.sh`, prune draws10_t1 entry AFTER —
the started_utc footgun); molmo2 endpoint ~08-08 → #19 box
obligations → K smoke ladder → attachment steer window.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 accruing from 15:01Z, ≤12 GPU-h gate). Older dated
snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

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
