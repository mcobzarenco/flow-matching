# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 17:33–18:0xZ (real `date -u`) — work session
(bounded, one item): **#17 molmo2 vision-unfreeze pre-reg DRAFT
posted** ([draft](posts/2026-08-07-prereg-molmo2-vision-unfreeze.md),
`3b6e0b8`) — the 17:04Z owner question's disposition, drafted while
the lit slice is fresh.*

**Status** (babysit 17:41Z):
- box molmo2 AR 40k — 25640/40k, loss 3.042, 2.193 s/step, vram
  67.07 ≤ 71. Probe 6.65@25500 (in-band, no ≥7.5 pair). Gate margin
  4.93. ~8.7 h to 40k → endpoint ~08-08 morning.
- local **ar100k_tsens_q4 rung t0.5** — 2912/4301, cumulative
  28.2 f/min, projection 2.5 ≤ 12 gate, ~0.8 h left on the rung.
  Rung roll t0.5 → t0.7 **~18:3xZ** (babysit `log` stem repoint at
  the first tick after); at the cumulative rate t0.7 ends ~21:0xZ,
  t1.3 ~23:3x–00Z → dT read opens late tonight (the 17:30 entry's
  "20–21Z" was optimistic; 3 × 2.5 h from 15:58 launch says ~00Z).

**Steering**: none (babysit-forced poll 17:41Z: no new messages;
`history -n 5`: our own posts + the answered 17:04Z question).

**Done**: this session — **#17 vision-unfreeze pre-reg DRAFT**
(`3b6e0b8`, loud DRAFT banner, execution blocked on finalization
amendment + owner go): one variable `--backbone-vision-lr 2e-6`
(0.1× text; full-FT tower per 2607.10172, never LoRA-on-SigLIP);
**primary = 10k screen** vs the banked baseline `step_010000`
checkpoint (both panel-eval'd with the 40k launcher's chained eval
verbatim; paired per-frame Δ CI95, null band 0.07 = seed-trio
spread; critical-frame re-pool robustness via the #16 instrument),
40k = escalation only (~110 GPU-h not spent before a ~27 GPU-h
screen). Memory ladder pre-registered (chunks 6→12 → decoder
activation-ckpt; matched downshift excluded — poisons the contrast;
~3–4 GiB tower adder on 67.07/71 makes the 150-step smoke
load-bearing). Declared blind spot: the panel can't see the MAPS
OOD tax. check.py 460 green. Queue: draft item done,
`idea17-molmo2-vision-unfreeze-execution` added (blocked,
owner_hold, post-attach-screen ~08-09+); validate green depth 2.

**Next**: `queue_cli.py next` → **idea19-tsens-dt-read-execution**
opens at rungs completion (~23:3x–00Z tonight; script landed,
record-only vs the decode-temperature page's written prior); then
`idea1-golden-ticket-prereg-draft` in GPU-busy windows. Dated
boundaries: tsens rung roll ~18:3xZ (babysit stem repoint t0.5 →
t0.7 at first tick after) → all rungs ~00Z → dT read; molmo2
endpoint ~08-08 morning → #19 box obligations → K smoke ladder →
attach-screen window. **Every GPU launch goes through
`run_detached.sh`.**

*Updated 2026-08-07 17:30–17:3xZ (real `date -u`) — tick (babysit):
both runs green, no steering. tsens window read 0.0 f/min again —
the known 160-frame flush quantization; adjudicated healthy per the
standing note (log mtime + cumulative), no live-watch needed this
time.*

**Status** (babysit 17:30Z):
- box molmo2 AR 40k — 25360/40k, loss 3.014, 2.199 s/step, vram
  67.07 ≤ 71, 28.6 steps/min window. Probe 7.10@25000 (in-band, no
  ≥7.5 pair). Gate margin 4.93. ~8.9 h to 40k → endpoint ~08-08
  morning.
- local **ar100k_tsens_q4 rung t0.5** — window 0.0 f/min over ~3 min
  (flush quantization, per the 16:53 note); log mtime 17:27:05Z
  (4 min old, inside the ~6-min flush cadence), latest line
  2592/4301, cumulative 28.1 f/min, projection 2.6 ≤ 12 gate,
  ~1.0 h remaining. Rung roll t0.5 → t0.7 **~18:3xZ** — babysit
  `log` stem repoint due at the first tick after; all rungs ~20-21Z
  → dT read.

**Steering**: none (`read`: only our own 17:30 work-session close;
`history -n 5`: no reactions).

**Done**: tick — babysit exit 0, both runs green; tsens 0.0-window
re-adjudicated healthy via log mtime + cumulative (standing note
applied, no escalation); `queue_cli.py validate` green (depth 3, 13
open); `run_work_next` already armed 17:30Z by the closing work
session — chained work session follows this tick. 16:37 work entry
rolled to archive. No Discord post (17:30 close current), no blog
build (no reader-visible change).

**Next**: chained work session → next CPU queue item (golden-ticket
/ vision-unfreeze pre-reg drafts); tsens rung roll ~18:3xZ (babysit
stem repoint t0.5 → t0.7 at the first tick after) → all rungs
~20-21Z → dT read against the decode-temperature page's written
prior (record-only); molmo2 endpoint ~08-08 morning → #19 box
obligations → K smoke ladder → attach-screen window (first save
validates async ckpt in production at 1250 cadence). **Every GPU
launch goes through `run_detached.sh`.**

*Updated 2026-08-07 16:57–18:1xZ (real `date -u`) — work session:
**#16 critical-frame re-pooling EXECUTED — every published ranking
holds** (pre-reg posted+committed before the read; `4773ba9` +
`3da7695`) + owner steering answered with a targeted lit slice
([vision-encoder-freeze](papers/vision-encoder-freeze.md),
`3ac7775`).*

**Status** (babysit 17:35Z):
- box molmo2 AR 40k — 25280/40k, loss 3.047, 2.191 s/step, vram
  67.07 ≤ 71. Probe 7.10@25000 (in-band vs the 5.97–7.18 recent
  band, no ≥7.5 pair). Gate margin 4.93. ~9.0 h to 40k → endpoint
  ~08-08 morning.
- local **ar100k_tsens_q4 rung t0.5** — 2592/4301 @ 49.7 f/min
  window, cumulative 29.0 f/min, projection 2.5 ≤ 12 gate. Rung
  roll t0.5 → t0.7 **~18:3xZ** (repoint the babysit `log` stem at
  the first tick after); all rungs ~20-21Z → dT read opens.

**Steering**: owner 17:04Z — "what evidence on unfreezing our
SigLIP encoder in molmo2, helpful or harmful?" Answered 17:2xZ from
the banked pages (VLM4VLA/APT/KI prior), then a targeted lit slice
found the missing pole and a correction was posted 17:5xZ: MAPS
(2511.19878) and the dual-encoder paper (2509.11417) are real
harm cases — in the OOD-retention regime, not ours. Net: both poles
real; our rung is adaptation-regime → unfreeze should help the
panel; recipe prior full-FT tower at low LR, never LoRA-on-SigLIP.
Disposition: `idea17-molmo2-vision-unfreeze-prereg-draft` queued
(draft CPU; execution post-attach-screen, owner-steered).

**Done**: this session —
(1) **idea16-critical-frame-repooling** (`4773ba9` pre-reg +
instrument BEFORE the read; `3da7695` results): the CI-MSE concern
tested on our own board at zero GPU cost. Frozen rule (chunk window
hits subgoal boundary | holding bracket | event), coverage 99.9%,
11,204 critical core frames. **All 10 pairwise gaps keep their
published sign with CI95 excluding 0**; separation vs state-copy
*widens* on critical frames (+6.18 → +6.74) — opposite of CI-MSE's
easy-frame-dilution mode. Robustness note on the leaderboard;
`critical_frame_repooling.py` (--selftest oracle) reusable at the
molmo2 endpoint. check.py 460 green ×3 commits.
(2) **Lit slice + papers page**
([vision-encoder-freeze](papers/vision-encoder-freeze.md), 4
sources, `3ac7775`) — see Steering; correction to the first reply
posted same session.
(3) Queue: idea16 done; `idea17-molmo2-vision-unfreeze-prereg-draft`
refilled; validate green depth 3.

**Next**: `queue_cli.py next` → **idea19-tsens-dt-read-execution**
opens at rungs completion (~20-21Z tonight; script landed,
record-only vs the decode-temperature page's written prior); then
golden-ticket + vision-unfreeze pre-reg drafts in GPU-busy windows.
Dated boundaries: tsens rung roll ~18:3xZ (babysit stem repoint
t0.5 → t0.7 at first tick after) → all rungs ~20-21Z → dT read;
molmo2 endpoint ~08-08 morning → #19 box obligations → K smoke
ladder → attach-screen window (first save validates async ckpt in
production at 1250 cadence). **Every GPU launch goes through
`run_detached.sh`.**

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

Session 17:33–18:0xZ: all-CPU bounded work session, 0 GPU-h new
(tsens + molmo2 accruing under their own gates) — queue-refill/
pre-reg: #17 vision-unfreeze pre-reg DRAFT posted (10k-screen
design vs baseline@10k, memory ladder, frozen reads incl.
critical-frame re-pool; execution owner-held post-attach-screen);
queue 1 done + 1 blocked execution item added, depth 2.

Session 16:57–18:1xZ: all-CPU work session, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — exploit/analysis +
owner-steered lit: #16 critical-frame re-pooling executed
(pre-reg → oracle-gated instrument → read; every published ranking
holds, separation widens on critical frames) + SigLIP-unfreeze
question answered with the vision-encoder-freeze papers page (both
poles + correction); queue 1 done, 1 refilled, depth 3.

Session 16:37–16:5xZ: all-CPU work session, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — exploit/infra + sanctioned
lit: attach-launch-save-cadence-prep landed (`c4555d4`, save-every
2500→1250 both arms + pre-reg amendment 2; pinned-buffer deferred
with stated arithmetic) + the offline-validation lit slice with
same-session papers page (5 sources; panel proxy measured ρ −0.61,
critical-frame re-pooling rung banked to #16); queue 1 done, 2
refilled, depth 3.
