# Now

















*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 11:15–11:2xZ (real `date -u`) — tick (babysit):
both runs green, no new steering; same picture as 11:04Z — the only
queued items stay boundary-blocked → normal exit, no work session
chained. Boundary now projects **~12:2xZ** (~1.1 h).*

**Status** (babysit 11:16Z, both green, exit 0):
- box molmo2 AR 40k — 16980/40k, loss 3.2421, 2.198 s/step, vram
  67.07 ≤ 71, probe low **6.64@16000** (latest 6.81@16500, gate
  margin 5.29; 6.6–6.9 oscillation band, normal); ~14.1 h + save
  pauses → endpoint ~08-08.
- local draws10_t1 — 23552/25800, window 29.2 f/min (content
  churn — judge on cumulative), cumulative 33.7 f/min → ~12.8 h
  total, **INSIDE the 24 GPU-h gate**; **~1.1 h to boundary
  (~12:2xZ)** → frozen reads + decode microbench + leaderboard
  rows.

**Steering**: none new (`read` empty; `history -n 5` shows only our
own 10:24–10:52Z posts, no reactions; owner last at 10:04–10:1xZ —
the leaderboard steering, fully executed).

**Done**: tick — babysit both green, exit 0; `queue_cli.py
validate` green (depth 2, 12 open). **No `run_work_next`**
(unchanged from 10:54Z/11:04Z): the queued microbench GPU run waits
on the draws10_t1 boundary and the F-then-joint pre-reg draft opens
after the seam-screen reads (~08-09+) — the boundary tick chains
the work session; never invent work to look busy. 09:49–10:3xZ
work-session entry rolled to archive. No Discord post (10:52Z post
is current), no blog build (no reader-visible change).

**Next**: draws10_t1 boundary ~12:2xZ → frozen reads
(`draws10_t1_results.py`) + decode microbench + leaderboard rows
(that tick arms the chained session); endpoint ~08-08 → #19 box
obligations → K smoke ladder → attachment steer window.

*Updated 2026-08-07 11:04–11:1xZ (real `date -u`) — tick (babysit):
both runs green, no new steering; same picture as 10:54Z — the only
queued items stay boundary-blocked → normal exit, no work session
chained. Boundary now projects **~12:2x–12:3xZ** (~1.3 h).*

**Status** (babysit 11:05Z, both green, exit 0):
- box molmo2 AR 40k — 16680/40k, loss 3.3092, 2.162 s/step, vram
  67.07 ≤ 71, probe low **6.64@16000** (latest 6.81@16500, gate
  margin 5.29; 6.6–6.9 oscillation band, normal); ~14.0 h + save
  pauses → endpoint ~08-08.
- local draws10_t1 — 23232/25800, window 29.3 f/min (content
  churn — judge on cumulative), cumulative 33.8 f/min → ~12.7 h
  total, **INSIDE the 24 GPU-h gate**; **~1.3 h to boundary
  (~12:2x–12:3xZ)** → frozen reads + decode microbench +
  leaderboard rows.

**Steering**: none new (`read` empty; `history -n 5` shows only our
own 10:24–10:52Z posts, no reactions; owner last at 10:04–10:1xZ —
the leaderboard steering, fully executed).

**Done**: tick — babysit both green, exit 0; `queue_cli.py
validate` green (depth 2, 12 open). **No `run_work_next`**
(unchanged from 10:54Z): the queued microbench GPU run waits on the
draws10_t1 boundary and the F-then-joint pre-reg draft opens after
the seam-screen reads (~08-09+) — the boundary tick chains the work
session; never invent work to look busy. No Discord post (10:52Z
post is current), no blog build (no reader-visible change).

**Next**: draws10_t1 boundary ~12:2x–12:3xZ → frozen reads
(`draws10_t1_results.py`) + decode microbench + leaderboard rows
(that tick arms the chained session); endpoint ~08-08 → #19 box
obligations → K smoke ladder → attachment steer window.

*Updated 2026-08-07 10:54–11:0xZ (real `date -u`) — tick (babysit):
both runs green, no new steering; no actionable CPU items this
window (both boundary-blocked) → normal exit, no work session
chained.*

**Status** (babysit 10:54Z, both green, exit 0):
- box molmo2 AR 40k — 16400/40k, loss 3.2713, 2.167 s/step, vram
  67.07 ≤ 71, probe **new low 6.64@16000** (gate margin 5.45);
  ~14.2 h + save pauses → endpoint ~08-08.
- local draws10_t1 — 22912/25800, window 107.7 f/min (content
  churn — judge on cumulative), cumulative 33.9 f/min → ~12.7 h
  total, **INSIDE the 24 GPU-h gate**; **~1.4 h to boundary
  (~12:2x–12:3xZ)** → frozen reads + decode microbench +
  leaderboard rows.

**Steering**: none new (`read` empty; `history -n 5` shows only our
own posts, no reactions; owner last at 10:04–10:1xZ — the
leaderboard steering, fully executed last session).

**Done**: tick — babysit both green, exit 0; `queue_cli.py
validate` green (depth 2, 12 open). Bookkeeping: the chained
10:1x–10:5xZ work session (endpoint-runbook git-audit CLEAN,
microbench prep, APT + siblings lit slices — commits
`ea8cfa9`/`49cbec4`/`6b2afaf`) had no now.md note; its footer
session note added below. **No `run_work_next`**: the only queued
CPU item (F-then-joint pre-reg draft) opens after the seam-screen
reads (~08-09+), and the microbench GPU run waits on the
draws10_t1 boundary — the boundary tick chains the work session;
never invent work to look busy. No Discord post (10:52Z post is
current), no blog build (no reader-visible change).

**Next**: draws10_t1 boundary ~12:2x–12:3xZ → frozen reads
(`draws10_t1_results.py`) + decode microbench + leaderboard rows
(that tick arms the chained session); endpoint ~08-08 → #19 box
obligations → K smoke ladder → attachment steer window.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

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
