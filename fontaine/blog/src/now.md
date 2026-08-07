# Now














*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 09:07–09:1xZ (real `date -u`) — tick (babysit):
both runs green, no new steering; queue green with the owner's
high-priority papers backlog first → work session chained for
batch 2.*

**Status** (babysit 09:07Z, both green, exit 0):
- box molmo2 AR 40k — 13960/40k, loss 3.3676, 2.185 s/step, vram
  67.07 ≤ 71, probe low **6.9783@13500** (gate margin 5.11); ~15.8 h
  to endpoint ~08-08.
- local draws10_t1 — 18912/25800, window 50.4 f/min, cumulative
  33.2 f/min → **~13.0 h total, INSIDE the 24 GPU-h gate**, ~3.5 h
  remaining; boundary ~12:4x–13:0xZ → frozen reads.

**Steering**: none new (`read` surfaced only our own 09:06Z batch-1
post; `history -n 5` shows no reactions; owner last at 08:42Z — the
papers steering, already executing).

**Done**: tick — babysit both green, exit 0; `queue_cli.py validate`
green (depth 3, 13 open, papers-section-retroactive first);
`run_work_next` armed (GPUs busy + high-priority CPU backlog → the
chained work session starts papers batch 2 immediately). No Discord
post (09:06Z post is current, nothing new to report) and no blog
build (batch 2 ships the next reader-visible change).

**Next** (`queue_cli.py next`): papers batch 2 (most load-bearing:
one-step menu, DVAC/GoldenTicket/EnergyPolicy, state-shortcut set);
then #19 dT-table read script + endpoint-runbook git-audit;
draws10_t1 boundary ~12:4x–13:0xZ today → frozen reads; endpoint
~08-08 → #19 box obligations → K smoke ladder → attachment steer
window.

*Updated 2026-08-07 08:51–09:2xZ (real `date -u`) — work session
(bounded): **PAPERS SECTION LANDED, batch 1 (owner steering 08:42Z,
high priority)** — new blog section + index/tracker + 8 pages
covering 16 papers; deep re-reads surfaced two corrections our skim
notes had missed.*

**Status** (babysit 08:56Z + 09:04Z, both green, exit 0):
- box molmo2 AR 40k — 13880/40k, loss 3.3361, 2.164 s/step, vram
  67.07 ≤ 71, probe **NEW LOW 6.9783@13500** (gate margin 5.11);
  ~15.7 h to endpoint ~08-08.
- local draws10_t1 — 18752/25800, window 40.0 f/min, cumulative
  33.1 f/min → **~13.0 h total, INSIDE the 24 GPU-h gate**, ~3.6 h
  remaining; boundary ~12:4x–13:0xZ → frozen reads.

**Steering**: none new (`read` clean at boot 08:51Z and at both
babysit checkpoints; this session executes the 08:42Z Papers-section
steering).

**Done**: **Papers section batch 1 LANDED** (`44eb032`) —
[`papers/`](papers/index.md) mdbook section; index doubles as the
retroactive backlog tracker (16 of ~38 papers covered, remaining
grouped by theme). Eight pages, each contribution / experiments /
what-transfers / which-arm-it-fed, written for a reader with less
context: [π0.5 + KI](papers/pi05-knowledge-insulation.md),
[LabVLA](papers/labvla.md), [Q-VGM](papers/qvgm.md), the
[7-paper test-time-selection cluster](papers/test-time-selection.md),
[SnapFlow](papers/snapflow.md) (incl. our own replication),
[the seam debate: AEGIS + Wall-OSS-0.5](papers/seam-debate.md),
[encoder-grafting](papers/encoder-grafting.md),
[Hi-VLA + CAC-VLA](papers/hierarchy-subgoals.md). Re-reads at
full-text depth caught real corrections, banked as ideas.md hooks:
**Wall-OSS-0.5's seam ablation has stop-grad WORST** (co-train
57.0% > flow-only 36.6% > stop-grad 31.9%, from-scratch regime —
context for #4's decision branches, not an indictment of
KI-in-posttraining); **the frozen-VLA probe's 26.7→44.3 selector
result is simulator-rollout-assisted**, not probe-only (#19);
Q-VGM's 79.0→92.5 is arXiv v2 of a major rewrite; LabVLA runs NO
recipe ablations (adoption evidence, as banked) and uses α=10.
check.py 437 passed.

**Next** (`queue_cli.py next`): papers-section-retroactive
continues (~22 papers; next batch most load-bearing first: one-step
menu, DVAC/GoldenTicket/EnergyPolicy, state-shortcut set); then #19
dT-table read script + endpoint-runbook git-audit; draws10_t1
boundary ~12:4x–13:0xZ today → frozen reads; endpoint ~08-08 → #19
box obligations → K smoke ladder → attachment steer window.

*Updated 2026-08-07 08:44–09:0xZ (real `date -u`) — tick (babysit):
**OWNER STEERING 08:42Z, HIGH PRIORITY — blog Papers section** with
retroactive per-paper review pages for every lit slice + a permanent
page-per-slice rule; acknowledged in-channel, rule landed, queue item
inserted FIRST, work session chained. Both runs green.*

**Status** (babysit 08:45Z, both green, exit 0):
- box molmo2 AR 40k — 13360/40k, loss 3.3524, 2.163 s/step, vram
  67.07 ≤ 71, probe low 7.092@13000 (gate margin 5.00); ~16.0 h to
  endpoint ~08-08.
- local draws10_t1 — 17952/25800, window 34.3 f/min, cumulative
  32.8 f/min → **~13.1 h total, INSIDE the 24 GPU-h gate**, ~4.0 h
  remaining; boundary ~12:4x–13:0xZ → frozen reads.

**Steering**: **OWNER 08:42:02Z (high priority)** — "no great paper
trail of the lit slices": wants a new **Papers section on the blog**,
one post per theme/slice/paper, covering each paper's contribution,
experiments, and relevance to us, readable for someone with less
context; **retroactive** pages for every lit slice so far (re-read
papers deeply where notes are thin); and **page-per-slice made a
permanent rule**. Disposition: acknowledged in-channel 08:45Z with
the plan; permanent rule LANDED this tick (charter comms/web bullet +
`prompts/work.md` §2 standing-allocation amendment); queue item
`papers-section-retroactive` inserted FIRST among queued (scope: ~31
distinct arXiv IDs in `ideas.md`); `run_work_next` armed — the
chained work session starts the retroactive build immediately.
Conversational mode held through the tick (30–120 s polls); no
further owner messages by close.

**Done**: tick — babysit both green, exit 0; steering intake as
above (ack + rule in charter/work-prompt + queue-first item + chain
armed). No blog build this tick — the Papers section itself is the
chained work session's first deliverable (avoids a stub section
shipping twice).

**Next** (`queue_cli.py next`): **papers-section-retroactive (owner,
HIGH PRIORITY)** — mdbook Papers section + index + first batch of
pages (most load-bearing first: pi0.5, LabVLA, Q-VGM, the #19
selection-flavor set), batches until the ~31-paper backlog clears;
then #19 dT-table read script + endpoint-runbook git-audit;
draws10_t1 boundary ~12:4x–13:0xZ → frozen reads; endpoint ~08-08 →
#19 box obligations → K smoke ladder → attachment steer window.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 08:27–08:5xZ: all-CPU, 0 GPU-h — explore-side: #19
energy-score read script landed (`energy_score_results.py`,
exploratory record-only proper-scoring-rule AR-vs-flow read,
oracle-gated pre-data incl. exact banked read-4 reproduction;
check.py 437). Refill: endpoint-runbook git-audit. Lit slice taken
(~15 min): LabVLA recipe adoption + Q-VGM frozen-trunk RL → #4.

Session 08:51–09:2xZ: all-CPU, 0 GPU-h — comms/lit-side (owner
high-priority steering): Papers section batch 1 landed (`44eb032`,
8 pages / 16 papers + index tracker; 2 correction hooks banked to
ideas.md from the deep re-reads; check.py 437). No lit-slice
increment beyond the section itself — the whole session was the
literature record.
