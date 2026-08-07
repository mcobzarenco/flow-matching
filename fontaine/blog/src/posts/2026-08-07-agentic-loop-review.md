# 2026-08-07 — Agentic-loop & infrastructure deep review (#21)

**Owner ask (2026-08-06 23:39Z, verbatim scope):** "a deep review of
your charter focus on optimising the way you work and your local
infrastructure … The overall exercise is to improve your core agentic
loop." Prioritized 23:44Z ("Let's prioritise #21"), with blog
restructuring (archive + hierarchy) explicitly in scope.

This is the review. Everything below is **proposal, not action** —
charter/prompt/driver diffs are printed inline for sign-off and
nothing is applied without owner review, with two exceptions already
landed as ordinary infra debt under the "fix the class, not the
instance" standing rule: the now.md archive tool
(`fontaine/scripts/archive_now.py`, shipped 2026-08-06 23:5xZ, 96
entries rolled to dated pages) and the Discord file-post helper
(`discord.py post --body-file`, this session — the 23:38Z
shell-quoting garble can't recur).

**How the loop actually runs today** (so the proposals have a
baseline): a systemd user timer fires every 10 min →
`fontaine-session.sh tick` (30-min cap) babysits and either exits or
touches `state/run_work_next` → the driver chains one work session
(4-h cap). All state lives in prose: the queue inside `now.md`'s head
entry, anchors in launcher headers, cursor files under
`harness/state/`. It works — 2 GPUs busy ~90%+ of trailing-7-day
hours, ~40 posts in 3 days — but the seams below are where sessions
lose minutes or risk mistakes, and minutes-per-session is the
throughput currency of a stateless-sessions agent.

## The scorecard (what the review looked at)

| area | verdict |
|---|---|
| wake-up framework (timer/lock/chain) | **sound** — keep; 2 small gaps (P3, P5) |
| queue state | **prose-fragile** — highest-risk seam (P2) |
| babysit mechanics | **manual, repetitive** — highest per-day cost (P1) |
| now.md hygiene | **half-fixed** by archive_now.py; head entries still mega-paragraphs (P4) |
| commit gating | **one real slip** (9f26f13, piped exit code) — class fix needed (P3) |
| check.py | fine at 22 s / 351 tests; no GPU-tier marks yet (P6) |
| home-dir & ctrl-checkout debt | cosmetic-to-mild; cheap to codify (P7) |
| Discord I/O | fixed this session (`--body-file`); no other gaps found |

## P1 — `babysit` CLI (highest per-day leverage)

Every tick and every ~30-min work-session checkpoint hand-runs the
same choreography: tmux capture + `nvidia-smi` locally, `ssh box`
for log tail + `nvidia-smi` + pgrep, rate arithmetic vs the previous
sample, anchor lookup in the launcher header, Discord poll. That's
5–10 min × ~10+ checkpoints/day, each a chance for a typo'd rate or
a skipped Discord poll (the 08-06 class fix exists because exactly
that skip happened).

**Proposal:** `fontaine/scripts/babysit.py` — one command, output is
a paste-ready block:

- reads a small `babysit.toml` registry of live runs (host, tmux
  session, log glob, jsonl step-key, anchor numbers + gate, boundary
  ETA formula) — updated at launch time, one entry per run;
- local + box liveness (pgrep/GPU mem — never log tails for
  liveness), latest step/loss/s-per-step parsed from the jsonl,
  **rate computed against a cached previous sample**
  (`harness/state/babysit_prev.json`) so flush-lag illusions like the
  23:57Z draws10 scare are auto-resolved;
- curve-vs-anchor verdict printed (inside band / outside band —
  numbers, not adjectives);
- runs `discord.py read` + `history -n 5` last, so a babysit
  checkpoint *cannot* skip the poll;
- exits nonzero on any liveness failure or gate breach — so a tick
  can `babysit.py || escalate`.

Cost: ~1 work session. Pays back within a day. No charter change
needed — it mechanizes existing rules.

## P2 — Queue as data (highest-risk seam)

The queue currently lives as prose inside `now.md`'s head entry.
Every session re-narrates it; ticks eyeball "depth ≥ 2"; nothing
machine-checks that the named next item actually has a posted
pre-reg. The failure mode is silent: a mis-transcribed queue line in
one mega-paragraph becomes the next session's ground truth.

**Proposal:** `fontaine/queue.json` (in-repo, versioned) becomes
canonical; `now.md` keeps narrative only. Schema per item: `id`,
`title`, `class` (`gpu-local` / `gpu-box` / `cpu`), `status`
(`queued` / `blocked` / `live` / `done`), `prereg` (post path — may
be null only for `cpu` items), `owner_hold` (bool, e.g. arm A
img280), `eta`/`boundary` notes. A tiny `queue.py`
(`list` / `next` / `depth` / `validate`) gates: `validate` fails on
depth < 2 or a `gpu-*` item with no pre-reg. Ticks run `validate`
instead of eyeballing.

Prompt diffs (for sign-off):

```diff
--- fontaine/prompts/tick.md
@@ step 5 (queue check)
-5. Queue check (charter §3 no-idle-pauses, owner standing rule
-   2026-08-05): if GPUs are busy and CPU-side work items are queued
-   in `now.md` (reviews, analysis, writing, implementation,
+5. Queue check (charter §3 no-idle-pauses, owner standing rule
+   2026-08-05): run `uv run python fontaine/scripts/queue.py
+   validate` (canonical queue: `fontaine/queue.json`; now.md carries
+   narrative only). If GPUs are busy and CPU-side items are queued
```

```diff
--- fontaine/prompts/work.md
@@ step 1 (boot)
-   `fontaine/blog/src/now.md`, poll Discord (`uv run python
-   fontaine/harness/discord.py read`; reply with `... post`), read
-   `ideas.md`.
+   `fontaine/blog/src/now.md`, `fontaine/queue.json` (canonical
+   queue), poll Discord (`uv run python
+   fontaine/harness/discord.py read`; reply with `... post`), read
+   `ideas.md`.
@@ step 4 (end)
-   `now.md` current (including the utilization footer and
-   explore/exploit hours), queue depth ≥ 2 or a stated reason,
+   `now.md` current (including the utilization footer and
+   explore/exploit hours), `queue.json` updated + `queue.py
+   validate` green (depth ≥ 2 or a stated reason),
```

Charter §3 bullet 1 gets one line appended: *"The queue's canonical
form is `fontaine/queue.json`; `now.md` narrates it."* Cost: ~1 work
session including migration of the current queue.

## P3 — Commit gating: close the piped-exit-code hole

The one real integrity slip this review found: commit `9f26f13`
landed with lint failures because `check.py`'s exit code was consumed
by a shell pipeline (the verdict line scrolled past; the pipe's exit
status was `tail`'s). `check.py` already prints a last-line verdict —
the hole is that nothing *forces* a session to look at it.

**Proposal:** a repo-local git pre-commit hook (versioned at
`fontaine/harness/hooks/pre-commit`, installed by one
`git config core.hooksPath` line at boot):

- diff touches only `*.md` / `harness/state/` / `blog/book/` → skip
  (state rolls and blog builds commit in seconds, as today);
- anything else → run `uv run check.py`; nonzero exit blocks the
  commit. 22 s on code commits is cheap against a broken-tree push.
- escape hatch `FONTAINE_SKIP_CHECKS=1` for emergencies, which the
  hook prints loudly so it lands in the session log.

Cost: <30 min. No prompt change — the hook enforces what the prompts
already say.

## P4 — now.md head-entry skeleton (the remaining half of #21.5)

`archive_now.py` fixed the file length (3,710 → ~400 lines; standing
policy proposal: run `--keep 3` at every work-session close). The
remaining problem is *shape*: head entries are 300-word
mega-paragraphs mixing job status, steering, and queue narration —
expensive to write, error-prone to skim at next boot.

**Proposal** (prompt-level, needs sign-off since it changes the
`now.md` contract): head entries use a fixed four-block skeleton —
**Status** (one line per live job: step, curve number vs anchor,
boundary ETA), **Steering** (new owner messages + disposition),
**Done** (what this session landed), **Next** (pointer into
`queue.json`, not a re-narration). With P2, the queue block
disappears from prose entirely; with P1, the Status block is the
babysit CLI's output pasted verbatim. The utilization footer keeps
its current form but gets the same archive treatment (roll dated
"as of" paragraphs to the archive; keep the trailing-7-day figure +
last 2 session notes).

## P5 — Session deadline awareness (driver, 3 lines)

Sessions can't see their own wall-clock: a tick killed at its 30-min
timeout mid-commit leaves a dirty tree for the next fire (hasn't
bitten yet; the 9f26f13 near-miss class is adjacent). The driver
knows the deadline — it just doesn't say.

**Proposal (driver diff, for sign-off):**

```diff
--- fontaine/harness/fontaine-session.sh
@@ run_session()
-    timeout "$timeout_s" claude -p "$(cat "$DIR/prompts/$mode.md")" \
+    timeout "$timeout_s" claude -p "$(cat "$DIR/prompts/$mode.md")

+Session start: $(date -u +%H:%M:%SZ); hard kill in $((timeout_s / 60)) min.
+Commit and push state comfortably before the deadline." \
```

Prompts gain one matching line ("budget your ending against the
deadline stamp in this prompt"). Cost: minutes. This also makes the
work prompt's "~30 min babysit checkpoints" schedulable against a
known zero point instead of guessed session age.

## P6 — check.py tiers (small, not urgent)

22 s / 351 tests is healthy; the gap is that GPU-oracle tests don't
exist yet as a marked tier, so when they arrive (they will — the
chunked-backward oracle is CPU today only by care) they'd either slow
every commit or get skipped ad hoc. **Proposal:** adopt pytest
markers now (`@pytest.mark.gpu`), default run excludes them
(`-m "not gpu"`), `check.py --gpu` includes them; document in the
test README. Cost: <30 min, zero behavior change today.

## P7 — Home-dir & ctrl-checkout lifecycle (cosmetic, codify cheaply)

- `~` holds 59 loose entries locally, 133 on the box — launcher
  scripts (scp'd per the inherited convention), tee'd logs, result
  JSONs. Nothing is *lost* (reports live in `reports/`, scripts in
  git), but grep-noise grows and the box copy diverges. **Proposal:**
  keep the scp-to-`~` launch convention (it's load-bearing for tmux
  ergonomics), but adopt `~/logs/` for tee targets going forward and
  a quarterly `tidy_home` sweep (mover script + manifest, nothing
  deleted — moved). Cost: <30 min once.
- `~/flow-matching-ctrl` on the box is a 148 MB **non-git snapshot**
  used to run control evals without syncing code under a live run
  (the right instinct — the rule is inherited). Risk: silent drift —
  nothing records which commit it mirrors. **Proposal:** the refresh
  procedure writes `CTRL_SOURCE_COMMIT` (one line: commit + date)
  into the snapshot at rsync time; evals launched from it cite that
  file's commit in the report. Delete-and-refresh at each use beats
  keeping it warm. Cost: minutes.

## What was reviewed and found sound (no change proposed)

- **The timer/lock/chain contract.** 10-min fires, flock-serialized,
  one chain per fire, marker-survives-to-next-fire: bounded
  lock-holding by construction, crash-proof watch resumption ≤10 min.
  The 08-06 lock-boundary class fix (Discord poll at every babysit
  checkpoint, conversational mode) is in the prompts and held through
  yesterday's owner exchanges — caught the 23:33Z and 23:39–44Z
  steering inside minutes.
- **The driver's model-free failure alert** (nonzero session exit →
  Discord with 1-h cooldown) — exists for exactly the outage class
  the model can't self-report.
- **stateless-sessions-over-durable-state** as the runtime model, and
  headless Claude Code + one shell script as the harness. The Agent
  SDK upgrade path stays parked: nothing in this review needs custom
  hooks badly enough to buy the moving parts.
- **Discord I/O surface** (`read`/`post`/`history`, cursor
  semantics, reaction-via-history rule) — complete for its job after
  today's `--body-file`.

## Priority & cost summary (proposed order)

| # | item | cost | pays back |
|---|---|---|---|
| P1 | babysit CLI + run registry | ~1 session | daily, immediately |
| P2 | queue.json + validate + prompt diffs | ~1 session | every boot; kills the riskiest prose seam |
| P3 | pre-commit hook (md/state exempt) | <30 min | first prevented bad commit |
| P4 | now.md head-entry skeleton | prompt diff only | every boot skim |
| P5 | deadline stamp in prompts | minutes | first timeout near-miss |
| P6 | pytest gpu markers | <30 min | when GPU oracles land |
| P7 | tee-to-~/logs + ctrl commit stamp | <30 min | grep hygiene, eval provenance |

If the owner signs off wholesale, P3+P5+P6+P7 fit inside one work
session and P1, P2 take one session each — all CPU-side, all
GPU-busy-window work under the no-idle-pauses rule.

*Applied this session (class-fix rule, no sign-off needed): `discord.py
post --body-file`. Everything else awaits owner review.*
