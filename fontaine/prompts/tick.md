Fontaine tick session (babysit). Take as long or as little as the
situation needs (owner steering 2026-08-05). Repo
`~/flow-matching`, branch `fontaine`. The charter
(`fontaine/charter.md`) governs; its "Session boot" section is this
prompt's contract.

1. `git pull --ff-only`. Read `fontaine/blog/src/now.md`.
2. Poll Discord: `uv run python fontaine/harness/discord.py read`
   (cursor-managed; replies via `... discord.py post "text"`). Owner
   messages are steering: acknowledge in-channel, record in
   `now.md`, apply at the next decision point.
   **Conversational mode — a chat never waits for the next tick**:
   if the owner is mid-exchange (a question, a follow-up likely),
   reply and STAY — sleep-poll the channel at 30–120 s intervals
   while the exchange is live, stretching the interval as it
   quiets; hand back to normal cadence after ~10 min of silence. If
   the tick's 30-min cap approaches mid-conversation, touch
   `fontaine/harness/state/run_work_next` and end — the chained
   work session rejoins the thread (`discord.py history` rebuilds
   recent context without moving the cursor).
3. If a training run is live: liveness by pgrep/GPU memory (never a
   log tail); latest step; curve vs the pre-registered anchors in
   the launcher header; anomaly scan (loss spikes, OOM, substitution
   flood, utilization collapse). Healthy → update `now.md`, exit.
   Anomalous → diagnose the mechanism (charter §6 discipline); kill
   only at a save boundary and only per the pre-registered gates;
   escalate per charter §7 when warranted.
4. If the run finished or died: post-process per charter §4 (panel
   score, reports, blog post, ledger row, artifact uploads), then
   launch the next queued pre-registered run. If that exceeds this
   tick's 30-min cap: do the urgent part, `touch
   fontaine/harness/state/run_work_next`, and end the session — the
   driver chains straight into a work session with a 4-h budget.
5. Queue check (charter §3 no-idle-pauses, owner standing rule
   2026-08-05): if GPUs are busy and CPU-side work items are queued
   in `now.md` (reviews, analysis, writing, implementation,
   literature slice), touch the `run_work_next` marker (see 4) —
   GPU-busy windows are work-item windows, never idle waits. Same if
   the queue is below depth 2 (the chained work session refills it).
   Only when the GPUs are idle-by-design AND the CPU-side queue is
   empty does the tick simply exit — and never invents GPU work to
   look busy.
6. Judgment call — holding the session open: through a critical
   window (fresh launch's first steps, an approaching eval boundary,
   a kill decision pending) you MAY babysit in-session with sleep
   polls (`sleep 900` etc.; single commands may run up to 1 h) —
   context is preserved and overlapping timer fires skip harmlessly
   off the lock. Mind the mode's wall-clock cap; for stable
   stretches prefer exiting — the timer's fresh sessions are cheaper
   and crash-proof.
7. End: commit + push state (`now.md` always; blog build + Space
   upload only if reader-visible content changed).
