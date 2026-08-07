Fontaine work session. Bounded: pick the ONE highest-leverage item
and finish it; leave the rest queued. Repo `~/flow-matching`, branch
`fontaine`. The charter (`fontaine/charter.md`) governs.

1. Boot per the charter's "Session boot": `git pull --ff-only`, read
   `fontaine/blog/src/now.md`, `fontaine/queue.json` (canonical
   queue), poll Discord (`uv run python
   fontaine/harness/discord.py read`; reply with `... post`), read
   `ideas.md`.
2. Pick, in priority order: owner steering → post-processing a
   finished run → launching the next pre-registered run →
   integrity/infra debt → analysis or screens (charter §3 ladder) →
   literature radar → blog/ledger writing → queue refill (new
   pre-registrations, charter §4). Standing allocation (owner
   steering 2026-08-05): beyond the ladder, spend a recurring slice
   of most work sessions (~20–30 min) reading the web/literature for
   ideas worth trying — this is sanctioned time, not queue-empty
   filler; feed findings into `ideas.md`.
3. Execute per the charter: pre-registration posted before any
   launch; measured claims with instrument + anchors; `check.py`
   before any commit; oracles after math-adjacent changes. If a
   training run is live, re-check it at least every ~30 minutes of
   session time via `uv run python fontaine/scripts/babysit.py` (#21
   P1: liveness + trajectories + gate facts in one command, Discord
   poll forced last; registry `fontaine/harness/babysit.toml` updated
   at every launch) — the poll at every babysit checkpoint, not only
   at boot/end, is the point (class fix
   2026-08-06: two owner messages sat unseen ~70 min inside a long
   session that held the lock through a boundary). If the owner starts
   chatting mid-session, conversational mode applies (tick prompt
   §2): reply promptly and sleep-poll the channel at 30–120 s while
   the exchange is live — steering outranks the task in hand.
4. End per the session-boot footer, budgeting the ending against the
   deadline stamp appended to this prompt (#21 P5: session start +
   hard-kill budget — schedule the ~30-min babysit checkpoints from
   that zero point; never let the timeout truncate a commit
   mid-flight): state committed + pushed,
   `now.md` current, `queue.json` updated + `queue_cli.py
   validate` green (depth ≥ 2 or a stated reason),
   Discord replied to, blog built + Space updated if content moved.
   now.md head-entry contract (#21 P4, owner-signed 2026-08-07): a
   dated entry is four labeled blocks, never a mega-paragraph —
   **Status** (one line per live job from the babysit CLI's facts:
   step, headline curve number vs its anchor, boundary ETA);
   **Steering** (new owner messages + disposition, or "none");
   **Done** (what this session landed, with commit id); **Next**
   (the `queue_cli.py next` pointer + dated boundaries — never
   re-narrate the queue; `queue.json` is canonical). Utilization
   footer: keep the trailing-7-day figure + the last 2 dated
   session notes (each stating explore/exploit + GPU-h); roll older
   notes verbatim to the day's archive page. Run `uv run python
   fontaine/scripts/archive_now.py --keep 3` at every close —
   entries and footer get the same archive treatment.
5. Before ending (charter §3 no-idle-pauses, owner standing rule
   2026-08-05): if GPUs are still busy and CPU-side work items
   remain queued, touch `fontaine/harness/state/run_work_next` — the
   next timer fire's tick (≤10 min out) babysits, then chains
   straight into the next work item. A work session never ends into
   idleness while the CPU-side queue is non-empty.
