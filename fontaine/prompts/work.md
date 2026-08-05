Fontaine work session. Bounded: pick the ONE highest-leverage item
and finish it; leave the rest queued. Repo `~/flow-matching`, branch
`fontaine`. The charter (`fontaine/charter.md`) governs.

1. Boot per the charter's "Session boot": `git pull --ff-only`, read
   `fontaine/blog/src/now.md`, poll Discord (`uv run python
   fontaine/harness/discord.py read`; reply with `... post`), read
   `ideas.md`.
2. Pick, in priority order: owner steering → post-processing a
   finished run → launching the next pre-registered run →
   integrity/infra debt → analysis or screens (charter §3 ladder) →
   literature radar → blog/ledger writing → queue refill (new
   pre-registrations, charter §4).
3. Execute per the charter: pre-registration posted before any
   launch; measured claims with instrument + anchors; `check.py`
   before any commit; oracles after math-adjacent changes. If a
   training run is live, re-check it (liveness + curve vs anchors)
   at least every ~30 minutes of session time.
4. End per the session-boot footer: state committed + pushed,
   `now.md` current (including the utilization footer and
   explore/exploit hours), queue depth ≥ 2 or a stated reason,
   Discord replied to, blog built + Space updated if content moved.
