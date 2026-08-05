Fontaine tick session (babysit). Budget: minutes, not hours. Repo
`~/flow-matching`, branch `fontaine`. The charter
(`fontaine/charter.md`) governs; its "Session boot" section is this
prompt's contract.

1. `git pull --ff-only`. Read `fontaine/blog/src/now.md`.
2. Poll Discord: `uv run python fontaine/harness/discord.py read`
   (cursor-managed; replies via `... discord.py post "text"`). Owner
   messages are steering: acknowledge in-channel, record in
   `now.md`, apply at the next decision point.
3. If a training run is live: liveness by pgrep/GPU memory (never a
   log tail); latest step; curve vs the pre-registered anchors in
   the launcher header; anomaly scan (loss spikes, OOM, substitution
   flood, utilization collapse). Healthy → update `now.md`, exit.
   Anomalous → diagnose the mechanism (charter §6 discipline); kill
   only at a save boundary and only per the pre-registered gates;
   escalate per charter §7 when warranted.
4. If the run finished or died: post-process per charter §4 (panel
   score, reports, blog post, ledger row, artifact uploads), then
   launch the next queued pre-registered run. If that exceeds a tick
   budget, continue as a work session (`fontaine/prompts/work.md`
   defines it — you may keep going in this session).
5. If idle and the queue is healthy: exit quickly — do NOT invent
   work. If the queue is below depth 2: continue into a work session
   to refill it.
6. End: commit + push state (`now.md` always; blog build + Space
   upload only if reader-visible content changed).
