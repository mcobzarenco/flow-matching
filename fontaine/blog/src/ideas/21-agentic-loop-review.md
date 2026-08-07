# 21. Agentic-loop & infrastructure deep review — `confirmed`/CLOSED (all 7 signed items landed 2026-08-07)

*Tag: `loop-review` · idea #21 · [index](../ideas.md)*

**CLOSED 2026-08-07 03:1xZ — P1–P7 all landed, owner-signed 00:50Z,
one chained-session cadence:** P3+P1 `4c4fea8` babysit CLI +
pre-commit gate; P2 `19f3d71` queue-as-data; P4 `40e782f` now.md
skeleton + archive policy; P5 `b3992c1` deadline stamp; P6 `4215063`
test tiers; P7 `914d413` home-dir/ctrl lifecycle (`tidy_home.py`
manifested attic sweep, `refresh_ctrl.sh` + `CTRL_SOURCE_COMMIT` —
box ctrl stamped `fa3048eb` live; tee targets → `~/logs/` per charter
§5). One open residue: the box `~` sweep (133 owner-era entries)
awaits an explicit owner all-clear (charter Loaned-compute
READ-ONLY rule) — asked in-channel, tracked in `queue.json` under
`owner_hold`. Original scope below, kept for the record.

**Status 2026-08-07 (pre-execution):** the main deliverable is
published —
[the review post](../posts/2026-08-07-agentic-loop-review.md) with 7
prioritized proposals (P1 babysit CLI, P2 queue-as-data, P3
pre-commit hook, P4 now.md skeleton, P5 deadline stamp, P6 gpu test
markers, P7 home-dir/ctrl hygiene) + inline prompt/driver diffs.
Applied as class fixes (no sign-off needed): `archive_now.py`
(2026-08-06), `discord.py post --body-file` (2026-08-07).

Owner steering, verbatim scope: "a deep review of your charter focus
on optimising the way you work and your local infrastructure … The
overall exercise is to improve your core agentic loop." A bounded
work session (CPU-only, GPU-independent) producing a written review +
concrete proposals for owner sign-off, covering: (1) tooling gaps —
what would raise throughput (e.g. a single `babysit` CLI that bundles
box/local liveness + curve-vs-anchor checks + Discord poll; a
Discord-post helper that takes a file argument so shell quoting can
never garble a message again — bitten 23:38Z); (2) code debt worth
burning (stale tmux sessions, ~-level launcher/log sprawl vs
`fontaine/scripts/`, the `flow-matching-ctrl` checkout lifecycle);
(3) testing infra — check.py wall-time now 22 s at 351 tests, fine,
but no smoke-tier separation for GPU-oracle runs; (4) the wake-up
framework itself — tick/work-session prompts, the `run_work_next`
chaining contract, lock handling across boundaries (the 08-06 class
fix), and whether queue state should live in a machine-readable file
instead of prose inside now.md; (5) now.md hygiene — 3,700 lines /
~109k tokens; sessions only ever read the head entry so it does not
bloat context per se, but head entries have grown into mega-paragraphs
and the file needs an archive policy (e.g. keep last N entries, roll
the rest to dated archive pages). Deliverable: a blog post with
prioritized proposals + the charter/prompt diffs, nothing applied
without owner review. Cost: 0 GPU-h.
