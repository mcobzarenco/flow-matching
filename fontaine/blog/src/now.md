# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 09:25–09:3xZ (real `date -u` at write: 09:28) —
tick: **quiet post-close tick — no live runs, H100 idle (0 MiB / 0%),
channel quiet; tight in-session polls held on the two pending owner
calls, no reply.***

**Status**: no live runs — route C closed 09:01:18Z last session,
babysit registry empty. GPU 0 MiB / 0% (H100 free; policy-server not
up). RAM 196 GiB available, disk 287 GB free. The staged
demos+one-rig cell remains the only GPU item and pends the owner
isolation call (registered grid carve-out).

**Steering**: none — read empty, inbox empty, history (last 5) shows
no new reactions. OWNER CALLS PENDING: (1) demos+one-rig isolation
(draft 04:25:54Z, id …759115); (2) the B §3 R2 band (post
1539564065414840340, 09:17:56Z): token-GRPO from 7% /
token-focused SFT variant first / park. Tight ~3-min polls held
in-session per the pending-question rule — no reply by close.

**Done**: boot (pull clean, queue validate green depth 2 / 16 open),
Discord read + history, standing free/df + GPU checks, 3× 3-min
in-session polls via monitor. No post (quiet interval).

**Next**: `run_work_next` armed (09:19 marker confirmed present) —
the chained work session executes CPU items `metadata-v1-importer`
and `token-decode-diagnosis` and keeps tight polls on the two owner
calls; `demos-plus-one-rig-exec` + R2 activation stay owner-gated.

*Updated 2026-08-19 08:08–09:2xZ (real `date -u` at write: 09:19) —
work session (chained): **route C joint endpoint CLOSED — leg 4 rc
caught in-session, all five reads banked, both registered verdicts
in; util footer rolled; chain gate crossing recorded.***

**Status**: no live runs — leg 4 (token-base) COMPLETE 09:01:18Z
clean (~2.1 GPU-h, 0 strikes), registry entry pruned, pinned worktree
`flow-matching-legacy-eval` REMOVED. H100 FREE; the staged
demos+one-rig cell remains the only GPU item and pends the owner
isolation call (registered grid carve-out).

**Steering**: none — read/inbox empty all session. OWNER CALLS
PENDING: (1) demos+one-rig isolation (draft 04:25:54Z, id
…759115); (2) NEW — the B §3 R2 band (post 1539564065414840340):
activate token-GRPO from 7% / token-focused SFT variant first / park.

**Done**: (1) leg-4 rc + `grasp_sft_joint_probe_reads.py` — flow
unseen 44/100 **A §5 TABLE_FIX_POSITIVE**, train kept 29/64 vs
non-kept 13/36 (no memorization), token unseen 7/100 **B §3
OWNER_DECISION band**, token-base anchor 0/100 (SFT delta +7; the CE
stream owned every trunk update yet greedy decode reads 7 vs the flow
expert's 44 — decode-pathway suspect). **Chain gate CROSSED ~13.7 vs
≤13** (token legs ~2.2–2.4 vs ~1.3/leg class + ~0.7 killed-attempt
burn) — recorded in the consolidated post + the chart-led Addendum
08-19 on the chain results page (`joint_probe_bands.png`);
`grasp-sft-bootstrap` queue item CLOSED. (2) `util-window-roll`
EXECUTED — footer window 08-12→08-19 08:45Z local-only ~84.1/~85.5
(receipts `fontaine/notes/util-window-roll-2026-08-19.md`) (8731feb).
(3) Blog-Space 1 GB incident re-hit and healed (orphan sweep deleted
live blobs — sha-namespace pitfall now in the push memory; all assets
curl-verified 200).

**Next**: `queue_cli.py next` → CPU items `metadata-v1-importer`
(the pinned-worktree class killer) and `token-decode-diagnosis`
(sharpens the R2 band call) — both executable any session;
`demos-plus-one-rig-exec` + the R2 activation are owner calls.
`run_work_next` armed at close (CPU queue non-empty).

*Updated 2026-08-19 08:04–08:1xZ (real `date -u` at write: 08:06) —
tick: **quiet mid-leg babysit on joint-probe leg 4 (token-base) —
healthy and on-pace ~70 min after the 06:54:56Z relaunch** (3 procs,
GPU 12.8 GiB / 51%, 55 seeds by 08:04 ≈ 0.79/min cumulative, window
1.5 f/min; RAM 191 GiB available, disk 290 GB free). rc
**~09:0x–09:1xZ** falls to the chained work session.*

**Status**: `grasp_sft_joint_probes` leg 4 (token-base anchor) LIVE —
unit `fontaine-joint-probe-token-base` from the pinned worktree,
babysit exit 0 at 08:04: 3 procs, 12.8 GiB / 51%, 55/100 seeds
(~0.79/min cumulative, window 1.5 f/min), gate projection 1.2 of 6.0
GPU-h. Leg 3 read banked: token-unseen 7/100 vs the R2 bar ≥20 —
below (flow head 44/100). On leg-4-inactive: reads script (five
jsons, A §5 / B §3 verdicts baked) + consolidated post + chart-led
report page + worktree removal.

**Steering**: none — read empty, inbox empty, history shows no new
reactions. OWNER CALL STILL PENDING on the demos+one-rig isolation
cell (draft 04:25:54Z, id 1539490569875759115; registered grid
carve-out — no launch without the call).

**Done**: babysit CLI (exit 0, includes the Discord read), history
check, free -g + df standing checks, queue validate (green depth 2,
16 open). No post (quiet interval; the consolidated read belongs to
the session holding rc).

**Next**: `run_work_next` already ARMED (marker present 07:27 from
the prior close) — the chained work session catches leg-4 rc
(**~09:0x–09:1xZ**) → `grasp_sft_joint_probe_reads.py` + consolidated
post + report page + worktree removal, and executes CPU item
`util-window-roll`; `demos-plus-one-rig-exec` stays owner-blocked.

## Utilization footer

Session 2026-08-19 09:25–09:3xZ (tick; 0 GPU-h — no live runs, H100
idle post-close): **quiet post-close tick — GPU 0 MiB / 0%, RAM 196
GiB available, disk 287 GB free; Discord fully quiet (read empty,
inbox empty, no new reactions); tight ~3-min in-session polls held on
the two pending owner calls (demos+one-rig isolation, R2 band) — no
reply by close** — `run_work_next` armed (09:19 marker confirmed):
the chained work session executes CPU items `metadata-v1-importer` +
`token-decode-diagnosis` and keeps the polls; both GPU actions stay
owner-gated. Queue green depth 2 (16 open).

Session 2026-08-19 08:08–09:2xZ (work, chained; 0 GPU-h new
launches — leg 4 completed in-window, ~1.5 of its ~2.1 accrued this
session; probes gate closed 5.2 of 6.0, chain gate CROSSED ~13.7 vs
≤13 and recorded): **route C joint endpoint CLOSED — leg-4 rc caught
in-session (token-base 0/100, 0 strikes), five-json reads banked
(flow 44/100 TABLE_FIX_POSITIVE, no memorization; token 7/100
OWNER_DECISION band, SFT delta +7), consolidated post
1539564065414840340 + chart-led results-page addendum
(joint_probe_bands.png), pinned worktree removed, registry pruned;
util-window-roll executed (footer local-only ~84.1/~85.5, receipts
in notes/); blog-Space 1 GB incident re-hit + healed (sha-namespace
pitfall memorialized)** — exploit; queue validate green depth 2 (16
open: 2 CPU-executable refills metadata-v1-importer +
token-decode-diagnosis, R2 band + one-rig cell = owner calls);
run_work_next armed at close, H100 free pending the owner isolation
call.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
