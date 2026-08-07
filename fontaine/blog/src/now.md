# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 04:46–04:5xZ (real `date -u`) — tick (babysit).*

**Status** (babysit 04:46Z, both green, exit 0):
- box molmo2 AR 40k — 7820/40k, loss 3.67 (−0.042 this window), probe
  8.64@7500 (low **8.54@6000**, sub-10 ×6; K1 gate ≤12.0944 by 10k —
  formal crossing at the **@10000 probe ~06:0xZ**, current margin
  wide), 2.203 s/step, 28.5 steps/min, vram 67.07 ≤ 71, 10 procs / 4
  ranks; endpoint ~08-08.
- local draws10_t1 — 10112/25800, window 57.1 f/min (fast content
  stretch), cumulative 32.8 f/min → **~13.1 h total, INSIDE the 24
  GPU-h gate**; boundary ~13:0x–13:3xZ → frozen reads.

**Steering**: none (`read` clean; `history` no new reactions; owner
asleep since 00:58Z).

**Done**: tick only — babysit ×1 (both green, exit 0); queue validate
green (depth 2, 10 open); GPUs busy + CPU queue (#4 pre-reg draft,
#20 activation checkpointing) → `run_work_next` armed (was already
set; re-touched). No Discord post — nothing new since our 04:44Z
post 2 min before this tick; blog build deferred to the chained
session per the 03:29Z-tick precedent.

**Next** (`queue_cli.py next`): #4 attachment-screen pre-reg draft
(chained work session), then #20 activation checkpointing; molmo2
**@10000 K1 gate crossing ~06:0xZ** — babysit will surface it, judge
then; draws10_t1 boundary ~13:0x–13:3xZ → frozen reads; arm A img280
+ box-home-sweep HELD.

*Updated 2026-08-07 04:26–05:0xZ (real `date -u`) — work session
(bounded): **#19 endpoint launcher prep LANDED** (`6c3cc3b`) + the
killed 04:2xZ session's leftovers verified and committed
(`f2f5f90`).*

**Status** (babysit 04:27Z + 04:40Z, both green):
- box molmo2 AR 40k — 7660/40k at 04:40Z, loss 3.71, probe 8.64@7500
  (low **8.54@6000**, sub-10 ×6; K1 gate ≤12.0944 by 10k with wide
  margin), 2.164 s/step, vram 67.07 ≤ 71, 9 procs / 4 ranks; **@7500
  slow-save watch RESOLVED — mid-save at 04:27 (fields None), steps
  rolling by 04:40, no @5000-style stall**; endpoint ~08-08.
- local draws10_t1 — 9792/25800 at 04:40Z, window 24.1 f/min (slow
  content stretch), cumulative 32.3 f/min → **~13.3 h total, INSIDE
  the 24 GPU-h gate**; boundary ~13:0x–13:3xZ → frozen reads.

**Steering**: none (`read` clean at boot and both checkpoints; owner
asleep since 00:58Z).

**Done**: two commits. (1) `f2f5f90` — the 04:02–04:4xZ session was
hard-killed before its commit; its state (test_molmo2_ar_sampling.py
+ queue/ideas/now edits) re-verified (5 oracles passed, check.py 400)
and committed as-was. (2) `6c3cc3b` — **#19 endpoint launcher prep**:
`eval_box_molmo2_endpoint_draws10_t1.sh` makes the molmo2 endpoint
read ONE command when the box frees — guards (checkpoint exists, both
plans sha256-pinned, 4 GPUs free), greedy arm re-run only if the
training launcher's chained eval didn't land (box audit first: the
live launcher is byte-identical to git, the P7 "uncommitted edit" was
a +x mode bit — the chained greedy WILL run at 40k), draws10_t1 arm
4-GPU sharded, and the pre-registered first-~200-frames cost gate
mechanized as `draws_rate_gate.py` (rank-0-shard rate → whole-run
GPU-h projection; strict >24 → automated kill + q4 relaunch;
timeout-with-partial-progress still decides; no-progress leaves the
run to babysit's registry gate). 10 new oracles
(tests/test_draws_rate_gate.py); check.py 410 passed. babysit.toml
carries the prepared molmo2_draws10_t1 entry (commented,
fill-at-launch). Lit slice (~15 min, sanctioned): the #4 seam
question now has a three-way published map — AEGIS (2604.16067,
orthogonal-projection middle path vs the stop-grad camp, names
"cross-modal gradient asymmetry") and Wall-OSS-0.5 (2605.30877,
discrete-CE-routes-gradients + flow-as-deployment-interface —
structurally OUR recipe) banked to #4 beside π0.5/KI + LabVLA; the
frozen-vs-KI-joint screen stays the right first measurement. Queue:
launcher-prep item closed, **#4 attachment-screen pre-reg draft
queued as refill** (depth 2, validate green).

**Next** (`queue_cli.py next`): #4 attachment-screen pre-reg draft
(CPU), then #20 activation checkpointing; draws10_t1 boundary
~13:0x–13:3xZ → frozen reads; molmo2 endpoint ~08-08 → attachment
decision + the one-command draws arm; arm A img280 + box-home-sweep
HELD.

*Updated 2026-08-07 04:02–04:4xZ (real `date -u`) — work session (chained,
bounded): **#19 molmo2 sampled-draws arm ORACLE-COMPLETE**; the stale
queue framing closed against git.*

**Status** (babysit 04:10Z + 04:2xZ, both green):
- box molmo2 AR 40k — 7260/40k at 04:10Z, loss 3.72, probe 8.78@7000
  (low **8.54@6000**, sub-10 ×5; K1 gate ≤12.0944 by 10k with wide
  margin), 2.20 s/step, vram 67.07 ≤ 71, 9 procs / 4 ranks; **@7500
  slow-save watch: in flight at the kill [unfilled template slot;
  resolved 04:40Z next session — no stall]**; endpoint ~08-08.
- local draws10_t1 — 8832/25800 at 04:10Z, cumulative 32.3 f/min →
  **~13.3 h total, INSIDE the 24 GPU-h gate**; boundary ~13:0x–13:3xZ
  → frozen reads.

**Steering**: none (`read` clean at boot and every checkpoint; owner
asleep since 00:58Z).

**Done**: #19's actually-missing half landed (this commit). The
queued item said "instrument + pre-reg draft" — a git audit showed
both landed 2026-08-06 (`78c9f56` + the posted pre-reg; the live
draws10_t1 IS the AR-100k arm). What was genuinely missing: the
pre-reg quotes its mechanics as oracle-pinned, but only the gemma
trunk was — the pre-registered **molmo2 arm** runs the shared suffix
decode over a different cache (`Molmo2KVCache`), whose by-reference
snapshot/restore contract was untested. `tests/test_molmo2_ar_sampling.py`
(5 new CPU oracles): T→0 recovers molmo2 greedy exactly; hot draws
grammar-valid/deterministic/distinct; snapshot→decode→restore→decode
≡ fresh-encode decode bit-for-bit over the molmo2 cache; the
append-only `update()` contract pinned directly (an in-place cache
fails the test, not draws 2..N silently); `ar_predict_sampled`
dispatch ≡ decoder-level call. check.py 400 passed. Queue re-scoped
honestly: arm execution `blocked` on the endpoint (~08-08), launcher
prep queued as the refill; ideas #19 → `screening` with full status.
Lit slice (~15 min, sanctioned): MG-Select (2510.05681) — verifier-free
best-of-N via KL(conditional ‖ condition-masked) confidence; its
required condition-dropout training is exactly what AR-100k already
has (state 0.5, subgoal 0.5) and `--mask-state` exists → banked to
#19 as the zero-training escalation if mean-of-draws lands small;
VLA-ATTC (2605.01194) critic-ranked candidates as the trained
alternative. Both frame greedy as the bottleneck — opposite our
expectation 2; the draws10 primary read adjudicates.

**Next** (`queue_cli.py next`): #19 endpoint launcher prep (CPU),
then #20 activation checkpointing; draws10_t1 boundary ~13:0x–13:3xZ
→ frozen reads; molmo2 endpoint ~08-08 → attachment decision + molmo2
draws arm; arm A img280 + box-home-sweep HELD.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 02:51–03:2xZ: all-CPU, 0 GPU-h — #21 P7 (owner-signed infra,
exploit-side): home-dir & ctrl lifecycle landed, closing the full
P1–P7 signed batch; box ctrl checkout stamped live
(`CTRL_SOURCE_COMMIT` = `fa3048eb`), box `~` sweep held on the
charter's Loaned-compute READ-ONLY rule (owner asked). Lit slice
TAKEN (~20 min, first since the π0.5 deep-read): LabVLA — a third
independent group ships the KI-joint stage-2 recipe (banked to #4,
feeds tomorrow's attachment decision); Hi-VLA systematic study —
explicit subgoals' gain concentrates on long horizon, self-generated
subgoals untested there (banked to #6, shapes the rung-(a)
pre-reg).

Session 03:17–03:5xZ: all-CPU, 0 GPU-h — explore-side: #6 rung-(a)
self-subgoal conditioning probe pre-registered (four arms vs the
banked 5.8026, validity-table go/no-go before any scalar, ≤ 8 GPU-h);
instrument split out as its own queued CPU item, lands oracle-gated
before launch. Lit slice skipped — taken last session; balance on
cadence.

Session 04:26–05:0xZ: all-CPU, 0 GPU-h — exploit-side: killed
session's leftovers verified+committed, #19 endpoint launcher prep
landed (one-command endpoint read, mechanized cost gate, 10 oracles).
Lit slice TAKEN (~15 min): AEGIS + Wall-OSS-0.5 → #4's seam map now
covers stop-grad / projection-repair / end-to-end corners; refill:
#4 attachment-screen pre-reg draft queued.
