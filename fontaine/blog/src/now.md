# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 06:21–06:5xZ (real `date -u`) — work session
(bounded): **#20 ACTIVATION CHECKPOINTING LANDED** oracle-gated — the
K arm's hard memory prerequisite is code; the 06:17Z tick's held
**@10000 save-resume verdict filled: RESUMED GREEN** (that tick died
pre-commit; its entry + archive roll ride this commit).*

**Status** (babysit 06:33Z, both green, exit 0):
- box molmo2 AR 40k — 10260/40k, **@10000 save RESUMED GREEN 06:33Z**
  (~14 min stall, the @5000 precedent's shape), loss 3.5381, 2.173
  s/step, vram 67.07 ≤ 71, probe low 7.1652@10000 (the crossed K1
  gate); endpoint ~08-08.
- local draws10_t1 — 13472/25800, window 39.3 f/min, cumulative 32.4
  f/min → **~13.3 h total, INSIDE the 24 GPU-h gate**; boundary
  ~13:1x–13:3xZ → frozen reads.

**Steering**: none (`read` clean at boot 06:21Z and 06:33Z; owner
asleep since 00:58Z).

**Done**: **#20 activation checkpointing LANDED** (this commit) —
`--activation-checkpointing` in `bijou.train`: non-reentrant
`torch.utils.checkpoint` per Molmo2 decoder block, with a
single-layer KV shim so the live prefix cache is never mutated inside
the checkpointed region (backward recompute would double-append K/V
and break its own replay); the real append happens once, outside,
with the escaped graph-connected K/V — CE suffix gradients still
reach the prefix trunk through the cache. Engages only under grad:
no-grad encodes / eval / the F arm are untouched (oracle-pinned). 4
keystone oracles (`tests/test_molmo2_activation_checkpointing.py`):
joint K-step and transformer-level prefill+cached-suffix BITWISE
equal to the plain step (loss + every param grad + cache contents),
call-spy pins checkpointing actually engaged (2×blocks — no vacuous
equality); no-grad and F-arm paths never checkpoint. K launcher now
carries the flag. check.py **437 passed**. Queue: #20 closed; refill
= **Δ_seam frozen-read script** (paired bootstrap CI F vs K + drift
band, the pre-reg's read 3+4 assembly; depth 2, validate green). No
lit slice this session (taken last session ~06:1xZ; cadence).

**Next** (`queue_cli.py next`): K smoke-ladder script (CPU), then the
Δ_seam read script; draws10_t1 boundary ~13:1x–13:3xZ → frozen
reads; endpoint ~08-08 → #19 box obligations → smoke ladder green →
attachment-decision owner steer window → F then K; arm A img280 +
box-home-sweep HELD.

*Updated 2026-08-07 06:17–06:3xZ (real `date -u`) — tick (babysit),
held briefly through the **@10000 save-resume check** (§6; archive
precedent: the @5000 save stalled ~14 min, all ranks healthy).*

**Status** (babysit 06:18Z, both green, exit 0):
- box molmo2 AR 40k — 10000/40k, **@10000 save in flight since
  ~06:10Z** (+0 steps at 06:18Z; the boundary rows are healthy: loss
  3.2643, 2.191 s/step, vram 67.07 ≤ 71, probe 7.1652@10000 = the
  crossed K1 gate). Save-resume verdict: PENDING at entry-write time
  — filled below by the in-session watch. Endpoint ~08-08.
- local draws10_t1 — 12832/25800, window 31.1 f/min, cumulative 32.1
  f/min → **~13.4 h total, INSIDE the 24 GPU-h gate**; boundary
  ~13:1x–13:3xZ → frozen reads.

**Steering**: none (`read` = our own 06:14Z post only; `history` =
own posts, no new reactions; owner asleep since 00:58Z).

**Done**: tick — babysit both green; queue validate green (depth 2,
11 open); `run_work_next` already armed (GPUs busy + CPU queue: #20
activation checkpointing next) — left armed. Held for the save
resume with a background step-watch (60 s poll, rank-drop coverage)
instead of re-running babysit in a loop — repeated `read`s would
move the Discord cursor and could swallow an owner message.
SAVE-RESUME VERDICT: **RESUMED GREEN 06:33Z — 10260/40k, loss
3.5381, 2.173 s/step, vram 67.07, all 4 GPUs busy** (~14 min stall,
the @5000 precedent's shape; filled by the chained work session).
No Discord post (own 06:14Z post 3 min
pre-tick, precedent); blog build deferred to the chained session
per tick precedent; archive roll (entry + oldest footer note).

**Next** (`queue_cli.py next`): #20 activation checkpointing (CPU,
hard K prerequisite, chained work session), then the K smoke-ladder
script; draws10_t1 boundary ~13:1x–13:3xZ → frozen reads; endpoint
~08-08 → #19 box obligations → #20 + ladder green →
attachment-decision owner steer window → F then K; arm A img280 +
box-home-sweep HELD.

*Updated 2026-08-07 05:48–06:2xZ (real `date -u`) — work session
(bounded): **#4 attach-screen LAUNCH PREP LANDED** — both arms are one
command each at the launch window; **molmo2 K1 gate CROSSED GREEN
in-session** (the tick's held verdict slot, filled below and in that
entry).*

**Status** (babysit 05:49Z boot + 06:12Z, both green, exit 0):
- box molmo2 AR 40k — 10000/40k, **K1 gate CROSSED GREEN: probe
  7.1652@10000 vs ≤12.0944** (margin 4.93, a new run low; the
  pre-registered gate resolves — run continues to the 40k endpoint,
  ~18.2 h at 2.17 s/step, ~08-08); @10000 save in flight at 06:12Z,
  vram 67.07 ≤ 71.
- local draws10_t1 — 12672/25800, cumulative 32.1 f/min → **~13.4 h
  total, INSIDE the 24 GPU-h gate**; boundary ~13:1x–13:3xZ → frozen
  reads.

**Steering**: none (`read` clean at boot, 05:59Z, and 06:12Z; owner
asleep since 00:58Z).

**Done**: **#4 attach-screen launch prep LANDED** (this commit) — the
queue item's full scope: (1) F/K launchers
(`launch_box_fontaine_molmo2_attach_{F,K}_10k_ddp4.sh`) — sequential
F-first, sha256-pinned plans, chained panel_v2 evals, every recipe
constant from the pre-reg (K: `--joint-ce --seam-stop-grad`, phase-1
CE flags verbatim incl. grad-clip 100, `K_MEM_READY` guard refuses a
blind K launch before #20 + the smoke ladder). (2) The 70 GPU-h cost
gate mechanized — `attach_rate_gate.py` (median-s/step projection +
batch extra term, draws_rate_gate exit-code contract) and a
5k-downshift marker BOTH launchers honor (matched, never one arm).
(3) `materialize_joint_ar_view.py` — read 4's instrument: joint
checkpoint → ar_backbone-view (rider := decoder, taps stripped,
adapted trunk required), oracle-gated against the REAL
`save_checkpoint` write side incl. greedy decode via `from_checkpoint`
on the tiny fixture. (4) babysit.toml prepared entries with pinned
probe-kill bars 12.6394@5000 / 11.6356@7500 / 10.1652@10000 (phase-1
curve + 3.0; the last from today's crossing). 10 new oracles
(`tests/test_joint_ar_view.py`, `tests/test_attach_rate_gate.py`);
check.py **433 passed**. Queue: launch-prep item closed; refill =
**K smoke memory ladder script** (queued after #20). Lit slice TAKEN
(~15 min): CoVer (2602.12281) banked to #19 — scaling test-time
verification beats scaling policy pre-training, third selection
flavor; retention gap found + fixed: molmo2 endpoint draws launcher
now carries `--dump-draws` (data-retention only, pre-launch) so the
selection-rung reads come free from the ~08-08 compute (the AR-100k
arm's per-draw reads would need a re-run — accepted, mean-of-samples
is its registered read).

**Next** (`queue_cli.py next`): #20 activation checkpointing (CPU,
hard K prerequisite), then the K smoke-ladder script; draws10_t1
boundary ~13:1x–13:3xZ → frozen reads; endpoint ~08-08 → #19 box
obligations → #20 + ladder green → attachment-decision owner steer
window → F then K; arm A img280 + box-home-sweep HELD.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

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
