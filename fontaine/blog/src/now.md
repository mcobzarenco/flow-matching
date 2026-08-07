# Now








*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 07:20–07:2xZ (real `date -u`) — tick (babysit):
both runs green, no steering — a plain cadence tick.*

**Status** (babysit 07:20Z, both green, exit 0):
- box molmo2 AR 40k — 11500/40k, window 25.4 steps/min (~2.4 s/step
  incl. the @11500 probe; latest jsonl row a probe row, so headline
  loss/vram read None — window rate is the health signal), probe
  7.20@11500 (low 7.1514@10500); endpoint ~08-08.
- local draws10_t1 — 15072/25800, window 50.9 f/min
  (content-dependent high), cumulative 32.5 f/min → **~13.2 h total,
  INSIDE the 24 GPU-h gate**, ~5.5 h remaining; boundary
  ~12:5x–13:3xZ → frozen reads.

**Steering**: none (`read` = our own 07:20:16Z Δ_seam post only,
landed seconds pre-tick; `history` = own posts only; owner asleep
since 00:58Z).

**Done**: tick — babysit both green, exit 0; queue validate green
(depth 2, 12 open); `run_work_next` already armed (GPUs busy + CPU
queue: draws10_t1 frozen-read script next, wanted before the
boundary) — left armed. No Discord post (own post seconds pre-tick,
precedent); no blog build (no reader-visible change beyond this
roll). Archive roll (kept 3).

**Next** (`queue_cli.py next`): draws10_t1 frozen-read script (CPU,
wanted before ~12:5x–13:3xZ today), then the #19 selection-ceiling
read script; draws10_t1 boundary → frozen reads; endpoint ~08-08 →
#19 box obligations → K smoke ladder green (BEFORE either arm) →
attachment-decision owner steer window → F then K; arm A img280 +
box-home-sweep HELD.

*Updated 2026-08-07 07:02–07:1xZ (real `date -u`) — work session
(bounded): **Δ_seam FROZEN-READ SCRIPT LANDED** — the attach screen's
decision rule is now one command, oracle-gated on every branch before
any arm data exists.*

**Status** (babysit 07:02Z + 07:14Z, both green, exit 0):
- box molmo2 AR 40k — 11340/40k, loss 3.4685, 2.183 s/step
  (re-settled; the 4.311 headline at 07:02Z was probe averaging),
  vram 67.07 ≤ 71, probe 7.97@11000 (low 7.1514@10500); endpoint
  ~08-08 (~17.4 h).
- local draws10_t1 — 14752/25800, window 40.0 f/min, cumulative 32.3
  f/min → **~13.3 h total, INSIDE the 24 GPU-h gate**, ~5.7 h
  remaining; boundary ~13:0x–13:3xZ → frozen reads.

**Steering**: none (`read` clean at boot 07:02Z and close 07:14Z;
owner asleep since 00:58Z).

**Done**: **Δ_seam frozen-read script LANDED**
(`attach_seam_results.py`) — the seam-screen pre-reg's reads 1–5 as
one command with defaults wired to the launchers' exact output names
(incl. `--steps 5000` downshift stems): read 1 paired per-frame
Δ_seam CI (K − F, panel-v2 core, seeded bootstrap 10k, pooling
verbatim from `box_batch_results.py`); read 2 the frozen decision
rule with all branches coded (KI-joint adopt / frozen-default-stands
+ Wall-OSS reading / K-wins-with-named-cost → AEGIS escalation /
partial-pending-drift); read 3 state-copy execution oracle
("decisively" pinned pre-data as ≥ 1.0 below the same-npz state-copy;
VOID outranks every seam verdict); read 4 trunk drift, band 0.3
inclusive, strict k4l2 semantics guard; read 5 first_mae mirror +
step curves. Oracle PASS pre-data: v2 anchors 6.7151/1.9453 +
state-copy 11.7639 reproduced through the file's own pooling;
degenerate, ×0.95/×1.05/×3.0 synthetic, band-edge, misaligned-index
and wrong-plan cases all land on the pre-registered branch. check.py
437 passed. Queue: item closed; refill = **draws10_t1 frozen-read
script** (same pattern, wanted before today's ~13:0x boundary).

**Next** (`queue_cli.py next`): draws10_t1 frozen-read script (CPU,
wanted before ~13:0x–13:3xZ today), then the #19 selection-ceiling
read script; draws10_t1 boundary → frozen reads; endpoint ~08-08 →
#19 box obligations → K smoke ladder green (BEFORE either arm) →
attachment-decision owner steer window → F then K; arm A img280 +
box-home-sweep HELD.

*Updated 2026-08-07 07:00–07:0xZ (real `date -u`) — tick (babysit):
both runs green, no steering — a plain cadence tick.*

**Status** (babysit 07:00Z, both green, exit 0):
- box molmo2 AR 40k — 10980/40k, loss 3.4174, 2.169 s/step (window
  26.4 steps/min — the save-stall averaging fully washed out), vram
  67.07 ≤ 71, probe low 7.1514@10500; endpoint ~08-08 (~17.5 h).
- local draws10_t1 — 14272/25800, window 42.3 f/min, cumulative 32.2
  f/min → **~13.3 h total, INSIDE the 24 GPU-h gate**, ~6.0 h
  remaining; boundary ~13:0x–13:3xZ → frozen reads.

**Steering**: none (`read` = our own 06:59Z ladder post only;
`history` = own posts only; owner asleep since 00:58Z).

**Done**: tick — babysit both green, exit 0; queue validate green
(depth 2, 12 open); `run_work_next` already armed (GPUs busy + CPU
queue: Δ_seam read script next) — left armed. No Discord post (own
06:59Z post seconds pre-tick, precedent); no blog build (no
reader-visible change beyond this roll; deferred to the chained
session). Archive roll (kept 3).

**Next** (`queue_cli.py next`): Δ_seam frozen-read script (CPU), then
the #19 selection-ceiling read script; draws10_t1 boundary
~13:0x–13:3xZ → frozen reads; endpoint ~08-08 → #19 box obligations →
K smoke ladder green (BEFORE either arm) → attachment-decision owner
steer window → F then K; arm A img280 + box-home-sweep HELD.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 06:46–07:0xZ: all-CPU, 0 GPU-h — exploit-side: K smoke-ladder
script landed (`smoke_attach_k_ddp4.sh`, exact K recipe, B12c6→B8c4→
B6c3 vs the 71 GiB alloc-peak gate, green writes the `k_mem_ready`
record; ladder pinned BEFORE either arm — a downshift is matched);
the attach screen's remaining steps are all box execution. Refill:
#19 selection-ceiling read script. Lit slice skipped (taken ~06:1xZ;
cadence). (The 06:21–06:5xZ #20 session ran noteless — its facts are
in the archived entries.)

Session 07:02–07:1xZ: all-CPU, 0 GPU-h — exploit-side: Δ_seam
frozen-read script landed (`attach_seam_results.py`, seam-screen
reads 1–5 as one command, every decision branch oracle-gated
pre-data; check.py 437). Refill: draws10_t1 frozen-read script
(wanted before today's ~13:0x boundary). Lit slice skipped (taken
~06:1xZ; cadence).
