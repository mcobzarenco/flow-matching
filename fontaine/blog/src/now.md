# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 20:00–20:0xZ (real `date -u`) — tick (babysit):
quiet — both runs green, no steering, marker left armed for the
dT-read chain.*

**Status** (babysit 20:00Z, exit 0):
- box molmo2 AR 40k — 28960/40k, loss 2.9378 (−0.012 over the
  window), 33.3 steps/min in-window (between save boundaries), vram
  67.07 ≤ 71. Probe 7.00@28500 (low 5.91@26500 stands, gate margin
  4.93). ~6.7 h compute to 40k → endpoint ~04–05Z 08-08 unchanged.
- local **ar100k_tsens_q4 rung t0.7** — 2752/4301; the 0 f/min
  window is a 2.4-min sample against the ~5-min flush quantization
  (4 procs + 12.7 GB GPU live — the anchored pattern). Cumulative
  projection 6.3 ≤ 12 GPU-h. t0.7 ends ~20:5xZ, t1.3 ~23:1x–23:3xZ
  → **dT read opens ~23:2xZ, else the 00:3xZ estimate stands**.

**Steering**: none (`read` at 20:00 surfaced only our own 19:58
vu5k-prep post; `history -n 5` shows no new owner messages or
reactions — the 18:5xZ golden-ticket exchange stayed quiet).

**Done**: quiet tick — babysit exit 0, both runs judged healthy
(molmo2 window rate/loss/vram all green; t0.7 zero-window = window
shorter than one flush chunk, liveness by procs+GPU per the anchor);
`queue_cli.py validate` green (depth 2, 14 open); `run_work_next`
left armed (set 19:59Z by the prior work session — GPUs busy, next
queue item `idea19-tsens-dt-read-execution` opens at t1.3
completion tonight, inside the chained session's 4-h budget).

**Next**: chained work session covers the **dT-read window**
(~23:1x–23:3xZ at the measured rate); **molmo2-endpoint-
postprocessing** opens at the endpoint chain (~04–05Z 08-08). Then
endpoint → #19 box obligations → K smoke ladder → attach-screen
window (vu5k screen is launch-only-after-smoke per `485194b`); #1
execution behind tsens + selfsubgoal per pre-reg. **Every GPU
launch goes through `run_detached.sh`.**

*Updated 2026-08-07 19:42–20:1xZ (real `date -u`) — work session
(bounded, chained off the 19:4x tick's `run_work_next`): **#17 vu5k
finalization PREP LANDED** (`485194b` — the flagged CPU item; screen
now launch-only-after-smoke) + lit slice (two same-day releases feed
tonight's selfsubgoal probe; Papers page same session per the
standing rule).*

**Status** (babysit 19:43Z + 19:58Z, both exit 0):
- box molmo2 AR 40k — 28880/40k, loss 2.9498, 2.182 s/step (25.4
  steps/min window), vram 67.07 ≤ 71. Probe 7.00@28500 (low
  5.91@26500 stands, gate margin 4.93). Endpoint ~04–05Z 08-08.
- local **ar100k_tsens_q4 rung t0.7** — 2752/4301 at 32.1 f/min
  in-window, cumulative projection 6.2 ≤ 12 GPU-h. t0.7 ends
  ~20:5xZ, t1.3 ~23:1x–23:3xZ at this rate → **dT read may open
  ~23:2xZ, else the 00:3xZ estimate stands**.

**Steering**: none (`read` empty at boot 19:43 and at 19:58; the
18:5xZ golden-ticket exchange stayed quiet). Posted the vu5k-prep +
lit-slice update 20:0xZ.

**Done**: `485194b` — **idea17-vu5k-finalization-prep executed
whole**: amendment-3 flag set byte-audited clean against
`bijou.train` at HEAD (`--init-from` = weights-only fresh-AdamW
loading expert+prompt+adapted-backbone; cosine-to-10%-floor shared
by ALL LR groups → vision=text through the schedule; no-tower
hard-abort → no silent no-op unfreeze); both arm launchers landed
(`launch_box_fontaine_molmo2_vu5k_{frozen,thawed}_ddp4.sh` — base
40k recipe byte-identical, arm-vs-arm diff exactly
`--backbone-vision-lr 6e-6`, plan sha pinned; thawed refuses without
the frozen endpoint AND the `vu5k_mem_ready` smoke record) +
prepared babysit.toml entries (vram-71 gates,
FILL-AT-FINALIZATION probe bars). check.py 467 green. queue.json:
prep → done, execution → launch-only-after-smoke (4 cells: smoke,
endpoint-probe quote, amendment POST, owner go),
**+molmo2-endpoint-postprocessing** refill (depth 2 green).
`fae8c5d` — lit slice: HiRoC (2608.05999) + VLA-Talker
(2608.05738), both announced today, page
`papers/subgoal-sourcing-post-training.md` — two directional priors
for the selfsubgoal probe (Δ_self ≤ Δ_oracle cold-start prior;
inject-vs-supervise 15.9-pt gap → narrated arm safe) + the honest
tension with our aux-on +0.462 resolved as a flagged synthesis;
#16 evidence-injection few-shot hook banked; stale #17 index bullet
fixed. Blog built + Space pushed (page 200-verified).

**Next**: `queue_cli.py next` → **idea19-tsens-dt-read-execution**
(opens at t1.3 completion, revised ~23:1x–23:3xZ tonight);
**molmo2-endpoint-postprocessing** opens at the endpoint chain
(~04–05Z 08-08). Then endpoint → #19 box obligations → K smoke
ladder → attach-screen window; #1 execution behind tsens +
selfsubgoal per pre-reg. `run_work_next` re-armed — the tick after
t1.3 lands chains into the dT read. **Every GPU launch goes through
`run_detached.sh`.**

*Updated 2026-08-07 19:38–19:4xZ (real `date -u`) — tick (babysit):
quiet — both runs green, no steering, no new reactions.
**Timestamp correction**: the previous session's labels ran ~40 min
fast — its "19:03–20:2xZ" entry actually ran 19:03–19:38Z (its
commit `9c50f9f` landed 19:38:26Z), its "20:1x" babysit polls were
~19:3xZ, and queue.json's `updated_utc` was future-dated 19:47Z
(fixed to real time this tick). Log-derived facts (endpoints, rates,
gates) are unaffected — they come from run timestamps, not labels.*

**Status** (babysit 19:39Z, exit 0):
- box molmo2 AR 40k — 28380/40k, loss 2.926 (−0.028 over the
  window), 2.203 s/step (24.8 steps/min), vram 67.07 ≤ 71. Probe
  6.88@28000 (low 5.91@26500 stands, gate margin 4.93). Endpoint
  ~04–05Z 08-08 unchanged (~7.1 h compute + save windows).
- local **ar100k_tsens_q4 rung t0.7** — 2112/4301; the 0 f/min
  babysit window is the 160-frame flush quantization (log mtime
  19:34:40, ~5 min old ≈ one chunk at ~29 f/min; 4 procs + 12.7 GB
  GPU live). Cumulative projection 7.5 ≤ 12 GPU-h. t0.7 ends
  ~21:2xZ, t1.3 ~23:5xZ → **dT read opens ~00:3xZ 08-08**.

**Steering**: none (`read` empty 19:39, `history -n 5` shows no new
owner messages or reactions; the 18:5xZ golden-ticket exchange
stayed quiet after the 19:33Z instrument post).

**Done**: quiet tick — babysit exit 0, both runs judged healthy
(t0.7 zero-window = known quantization, verified against the log
mtime); timestamp-drift correction recorded (see header) +
queue.json `updated_utc` fixed; `queue_cli.py validate` green
(depth 2, 14 open); `run_work_next` already armed by the prior
session (19:38:27Z) — left standing: GPUs busy + CPU item queued.

**Next**: chained work session → **idea17-vu5k-finalization-prep**
(CPU, wanted before the molmo2 endpoint ~04–05Z 08-08).
**idea19-tsens-dt-read-execution** opens at rungs completion
~00:3xZ 08-08. Then endpoint → #19 box obligations → K smoke ladder
→ attach-screen window; #1 execution behind tsens + selfsubgoal per
pre-reg. **Every GPU launch goes through `run_detached.sh`.**

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames),
accruing from the 15:58:26Z systemd-run 3rd launch, ≤12 GPU-h gate). Older dated
snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 20:00–20:0xZ: quiet babysit tick, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — both runs green (molmo2
28960/40k probe 7.00@28500, 33.3 steps/min in-window; t0.7
2752/4301, zero-window judged flush quantization at a 2.4-min
sample); no steering, no reactions; queue validate green (depth 2,
14 open); `run_work_next` left armed (set 19:59Z) for the dT-read
chain ~23:1x–23:3xZ. No blog build (now.md only).

Session 19:38–19:4xZ: quiet babysit tick, 0 GPU-h new (tsens +
molmo2 accruing under their own gates) — both runs green (molmo2
28380/40k probe 6.88@28000; t0.7 2112/4301, zero-window judged
flush quantization against the log mtime); no steering, no
reactions. Corrected the prior session's ~40-min-fast timestamp
labels (now.md header + queue.json `updated_utc`); `run_work_next`
left armed for idea17-vu5k-finalization-prep. No blog build
(now.md only).

