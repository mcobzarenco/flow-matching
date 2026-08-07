# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-07 05:46–06:1xZ (real `date -u`) — tick (babysit),
held open through the **@10000 K1 gate crossing** (~06:08Z, judgment
call §6: pre-registered gate resolution inside the session window).*

**Status** (babysit 05:46Z, both green, exit 0):
- box molmo2 AR 40k — 9400/40k, loss 3.5371, probe low **7.67@8500**
  (8.26@9000; K1 gate ≤12.0944 by 10k — crossing held in-session,
  verdict below), 2.196 s/step, vram 67.07 ≤ 71; endpoint ~08-08.
- local draws10_t1 — 11872/25800, cumulative 32.2 f/min → **~13.4 h
  total, INSIDE the 24 GPU-h gate**; boundary ~13:0x–13:3xZ → frozen
  reads.

**Steering**: none (`read` clean; `history` = our own posts only, no
new reactions; owner asleep since 00:58Z).

**Done**: tick + gate watch — babysit at 05:46Z (both green);
queue validate green (depth 2, 11 open); `run_work_next` already
armed (GPUs busy + CPU queue: #4 launch prep, #20 checkpointing) —
left armed. Held to ~06:09Z for the @10000 probe — verdict slot below, filled by
the in-session re-poll (an unfilled slot means the session died
pre-resolution; margin at 9000 was 3.84 under the threshold).
GATE @10000: **CROSSED GREEN 06:0xZ — probe 7.1652 vs ≤12.0944**
(margin 4.93, a new run low; filled by the chained work session — the
tick ended at commit and `run_work_next` chained straight into it).

**Next** (`queue_cli.py next`): #4 attach-screen launch prep (CPU,
chained work session), then #20 activation checkpointing; draws10_t1
boundary ~13:0x–13:3xZ → frozen reads; screen execution opens at
endpoint → #19 box obligations → #20 + launch prep →
attachment-decision owner steer window; arm A img280 +
box-home-sweep HELD.

*Updated 2026-08-07 05:07–06:0xZ (real `date -u`) — work session
(bounded): **#4 attach-screen instrument LANDED, oracle-gated** — all
three pre-registered parts; the K arm is now launchable code.*

**Status** (babysit 05:08Z boot + 05:34Z, both green, exit 0):
- box molmo2 AR 40k — 9080/40k, loss 3.6347, probe **new low
  7.67@8500** (8.26@9000, sub-10 ×10; K1 gate ≤12.0944 by 10k —
  formal crossing at the **@10000 probe ~06:0xZ**, margin huge),
  2.215 s/step, vram 67.07 ≤ 71; endpoint ~08-08.
- local draws10_t1 — 11552/25800, cumulative 32.4 f/min → **~13.3 h
  total, INSIDE the 24 GPU-h gate**; boundary ~13:0x–13:3xZ → frozen
  reads.

**Steering**: none (`read` clean at boot and both checkpoints; owner
asleep since 00:58Z).

**Done**: **#4 attachment-screen instrument LANDED** (this commit),
all three parts oracle-gated per the pre-reg. (1) Molmo2 residual
exports — the trunk-side tap protocol existed since WP1 (queue-title
audit paid off); the wiring was the gap: `Molmo2Encoder
residual_exports` + `Molmo2PromptConfig` field, `molmo2_residual_taps`
pins the rule (stride 3, last tap = final layer; 36 ⇒ 2,5,…,35),
`molmo2_residual_expert_config` mirrors trunk geometry, the
ar_backbone-only guard lifted for `--decoder flow
--conditioning-streams residual`, checkpoint save/load round-trips
(molmo2 flow checkpoints now load via `from_checkpoint`). (2)
`--seam-stop-grad` — taps detached before adapter projection in
`BijouModel.encode`. (3) `--joint-ce` — the K arm: Molmo2ARDecoder
rider beside the flow expert, CE suffix inside autocast + fp32 flow
outside, three-normalizer chunked-backward form, rider tables at
decoder-lr, saved as `joint_ce.safetensors`, and continued from the
endpoint's `expert.safetensors` under `--backbone-init-from` — that
last pinned in a **pre-reg AMENDMENT** ("decoder fresh" = the flow
expert; fresh CE tables would contradict "continuing verbatim").
13 new oracles (`tests/test_molmo2_residual.py`): taps byte-match
trunk hidden states, cache bit-identical with/without taps, stream
contract + padding invariance, stop-grad zero/nonzero with the
naive-joint negative control, and both α-edges **bitwise through the
real BijouTrainStep** (flow half ≡ F-arm step; trunk grads ≡ phase-1
CE step). check.py 423 passed. Queue: instrument item closed;
**launch-prep item queued** as refill (F/K scripts + the
joint-checkpoint AR-view materializer for the trunk-drift read);
validate green (depth 2, 11 open).

**Next** (`queue_cli.py next`): #4 attach-screen launch prep (CPU),
then #20 activation checkpointing (hard K prerequisite); molmo2
**@10000 K1 gate crossing ~06:0xZ** — babysit surfaces it, judge
then; draws10_t1 boundary ~13:0x–13:3xZ → frozen reads; screen
execution opens at endpoint → #19 box obligations → #20 + launch prep
→ attachment-decision owner steer window; arm A img280 +
box-home-sweep HELD.

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
