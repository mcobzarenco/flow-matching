# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-07 05:04–05:1xZ (real `date -u`) — tick (babysit).*

**Status** (babysit 05:04Z, both green, exit 0):
- box molmo2 AR 40k — 8300/40k, loss 3.704 (+0.06 this 100-step
  window, jitter — trend intact), probe 8.64@8000 (low **8.54@6000**,
  sub-10 ×7; K1 gate ≤12.0944 by 10k — formal crossing at the
  **@10000 probe ~06:0xZ**, margin wide), 2.169 s/step, vram 67.07 ≤
  71; endpoint ~08-08.
- local draws10_t1 — 10752/25800, window 37.7 f/min, cumulative 32.9
  f/min → **~13.1 h total, INSIDE the 24 GPU-h gate**; boundary
  ~13:0x–13:3xZ → frozen reads.

**Steering**: none (`read` surfaced only our own 05:03Z pre-reg post;
`history` no new reactions; owner asleep since 00:58Z).

**Done**: tick only — babysit ×1 (both green, exit 0); queue validate
green (depth 2, 11 open); GPUs busy + CPU queue (#4 attach-screen
instrument, #20 activation checkpointing) → `run_work_next` armed.
Drive-by: `queue.json` `updated_utc` was future-dated 05:17Z by the
previous session (committed 05:02Z) — corrected to real time. No
Discord post — our pre-reg post landed 1 min before session start;
blog build deferred to the chained session per tick precedent.

**Next** (`queue_cli.py next`): #4 attach-screen instrument (CPU,
chained work session), then #20 activation checkpointing; molmo2
**@10000 K1 gate crossing ~06:0xZ** — babysit surfaces it, judge
then; draws10_t1 boundary ~13:0x–13:3xZ → frozen reads; arm A img280
+ box-home-sweep HELD.

*Updated 2026-08-07 04:48–05:1xZ (real `date -u`) — work session
(bounded): **#4 attachment seam screen PRE-REGISTERED** — the
molmo2 stage-2 attachment decision is executable at the endpoint.*

**Status** (babysit 04:48Z boot + 05:00Z, both green, exit 0):
- box molmo2 AR 40k — 8200/40k, loss 3.6444 (−0.038 this window),
  probe 8.64@8000 (low **8.54@6000**, sub-10 ×7; K1 gate ≤12.0944 by
  10k — formal crossing at the **@10000 probe ~06:0xZ**, margin
  wide), 2.192 s/step, vram 67.07 ≤ 71; endpoint ~08-08.
- local draws10_t1 — 10592/25800, window 40.0 f/min, cumulative 32.8
  f/min → **~13.1 h total, INSIDE the 24 GPU-h gate**; boundary
  ~13:0x–13:3xZ → frozen reads.

**Steering**: none (`read` clean at boot and checkpoint; owner asleep
since 00:58Z).

**Done**: **#4 attachment-screen pre-reg POSTED**
([post](posts/2026-08-07-prereg-molmo2-attach-screen.md), this
commit) — the queue wanted it before the endpoint so the attachment
decision is executable when it opens. Two arms, the seam the ONLY
contrast: **F** (hard-frozen trunk, our default = "extreme KI") vs
**K** (KI-joint: phase-1 CE objective continuing verbatim at
`backbone-text-lr 2e-5` + stop-grad seam, α=1 fixed — no tuning, per
KI); naive joint NOT re-measured. Matched 10k steps / eff-48 /
sequential on the box, F first. Surface held constant: residual
conditioning with the molmo2 tap rule pinned (gemma's rule is
KV-share-structural and doesn't transfer) — 12 taps @ stride 3,
layers 2,5,…,35, expert depth 12 h1024; the depth-of-reads dial (#4
arm 1) stays open, explicitly NOT measured. Frozen reads: Δ_seam
paired per-frame CI on panel_v2 heun30/draws1/stable; K trunk-drift
diagnostic (greedy AR panel vs the 40k endpoint number, band 0.3) as
the language-following analog; frozen decision rule — ties → frozen
default stands. Gates: vram ≤71, K1-style probe kill (phase-1 curve
+3.0 at ≥5k), 70 GPU-h ceiling with matched 5k downshift
(draws_rate_gate mechanization pattern). Instrument does NOT exist
yet — queued oracle-gated (molmo2 residual exports + guard lift,
seam stop-grad, joint CE+flow with α-edge oracles); **#20 activation
checkpointing is a hard K prerequisite** (phase 1 already at 67/71
GiB with no expert). Also: `posts/index.md` had drifted 16 posts
behind SUMMARY.md (everything since mid-08-06) — regenerated in
SUMMARY order. check.py 410 passed.

**Next** (`queue_cli.py next`): #4 attach-screen instrument (CPU),
then #20 activation checkpointing; molmo2 **@10000 K1 gate crossing
~06:0xZ** — babysit surfaces it, judge then; draws10_t1 boundary
~13:0x–13:3xZ → frozen reads; screen execution opens at endpoint →
#19 box obligations → instruments + #20 → attachment-decision owner
steer window; arm A img280 + box-home-sweep HELD.

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
