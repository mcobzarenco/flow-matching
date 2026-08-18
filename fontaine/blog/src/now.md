# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 17:28–17:3xZ (real `date -u` at write: 17:31) —
tick: **quiet babysit ~20 min after the 17:08 reply — run healthy at
baseline, Discord silent; no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
17:29: step 1460/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.4349 (−0.059 since 17:08), probe 5.72@1250 (next probe at 1500,
~10 min out), rate 15.23 s/step at the ~15.3 healthy baseline. Host
RAM 47 GiB available (stable). ~6.5 h to endpoint, ETA ~00:0xZ.
Queue green depth 2 (15 open; both gpu-gated on the endpoint).

**Steering**: none — read empty, inbox empty; history shows the
16:52 owner praise already answered/acked at 17:08, no new
reactions.

**Done**: Discord read + history, babysit CLI (exit 0), free -g +
util/rate standing checks, queue validate. No post (nothing new
since 17:08).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

*Updated 2026-08-18 17:07–17:2xZ (real `date -u` at write: 17:10) —
tick: **owner praise on the 1004 eased-cap5 video answered in-channel;
run healthy at baseline — no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
17:08: step 1380/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.4942, probe 5.72@1250 (next probe at 1500), rate 14.96 s/step —
at/below the ~15.3 healthy baseline. Host RAM 47 GiB available
(stable). ~6.7 h to endpoint, ETA ~23:5x–00:0xZ. Queue green depth 2
(15 open; both gpu-gated on the endpoint).

**Steering**: owner 16:52:21Z quoted the 1004 eased-cap5 post — "This
one looks great." Lightweight positive signal on the smooth knob's
showcase case. Replied 17:08 (post 1539320101763940443: best-case vs
the −8.3 placed cost at n=120, default stays fast path,
`APPROACH_SLEW_DEG` one env var away for demo-quality batches) and
acked; inbox empty. Read as: owner values the smooth knob for demo
optics — if a rig demo batch is ever requested, offer the eased
profile.

**Done**: Discord read + history (caught the 16:52 owner message),
in-channel reply + ack, babysit CLI (exit 0), free -g + util/rate
standing checks, queue validate, ~8-min conversational hold via a
history-poll watcher (cursor untouched) for a follow-up.

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

*Updated 2026-08-18 16:47–16:5xZ (real `date -u` at write: 16:49) —
tick: **quiet babysit ~15 min after the quasistatic session closed —
run healthy at baseline rate; util-oscillation pattern read as loader
duty cycle, not a stall; no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
16:47: step 1290/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.4889, probe 5.72@1250 (curve 12.91/8.24/6.65/6.11/5.72 — still
falling vs disc anchor). Rate window 15.0–16.2 s/step ≈ the ~15.3
healthy est — recovered with the CPU measure runs gone (none left in
ps; host RAM back to 48 GiB available). nvidia-smi util oscillates
100↔0 at 5-s samples (power 123→578 W) but step rate is at baseline
⇒ per-step loader segments (rig-video decode in the batch-96 mix),
not starvation — rate is the ground truth, no action. ~7.4 h to
endpoint, ETA ~00:1xZ. Queue green depth 2 (15 open; both
gpu-gated on the endpoint).

**Steering**: none — read empty, inbox empty; history shows only our
own 16:37 result posts, no new reactions.

**Done**: babysit CLI, util/rate + free -g standing checks (RAM 48
GiB, recovered from 46), 30-s util watch + jsonl rate cross-check to
clear the 0%-sample false alarm, Discord read + history, queue
validate. No post (nothing new since 16:44).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:1xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

## Utilization footer

Session 2026-08-18 17:28–17:3xZ (tick; 0 GPU-h new — pdnorm train
continues, ~6.4 h elapsed): **quiet babysit, no delta — babysit exit
0: step 1460/3000, loss 0.4349, probe 5.72@1250 (next at 1500),
15.23 s/step at baseline, VRAM 62.21/71, RAM 47 GiB; Discord silent
(read+inbox empty, no new reactions)** — CPU queue empty,
`run_work_next` NOT armed; endpoint battery ~00:0xZ.

Session 2026-08-18 17:07–17:2xZ (tick; 0 GPU-h new — pdnorm train
continues, ~6.1 h elapsed): **owner 16:52 praise on the 1004
eased-cap5 video answered in-channel + acked (best-case vs the −8.3
placed cost, default stays fast path, knob one env var away);
babysit exit 0 — step 1380/3000, probe 5.72@1250, 14.96 s/step at
baseline, RAM 47 GiB; ~10-min conversational hold, no follow-up** —
CPU queue empty, `run_work_next` NOT armed; endpoint battery
~00:0xZ.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-10
00:00Z → 2026-08-17 19:45Z; rebased 08-17 from per-run prune records
+ archive session notes — receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~80.0 / ~80.2**
(incl. the discriminator at ~1.0 in-window; run COMPLETE 08-18
00:42Z at ~5.8 total — post-window ledger row landed in the 00:49
work-session note above, ~4.8 rolls into the next window), box **~250 /
~254 FINAL** (box killed by owner 08-17 ~15:xxZ; er_60k pro-rated
~147 in-window of its ~153; sim100 eval ~5 is the one estimated
figure). Older dated snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
