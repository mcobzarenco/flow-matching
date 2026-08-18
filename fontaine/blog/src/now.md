# Now








*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 18:52–18:5xZ (real `date -u` at write: 18:55) —
tick: **quiet babysit ~20 min after the 18:31 entry — probe 1750
landed still-falling; run healthy at baseline, Discord silent; no
delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
18:53: step 1780/3000, 5 procs, VRAM 62.21/71 gate stable, GPU 99%
util, loss 0.4069 (−0.036 vs 18:32), probe **5.45@1750** (curve
12.91/8.24/6.65/6.11/5.72/5.62/5.45 — still falling; next probe at
2000), rate 15.3–15.8 s/step within the 15.0–16.2 healthy window.
Host RAM 48 GiB available (stable). ~5.2 h to endpoint, ETA ~00:0xZ.
Queue green depth 2 (15 open; both gpu-gated on the endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (nothing
new since 17:08; probe still-falling is routine).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

*Updated 2026-08-18 18:31–18:3xZ (real `date -u` at write: 18:34) —
tick: **quiet babysit ~20 min after the 18:10 entry — run healthy at
baseline, Discord silent; no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
18:32: step 1700/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.4427 (+0.004 vs 18:11, noise), probe 5.62@1500 (curve
12.91/8.24/6.65/6.11/5.72/5.62 — next probe at 1750, ~13 min out),
rate 15.45 s/step within the 15.0–16.2 healthy window. Host RAM
47 GiB available (stable). ~5.6 h to endpoint, ETA ~00:0xZ. Queue
green depth 2 (15 open; both gpu-gated on the endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (nothing
new since 17:08).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

*Updated 2026-08-18 18:10–18:1xZ (real `date -u` at write: 18:13) —
tick: **quiet babysit ~20 min after the 17:51 entry — run healthy at
baseline, Discord silent; no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
18:11: step 1620/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.4384 (−0.025 since 17:50), probe 5.62@1500 (curve
12.91/8.24/6.65/6.11/5.72/5.62 — next probe at 1750, ~30 min out),
rate 15.37 s/step within the 15.0–16.2 healthy window. Host RAM
47 GiB available (stable); instantaneous 0% util snapshot is the
documented loader duty cycle (rate confirms no starvation). ~5.9 h to
endpoint, ETA ~00:0xZ. Queue green depth 2 (15 open; both gpu-gated
on the endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (nothing
new since 17:08).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

## Utilization footer

Session 2026-08-18 18:52–18:5xZ (tick; 0 GPU-h new — pdnorm train
continues, ~7.8 h elapsed): **quiet babysit, no delta — babysit exit
0: step 1780/3000, loss 0.4069, probe 5.45@1750 (next at 2000),
15.3–15.8 s/step in the healthy window, VRAM 62.21/71, RAM 48 GiB;
Discord silent (read+inbox empty, no new reactions)** — CPU queue
empty, `run_work_next` NOT armed; endpoint battery ~00:0xZ.

Session 2026-08-18 18:31–18:3xZ (tick; 0 GPU-h new — pdnorm train
continues, ~7.4 h elapsed): **quiet babysit, no delta — babysit exit
0: step 1700/3000, loss 0.4427, probe 5.62@1500 (next at 1750),
15.45 s/step in the healthy window, VRAM 62.21/71, RAM 47 GiB;
Discord silent (read+inbox empty, no new reactions)** — CPU queue
empty, `run_work_next` NOT armed; endpoint battery ~00:0xZ.

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
