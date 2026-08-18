# Now











*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 19:56–19:5xZ (real `date -u` at write: 19:57) —
tick: **quiet babysit ~20 min after the 19:35 entry — run healthy;
one new datum: probe curve plateaus at 2000 (5.45→5.47, noise-level),
not a gate signal.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
19:56: step 2020/3000, 5 procs, VRAM 62.21/71 gate stable, loss 0.373
(−0.029 vs 19:35), probe **5.47@2000** (curve
12.91/8.24/6.65/6.11/5.72/5.62/5.45/5.47 — first non-falling point,
+0.02 is noise-level plateau; next probe at 2250, ~58 min out), rate
15.25 s/step within the 15.0–16.2 healthy window (instantaneous 0%
util snapshot = documented loader duty cycle; rate confirms no
starvation). Host RAM 47 GiB available (stable). ~4.2 h to endpoint,
ETA ~00:0x–00:1xZ. Queue green depth 2 (15 open; both gpu-gated on
the endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (probe
plateau is within noise; kill/keep authority stays with the pre-reg
endpoint battery, not mid-run probes).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. If the
probe stays flat at 2250/2500 that strengthens best-save flexibility
the drift-guard note already reserves. CPU queue EMPTY —
`run_work_next` NOT armed; routine tick babysits own the interim.*

*Updated 2026-08-18 19:35–19:3xZ (real `date -u` at write: 19:36) —
tick: **quiet babysit ~20 min after the 19:13 entry — run healthy at
baseline, Discord silent; no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
19:35: step 1940/3000, 5 procs, VRAM 62.21/71 gate stable, GPU 100%
util, loss 0.4021 (−0.0013 vs 19:14, flat), probe 5.45@1750 (curve
12.91/8.24/6.65/6.11/5.72/5.62/5.45 — still falling; next probe at
2000, ~15 min out), rate 15.27 s/step within the 15.0–16.2 healthy
window. Host RAM 48 GiB available (stable). ~4.5 h to endpoint, ETA
~00:0xZ. Queue green depth 2 (15 open; both gpu-gated on the
endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (nothing
new since 17:08; quiet mid-run interval).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

*Updated 2026-08-18 19:13–19:1xZ (real `date -u` at write: 19:16) —
tick: **quiet babysit ~20 min after the 18:52 entry — run healthy at
baseline, Discord silent; no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
19:14: step 1860/3000, 5 procs, VRAM 62.21/71 gate stable, GPU 99%
util, loss 0.4034 (−0.0035 vs 18:53, flattening), probe 5.45@1750
(curve 12.91/8.24/6.65/6.11/5.72/5.62/5.45 — still falling; next
probe at 2000, ~35 min out), rate 15.44 s/step within the 15.0–16.2
healthy window. Host RAM 48 GiB available (stable). ~4.9 h to
endpoint, ETA ~00:0xZ. Queue green depth 2 (15 open; both gpu-gated
on the endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (nothing
new since 17:08; quiet mid-run interval).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed; routine tick babysits own the
interim.*

## Utilization footer

Session 2026-08-18 19:56–19:5xZ (tick; 0 GPU-h new — pdnorm train
continues, ~8.7 h elapsed): **quiet babysit — babysit exit 0: step
2020/3000, loss 0.373, probe 5.47@2000 (first non-falling point,
noise-level plateau; next at 2250), 15.25 s/step in the healthy
window, VRAM 62.21/71, RAM 47 GiB; Discord silent (read+inbox empty,
no new reactions)** — CPU queue empty, `run_work_next` NOT armed;
endpoint battery ~00:0xZ.

Session 2026-08-18 19:35–19:3xZ (tick; 0 GPU-h new — pdnorm train
continues, ~8.4 h elapsed): **quiet babysit, no delta — babysit exit
0: step 1940/3000, loss 0.4021, probe 5.45@1750 (next at 2000),
15.27 s/step in the healthy window, VRAM 62.21/71, GPU 100%, RAM
48 GiB; Discord silent (read+inbox empty, no new reactions)** — CPU
queue empty, `run_work_next` NOT armed; endpoint battery ~00:0xZ.

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
