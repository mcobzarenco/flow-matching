# Now













*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 20:38–20:4xZ (real `date -u` at write: 20:41) —
tick: **quiet babysit ~20 min after the 20:16 entry — run healthy,
Discord silent; loss wiggled to the top of its noise band (watch
item, no action).***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
20:38: step 2180/3000, 5 procs, VRAM 62.21/71 gate stable, GPU 100%
util (in-CLI 0% snapshot = documented loader duty cycle; direct
re-check 100%), loss 0.4287 (+0.0455 vs 20:17 — top of the recent
0.37–0.41 band, largest single-interval wiggle so far; watch, not
anomaly), probe 5.47@2000 (next at 2250, ~18 min out — the
plateau-real? datum lands next tick), rate 15.31 s/step within the
15.0–16.2 healthy window. Host RAM 47 GiB available (stable). ~3.5 h
to endpoint, ETA ~00:0x–00:1xZ. Queue green depth 2 (15 open; both
gpu-gated on the endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (quiet
mid-run interval; loss wiggle is intra-band).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. Next tick
reads probe@2250 (plateau-real?) and re-checks the loss band. CPU
queue EMPTY — `run_work_next` NOT armed; routine tick babysits own
the interim.*

*Updated 2026-08-18 20:16–20:1xZ (real `date -u` at write: 20:17) —
tick: **quiet babysit ~20 min after the 19:56 entry — run healthy at
baseline, Discord silent; no delta.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
20:17: step 2100/3000, 5 procs, VRAM 62.21/71 gate stable, GPU 99%
util, loss 0.3832 (+0.0102 vs 19:56, noise), probe 5.47@2000 (curve
12.91/8.24/6.65/6.11/5.72/5.62/5.45/5.47 — plateau holding; next
probe at 2250, ~38 min out), rate 15.21 s/step within the 15.0–16.2
healthy window. Host RAM 47 GiB available (stable). ~3.8 h to
endpoint, ETA ~00:0xZ. Queue green depth 2 (15 open; both gpu-gated
on the endpoint).

**Steering**: none — read empty, inbox empty; history shows the 16:52
owner praise already answered/acked at 17:08, no new reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks, queue validate. No post (quiet
mid-run interval; probe 2250 will say whether the 2000 plateau is
real).

**Next**: unchanged — **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1), then **grasp-sft-bootstrap** probe legs 3/4. If the
probe stays flat at 2250/2500 that strengthens best-save flexibility
the drift-guard note already reserves. CPU queue EMPTY —
`run_work_next` NOT armed; routine tick babysits own the interim.*

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

## Utilization footer

Session 2026-08-18 20:38–20:4xZ (tick; 0 GPU-h new — pdnorm train
continues, ~9.4 h elapsed): **quiet babysit — babysit exit 0: step
2180/3000, loss 0.4287 (top of the noise band, watch item), probe
5.47@2000 (next at 2250), 15.31 s/step in the healthy window, VRAM
62.21/71, GPU 100%, RAM 47 GiB; Discord silent (read+inbox empty, no
new reactions)** — CPU queue empty, `run_work_next` NOT armed;
endpoint battery ~00:0xZ.

Session 2026-08-18 20:16–20:1xZ (tick; 0 GPU-h new — pdnorm train
continues, ~9.0 h elapsed): **quiet babysit, no delta — babysit exit
0: step 2100/3000, loss 0.3832, probe 5.47@2000 (plateau holding;
next at 2250), 15.21 s/step in the healthy window, VRAM 62.21/71,
GPU 99%, RAM 47 GiB; Discord silent (read+inbox empty, no new
reactions)** — CPU queue empty, `run_work_next` NOT armed; endpoint
battery ~00:0xZ.


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
