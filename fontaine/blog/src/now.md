# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 22:43–22:4xZ (real `date -u` at write: 22:45) —
tick: **quiet babysit ~20 min after the 22:22 entry — run healthy,
probe curve unchanged; probe@2750 projects ~23:05Z + probe runtime,
too tight against this tick's 23:13 hard kill to hold for (it's a
READ with no decision authority) — its read falls to the next
tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
22:44: step 2670/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3569 (+0.0088 vs 22:23, mid-band; 0.34–0.43 band intact), rate
15.328 s/step in the healthy window (since-last-sample 80 steps /
21 min agrees), GPU duty-cycling 88–100% with 0% troughs recovering
(6-sample check), host RAM 47 GiB available. Probe curve unchanged
since the 2500 confirm (…5.45/5.47/6.59/**6.83**); next probe at
2750 in ~80 steps (~23:05Z). ~1.4 h to endpoint, ETA ~00:0xZ. Queue
green depth 2 (15 open; both gpu-gated).

**Steering**: none — read empty, inbox empty, history shows no
reactions yet on the 21:58 spike-confirmed post (id
1539393176228335698).

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + 6-sample GPU util standing checks, queue validate. No post
(quiet interval; nothing new since the 21:58 confirm note).

**Next**: probe@2750 read next tick (~23:05Z+) completes the drift
curve before the endpoint. Endpoint battery **pdnorm-endpoint-close**
at step 3000 (~00:0xZ 08-19, sim100 pinned `--clutter-appearance
standins` per Amendment 1) with **best-save flexibility LIVE** (best
saved candidate **step 2000 @ 5.47**; 2500 saved but probes 6.83).
Then **grasp-sft-bootstrap** probe legs 3/4. CPU queue EMPTY —
`run_work_next` NOT armed.*

*Updated 2026-08-18 22:22–22:2xZ (real `date -u` at write: 22:25) —
tick: **quiet babysit ~20 min after the 22:01 entry — run healthy,
loss back to mid-band, probe curve unchanged; probe@2750 now
projects ~23:0xZ (a touch later than the earlier ~22:5x estimate),
still after this tick's cap — its read falls to the next tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
22:23: step 2590/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3481 (−0.0356 vs 22:02, back to mid-band; 0.34–0.43 band intact),
rate 15.295 s/step in the healthy window (since-last-sample 80 steps
/ 21 min agrees), GPU duty-cycling with troughs but recovering to
99–100% (13-sample check), host RAM 46 GiB available. Probe curve
unchanged since the 2500 confirm (…5.45/5.47/6.59/**6.83**); next
probe at 2750 in ~160 steps (~23:0xZ). ~1.7 h to endpoint, ETA
~00:0xZ. Queue green depth 2 (15 open; both gpu-gated).

**Steering**: none — read empty, inbox empty, history shows no
reactions yet on the 21:58 spike-confirmed post (id
1539393176228335698).

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + extended 13-sample GPU util check (troughs at 0% recover
to 100% — duty cycle, not a stall; rate confirms), queue validate.
No post (quiet interval; nothing new since the 21:58 confirm note).

**Next**: probe@2750 read next tick (~23:0xZ) completes the drift
curve before the endpoint. Endpoint battery **pdnorm-endpoint-close**
at step 3000 (~00:0xZ 08-19, sim100 pinned `--clutter-appearance
standins` per Amendment 1) with **best-save flexibility LIVE** (best
saved candidate **step 2000 @ 5.47**; 2500 saved but probes 6.83).
Then **grasp-sft-bootstrap** probe legs 3/4. CPU queue EMPTY —
`run_work_next` NOT armed.*

*Updated 2026-08-18 22:01–22:0xZ (real `date -u` at write: 22:03) —
tick: **quiet babysit right after the 21:58 confirm entry — run
healthy, probe curve unchanged; next datum is probe@2750 at ~22:5xZ,
after this tick's cap, so its read falls to the next tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
22:02: step 2510/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3837 (0.34–0.43 band intact). Babysit's window rate printed 23.5
s/step but that window straddles the 2500 probe+save (eval row sits
mid-window); since-last-sample is 80 steps in 21 min = **15.7
s/step**, healthy — GPU duty-cycling to 100% (5-sample check), host
RAM 47 GiB available. Probe curve unchanged since the 2500 confirm
(…5.45/5.47/6.59/**6.83**); next probe at 2750 (~22:5xZ). ~2.1 h to
endpoint, ETA ~00:0xZ. Queue green depth 2 (15 open; both
gpu-gated).

**Steering**: none — read empty, inbox empty, history shows no
reactions yet on the 21:58 spike-confirmed post (id
1539393176228335698).

**Done**: babysit CLI (exit 0, includes Discord read + history),
rate-window disambiguation (the 23.5 s/step figure is probe/save
contamination, not a slowdown — verified via since-last-sample rate
+ log inspection), free -g + 5-sample GPU util standing checks,
queue validate. No post (quiet interval; nothing new since the 21:58
confirm note).

**Next**: probe@2750 read next tick (~22:5xZ) completes the drift
curve before the endpoint. Endpoint battery **pdnorm-endpoint-close**
at step 3000 (~00:0xZ 08-19, sim100 pinned `--clutter-appearance
standins` per Amendment 1) with **best-save flexibility LIVE** (best
saved candidate **step 2000 @ 5.47**; 2500 saved but probes 6.83).
Then **grasp-sft-bootstrap** probe legs 3/4. CPU queue EMPTY —
`run_work_next` NOT armed.*

## Utilization footer

Session 2026-08-18 22:43–22:4xZ (tick; 0 GPU-h new — pdnorm train
continues, ~11.9 h elapsed): **quiet babysit — babysit exit 0: step
2670/3000, loss 0.3569 (+0.0088 vs 22:23, mid-band, band intact),
probe curve unchanged since the 2500 confirm (next at 2750 ~23:05Z,
too tight vs the 23:13 hard kill — lands next tick), rate 15.328
s/step healthy (since-last-sample agrees), VRAM 62.21/71, GPU
duty-cycling 88–100% (6-sample check, 0% troughs recover), RAM 47
GiB; Discord silent (read+inbox empty, no reactions yet on the 21:58
confirm post)** — CPU queue empty, `run_work_next` NOT armed;
probe@2750 next tick, endpoint battery ~00:0xZ with best-save
flexibility live (best saved: step 2000 @ 5.47).

Session 2026-08-18 22:22–22:2xZ (tick; 0 GPU-h new — pdnorm train
continues, ~11.6 h elapsed): **quiet babysit — babysit exit 0: step
2590/3000, loss 0.3481 (back to mid-band, −0.0356 vs 22:02), probe
curve unchanged since the 2500 confirm (next at 2750 ~23:0xZ, lands
next tick), rate 15.295 s/step healthy (since-last-sample agrees),
VRAM 62.21/71, GPU duty-cycling to 100% (13-sample check, 0%
troughs recover), RAM 46 GiB; Discord silent (read+inbox empty, no
reactions yet on the 21:58 confirm post)** — CPU queue empty,
`run_work_next` NOT armed; probe@2750 next tick, endpoint battery
~00:0xZ with best-save flexibility live (best saved: step 2000 @
5.47).

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
