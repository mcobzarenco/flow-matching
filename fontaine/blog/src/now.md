# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 23:46–23:5xZ (real `date -u` at write: 23:50) —
tick: **final-stretch babysit — endpoint (~00:07Z) lands inside this
tick's window but the endpoint battery is hours of work, far past the
00:17Z hard kill — `run_work_next` ARMED; the chained 4-h work
session owns the endpoint: final probe + save, then the
pdnorm-endpoint-close battery with best-save flexibility live.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
23:47: step 2920/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3137 (−0.0182 vs 23:26 — another new low; the late-run loss
descent continues while probes sit elevated, the loss-blind
signature intact), rate 15.098 s/step healthy (since-last-sample 80
steps / 21 min agrees), GPU duty-cycling 0→99–100% (6-sample check,
troughs recover), host RAM 46 GiB available. ~80 steps to endpoint,
ETA ~00:07Z. Queue green depth 2 (15 open; both gpu-gated).

**Steering**: two new 👍 reactions in history — on the 21:58
spike-confirm post (id 1539393176228335698) and the 23:05 retrace
post (id 1539410046666936431); owner agreement with both reads,
recorded. Read + inbox empty otherwise.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + 6-sample GPU util standing checks, queue validate,
`run_work_next` armed. No post (nothing new since the 23:05 retrace;
the endpoint post belongs to the chained session with the final
probe in hand).

**Next**: chained work session catches step 3000 (~00:07Z 08-19):
final probe + save, then the **pdnorm-endpoint-close** battery
(sim100 pinned `--clutter-appearance standins` per Amendment 1) with
**best-save flexibility LIVE**: candidates endpoint-3000 (probe TBD)
vs **step 2000 @ 5.47**. Then **grasp-sft-bootstrap** probe legs
3/4.*

*Updated 2026-08-18 23:25–23:3xZ (real `date -u` at write: 23:28) —
tick: **quiet babysit ~20 min after the 23:04 entry — run healthy in
its final stretch, probe curve complete (no probe boundary left
before 3000); endpoint projects ~00:07Z, past this tick's 23:55 hard
kill — the endpoint battery (pdnorm-endpoint-close) falls to the
next tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
23:26: step 2840/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3319 (+0.0018 vs 23:05's new low 0.3301 — holding at the low end,
below the old 0.34–0.43 band), rate 15.172 s/step healthy
(since-last-sample 80 steps / 21 min agrees), GPU duty-cycling 0–100%
with troughs recovering (6-sample check), host RAM 45 GiB available.
Probe curve final: 6.11/5.72/5.62/**5.45/5.47**/6.59/6.83/**6.32** —
next datum is the endpoint itself (~160 steps, ~40 min). Queue green
depth 2 (15 open; both gpu-gated).

**Steering**: none — read surfaced only our own 23:05 probe post
(cursor catch-up), inbox empty; history shows no new reactions (the
👍 on the 21:01 post was already recorded).

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + 6-sample GPU util standing checks, queue validate. No post
(quiet interval; nothing new since the 23:05 retrace post).

**Next**: endpoint at step 3000 (~00:07Z 08-19) — final probe + save,
then the **pdnorm-endpoint-close** battery (sim100 pinned
`--clutter-appearance standins` per Amendment 1) with **best-save
flexibility LIVE**: candidates are endpoint-3000 (probe TBD) vs
**step 2000 @ 5.47**. Then **grasp-sft-bootstrap** probe legs 3/4.
CPU queue EMPTY — `run_work_next` NOT armed.*

*Updated 2026-08-18 23:04–23:1xZ (real `date -u` at write: 23:08) —
tick: **probe@2750 read landed — 6.32, a partial retrace of the
spike (6.83@2500 → 6.32@2750, train-probe in lockstep 6.75 → 6.10);
still well above the 1750–2000 plateau, best-saved candidate
unchanged (step 2000 @ 5.47). Posted in-channel (id
1539410046666936431). Mid-run probe curve is now complete — next
datum is the endpoint itself.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
23:05: step 2760/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3301 (−0.0268 vs 22:44 — a new low, first dip below the recent
0.34–0.43 band; favorable direction, not an anomaly), window rate
19.0 s/step is 2750-probe-contaminated (same pattern as at 2500) —
since-last-sample is 90 steps / 21 min ≈ **14 s/step**, healthy; GPU
100%, host RAM 46 GiB available. Probe curve final mid-run shape:
6.11/5.72/5.62/**5.45/5.47**/6.59/6.83/**6.32** — drift is real but
non-monotone. ~1 h to endpoint, ETA ~00:0xZ. Queue green depth 2 (15
open; both gpu-gated).

**Steering**: none new — read empty, inbox empty; history's 👍 on the
21:01 spike-watch post was already recorded in an earlier tick, no
reactions yet on the 21:58 confirm post (id 1539393176228335698).

**Done**: babysit CLI (exit 0, includes Discord read + history),
probe@2750 read + train-probe lockstep check (grepped the train log
directly), free -g standing check, queue validate, in-channel retrace
post (id 1539410046666936431).

**Next**: endpoint at step 3000 (~00:0xZ 08-19) — final probe + save,
then the **pdnorm-endpoint-close** battery (sim100 pinned
`--clutter-appearance standins` per Amendment 1) with **best-save
flexibility LIVE**: candidates are endpoint-3000 (probe TBD) vs
**step 2000 @ 5.47**; 2500 saved but probes 6.83, 2750 not a save
boundary. Then **grasp-sft-bootstrap** probe legs 3/4. CPU queue
EMPTY — `run_work_next` NOT armed.*

## Utilization footer

Session 2026-08-18 23:46–23:5xZ (tick; 0 GPU-h new — pdnorm train
continues, ~12.9 h elapsed): **final-stretch babysit — babysit exit
0: step 2920/3000, loss 0.3137 (−0.0182 vs 23:26, another new low
while probes sit elevated — loss-blind signature intact), rate
15.098 s/step healthy (since-last-sample agrees), VRAM 62.21/71, GPU
duty-cycling 0→99–100% (6-sample check, troughs recover), RAM 46
GiB; two new 👍 reactions recorded (21:58 confirm + 23:05 retrace
posts — owner agreement), read+inbox otherwise empty** — endpoint
~00:07Z lands inside the window but the battery exceeds the 00:17Z
hard kill → `run_work_next` ARMED; the chained 4-h work session owns
the endpoint (final probe + save) and the pdnorm-endpoint-close
battery with best-save flexibility live (best saved: step 2000 @
5.47).

Session 2026-08-18 23:25–23:3xZ (tick; 0 GPU-h new — pdnorm train
continues, ~12.5 h elapsed): **quiet babysit — babysit exit 0: step
2840/3000, loss 0.3319 (+0.0018 vs 23:05's new low 0.3301, holding
at the low end), probe curve complete (no probe boundary left before
3000; next datum is the endpoint), rate 15.172 s/step healthy
(since-last-sample agrees), VRAM 62.21/71, GPU duty-cycling 0–100%
(6-sample check, troughs recover), RAM 45 GiB; Discord silent (read
surfaced only our own 23:05 post, inbox empty, no new reactions)** —
CPU queue empty, `run_work_next` NOT armed; endpoint ~00:07Z lands
past this tick's 23:55 hard kill — next tick owns the
pdnorm-endpoint-close battery with best-save flexibility live (best
saved: step 2000 @ 5.47).

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
