# Now
















*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 21:40–22:0xZ (real `date -u` at write: 22:00) —
tick: **probe@2500 read in-session (held open for the ~21:58Z datum)
— spike CONFIRMED: eval 6.59@2250 → 6.83@2500, train-probe in
lockstep 6.44 → 6.75, loss fully healthy. Sustained loss-blind
drift, not a transient. Still a READ per pre-reg; best-save
flexibility now LIVE at endpoint choice.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
21:41: step 2430/3000 (2500+ by write), 5 procs, VRAM 62.21/71 gate
stable, loss 0.3702@2500 (0.35–0.43 band intact, grad_norm 2.2–2.3),
rate 15.07–15.09 s/step in the 15.0–16.2 healthy window, GPU 99%.
**Probe eval_chunk_mae 6.83@2500** (curve
…5.45/5.47/6.59/**6.83**) — two consecutive elevated points, rise
even steepened (+0.24 on top of +1.11); train-side probe rose in
lockstep both times, so this is the mixed-v2-class loss-blind drift,
confirmed. ~2.1 h to endpoint, ETA ~00:0xZ. Queue green depth 2 (15
open; both gpu-gated).

**Steering**: none new — read empty, inbox empty, history shows no
new reactions beyond the already-recorded 👍 on the 21:01 watch post.

**Done**: babysit CLI (exit 0, includes Discord read + history),
queue validate, **in-session hold 21:43–21:58 for the probe@2500
datum** (charter §6 judgment call — the confirm/deny was worth the
hold), **Discord post** (spike-confirmed note, id
1539393176228335698).

**Next**: endpoint battery **pdnorm-endpoint-close** at step 3000
(~00:0xZ 08-19, sim100 pinned `--clutter-appearance standins` per
Amendment 1) now runs with **best-save flexibility LIVE**: endpoint
choice re-opens to the best-probing saved checkpoint — best saved
candidate **step 2000 @ 5.47** (2500 saved but probes 6.83; final
probe at 2750 ~22:5xZ, endpoint 3000 read completes the curve).
Then **grasp-sft-bootstrap** probe legs 3/4. CPU queue EMPTY —
`run_work_next` NOT armed.*

*Updated 2026-08-18 21:19–21:2xZ (real `date -u` at write: 21:21) —
tick: **quiet babysit ~20 min after the 20:58 spike entry — run
healthy, loss flat; owner 👍 on the spike watch note (agreement with
the READ/no-action call). Probe@2500 lands ~21:58Z, after this
tick's cap — confirm/deny read falls to the next tick.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
21:20: step 2350/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3544 (−0.0002 vs 20:59, flat — the 20:38 top-of-band wiggle stays
resolved), rate 15.06 s/step in the 15.0–16.2 healthy window, GPU
duty-cycling to 100% (5-sample check), host RAM 46 GiB available
(stable). Probe curve unchanged since the 2250 spike
(…5.45/5.47/**6.59**); next probe at 2500 in ~150 steps (~21:58Z) is
the spike's confirm/deny — also a save boundary. ~2.7 h to endpoint,
ETA ~00:0xZ. Queue green depth 2 (15 open; both gpu-gated).

**Steering**: **owner 👍x1 on the 21:01 spike watch post** (id
1539378620609339457) — lightweight agreement with the pre-reg
READ/no-action stance and the endpoint-battery kill/keep authority;
recorded per the reaction-steering rule, no reply owed (reaction,
not message). Read/inbox otherwise empty.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + util/rate standing checks (GPU duty cycle verified by
5-sample poll), queue validate. No post (quiet mid-run interval; the
2500 datum is what's worth posting, and it lands next tick).

**Next**: probe@2500 read next tick (~21:58Z+) — if elevation holds,
best-save flexibility goes live at endpoint choice (best saved
candidate: step 2000 @ 5.47). Endpoint battery
**pdnorm-endpoint-close** unchanged at step 3000 (~00:0xZ 08-19,
sim100 pinned `--clutter-appearance standins` per Amendment 1), then
**grasp-sft-bootstrap** probe legs 3/4. CPU queue EMPTY —
`run_work_next` NOT armed.*

*Updated 2026-08-18 20:58–21:0xZ (real `date -u` at write: 21:01) —
tick: **probe SPIKE at 2250 — 5.47→6.59 (+1.11), first real anomaly
of the run; loss trace fully healthy, so this is the loss-blind
drift signature. READ per pre-reg, no action; posted in-channel.
Confirm/deny at probe@2500 (~21:5xZ).***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
20:59: step 2270/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3546 (−0.0741 vs 20:38 — the 20:38 top-of-band wiggle resolved as
noise), rate 14.84 s/step in the healthy window, host RAM stable.
**Probe eval_chunk_mae 6.59@2250** (curve
12.91/8.24/6.65/6.11/5.72/5.62/5.45/5.47/**6.59** — +1.11 after the
2000 plateau). Dug into `train_log.jsonl` + unit journal: train-side
probe MAE rose in lockstep (5.41→6.44) so it is NOT eval-set noise;
training loss 0.35–0.43 the whole 1950–2280 window, zero spikes,
grad_norm 2.0–3.1 normal — invisible-to-loss, the mixed-v2 8x drift
class the pre-reg drift-guard note anticipated (that guard's own
1000−500 window passed long ago at −2.13). One point, could still be
a transient; probe@2500 (~21:5xZ, also a save boundary) is the
confirm/deny. ~3.0 h to endpoint, ETA ~00:0xZ. Queue green depth 2
(15 open; both gpu-gated).

**Steering**: none — read empty, inbox empty; history shows the
16:52 owner praise already answered/acked at 17:08, no new
reactions.

**Done**: babysit CLI (exit 0, includes Discord read + history),
loss/grad-norm spike scan of the 1950–2280 window, per-dataset
breakdown check (probe rows carry only pooled MAE — slice breakdown
lands with the endpoint battery), queue validate, **Discord post**
(probe-spike watch note, id 1539378620609339457).

**Next**: probe@2500 read next tick (~21:5xZ) — if elevation holds,
the reserved best-save flexibility goes live at endpoint choice
(best saved candidate: step 2000 @ 5.47; 1750's 5.45 was not a save
boundary). Endpoint battery **pdnorm-endpoint-close** unchanged at
step 3000 (~00:0xZ 08-19, sim100 pinned `--clutter-appearance
standins` per Amendment 1), then **grasp-sft-bootstrap** probe legs
3/4. CPU queue EMPTY — `run_work_next` NOT armed.*

## Utilization footer

Session 2026-08-18 21:40–22:0xZ (tick; 0 GPU-h new — pdnorm train
continues, ~11.0 h elapsed): **probe@2500 read in-session (held
21:43–21:58 for the datum) — spike CONFIRMED: eval 6.59→6.83,
train-probe lockstep 6.44→6.75, loss healthy (0.3702@2500,
grad_norm 2.3). Sustained loss-blind drift; READ per pre-reg,
best-save flexibility LIVE at endpoint (best saved: step 2000 @
5.47). Posted in-channel (id 1539393176228335698). Run otherwise
healthy: 15.07 s/step, VRAM 62.21/71, ETA ~00:0xZ; Discord
otherwise silent** — CPU queue empty, `run_work_next` NOT armed;
probe@2750 + endpoint battery ~00:0xZ own the next reads.

Session 2026-08-18 21:19–21:2xZ (tick; 0 GPU-h new — pdnorm train
continues, ~10.3 h elapsed): **quiet babysit — babysit exit 0: step
2350/3000, loss 0.3544 (flat), probe curve unchanged since the 2250
spike (next at 2500 ~21:58Z, lands next tick), 15.06 s/step in the
healthy window, VRAM 62.21/71, GPU duty-cycling to 100%, RAM 46 GiB;
owner 👍 on the 21:01 spike watch post (agreement with READ/no-action,
recorded), read+inbox otherwise empty** — CPU queue empty,
`run_work_next` NOT armed; probe@2500 confirm/deny next tick,
endpoint battery ~00:0xZ.

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
