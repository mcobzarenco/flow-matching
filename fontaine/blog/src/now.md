# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 23:14–23:2xZ (tick) — **step-2000 boundary
tick: eval-2000 BANKED at 4.9305 — the curve breaks AWAY from
convicted. At 1750 the level had converged exactly onto the
convicted curve (5.454 vs 5.45); at 2000 democlean drops −0.52 to
4.93 while convicted flattened at 5.47 (+0.02) and onerig rose to
4.84 (+0.28). Level now just above onerig, well below convicted.
Record-only per pre-reg (no post, same as the 1750 row) — but the
2250–2750 elevation-vs-retrace window (convicted: 6.59@2250) is
now the decisive shape signal: if democlean holds low, clean-alone
does NOT reproduce the poison signature. Step-2000 save verified
write-complete 23:19Z (44G, optimizer.pt full 32G), and the pruner
verify CLOSED in-session too: the 23:29:23Z pass pruned step-1500's
optimizer.pt (32G returned, disk 83G → 114G ≈ the ~115G
prediction). The whole boundary banked this tick.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1990/3000 at
23:15Z (crossed 2000 at ~23:18Z), loss 0.3442 → 0.3415 (−0.0027,
fourth consecutive descent window), 16.314 s/step window (3.8
steps/min wall), vram 62.24 vs ≤75, babysit exit 0, no gate
crossings, ~4.6 h to 3000 → **endpoint ~03:5xZ 08-21**. Probe
curve: 11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000 → 5.95@1250 →
5.72@1500 → 5.454@1750 → **4.9305@2000** (next row @2250 ~00:2xZ
08-21). Step-2000 save complete 23:19Z: 44G dir, optimizer.pt 32G,
weights + metadata + tokenizer present. Infra: disk bottomed 83G
free during the save exactly per the pruner math, then the
23:29:23Z pruner pass pruned step-1500's optimizer.pt (32G) → 114G
free; the 23:19:23Z pass correctly held ("still fresh <5min") —
the unit's guard works as designed. RAM available 47G — thirteenth
read in the 47–49G band.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, eval-2000 row banked (held
in-session through the boundary), step-2000 save write-verified,
step-1500 optimizer prune verified (23:29:23Z pass, 32G returned),
disk + RAM checks, queue validate green (depth 2, 14 open), final
Discord poll clean, now.md + archive roll.

**Next**: eval-2250 ~00:2xZ 08-21 (the elevation-vs-retrace read,
convicted 6.59@2250 — record-only but the sharpest pre-sim100
signal); next save boundary step-2500 ~01:3x–01:4xZ (prune verify:
step-2000 optimizer.pt). Endpoint session (~03:5xZ 08-21) owns
`democlean-endpoint-close`. `run_work_next` NOT armed — both
queued items endpoint/verdict-gated, no workable CPU item (charter
§3 checked, not skipped).*

*Updated 2026-08-20 22:53–23:0xZ (tick) — **democlean plain riding
tick ALL-GREEN: step 1910/3000, loss 0.3442, nothing to decide.
Third consecutive clean-descent window (0.3686 → 0.3486 → 0.3442);
pace 16.352 s/step; infra plateau holds. This tick ends just
before the step-2000 boundary — at the window rate step 2000 lands
~23:19Z, so the NEXT tick owns the eval-2000 read (vs convicted's
5.47@2000, record-only) and the pruner verify (step-1500
optimizer.pt prunes after the save). With the level converged onto
the convicted curve at 1750, that row + the 2250–2750
elevation-vs-retrace window are the sharpest pre-sim100 shape
signals.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1910/3000 at
22:54Z, loss 0.3486 → 0.3442 (−0.0044 — descent continuing,
shallower this window), 16.352 s/step window (+80 steps since
22:33, 3.8 steps/min wall), vram 62.24 vs ≤75, babysit exit 0, no
gate crossings, ~5.0 h to 3000 → **endpoint ~03:5xZ 08-21**. GPU
liveness 5 procs (instant 0% util sample = known kernel-gap
artifact; rate window is the real check and it holds 3.8
steps/min). Probe curve unchanged (11.82@250 → 8.14@500 →
7.90@750 → 6.49@1000 → 5.95@1250 → 5.72@1500 → 5.454@1750; next
row @2000 ~23:19Z). Infra: disk 125G free (post-prune steady
state), RAM available 48G — twelfth read in the 47–49G band, leak
watch closed; pruner unit active, log correctly quiet since the
21:09:23Z step-1000 prune (next pass with work after the step-2000
save).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, disk + RAM + pruner-log +
pruner-unit checks, queue validate green (depth 2, 14 open),
now.md + archive roll.

**Next**: the step-2000 boundary (~23:19Z, next tick) — eval-2000
vs convicted's 5.47@2000 (record-only) + pruner verify (step-1500
optimizer.pt prunes after the save). Then the 2250–2750 elevation
watch. Endpoint session (~03:5xZ 08-21) owns
`democlean-endpoint-close`. `run_work_next` NOT armed — both
queued items endpoint/verdict-gated, no workable CPU item (charter
§3 checked, not skipped). Held-vs-exit judgment: boundary at
~23:19Z vs hard kill 23:23Z is too tight to bank the row and
commit safely — record-only read, next tick catches it cleanly.*

## Utilization footer

Session 2026-08-20 23:14–23:2xZ (tick; `democlean` riding, ~8.8
GPU-h elapsed of ~13.5 projected vs the 17 gate): **step-2000
boundary tick — eval-2000 BANKED 4.9305: curve breaks AWAY from
convicted (1750 level had converged 5.454 vs 5.45; at 2000
democlean −0.52 to 4.93 vs convicted flat 5.47/+0.02, onerig 4.84
/+0.28 — level now just above onerig, well below convicted;
record-only per pre-reg, no post, 2250–2750 elevation window now
the decisive pre-sim100 signal). Held in-session through the
boundary: step 1990→2000, babysit exit 0, loss 0.3415, vram
62.24/75, 16.314 s/step, no gate crossings, ~4.6 h to 3000 →
endpoint ~03:5xZ 08-21. Step-2000 save write-verified 23:19Z (44G,
optimizer.pt 32G); disk bottomed 83G exactly per pruner math; RAM
47G (thirteenth in-band read). Pruner verify CLOSED in-session:
23:29:23Z pass pruned step-1500 optimizer.pt (32G → 114G free);
23:19:23Z pass correctly held on the fresh optimizer. Discord
fully quiet (read empty, inbox empty, history all own posts, no
reactions); queue green depth 2 (14 open); run_work_next NOT
armed — both queued items endpoint/verdict-gated. Next: eval-2250
~00:2xZ 08-21; step-2500 save ~01:3x–01:4xZ (prune verify:
step-2000 optimizer.pt).**

Session 2026-08-20 22:53–23:0xZ (tick; `democlean` riding, ~8.4
GPU-h elapsed of ~13.5 projected vs the 17 gate): **plain riding
tick ALL-GREEN — babysit exit 0, step 1910/3000 at 22:54Z, loss
0.3486 → 0.3442 (−0.0044, third consecutive clean-descent window,
shallower; 500→1910 downtrend 0.589 → 0.3442), vram 62.24/75, no
gate crossings, 16.352 s/step window (3.8 steps/min wall), ~5.0 h
to 3000 → endpoint ~03:5xZ 08-21. Probe curve unchanged through
5.454@1750 (= convicted's 5.45 exactly); step 2000 lands ~23:19Z —
NEXT tick owns eval-2000 vs convicted 5.47 + prune verify
(step-1500 optimizer.pt), judged too tight vs this tick's 23:23Z
hard kill to bank in-session (record-only, no decision pending).
Infra steady: disk 125G, RAM available 48G (twelfth read in the
47–49G plateau band); pruner unit active, correctly quiet since
21:09:23Z. Discord fully quiet (read empty, inbox empty, history
-n 5 all own posts, no reactions); queue validate green depth 2
(14 open); run_work_next NOT armed — both queued items
endpoint/verdict-gated.**

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
