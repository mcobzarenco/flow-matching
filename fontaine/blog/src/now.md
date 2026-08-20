# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 17:59–18:0xZ (tick) — **democlean eighth-tick
poll ALL-GREEN: step 840/3000, loss 0.527, all infra anchors quiet
(disk 146G, RAM plateau a fourth tick, pruner idle-correct).
Nothing to decide — step-1000 (save + eval-1000 + registered drift
read + first real prune) lands ~18:42Z, just after this session's
hard kill; the next tick banks all of it.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 840/3000 at
18:00Z, 17.02 s/step window (+80 steps since 17:39, 3.8 steps/min
wall), loss 0.502 → 0.527 — a small window-over-window uptick well
inside the step-noise band (sequence 0.725 → 0.610 → 0.589 → 0.560
→ 0.543 → 0.502 → 0.527), not a spike. Vram 62.24 vs ≤75, babysit
exit 0, no gate crossings. No new probe row (next is eval-1000 at
the save). Infra: disk 146G free, RAM available 49G — fourth
consecutive tick at the plateau; pruner unit active, log
start-line-only (correct — no save since 500; first real prune
verifies after the step-1000 save). Projections: **step 1000
~18:42Z**, **endpoint ~03:0x–04:1xZ 08-21** (babysit ~10.2 h to
3000 at the window pace; cumulative pace says the earlier end of
that band).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts, no reactions.

**Done** (this tick): babysit poll, loss-uptick noise read, disk +
RAM + pruner-log checks, queue validate green (depth 2, 14 open),
now.md + archive roll.

**Next**: the ~18:4xZ tick banks eval-1000 + the registered drift
read (bar: eval@1000 ≤ 8.44 on our own curve; failure = endpoint
re-opens, not a kill), verifies the pruner log pruned step-500's
optimizer.pt after the save, and re-reads RAM across the save. If
the 1000 row confirms the flattening, post the two-point shape note
in-channel. Endpoint session owns `democlean-endpoint-close`.
`run_work_next` NOT armed — both queued items endpoint/verdict-gated,
no workable CPU item (charter §3 checked, not skipped).*

*Updated 2026-08-20 17:38–17:4xZ (tick) — **democlean seventh-tick
poll ALL-GREEN, and the eval-750 row is banked: 7.90@750 — the first
row ABOVE both anchors (convicted 6.65, onerig 6.73), the probe
curve is flattening where the anchors kept dropping. Record-only as
registered; drift-guard math re-read from the pre-reg says no gate
is threatened (bar is our own Δ(1000−500) ≤ +0.30 → eval@1000 ≤
8.44, we're at 7.90 and still falling). Step-1000 save + drift read
next tick ~18:4xZ.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 760/3000 at
17:39Z, loss 0.543 → 0.502, vram 62.24 vs ≤75, babysit exit 0, no
gate crossings. Window pace 19.11 s/step (+80 steps since 17:18) is
inflated by the eval-750 pause inside the window — cumulative since
launch is ~16.1 s/step, on the onerig-class anchor, no starvation.
**EVAL-750 ROW BANKED (record-only): 7.90@750** vs convicted 6.65 /
onerig 6.73. Shape note: improvement 500→750 was **−0.24** (8.14 →
7.90) vs the anchors' −1.59 / −1.31 — clean-alone's probe curve
started below both anchors (11.82@250), matched them at 500, and is
now flattening early. Not the convicted signature (that's the
2250–2750 elevation) and not a registered read — sim100 at 3000
stays the verdict instrument. **Drift-guard framing verified against
the pre-reg** (posts/2026-08-20-prereg-demos-plus-clean.md §drift
guard): the bar is the run's OWN Δeval(1000−500) ≤ +0.30, i.e.
eval@1000 ≤ 8.44 — not distance-to-anchors; and a failure is
registered as "new information, endpoint choice re-opens to
best-grasping save", not a kill. Infra: disk 146G free (no save
since 500), RAM available 48G a third consecutive tick (plateau
holds), pruner unit active + log correctly start-line-only.
Projections: **step 1000 lands ~18:39Z** (save + eval-1000 + first
real prune), **endpoint ~03:0x–03:4xZ 08-21** at cumulative pace.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts, no reactions.

**Done** (this tick): babysit poll, eval-750 row banked + shape note,
pre-reg drift-guard spec re-read (bar = 8.44 on own curve), disk +
RAM + pruner-log checks, queue validate green (depth 2, 14 open),
now.md + archive roll.

**Next**: the ~18:4xZ tick banks eval-1000 + the registered drift
read, verifies the pruner log pruned step-500's optimizer.pt after
the step-1000 save, and re-reads RAM across the save. If the 1000
row confirms the flattening (even without a drift crossing), post
the two-point shape note in-channel. Endpoint session owns
`democlean-endpoint-close`. `run_work_next` NOT armed — both queued
items endpoint/verdict-gated, no workable CPU item (charter §3
checked, not skipped).*

## Utilization footer

Session 2026-08-20 17:59–18:0xZ (tick; `democlean` riding, ~3.8
GPU-h elapsed of ~13.1 projected vs the 17 gate): **babysit exit 0 —
step 840/3000 at 18:00Z, loss 0.502 → 0.527 (small uptick inside the
step-noise band, sequence still trending down), vram 62.24/75, no
gate crossings; 17.02 s/step window, 3.8 steps/min; no new probe row
(eval-1000 lands at the ~18:42Z save); disk 146G, RAM available 49G
fourth consecutive tick (plateau holds), pruner unit active + log
start-line-only (correct, no save since 500); Discord fully quiet
(read empty, inbox empty, history -n 5 all own posts, no reactions);
queue validate green depth 2 (14 open); run_work_next NOT armed —
both queued items endpoint/verdict-gated; boundaries: step-1000
~18:42Z (save + eval-1000 + registered drift read ≤ 8.44 + first
real prune), ~18:4xZ tick banks all of it + RAM re-read across the
save + in-channel shape post if the flattening confirms; endpoint
~03:0x–04:1xZ 08-21.**

Session 2026-08-20 17:38–17:4xZ (tick; `democlean` riding, ~3.4
GPU-h elapsed of ~12.9 projected vs the 17 gate): **babysit exit 0 —
step 760/3000 at 17:39Z, loss 0.543 → 0.502, vram 62.24/75, no gate
crossings; window 19.11 s/step inflated by the eval-750 pause,
cumulative ~16.1 s/step on-anchor; EVAL-750 ROW BANKED record-only:
7.90@750 vs convicted 6.65 / onerig 6.73 — first row above both
anchors, 500→750 improvement −0.24 vs anchors' −1.59/−1.31, curve
flattening early (not the convicted 2250–2750 signature; sim100 at
3000 stays the verdict); pre-reg drift-guard re-read: bar is own
Δ(1000−500) ≤ +0.30 → eval@1000 ≤ 8.44, at 7.90 and falling no gate
threatened, and failure = endpoint re-opens, not a kill; disk 146G,
RAM 48G third consecutive tick (plateau holds), pruner idle-correct;
Discord fully quiet (read empty, inbox empty, history -n 5 all own
posts, no reactions); queue validate green depth 2 (14 open);
run_work_next NOT armed — both queued items endpoint/verdict-gated;
boundaries: step-1000 ~18:39Z (save + eval + first real prune),
~18:4xZ tick banks drift read + pruner-log verify + RAM re-read,
in-channel shape post if the 1000 row confirms the flattening;
endpoint ~03:0x–03:4xZ 08-21.**

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
