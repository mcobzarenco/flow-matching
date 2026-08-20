# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 19:03–19:0xZ (tick) — **democlean plain riding
tick ALL-GREEN: step 1070/3000, loss 0.4471, nothing to decide.
First post-boundary tick confirms the save cycle left everything
clean: disk steady at 135G, RAM plateau holds at 49G, pruner log
correctly quiet since its 18:49Z first real prune.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1070/3000 at
19:03Z, loss 0.4611 → 0.4471 (downtrend intact), 16.229 s/step
window (+70 steps since 18:42, 3.4 steps/min wall — save+eval pause
in-window), vram 62.24 vs ≤75, babysit exit 0, no gate crossings,
~8.7 h to 3000 → **endpoint ~03:4xZ 08-21**. Probe curve unchanged
(11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000; next row @1250,
record-only, no save). Infra: disk 135G free (no save since 1000),
RAM available 49G — plateau holds post-save-cycle; pruner unit
active, log unchanged since the 18:49:22Z step-500 prune (correct —
next pass with work is after the step-1500 save).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, disk + RAM + pruner-log checks,
queue validate green (depth 2, 14 open), now.md + archive roll.

**Next**: plain riding ticks — next boundary is the step-1500 save
(~21:0xZ, eval-1500 row + prune verify); probe row @1250 record-only.
Endpoint session (~03:4xZ 08-21) owns `democlean-endpoint-close`
(sim100 + panel guard + paired reads + verdict grid).
`run_work_next` NOT armed — both queued items endpoint/verdict-gated,
no workable CPU item (charter §3 checked, not skipped).*

*Updated 2026-08-20 18:41–18:5xZ (tick) — **step-1000 boundary
BANKED end-to-end: eval-1000 = 6.4894, registered drift read PASSES
decisively (delta(1000−500) = −1.65 vs the ≤ +0.30 bar) — endpoint
stays step 3000, nothing re-opens. And the 750-row early-flattening
flag is RETRACTED: 750→1000 falls −1.41, steeper than both anchors
over the same span. Shape post in-channel 18:43Z. Save complete,
first real prune verified, RAM flat across the save.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1000/3000 at
18:42Z, loss 0.4611 (0.4725 → 0.4611, downtrend intact), 16.325
s/step, vram 62.24 vs ≤75, babysit exit 0, no gate crossings, ~9.1 h
to 3000 → **endpoint ~03:4xZ 08-21**. Probe curve 11.82@250 →
8.14@500 → 7.90@750 → **6.49@1000** — still above both anchors in
level (convicted 6.11, onerig 5.83 @1000) but converging; nothing
resembling the convicted 2250–2750 elevation. sim100 at 3000 stays
the verdict instrument; next probe row @1250 (record-only, no save).
Infra across the save: step-1000 checkpoint fully written by 18:43Z
(44G, vision backbone hardlinked); pruner pass at 18:49:22Z pruned
step-500's optimizer.pt (32G back, weights untouched — first real
prune verified in-log); disk 135G free post-prune (net ~−11G per
save cycle, on the ~124G peak-need model); RAM available 48G —
plateau holds across a full save cycle, leak watch stays closed.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts, no reactions.

**Done** (this tick): babysit poll, eval-1000 drift read (PASS,
registered), flattening-flag retraction + in-channel shape post,
save + first-prune verify, RAM re-read across the save, disk check,
queue validate green (depth 2, 14 open), now.md + archive roll.

**Next**: plain riding ticks — next boundary is the step-1500 save
(~21:0xZ, eval-1500 row + prune verify); probe rows @1250/@1500
record-only. Endpoint session (~03:4xZ 08-21) owns
`democlean-endpoint-close` (sim100 + panel guard + paired reads +
verdict grid). `run_work_next` NOT armed — both queued items
endpoint/verdict-gated, no workable CPU item (charter §3 checked,
not skipped).*

## Utilization footer

Session 2026-08-20 19:03–19:0xZ (tick; `democlean` riding, ~4.9
GPU-h elapsed of ~13.5 projected vs the 17 gate): **plain riding
tick ALL-GREEN — babysit exit 0, step 1070/3000 at 19:03Z, loss
0.4611 → 0.4471, vram 62.24/75, no gate crossings, 16.229 s/step
window (save+eval pause in-window), ~8.7 h to 3000 → endpoint
~03:4xZ 08-21. First post-boundary check: disk steady 135G, RAM
available 49G (plateau holds post-save-cycle), pruner log correctly
quiet since the 18:49:22Z step-500 prune. Discord fully quiet (read
empty, inbox empty, history -n 5 all own posts, no reactions);
queue validate green depth 2 (14 open); run_work_next NOT armed —
both queued items endpoint/verdict-gated. Next boundary: step-1500
save ~21:0xZ (eval-1500 + prune verify).**

Session 2026-08-20 18:41–18:5xZ (tick; `democlean` riding, ~4.5
GPU-h elapsed of ~13.4 projected vs the 17 gate): **step-1000
boundary banked end-to-end — babysit exit 0 at step 1000/3000
18:42Z, loss 0.4611, vram 62.24/75, no gate crossings, 16.325
s/step, ~9.1 h to 3000 → endpoint ~03:4xZ 08-21. Eval-1000 =
6.4894: registered drift read PASSES (delta(1000−500) −1.65 vs the
≤ +0.30 bar), endpoint stays 3000; 750-row flattening flag
RETRACTED (750→1000 −1.41, steeper than convicted −0.54 / onerig
−0.90; level still above both — 6.11/5.83 — but converging); shape
post in-channel 18:43Z. Save complete 18:43Z (44G), pruner first
real prune verified in-log at 18:49:22Z (step-500 optimizer.pt, 32G
reclaimed, weights untouched), disk 135G free post-prune (net ~−11G
per save cycle, on the ~124G peak-need model), RAM available 48G
flat across the save cycle (leak watch stays closed). Discord
otherwise quiet (read
empty, inbox empty, history -n 5 all own posts + our shape post, no
reactions); queue validate green depth 2 (14 open); run_work_next
NOT armed — both queued items endpoint/verdict-gated. Next boundary:
step-1500 save ~21:0xZ.**

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
