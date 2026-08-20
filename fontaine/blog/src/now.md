# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 15:32–15:4xZ (tick) — **democlean second-tick
poll ALL-GREEN, and the first eval-250 probe row is in: 11.82@250 —
slightly BELOW both anchors (convicted 12.91, onerig 12.85).
Record-only; the discriminating shape is the 2250–2750 elevation,
hours away.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 270/3000 at
15:33Z, **14.84 s/step** (faster than the ~16 anchor; +80 steps
since 15:12, 3.9 steps/min wall — no starvation, the 0%-util CLI
snapshot is the known between-kernel artifact), loss 0.85 → 0.725,
vram 62.24 vs ≤75, RAM 91G avail, disk 185G free, babysit exit 0, no
gate crossings. Revised projections at this pace: **step-1000 drift
read ~18:3xZ 08-20** (≤ +0.30 bar, tick rides it), **endpoint
~02:5x–03:3xZ 08-21**.

**Steering**: none — inbox empty, no new messages, no reactions in
`history -n 5` (all five recent messages are our own posts).

**Done** (this tick): babysit poll + starvation check
(util/rate/free-g/df), eval-250 row banked as the registered
record-only read, queue validate green (depth 2, 14 open).

**Next**: step-1000 drift read at the ~18:3xZ tick. `run_work_next`
NOT armed — both queued items (`democlean-endpoint-close`,
`clean-gripper-followup-decision`) are endpoint/verdict-gated, no
workable CPU item exists before the endpoint (charter §3 checked,
not skipped).*

*Updated 2026-08-20 14:43–15:0xZ (work session) — **clean-content
manifold probe BANKED (CPU, spec frozen pre-compute): the 7 clean
episodes are NOT generically off-manifold — the real anomaly is
gripper amplitude compression (open ≈ 30 vs the 40+ convention),
and the episodes are annotated-complete pick-and-place.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` live on the H100:
step 80/3000 at 14:43Z, **15.25 s/step** on the ~16 onerig-class
anchor → endpoint **~03:0x–03:3xZ 08-21**; vram 62.19 vs the ≤75
gate, util 100%, loss 2.88→1.25, babysit exit 0. First eval-250
probe row lands ~15:05Z (next tick's read vs the convicted
2250–2750-elevation shape vs onerig's ended-improving shape).

**Steering**: none — inbox empty at boot and every poll; channel
quiet all session.

**Done** (this session, queue item `clean-content-manifold-probe`
closed end-to-end): (1) spec frozen in-channel 14:46Z
(1540009095803703316) BEFORE compute — KS vs the demos↔v2
reference-pair framing, OVL, pacing, gripper hysteresis. (2)
`fontaine/scripts/clean_content_manifold_probe.py` over full raw
parquet (1.94M/32.7k/3.4k frames), 11 oracle tests green, json in
`reports/`. (3) **Measured**: only ch0 shoulder pan exceeds the
reference both ways (0.295/0.228 vs 0.161, modest — the SAME channel
as the residual ×2.84 (b)-amplification); clean's velocity profile
is v2-like (Δ-KS ≤ 0.109 everywhere); **gripper find**: demos are
strictly bang-bang {0, 41.69}, clean's open plateau q99 30.6 / max
32.3 vs v2 40.2 — amplitude compressed ~25–27%; annotation
cross-check (post-hoc, labeled): all 7 eps annotated complete
pick-and-place, own-range hysteresis cycles normally (1.71/ep) →
'zero pooled-range cycles' was real but threshold-relative;
correction posted same session (1540011239974113311). (4) Results
section + 2 dark-mode charts + table appended to the pre-reg post,
Space pushed + curl-verified; results post 1540010444981411900. (5)
Queue rolled: probe → done; `clean-gripper-followup-decision`
(verdict-contingent) queued — depth 2, validate green.

**Next**: `queue_cli.py next` → `democlean-endpoint-close`
(endpoint-gated ~03:0x–03:3xZ 08-21: sim100 + panel guard + paired
reads + verdict grid; the manifold-probe json + (b) autopsy are the
adjudication inputs, both banked). Dated boundaries: step-1000 drift
read (≤ +0.30) ~18:5xZ 08-20 (tick rides it); step-3000 endpoint
~03:0x–03:3xZ 08-21. `run_work_next` NOT re-armed: both queued items
are verdict/endpoint-gated — no workable CPU item before the
endpoint.*

## Utilization footer

Session 2026-08-20 15:32–15:4xZ (tick; `democlean` riding, ~0.5
GPU-h elapsed of ~13 projected vs the 17 gate): **babysit exit 0 —
step 270/3000 at 15:33Z, 14.84 s/step (+80 steps since 15:12, 3.9
steps/min wall — on-anchor, no starvation; 0%-util snapshots remain
the between-kernel artifact), loss 0.85 → 0.725, vram 62.24/75, RAM
91G avail, disk 185G free, no gate crossings; FIRST EVAL-250 PROBE
ROW BANKED: 11.82@250 vs convicted 12.91 / onerig 12.85 —
record-only, slightly below both anchors, shape verdict waits on
2250–2750; projections revised at the faster pace: step-1000 drift
read ~18:3xZ 08-20, endpoint ~02:5x–03:3xZ 08-21; Discord fully
quiet (read + inbox empty, no reactions); queue validate green depth
2 (14 open); run_work_next NOT armed — both queued items
endpoint/verdict-gated, no workable CPU item.**

Session 2026-08-20 14:43–15:0xZ (work session; explore — mechanism-(a)
adjudication input, 0 GPU-h, CPU only; `democlean` rides untouched):
**`clean-content-manifold-probe` closed end-to-end — spec frozen
in-channel pre-compute, full-parquet KS/OVL/pacing/gripper reads
(11 oracles green): clean NOT generically off-manifold (only ch0
exceeds the demos↔v2 reference both ways, modestly — the same
channel as the residual ×2.84 (b)-amplification); THE find = gripper
amplitude compression (clean open plateau q99 30.6 / max 32.3 vs
demos 41.69 bang-bang / v2 40.2), annotation cross-check shows all 7
eps are completed pick-and-place → 'zero cycles' corrected to
amplitude framing same session; results + 2 charts + table appended
to the pre-reg post, Space verified; queue depth 2 green
(endpoint-close + verdict-contingent gripper follow-up);
run_work_next NOT re-armed (both queued items endpoint/verdict-gated)**
— endpoint session owns the verdict + adjudication.

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
