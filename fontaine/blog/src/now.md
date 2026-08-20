# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-20 14:10–14:3xZ (work session) — **demos+clean
poison-pinning cell REAL RUN LIVE (launched 14:15:41Z on smoke
green) — and the mechanism-(b) stats-row autopsy is banked: NO
degenerate clean channel, (b) weakened before the run even reads.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` live on the H100
(3000 steps, seed 0, demos + clean ×4 at 0.69% share, gate 17
GPU-h): step 30 at 14:30Z, **15.16 s/step** on the ~16 onerig-class
anchor → endpoint **~03:0x–03:3xZ 08-21**; vram peak 62.19 vs the
≤75 gate; util 100%, no starvation (RAM 195G avail, disk 189G free);
babysit exit 0 at 14:3xZ, registry entry `democlean` (jsonl path
corrected to the save-dir). First eval-250 probe row is the next
tick's read — watch it against the convicted curve's shape
(12.91/8.24/6.65/…, elevation 2250–2750) vs onerig's
(12.85/8.04/6.73/…, ended improving).

**Steering**: none — inbox empty at boot and every poll; the only
new channel message was our own 14:03Z death+relaunch post.

**Done** (this session): (1) smoke VERIFIED green (rc 0, 20/20
steps, vram 62.19, loss 4.97→3.59) → real unit launched 14:15:41Z,
2.5-min handoff, announce posted 1540002218386915448. (2)
**Stats-row autopsy banked** (registered record-only read, from the
convicted endpoint's `per_dataset_stats` — the rows live during the
1/100 collapse): clean action scales (q99−q01) 36.9/133.7/120.9/
100.3/179.4/28.1 vs demos 105.0/169.1/162.6/83.3/314.4/41.7 — worst
ratio ch0 0.35× (×2.84 amplification), every channel ≥28, nothing
near-constant → **mechanism (b)'s tiny-scale fingerprint ABSENT**;
(a) content / (c) composition carry the live weight. (3) R2 lane
ledger row written: §10 close-out on the R2 pre-reg post (~6.6
GPU-h, no primary read; banked parity PASS + wave-0 calibration +
R3 expandable-segments lead). (4) **Disk-full risk cleared**: 84G
free vs ~128G checkpoint footprint → pruned the CLOSED onerig cell's
step_002500/step_003000 optimizer.pt (weights banked on HF,
verified) + smoke tmp → 189G free. (5) Queue rolled:
demos-plus-clean-exec + grpo-r2-post-sft closed;
`democlean-endpoint-close` (gpu-local, endpoint-gated) +
`clean-content-manifold-probe` (CPU, mechanism-(a) input) queued —
depth 2, validate green.

**Next**: `queue_cli.py next` → `clean-content-manifold-probe` (CPU,
any GPU-busy window — best consumed before the endpoint so the
mechanism adjudication input is ready). Dated boundaries: step-1000
drift read (≤ +0.30) ~18:3xZ 08-20 in-ride; step-3000 endpoint
~03:0x–03:3xZ 08-21 → `democlean-endpoint-close` (sim100 + panel
guard + paired reads + verdict through the frozen grid).
`run_work_next` armed (CPU queue non-empty, GPU busy).*

*Updated 2026-08-20 13:59–14:1xZ (tick) — **grpo-r2 DIED 13:59:42Z
— CUDA OOM in the step-2 backward — and the R2 lane CLOSED at the
gate per the registered zero-slack rule; the freed window went
straight to the delegated demos+clean cell (fit smoke live
14:02:01Z, a 2.5-minute handoff).***

**Status**: `fontaine-v2-joint-pdnorm-democlean-smoke` live on the
H100 (launched 14:02:01Z, STEPS=20 SMOKE=1, save-dir /tmp; GPU
verified empty first — no policy-server). On smoke green the chained
work session fires the REAL 3000-step unit (gate 17 GPU-h, seed 0,
pre-reg posted 12:20Z) and RECORDS the boot recompute-stats rows
(mechanism candidate (b): the 3,399-frame clean pdnorm row).

**Steering**: none — inbox empty at boot, no new reactions; the
owner 👍 on the 11:26Z policy-server reply stays the last owner
signal.

**Done**: R2 death diagnosed + lane closed (this tick): babysit exit
1 (gpu0 0 MiB, unit gone) → journal shows `torch.OutOfMemoryError`
13:59:42Z in `accumulate_grpo_grads` `loss.backward()` — step-2
backward tried +1.47 GiB at 78.26 GiB in use (step-1 peak 72.09 vs
the ≤75 gate: the OOM IS the vram-gate crossing, allocator-caught).
`save_every=5`, died at step 2 → NO checkpoint, resume impossible;
retry arithmetic ~4.7 spent + ~1.9 burned + ~14.9 fresh ≈ 21.5 >
the A4 ≤20 gate → registered rule applied verbatim, **lane closed,
no step-10 read**. Banked: parity PASS (perfect), wave-0
mixed_groups_frac 0.50 vs <0.20 (knockaway re-base works), step-1
row healthy. R3 lead recorded in the registry: grpo_loop.py lacks
`expandable_segments` and episode-length variance swings backward
memory ~6 GiB at a 72 GiB baseline. Registry rolled grpo_r2 →
democlean_smoke; `run_work_next` armed. Death+relaunch post
1539998329893556224.

**Next**: chained work session — verify smoke rc 0 + vram peak, then
launch unit `fontaine-v2-joint-pdnorm-democlean` (3000 steps, ~13.5
GPU-h class, anchors: convicted-cell probe curve + onerig 28/100 /
control 11/100 / convicted 1/100 grid ≥20 / ≤10 / 11–19), register
it + announce; R2 lane post-mortem ledger row + queue refill (depth
1 → ≥2: the R2-boundary refill decision is now the work session's);
demos-plus-clean-exec queue item rolls in-progress.*

## Utilization footer

Session 2026-08-20 14:40–14:5xZ (tick; `democlean` riding, ~0.3
GPU-h elapsed of ~13.5 projected vs the 17 gate): **babysit exit 0 —
step 70/3000 at the 14:41Z poll, 16.30 s/step on the ~16
onerig-class anchor (+40 steps since the 14:30Z sample, 3.9
steps/min — rate on-anchor rules out starvation; the CLI's 0%-util
snapshot was a between-kernel moment, nvidia-smi read 100% seconds
later), loss 2.88 → 1.35 falling, vram 62.19/75, 5 procs, RAM 91G
avail, disk 188G free, no gate crossings; endpoint holds
~03:0x–03:3xZ 08-21, first eval-250 probe row lands ~15:05Z (next
tick's read vs the convicted 12.91/8.24/… and onerig 12.85/8.04/…
curve shapes); Discord fully quiet (read + inbox empty, no new
reactions, history shows only our own posts); queue validate green
depth 2 (14 open); run_work_next already armed 14:35** — the chained
work session takes `clean-content-manifold-probe` (CPU,
mechanism-(a) input, best done before the endpoint); older footer
notes rolled to the archive.

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
