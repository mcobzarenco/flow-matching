# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 19:04Z (tick) — **onerig healthy at first
post-warmup read; work session chained for the R2 parity fix.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` step 130/3000,
15.127 s/step — warmup pace fully resolved into the smoke/mixed-cell
band (15.1–15.4); loss 1.0273, 62.19 GiB vs the 71 gate, 84% util, 5
procs. ~12.1 h to endpoint → ETA holds ~07:1xZ 08-20; step-1000
drift read ~22:3xZ remains the next tick duty (READ not kill, Δ ≤
+0.30 raw).

**Steering**: none — read + inbox empty, history clean (no
reactions).

**Done**: babysit poll (healthy, no gate crossings); queue validate
OK (depth 2, 16 open); `run_work_next` touched 19:04:42Z — GPU busy
+ CPU item queued (`grpo-r2-serving-parity-fix`, the R2 launch
gate).

**Next**: chained work session takes `grpo-r2-serving-parity-fix`
(path diff + parity oracle; the cheap GPU parity read waits for the
post-onerig window). Tick duties: 22:3xZ drift read, endpoint
~07:0xZ 08-20.*

*Updated 2026-08-19 17:5x–18:4xZ (real `date -u` at write: 18:35, same
work session continued) — **the R2 relaunch exposed a DEEPER break and
was KILLED 18:06:48Z: the loop's serving stack (MolmoAct2DiscreteStack
+ hardcoded official shim, er60k-era) is inert on v2 corrected-table
checkpoints — the A4 substrate fix was necessary but not sufficient.
R2 lane PARKED on a serving-parity fix (now a launch gate).
Demos+one-rig took the GPU 18:22:47Z on the banked owner GO.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm_onerig` LIVE (unit
`fontaine-v2-joint-pdnorm-onerig`, launched 18:22:47Z after a green
fit smoke) — step 10+ at warmup pace (20.8 s/step; smoke measured
~15.4, mixed-cell precedent ~15.1–15.3), 66.4 GiB / 83–100% util, no
starvation, RAM 91G, disk 214G free. Endpoint ETA ~07:0xZ 08-20;
step-1000 drift read ~22:3xZ tonight (READ not kill). Gates vram 71 /
17 GPU-h. R2 lane: killed relaunch burned ~0.33 GPU-h (lane total
~4.0); artifacts banked (`loop_wave0abort_patched/`, killed `loop/`,
`wave0_diag/`).

**Steering**: none — read + inbox empty at every poll (one babysit
read was piped through head against the never-truncate rule;
recovered immediately: inbox empty, history clean, nothing lost).
Kill + root-cause post 18:08Z, onerig launch post 18:36Z — decide +
announce, no GO asks.

**Done (this half)**: (1) R2 relaunch ridden to the step-0 eval row
(+17 min, on the re-measured gap): 0/20 with ALL 20 scenes bit-frozen
under VERIFIED standins (meta + worker seam) — vs the greedy anchor
leg's 59/100 visible displacement, P≈2e-8 → loop path inert
independent of substrate; my wave0_diag probe had confounded
driver-with-substrate (no sequential-patched control). (2) Class
pinned: R1-B on the released ckpt through the SAME loop stack
interacted (knockaway 0.33–0.45, wave successes 3–4) — the break is
v2-checkpoint-specific; suspicious seam spotted
(`grpo_replay._batch` reuses ACTION quantiles as `state_stats`).
(3) Run killed 18:06:48Z on that evidence (saved ~1 GPU-h to the
wave-0 re-fire); registry pruned with the full postmortem. (4)
`grpo-r2-serving-parity-fix` queued as the R2 launch gate;
`grpo-r2-boundary-legs-launcher` blocked on it. (5)
`demos-plus-one-rig-exec` EXECUTED (closed superseded-by-execution
per the pdnorm precedent): smoke green, unit live, preamble verified
(2 datasets, clean dropped, v2 ×4 = 6.30% share), babysit entry
live; `onerig-endpoint-close` refilled (frozen grid ≥20 / ≤10 /
11–19; anchors demosonly 11, mixed 1).

**Next**: `queue_cli.py next` → CPU item
`grpo-r2-serving-parity-fix` (diff the two serving paths on v2,
parity oracle, launcher-gated); onerig boundaries: step-1000 drift
read ~22:3xZ 08-19 (tick duty), endpoint ~07:0xZ 08-20 →
`onerig-endpoint-close`. R2 relaunch only on parity green +
re-registration (A5).*

*Updated 2026-08-19 16:21–17:5xZ (real `date -u` at write: 17:49) —
work session (chained): **R2 wave-0 gate FIRED (mixed 0.0 < 0.20,
17:19Z) → substrate bug convicted in ~20 min → fixed + RELAUNCHED
17:46:56Z. The loop rendered the 08-18 `patched` production default
while every R2 anchor + preflight ran `standins` — the stand-ins-era
policy is fully inert on patched (64/64 wave episodes zero
interaction, distances bit-frozen). Probe driver on the SAME seeds
under standins: 6/8 interact — seed band exonerated. Boundary-reads
instrument also landed; the endpoint is now reads-not-code.***

**Status**: `grpo_r2` LIVE (relaunch 17:46:56Z, unit `grpo-r2`) —
same A3.4 frozen argv + `--clutter-appearance standins` (A4); first
poll 3 procs, 28.8 GiB / 64%, step-0 eval rolling. Gates re-armed
fresh (wave-0 mixed 0.20, knockaway wave0 self-capture, kl_stop
0.06). Budget: ~3.7 GPU-h spent pre-relaunch (preflight 2.25 +
aborted patched wave ~1.2 + probe 0.24), expected total ~18.5, gate
≤20 (A4 re-price, supersedes A3.5's 15). Measured startup gaps: eval
row ~+16 min (~18:0xZ), wave-0 gate row ~+70 min (~18:5xZ) — riding
that read in-session. RAM fine, disk 227 GB free (92%).

**Steering**: none — read + inbox empty all session; abort +
diagnosis + relaunch posted 17:48:09Z (decide + announce, no GO ask
per the standing rule).

**Done**: (1) `grpo-r2-boundary-reads-instrument` EXECUTED + CLOSED
(0a405a2, check.py 1083 green): `grpo_r2_boundary_verdict.py` — all
three A3.4 endpoint legs mechanized (PRIMARY paired per-seed exact
sign test vs 7/100 with oracle-pinned band edges; sampled vs
preflight floor record-only, decode-gap movement priced; flow
euler-10 vs 44/100 with the material line at the exact 5% tail
≤35/100), loud per-leg provenance guards, `overall_surface` combines
mechanically. (2) Wave-0 abort postmortem: gate fired 17:19Z, zero
interaction across 64 episodes diagnosed → probe driver A/B on the
same seeds convicted the substrate (patched vs standins), receipts
`outputs/sim/grpo_r2/wave0_diag/` + `loop_wave0abort_patched/`. (3)
Fix landed (4914f80): `--clutter-appearance` on `sim.grpo_loop`
(default patched, zero change elsewhere), threaded through wave +
eval seams, meta-recorded, launcher pins standins, parse-check
asserts, oracles green. A4 postmortem+re-price on the pre-reg page.
(4) R2 relaunched on the standing PASS verdict; registry updated
(started_utc, gate 20, measured startup gaps).

**Next**: in-session — ride to the wave-0 gate row (~18:5xZ): mixed
≥0.20 (predicted 0.487) = calibration read PASSES and the run
proceeds; below = a REAL calibration fail this time (group shape is
the amendment path). `queue_cli.py next` → CPU item
`grpo-r2-boundary-legs-launcher` (stage the endpoint's three GPU
legs as one command); R2 boundary ~step 10 (~0x:xxZ 08-20) → three
legs + `grpo_r2_boundary_verdict` (instrument banked), then
`demos-plus-one-rig-exec` takes the GPU (owner GO banked).*

*Updated 2026-08-19 16:17–16:2xZ (real `date -u` at write: 16:20) —
tick: **first babysit poll of live `grpo_r2` — healthy in the
pre-registered startup window. Liveness by procs+GPU (3 procs,
28.8 GiB, 62→100% util); babysit exit 1 "no parseable rows" is the
KNOWN startup read (first train.jsonl row ~17:1xZ). Discord fully
quiet; `run_work_next` already armed at 16:13, tick closes fast.***

**Status**: `grpo_r2` LIVE and healthy 7 min post-launch — 3 procs,
28.8 GiB / 100% util (62% momentarily mid-poll: step-0 eval + wave-0
rollout phase, replan/env cycles). No gate crossing (exit 3 did not
fire); nothing to judge yet — wave-0 gates (mixed ≥0.20 predicted
0.487, knockaway self-baseline) read at the first heartbeat row
~17:1xZ. RAM 162 GiB available, disk 227 GB free (92%).

**Steering**: none — read + inbox empty, history shows nothing after
our 16:11:50Z launch post, no reactions. No owner calls pending
(both closed 13:25Z).

**Done**: boot (pull clean), babysit CLI (exit 1 = the registry's
declared startup gap, procs+GPU confirm liveness), history + inbox
checks, queue validate green (depth 2, 15 open), standing
GPU/RAM/disk checks. No post owed (launch post 16:11Z is current;
next post-worthy event is the first heartbeat read).

**Next**: chained work session (marker armed 16:13) takes CPU item
`grpo-r2-boundary-reads-instrument` and reads the first train.jsonl
row ~17:1xZ (wave-0 gate judgment); ride cadence ~30-min babysit;
boundary ~step 10 (~0x:xxZ 08-20) → boundary legs per A3.4/A3.5,
then `demos-plus-one-rig-exec` takes the GPU.*

## Utilization footer

Session 2026-08-19 17:5x–18:4xZ (work, same session cont.; ~0.33
GPU-h banked — killed R2 relaunch — + `onerig` live from 18:22:47Z,
~13 expected / gate 17): **R2 relaunch step-0 eval read 0/20 ALL
scenes frozen under verified standins (P≈2e-8) → loop serving stack
convicted on v2 checkpoints (R1-B/released interacted through it) →
KILLED 18:06:48Z, lane parked on `grpo-r2-serving-parity-fix` (launch
gate); demos+one-rig fired on the banked GO (smoke green, preamble
verified, 66 GiB / 83–100%)** — exploit (registered lane + integrity);
queue green depth 2 (16 open). Disk 214 GB free (93%).

Session 2026-08-19 16:21–17:5xZ+ (work, chained; ~1.44 GPU-h banked
this session — aborted patched wave ~1.2 + diagnosis probe 0.24 —
plus `grpo_r2` relaunched 17:46:56Z riding, ~18.5 lane total
expected / gate ≤20 per A4): **boundary-reads instrument landed
(0a405a2) → wave-0 gate FIRED 17:19Z (mixed 0.0) → substrate bug
convicted (loop rendered patched, anchors standins; probe A/B on the
same seeds: 6/8 interact under standins) → fix + A4 + RELAUNCH
17:46:56Z (4914f80)** — exploit (registered lane + integrity fix);
queue green depth 2 (15 open: instrument closed,
`grpo-r2-boundary-legs-launcher` refilled). Disk 227 GB free (92%).

Session 2026-08-19 16:17–16:2xZ (tick; `grpo_r2` riding, ~0.2 GPU-h
elapsed of ~14 expected): **first poll of live R2 healthy in the
declared startup window (procs+GPU liveness, babysit exit 1 = known
train.jsonl gap until ~17:1xZ); Discord quiet, no gate crossings;
`run_work_next` already armed — fast close** — queue green depth 2
(15 open). Disk 227 GB free (92%).

Session 2026-08-19 13:07–16:1xZ (work, chained; ~2.25 GPU-h banked —
preflight leg 0 — + `grpo_r2` live from 16:10Z, ~14 expected / gate
≤15): **owner agree-with-recs mid-session → A3 ACTIVATED end-to-end:
launch kit landed (570e53e) → preflight PASS (sampled 8/100 vs greedy
7) → R2 FIRED 16:10:02Z; demos+one-rig GO banked, fires at the next
free GPU boundary; disk sweep executed in the ride window (~41G
freed, 16/16 bitwise audit)** — exploit (registered activation +
infra debt); queue green depth 2 (15 open: `grpo-r2-launch-kit` +
`disk-retirement-sweep-banked-sources` closed,
`grpo-r2-boundary-reads-instrument` refilled). Disk 231 GB free
(92%).


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
