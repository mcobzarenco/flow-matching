# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 18:19–18:2xZ (real `date -u` at stamp: 18:24) —
tick, babysit: **owner 18:19Z caught a real shim discrepancy — the
official LeRobot v3.0→v2.1 conversion sign-flips shoulder_lift
((−1,+90) = 90−arm); our fitted map used (+1,+180). The INERT 0.00×20
release read is now SUSPECT on lift; official-map rerun queued as first
GPU claim.***

**Status**: no live jobs, GPU idle. Queue validate green (depth 4, 14
open) — new item `release-eval20-officialmap` is first GPU claim;
`run_work_next` armed. Driver-guard 18:14Z straggler alert: pid was a
bare `-zsh` session child, no job attached — noise, nothing to relaunch.

**Steering**: owner 18:19:08Z — does our shim match
irenegracekp/molmoact2-so101 `inference.py` (offsets `0,90,90,0,0,0`,
signs `1,-1,1,1,1,1`, the documented v3.0→v2.1 SO-100/101 conversion)?
Verified on the real tables (CPU, same session): **4/6 joints match
exactly** (pan/wrist_flex/gripper identity; elbow +90 — our override
landed the official value). **shoulder_lift MISMATCHES**: the mirror
(−1,+90) QUALIFIED in our fit and covers the release box better (7.5%
vs 27.9% uncovered) but lost to the pre-registered MIRROR_MARGIN=0.25
rule by 20.4 pt — the gate rejected a real mirror (box's panel snap
+180 has the same exposure). wrist_roll ambiguous: ours −90 vs official
identity, both 61% uncovered (span mismatch); identity clamps sim wrist
home (77.6°) above the box ceiling (43.5°) — our −90 may absorb a
rig-specific zero. **Consequence**: wrong lift sign direction-inverts
decoded lift motion — matches the filmed swing-down-and-park; the
first-action detector is sign-blind at rest (any bijection preserves
action≈state). Full comparison + rerun plan posted 18:2xZ. Live
exchange 18:32–18:3xZ: owner *how is it going?* → status posted;
owner 18:34:34Z — *running the seeds now with the snippet's map?
Update me on episodes 1 by 1* → confirmed 18:36Z: snippet map EXACTLY
as primary (wrist_roll identity per snippet; our −90 arm optional
secondary), per-episode in-channel posts as rows land (completion
order under workers=8; strict-sequential offered if wanted — check
channel before launch). Steering recorded in the queue item. Open
asks: v3-rerun unhold + arm set (15:13Z), GRPO memo review,
disk-draws sign-off.

**Done**: per-joint audit banked (this entry + queue item);
`release-eval20-officialmap` queued (sign-carrying override CLI
extension → tripwires under official map → same 20 seeds, ≤0.4 GPU-h,
amendment on the existing pre-reg page, INERT claim to be explicitly
re-dispositioned); `run_work_next` armed.

**Next**: chained work session executes the official-map rerun, then
CPU lanes (`lit-sim-improvement-levers`, `sim-wrist-compositing`).
v3-rerun still pends the owner unhold. `queue.json` canonical.*

*Updated 2026-08-12 17:29–18:1xZ (real `date -u` at stamp: 18:10) —
work session, bounded: **release-eval20-convmap DONE — the released
MolmoAct2 checkpoint, unit-shimmed into the sim, is INERT: progress
0.00 on all 20 seeds, the boat never touched; the shim itself is
verified (first-action 2.98° vs contract anchor 6.31°), so units are
demonstrably NOT the blocker — scene/task grounding is.***

**Status**: no live jobs, GPU idle again. The one GPU claim ran ridden
end-to-end (~0.19/0.5 GPU-h gate: 3 tripwire probes + one 20-seed
parallel arm, 5.5 min at workers=8, first-poll 100% util / 20.8 GB).
Queue validate green (depth 3, 13 open).

**Steering**: no new owner messages this session (polled at boot,
pre-post, close). Executed the standing owner prio 17:13:24Z with its
17:22Z-👍'd design. Open asks unchanged: v3-rerun unhold + arm set
(15:13Z), GRPO probe memo review, disk-draws sign-off.

**Done** (commits `5b3783e`, close-out): branch REBASED onto latest
main (brings the box's `--molmo-norm`/`fit_convention_map` machinery,
`4d54490`/`63155d4`). (1) Instrument: `sim/convmap.py` seam
(fit + explicit per-joint overrides, off-contract `_convmap`
provenance in rows) + `--convmap-seam-stats`/`--convmap-override` on
the parallel driver (state-in A, action-out A⁻¹ through the policy's
own convention-map path); tripwire script
`fontaine/scripts/convmap_tripwires.py`; 3 oracles; checks 791 green.
(2) Tripwires did real work: gated fit gave lift+180 only; coverage
caught elbow (identity leaves 56% of the rig range below the release
floor; +90 → 10%) and the first-action probe caught wrist_roll
(identity delta 34.5° = sim home 77.6° minus release ceiling 43.5°,
the clamp signature; −90 → 0.97°). Final map lift+180 elbow+90
wrist_roll−90; first-action 2.98° < anchor 6.31° = the note's
predicted collapse. (3) The read: INERT 0.00 × 20 — not frozen;
smooth, repeatable swing to the same off-task park every seed,
wrist cam ending off-table. 0 knock-aways, 0 approaches. Paired:
release−step2000 +0.46 [−0.01,+1.11] (pure knock-away artifact),
release−step500 −0.02 (noise). Clean bracket: 500–2000 ft steps buy
scene-directed reaching from a unit-corrected base that does nothing
task-relevant here. (4) Cross-check banked + posted for the box: lift
+180 AGREE; elbow +90 agrees only past the midpoint gate's 2.2°
near-tie (estimator under-translates rig-table-shaped joints —
suggested coverage-fraction tiebreak); wrist_roll −90 empirical,
consistent with the ±90 wrap family; wrist spans stay 53–61%
uncovered under any offset (release's narrower wrist workspace —
lower-bound caveat). Artifacts: rows + 20 videos + chart (dark,
per-seed) on the reports Space `release_convmap/` (curl 200);
pre-reg page carries full results; Discord 2 posts ~18:0xZ.

**Next**: `queue_cli.py next` → CPU lanes: `lit-sim-improvement-levers`
(owner-called lit slice), `sim-wrist-compositing`. GPU idle pending
the v3-rerun unhold (15:13Z ask — the re-baseline carrier).
`grpo-signal-probe` owner_hold. `queue.json` canonical.*

*Updated 2026-08-12 17:20–17:2xZ (real `date -u` at stamp: 17:25) —
tick, babysit: **new owner prio landed and is queued — run the RELEASED
MolmoAct2 checkpoint in sim through a unit shim; ack + design posted,
owner 👍, chained work session armed to execute.***

**Status**: no live jobs, GPU idle (0% / 0 MiB). Queue validate green
(depth 4, 14 open) — new item `release-eval20-convmap` is FIRST GPU
claim. The 17:15Z harness exit-1 alert was benign: the 15:31Z work
session died on API-529 overload retries AFTER its work was committed
(log-verified); nothing lost.

**Steering**: owner 17:13:24Z — *run the released checkpoint directly;
molmoact2 normalizes actions by global quantile stats assuming v2.1
lerobot format; read the note in depth* + attached box-side note on
molmoact2 unit contracts (committed:
`fontaine/notes/molmoact2-unit-contracts-box-note.md`). Read in depth;
key mechanics: the release's q01/q99 table is a *unit contract* (lift
box [+45.2, +186.1]) near-disjoint from our rig table ([−103.7,
+48.6]); tag equality ≠ table equality; raw-in-v3-sim is meaningless
(state below box floor → blind), so we execute the note's case 3 — a
per-joint affine shim (state-in / action-out), labeled off-contract
`_convmap`, lower-bound interpretation. Ack + 4-step plan posted
17:22Z, **owner 👍 confirmed**. Open asks: v3-rerun unhold (15:13Z),
GRPO probe memo review, disk-draws sign-off.

**Done**: `release-eval20-convmap` queued (owner prio, pre-reg page
`posts/2026-08-12-prereg-release-eval20-convmap.md` with the two
mandatory pre-GPU tripwires from the note: A⁻¹(box) workspace
coverage + first-action-vs-state unit-bug detector, ≤0.5 GPU-h gate);
box note committed into the repo; converted release located on disk
(`~/marius-convert-gate/converted/molmoact2_so100_101_release` — no
conversion step needed); exit-1 alert root-caused benign;
`run_work_next` armed.

**Next**: chained work session executes `release-eval20-convmap`
(shim → tripwires → 20 seeds parallel, paired vs step-500/step-2000
corrected arms; cross-check bank: our sim calibration's implied
lift/elbow map vs the box's fit_convention_map snap). Then CPU lanes:
`lit-sim-improvement-levers`, `sim-wrist-compositing`. v3-rerun still
pends the owner unhold. `queue.json` canonical.*

## Utilization footer

Session 2026-08-12 18:19–18:2xZ (tick, babysit; 0 new GPU-h — GPU
idle): owner 18:19Z asked whether our convmap shim matches the official
LeRobot v3.0→v2.1 conversion (linked inference.py) → audited on the
real tables same session: 4/6 joints match; **shoulder_lift does NOT**
(official (−1,+90) mirror qualified in our fit, covered better, lost
only to the MIRROR_MARGIN rule) and wrist_roll is ambiguous both ways.
INERT 0.00×20 read flagged suspect on lift (sign inverts decoded lift
motion; first-action detector sign-blind at rest); full comparison
posted in-channel, `release-eval20-officialmap` queued first GPU claim
(≤0.4 gate), run_work_next armed. Driver-guard straggler alert
dispositioned noise (bare zsh). Archive roll: 1 main entry (16:55
work), 2 footer notes (17:20 tick, 15:31 work).

Session 2026-08-12 17:29–18:1xZ (work, bounded; **+~0.19 GPU-h** —
convmap tripwire probes + one 20-seed parallel arm, ridden end-to-end;
exploit, owner prio): release-eval20-convmap DONE same session as
queued — rebase onto box's molmo_norm machinery, seam instrument +
tripwires + oracles, both pre-GPU gates dispositioned (elbow and
wrist_roll overrides earned by coverage + first-action evidence, not
assumed), 20-seed read INERT 0.00×20 with verified shim, cross-check
banked to the box in-channel. No steering traffic; 2 result posts.

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames), 3rd
launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3 rungs
(+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
23:24Z–02:37Z 08-08 **COMPLETE +~3.2 GPU-h (≤ 8 gate)**;
 08-08 daytime: local rung-(b) preflight+stage1
08:49–10:15Z **+~1.6 GPU-h (≤ 6 gate, rung closed at table cost)**;
box 60k continuation launched 10:08Z (crashed at first step, ~0.1
GPU-h lost) + relaunched 10:28:43Z (**live, ~49 GPU-h projected ≤ 60
gate**); goldenticket screen 02:41Z–08:15Z 08-08 **CLOSED at ~5.55 GPU-h ≤ 6
gate** (s1 ~1.7 + s2 ~0.85 + s3 2.99); box molmo2 chain: 40k train
to ~04:0xZ, greedy ~1.7 GPU-h, draws10_t1 04:54–07:22Z **~10 GPU-h
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box 60k continuation COMPLETE 08-08 ~23:4xZ
(~49 GPU-h ≤ 60 gate, chained evals incl.); local subgoal-swap arms
08-09 ~02:1x–03:42Z +~1.5 GPU-h ≤ 3 gate; box K-smoke ladder 08-09
04:02–04:39Z **+~0.5 GPU-h ≤ 6 gate (rung 1 GREEN first try)**; box
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + panel_v2
eval COMPLETE ~08:01Z (+~1.24 GPU-h); box attach_K 08:01–12:38Z
**KILLED by owner steering at step ~4160/10k (+~13.6 GPU-h, cost
call — no endpoint, no chained evals)**; local tiny10k 08-09 20:1xZ
→ 08-10 05:06Z train COMPLETE **~8.7/15 GPU-h incl. OOM replay** +
chained panel_v2 eval COMPLETE 08-10 05:45Z (+~0.6 GPU-h, **~9.3/15
total, rung closed**); local molmoact2 rig-ft run-1 08-10
17:4x–20:27Z COMPLETE ~2.7/12 GPU-h; local er35k owner-request evals
08-10 20:5x–00:41Z 08-11 ~2.2/8 GPU-h; local molmoact2 port parity
reads 08-10/11 ~0.7 GPU-h; local molmoact2_ae_ours (port item 4)
08-11 05:19–06:56Z **COMPLETE ~1.9/6 GPU-h (port total ~2.6/8)**).
Older dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).
