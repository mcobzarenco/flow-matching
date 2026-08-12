# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-12 15:31–17:0xZ (real `date -u` at stamp: 16:55) —
work session, bounded: **the owner-prio flipped-physics rerun is
CLOSED — including an owner-caught render bug whose fix OVERTURNED the
first readout (MuJoCo `sameframe` fast path; corrected read:
knock-aways 6→2, the −12.3 cm catastrophe dissolves, paired +0.75 cm
CI-crossing) — plus an owner-extension step-500 arm: the EARLIER
checkpoint reads slightly better (paired +0.48, 9/3/8).***

**Status**: no live jobs — `ftrig_eval20_flip_parallel` COMPLETE (5
arms total: 15:42–15:59Z both arms, 16:2xZ corrected postflip_v2,
16:4xZ step-500 extension; all rc=0, ridden in-session; first-poll
100% util / 20.9 GB; ~0.45/0.5 GPU-h). GPU idle again, pending the
owner's v3-rerun unhold (15:13Z ask, still open). Registry pruned to
a completion note. Queue validate green (depth 3, 13 open).

**Steering** (live exchange 16:07–16:5xZ): owner prio 15:27:11Z
(flipped-physics rerun, parallel) → executed; results 16:02Z. Owner
16:07:24Z: *no difference between the videos, bracket still into the
table* → root-caused same session (below), fix + corrected rerun +
corrected numbers in-channel 16:31Z. Owner 16:27:55Z: *what is
`sameframe`?* → explainer posted 16:32Z. Owner 16:37:48Z: *try the
step-500 ftrig checkpoint too, same 20 seeds* → converted + run +
numbers in-channel 16:54Z. Open asks: v3-rerun unhold + arm set
(15:13Z), GRPO probe memo review, disk-draws sign-off.

**Done** (commits `c68ea06`, `49d883f` + correction close-out): queue
item `ftrig-eval20-flipped-parallel` CLOSED. (1) Instrument:
`SO101Sim(flip_camera_mount=)` toggle + parallel-driver merged-stats
fallback + `--no-mount-flip`; harness oracle 5/5, check.py 773 green.
(2) First paired read (both arms parallel workers=8, same 20 seeds):
~null, 18/20 bit-identical — **superseded**: it measured only the
collision boxes. (3) **Owner-caught bug, root-caused**: MuJoCo stamps
geoms whose frame coincides with a precomputed frame with
`geom_sameframe`, and `mj_kinematics` then never reads
`geom_pos/quat` — the bracket's visual mesh (flag 2) silently ignored
the runtime flip edit, so every video rendered the bracket
table-side. One-line fix (clear the flag after editing); verified by
hand-computed world-pose prediction (mesh (74,10,48)→(137,−22,149) mm,
ceiling-side by the camera). (4) Corrected postflip rerun: the
bit-identity oracle failed CORRECTLY — 13/20 seeds changed (the
bracket is visible in the top cam; policy input changed; fixed render
is MORE real-matching). TRUE flip effect: knock-aways 6→2 (s4 −12.3
→ −0.05, s5 −5.5 → +0.1), mean −1.21 → −0.46 cm, paired +0.75 cm
CI95 [−0.33, +2.26], 9 exact ties; character shift shoving→freezing
(encoder-OOD probe remains the named follow-up). Physics-side claims
(control loss −62%, sweep 31.9%→1.4%) box-driven — stand. **Lesson
registered: every runtime `geom_pos/quat` edit must clear
`geom_sameframe`** (existing runtime edits audited: cameras/materials
unaffected). Banked incidentals: lockstep-parallel bit-reproducibility
at workers=8; parallel-vs-seq outcome drift (11/20 seeds >0.1 cm, max
6.0). (5) **Owner-extension step-500 arm** (16:37Z ask → 16:54Z
numbers): checkpoint converted fresh
(`outputs/converted/molmoact2_rig_r1_step500`), same 20 seeds, fixed
sim — mean +0.02 vs step-2000's −0.46 cm, paired +0.48 CI95
[−0.06, +1.13] (9 better/3 worse/8 tied), knock-aways 1 vs 2, day's
best approach s0 +1.59 cm: the extra 1500 ft steps buy no sim-side
competence (consistent with fine-tune narrowing toward rig
appearance). Rows + 80 videos + stills on fontaine-reports
`/ftrig_eval20_flip_parallel/` (curl 200); pre-reg page carries
results + correction + extension; Discord
16:02/16:31/16:32/16:47/16:54Z.

**Next**: `queue_cli.py next` → CPU lanes: `lit-sim-improvement-levers`,
`sim-wrist-compositing`. GPU: idle until the owner answers the
v3-rerun unhold ask (15:13Z — the rerun is the re-baseline carrier
for every banked sim row post-flip, now WITH the render fix in);
`grpo-signal-probe` owner_hold. `queue.json` canonical.*

## Utilization footer

Session 2026-08-12 17:29–18:1xZ (work, bounded; **+~0.19 GPU-h** —
convmap tripwire probes + one 20-seed parallel arm, ridden end-to-end;
exploit, owner prio): release-eval20-convmap DONE same session as
queued — rebase onto box's molmo_norm machinery, seam instrument +
tripwires + oracles, both pre-GPU gates dispositioned (elbow and
wrist_roll overrides earned by coverage + first-action evidence, not
assumed), 20-seed read INERT 0.00×20 with verified shim, cross-check
banked to the box in-channel. No steering traffic; 2 result posts.

Session 2026-08-12 17:20–17:3xZ (tick, babysit; 0 new GPU-h — GPU
idle): owner prio 17:13:24Z landed (released-checkpoint-in-sim +
unit-contracts note) → note read in depth, design ack posted 17:22Z
(case-3 shim, off-contract `_convmap`, two pre-GPU tripwires), owner
👍; item queued with pre-reg page as first GPU claim; exit-1 harness
alert root-caused benign (API-529 storm post-commit); run_work_next
armed → the prio item rides the chained work session. Archive roll:
1 main entry (11:25 work), 2 footer notes (15:11 tick, 13:10 work).

Session 2026-08-12 15:31–17:0xZ (work, bounded; **+~0.45 GPU-h** —
ftrig_eval20_flip_parallel, 5 arms × 5.4 min at workers=8, ridden
end-to-end; exploit, owner prio): flipped-physics rerun closed
(~25 min ask→numbers), then an owner-caught render bug OVERTURNED the
first readout — MuJoCo `geom_sameframe` was swallowing the runtime
mesh flip; fix + corrected rerun same session. TRUE flip effect:
knock-aways 6→2, paired +0.75 cm (CI crosses zero), 13/20 seeds moved
via the vision channel. Owner-extension step-500 arm: earlier
checkpoint slightly better (paired +0.48, 9/3/8). Lesson registered
(clear `sameframe` on runtime geom edits) + two incidentals (parallel
bit-reproducibility; oracle-FAIL drift at outcomes). 4 owner messages
dispositioned live; v3-rerun unhold ask still open.

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
