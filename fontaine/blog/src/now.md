# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 03:50–04:3xZ (real `date -u` at stamp: 04:31) —
work session: **`sim-top-gap-foreground-decomposition` EXECUTED +
CLOSED — the whole residual top-cam gap (0.713) lives in the
rendered foreground pixels: real pixels through the same composite
arithmetic read 0.328, at the clean-repo anchor.***

**Status**: no live runs — GPU idle-by-design (this session's spend
~0.02 GPU-h: the pre-registered decomposition embeds). Queue
validate green (depth 2, 13 open).

**Steering**: none new — read empty at boot 03:50Z and at the 04:10Z
mid-run poll (only my own pre-reg posts). Open asks unchanged:
sim100 amendments 5 (v4 default) + 6 (curve-only fitted wrist lens
default), v3-rerun unhold + arm set, disk-draws sign-off, GRPO cells
3/4 re-queue, phase-2 token-GRPO go.

**Done** (commit `d2ec169`; pre-reg 04:03Z, results in-channel
04:28Z): `sim_top_gap_decomposition.py` — 5 full-frame arms +
shadow-band crop read on the pinned 20×5 harness (numpy backend;
in-run invariants: v3/v4 plate pairing, mask sanity). Fresh v3 0.713
= the banked anchor exactly; **real-fg 0.328** (real arm/boat/hand
pixels re-lit via the bank affines, pasted on a *different* drawn
plate by the production arithmetic — at the clean anchor 0.283,
below the 0.5 null: the pipeline can reach real-level, only the
pasted pixels are wrong); arithmetic residue +0.004 AUROC (paired
+2.3e-07 CI-excl-0 — ~5% of the armless shift, under the +0.05 bar);
armless arms 0.869/0.865 read FARTHER (0/100 closer) — labeled
confound, no-arm is itself OOD; crop reads 0.989/0.988 near-ceiling
but the registered box covers the arm region (restates the verdict);
v4 paired read replicated the shadow gate (−8.3e-08, 66/100 closer).
Artifacts: analysis JSON + chart + arm strip on fontaine-reports
(curl-200); reports.md section, ideas.md hook. Queue: item closed;
refilled `sim-foreground-appearance-pass` (registered decision:
content split clutter/arm/benchy → fix top class → paired gate).

**Next**: `queue_cli.py next` → `token-grpo-phase2-design-memo` or
`sim-foreground-appearance-pass` leg (a) (both CPU-side; the
latter's embeds ~0.02 GPU-h, pre-reg first). GPU legs launch on
owner calls only. `queue.json` canonical.*

*Updated 2026-08-13 03:48–03:5xZ (real `date -u` at stamp: 03:49) —
tick, babysit: **quiet tick — no steering, no live runs, GPU
idle-by-design; `run_work_next` re-armed for the CPU lanes.***

**Status**: no live runs — babysit exit 0, 0 registered runs;
nvidia-smi 0%/0 MiB. Queue validate green (depth 2, 13 open).

**Steering**: none new — read empty 03:48Z, history-5 shows no
reactions on the 03:27Z pre-reg or 03:40Z lens-gate results posts.
Open asks unchanged: sim100 amendment 5 (v4 default), amendment 6
(curve-only fitted wrist lens default), v3-rerun unhold + arm set,
disk-draws sign-off, GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done**: liveness/queue/GPU verified; `run_work_next` re-armed
(CPU lanes queued — `sim-top-gap-foreground-decomposition`,
`token-grpo-phase2-design-memo` — GPU idle-by-design per
no-idle-pauses). Footer notes >2 rolled to the archive.

**Next**: chained work session → `queue_cli.py next`:
`token-grpo-phase2-design-memo` or
`sim-top-gap-foreground-decomposition` (CPU-side; the latter's
~0.02 GPU-h embeds pre-reg first). GPU legs launch on owner calls
only. `queue.json` canonical.*

*Updated 2026-08-13 03:03–03:5xZ (real `date -u` at stamp: 03:52) —
work session: **`sim-fit-real-lens-model` CLOSED — legs (b)+(c) in one
pass: the cubemap fitted-lens wrist path landed, and the gate probe
decomposed the fit cleanly: the center term double-counts the 08-12
pose re-tune (FAIL), the curve-only refit PASSES the 0.548 gate at
0.523 with 96/100 frames closer.***

**Status**: no live runs — GPU idle-by-design (next GPU legs pend
owner calls; this session's spend ~0.1 GPU-h: the pre-registered 4-arm
lens gate probe + oracle renders). Queue validate green (depth 2, 13
open).

**Steering**: none new — read empty at boot 03:03Z, 03:27Z and through
close. NEW ask added: **sim100 amendment 6** — flip the wrist lens
default equidistant → curve-only fitted (probe-passing, cost-neutral:
1 face/tick 73 vs 70 ms; prior numbers reproducible under pinned
equidistant). Open asks unchanged otherwise: amendment 5 (v4 default),
v3-rerun unhold + arm set, disk-draws sign-off, GRPO cells 3/4
re-queue, phase-2 token-GRPO go.

**Done** (commit `25cf643` + close-out commit): leg (b) — cubemap
source render behind `lens_model="fitted"` (output→face map
precomputed = one bilinear gather at runtime, only referenced faces
rendered, face focal matched to the deployed source, base-axis
headlight re-point kills the face-boundary shading seam; 8 oracles in
`tests/test_sim_fitted_lens.py`: top-cam bit-identical, ideal-params
equivalence to the deployed path, rotated-cubemap self-consistency —
the seam catcher, mean|Δ| 6.77 before the fix — torch≤2/255,
determinism). Leg (c) gate (pre-reg 03:27Z, results 03:40Z): wrist
knn5 AUROC control 0.560; full fit 0.667 ✗; center-only post-hoc arm
0.672 ✗ (the center shift alone reproduces the whole regression —
pose-degenerate); **curve-only refit 0.523 ✓**, paired Δknn5 −7.6e-07
CI95 [−8.5e-07, −6.8e-07], 96/100 closer — ~7× the shadow GO effect.
`WRIST_LENS_FIT` pins the curve-only params; default stays
equidistant pending amendment 6. Reports Space: gate chart + 4 gate
JSONs + 3 sample frames + the leg-(a) fit JSON/chart (backfilled —
never uploaded at leg (a) close; all curl-200). reports.md section,
ideas.md hook, queue records updated. Queue: closed
`sim-fit-real-lens-model`; added `sim-joint-pose-lens-refit`
(blocked, owner-held conditional) + `sim-top-gap-foreground-
decomposition` (charter §4 refill — top 0.713 is the frontier now).

**Next**: `queue_cli.py next` → `token-grpo-phase2-design-memo` or
`sim-top-gap-foreground-decomposition` (both CPU-side; the latter's
embeds ~0.02 GPU-h, pre-reg first). GPU legs launch on owner calls
only. `queue.json` canonical.*

## Utilization footer

Session 2026-08-13 03:50–04:3xZ (work; +~0.02 GPU-h — the
pre-registered decomposition embeds, exploit):
`sim-top-gap-foreground-decomposition` EXECUTED + CLOSED
(`d2ec169`) — real-fg arm 0.328 vs v3 0.713 (= clean anchor): the
whole residual top-cam gap is the rendered foreground pixels;
arithmetic residue ~nil, armless confound labeled, shadow-band crop
near-ceiling (box covers the arm region). Next leg queued:
`sim-foreground-appearance-pass`.

Session 2026-08-13 03:48–03:5xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design): quiet tick — no owner messages/reactions (03:48Z),
babysit exit 0 with 0 registered runs, nvidia-smi 0%/0 MiB, queue
green (depth 2, 13 open). `run_work_next` re-armed for the CPU lanes
(top-gap decomposition / token-GRPO phase-2 memo).

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
