# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 01:48–02:5xZ (real `date -u` at stamp: 02:58) —
work session: **`sim-composite-contact-shadows` CLOSED, gate GO — the
real arm's shadow measured from the episodes themselves, the v4
render style casts it, and the encoder moves ~10% of the remaining
top-cam gap.***

**Status**: no live runs — GPU idle-by-design (next GPU legs pend
owner calls; the only spend this session was the ~0.04 GPU-h paired
gate probe, pre-registered in-channel 02:50Z before launch). Queue
validate green (depth 2, 12 open).

**Steering**: none new — read empty at boot 01:48Z and through the
session; no reactions yet on the 02:50Z pre-reg or 02:56Z results
posts. NEW ask added: **sim100 amendment 5** — flip the default
render style v3 → v4 (costless, +1 depth pass/frame; all prior
numbers reproducible under pinned v3). Open asks unchanged
otherwise: v3-rerun unhold + arm set, disk-draws sign-off, GRPO
cells 3/4 re-queue, phase-2 token-GRPO go.

**Done** (commit `8f35560` + close-out commit): leg (a) light fit —
`fit_contact_shadow.py` (200 frames × 25 bank episodes, sim-replayed
arm silhouette vs frame/plate darkening): the shadow is real and
directional, contrast +0.091 CI95 [0.081, 0.100] vs ring control,
zenith 30°/azimuth 112.5° (85% bootstrap), strength 0.392, σ 24 px.
`sim/shadow.py` projector + `render_style="v4"` (12 oracles: wrist
bit-identical to v3, zero-strength v4 ≡ v3, torch ≤2/255,
conventions pinned analytically). Paired gate seeds 0..99: top knn5
AUROC 0.721 → 0.715 (fresh v3 arm — banked 0.673 anchor predates the
bracket flip), paired Δknn5 −1.04e-07 CI [−1.53e-07, −5.6e-08],
66/100 seeds closer, wrist 100/100 tied → **GO recorded, default
stays v3 pending amendment 5**. Reports Space: fit JSON + both gate
JSONs + chart + v4 sample (all curl-200). reports.md section,
ideas.md hook updated. Queue refilled:
`token-grpo-phase2-design-memo` (charter §4).

**Next**: `queue_cli.py next` → `sim-fit-real-lens-model` leg (b)
(cubemap→equirect→fitted-lens render path, CPU-side) or the
token-GRPO phase-2 design memo. GPU legs launch on owner calls only.
`queue.json` canonical.*

*Updated 2026-08-13 01:44–01:5xZ (real `date -u` at stamp: 01:47) —
tick, babysit: **quiet tick — no steering, GPU idle as declared; one
housekeeping kill: the stale boxsync loop (polling the dead 08-05
box for 6 days) found and stopped.***

**Status**: no live runs — registry carries the declared reason
(next GPU legs pend owner calls); nvidia-smi 0%/0 MiB, no stray
compute procs. Queue validate green (depth 2, 12 open).

**Steering**: none new — read empty 01:45Z, history-5 shows no
reactions on the 01:10Z probe results post or the 01:40Z lens post.
Open asks unchanged: v3-rerun unhold + arm set, disk-draws sign-off,
GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done**: process sweep surfaced `boxsync_loop.sh` still running
since 08-06 23:44Z in a tmux pane — ssh-polling the retired 4xH100
box (192.222.55.210) every 20 min; box confirmed unreachable
(connection timeout), all its registry entries historical → loop +
hung ssh killed. `run_work_next` re-armed (CPU lanes queued, GPU
idle-by-design). Footer notes >2 rolled to the archive.

**Next**: chained work session → `sim-composite-contact-shadows`
(queue head) or lens leg (b) render path; phase-2 token-GRPO design
memo open. GPU legs launch on owner calls only.*

*Updated 2026-08-13 01:18–01:5xZ (real `date -u` at stamp: 01:41) —
work session (chained by the 01:14 tick): **wrist lens fit leg (a)
DONE — the real lens is measurably not ideal-equidistant.** Plumb-line
θ→r fit on the 150 pinned real wrist frames (pure CPU, no rig time):
optical center 22 px left / 14 px below the image midpoint (~5σ), and
the curve compresses the periphery −12.8 px at the frame corner vs
the deployed equidistant assumption (CI95 [−17.2, −10.0], excludes
0). Results + house chart posted in-channel 01:40Z.*

**Status**: no live runs — GPU idle-by-design (registry carries the
declared reason; next GPU legs pend owner calls). Queue validate
green (depth 2, 12 open).

**Steering**: none new — read empty at boot 01:18Z and at close; no
reactions yet on the 01:10Z probe results post or the 01:40Z lens
post. Open asks unchanged: v3-rerun unhold + arm set, disk-draws
sign-off, GRPO cells 3/4 re-queue, phase-2 token-GRPO go.

**Done** (commit `5581d6d`): `sim-fit-real-lens-model` leg (a) —
plumb-line fit instrument (`fontaine/scripts/fit_lens_plumbline.py`:
Canny → PCA/quadratic-filtered seam chains, 382 chains from 132/150
frames; Nelder-Mead over (cx, cy, k₂, k₄) with center-only /
curve-only decompositions + frame bootstrap), synthetic-recovery
oracles (`tests/test_lens_plumbline.py`, 4 tests), house dark chart
(`lens_fit_chart.py`). Plank straightness RMS 1.07 → 0.90 px;
fitted params are the stage-2 resampler spec for leg (b).
check.py 801 green. Queue item updated with the leg-(a) record.

**Next**: `queue_cli.py next` → `sim-composite-contact-shadows`
(queue head, CPU) or lens leg (b) cubemap→equirect→fitted-lens
render path (same item, CPU-side first); phase-2 token-GRPO design
memo also open. GPU legs launch on owner calls only. `queue.json`
canonical.*

## Utilization footer

Session 2026-08-13 01:48–02:5xZ (work; +~0.04 GPU-h — the paired
contact-shadow gate probe, exploit): `sim-composite-contact-shadows`
CLOSED with gate GO — real-arm shadow fit (contrast +0.091
CI-excl-0, zen 30°/az 112.5°), v4 render style + 12 oracles
(`8f35560`), paired probe closed ~10% of the remaining top-cam knn5
excess (Δknn5 CI-excl-0, 66/100 seeds closer; wrist 100/100 tied).
Default flip = sim100 amendment 5, owner ask posted 02:56Z. Queue
refilled with `token-grpo-phase2-design-memo`.

Session 2026-08-13 01:44–01:5xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design): quiet tick — no owner messages/reactions (01:45Z),
registry reason stands, nvidia-smi 0%/0 MiB, queue green (depth 2,
12 open). Housekeeping: killed the stale `boxsync_loop.sh` (up since
08-06 23:44Z, ssh-polling the dead 4xH100 box every 20 min;
unreachable, connection timeout). `run_work_next` re-armed for the
CPU lanes.

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
