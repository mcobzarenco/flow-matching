# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 04:09–07:2xZ (real `date -u` at write: 07:15) —
work session: **sim-visual-matching CLOSED end-to-end — pre-reg,
every named appearance axis landed (ships as `render_style="v1"`
default), and the registered bar was MISSED honestly: top-cam 5-NN
AUROC 0.890 → 0.876 vs the ≤0.790 target. The encoder separates sim
from real on signal beyond scene layout, lens geometry and color
statistics; real-frame inpainting queued as the named lever.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB
(probe reads ran foreground, ~0.12/0.5 GPU-h gate). Next items:
**sim-visual-inpainting** + **sim-wrist-periphery-fix** (both CPU +
probe minutes); **sim100-v1-rerun** flipped to owner_hold (probe
gate missed; 20-seed behavioral spot-check offered in-channel — the
geometry fixes change where things appear without moving encoder
AUROC, er60k's reach-over-the-table fingerprint is a
pinhole-vs-fisheye signature).

**Steering**: none — Discord read at boot and pre-post both empty of
owner messages (owner asleep since 01:11Z); their promised pre-reg
posted 04:15Z, results + owner-decision ask posted 07:1xZ.

**Done**: **sim-visual-matching CLOSED** (commits 7ae1c8c pre-reg,
5c281f2 close): reset-render probe instrument
(`sim_encoder_ood_probe.py --render-resets/--appearance-draws`,
tick-0 validation 0.887≈0.885), v0-render baseline 0.890/0.835
(tripwire passed), then texture rebuild (plank direction/scale,
central-band stats matched to ~2/255), real clutter layout, wrist-cam
re-pose (menagerie had the moving jaw mirrored + camera staring into
the gripper body), 72°-source center-matched equidistant fisheye,
AWB grade, sensor blur/noise (labeled amendment), appearance-jitter
RNG; physics oracles 5 green (qpos bit-identical across appearance
seeds/render styles, spawn stream bit-matches banked sim100).
Reads: scene 0.892 / fisheye 0.874 / grade 0.881 / sensor 0.876 vs
bar 0.790 — miss; wrist content-sensitive (0.786 scene-only best,
fisheye+grade regress to 0.900); sensitivity 20×5: jitter moves
per-seed k ~3%, sim ~10× too homogeneous. 6 probe jsons + 2
before/after composites on fontaine-reports (curl 200), results post
+ reports.md section, queue: inpainting + wrist-periphery queued,
rerun owner_hold.

**Next**: `queue_cli.py next` → **sim-visual-inpainting** (or the
owner's rerun call if it lands first). `run_work_next` armed. No
dated boundaries — `queue.json` canonical.*

*Updated 2026-08-12 04:05–04:1xZ (real `date -u` at write: 04:07) —
tick (babysit): **quiet tick — GPU idle-by-design between the OOD
probe close and the sim-visual-matching pre-reg; no owner messages
or reactions; chained work session already armed.***

**Status**: no live jobs — registry empty (`no_live_runs_reason`
current), `nvidia-smi` 0% / 0 MiB. Next item:
**sim-visual-matching** (CPU + render minutes; pre-reg combines the
measured OOD baseline — move top-cam 5-NN AUROC 0.885 → ~0.5 —
with matching v1 + a 20-seed texture-sensitivity read), then
**sim100-v1-rerun**.

**Steering**: none — Discord read empty; history shows our 03:47Z
probe-result post as the latest message, no new reactions (owner
asleep since 01:11Z).

**Done**: babysit CLI skipped per registry (no live entry); queue
validate green (depth 2, 11 open); `run_work_next` confirmed armed →
work session chains at tick end. Archive roll: 00:40 body entry +
03:25/00:40 footer notes.

**Next**: chained work session → **sim-visual-matching** pre-reg
(in-channel first, promised 01:30Z; owner goal: ≥1 success on the
100 seeds). No dated boundaries — `queue.json` canonical.*

*Updated 2026-08-12 03:30–03:5xZ (real `date -u` at write: 03:49) —
work session: **encoder OOD probe CLOSED end-to-end — the visual gap
is REAL and measured at the policy's eyes, top-cam-heavier, but sim
sits at the EDGE of the real manifold, not off it. The
sim-visual-matching pre-reg now has its baseline: move top-cam 5-NN
AUROC 0.885 → ~0.5.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB
(probe ran foreground, ~0.02 GPU-h). Next GPU item:
**sim100-v1-rerun** (queued this session, pends the visual-matching
landing; probe re-read is its go/no-go gate).

**Steering**: none — Discord read empty at boot (owner asleep since
01:11Z); their 01:11Z visual-gap question now has a measured answer
in-channel (launch note + results post + chart, 03:3x–03:4xZ).

**Done**: **sim-encoder-ood-probe CLOSED** (this commit): launch note
pre-GPU, `fontaine/scripts/sim_encoder_ood_probe.py` (er_60k
eval-mount vision trunk, pinned frames: 300 sim er60k-arm + 300 real
v2 A/B-split + 100 clean anchor per camera), AUROC oracle tests (5
green). Reads: centroid AUROC top 0.802 / wrist 0.707; 5-NN
secondary top 0.885 (ratio 1.54×) / wrist 0.828 (1.33×); clean
control INSIDE the real spread (0.26/0.28) = shift is sim-specific;
sim renders 7× too homogeneous (lighting/blur diversity is part of
the gap); per-tick flat = scene not poses. Artifacts: analysis json +
strip chart on fontaine-reports (curl 200 ×2), reports.md section,
house dark chart via `sim_encoder_ood_chart.py`. Queue:
sim-visual-matching enriched with the measured baseline,
sim100-v1-rerun queued as successor.

**Next**: `queue_cli.py next` → **sim-visual-matching** (CPU + render
minutes; pre-reg promised in-channel 01:30Z combines this baseline +
matching v1 + 20-seed texture-sensitivity read). `run_work_next`
armed. No dated boundaries — `queue.json` canonical.*

## Utilization footer

Session 2026-08-12 04:09–07:2xZ (work, exploit; ~0.12 GPU-h foreground
probe reads): sim-visual-matching closed — pre-reg + all appearance
axes landed as render_style v1 (texture, layout, wrist re-pose,
fisheye, grade, sensor, jitter), registered bar missed honestly (top
5-NN AUROC 0.890→0.876 vs ≤0.790), inpainting + wrist-periphery
queued, sim100-v1-rerun flipped owner_hold with a spot-check ask
in-channel. GPU otherwise idle; run_work_next armed.

Session 2026-08-12 04:05–04:1xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design between the OOD-probe close and the visual-matching
pre-reg): quiet tick. Registry empty, nvidia-smi 0%/0 MiB. Discord
read empty; no new reactions (owner asleep since 01:11Z). Queue
validate green (depth 2, 11 open); run_work_next armed → the
sim-visual-matching pre-reg chains next. Archive roll: 00:40 body
entry + 03:25/00:40 footer notes.

Session 2026-08-12 03:30–03:5xZ (work, exploit; ~0.02 GPU-h foreground
probe): sim-encoder-ood-probe closed end-to-end — launch note, probe
script + AUROC oracles, measured baseline (top 5-NN AUROC 0.885 /
wrist 0.828; clean control inside the real spread), json + chart on
fontaine-reports, results in-channel, reports.md section;
sim100-v1-rerun queued as successor. GPU idle otherwise;
run_work_next armed for the sim-visual-matching pre-reg.

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
