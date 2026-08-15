# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-15 14:26–14:2xZ (real `date -u` at stamp: 14:27) —
tick: **quiet hold — GPU owner-reserved and idle (0%), no launches;
owner is building on main (checkpoint-format commit, no message).***

**Status**: no live jobs; GPU 0% / 0 MiB — still **RESERVED BY THE
OWNER** (13:35Z), untouched. No babysit entries.

**Steering**: none — Discord read empty, inbox empty, history shows
nothing new past the recorded 🎉. Retrain arm pick
(continue-from-2k vs from-base) and GPU release both still
**owner-pending**. Context (not steering): owner pushed `4fd6875` to
**main** at 13:56Z — phase 3 VLA checkpoint format
(`bijou/checkpoint.py`: VLAMetadata schema v1, write/validate
checkpoint, backbone snapshot mirroring, `convert_legacy`). They're
actively building; the chained work session should skim it for
interaction with our retrain/export tooling (e.g. the step2000-hf
export path and `--norm-stats-from` seam).

**Done**: Discord + history polls, GPU/process check, queue validate
OK depth 3 (17 open), `run_work_next` confirmed armed (14:24 touch).
No posts (nothing owner-facing changed). 0 GPU-h.

**Next**: chained work session takes the R2 draft amendment
(token-SFT-before-token-GRPO seam) and should read `4fd6875` for
checkpoint-format implications; retrain launch stays parked until
the owner picks an arm AND frees the GPU.*

*Updated 2026-08-15 13:53–14:2xZ (real `date -u` at stamp: 14:22) —
work session: **image-augment-sim2real LANDED — `--image-augment`
train-time sim2real photometric recipe in bijou.train, v0 params
pre-registered; GPU untouched (owner-held).***

**Status**: no live jobs; GPU 0% / 0 MiB — still **RESERVED BY THE
OWNER** (13:35Z), untouched all session. No babysit entries.

**Steering**: no new messages — Discord read + inbox empty at boot
and mid-session; the close poll surfaced an owner **🎉 on the
14:21Z landing post** (acknowledgment recorded, no ask). Retrain arm
pick (continue-from-2k vs from-base) and GPU release both still
**owner-pending**.

**Done** (commit `09129af` + close commit): queue item
`image-augment-sim2real` DONE — `bijou/image_augment.py` (v0 recipe
frozen: crop/translate 0.90–1.0, brightness ±0.15, contrast/sat
0.7–1.3, hue ±0.05, gamma log-U(0.8, 1.25), noise p=.5, blur p=.25,
JPEG p=.25 q40–85), `Collator.image_augment` per-frame gate at the
CameraFrame seam; p=0 is a **bitwise** pin (identity pass-through,
zero RNG — 11 oracles incl. the probe-clone convention; eval-side
collators never set the field). check.py green (865). Pre-reg page
live with a clean-vs-7-draws grid on a real stage-B frame
([page](posts/2026-08-15-prereg-image-augment-sim2real.html),
curl-200); in-channel post 1538191003574607885 incl. the composition
recommendation (`--image-augment 0.8` on the owner-picked retrain
arm: direct = confounded vs the 28/100 floor, follow-up arm = clean
A/B ~2.9 GPU-h — owner's call). 0 GPU-h.

**Next**: `queue_cli.py next` → grasp-sft-bootstrap retrain is
**owner-pending** (arm pick + GPU release must both clear before any
launch). Remaining CPU slice: the R2 draft amendment
(token-SFT-before-token-GRPO seam on the grpo-r2-post-sft item).
`run_work_next` armed.*

*Updated 2026-08-15 13:50–13:5xZ (real `date -u` at stamp: 13:51) —
tick: **quiet hold — GPU owner-reserved and idle (0%), no launches;
everything actionable is owner-gated or queued for the chained work
session.***

**Status**: no live jobs; GPU 0% / 0 MiB — still **RESERVED BY THE
OWNER** (13:35Z), untouched since the probe handoff at 13:41Z. No
babysit (registry pruned last session).

**Steering**: none new — Discord read empty, inbox empty, history
shows no new reactions; the 13:35Z exchange quiet ~20 min, so
conversational mode handed back. Retrain arm pick (continue-from-2k
vs from-base) and GPU release both still **owner-pending**.

**Done**: Discord + history polls, GPU/process check, queue validate
OK depth 4 (18 open), `run_work_next` confirmed armed (13:44 touch).
No posts (quiet tick, nothing owner-facing changed). 0 GPU-h.

**Next**: chained work session takes `image-augment-sim2real` (the
executable no-GPU slice) and the R2 draft amendment item; retrain
launch stays parked until the owner picks an arm AND frees the GPU.*

## Utilization footer

Session 2026-08-15 14:26–14:2xZ (tick; 0 GPU-h): quiet hold — GPU
owner-reserved and idle (0%), no launches; Discord/inbox/history
empty, retrain arm pick + GPU release both owner-pending; noted owner
commit `4fd6875` on main (phase 3 VLA checkpoint format) for the
chained work session to skim; queue validate OK depth 3 (17 open),
`run_work_next` confirmed armed (R2-amendment CPU slice).

Session 2026-08-15 13:53–14:2xZ (work; exploit; 0 GPU-h): queue item
`image-augment-sim2real` landed end to end — `--image-augment` in
bijou.train (v0 recipe, bitwise off-path pin, 11 oracles, check.py
865 green), pre-reg page + grid live on the Space, in-channel post;
GPU owner-held and untouched; queue validate OK depth 3 (17 open),
`run_work_next` armed (R2-amendment CPU slice remains).

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
