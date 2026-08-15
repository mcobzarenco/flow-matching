# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-15 12:42–13:4xZ (real `date -u` at stamp: 13:44) —
work session: **probe COMPLETE — no memorization signature (trained
9/64 vs unseen 28/100); full remit discharged; GPU handed to the
owner on their order.***

**Status**: no live jobs. `fontaine-grasp-sft-step2000-probe` DONE
clean at ~13:41Z (~3.4/4.0 GPU-h, 0 strikes, babysit entry pruned) —
FINAL three-way: trained-kept **9/64 (14%)**, expert-failed **9/36
(25%)**, unseen **28/100 (28%)** vs base anchor 9/100; the
memorization signature is decisively absent (inversion ~2 SE,
suggestive only). **GPU 0% — RESERVED BY THE OWNER** (13:35Z: "I'll
actually need the gpu"); no launches until they free it.

**Steering**: three owner messages, all replied + acked. (1) 13:09Z
continue-from-2k-under-corrected-table question → answered
(first-class supported via `--norm-stats-from` on the step2000-hf
export; recommended as primary arm over from-base — same cost, warm
features; expect early loss spike from the I/O rescale, wrist_roll
~3×); amendment proposed, **owner pick pending**. (2) 13:09Z sim2real
image augmentation → answered (nothing image-side wired today;
train-time photometric aug = the cheap insertion point) + queued
`image-augment-sim2real` (CPU item). (3) 13:35Z "nothing right away,
I'll need the gpu, ping me at job end" → finish-ping + final
comparison posted at the boundary (1538180830470602903).

**Done** (commits `75a0379` + close commit): step2000 delta uploaded
(590/705 tensors, rig-r1 pattern) →
`fontaine-checkpoints/molmoact2_grasp_sft_stagec_ar_step2000`;
retrain prep landed — `build_corrected_norm_stats.py` (5 oracles) →
corrected artifact (wrist_roll q01/q99 → ±157.2), base converted
under it (`molmoact2_base_corrected_stats_v0`, rows verified baked);
retrain pre-reg DRAFT posted (page + in-channel, owner-gated); probe
reads run + `probe_bands` chart + probe section live on the chain
results page; babysit pruned; blog ×2 Space pushes (both curl-200).
In-session GPU launched: 0 (probe ride-through ~0.9 GPU-h of its
3.4 total).

**Next**: `queue_cli.py next` → grasp-sft-bootstrap retrain decision
is **owner-pending** (continue-from-2k vs from-base + go; GPU also
owner-held — both must clear before any launch).
`image-augment-sim2real` (CPU, queued 13:4xZ) is the executable
no-GPU slice. R2 draft amendment (token-SFT-before-token-GRPO) still
owed on its own item. `run_work_next` armed.*

*Updated 2026-08-15 12:40–12:4xZ (real `date -u` at stamp: 12:41) —
tick: **train arm riding green at ~seed 1046/1099 (ETA ~13:5xZ);
owner 👍 on the 9/100-anchor correction recorded.***

**Status**: **LIVE** — `fontaine-grasp-sft-step2000-probe` train arm:
~seed 1046/1099 at 12:40Z, 4 procs, GPU 38%/11.9 GiB, 11.2 f/min
window, cumulative projection 2.5 vs the 4.0 GPU-h gate. Babysit exit
0, no gate crossings. Mid-arm tally (posted 12:39:56Z by the prior
session, 45/100 done): trained spawns 6/37, expert-failed 4/9, unseen
28/100 — no memorization signature so far.

**Steering**: owner **👍 on the 12:01Z anchor-correction reply** — agreement registered: **9/100 released-base is the
primary anchor** (causal SFT read 9 → 28 ≈ 3.1×, a floor given the
corrupt table); ftrig4k/W0 demoted to context rows. Ledger + probe
report page must carry 9/100 as headline comparator. Inbox empty, no
new messages; the 11:58/12:03 exchange quiet since 12:04Z reply.

**Done**: babysit + Discord polls, queue validate OK depth 3 (17
open), `run_work_next` confirmed armed (11:54 touch). No posts (quiet
tick, nothing owner-facing changed). 0 GPU-h.

**Next**: train-arm boundary ~13:5xZ belongs to the chained work
session — reads script → three-way comparison post (9/100 as primary
anchor), then the queued remit: step2000 delta upload, owner-gated
corrected-table `bijou.train` retrain prep, probe report page.*

## Utilization footer

Session 2026-08-15 13:50–13:5xZ (tick; 0 GPU-h): quiet hold — GPU
owner-reserved and idle (0%, untouched since the 13:41Z handoff), no
launches; Discord/inbox/history empty, retrain arm pick + GPU release
both owner-pending; queue validate OK depth 4 (18 open),
`run_work_next` confirmed armed for the CPU items
(`image-augment-sim2real`, R2 amendment).

Session 2026-08-15 12:42–13:4xZ (work; exploit; 0 GPU-h launched —
probe ride-through ~0.9 of its 3.4 total): remit discharged end to
end — step2000 delta uploaded, corrected-table retrain prep landed
(table artifact + base conversion + owner-gated pre-reg DRAFT), probe
boundary executed (train arm banked, no-memorization read posted),
report page + chart live; 3 owner messages replied+acked, GPU handed
to the owner at their 13:35Z ask; queue depth 4, `run_work_next`
armed (CPU item queued).

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
