# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 14:53–15:2xZ (real `date -u` at write: 15:20) —
work session: **rigonly CLOSED CLEAN 14:52Z (~10.5/12 GPU-h) and the
drift-saga consolidated page is LIVE — the queue-next chart-led record
of the whole investigation, with the rigonly ambiguous-leaning-drift
verdict folded in. Owner agreed with the ambiguous reading 15:07Z;
the discriminator go/no-go ask is in-channel.***

**Status**: NO live runs — box 8×A100 idle (rigonly unit inactive,
1000/1000, all 4 saves on disk) + local H100 idle (eval chain done
14:17:56Z). All three box runs' train logs rsynced local BEFORE any
cleanup (`outputs/train/rigonly_artifacts/`); saves kept on box
(rigonly 250–1000, mixedv2 + demosonly 500/1000; diagnostic
checkpoints, curves fully banked — not uploaded, consistent with the
demosonly/mixedv2 precedent). Next GPU leg = the staged 1-GPU
discriminator, OWNER-GATED (ask posted 15:14Z, msg 1538929076079689849).

**Steering**: 15:07Z "Agreed with your ambiguous reading" → replied
15:14Z (the verdict post opens as the reply) + acked same-minute.
Discriminator question pending — tight-polling per the standing rule.

**Done**: drift-saga report page live + curl-verified
([page](posts/2026-08-17-sft-drift-saga.md), commit `7d80edd`): 4
dark-mode charts via `sft_drift_saga_charts.py` (2×2 curve grid, the
indexed-drift overlay demosonly +2.93 / mixedv2 +2.33 / rigonly +0.69
/ run-2 −0.92, two-rulers loss-vs-MAE, head-asymmetry bars), curves
banked `reports/curve__sft_drift_saga.json` + mirrored to the reports
Space (curl 200); rigonly babysit entry PRUNED with completion record
+ no_live_runs_reason declared; queue: `sft-drift-saga-report-page`
DONE, `sft-drift-discriminator-run` added (blocked, owner_hold,
prereg → the frozen launcher header), depth-1 reason stated
(experimental frontier deliberately owner-gated); blog built + Space
pushed.

**Next**: owner's discriminator call (on GO: cut the formal pre-reg
post from the script header BEFORE launch, babysit entry, first-poll
util check; alternative offered: rigonly continuation past 1000).
`queue_cli.py next` → `sft-v1-eval-chain-html-panel` (CPU).
Owner-pending: discriminator go, G1-miss ride 👍, augment-report
reaction, disk composite exemption, approach redesign go, v2.1 bands,
ckpt-format, morning-veto items.*

*Updated 2026-08-17 14:27–14:4xZ (real `date -u` at write: 14:33) —
tick: **both promised boundaries banked — eval-chain ALL DONE
14:17:56Z, leg 3 endpoint token-with-fix **14/100** (vs step500 token
16/100: the token head is ~flat across training while flow stayed
collapsed 4→5 — head asymmetry holds at both ends); rig-only @500
eval MAE **8.82** / train **4.62**, DOWN from @250's 9.24/5.53 on
both slices — opposite of the drift signature so far.***

**Status**: `grasp_sft_rigonly_8xa100` step ~690/1000 at this poll,
~3.8 s/step, 8×99% util, losses falling (0.67); @750 ridden
in-session: eval MAE **9.15** / train **4.03** — eval wobbled up from
@500's 8.82 (still below @250's 9.24; holdout is 6 episodes) while
train fell monotone 5.53→4.62→4.03. @1000 landed 14:52Z at the
session wire: eval **9.51** / train **4.23** — eval rose monotone
from 500 (dip-then-rise, the drifting-run SHAPE, ending above @250)
and train ticked up for the first time. AMBIGUOUS-LEANING-DRIFT
posted honestly (magnitude +0.69 vs demosonly's +2.9 over the same
span; 6-ep holdout); if real ⇒ recipe/stack, discriminator is the
next cut. Full verdict + charts owed by the chained work session
(healthy = corpus implicated, drifting = recipe/stack convicted; the
staged 1-GPU discriminator is the complementary cut, owner decides;
rsync eval artifacts local BEFORE any box cleanup). Local
H100 FREE as of 14:17:56Z (chain done, ~6.2/12 GPU-h).

**Steering**: none new (inbox empty, `read` empty of owner messages;
history — no new reactions).

**Done**: leg-3 result computed from `token_s0.json` (14 successes,
seeds listed; median progress 0.69 cm, 54/100 moved >0.5 cm —
consistent with the 3/20 seeds-100-119 sample at 15%); combined
verdict + @500 read posted (1538917693032243293); `sft_v1_eval_chain`
babysit entry PRUNED with its completion record; queue +1
(`sft-v1-eval-chain-html-panel`, CPU) → depth 2 validated;
`run_work_next` armed (box busy + CPU items queued); 08:52 entry + 2
footer notes rolled to the [08-17 archive](archive/now-2026-08-17.md).

**Next**: chained work session — drift-saga report page (queued,
draftable now; finalize slot for the rigonly verdict) + eval-chain
HTML panel; rig-only @1000 boundary ~15:0xZ (post-process per charter
§4: MAE curve verdict in-channel, rsync eval artifacts local BEFORE
any box cleanup, then the discriminator question to the owner).
Owner-pending: G1-miss ride 👍, augment-report reaction, disk
composite exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

*Updated 2026-08-17 10:19–13:4xZ (real `date -u` at write: 13:08,
amended 13:41) — work session: **the day the story flipped twice. v1
endpoint tail closed by reconstructing sim100 from logs (the box wipe
had destroyed the merged artifacts — disclosed); owner burst (10
messages) killed the mixed v2 run and launched demos-only; that run
REPRODUCED the MAE drift under a demos-native table — mix/table
exonerated — and was killed too; the owner's rig-only data-axis cut
is now live. Plus: run-2's step500 TOKEN head reads 16/100 — the flow
collapse was head-specific.***

**Status**: (1) `grasp_sft_rigonly_8xa100` on the box since 13:34:08Z
(unit `fontaine-grasp-sft-rigonly`, owner-designed data-axis cut:
rig datasets only, 2 ds / 51 eps / 32,431 frames ~3 epochs, 1000
steps, save+eval 250, recipe otherwise verbatim incl. the full
distributed stack, rig-native recompute table): boundary ~15:0xZ —
drift on known-good rig data convicts the recipe/stack, health
implicates the sim-demo corpus. Predecessor demosonly KILLED 13:30Z
at ~1350 (drift fully reproduced: eval 3.46→3.24→4.22→5.27→6.17,
train 3.69→3.32→3.86→4.60→5.62, monotone from 500, losses falling
throughout; saves 500/1000 kept). The 1-GPU single-delta
discriminator stays STAGED on the box
(`launch_box_grasp_sft_v2_demosonly_1gpu_discriminator.sh`) as the
complementary cut. (2) `sft-v1-eval-chain` local H100, leg 3 of 3
(endpoint token-fixed sim100) since 12:12:02Z, ETA ~14:1xZ, 4.6/12
GPU-h projected — the owner's full-100 endpoint token number; leg 2
banked in-session.

**Steering** (8 messages, all replied + acked same-hour): sim100
board reminder (10:20) + probe-protocol question (10:24) → both
answered from banked artifacts; sim20-on-step500 order (10:54, they
rsynced the ckpt themselves 10:57) → run + result posted 0/20 with
paths; kill-mixed + demos-only order (11:27/11:28) → executed
11:38:30Z with delta posted pre-launch; exact-sim-command ask (11:30)
→ verbatim command posted; losses-down-MAE-up question (11:40) →
two-rulers answer (normalized/tokenized loss space vs raw-degree MAE;
1/(q99−q01)² channel weighting + clamped targets).

**Done**: (a) **v1 endpoint boundary tail CLOSED via log
reconstruction** (`d464ac6`, `afe7d44`): the 05:5xZ box `outputs/`
wipe had deleted the merged sim100 jsons + videos before their
rsync-local step — per-seed data reconstructed exactly from the
surviving shard logs (5/100, 0/100, moved 51, median 8.65 all
reproduce; videos = only true loss), incident disclosed in-channel +
results page, [results page](posts/2026-08-16-grasp-sft-v1-results.md)
finalized + registered in SUMMARY (was 404), v1endpoint HTML report
live on the reports Space, memory rule upgraded near-miss→realized.
(b) **Correction on the record**: run-2 step500 flow is **4/100** not
the tick-posted 2/100 (results page + queue fixed, posted). (c)
**sim20 on mixed-v2 step500: 0/20** vs run-2's 1/20 same seeds
(honest no-anchor-at-500 framing). (d) **Mixed v2 killed** (owner
order, step ~1150, ~2.6 GPU-h; MAE curve banked) → **demos-only
launched 11:38:30Z** (`a58251f`), banner verified 1 ds / 4500 eps /
1.75M frames. (e) **Eval-chain leg 2: run-2 step500 token 16/100** —
flow 4 vs token 16 at the same step; CE weights channels uniformly,
flow MSE ∝ 1/(q99−q01)² — the table poisoned the flow head's loss
weighting specifically. (f) v2 + demosonly endpoint kits staged
(`698298e`, `5cfe517`: box eval scripts, upload scripts, report
`--run v2`, v2endpoint HTML preset). (g) Queue truth-up: 3 stale
statuses corrected, +3 items, kit item closed same-session.

**Next**: rigonly boundary ~15:0xZ (tick chain: MAE-curve verdict vs
the drifting-run signature, then the next cut — staged 1-GPU
discriminator or owner's pick). Leg-3 boundary ~14:1xZ (tick rides
it: full-100 endpoint token vs step500's 16 — degradation read).
`queue_cli.py next` → `sft-drift-saga-report-page` (CPU, draftable).
Steering additions 13:27/13:30 (both served): DDP-prior push-back →
agreed + honest delta-list refinement; kill + rig-only order →
executed 13:34:08Z. Owner-pending: G1-miss ride 👍, augment-report
reaction, disk composite exemption, approach redesign go, v2.1
bands, ckpt-format, morning-veto items.*

## Utilization footer

Session 2026-08-17 14:53–15:2xZ (work, exploit; box: rigonly ridden to
its 14:52Z close ≈ 10.5/12 GPU-h claimed at completion; local idle,
zero new GPU-h): **drift-saga consolidated page shipped same-session
as the rigonly verdict (4 charts, curves banked + mirrored), babysit
pruned + no-live-runs declared, queue truth-up (+discriminator item,
owner-gated), owner 15:07Z agreement replied + acked, discriminator
ask posted** — GPUs idle by design pending the owner's word,
`run_work_next` armed for the CPU queue.

Session 2026-08-17 14:27–14:4xZ (tick; box busy with rig-only ~690/1000
ridden not claimed; local H100 freed 14:17:56Z by the chain's ALL
DONE): **eval-chain closed at ~6.2/12 GPU-h — leg 3 endpoint
token-fixed 14/100 banked + posted (token head ~flat 16→14 across
training vs flow collapsed 4→5); rig-only @500 read posted (8.82/4.62
falling, anti-drift so far); babysit entry pruned, queue +1 (HTML
panel), depth 2** — inbox clear, `run_work_next` armed.

Session 2026-08-17 10:19–13:5xZ (work, exploit; box: mixed v2 ridden
to the owner kill at ~1150 ≈ +2.6 GPU-h, demosonly launched
11:38:30Z → killed 13:30Z at ~1350 ≈ +4 GPU-h with the drift
REPRODUCED, rig-only cut launched 13:34:08Z live ~1.3 proj / 12
gate; local: sim20 on mixed step500 +~0.5 GPU-h owner-ordered, eval
chain legs 2–3 ridden not claimed): **v1 endpoint tail closed via
log reconstruction (wipe incident disclosed), 10 owner messages
served, two runs killed on their signatures and the data-axis cut
launched (mix/table exonerated, config-delta table honest-refined,
1-GPU discriminator staged), run-2 step500 token 16/100 banked
(flow-specific collapse), 2/100→4/100 correction posted** — queue
depth 1 with stated reason, `run_work_next` armed at close.

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
