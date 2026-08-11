# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-11 04:21–04:2xZ (real `date -u` at write: 04:22) —
tick (babysit): **quiet green tick right after the 04:2x work close —
box healthy, channel quiet**; run_work_next armed for the chained
work session (port item 4 / endpoint prep).*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — count
46,820, 30.5 f/min window, babysit exit 0 (8 procs, util 94–100%,
vram ~71.8×4 under the 77 bar), gate projection 118.1/155 GPU-h.
Rungs unchanged since the work close (latest 5.40@46000 / 5.40@46500;
run-best **5.10@44500** stands; 5.1–5.6 band holds). Record-only to
endpoint **@60000 ~12:3xZ** → chained panel_v2; next save boundary
@50000 ~06:2xZ. Local H100 FREE.

**Steering**: none — read empty; history -n 5 shows only our own
five posts, no new reactions.

**Done**: babysit exit 0; Discord read (empty) + history (clean);
queue validate OK depth 2 (9 open); run_work_next confirmed armed
(04:21 marker un-consumed — the chained work session is still
ahead); aged 08-11 entries (03:20 tick, 00:50 work) + their footer
notes rolled to [archive](archive/now-2026-08-11.md).

**Next**: unchanged from the 04:2x close — chained work session
picks between **er60k-endpoint-postprocess** (time-sensitive:
endpoint @60000 ~12:3xZ → chained panel_v2 → paired CI95 vs banked
40k 6.0079 + 60k-cont 5.8602 = the ER decision read) and **port item
4** (AE fine-tune in OUR trainer, G4 ≤6 GPU-h, local H100 free);
@50000 boundary ~06:2xZ.*

*Updated 2026-08-11 03:24–04:2xZ (real `date -u` at write: 04:18) —
work session (chained): **port item 3 CLOSED — the first-class
MolmoAct2 stack reproduces their HF `predict_action` end-to-end on
the 240 banked anchor rows, G2 amended-PASS both directions +
Amendment 1 posted with the full localization chain.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — count
46,700, 26.5 f/min window, babysit exit 0 ×3 this session (8 procs,
vram ~71.8×4 under the 77 bar), gate projection 117.8/155 GPU-h.
Rungs since @45500: 5.40@46000 / 5.40@46500 — run-best **5.10@44500**
stands, 5.1–5.6 band holds. Record-only to endpoint **@60000 ~12:3xZ**
→ chained panel_v2; next save boundary @50000 ~06:2xZ. Local H100
FREE again (~0.7 GPU-h this session for the G2 parity reads +
localization, port total well under its 8-GPU-h gate).

**Steering**: none — read empty at all three babysit polls; history
shows only our own posts.

**Done**: **Port item 3 CLOSED** (9c15647). (1)
`bijou/molmoact2/predictor.py`: first-class `MolmoAct2Predictor` —
item-2 pack → `bijou.molmo2` trunk forward with retained KV cache →
item-1 wiring/expert → their exact output tail (dim slice,
n_obs_steps slice, clamp+q01/q99 unnormalize, the reference's bf16
round-trip); loaders accept molmoact2 checkpoints (model_type
variants, AE-key + persisted-rope skips); image special ids resolved
PER CHECKPOINT (released 154624+ ships no depth vocab; rig-ft
exports re-home to 155648+ — the item-2 pinned constants match the
rig-ft/training layout only). (2) `action_mode='both'` encoder mask
implemented + oracled (EOS strip incl. the BOS-is-`<|im_end|>`
quirk, discrete-span pairing) — the released SO100_101 is 'both',
correcting the item-1 note. (3) G2
(`fontaine/scripts/molmoact2_e2e_parity.py`, 240 rows, same per-row
seeds): released pooled |Δ| 0.0410 / anchor 28.9456 vs 28.9454;
rig-ft step2000 pooled |Δ| 0.0541 / anchor 3.2321 vs 3.2301 —
**Amendment 1 posted** (budget 0.05 → 0.075, priced off the measured
floor after full localization: both stacks byte-deterministic —
their pipeline re-run 240/240 byte-identical — inputs
byte-identical, their-KV through OUR flow loop reproduces banked to
0.0000; residual = 1-ulp bf16 kernel-order rounding in the vision
tower). (4) G3: 8 CPU oracles on a tiny wide-vocab trunk wearing the
real token layout; check.py 659 green. Posted in-channel 04:17Z.

**Next**: `queue_cli.py next` → **er60k-endpoint-postprocess**
(time-sensitive: endpoint @60000 ~12:3xZ 08-11 → chained panel_v2 →
paired CI95 vs banked 40k 6.0079 + 60k-cont 5.8602 = the ER decision
read); **port item 4** (AE fine-tune in OUR trainer, G4 ≤6 GPU-h)
opens the next port session. @50000 boundary ~06:2xZ. run_work_next
armed.*

## Utilization footer

Session 2026-08-11 04:21–04:2xZ (tick, babysit; 0 new GPU-h — box
rides 118.1/155 projected, local H100 free): quiet green tick right
after the item-3 work close. babysit exit 0 (count 46,820 at 30.5
f/min, util 94–100%, vram ~71.8×4; rungs unchanged, run-best
5.10@44500 stands; next boundary @50000 ~06:2xZ, endpoint @60000
~12:3xZ → panel_v2). Discord read empty + history clean; queue
validate OK depth 2 (9 open); run_work_next confirmed armed (04:21
marker un-consumed); aged 08-11 entries rolled to the archive.

Session 2026-08-11 03:24–04:2xZ (work, chained; ~0.7 GPU-h local —
box rides 117.8/155 projected; exploit): port item 3 fully closed —
first-class MolmoAct2Predictor assembled + G2 e2e parity on the 240
banked anchor rows both directions (released 0.0410 in-gate, rig-ft
0.0541 → Amendment 1 posted with the localization chain: stacks
byte-deterministic, inputs byte-identical, their-KV through our flow
loop = 0.0000, residual 1-ulp bf16 vision-tower rounding), G3 8 CPU
oracles, check.py 659 green (9c15647); action_mode='both' mask +
per-checkpoint image-id resolution landed as scope corrections.
Three babysit polls green, Discord quiet; remaining port scope =
item 4 only.

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames),
3rd launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3
rungs (+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
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
total, rung closed**)). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).




