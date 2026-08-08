# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-08 15:28–16:0xZ (real `date -u`) — work session
(bounded): the meta-report's frame-mining stage EXECUTED end-to-end in
the GPU-quiet window — the owner's "ambiguous frames" found
automatically, and the report's central question answered early: the
subgoal gain does NOT concentrate on them. Standing lit slice landed
the null's interpretive frame same-session.*

**Status** (babysits 15:29/15:5x/16:0xZ, exit 0): box **molmo2_ar60k LIVE
+ healthy**: step ~47,900/60,000, probe 6.58@47,500 flat in the
6.40–6.87 band (1.63 under the 8.21 bar, ×3 never armed), loss 2.77
falling, 2.19 s/step, vram 73.84 no new peak; **50,000 save boundary
~16:5xZ**, ~7.3 h to the 60k close (~23Z). Local GPU: 12-min embed
unit (fontaine-framemining-embed) ran and exited clean; idle again
for the post-23Z perf ladder.

**Steering**: none new (poll clear at both babysits; owner 👍-acked
both 15:25Z answers). P1 relative-bound adjudication still pending.

**Done** (this session,
[post](posts/2026-08-08-framemining-aliased-frames.md)): the
`fieldcond-subgoal-meta-report` frame-mining stage, instrument to
verdict same-session: (1) `frame_mining.py` landed (embed / mine /
sheet; check.py 500 green) — 17,204 core panel frames embedded with
the **frozen Gemma-4 E2B tower = AR-100k's own frozen eye**
(alignment oracle vs the banked npz every row, actions included);
(2) within-dataset NN mining banked
(`analysis__framemining_ar100k_k4l2.json` + flagged npz + a 12-pair
contact sheet that IS the owner's ask — cylinder mid-place vs
placed, mug pre/post-grasp, chess boards); (3) **concentration read
(pinned pre-execution): clean NULL** — flagged−rest Δ_oracle −0.003
[CI −0.205, +0.176], ρ −0.01 on 14,064 frames; gain flat across
aliasing except ~zero on the least-aliased decile. Story for the
report: the subgoal slot is a **uniform prior, not a disambiguator**;
the +29% aliased-frame error floor (miner validated, ρ 0.41 vs
baseline MAE) is the #11 history-arm prize. Ideas #6/#11 hooks +
queue amendment landed. Then the **standing lit slice** (papers page
same-session per the permanent rule:
[conditioning-shortcuts](papers/conditioning-shortcuts.md),
2602.24143 + 2605.20856): the flat gain has a published family —
"robust skills, brittle grounding" (conditioning consumed as a coarse
prior; compositional holdout 44%→0%; 10k→100k demos buys ~nothing)
and DISC's task-state entanglement mechanism + structural-decoupling
fix. Missing cell for our slot named: a subgoal-swap sensitivity read
(presence −0.29 / channel +0.043 / CONTENT = the open triangle) —
meta-report open-questions candidate. #6/#17 hooks landed.

**Next**: `queue_cli.py next` boundaries: **50,000 save ~16:5xZ**
(routine), **60k close ~23Z** → chained eval → fields panel → perf
box ladder + noise-ladder stage 2 in the post-close window; the
meta-report composes the banked mining artifacts with the fields
numbers after that. Chained work armed (`run_work_next`).

*Updated 2026-08-08 15:23–15:4xZ (real `date -u`) — tick (babysit):
run healthy; and a SECOND missed-steering catch this day — **two
owner questions (14:40Z + 14:49Z) had scrolled past the cursor
unanswered** during the perf-exec window; found via `history`,
both answered from code this tick (~45 min latency).*

**Status** (15:2xZ babysit exit 0): box **molmo2_ar60k LIVE +
healthy**: step 47,540/60,000, probe 6.58@47,500 flat in the
6.40–6.87 band (1.63 under the 8.21 bar, ×3 never armed), loss
2.786 falling, 2.21 s/step, vram 73.84 no new peak; ~7.5 h to the
60k close (~23Z). Local GPU idle-by-design (perf ladder waits for
the box's post-23Z window). Queue validate green depth 5 (15 open).

**Steering** (two owner questions, both answered in-channel):
(1) 14:40Z *"is the SigLIP2→LLM connector frozen? vision-lr or
text-lr?"* — answered: connector (2×2 attn-pool + gated
`image_projector`, `bijou/molmo2/vision.py`) is inside
`backbone.vision` → **`--backbone-vision-lr`'s group**
(`encoders/molmo2.py:447`); the 60k run passes no vision-lr, so
tower AND connector are **frozen** (text trunk 2e-5 + head 1e-4
train). (2) 14:49Z *"40k report: headline chunk_mae 6.008 vs Q2
true-outcome MAE 5.877 — what's the first conditioned on?"* —
answered: **same single TRUE-label-conditioned pass; Q2 is a
bucketing of the same scores, not a counterfactual**
(`eval/cli.py:1404`; unlabeled frames render no outcome bracket =
unconditioned marginal, `interface.py:456`). 6.008 = frame-weighted
pool over all 17,204 frames {success 5.877, partial 6.315, failure
6.894, unlabeled 6.290}; the gap is bucket composition, not a
conditioning delta (the forced-success counterfactual is Q3).
P1 relative-bound adjudication still pending with the owner.

**Done**: babysit + 2 code-grounded answers posted; conversational
window held with a monitor (no further owner replies by close).
Process note: this is the day's second cursor-slip — the read-cursor
moves on any session's poll, but a heads-down session can read
without handling. `history` at every tick is the safety net; a
harness-level unacked-owner-message guard is worth an idea entry.

**Next**: **50,000 save ~16:5xZ** (routine), **60k close ~23Z** →
chained eval → fields panel → perf box ladder (P1 in/out per owner
adjudication) + noise-ladder stage 2 in the post-close window.
Chained work session armed (`run_work_next`) — queue has CPU-side
items and the box is busy.

*Updated 2026-08-08 14:0x–15:3xZ (real `date -u`) — work session
EXTENDED by owner steering (14:04Z + 14:10Z mid-close): the perf
pass-1 **execution** ran same-session at owner prio. Net: branch
built + bitwise-oracled green, one-step parity executed (one honest
gate FAIL, owned), a **real `--activation-checkpointing` CUDA bug
found** before it could crash a box launch, and the bench ladder
relocated to the box's true recipe after the local single-GPU form
proved structurally OOM.*

**Status** (15:2xZ): box **molmo2_ar60k LIVE + healthy**: step
~47,540/60,000, **47,500 boundary judged routine PASS** (probe
6.58@47,500, flat in the 6.40–6.87 band, 1.63 under the 8.2075 bar,
×3 never armed), loss 2.786 falling, 2.21 s/step, vram 73.84 no new
peak; ~7.6 h to the 60k close (~23Z). Local GPU free again
(parity-only unit exited 15:04Z).

**Steering** (mid-session, all handled): 14:04Z "why training only?"
— answered (decode byte-anchors + the win is training-side); 14:10Z
"scope good; **prio training speed with clear benchmarks**; what's on
the local GPU?" — answered (idle) and executed: `molmo2-perf-pass1-exec`
ran immediately. 15:0xZ **P1 adjudication pending with the owner**:
the one-step loss bound failed as banked (8.70e-3 abs vs frozen
1e-3; = 5.1e-4 *relative* at init-scale loss 16.9 — my calibration
flaw, owned in-channel). Default per frozen rules: P1 dropped;
owner may approve a relative-bound amendment before the box ladder.

**Done** (this block, commits `410e1aa` + `553aae1`): branch
`perf-pass1` (P1-only `00cdafe`, full `22e8148`, check.py 500 green
at both, pushed); **bitwise oracle GREEN 118/118 hashes**
(`perf_pass1_bitwise_oracle.py`, HEAD vs branch: logits, loss, wte
both regimes, every param grad — P3a/P3b/P4 value-identity proved);
one-step parity executed locally (grad-norm PASS rel 8.1e-3, cuDNN
fwd+bwd no crash — expectation 4 holds at one step; loss bound FAIL
as above); **single-GPU full-recipe bench proven structurally OOM**
(unsharded AdamW states → 78.2/79.18 GiB by step 2 at ANY batch —
chunked backward makes activations batch-invariant, receipts in 5
launch-round logs); **act-ckpt CUDA bug found + filed on idea #20**
(checkpoint recompute escapes the `sdpa_kernel` pin → backend
mismatch abort; the review's lineage-flip rec had a latent box
crash; prerequisite fix named); box-recipe ladder launcher landed
(`box/perf_pass1_bench_box_ddp4.sh`, supersedes the transfer
smoke). Babysits green throughout; 47,500 boundary PASS posted.

**Next**: `queue_cli.py next` boundaries: **50,000 save ~16:5xZ**
(routine), **60k close ~23Z** → chained eval → fields panel → then
the post-close GPU window runs the **perf box ladder** (needs the
`perf-pass1` worktree on the box — prereq in the launcher header;
P1 in or out per the owner's adjudication) and **noise-ladder
stage 2**. Every GPU launch via `run_detached.sh`.

## Utilization footer

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
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box + local both
idle from ~08:15Z pending the next pre-registered launches). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-08 15:28–16:0xZ (work, bounded; exploit+explore,
~0.2 GPU-h local): meta-report **frame-mining stage EXECUTED**
(`29813f0`): frame_mining.py landed, 17,204 panel frames embedded
with the frozen Gemma-4 E2B tower (12-min detached unit, alignment
oracle every row), NN mining + pinned concentration read banked —
**clean NULL** (flagged−rest Δ_oracle −0.003, ρ −0.01; subgoal slot
= uniform prior, not disambiguator; +29% aliased error floor = #11
prize), post + 2 charts + 12-pair contact sheet live, Discord
posted. Standing lit slice: conditioning-shortcuts papers page
(2602.24143 + 2605.20856) — the null's interpretive frame + the
subgoal-swap missing cell; #6/#11/#17 hooks. Babysits 15:29/15:5x/16:0x
green; queue green depth 5.

Session 2026-08-08 15:23–15:4xZ (tick): babysit exit 0 (47,540,
probe 6.58@47,500 in-band, vram flat), 0 GPU-h new; **second
missed-steering catch of the day via `history`**: owner questions
14:40Z (connector frozen? → yes, vision group, no vision-lr passed)
+ 14:49Z (chunk_mae 6.008 vs Q2 5.877 → same true-label pass, Q2 is
a bucketing; 6.008 pools all buckets) both answered from code
in-channel (~45 min latency); conversational window held via
monitor; queue green depth 5; `run_work_next` re-armed. Archive
roll (head entry + 3 oldest footer notes).

Session 2026-08-08 14:0x–15:3xZ (work EXTENSION, owner-steered;
exploit, ~0.4 GPU-h local): perf pass-1 EXECUTED at owner prio
(`410e1aa`+`553aae1`): branch built (P1 `00cdafe` / full `22e8148`),
bitwise oracle 118/118 GREEN, one-step parity run (grad-norm PASS,
loss bound FAIL as banked — 5.1e-4 relative at init scale, owned;
P1 adjudication with owner), single-GPU bench proven structurally
OOM -> ladder moved to box true recipe (launcher landed),
**act-ckpt CUDA bug found** (recompute escapes sdpa_kernel pin;
idea #20, prerequisite fix named). 47,500 boundary routine PASS
(probe 6.58 in-band). Babysits green; Discord ×5; queue green.
