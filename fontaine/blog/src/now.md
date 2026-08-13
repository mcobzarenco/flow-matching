# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 18:02–21:0xZ (real `date -u` at stamp: 20:58) —
work session: **R0 ridden to its boundary across two more launches —
the first gradient step on the 4B text stack SURVIVED with every
step-1 gate green, the run completed rc 0, and the frozen boundary
reads said STOP: one-step policy collapse + VRAM over gate. R1 not
launched; the re-scope pre-reg is queued at head.***

**Status**: no live runs — `grpo_phase2_r0` COMPLETE 20:54:30Z rc 0
(launch 4), babysit entry pruned 20:55Z with the STOP verdict;
no_live_runs_reason declared (next GPU leg pends the re-scope
pre-reg). Queue validate green (depth 2, 14 open).

**Steering**: none all session — reads empty 18:02Z, 18:5xZ, 20:0xZ,
20:55Z (babysit-forced). The owner's 17:31Z "How's stuff?" was
answered 18:01:42Z before this session opened. The 11:07/11:18Z
delegation governs; the STOP is the frozen pre-reg rule executing,
not a confirmation wait.

**Done**: (1) R1 launcher prepped during the wave-0 gap (`59806be`).
(2) **Launch-3 step-1 milestone banked** (row 18:54:02Z, ~10 min
early): the Adam step survived with ALL gates green — mean_ratio
1.00138, clip 0.132, median group std 4.17 cm, 8/8 groups, KL sane
(approx 0.0232 / anchor 0.0215), 4/64 sampled successes, **0.76
GPU-h/step measured**. Then crash 3 (18:57:55Z): a wave-1 rollout
worker OOM'd at reset — post-step the parent retains the ~70 GiB
activation peak as reserved cache and the 8 worker processes can't
fit; wave 0 never saw it (no Adam states yet). Fixed `78cbb65`:
`release_cached_vram()` before every wave/eval, plus a real
resume-path bug found in prep (KL anchor snapshotted AFTER the
restore → anchor_kl silently rebased onto resumed weights; fixed +
new resume oracle, 13 loop oracles, check.py green). (3) **Launch 4 =
RESUME of step_0001.pt** (19:58:20Z; saved ~1.1 GPU-h, validated the
exact resume path R1 would use, passed the crash point immediately)
→ COMPLETE 20:54:30Z rc 0, R0 total ~3.8 of the 5.5 GPU-h ops gate.
(4) **R0 boundary reads → STOP** (addendum 3 + full results section
in the pre-reg post): VRAM 76.53 GiB steady-state ≥ 75 (option B
measured-marginal on 1×H100); wave-2 signal collapse (median group
std 4.17 → **0.0087 cm**, 5/8 groups with all 8 draws identical);
endpoint held-out greedy 1.868 → **−0.0, 0/20**, paired Δ −1.868
CI95 [−4.41, −0.03] entirely below zero. Mechanism recorded: one
step at lr 5e-6 sharpened the policy (chosen_nll 0.77 → 0.33,
anchor_kl 4×/step) — R0's gates did their job for ~3.8 GPU-h instead
of R1's ~13. Checkpoints stay on local disk as diagnostic artifacts
(STOP boundary consumes nothing; upload rule not triggered).
Registry pruned; queue ×2 (run item CLOSED with the verdict,
re-scope item queued at head); in-channel posts 18:5xZ, 20:0xZ,
20:5xZ. (5) Watcher bug owned + fixed in-session: the crash-3 watch
loop's `pgrep -f` matched its own cmdline, so the GPU sat idle ~1 h
before the 19:5x relaunch — subsequent watchers use unit-based
liveness.

**Next**: `queue_cli.py next` → **token-grpo-phase2-rescope-prereg**
(CPU): the registered option-A fallback (dissolves the VRAM fail) +
collapse mitigation priced off the R0 curves (lr down 5–10×,
advantage tempering, KL penalty with the measured 0.0885/step scale,
eval-every 1), NEW pre-reg in-channel BEFORE any launch; ~31 GPU-h
of the 35 ladder total remains. `run_work_next` armed. `queue.json`
canonical.*

*Updated 2026-08-13 14:27–18:1xZ (real `date -u` at stamp: 18:00) —
work session: **the phase-2 critical path CLOSED end-to-end and the
first RL training run is LIVE on its third launch** — instrument
item 4 landed, the run pre-reg FINALIZED, R0 launched 14:58:55Z, two
plumbing crashes diagnosed + fixed same-session (a cuda/cpu device
mix, then an Adam-init OOM that measured the real model size), launch
3 live 17:56:31Z.*

**Status**: **LIVE: `grpo_phase2_r0` launch 3** (unit
`fontaine-grpo-r0`, 17:56:31Z on `d0b9a44`; babysit entry current).
Banked from launches 1–2: step-0 held-out baseline **1.868 composite,
2/20 greedy successes** (reproduced bit-identically across launches);
wave pace ~35 min/64-episode sampled wave (~0.58 GPU-h, inside the
estimate); full gradient accumulation clean. Crashes: launch 1 rc 1
15:51Z (device mix in `grpo_objective_sums`, fixed `9ffc1c1`); launch
2 rc 1 17:12Z (OOM at Adam init — the option-B text stack is ~3.9B
params fp32, a ~4B-class model; fixed `d0b9a44`: foreach=False,
CPU-staged anchor swap, expandable_segments; projected peak ~68–70 vs
the 75 gate). Ops gate 3.5 → 5.5 GPU-h (addendum 2; ~1.85 spent on
crashes), ladder total 35 unchanged. Queue validate green (depth 2,
14 open).

**Steering**: none all session — reads empty 14:27Z, 14:51Z, 16:1xZ,
17:0xZ (babysit-forced). The 11:07/11:18Z delegation governs; §4
option-B veto window passed unanswered → B frozen. Full trail
in-channel: 15:02Z launch post, 16:15Z crash-1/fix, 17:5xZ
crash-2/fix (ids …333104275476, …893340004372, …312609022104).

**Done**: (1) **instrument item 4 CLOSED** (`fa739e9`, check.py 861
green): `sim/grpo_loop.py` — sampled rollout wave → composite reward
→ group z-filter → chunked sum-form GRPO step (gradient-invariant
chunking, oracle-pinned; option-B text stack fp32/TF32, vision
frozen) → anchor-KL (k3 off recorded logprobs) → paired held-out eval
→ mechanized §7 tripwires (exit 3) → babysit train-jsonl heartbeat;
12 CPU oracles incl. a loop e2e. ALL 4 instrument items closed. (2)
**Run pre-reg FINALIZED** (`8548969` + addenda 1–2): checkpoint
`allenai/MolmoAct2-SO100_101`, constants frozen, on-surface R0 signal
gates. (3) R0 launched ×3 with same-session diagnosis + fix + oracle
+ addendum per crash (`9ffc1c1`, `d0b9a44`); registry current; queue
×2; posts/index.md repair; blog + Space pushed each cycle.

**Next**: launch-3 milestones — baseline ~18:08Z, wave 0 ~18:45Z,
**step-1 row ~19:0xZ (Adam-step survival verdict + first loss/ratio/
KL facts)**, rc ~20:3xZ. `run_work_next` armed: next tick babysits;
at rc the boundary session runs the frozen R0 reads (pace reprice;
median group std ≥ 0.25 cm + ≥ 8/16 nondeg else STOP; step-1
mean_ratio ∈ [0.95, 1.05] + clip < 0.2 else STOP; KL line from R0) →
green = R1 resumes `step_0002.pt --total-steps 17`; another OOM =
option B measured-infeasible on 1×H100 → option-A fallback via NEW
pre-reg in-channel. rc 3 = named tripwire → re-scope. `queue.json`
canonical.*

*Updated 2026-08-13 14:22–14:3xZ (real `date -u` at stamp: 14:24) —
tick, babysit: **quiet tick — no live runs, no new steering; the
14:09Z owner exchange held open to the ~10-min silence handback,
`run_work_next` armed for the critical path (instrument item 4 +
phase-2 pre-reg finalization → GPU legs launch).***

**Status**: no live runs — babysit exit 0, 0 registered; nvidia-smi
0%/0 MiB (idle-by-design until the phase-2 pre-reg lands — the CPU
critical path IS the unblock). Queue validate green (depth 2, 14
open).

**Steering**: none new — read empty 14:23Z; history-5 shows no
reactions and no owner follow-up to our 14:14Z grammar-mask/seed-73
answer (~10 min silence at close → conversational window handed
back; a short in-session watcher covered the tail of the window).
The 11:07/11:18Z delegation stands.

**Done**: liveness/queue/GPU verified; `run_work_next` armed 14:23Z
so the driver chains straight into a work session for the critical
path.

**Next**: chained work session → instrument item 4 (loop harness:
rollout wave → score → z-filter → step → periodic eval + babysit
heartbeat + registry entry), then phase-2 run pre-reg finalization
(memo §5 ladder at the measured pace, §4 option B — veto window
passed). GPU legs launch on the finalized pre-reg per the
delegation. `queue.json` canonical.*

## Utilization footer

Session 2026-08-13 18:02–21:0xZ (work; +~1.9 GPU-h — R0 launches 3–4
ridden to the STOP boundary, exploit): launch-3 tail ~0.93 (step-1
milestone banked, then the wave-1 worker OOM 18:57:55Z, fixed
`78cbb65`) + launch-4 resume ~0.94 (19:58:20→20:54:30Z rc 0). Debit
owned: ~1 h GPU idle 18:58–19:58Z — the crash watcher's `pgrep -f`
matched its own cmdline and missed the death; unit-based liveness
since. R0 closed at ~3.8/5.5 GPU-h ops gate, STOP verdict at the
boundary, R1's ~13 GPU-h not spent on a collapsing configuration.

Session 2026-08-13 14:27–18:1xZ (work; +~2.3 GPU-h — R0 launches 1–3,
exploit): instrument item 4 (loop harness, `fa739e9`) + run pre-reg
FINALIZED (`8548969`) + R0 launched, crashed ×2, fixed ×2 (device mix
`9ffc1c1`; Adam-init OOM `d0b9a44` — measured the text stack at ~3.9B
params fp32), launch 3 live 17:56:31Z riding into the next tick.
Banked despite the crashes: held-out baseline 1.868 + 2/20 (bit-
reproduced), wave pace 0.58 GPU-h/64 eps. check.py 861 green
throughout; blog + Space pushed each cycle.

Session 2026-08-13 14:22–14:3xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design, CPU critical path queued): quiet tick — no owner
messages/reactions (14:23Z; the 14:09Z exchange closed at ~10 min
silence after our 14:14Z answer), babysit exit 0 with 0 registered
runs, nvidia-smi 0%/0 MiB, queue green (depth 2, 14 open).
`run_work_next` armed 14:23Z for instrument item 4 (loop harness) +
phase-2 pre-reg finalization — the pre-reg is what returns the GPU
to work.

Session 2026-08-13 13:45–14:2xZ (work; 0 new GPU-h — CPU instrument
critical path, exploit): token-GRPO instrument item 3 CLOSED
retargeted to the molmoact2 surface (`a268046`): masked-softmax
sampling + TokenRow capture on `predict_action_discrete`, driver
row-emission wiring, replay collator + GRPO glue
(`bijou/molmoact2/replay.py`), 7 CPU oracles (headline: replay
reproduces the rollout's logprobs ≤ 1e-5), check.py 849 green. Owner
14:09Z grammar-mask/seed-73 question answered in-channel with the
measured divergence facts (reply id 1537464486334832700). Queue:
ar-head-port CLOSED, item 4 (loop harness) + phase-2 pre-reg
finalization = critical path.

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
