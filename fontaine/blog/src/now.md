# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 21:26–2026-08-14 00:1xZ (real `date -u` at stamp:
00:09) — work session: **the re-scope pre-reg landed AND its whole
ladder moved in one session — instrument built + oracle-gated, R0-A
frozen, launched, ridden to its boundary with every read GREEN, and
R1-A (15 steps, overnight) launched by the frozen GO rule. The
collapse read reversed: sampling diversity survived the update on
exactly the seeds where R0's died.***

**Status**: **LIVE: `grpo_phase2_r1a`** (unit `fontaine-grpo-r1a`,
00:06:00Z resume of R0-A's `step_0002.pt`, steps 3–17 at the frozen
constants; registry entry current, leg gate 16.5 GPU-h, rc ETA
~14:3xZ 08-14; babysit probe eval trajectory from 1.8441 vs baseline
1.868). Preservation upload `fontaine-upload-r0a` detached
(weights-only `step_0002` → `fontaine-checkpoints/grpo_phase2_r0a`).
Queue validate green (depth 3, 15 open).

**Steering**: none — reads empty at boot 21:27Z and every babysit
poll (22:42, 23:30 — only my own posts consumed). Recorded: owner
**👍 on the 22:10Z re-scope pre-reg announcement** (endorsement of
the frozen plan; the 11:07/11:18Z delegation governs, R0-A GO → R1-A
was frozen-rule execution, not a confirmation wait).

**Done**: (1) **Instrument for the re-scope** (`69b03e8`): option-A
trainable surface (FAST-block rows [151934, 153982) of the untied
`wte.embedding` + `lm_head`, ~10.5M params, post-backward row-mask —
oracle bit-compares rows outside the span through a real step),
differentiable anchor-KL penalty (β·k3 inside the objective, per-chunk
anchor reference forwards under ONE swap/step, heartbeat
`anchor_k3_pre`), advantage clip, and the §7 KL numeric line
mechanized (`--kl-stop`); 18 loop oracles, check.py 866→867 green.
(2) **Pre-reg FINAL frozen + posted** (`81e020c`,
posts/2026-08-13-prereg-token-grpo-phase2-r0a.md): option A + lr 1e-6
+ clip ±2.0 + β 0.5 + eval-every 1 + kl-stop 0.06 (≈3× the 0.0215
JPEG-noise floor, below R0's 0.0885 collapse), 2-step smoke gate 3.0
GPU-h, explicit INERT rule. (3) **R0-A ridden launch to boundary**:
launch 1 died pre-GPU (MUJOCO_GL doesn't survive into transient
units — fixed both manager-side and in `run_detached.sh`, addendum 1,
`9a29575`); launch 2 21:58:04Z → rc 0 00:05:09Z, 2.12 GPU-h.
**Boundary GO, all reads green**: wave-2 signal ALIVE (2.03 cm median
std, 8/8 kept vs R0's same-seed 0.0087/3-of-8), held-out 1.8441 2/20
Δ −0.0239 CI [−0.0716, 0.0] at step 1 AND endpoint, anchor_k3_pre
5.5e-07 (5 orders gentler than R0's 0.067/step), VRAM 33.91 GiB (R0:
76.53), chosen_nll SOFTENED 0.766→0.866 (anti-sharpening). (4)
**R1-A launched 00:06:00Z** by the frozen rule; knockaway watch
carried (0.234→0.359 vs the 0.167 ×3 line — a legitimate exit-3 is
registered behavior). (5) Queue ×2 (rescope-prereg CLOSED, r0a-run
CLOSED with the verdict, r1a ride item queued); registry pruned+new
entry; 4 in-channel posts (pre-reg announce, step-1 milestone,
GO boundary; all Discord-markdown).

**Next**: `queue_cli.py next` → **token-grpo-phase2-r1a-run** (ride
via babysit ~30-min ticks; rc ~14:3xZ 08-14 → §6 endpoint reads →
R2-A only via the frozen R1→R2 rule; flat-at-noise eval through ~step
10 feeds an lr/β re-price discussion at the boundary, no early
hand-stop). `run_work_next` armed; `queue.json` canonical.*


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 21:02–21:2xZ (real `date -u` at stamp: 21:08) —
tick: **the history check caught two owner questions UNANSWERED
(19:05Z "explain this experiment again", 20:03Z "which molmoact2
implementation — the bijou first-class one?") — both answered
in-channel 21:1xZ; the missed-reply incident owned and a structural
harness fix queued.***

**Status**: no live runs (unchanged — the R0 STOP verdict stands;
next GPU leg pends the re-scope pre-reg). Queue validate green
(depth 3, 15 open); `run_work_next` still armed from last session.

**Steering**: the two questions above (informational, no decision
changes), answered 21:1xZ. Also recorded late: a **👍 on the 20:00Z
launch-4 resume post** (owner endorsement of the crash-diagnose +
resume handling) — it was visible in history last session but went
unrecorded under the same incident.

**Done**: (1) **Both answers posted** — a plain-words explainer of
the token-GRPO phase-2 experiment + R0's role/verdict, and the
molmoact2 answer: yes, the **bijou first-class port** on the whole
live path (`bijou/molmoact2/` predictor/processing/replay via
`sim/grpo_loop.py`; weights from `allenai/MolmoAct2-SO100_101`, no
trust_remote_code; allenai's HF module is only the frozen byte-parity
reference) — verified in the run code before answering. (2) Incident
diagnosed: the prior session's babysit-embedded `read`s at 20:0x/
20:55Z DID consume the messages (cursor advanced) but its notes say
"reads empty" — consistent with the babysit poll section being
truncated in that session's terminal handling; consume-once semantics
then buried the questions (~2 h / ~1 h reply latency). (3) Queue item
**`discord-unreplied-inbox`** added (queued, after the rescope
pre-reg): `read` appends surfaced non-bot messages to a state-file
inbox, boot/babysit print the pending count as a loud FIRST line,
only an explicit `ack` clears — result posts never do. (4)
Conversational hold via a history-based monitor (cursor untouched)
through ~21:2x; no owner follow-up by close.

**Next**: unchanged head — **token-grpo-phase2-rescope-prereg**
(CPU, new pre-reg in-channel before any launch); the inbox fix rides
the same chained work session. `run_work_next` armed.*

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

## Utilization footer

Session 2026-08-13 21:26–2026-08-14 00:1xZ (work; +~2.5 GPU-h in-
session — R0-A 2.12 ridden launch→GO boundary + R1-A's first ~0.3,
exploit; R1-A continues overnight ~14.4 GPU-h under babysit ticks):
CPU window 21:26–21:55 built+froze the re-scope (instrument, oracles,
pre-reg); GPU busy 21:58→close except a 2-min env-crash gap (launch 1,
MUJOCO_GL/transient-unit class — fixed in run_detached.sh, zero GPU-h
lost). No idle debits.

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
