# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 12:36–12:5xZ (real `date -u`) — tick (babysit →
conversational): **owner steering burst, three messages in 10 min —
attach_K KILLED on owner instruction (cost call, ~4× F per step), a
docs-modernization pass prioritized ahead of an owner main-rebase,
and a brand-new top-priority run spec: molmo2 from base 4B, 100k
steps, vision unfrozen from step 0, AdamC optimizer (implement
first, parameter sheet for approval before launch).***

**Status**: NO live runs — `fontaine_molmo2_flow_kijoint_10k_ddp4`
(attach_K) stopped 12:38Z at step ~4160/10k per owner instruction
(unit fontaine-attach-k; box GPUs verified 0 MiB ×4; checkpoints
through step_003750 retained on box, not uploaded — partial arm,
nothing consumes it; ~13.6 GPU-h spent 08:01–12:38Z). Probes were
healthy at kill (10.9664@4000 best, ~1.7 under the 5k bar) — this
was a COST kill (3.74 s/step vs F's 0.92), not a gate. Local GPU
free. Δ_seam matched read + read-4 AR-view drift are OFF (no K
endpoint); the attach screen closes on F evidence.

**Steering** (owner, 12:28:59Z / 12:31:43Z / 12:37:56Z + 👍 on our
12:37Z reply): (1) **docs pass prioritized** — update `docs/`
(architecture etc.) to reflect the current codebase/models in
standard ML language, no internal vocabulary (rungs/panels/idea
numbers), for an ML expert; README must state `fontaine/...` = the
research agent, rest = shared codebase (owner will rebase main on
fontaine and develop with local agents); tech-debt sweep at my
discretion. (2) **kill attach_K** — "way too slow per step";
executed 12:38Z. (3) **new molmo2 run from base 4B, TOP priority
("let's start with it")** — 100k steps, eff-batch 32 (8/rank),
vision encoder unfrozen from step 0 with `--{backbone,text}-
vision-lr 2e-5`, warmup 1000, **AdamC** per arxiv 2506.02285v1
(AdamW + time-varying per-group decay; implement efficiently,
mindful of tied/shared layers e.g. Gemma lm_head; read the owner's
shared conversation claude.ai/share/52f07abb… as part of
implementing); **in-depth description of ALL run parameters for
owner approval BEFORE launch**. All three acknowledged in-channel
(12:37Z + 12:40Z posts). ⚠ Process near-miss ×2: the 12:28/12:31Z
messages never surfaced via `read` (cursor already past them —
history check caught them), and the 12:37:56Z spec was consumed by
a `head -4`-truncated babysit read, recovered via the cursor
snowflake timestamp + history. New standing rule (memory): NEVER
pipe read/babysit output through head/tail; cross-check cursor
timestamp vs history each poll.

**Done**: kill executed + verified (procs gone, 4×0 MiB); babysit
attach_K entry pruned (kill note in babysit.toml); queue updated —
`idea4-attach-screen-execution` CLOSED (owner-kill note, F-side
complete), `owner-molmo2-adamc-run-prep-0809` added at HEAD,
`owner-docs-pass-0809` added second,
`molmo2-stage2-attachment-decision` re-scoped to F-only basis
(unblocked, after docs pass), f-then-joint draft re-anchored (must
argue against the measured 4× step cost); validate green depth 4
(9 open). Both owner replies posted (kill readout + AdamC plan:
paper + shared conversation first, thin AdamW variant with
per-group time-varying decay, tied-lm_head group-partition audit +
tests, then the full parameter sheet; no launch without sign-off).
`run_work_next` armed.

**Next**: chained work session (4-h budget) executes in owner order:
**AdamC implementation → parameter sheet posted for approval →
docs pass (a/b/c)**; launch of the 100k run ONLY after explicit
owner approval (box GPUs free and waiting). Then stage-2 memo
(F-only basis) + `lit-radar-hooks-0811a` in any gap.

*Updated 2026-08-09 12:16–12:4xZ (real `date -u`) — work session
(bounded, chained via `run_work_next`): **the radar backlog cleared
TWICE over — four papers deep-read, four pages landed same session
(QDepth-VLA, ForesightFlow, CLP fewer-layers, Qwen-VLA); two fresh
production frozen-first votes filed on #4's ledger hours before
tonight's Δ_seam read, and the selection cluster gets its first
direct evidence that selector shape beats selector size.***

**Status**: attach_K healthy at the 12:22Z poll — step 3940/10k,
loss 3.13, 3.776 s/step (endpoint ~18:3xZ holds), vram 59.07 ≤ 71,
liveness 7 procs / 4 GPUs. Probe **11.2033@3500** (best); first
kill-bar 12.6394 binds ≥5k (~13:3xZ) with ~1.4 margin. CE aux flat.
Local GPU free.

**Steering**: none — `read` clean at boot and at the 12:22Z babysit;
nothing from the owner after the answered 11:43:03Z loss_action
question, no new reactions.

**Done**: **three lit queue items executed same session they were
queued** (`lit-radar-hooks-0809b` → `-0810a` → `-0810b`, each
refill consumed in-window per the standing precedent), four papers
pages: (1) [QDepth-VLA](papers/qdepth-vla.md) 2510.14836 — third
aux-spatial recipe class (expert-generative VQ depth tokens,
monocular pseudo-labels, tokens RIDE the inference context unlike
VEGA/SF); ablation split carried loudly (−2.9 loss vs −8.5 expert:
the scaffold, not the geometry, carries most of the win) → #11/#17/
#5. (2) [ForesightFlow](papers/foresightflow-self-scored-bestofk.md)
2606.04968 — seventh selection flavor; the K-sweep is the evidence
anchor (separate 500M critic FLAT K=1→5, self-scored +5.0 = third
strike on post-hoc probe selectors); 1-NFE endpoint preview
instrument (τ 0.83, ~97% gain retained) → #19/#1/#12/#16.
(3) [CLP fewer-layers](papers/fewer-layers-clp.md) 2606.20246 —
33–50% of finetuned-VLA depth is CKA twins (8/16 DiT expert layers
free); throughput fourth lever class, CKA map banked as a
one-forward-pass diagnostic → #17. (4)
[Qwen-VLA](papers/qwen-vla-early-fusion.md) 2605.30280 —
early-fusion pole staked; Stage I trains the expert trunk-FROZEN =
**F-then-joint production vote #2 beside RDT2, filed pre-Δ_seam**;
τ=0.6 deploy sharpening = production cool-side dT sighting →
#17/#4/#19/#16. Two sweeps: no stage-2/actckpt re-ranker found; 2
new hooks banked (SEAM 2607.04609 boundary-jerk, Robot Critics
2606.21572). Papers-index integrity fix (2 stale "unread" rows →
page links); 2 future-dated queue stamps caught at write time and
corrected against `date -u` (the 78cace5 class — my pacing sense
runs fast; stamp at write, not at projected finish).

**Next**: 5k kill-bar binds ~13:3xZ (probe must be < 12.6394 —
currently 11.20; babysit before session end catches or brackets
the crossing); endpoint ~18:3xZ → chained panel_v2 + AR-view drift
panel → **Δ_seam frozen read (runbook staged, pre-audited)** →
stage-2 decision. `queue_cli.py next` → `lit-radar-hooks-0811a`
(any GPU-busy window).

*Updated 2026-08-09 12:12–12:2xZ (real `date -u`) — tick (babysit):
**attach_K healthy past the run's midpoint approach — probe margin
~1.4 held, all quiet; queue armed for the next lit slice.***

**Status**: attach_K healthy at the 12:13Z poll — step 3800/10k,
loss 3.10, 3.78 s/step (13.1 steps/min window; endpoint ~18:3xZ
holds), vram 59.07 ≤ 71, liveness 7 procs / 4 GPUs. Probe
**11.2033@3500** (best); first kill-bar 12.6394 binds ≥5k (~13:2xZ)
with ~1.4 margin. CE aux flat. Local GPU free.

**Steering**: none — `read` clean; `history` shows nothing from the
owner after the answered 11:43:03Z loss_action question and no new
reactions on our 11:48Z answer or the 12:12Z session post.

**Done**: babysit poll (exit 0, facts above — trajectories nominal,
no anomaly beyond the CLI facts: loss stepping down 3.21 → 3.10,
probe monotone-improving since 2500); queue validate green (depth 2,
8 open); `run_work_next` armed (chained work session takes
`lit-radar-hooks-0809b` — QDepth-VLA + fresh sweep; banked radar
backlog is empty).

**Next**: 5k kill-bar binds ~13:2xZ (probe must be < 12.6394 —
currently 11.20; next tick catches the crossing); endpoint ~18:3xZ
→ chained panel_v2 + AR-view drift panel → **Δ_seam frozen read
(runbook staged, pre-audited)** → stage-2 decision.

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
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box 60k continuation COMPLETE 08-08 ~23:4xZ
(~49 GPU-h ≤ 60 gate, chained evals incl.); local subgoal-swap arms
08-09 ~02:1x–03:42Z +~1.5 GPU-h ≤ 3 gate; box K-smoke ladder 08-09
04:02–04:39Z **+~0.5 GPU-h ≤ 6 gate (rung 1 GREEN first try)**; box
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + panel_v2
eval COMPLETE ~08:01Z (+~1.24 GPU-h); box attach_K 08:01–12:38Z
**KILLED by owner steering at step ~4160/10k (+~13.6 GPU-h, cost
call — no endpoint, no chained evals)**). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 12:12–12:2xZ (tick, babysit; 0 GPU-h): attach_K
step 3800/10k healthy (loss 3.10, 3.78 s/step, probe 11.2033@3500
best, vram 59.07 ≤ 71; 5k kill-bar margin ~1.4, binds ~13:2xZ,
endpoint ~18:3xZ). Discord clean — read empty, history nothing new
after our 12:12Z session post, no new reactions. Queue validate
green depth 2 (8 open); run_work_next armed (lit-radar-hooks-0809b
next: QDepth-VLA + fresh sweep). Stable stretch → exited rather
than held; next tick catches the 5k crossing.

Session 2026-08-09 12:36–12:5xZ (tick, babysit → conversational; 0
GPU-h new): OWNER STEERING BURST — attach_K killed 12:38Z on owner
instruction (step ~4160/10k, ~13.6 GPU-h spent, cost call: 3.74
s/step vs F's 0.92; box 0 MiB ×4, ckpts to 3750 retained; Δ_seam +
read-4 OFF, screen closes on F evidence); docs-modernization pass
prioritized (plain ML language, README fontaine-vs-shared split,
pre-rebase); NEW top-priority run spec: molmo2 base-4B 100k,
eff-batch 32, vision unfrozen from step 0 (lr 2e-5), warmup 1000,
AdamC (2506.02285) — implement first, parameter sheet for owner
approval before launch. Both replies posted; 👍 on the kill/docs
reply. Queue: attach item closed, adamc-prep + docs-pass items at
head, stage-2 re-scoped F-only; validate green depth 4.
Consume-once near-miss ×2 (cursor skip + head-truncated read) →
new standing rule banked in memory: never truncate read/babysit
output. run_work_next armed (AdamC first, docs pass second).
