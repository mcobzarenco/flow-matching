# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 13:42–14:1xZ (real `date -u`) — work session
(bounded, one item): **the #4 stage-2 attachment decision is CLOSED —
frozen default stands, memo posted from banked artifacts — and
adamc_100k survived its microbatch-1 first backward and is running
healthy at full utilization.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE (launch 3, 13:40Z,
chunks 8 / microbatch 1) — step ~480/100k at the 14:06Z babysit, GPUs
97–100% ×4 (first-poll starvation check clean), **2.62–2.75 s/step**,
vram alloc peak 70.4 vs the 77 bar, loss 16.33@20 → 7.59@160 falling
smoothly through warmup, CE-aux 1.17, grad-norm 283→31 (record-only
AdamC watch). Banners verified: AdamC λ=1e-5 partition
4074.7M/2.6M/0.6M, E1 dataset gate exact. **Projection ~75 h wall ≈
300 GPU-h → endpoint ~08-12 ~17:00Z**; babysit gate raised 260→310
(declared in-channel — the OOM-forced microbatch-1 restart is the
whole gap vs the 1.7–2.1 estimate; stop+act-ckpt alternative offered
to the owner, default let-it-run). First async-save line owed at step
5000 (~17:2xZ).

**Steering**: none new — read clean at boot, 14:06Z and 14:0xZ polls;
the 13:48Z first-poll/gate post and the 14:05Z memo post are
unanswered so far (tight-poll rule armed for the gate question).

**Done** (`e4b0ba5`): **stage-2 attachment decision memo posted**
([post](posts/2026-08-09-molmo2-stage2-attachment-decision.html)) —
frozen default ADOPTED for the Molmo2 trunk class; KI-joint
closed-unmeasured (honesty flag up front: no Δ_seam CI exists, K was
owner-killed at ~4160). Basis: F panel 9.4157 vs state-copy 11.7639
(2× the decisive bar), 8 matched probe evals K−F mean +0.208 (K
ahead 2/8, CE branch healthy throughout — trunk fine, not paying),
measured 4.11× step cost, RDT2/Qwen-VLA frozen-first votes; Wall-OSS
reading recorded. Probe-curve chart landed
(`attach_screen_probe_chart.py`, eval-report dark theme). Priced
residuals: Δ_seam@3750 rescue read ~2.5 GPU-h (own pre-reg);
f-then-joint draft UNBLOCKED (must argue vs 4×); depth-of-reads open.
Idea #4 ledger → `decided`; queue item DONE; blog built + Space
pushed, memo page curl-verified 200.

**Next**: `queue_cli.py next` → `idea4-f-then-joint-prereg-draft`
(CPU, in the run's shadow; natural target = the adamc_100k endpoint)
or `lit-radar-hooks-0811a`/`docs-pass-followups-0809` in any gap.
adamc_100k boundaries: first save + async-save line ~17:2xZ; first
kill-bar comparison binds at eval@2500 vs @10k (~08-10); endpoint
~08-12 ~17:00Z → chained k4l2 panel (--report) → leaderboard row +
grad-norm chart.

*Updated 2026-08-09 12:47–13:5xZ (real `date -u`) — work session
(4-h budget): **both owner top-priority items closed — AdamC
implemented, oracle-tested and LAUNCHED as the new 100k run from
base Molmo2-4B (after a three-message approval exchange, including a
λ override caught before step 1), and the docs modernization pass
landed for the owner's main-rebase.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE on the box (unit
`fontaine-adamc-100k`, relaunched 13:30Z after the λ override) —
base Molmo2-4B, 100k steps, eff-batch 32 (8/rank ×4, microbatch 2),
vision tower unfrozen from step 0 (banner: 439.1M vision params @
2e-5), text 2e-5, decoder 1e-4, warmup 1000, **AdamC λ=1e-5**, seed
1, save 5000, ZeRO-1 + chunked backward + async saves. Banners
verified: E1 dataset gate exact (878/38,571/18,636,749), AdamC
partition 4074.7M corrected / 2.6M head / 0.6M 1-D. In dataloader
spin-up at write time — first log window's measured s/step + vram
peak owed to the channel (babysit `adamc_100k` entry live: kill bars
NaN/inf, @10k<@2500, >25×3 after 5k, 77 GiB near-OOM watch, 260
GPU-h gate; grad-norm = record-only AdamC watch).

**Steering** (13:19:10Z + 13:24:10Z, both actioned same session):
(1) approvals on the parameter sheet — text+vision 2e-5 confirmed,
seed 1, save-every 5000, **no smoke, launch the real run** (OOM ⇒
restart at microbatch 1); λ pushed back ("0.1 high — what's
standard?") → grounded answer posted (openpi ≈0, OpenVLA finetunes
0.01), launched at 0.01. (2) **λ override 13:24Z: use the 40k/60k
lineage value 1e-5** — caught before the first optimizer step
(run was in model-load), stopped, relaunched clean at 1e-5
(amendment 2 on the sheet). ⚠ Process: the 13:19Z reply sat unseen
~35 min while I was heads-down in the docs pass — new memory rule:
after asking the owner anything, poll every ~3–5 min until answered.

**Done**: (1) **AdamC** (`401d6f7`): `--optimizer adamc` = stock
fused AdamW with per-group time-varying decay λ̂=λ·γt/γmax; partition
corrected/head/no-decay with tied-lm_head care (Gemma AR decoder's
tied embed-head routed as one param, one group; unaudited decoders
refuse; BOTH optimizer modes now hard-assert disjoint exact cover of
the trainable set); 10 new oracles incl. bitwise AdamW equivalence
at peak lr + the ZeRO-1 wrapper→local sync contract; check.py 584
green. (2) **Parameter sheet + 2 amendments**
([post](posts/2026-08-09-prereg-molmo2-adamc-100k.html)) posted
before launch; launcher `launch_box_fontaine_molmo2_adamc_100k_ddp4.sh`
(63b977c + λ fix); box synced via the git side-branch route (GitHub
key absent on box). (3) **Docs pass** (`e7144c3`, owner 12:28Z
request): README two-trunk + fontaine-vs-shared split; architecture
.md modernized end-to-end (Molmo2 in intro/§1/§2, curated-plan
ledger in §7, shipped-flag demotions in §8, CLI-default corrections,
residual/seam/snapflow documented, §5 gains AdamC + memory machinery
+ async saves); 4 historical docs got archive headers; subagent
staleness audit against HEAD drove the pass; deferred tail queued as
`docs-pass-followups-0809`.

**Next**: `queue_cli.py next` → `molmo2-stage2-attachment-decision`
memo (F-only basis, CPU) in the run's shadow;
`docs-pass-followups-0809` + `lit-radar-hooks-0811a` in any gap.
adamc_100k boundaries: first kill-bar reads bind at the eval@2500 →
@10k comparison (~08-10); endpoint ~08-11/12 → chained k4l2 panel
(--report) → leaderboard row + grad-norm chart.


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

Session 2026-08-09 13:42–14:1xZ (work session, bounded; exploit; 0
new GPU-h launched — adamc_100k rides from last session, ~1.7 GPU-h
accrued to the 14:06Z poll vs 310 gate): stage-2 attachment decision
CLOSED (memo posted, frozen default stands, KI-joint
closed-unmeasured; f-then-joint draft unblocked; queue validate green
depth 3, 8 open). adamc_100k survived the microbatch-1 first backward
(launch 3): 2.62–2.75 s/step, vram 70.4/77, util 97–100%×4, banners
verified; projection ~300 GPU-h → gate 260→310 declared in-channel
with the act-ckpt alternative offered. Discord: first-poll facts +
memo posted; no owner traffic. run_work_next armed.
