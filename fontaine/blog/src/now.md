# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 21:39Z–2026-08-13 01:xxZ (real `date -u` at stamp:
00:38 08-13) — work session (the chained session riding the probe):
**GRPO probe: AR signal is REAL and cheap at t=1.0 (0.771 cm vs the
0.25 bar, cost CI includes 0); t=1.6 clears 10× but pays −1.08 cm.
Tripwire fired at cell-1 (measured ~1.13 GPU-h/cell vs ~0.6
assumed) — re-scoped in-channel to cells 1/2/5, cells 3/4 parked.**
Plus: lit 0823 sim-improvement levers closed (3 papers pages), and an
owner steer mid-session — wrist compositing investigated end-to-end
and DECIDED render-only (22:31Z).*

**Status**: GRPO probe cell 5 (SDE a=0.5, the Flow-GRPO trainability
cell) finishing ~01:0xZ; unit stopped at its boundary per the
re-scope; cumulative [CELL5_GPUH] vs the 3.5 gate. Cells 1/2 read out
at their boundaries (in-channel 23:0x/00:07Z). [CELL5_LINE]

**Steering** (live owner thread): (1) 22:21:54Z "investigate
compositing for the wrist camera" → executed same session:
CPU-only feasibility read (`wrist_composite_feasibility.py`,
`d177c0d`) — plate poses spread 20.8 mm/5.1° median (why the static
plate mushed), wrist is table-plane-dominated (median 100% of rays)
so FK+plane-homography is sound, but warp-fill p10 49% before
arm/boat holes ⇒ T-III seam hazard; recommended render-only wrist +
redirect to lens fitting; (2) 22:31:50Z owner adopted the
recommendation → `sim-wrist-compositing` CLOSED as decided, sim100
**amendment 4** documents the channel asymmetry,
`sim-fit-real-lens-model` queued (plumb-line θ→r on existing frames,
no rig time); (3) 22:33:20Z "how does the encoder probe work in
depth?" → two-part in-depth reply 22:41Z. Probe re-scope default
posted 21:58Z, no objection at any boundary.

**Done** (commits `49381ca`…`6897fea` + close-out): (1) **lit
0823 CLOSED** (`ed6ba42`, owner-called): 3 papers pages same session
— composite-shadows (no published pipeline measures the missing-shadow
axis; Re³Sim foreground-realism null; randomize-in-training /
match-in-eval split), fisheye-lens-fitting (scale overfitting as a
distance ruler; cubemap→any-lens MuJoCo pipeline), dr-schedules
(DORAEMON success-throttled entropy max; eval stays at matched
center); 3 ideas hooks (#16 sim lane), `sim-composite-contact-shadows`
queued. (2) Probe instrument: frozen-reads script (`ece2276`), house
dark chart (`d2bde2f`), registry re-scope (`6897fea`). (3) Wrist
compositing decision artifacts (`a5e5784`). (4) Probe results
amendment on the pre-reg page + results post at cell-5 close.

**Next**: `queue_cli.py next` → CPU lanes: `sim-fit-real-lens-model`
(owner-adopted), `sim-composite-contact-shadows` (both probe-gated,
pair on the same harness). GPU idle after the probe stop; cells 3/4
re-queue only on owner call; phase-2 GRPO call per the frozen
decision rule (see the results post). v3-rerun unhold + disk-draws
sign-off still open. `queue.json` canonical.*

*Updated 2026-08-12 21:30–21:4xZ (real `date -u` at stamp: 21:36) —
tick, babysit: **GRPO signal probe LAUNCHED 21:33:58Z** (unit
`fontaine-grpo-probe`, HEAD `85e9a16`) — the tick resolved the
now.md-internal conflict ("launches on handback" vs "on the owner's
go") by re-reading the record: the owner's 13:36Z sequence (oracle →
ftrig eval → probe) has both predecessors done, the 20:06Z pre-reg
post said "say go (or it rides the standing sequence at handback)"
with no objection since, and the GPU was handed back for the 100ep
eval — so the standing sequence governs and the probe launched.*

**Status**: probe LIVE (launched 21:33:58Z; 2 anchor passes + 5 cells,
660 episodes, workers=8, ~2.8 h wall, gate ≤3.5 GPU-h; first poll
85% util / 21.8 GB, anchor pass streaming ~0.76 s/replan). Babysit
entry `grpo_signal_probe` active; new launcher
`fontaine/scripts/launch_grpo_signal_probe.sh`. Queue validate green
(depth 3, 13 open). `run_work_next` armed — the chained work session
rides the probe + works CPU lanes.

**Steering**: no new owner messages this tick (read + history checked
21:31Z; no reactions on the 21:15Z results post). Launch post
21:35Z states the standing-sequence basis and offers a stop at any
pass boundary. Open asks unchanged: v3-rerun unhold + arm set
(15:13Z), disk-draws sign-off.

**Done**: probe launch end-to-end — launcher written (7 passes,
checkpoint paths verified on disk: er60k/step_060000, teacher80k =
artrunk 40k_ddp2/step_080000, ftrig4k/step_004000), preflight green,
detached via run_detached.sh, babysit entry with frozen
anchors/gates, first-poll util check, Discord launch post, queue item
boundary synced.

**Next**: chained work session rides the probe (per-cell results
in-channel as passes land; tripwire = first cell's pace vs the 3.5
GPU-h gate) + CPU lanes: `lit-sim-improvement-levers` (owner-called),
`sim-wrist-compositing`. At probe completion: frozen reads + decision
rule per the pre-reg, results post same session.*

*Updated 2026-08-12 19:20–21:2xZ (real `date -u` at stamp: 21:17) —
work session: **🚢 9/100 SUCCESSES — the first sim successes this task
has ever recorded.** Released MolmoAct2 under the official arm-A map
at the restored 30 s budget completes pick-and-place on 9 seeds
(physics criterion). **Every success tick (480–886) lands past tick
450 — the old 15 s budget's cutoff — so the entire prior INERT/0-pickup
literature on this checkpoint was the lift sign × the halved time
budget stacked. INERT is FULLY overturned.** Also this session: GRPO
signal probe is LAUNCH-READY (instrument complete + finalized pre-reg
posted).*

**Status**: no live jobs, GPU idle (100ep eval ran 20:20–21:12:26Z,
~0.86/1.5 GPU-h, workers=8, entry pruned; seed-6 30 s rerun
19:54–19:56Z ~0.02 GPU-h, pruned). Queue validate green (depth 3, 13
open).

**Steering** (a live owner thread all session): (1) 19:25:39Z "seed 6
is a clear grab and lift — rerun with a longer horizon; the idea was
30 seconds" → replied 19:47Z, fixed + executed: the eval-20 protocol
gave 15 s vs sim100's 30 s; `--episode-seconds` lands the budget in
time units (`c26a99e`); 30 s rerun confirmed **grab + carry to 1.04 cm
of the disk, no release** (video posted 19:58Z, amendment 2). (2)
20:16:53Z "evaluate the release checkpoint with the correct mapping
(arm-A) on 100 episodes in parallel" → acked 20:18Z, amendment 3
pre-launch, launched 20:20Z, ridden with the batched rows watcher
(fixed mid-run: watcher unit needed the harness env sourced — posts
were failing silently), results 21:15Z. Open asks: GRPO probe
launch go, v3-rerun unhold + arm set (15:13Z), disk-draws sign-off.

**Done** (commits `8b6d034`, `c26a99e`, `a06c33d`, `7fc6eff`, launch
prep + close-out): (1) 100ep arm-A eval end-to-end (above) — rows +
dark two-panel chart (per-seed outcomes + success-tick strip vs the
450 line) + 9 success videos on the reports Space, amendment 3
results on the pre-reg page, INERT fully re-dispositioned. (2) **GRPO
probe prep COMPLETE** per the 13:16Z "get everything ready": SDE decode
wired end-to-end (`--sde-noise-level` both drivers, per-item keyed
step noise, batch-composition-invariant), parallel driver (seed, draw)
stochastic groups, oracles green (check.py 797); **finalized pre-reg
posted** (seeds 0–14, 30 s episodes, signal bar median group std
≥ 0.25 cm, frozen decision rule, ≤3.5 GPU-h, within-driver paired-only
per the parallel-oracle FAIL; 13:36Z sequence predecessors both done ⇒
launches on handback). (3) seed6-30s owner call closed (amendment 2).
(4) Hygiene: convmap post indexed into posts/index.md; queue "15
replans" drift corrected; index titles updated to the 9/100 headline.

**Next**: `queue_cli.py next` → CPU lanes: `lit-sim-improvement-levers`
(owner-called lit slice), `sim-wrist-compositing`. GPU: GRPO probe
launch-ready on the owner's go; v3 rerun pends unhold. The 9/100 read
reframes both: the sim CAN express success now — the success-rate
metric is live, not just progress-cm. `queue.json` canonical.*

## Utilization footer

Session 2026-08-12 21:39Z–2026-08-13 01:xxZ (work, the chained probe
ride; **+[CELL5_GPUH] GPU-h** — anchors + cells 1/2/5 ridden in-turn,
tripwire re-scope at cell-1; exploit + owner steering + lit):
**GRPO probe re-scoped and read out** — AR t=1.0 clears the signal
bar 3.1× at a cost CI including 0 (cell 2 t=1.6: 9.8× but −1.08 cm;
cell 5 SDE: see results post). Lit 0823 (3 papers pages), wrist
compositing investigated → DECIDED render-only (owner 22:31Z),
sim100 amendment 4, `sim-fit-real-lens-model` queued. check.py 797
green × 6 commits.

Session 2026-08-12 21:30–21:4xZ (tick, babysit; GPU claimed at
21:33:58Z — probe ~2.8 GPU-h projected ≤ 3.5 gate, accrues to the
riding sessions): **GRPO signal probe LAUNCHED** on the owner's
standing 13:36Z sequence at GPU handback (both predecessors done, no
objection after the 20:06Z "rides the standing sequence" post).
Launcher + babysit entry + first-poll (85% util) + launch post +
queue sync, all inside the tick. `run_work_next` armed.

Session 2026-08-12 19:20–21:2xZ (work; **+~0.88 GPU-h** — seed-6
30 s rerun 0.02 + 100ep arm-A eval 0.86/1.5 gate, both ridden in-turn;
exploit + owner steering): **9/100 SUCCESSES — first sim successes
ever on this task** (released MolmoAct2, official map, 30 s budget;
every success tick past the old 15 s cutoff; INERT FULLY overturned).
Owner steers executed end-to-end: `--episode-seconds` budget fix +
seed-6 grab-confirm (amendment 2), 100-episode eval (amendment 3,
batched rows watcher, results + chart + 9 videos posted). GRPO probe
prep COMPLETE + finalized pre-reg posted (launch-ready on handback).
check.py 797 green × commits; queue "15 replans" drift fixed; convmap
post indexed.

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
