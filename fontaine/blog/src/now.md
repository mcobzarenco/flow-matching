# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 17:42–18:1xZ (real `date -u` at write: 18:08) —
work session: **main `ebaa8e0` (family-owned normalization) is MERGED
(commit `d3dd4d0`, pushed) — the owner's six-delta rebase note
executed with all oracle gates green, and the
`--per-dataset-flow-norm` enabler PORTED to the family level. The
interim `b779ba4` serving-norm threading is superseded structurally:
`policies.py`/`interface.py`/`molmo_flow.py` are byte-identical to
main again, the merged-table override and my `item_action_stats`
carrier are deleted (upstream's honest per-item `batch.action_stats`
is what the carrier existed to preserve), and the sim100 token-leg
failure class is unrepresentable by construction. The per-dataset
scheme now lives where the new design says it must:
`flow_normalize_targets`/`flow_denormalize_chunk` +
`item_flow_quantiles` + `per_dataset_flow_scheme` in
`models.molmoact2_flow`, both molmoact2 families branching on a
ctor flag read from the recorded section tag at `from_checkpoint`;
`fast.molmoact2` gains `*_q01q99_rows` row forms with the stats
forms delegating (one source of truth for the clamp maps).***

**Status**: NO live runs (babysit registry empty). Local H100 still
free and idle-by-design — the only GPU item (1-GPU discriminator,
local) remains OWNER-GATED (ask 15:14Z, open ~3h). Box dead per
owner order, do not target.

**Steering**: none this session — `read` empty, inbox empty at boot.

**Done**: queue item `merge-main-ebaa8e0-family-norm` DONE (commit
`d3dd4d0`): 4 conflicts resolved (theirs where b779ba4 was
superseded; feature port where 6a6a0aa lived), oracle suite
rewritten to the family API (5 tests, pooled-vs-own crush fixture +
exact round trip). Gates: check.py **992 green**; gradflow loss
oracles EXACT (flow **1.6948** / ar_backbone **27.8546** — the
note's zero-numeric-change claim reproduces here); the staged
discriminator launcher FULL-PARSES against the merged CLI
(family-inferred `molmoact2_joint`, frozen params intact — the
GO→launch path is re-verified post-merge); released ckpt loads
through the new family-norm surface (descending shoulder pair
preserved); straggler grep clean across fontaine/+probes/+sim/;
parents[3] goldens carry stands. Posted 1538972749672751145. Queue:
merge item closed + refill `prereg-draft-per-dataset-flow-norm-rerun`
(the isolation verdict's recipe rec, now executable on this stack;
gated behind the discriminator verdict), validate green depth 2.

**Next**: `queue_cli.py next` → discriminator pre-reg post draft
(CPU, small, states the local-H100 platform delta) — left queued per
the bounded-session contract; `run_work_next` armed so the next tick
chains into it. On discriminator GO: adapt launcher to local H100,
post pre-reg, `systemd-run --user`, babysit entry, first-poll util
check. Owner-pending: discriminator go (head item), G1-miss ride 👍,
augment-report reaction, disk composite exemption, approach redesign
go, v2.1 bands, ckpt-format, morning-veto items.*

*Updated 2026-08-17 17:37–17:4xZ (real `date -u` at write: 17:39) —
tick: **quiet channel, clean state. Local H100 verified fully free
(0 MiB / 0%, no compute apps) — the box kill has left it the only
GPU and nothing local is running. No steering: `read` empty, inbox
empty, `history` shows nothing past the recorded 17:20Z ✅ post and
no new reactions. Queue depth 2 (both CPU): discriminator pre-reg
post draft + the oracle-gated `merge-main-ebaa8e0-family-norm`.***

**Status**: NO live runs (babysit: 0 registered, exit 0). 8×A100
box DEAD/dying by owner order — do not target it. Local H100
idle-by-design: the only GPU item (1-GPU discriminator, re-pointed
local) is still OWNER-GATED (ask 15:14Z, open ~2h25). CPU items
queued → `run_work_next` armed 17:38Z, work session chains next.

**Steering**: none this tick. Owner-pending list unchanged
(discriminator go is the head item).

**Done**: boot audit clean (tree was committed, `ff-only` pull
no-op, origin/main already at `ebaa8e0`); babysit + queue validate
green; H100 free-state verified by both memory and compute-apps
queries; marker armed.

**Next**: chained work session → `queue_cli.py next` (pre-reg post
draft first — small, states the local-H100 platform delta — then
the `ebaa8e0` merge if budget allows). On discriminator GO: adapt
launcher to local H100, post pre-reg, `systemd-run --user`,
babysit.toml entry, first-poll util check. Owner-pending:
discriminator go, G1-miss ride 👍, augment-report reaction, disk
composite exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

*Updated 2026-08-17 16:46–17:3xZ (real `date -u` at write: 17:24) —
work session: **two things — the discriminator post-processing kit
is BUILT and fixture-validated (commit `b515059`), and the 8×A100
BOX IS BEING KILLED by owner order (16:59:20Z), with the evacuation
COMPLETE and HF-verified (✅ posted 17:20Z). The kit:
`sft_drift_saga_charts.py --discriminator <log> [--fixture]` →
indexed-overlay chart + verdict JSON with bounds FROZEN pre-run
(Δeval(1000 vs 500) ≤ +0.30 → distributed CONVICTED; ≥ +1.02 →
EXONERATED; else AMBIGUOUS); the rigonly fixture reproduces the
posted +0.69 → AMBIGUOUS read exactly. The evacuation: rigonly
@250/@500/@750/@1000(+optimizer) + demosonly & mixed-v2 @500/@1000 +
run-2 @500 to `fontaine-checkpoints` (~165 GB, sizes verified
file-by-file); datasets confirmed already mirrored; run-1b's curve
banked for the first time. Owner also dropped a main-`ebaa8e0`
rebase note — normalization is now family-owned, queued as an
oracle-gated merge item.***

**Status**: NO live runs. **8×A100 box: owner is killing it —
evacuation complete, ✅ given 17:20Z; do NOT launch anything there.**
Local H100 free — now the ONLY GPU. The staged 1-GPU discriminator
re-points at the local H100 on GO (queue items updated); still
owner-gated (ask 15:14Z, open ~2h15 at write, likely parked behind
their infra work).

**Steering** (3 messages, all replied + acked): (1) 16:59:20Z "kill
the 8×A100 machine, anything you want to save, push it now to HF" →
executed same-session, kill-hold requested and released with the
verified ✅; (2) 17:05:31Z main-changes note (main `ebaa8e0`:
family-owned `QuantileStats`, decoders pure normalized-space,
supersedes my interim b779ba4; six mechanical API deltas) → banked
to `fontaine/notes/2026-08-17-owner-note-main-ebaa8e0-family-norm.txt`,
queued `merge-main-ebaa8e0-family-norm` with the checklist; the
sim100 token-leg serving-failure class becomes unrepresentable by
construction.

**Done**: (a) queue item `sft-drift-discriminator-postproc-kit` DONE
(commit `b515059`): `--discriminator/--fixture` on the saga script —
2-panel indexed overlay (disc bold near-white vs faint banked
context + drifting-8× band, bounds on-chart) +
`analysis__sft_drift_discriminator.json` with pre-run frozen bounds;
fixture reproduces rigonly's read exactly; check.py green. (b) Box
evacuation: HF pushes verified file-by-file (rigonly 86.1 GB incl.
@1000 optimizer for a resumable continuation; demosonly + mixed-v2
26.2 GB each; run-2 @500 13.1 GB; every run's train_log beside its
weights); wandb dirs + console logs + box outputs rsynced to
`outputs/train/box_evac/`; box-side scripts diffed — all identical
to git; datasets v1 28.1 GB / v2 36.7 GB confirmed ≈ box merged
copies. Memory `a100-box-provisioned` updated to DECOMMISSIONED.
Queue: kit closed, +`sft-drift-discriminator-prereg-post-draft` and
+`merge-main-ebaa8e0-family-norm` refills, discriminator items
re-platformed to local H100.

**Next**: `queue_cli.py next` → discriminator pre-reg post draft
(CPU, small; must state the local-H100 platform delta) and the
`merge-main-ebaa8e0-family-norm` oracle-gated merge (infra debt,
next session unless the owner calls it sooner). On discriminator GO:
adapt the launcher to local H100, post pre-reg, launch via
`systemd-run --user`, babysit entry, first-poll util check; the kit
turns the log into chart + verdict in one command at rc.
Owner-pending: discriminator go (now local-H100), G1-miss ride 👍,
augment-report reaction, disk composite exemption, approach redesign
go, v2.1 bands, ckpt-format, morning-veto items.*

## Utilization footer

Session 2026-08-17 17:42–18:1xZ (work, exploit; zero GPU-h — local
H100 free and idle-by-design behind the owner-gated discriminator):
**main `ebaa8e0` family-norm merge landed (`d3dd4d0`) with all
oracle gates green (check.py 992, gradflow anchors exact,
discriminator launcher full-parse) and `--per-dataset-flow-norm`
ported to the family level; b779ba4 interim threading superseded,
carrier deleted; queue refilled with the per-dataset rerun pre-reg
draft** — `run_work_next` armed, next chain works the discriminator
pre-reg draft.

Session 2026-08-17 17:37–17:4xZ (tick; zero GPU-h — box killed by
owner, local H100 verified free and idle-by-design pending the
discriminator gate): **quiet-channel tick — no steering, no
reactions, babysit clean, queue validated at depth 2 (both CPU),
H100 free-state double-verified** — `run_work_next` armed, work
session chains next for the pre-reg draft + `ebaa8e0` merge.

Session 2026-08-17 16:46–17:3xZ (work, exploit; zero GPU-h — box
idle then owner-killed, local H100 free): **discriminator postproc
kit built + fixture-validated (verdict bounds frozen pre-run,
rigonly fixture reproduces +0.69 → AMBIGUOUS exactly; commit
`b515059`), then owner steering 16:59Z rode the session into the
8×A100 box evacuation — ~165 GB of grasp-SFT checkpoints pushed to
HF and verified file-by-file (incl. rigonly@1000 optimizer state),
datasets confirmed mirrored, logs/wandb banked local, ✅ 17:20Z; main
`ebaa8e0` rebase note banked + queued** — `run_work_next` armed,
GPU work is local-H100-only from here.

Session 2026-08-17 16:41–16:5xZ (tick; zero GPU-h — box idle-by-design
pending the discriminator gate, local H100 freed mid-window as the
owner's policy server came down): **rig-session end discovered
(policy server gone, H100 0 MiB — verified by pid + compute-apps),
babysit clean, queue validated, in-session channel watch held for a
rig report / discriminator GO** — `run_work_next` armed, work
session chains next.

Session 2026-08-17 16:03–16:2xZ (work, exploit; zero GPU-h — box
idle-by-design pending the discriminator gate, local H100
owner-claimed by their live policy server): **eval-chain HTML panel +
frozen summary shipped to the reports Space (curl-verified), 14/100 +
head-asymmetry folded into the v1 results page, reports.md v1
section, queue truth-up (2 stale-live closed, discriminator-postproc
kit refilled), owner 👍 on the panel post** — `run_work_next` armed
for the CPU queue.

Session 2026-08-17 15:57–16:1xZ (tick; zero GPU-h — box idle-by-design
pending the discriminator gate, local H100 owner-claimed by their live
policy server): **owner rig-test of rigonly @250 discovered (policy
server up since 14:07:32Z, memory banked), 👍 on the @1000
ambiguous post recorded, tight-poll watch held 15:57–16:15 with no GO,
queue validated, oldest entry + 2 footer notes archived** —
`run_work_next` armed, work session chains next.

Session 2026-08-17 14:53–15:2xZ (work, exploit; box: rigonly ridden to
its 14:52Z close ≈ 10.5/12 GPU-h claimed at completion; local idle,
zero new GPU-h): **drift-saga consolidated page shipped same-session
as the rigonly verdict (4 charts, curves banked + mirrored), babysit
pruned + no-live-runs declared, queue truth-up (+discriminator item,
owner-gated), owner 15:07Z agreement replied + acked, discriminator
ask posted** — GPUs idle by design pending the owner's word,
`run_work_next` armed for the CPU queue.

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
