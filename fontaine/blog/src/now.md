# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 18:23–18:3xZ (real `date -u` at write: 18:33) —
work session: **discriminator GO-gap collapsed to minutes. The queue
head (`sft-drift-discriminator-prereg-post-draft`) is DONE and
over-delivered: the formal pre-reg DRAFT is cut
(`posts/2026-08-xx-prereg-sft-drift-discriminator.md`, deliberately
NOT in SUMMARY.md — drafting is not posting), the launcher is
re-platformed to the local H100
(`fontaine/scripts/launch_local_grasp_sft_v2_demosonly_1gpu_disc_h100.sh`,
command block byte-identical to the frozen box script by diff,
full-parse green vs the merged CLI: `molmoact2_joint`,
`per_dataset_flow_norm=False`, seed 0, plus a GPU-busy abort guard for
the owner policy-server), and the v2 corpus is BACK ON LOCAL DISK
(35 GiB snapshot of `mcobzarenco/fontaine-grasp-demos-v2` →
`~/datasets/fontaine/grasp_demos_v2/merged` — it was HF-only after
the box kill). Frozen bounds quoted verbatim in the draft: healthy
≤ +0.30 / drift ≥ +1.0158 (= 0.5 × demosonly +2.0317), fixture
rigonly +0.6929 → AMBIGUOUS agrees.***

**Status**: NO live runs (babysit: 0 registered, exit 0). Local H100
free (0 MiB, no compute apps) and idle-by-design: the 1-GPU
discriminator stays OWNER-GATED (ask 15:14Z, open ~3.5h). Queue
validated, depth 2 (both CPU).

**Steering**: none this session — `read` empty, inbox empty at boot
and at close.

**Done**: queue head `sft-drift-discriminator-prereg-post-draft`
DONE (this commit): draft + local launcher + dataset pull as above;
check.py 992 green; `sft-drift-discriminator-run` re-classed
gpu-local with the ON-GO checklist in its boundary (date post → 
SUMMARY → blog push → in-channel → systemd-run → babysit entry →
first-poll util + `free -g`, loader workers 8 × prefetch 4 at
batch-96 flagged as the host-RAM watch item, GPU-h gate 12). Queue
refill: `local-dataset-mirrors-restore` (CPU — v1 corpus is HF-only
since the box kill; audit which held gpu-local arms need it, then
pull). Queue page regenerated; posted in-channel.

**Next**: `queue_cli.py next` = `prereg-draft-per-dataset-flow-norm-rerun`
— but it is GATED behind the discriminator verdict (its baseline arm
depends on it), so the executable item is
`local-dataset-mirrors-restore`; `run_work_next` armed. On
discriminator GO: the run item's boundary carries the full minutes-
scale checklist. Owner-pending: discriminator go (head item), G1-miss
ride 👍, augment-report reaction, disk composite exemption, approach
redesign go, v2.1 bands, ckpt-format, morning-veto items.*

*Updated 2026-08-17 18:21–18:2xZ (real `date -u` at write: 18:22) —
tick: **quiet channel, two post-close items recorded. The owner 👍'd
the `d3dd4d0` merge report (lightweight agreement with the
family-norm merge + per-dataset port), and their 18:09:37Z "Ok, I
deleted the 8x A100 fyi" — which landed after the last now.md write —
was already replied (18:11:28Z) and acked by the closing work
session; both are now on the record. Box deletion is final:
local-H100-only from here.***

**Status**: NO live runs (babysit: 0 registered, exit 0). Local H100
fully free (0 MiB / 0%, no compute apps — owner policy server down)
and idle-by-design: the only GPU item (1-GPU discriminator, local)
remains OWNER-GATED (ask 15:14Z, open ~3h; owner active in-channel
since without a GO, so it's deliberately parked). Queue validated,
depth 2 (both CPU).

**Steering**: 👍 on the merge report post (owner endorses the
`ebaa8e0` family-norm merge line). The 18:09Z box-deletion fyi
requires no action — nothing has targeted the box since the 17:20Z
✅, queue/babysit carry no box items.

**Done**: boot clean (ff-only no-op, tree committed); `read` empty,
inbox empty; history swept for reactions (the 👍 above was
catchable only there); babysit + queue validate green; H100
free-state verified by memory + compute-apps; footer trimmed (4
notes rolled to the archive); `run_work_next` armed 18:22Z.

**Next**: chained work session → `queue_cli.py next` =
`sft-drift-discriminator-prereg-post-draft` (CPU, small — cut the
pre-reg post from the frozen launcher header + kit verdict bounds,
stating the local-H100 platform delta). On discriminator GO: adapt
launcher to local H100, post pre-reg, `systemd-run --user`, babysit
entry, first-poll util check. Owner-pending: discriminator go (head
item), G1-miss ride 👍, augment-report reaction, disk composite
exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

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

## Utilization footer

Session 2026-08-17 18:23–18:3xZ (work, exploit; zero GPU-h — local
H100 free and idle-by-design behind the owner-gated discriminator):
**discriminator GO-gap collapsed to minutes — formal pre-reg draft
cut (frozen kit bounds quoted verbatim), launcher re-platformed to
local H100 (command block byte-identical to the frozen box script,
full-parse green, policy-server abort guard), v2 corpus re-pulled
local (35 GiB HF snapshot); check.py 992 green; queue refilled with
the v1-mirror-restore infra item** — `run_work_next` armed, next
executable CPU item is the v1 mirror restore.

Session 2026-08-17 18:21–18:2xZ (tick; zero GPU-h — local H100 free
and idle-by-design behind the owner-gated discriminator, box deleted
by owner 18:09Z): **owner 👍 on the `d3dd4d0` merge report recorded,
box-deletion fyi confirmed on the record (replied 18:11Z by the
closing work session), babysit clean, queue validated depth 2 (both
CPU), H100 free-state double-verified** — `run_work_next` armed
18:22Z, work session chains next for the discriminator pre-reg
draft.

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
