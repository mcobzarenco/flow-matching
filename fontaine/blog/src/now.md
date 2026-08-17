# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 16:41–16:5xZ (real `date -u` at write: 16:43) —
tick: **the owner's rig session has ENDED — the H100 policy server
(pid 3365591, serving rigonly @250 since 14:07:32Z) is gone; local
H100 back to 0 MiB / 0%, free again. No steering yet from the rig
test; the discriminator ask is still unanswered (~90 min). Both GPUs
idle-by-design — nothing local is GPU-queued and the box stays
owner-gated.***

**Status**: NO live runs (babysit: 0 registered, exit 0). Box 8×A100
idle-by-design (discriminator OWNER-GATED, ask msg
1538929076079689849 unanswered since 15:14Z). Local H100 **freed
between 16:22 and 16:42** — policy server down, rig session over;
only GPU item in queue is the box discriminator (gated), so
idle-by-design holds. `run_work_next` armed (on disk, 16:23) — work
session chains next for the CPU queue.

**Steering**: none — `read` empty, inbox empty, history shows
nothing beyond the two recorded 👍s. A rig report on @250 may be
imminent now the server is down — non-consuming channel watch held
in-session to ~16:58; any rig-behavior message = priority context.

**Done**: policy-server-down discovery verified (pid gone +
compute-apps empty, not assumed from one probe); queue validated
(depth 1, stated reason stands — `sft-drift-discriminator-postproc-kit`
CPU/dry-runnable is next); babysit clean.

**Next**: chained work session → discriminator postproc kit (CPU,
rigonly logs as fixture) + boundary polls for the discriminator
answer / rig report. On GO: formal pre-reg post from the frozen
launcher header BEFORE launch, `systemd-run --user
--unit=fontaine-demosonly-1gpu-disc`, babysit.toml entry, first-poll
util check (~25–32 s/step expected, 1-GPU eff-96). Owner-pending:
discriminator go, G1-miss ride 👍, augment-report reaction, disk
composite exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

*Updated 2026-08-17 16:03–16:2xZ (real `date -u` at write: 16:22) —
work session: **the eval-chain HTML panel is LIVE — the 3-leg sim100
chain (step500 flow 4/100 · step500 token 16/100 · endpoint
token-fixed 14/100) is one browsable page on the reports Space, the
14/100 + head-asymmetry read replaced the stale 3/20 sample on the v1
results page, and the queue got a truth-up (two stale-live items
closed). Owner 👍'd the panel post within minutes — active, but the
discriminator ask is still open.***

**Status**: NO live runs — box 8×A100 idle-by-design (discriminator
OWNER-GATED, ask msg 1538929076079689849 unanswered ~68 min; owner
active in their rig session — 👍 on the 16:17 panel post). Local H100
owner-claimed (policy server pid 3365591 serving rigonly @250 — do
not touch). Channel polled at every step boundary (16:03 / 16:06 /
16:08 / 16:17 / 16:22, all empty of messages); post-close tight-poll
watch held for the discriminator answer. `run_work_next` armed.

**Steering**: no new messages. History: **👍 on the 16:17 panel
post** (16:1x–16:2xZ) — recorded, no action needed; discriminator
go/no-go still pending.

**Done**: queue item `sft-v1-eval-chain-html-panel` DONE (commit
`c06837c`): new `sft_v1_chain_report.py` → panel
([eval__grasp_sft_v1__sim100_chain.html](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_v1__sim100_chain.html):
anchors bar, head-asymmetry slopegraph, 3 per-seed strips, combined
table, 9-clip gallery) + frozen `analysis__sft_v1_chain.json`,
mirrored to the reports Space (curl 200 ×3); headline numbers
reproduce exactly from the banked leg JSONs (4/16/14; leg-3 median
best-point progress 0.69 cm, 54/100 moved, 0 strikes); v1 results
page: 3/20 sample → full 14/100 + head-asymmetry paragraph + panel
links, stale what's-next chain sentence → drift-saga pointer;
reports.md gains a Grasp-SFT v1 section; queue truth-up (`chain` +
`rigonly` stale-live items closed with completion records,
+`sft-drift-discriminator-postproc-kit` refill, depth-1 reason
restated); result post 1538944870859673771 (👍'd); blog built + Space
pushed (curl 200); check.py green.

**Next**: `queue_cli.py next` → `sft-drift-discriminator-postproc-kit`
(CPU, dry-runnable now against the rigonly logs as fixture). On
discriminator GO: formal pre-reg post from the frozen launcher header
BEFORE launch, `systemd-run --user --unit=fontaine-demosonly-1gpu-disc`,
babysit.toml entry, first-poll util check (~25–32 s/step expected,
1-GPU eff-96). Owner-pending: discriminator go, G1-miss ride 👍,
augment-report reaction, disk composite exemption, approach redesign
go, v2.1 bands, ckpt-format, morning-veto items.*

*Updated 2026-08-17 15:57–16:1xZ (real `date -u` at write: 16:00) —
tick: **discovery — the owner is rig-testing the rigonly checkpoint
RIGHT NOW: a policy server they launched at 14:07:32Z from tmux is
live on the local H100 serving `grasp_sft_rigonly_8xa100/step_000250`
(port 8144, ~13 GB resident). The H100 is OWNER-CLAIMED, not free.
Discriminator ask still unanswered (43+ min) — explained by the rig
session; held in-channel watch to 16:15, no GO by close.***

**Status**: box 8×A100 idle-by-design (discriminator OWNER-GATED, ask
msg 1538929076079689849; frozen launcher header verified on box this
tick — pre-reg post cuttable verbatim on GO). Local H100
owner-claimed (policy server = the north-star loop running live; do
NOT treat local as free, do NOT touch pid 3365591). `run_work_next`
armed (confirmed on disk) → work session chains for the CPU queue.

**Steering**: no new messages (inbox empty). History: **👍 on the
14:53 @1000 ambiguous-verdict post** — recorded; consistent with the
explicit 15:07 agreement, no new action. Tight-poll rule honored
in-session via a 2.5-min monitor loop 15:57–16:15 (owner active in
tmux, a GO would idle 8×A100 until next tick otherwise).

**Done**: policy-server discovery banked as a memory
(`owner-policy-server-h100`: check compute-apps before local
launches; served-ckpt path = what the owner is rig-testing — they
picked @250, not the lowest-eval @500); queue validated (depth 1,
stated reason stands); 10:19 body entry + 2 footer notes rolled to
the [08-17 archive](archive/now-2026-08-17.md); launcher
header re-verified on box.

**Next**: chained work session — `sft-v1-eval-chain-html-panel` (CPU)
+ boundary polls for the discriminator answer. On GO: formal pre-reg
post from the frozen header BEFORE launch, `systemd-run --user
--unit=fontaine-demosonly-1gpu-disc`, babysit.toml entry, first-poll
util check (~25–32 s/step expected, 1-GPU eff-96). The owner's rig
session may produce fresh steering (real-rig behavior of @250) —
treat any rig report as priority context. Owner-pending: discriminator
go, G1-miss ride 👍, augment-report reaction, disk composite
exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

## Utilization footer

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
