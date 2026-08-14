# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 00:36–02:2xZ (real `date -u` at stamp: 02:16) —
work session: **`sim-arm-photometric-links` EXECUTED end-to-end
(`4515ab4`) — mined the real arm's pixels at recorded poses, fitted a
material grade through the production composite, and the registered
probe read passed BOTH bars: the missing term was shine, not paint.***

**Status**: **LIVE: `grpo_phase2_r1a`** — babysits 00:36/00:48/01:02/
01:08/01:34/02:04/02:15 all exit 0: 3 procs, GPU ~100% at ~34 GiB
(75-gate headroom ~41 GiB). First fresh rows landed: step 3 (loss
0.0385, eval 1.8441 — flat-at-noise, as the accumulation question
expects) and step 4 (loss 0.0343), ~2880 s/step incl. per-step eval
→ ~10.4 h to step 17 at the 02:15 read, rc within the ~14:3xZ ETA.
No tripwires, knockaway streak quiet.

**Steering**: none — inbox empty and read empty at every babysit poll.
NEW ASK OUT (02:1xZ, with the results post): promote
`arm_photometrics='v1'` into the production v3/v4 default? (Same
contract as the clutter-patch promotion ask, 05:40Z 08-13, still
open — they could flip together.)

**Done**: **sim-arm-photometric-links CLOSED** (`4515ab4`, pre-reg
posted 01:53Z BEFORE the read): (1) mining — sim posed at the
recorded joints of 142 real v2 frames, silhouette projected through
the production fisheye, per-body FFT darkness-snap ±60 px + ring +
absolute-darkness guards → 436k real PLA px + 77k servo px; real arm
reads brighter than the flat recolor (median luma 66 vs 54),
cool-cast, 16–18% glints vs sim's 5%/0%; (2) fit — albedo per channel
solved through the production composite, spec×shin by grid; both
populations chose the specular ceiling (1.0, shin 0.1), loss ↓8.5×/
2.3×; (3) opt-in `arm_photometrics="v1"` (default byte-identical,
zero RNG draws, 5 oracles, check.py 874→879); (4) registered 20×5
read GREEN — in-run v3 0.713 dead-center, **PRIMARY v3_photo CI95
[−3.08e-07, −1.38e-07] < 0 (0.713→0.698, 72/100), MECHANISM
only_links CI95 < 0 (0.705→0.652, 96/100) ≈ the no_mount amputation
ceiling without amputating**. Artifacts on fontaine-reports
(curl-200): chart, before/after strip, mining overlay, 3 JSONs;
reports.md section + ideas.md hook; results + promotion ask
in-channel 02:1xZ. Queue: item done; NEW —
`sim-arm-photometrics-promotion` (owner_hold),
`sim-mount-material-split` (the mount is WHITE in reality, black in
sim — per-pixel worst offender), `sim-arm-texture-followup` (print
layers + servo glint tail). Validate green (depth 3, 16 open).

**Next**: `queue_cli.py next` → **token-grpo-phase2-r1a-run** (ride
via ~30-min ticks to rc ~14:3xZ 08-14 → §6 endpoint reads → R2-A only
via the frozen rule). `run_work_next` armed — GPU busy,
`sim-mount-material-split` (CPU) is the next executable work item per
no-idle-pauses.*

*Updated 2026-08-14 00:34–00:4xZ (real `date -u` at stamp: 00:37) —
tick, babysit: **quiet tick — R1-A healthy 28 min into its overnight
leg, no steering, no anomalies.***

**Status**: **LIVE: `grpo_phase2_r1a`** — babysit 00:34:31Z exit 0:
3 procs, GPU 100% at 34.3 GiB (75-gate headroom ~41 GiB), step 2/17
(registered resume state; the step-3 fresh row lands ~01:0xZ,
just past this tick's cap — the next session catches it). Probe
1.87@0 → 1.84@1-2 vs baseline 1.868, flat-at-noise as the
accumulation question expects this early. Knockaway streak fresh.
rc ETA ~14:3xZ.

**Steering**: none — inbox empty, read empty at 00:34Z and at the
babysit poll; history shows no new reactions (👍 on the 22:10Z
pre-reg post already recorded; nothing on the 00:08Z GO post or the
00:32Z inbox-fix post).

**Done**: babysit poll (facts above); queue validate green (depth 2,
14 open).

**Next**: unchanged — ride **token-grpo-phase2-r1a-run** via ~30-min
ticks to rc ~14:3xZ; step-3 row is the first accumulation datapoint.
`run_work_next` stays armed (00:33 marker) — GPU busy and
`sim-arm-photometric-links` (CPU) queued; the chained work session
takes it per no-idle-pauses.*

*Updated 2026-08-14 00:20–00:4xZ (real `date -u` at stamp: 00:33) —
work session: **the `discord-unreplied-inbox` harness fix landed
(`2a362a1`) — the 08-13 missed-reply class is structurally closed:
consumed owner messages persist in an inbox until an explicit ack,
and the pending count prints as a truncation-proof first line in
read AND babysit.***

**Status**: **LIVE: `grpo_phase2_r1a`** — boot babysit 00:20:44Z
exit 0: 3 procs, GPU 100% at 34.2 GiB (75-gate headroom 41 GiB),
step 2/17 (registered resume state; first fresh row is step 3
~01:0xZ). Probe 1.87@0 → 1.84@1-2 vs baseline 1.868 — flat-at-noise
as expected this early. rc ETA ~14:3xZ.

**Steering**: none — read empty at boot 00:20Z and at the babysit
poll; history shows no new reactions.

**Done**: **discord-unreplied-inbox CLOSED** (`2a362a1`): `read`
appends every surfaced non-bot message to
`state/discord_unreplied.jsonl` (dedupe by id); `read` and babysit
print the pending count as a loud FIRST line (babysit re-checks
after its final poll); only an explicit `discord.py ack <id>` clears
— result posts never do; `discord.py inbox` reprints entries in full.
7 oracles in `tests/test_discord_inbox.py`, check.py 867→874 green;
ack contract added to tick.md + work.md; in-channel post 00:3xZ
closes the 21:05Z "being fixed" promise. Queue item closed
(validate green, depth 2, 14 open).

**Next**: `queue_cli.py next` → **token-grpo-phase2-r1a-run** (ride
via ~30-min babysit ticks to rc ~14:3xZ 08-14 → §6 endpoint reads →
R2-A only via the frozen rule). `run_work_next` armed —
`sim-arm-photometric-links` (CPU) is queued and the GPU is busy; the
chained work session takes it per no-idle-pauses.*

## Utilization footer

Session 2026-08-14 00:36–02:2xZ (work; ~0.02 GPU-h decided — the probe
embeds, run alongside R1-A which accrued ~2.1 GPU-h of its ~14.4 leg
under 7 in-session babysits; CPU item, exploit-sim):
sim-arm-photometric-links executed end-to-end inside the GPU-busy
window (mine → fit → sim patch → pre-reg → registered read GREEN,
`4515ab4`); promotion ask out; queue depth 3 (16 open).
`run_work_next` armed for sim-mount-material-split.

Session 2026-08-14 00:34–00:4xZ (tick, babysit; 0 new GPU-h decided —
R1-A live and healthy, ~0.45 GPU-h accrued on its ~14.4 leg): quiet
poll, no anomalies, no steering, inbox empty; queue green (depth 2,
14 open). `run_work_next` stays armed for sim-arm-photometric-links;
step-3 fresh row ~01:0xZ lands with the next session.

Session 2026-08-14 00:20–00:4xZ (work; 0 new GPU-h decided — R1-A
live throughout, ~0.4 GPU-h accrued on its ~14.4 leg; CPU item,
exploit-infra): discord-unreplied-inbox harness fix built, oracled,
landed (`2a362a1`) inside the GPU-busy window. `run_work_next` armed
for sim-arm-photometric-links.

Session 2026-08-14 00:18–00:2xZ (tick, babysit; 0 new GPU-h decided —
R1-A live and healthy, ~0.2 GPU-h accrued on its ~14.4 leg): quiet
poll, no anomalies, no steering; R0-A Hub upload verified complete.
`run_work_next` armed for the inbox-fix CPU item.

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
