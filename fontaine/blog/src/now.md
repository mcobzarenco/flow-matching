# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-11 18:38–18:4xZ (real `date -u` at write: 18:42) —
tick (babysit): **quiet tick — GPUs free, no new messages;
hallucinated queue clock fixed; `run_work_next` already armed for
sim-fixes-reset-contact.***

**Status**: no live jobs — registry empty (no_live_runs_reason
current), `nvidia-smi` 0% / 0 MiB. 100%-sim lane in effect.

**Steering**: Discord read empty; history clean — owner's 18:15Z
100%-sim call already acked (👍 recorded last session); no reactions
yet on the 18:36Z sim-lit-review summary.

**Done**: queue `updated_utc` hallucinated-clock audit hit — stamp
said 18:55Z but was committed 18:37Z; corrected to 18:39Z, validate
green (depth 2, 12 open). 17:38 body entry + 3 footer notes rolled to
the archive.

**Next**: `run_work_next` armed (18:37, pre-existing) — chained work
session: **sim-fixes-reset-contact** (CPU, blocks the 100-seed
pre-reg), then `sim-servo-sysid`. No dated boundaries — `queue.json`
canonical.*

*Updated 2026-08-11 18:13–18:4xZ — work session: **sim-lit-review
CLOSED (owner sim lane confirmed 18:15Z): 3 Papers pages live
same-session; the 100-seed protocol now has its design citations and
a fix list with named mechanisms.***

**Status**: no live jobs — GPU 0% / 0 MiB (inference-only steer
respected; research was CPU + web only).

**Steering**: owner 18:15Z answered the lane question — **"your
focus is 100% simulations, I have a local agent working on the
molmo_flow migration plan"** — acked in-channel 18:17Z (👍 received);
`molmo-flow-step1-cli-rule` parked owner_hold (owner-side lane,
steps 2–8 too). No other messages.

**Done**: **sim-lit-review CLOSED** — three Papers pages
([sim-as-eval](papers/sim-as-eval.md),
[SO-101 sim landscape](papers/so101-sim-landscape.md),
[contact fidelity](papers/sim-contact-fidelity.md)) via 3 parallel
research agents, links fetch-verified, Space pushed (all four pages
curl-200), summary + owner-facing headlines in-channel 18:34Z.
Substance: continuous distance metric vindicated (up to 70% fewer
trials than binary, 2603.13616); controller sysid is the first-order
eval-fidelity lever (SIMPLER ablation) and an asset diff surfaced a
56× kp discrepancy (menagerie 998.22/±2.94 — exactly the review's
measured saturation — vs TheRobotStudio 17.8/±3.35; BAM's identified
STS3215 model banked as prior); census: no public SO-101 sim eval
with a continuous metric exists; all four sim-review findings have
documented mechanisms + named fixes (CoACD threshold-not-cap or
native SDF which also closes the CC-BY-ND asset hazard; priority
override is spec → explicit jaw–boat pair + condim≥4 + elliptic
cones). ideas #16 fed. Queue: +`sim-fixes-reset-contact`,
+`sim-servo-sysid` (both CPU, from the fix list);
`sim-policy-eval-100seeds` boundary updated with the design
citations.

**Next**: `queue_cli.py next` → **sim-fixes-reset-contact** (CPU,
blocks the 100-seed pre-reg), then `sim-servo-sysid`.
`run_work_next` armed. No dated boundaries — `queue.json` canonical.*

*Updated 2026-08-11 18:07–18:3xZ — tick (babysit): **owner landed the
molmo_flow migration plan on main (§8.13, `128a863`, 17:50Z) — read,
rebased onto, queued as step-1 item; priority-vs-sim-lane question
posted in-channel.***

**Status**: no live jobs — GPU 0% / 0 MiB (inference-only steer
respected). Driver-guard straggler pid 35366 checked: a 6-day-old
idle tmux zsh, not a job — nothing to relaunch.

**Steering**: no new owner messages/reactions in-channel this tick.
But main moved 36afff0 → **128a863**: owner design record
**architecture §8.13 molmo_flow** (MolmoAct2 action expert as a
first-class bijou decoder; 10 registered decisions incl. ascending-t
convention for all new flow code, parallel copy w/ byte-parity
oracle, conversion-first loading, decoder-owned q01/q99, joint_ce
narration rider, `--insulate-expert` KI seam; steps 1–8 with gates,
est. 5–6 sessions ≤10 GPU-h; "plan approved in owner session
2026-08-11; step 1 (CLI rule) next"). Treated as steering: it
post-dates the 17:07Z sim pivot, so lane priority is ambiguous —
**asked in-channel 18:19Z (a) sims first / (b) step 1 first / (c)
interleave; defaulting to (a)** until answered. Tight-poll owed.

**Done**: (1) `fontaine` rebased onto main @128a863 — clean, 17
commits replayed, zero conflicts (docs-only commit; check.py 688
green via the pre-commit hook at push). (2) Queue: +`molmo-flow-step1-cli-rule` (CPU,
gates verbatim from the record) behind sim-lit-review;
`ae-on-our-trunk-prereg-draft` re-statused **absorbed** by §8.13
step 7 (owner-confirm pending); validate green, 11 open. (3)
Straggler triaged benign; ack + priority ask posted 18:19Z.

**Next**: `run_work_next` armed (pre-existing) — chained work
session: **sim-lit-review** under default (a), pivoting to
molmo_flow step 1 if the owner calls (b)/(c); rejoin the Discord
thread via `history` first. `rig-mixture-screen-exec` stays
owner-held. No dated boundaries — `queue.json` canonical.*

## Utilization footer

Session 2026-08-11 18:38–18:4xZ (tick, babysit; 0 new GPU-h — GPUs
free): quiet tick. Registry empty, nvidia-smi 0%/0 MiB. Discord read
empty; history clean (18:15Z 100%-sim call already acked + 👍; no
reaction yet on the 18:36Z summary). Queue validate OK (depth 2, 12
open); hallucinated updated_utc stamp (18:55Z written at 18:37Z)
fixed to 18:39Z. run_work_next already armed → sim-fixes-reset-contact
chains next. 17:38 body entry + 3 footer notes rolled to the archive.

Session 2026-08-11 18:13–18:4xZ (work, explore-lit; 0 GPU-h — CPU +
web research only): sim-lit-review CLOSED — 3 Papers pages
(sim-as-eval, so101-sim-landscape, sim-contact-fidelity) live +
curl-verified; owner lane call 18:15Z (100% sims) acked +
molmo_flow item parked owner-side; 56× servo-kp discrepancy found
vs upstream; queue +2 fix items (sim-fixes-reset-contact,
sim-servo-sysid); ideas #16 fed. run_work_next armed for
sim-fixes-reset-contact.

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
