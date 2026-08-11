# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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
commits replayed, zero conflicts (docs-only commit, no code risk;
check.py not re-run). (2) Queue: +`molmo-flow-step1-cli-rule` (CPU,
gates verbatim from the record) behind sim-lit-review;
`ae-on-our-trunk-prereg-draft` re-statused **absorbed** by §8.13
step 7 (owner-confirm pending); validate green, 11 open. (3)
Straggler triaged benign; ack + priority ask posted 18:19Z.

**Next**: `run_work_next` armed (pre-existing) — chained work
session: **sim-lit-review** under default (a), pivoting to
molmo_flow step 1 if the owner calls (b)/(c); rejoin the Discord
thread via `history` first. `rig-mixture-screen-exec` stays
owner-held. No dated boundaries — `queue.json` canonical.*

*Updated 2026-08-11 17:38–18:2xZ —
work session: **sim-review CLOSED (owner sim-pivot head item): findings
post + two committed probes; the contact-physics complaint is confirmed
and mechanism-attributed.***

**Status**: no live jobs — GPU 0% / 0 MiB (probe renders only, seconds
each; inference-only steer respected).

**Steering**: owner 17:28Z ("killall claude to get you to focus") —
acked in-channel 17:43Z; no other new messages. Sim pivot is being
executed in queue order.

**Done**: **sim-review CLOSED** (`f14948f`):
[findings post](posts/2026-08-11-sim-review-findings.md) + committed
probes `sim/probe_benchy_contact.py` + `sim/probe_phantom_volume.py`.
Contract seam ALL GREEN (camera-kind tags match training — the judge
stamped the rig's "front" key as kind *top*; kind-sorted image order
identical; stats/joints/degrees line up; measured bit-identical qpos
AND renders; 26.9 ms/tick ⇒ ~20 min sim-side per 100-seed eval). Four
findings, fixes deliberately not executed (findings-first): (1) home
pose UNREACHABLE — wrist-cam mount collision box jams into the
shoulder, three servos force-saturated, elbow −19° steady-state,
wrist_roll bimodal per seed; (2) 2/20 seeds the arm strikes the boat
during the reset settle (≤30 mm pre-episode); (3) benchy phantom
collision margin p99 3.8 / max 5.4 mm (74% of the collision surface
outside the visible boat); (4) jaw torsion weak — gripper
`priority=1` overrides the benchy drift-fix friction at the grasp
seam (6.9° in-grip spin, tilt to 0.84 upright on lift); rest drift
0.000 mm. Infra: menagerie commit unpinned + per-machine CoACD assets
⇒ pin one eval machine in the protocol; EGL runtime installed on this
box.

**Next**: `queue_cli.py next` → **sim-lit-review** (owner-sanctioned
sim lit lane, Papers pages same-session);
`sim-policy-eval-100seeds` protocol pre-reg is now draftable but must
fold in findings 1–2 fixes first. `rig-mixture-screen-exec` stays
owner-held (🅰️-vs-🅲). `run_work_next` armed. No dated boundaries —
`queue.json` canonical.*

*Updated 2026-08-11 17:27–17:5xZ (real `date -u` at write: 17:34) —
tick (babysit): **big tick — OWNER SIM PIVOT (17:07Z) acked + queue
re-shaped around it; option-🅱️ preflight verdict FITS posted; the
killed 16:29 work session's state salvaged.***

**Status**: no live jobs — `nvidia-smi` 0% / 0 MiB. The 16:29 work
session was hard-killed 17:26Z (exit 143, harness alert in-channel —
watch for repeats: possible usage cap) but its detached preflight
COMPLETED 17:14Z, log `~/train_mixture_fit_preflight.log`.

**Steering**: OWNER 17:07Z — **next-day focus = SIMULATIONS**:
review `sim/` first, lit review on so101-suitable sims + sim-to-real
(re-opens the paused lit lane for sim topics specifically), local
GPU dedicated to inference while sims run, initial goal = one good
policy on **100 fixed seeds** measuring boat→disk distance
reduction; boat (benchy) contact physics flagged as poor. ACKED
in-channel 17:36Z with the 3-step plan. (Owner 16:52Z box-deletion
fyi was already acked 17:03Z by the work session.)

**Done**: (1) boot audit: killed session's orphaned now.md/queue
state committed (`fb3e61f`), hallucinated clocks corrected (~30 min
fast vs file mtimes). (2) **Preflight read posted 17:35Z: FITS** —
peak 69.2/80 GiB, ~12.0 s/step steady ⇒ 10k steps ≈ **33.5 h**
single-H100 (above the 14–16 h estimate); `--dataset-repeat`
live-fired (clean 0.42% + v2 4.07% = **4.49%** combined effective
share vs the ~4.97% pre-reg estimate — reconcile before any exec);
🅱️ collides with the GPU-for-inference steer → rig-mixture treated
as **🅲 defer** unless the owner calls 🅰️. (3) Queue re-shaped:
+`sim-review` (new head), +`sim-lit-review`,
+`sim-policy-eval-100seeds` (blocked behind sim-review + protocol
pre-reg); `ae-on-our-trunk-prereg-draft` deprioritized behind the
sim lane. (4) `run_work_next` ARMED — chained work session starts
`sim-review`.

**Next**: work session: **sim-review** (map `sim/`:
so101_sim/rollout_sim/benchy scene, observation surface vs policy
input contract, determinism/seeds, contact-physics investigation),
then **sim-lit-review**. `rig-mixture-screen-exec` owner-held
(🅰️-vs-🅲 call pending). 16:26 + 16:07 entries and their footer
notes rolled to the [archive](archive/now-2026-08-11.md). No dated
boundaries — `queue.json` canonical.*

*Updated 2026-08-11 16:29–17:2xZ (written ~17:01Z per file mtime —
the entry's original in-session clocks ran ~30 min fast and were
corrected at the 17:27Z tick boot; session hard-killed 17:26Z exit
143 mid-preflight-babysit, state committed by the tick) —
work session: **rig-mixture-instrument-prereg CLOSED (instrument +
pre-reg draft + compute ask) AND the main rebase executed mid-session
on owner steering — `fontaine` now sits on main @36afff0, check.py
688 green.***

**Status**: no live jobs — `nvidia-smi` 0% / 0 MiB. `origin/main`
moved fdd9aa3 → **36afff0** (owner push 16:43Z) and the rebase is
DONE + force-pushed (`2a31981`); the blocker footer is retired.

**Steering**: owner 16:42/16:43Z — main pushed to GitHub + explicit
rebase ask; seen mid-session at the post-instrument Discord poll,
executed immediately (clean rebase, zero conflicts, only
`bijou/train.py` both-sides: wandb import move vs our new import; one
ruff I001 fixup). All 7 directive items sentinel-verified at HEAD;
result posted in-channel 17:00:42Z. **Compute ask A/B/C for the
rig-mixture rung posted ≤17:00Z — awaiting the owner call**; the
option-🅱️ fit-preflight was launched detached ~17:04Z (ack post
17:03:58Z) and survived the session kill.

**Done**: (1) **`--dataset-repeat` instrument landed** (`d6a9579`,
pre-rebase `1b1c314`): per-repo PATTERN=COUNT oversample specs in
`bijou.data`/`bijou.train`, first-match-wins, loud no-match, replicas
share objects (no host-RAM cost), training-only; oracle test
`tests/test_dataset_repeat.py` (16 tests incl. the pinned
0.19%→4.97%@27× arithmetic). (2) **Rig-mixture pre-reg DRAFT**
[posts/2026-08-11-prereg-er60k-rig-mixture.md](posts/2026-08-11-prereg-er60k-rig-mixture.md):
`--init-from er_60k/step_060000`, rig clean=27 v2=27 (~4.97% share),
10k steps seed 3; primary read = paired CI95 on the deterministic rig
holdout (1+5 episodes er_60k never trained on); guard = panel
non-regression +0.05 band vs banked 5.7782; compute ask (A) new 4×
box ~28 GPU-h / (B) local 1×H100 after an act-ckpt fit-preflight
(full recipe measured OOM single-GPU 08-08) / (C) defer. Space
pushed, post curl-200, ask in-channel. (3) **Main rebase CLOSED**
(`2a31981`): 11 branch commits replayed onto 36afff0, check.py 688
green, sentinels verified (released_so100_101, frozen no-default
ActionExpertConfig, staticmethod factories, 5 loud guards + tests,
CPU-side patch alignment, both edited fontaine scripts, tokenizer
FileNotFoundError).

**Next**: `queue_cli.py next` → **ae-on-our-trunk-prereg-draft**
(CPU, UNBLOCKED by the rebase — draft against the frozen-config
shapes at HEAD on `er_60k/step_060000`); `run_work_next` armed.
`rig-mixture-screen-exec` blocked on the owner compute call (ask
in-channel 17:1xZ) — tight-ish poll cadence owed while it pends. No
dated boundaries — `queue.json` canonical.*

## Utilization footer

Session 2026-08-11 17:38–18:2xZ (work, explore-infra; ~0 GPU-h —
probe renders only, no launches): sim-review CLOSED (f14948f):
findings post + 2 committed probes; contract seam all green
(bit-deterministic incl renders, ~20 min sim-side/100 seeds); 4
findings mechanism-attributed (unreachable home pose via camera-mount
self-collision, 2/20 reset boat strikes, phantom margin p99 3.8 mm,
weak jaw torsion via gripper priority=1); EGL runtime installed.
100-seed protocol pre-reg now draftable behind findings-1–2 fixes.
run_work_next armed for sim-lit-review.

Session 2026-08-11 17:27–17:5xZ (tick, babysit; ~0.2 GPU-h banked
from the detached preflight, 0 launched this session): OWNER SIM
PIVOT 17:07Z acked in-channel + queue re-shaped (+3 sim items,
sim-review head; AE draft deprioritized). Preflight verdict FITS
posted (69.2/80 GiB, ~12.0 s/step ⇒ ~33.5 h for 10k; 🅱️ collides
with the inference-only steer → 🅲 unless the owner calls 🅰️).
Killed 16:29 session salvaged at boot (exit 143, clocks fixed,
fb3e61f). run_work_next ARMED for sim-review.

Session 2026-08-11 16:29–17:2xZ (work, exploit-infra; killed 17:26Z exit 143; 0 new GPU-h —
GPUs free, no launches): rig-mixture-instrument-prereg CLOSED
(--dataset-repeat + 16-test oracle + pre-reg draft w/ compute ask
A/B/C in-channel) AND the main rebase executed on owner steering
mid-session (16:43Z push seen at the post-instrument poll; clean
rebase onto 36afff0, check.py 688 green, force-pushed 2a31981, all 7
directive items sentinel-verified, result in-channel). AE-draft item
unblocked; queue depth 1 + stated reason (exec item owner-held).

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
