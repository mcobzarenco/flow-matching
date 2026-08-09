# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 11:11–11:3xZ (real `date -u`) — tick (babysit,
then conversational): **the 2500 probe uptick resolved as NOISE —
probe@3000 = 11.6124, a new best; then an owner throughput question
landed mid-close and was answered in-channel same tick.***

**Status**: attach_K healthy at the 11:12Z poll — step 2880/10k,
loss 3.25, 3.822 s/step (endpoint ~18:3xZ holds), vram 59.07 ≤ 71,
liveness 7 procs / all 4 GPUs loaded. Probe 11.67@2000 →
12.42@2500 → **11.6124@3000** (caught via a background watcher on
the box jsonl): the uptick was noise, the trajectory resumes
downward, and the first kill-bar 12.6394 (binds at ≥5k, ~13:2xZ)
now has ~1.0 of margin. CE aux flat. Local GPU free.

**Steering**: owner 11:14:53Z (caught on the pre-close `read`):
*where are we on increasing training throughput for molmo2 AR?*
Answered in-channel 11:25Z with the assembled record: (1) the 08-08
review's 8 findings; (2) **pass-1 executed and killed by its own
frozen rule** — true-recipe box ladder A 2.251 / B(+cuDNN suffix)
2.495 (**−10.8%**) / C(bundle) 2.415 (**−7.3%**) s/step, both
SLOWER, the 13× local microbench transfer falsified, P1 doubly dead
(parity loss-bound fail too); bitwise-safe subset landed `6a4b45e`
with no speed claim; (3) the live lever is **#20 actckpt** (crash
fixed `913fdc4`, flag field-validated on the K arm right now; 4-rung
ladder pre-reg drafted, ADOPT iff ≤1.02× control AND alloc ≤63 GiB,
frees batch 12→16–20/GPU; blocked on a fresh AR-trunk launch —
nothing AR-trunk is training now, so no run currently pays the
cost); (4) ViT SDPA / valid-row CE / fused RMSNorm unmeasured solo
(bundling hides sign), parked. Reply-watch held ~8 min after the
answer — quiet → normal cadence (chained work session rejoins if
the thread continues). No new reactions in `history`.

**Done**: babysit poll (exit 0, facts above); in-session hold for
the step-3000 probe (charter §6 — cheapest resolution of the
uptick watch item); queue validate green (depth 2, 8 open);
`run_work_next` confirmed armed from the 11:08Z close (the chained
work session picks up `lit-radar-hooks-17`).

**Next**: 5k kill-bar binds ~13:2xZ (probe must be < 12.6394 —
currently 11.61); endpoint ~18:3xZ → chained panel_v2 + AR-view
drift panel → **Δ_seam frozen read (runbook staged, pre-audited)**
→ stage-2 decision.

*Updated 2026-08-09 10:36–11:1xZ (real `date -u`) — work session
(bounded): **the #6 post-mortem map read out same session — KL is
rank-NOISE (not a reversed compass), SC was the better axis all
along at ~6× too weak, and the family failed twice independently;
plus a live owner exchange on compute-matched a(t)/b(t) schedules
that seeded the lit slice (LP-FT + VLM4VLA pages) and two more
queue items executed.***

**Status**: attach_K healthy at the ~11:06Z poll — step 2780/10k,
loss 3.20, 3.817 s/step (endpoint ~18:3xZ holds), vram 59.07 ≤ 71;
probe **11.67@2000 → 12.42@2500, an uptick** — still under the first
kill-bar 12.6394 which binds only at ≥5k (~13:0xZ), watch item for
the next poll. CE aux flat. Local GPU free.

**Steering**: owner 10:38Z (mid-babysit): *shouldn't F-vs-K be
compute-matched — frame it as loss a(t)·AR + b(t)·flow under a fixed
budget, what curves do you want?* Answered in-channel 10:48Z (two
posts): K pays ~4.1×/step (~14 vs 58 GPU-h per 10k) so matched-steps
over-serves K; the screen is deliberately the *mechanism* read with
an asymmetric rule — K ≤ F at matched steps ⇒ K dominated on the
whole compute axis (every constant-a>0 schedule dies in one run);
K > F ⇒ the win gets priced against 4× via a compute-matched
follow-up arm; F-then-joint is the cheapest non-constant a(t)
already queued. Owner 10:40Z: taps design 👍 (ack'd). No further
replies through 11:0xZ.

**Done**: (1) **`idea6-mcselect-postmortem` READ OUT** (`9939e33`):
`mcselect_postmortem.py` (reuses mcres/bbr/bijou scorers verbatim;
oracle: planted monotone fixture exact hand arithmetic + 6 abort
branches) → analysis json + raw sidecar npz + dated addendum with 2
dark-mode charts on the
[results post](posts/2026-08-09-mcselect-results.md). THE MAP:
per-row Spearman(KL, err) **+0.012 [−0.005, +0.029]** (rank-noise;
oracle-best UNIFORM on the KL axis, 0.498 vs 0.5, excess at BOTH
extremes ⇒ argmin fails too; harm is magnitude-driven — value-level
rho +0.126, winner's curse); SC **−0.030 [−0.046, −0.014]**
right-signed but ~6× too weak for an argmax (oracle-best at SC-top
30.1% vs 12.6% null); axes mutually uncorrelated (+0.032) — two
independent failures. Calibration bar for any learned-verifier
pre-reg: free rank signal tops at |rho| ≈ 0.03 toward the real
−0.250 ceiling. #6 escalation stays CLOSED. (2)
**`attach-seam-readout-audit` executed same session it was queued**:
attach_seam_results.py oracle green at HEAD, all stems verified
against the box files + launcher %06d padding, dry-run confirms the
clean pre-rsync abort; 3-step runbook staged into the attach_K
babysit anchors — tonight's Δ_seam read is copy-paste. (3)
**`lit-unfreeze-schedules` executed** (owner-steered slice, 2 papers
pages): [LP-FT](papers/lpft-two-phase-schedules.md) (2202.10054 +
NTK 2405.16747 — f-then-joint's THIRD citation, first with matched
frozen control + the feature-distortion theorem; compute-Pareto case
for step-function a(t); explicitly silent on F-vs-K since K's
stop-grad blocks the distortion channel) and
[VLM4VLA](papers/vlm4vla-trunk-ablation.md) (2601.03309 — frozen
vision encoder loses uniformly across 9 trunks × 3 sims ⇒ external
prior for #17's thawed arm; VQA→control proxy collapse off-Calvin ⇒
trunks are priced by panel screens only; NOT compute-matched, caveat
loud). index/SUMMARY/ideas #4 + #17 hooks updated.

**Next**: `queue_cli.py next` → attach_K kill-bars first BIND at
5000 (~13:0xZ; probe uptick watch); endpoint ~18:3xZ → chained
panel_v2 + AR-view drift panel → **Δ_seam frozen read (runbook
staged, pre-audited)** → stage-2 decision (unblocks the
triple-cited f-then-joint draft). Queue depth 2
(`lit-radar-hooks-17` executable any GPU-busy window).

*Updated 2026-08-09 10:29–10:5xZ (real `date -u`) — tick
(conversational): **a dropped owner conversation caught and repaired
— the 08:16Z "why does KI-joint exist" question AND the 09:53Z "did
you miss my previous message?" follow-up had both been
cursor-consumed unanswered; answered in-channel 10:36Z, reply-watch
held through the tick.***

**Status**: attach_K healthy at the 10:29Z poll — step 2240/10k,
loss 3.26, 3.803 s/step steady (endpoint ~18:3xZ holds), vram 59.07
≤ 71, probe 15.92@500 → 13.08@1000 → 13.01@1500 → **11.67@2000,
already under the first kill-bar (12.64@5k) three probes early**;
CE-health aux ~2.6 flat (no drift signal). Local GPU free. Babysit
exit 0.

**Steering**: **two owner messages had been missed** (consumed by
`read` during earlier run-triage, never replied — the owner had to
ping). Both answered 10:36Z: (1) *why KI*: the arms are
gradient-decoupled but NOT equivalent — K's trunk keeps taking CE
steps on the robot-episode stream (text-lr 2e-5), so the residual
taps the expert reads keep adapting to the deployment distribution;
the π0.5-KI bet is that insulated adaptation outweighs the
moving-target cost the owner named, Δ_seam prices exactly that, and
F tying ⇒ frozen also wins on cost (no trunk backward). Drift is
instrumented (CE-health watch + read-4 |Δ_AR| ≤ 0.3). (2) *what the
expert attends*: NOT K/V export like the Gemma-4 path — Molmo2's
uniform full-attention stack has no KV-share boundary, so the pinned
rule is residual taps: hidden states after layers 2, 5, …, 35
(stride 3, last tap on the final layer; 12 taps = 12 expert layers)
through learned expert-side adapters into the trunk's GQA geometry
(8 kv-heads × head_dim 128, RoPE θ=5M), stop-grad on the taps.
Feedback memory recorded: `read` is consume-once — every owner
message it surfaces gets a same-session in-channel reply; result
posts don't count.

**Done**: the two in-channel answers; babysit poll (facts above);
queue validate green (depth 2, 8 open); archive roll (keep-3).

**Next**: attach_K kill-bars first BIND at step 5000 (~13:0xZ);
endpoint ~18:3xZ → chained panel_v2 + AR-view drift panel → Δ_seam
frozen read at matched endpoints → stage-2 decision. CPU window
(chained work session, `run_work_next` armed):
`idea6-mcselect-postmortem` (record-only, banked dump) + rejoin the
owner thread if it continues (`history` rebuilds context).

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
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + chained
panel_v2 eval live (~1–2 GPU-h; batch gate 70, rate-gate projection
50.3 incl. K estimate)). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 10:29–10:5xZ (tick, conversational; 0 GPU-h):
recovered a dropped owner exchange — the 08:16Z KI-rationale
question and the 09:53Z cross-attention follow-up had been
cursor-consumed unanswered; both answered in-channel 10:36Z (KI =
insulated trunk adaptation vs the moving-target cost, Δ_seam prices
it; Molmo2 attach = residual taps 2,5,…,35 via adapters, not K/V
export), history-diff reply-watch held through the tick, feedback
memory recorded (read is consume-once — same-session replies
mandatory). attach_K healthy: probe 11.67@2000, already under the
5k kill-bar. Queue validate green depth 2; run_work_next armed.

Session 2026-08-09 11:11–11:3xZ (tick, babysit → conversational; 0
GPU-h): attach_K step 2880/10k healthy (3.822 s/step, vram 59.07 ≤
71, endpoint ~18:3xZ); held through the step-3000 probe boundary —
11.6124@3000, new best: the 2500 uptick was noise, first kill-bar
(12.64, binds ≥5k ~13:2xZ) has ~1.0 margin. Owner 11:14:53Z
throughput question answered in-channel 11:25Z (pass-1 killed by
its own rule at −7.3%/−10.8% on the true recipe, subset landed
speed-claim-free, #20 actckpt = the staged lever, ladder blocked on
the next fresh AR-trunk launch); reply-watch ~8 min, quiet. Queue
validate green depth 2; run_work_next armed (work session rejoins
the thread via history if it continues).
