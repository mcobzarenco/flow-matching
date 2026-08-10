# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 04:03–04:2xZ (real `date -u` at write: 04:15) —
tick (babysit): **tiny10k HOST-RAM OOM at step ~9,060 → RESUMED from
step_008750.** The 04:00:55Z systemd oom-kill (host RAM, not GPU —
vram 36.6/80, loss 0.13x, probe **9.37@9000** run-best at death) cost
~310 steps; relaunched 04:06Z as unit `fontaine-tiny10k-r8750` with
full `--resume` (optimizer/scheduler verified at lr 1.38e-05, exactly
on the pre-kill cosine), fresh shuffle seed 1 (trainer-enforced +
standing resume-seed policy; eval-seed stays 0 → probe ladder
comparable), workers 10→6 for RAM headroom. One rc2 false start:
`--backbone-init-from` is mutually exclusive with `--resume` —
dropped. **New endpoint ~05:1xZ** + chained panel_v2 @10000; the
Δ_capacity read still lands this morning.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~8,120, probe … 15.75@7500 (transient) → **7.65@8000** run-best
(matched deltas … −0.48, +7.11 transient, **−0.98**), 26.3 st/min,
util 95–96% ×4, vram ~71.7 ×4 vs 77 bar, ~20.9/155 GPU-h; endpoint
~08-11 ~12:00Z. `fontaine-tiny10k-r8750` LIVE local — resumed at
step 8,750/10,000, rungs @9000–@9500 re-run on the resumed path,
then the @10000 primary read vs banked F@10k 9.4157; endpoint
~05:1xZ + chained panel eval.

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 1 → diagnosed: tiny10k unit oom-killed by
the host at 04:00:55Z (journal), GPU empty, step_008750 async save
complete on disk. Wrote `launch_local_tiny10k_resume8750.sh`
(verbatim recipe + the three resume deltas above), launched via
systemd-run, verified the resume banner (expert + adapted backbone +
optimizer at step 8750). Incident + relaunch posted in-channel.
babysit.toml re-pointed (r8750 log, OOM+RESUME history in boundary).
Queue validate OK: depth 0 pickable WITH stated depth_reason (lit
pause). run_work_next left unarmed — the ~05:1x–05:3xZ tick chain
owns tiny10k post-processing (panel_v2 → Δ_capacity read). Body +
footer rolled per last-2 (03:23 block + 03:34 note → 08-10 archive).

**Next**: tiny10k endpoint ~05:1xZ → chained panel_v2 → Δ_capacity
read @10k (vs banked F@10k 9.4157) — tiny was 0.05 UNDER F's
endpoint at step 9000 pre-kill; |Δ|≤0.3 "prior confirmed" vs "tiny
wins" leaning tiny-wins, the @10k paired CI95 decides. Watch the
resumed run's first probe rung (@9000 re-run, fresh data order) for
seed-1 consistency with the pre-kill 9.37. er_60k rungs record-only
to endpoint ~08-11 ~12:00Z; @7500-class transient recurrence
upgrades to a posted fact. No lit refills until the owner
re-enables.*

*Updated 2026-08-10 03:45–04:0xZ (real `date -u` at write: 04:00) —
tick (babysit): **er_60k spike-and-recover — probe **15.75@7500**
(a 2× excursion off 8.30@7000) resolved at @8000 into **7.65 NEW
RUN-BEST**, Δ −0.98 vs the 40k's 8.6371 matched, the largest
negative delta of the run. Anomaly scan on the spike: flow loss FLAT
through it (3.65–3.73 over steps 7300–7800), train_mae spiked and
recovered in lockstep with eval (16.93 → 7.71) → a one-rung
decode-probe excursion, not training divergence; the 40k baseline
never spiked like this at any rung; kill line (>25 ×3) never
approached. Held the session through both eval boundaries (§6) to
see it resolve. tiny10k **9.37@9000** run-best — already 0.05 UNDER
the banked F@10k endpoint 9.4157, with @9500 + the @10000 primary
read still to come.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
8,000, probe … 8.30@7000 → 15.75@7500 → **7.65@8000** (matched
deltas −0.57, +0.12, −0.38, −0.47, +0.44, −0.73, −0.48, +7.11
transient, **−0.98**), 25.4 st/min window, util 100% ×4, vram ~71.7
×4 vs 77 bar, ~20.7/155 GPU-h; endpoint ~08-11 ~12:00Z.
`fontaine-tiny10k` LIVE local — step ~9,040, probe 9.62@8500 →
**9.37@9000** (rung @9500 remains, then the @10000 primary read vs F
9.4157), 23.6 f/min, ~7.8/15 GPU-h; endpoint ~04:4xZ (≈960 steps
left).

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 0, then the anomaly scan above (remote
log: loss window + full probe ladder with train_mae pairs) and an
in-session hold through the @8000/@9000 boundaries — both resolved
green; record-only, no escalation (the spike self-resolved within
one rung; next tick's Δ_capacity post carries it as a rider). Queue
validate OK: depth 0 pickable WITH stated depth_reason (lit pause; 8
open = 2 live + 6 owner-gated/blocked). run_work_next left unarmed —
no CPU items open; the ~04:4xZ tick chain owns tiny10k
post-processing (panel_v2 → Δ_capacity read). Body + footer rolled
per last-2 (03:12 block + 03:23 note → 08-10 archive).

**Next**: tiny10k @9500 rung then endpoint ~04:4xZ → chained
panel_v2 → Δ_capacity read @10k (vs banked F@10k 9.4157) — with tiny
already under F's endpoint at step 9000, |Δ|≤0.3 "prior confirmed"
vs "tiny wins" is now leaning tiny-wins; the @10k paired CI95
decides. er_60k rungs record-only to endpoint ~08-11 ~12:00Z; watch
for @7500-class transient recurrence — a repeat upgrades it from
record-only to a posted fact. No lit refills until the owner
re-enables.*

*Updated 2026-08-10 03:34–03:3xZ (real `date -u` at write: 03:36) —
tick (babysit): **tiny10k **9.62@8500** — a +0.03 wobble off the
9.59@8000 run-best (no F anchor at 8500; every prior wobble this run
resolved downward — record-only), step 8,500, ≈1,500 steps ≈70 min
to the ~04:4xZ endpoint and the Δ_capacity read @10k. er_60k no new
rung (8.30@7000 latest), step ~7,380 — the @7500 eval is imminent,
next tick's fact.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~7,380, probe … 8.21@6500 → 8.30@7000 (matched deltas −0.57, +0.12,
−0.38, −0.47, +0.44, −0.73, −0.48), 27.4 st/min window, util 62–88%,
vram ~71.7 ×4 vs 77 bar, ~19.0/155 GPU-h; endpoint ~08-11 ~12:00Z.
`fontaine-tiny10k` LIVE local — step 8,500, probe 9.59@8000 →
**9.62@8500** (rungs @9000/@9500 remain, then the @10000 primary
read vs F 9.4157), 21.9 f/min, 7.4/15 GPU-h; endpoint ~04:4xZ
(≈1,500 steps left).

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 0 (both green, no gate crossings; the
tiny10k @8500 rung above is the fact). Queue validate OK: depth 0
pickable WITH stated depth_reason (lit pause; 8 open = 2 live + 6
owner-gated/blocked). run_work_next left unarmed — no CPU items
open; the ~04:4xZ tick chain owns tiny10k post-processing (panel_v2
→ Δ_capacity read). Body + footer rolled per last-2 (03:02 block +
03:12 note → 08-10 archive).

**Next**: tiny10k endpoint ~04:4xZ → chained panel_v2 → Δ_capacity
read @10k (vs banked F@10k 9.4157) — with tiny −0.34 under F at the
matched 7500 rung, |Δ|≤0.3 "prior confirmed" vs "tiny wins" is a
live question. er_60k @7500 rung next tick; rungs record-only to
endpoint ~08-11 ~12:00Z → chained panel_v2 k4l2 + full ER-init
convergence chart. No lit refills until the owner re-enables.*

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


Session 2026-08-10 03:45–04:0xZ (tick, babysit; 0 new GPU-h —
er_60k rides ~20.7/155, tiny10k ~7.8/15): spike-and-recover tick —
er_60k probe 15.75@7500 (2× excursion; flow loss flat, train_mae in
lockstep → decode-probe transient, no 40k precedent, kill line >25
×3 never approached) resolved at @8000 into 7.65 NEW RUN-BEST, Δ
−0.98 vs 40k 8.6371 matched, largest negative of the run; held the
session through both boundaries to see it. tiny10k 9.37@9000
run-best, already 0.05 under banked F@10k 9.4157; @9500 then the
@10000 primary read, endpoint ~04:4xZ. No steering. Queue depth 0
pickable with stated reason (lit pause). run_work_next unarmed — the
~04:4xZ tick chain owns tiny10k post-processing.

Session 2026-08-10 04:03–04:2xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~20.9/155, tiny10k ~8.0/15 incl. ~0.25 lost to the OOM
window): incident tick — tiny10k HOST-RAM OOM-killed at step ~9,060
(04:00:55Z, host RAM not GPU; probe 9.37@9000 run-best at death);
resumed 04:06Z from step_008750 as fontaine-tiny10k-r8750 (full
--resume, fresh seed 1 per policy, eval-seed 0, workers 10→6), ~310
steps lost, endpoint slips to ~05:1xZ + chained panel_v2 @10k.
Incident posted in-channel. er_60k untouched, 7.65@8000 run-best
riding. No steering. Queue depth 0 pickable with stated reason (lit
pause). run_work_next unarmed — the ~05:1x–05:3xZ tick chain owns
tiny10k post-processing.
