# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 04:35–05:0xZ (real `date -u` at write: 04:52) —
tick (babysit): **both rungs caught in-session (§6 hold): tiny10k
9.5045@9500** — record-only, sits between the resumed-path 9.56@9000
re-run and the pre-kill run-best 9.37@9000, band consistent; **er_60k
7.82@9000** — descent resumed off the 8.29@8500 wobble, second-best
rung of the run (behind 7.65@8000), no @7500-class recurrence
(matched Δ vs the 40k curve at 9000 computes next tick — the curve
is banked on the box, not locally). tiny10k step ~9,640 @~22.7
st/min steady (s_per_step 2.64–2.65 — recovery holds), **endpoint
~05:1xZ IMMINENT** → the next tick owns step-10000 + chained
panel_v2 + the Δ_capacity read vs banked F@10k 9.4157. Host RAM
94/221, 126 available — stable vs 93 last tick, growth flat.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~9,400, probe … 7.65@8000 → 8.29@8500 → **7.82@9000**, 26.7 st/min,
util 57–100% ×4, vram ~71.7 ×4 vs 77 bar, projection 24.1/155
GPU-h; endpoint ~08-11 ~12:00Z. `fontaine-tiny10k-r8750` LIVE local
— step ~9,640/10,000 at ~22.7 st/min, loss 0.13x in-band; endpoint
~05:1xZ + chained panel_v2 = the Δ_capacity primary read.

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×2 exit 0 (both live both polls). §6 hold for the
double rung window: until-loop watcher caught tiny @9500 in-session;
the er @9000 leg of the watcher silently failed (grepped the box's
log path locally — it doesn't exist on this host), caught instead by
the second babysit pass, which polls over ssh. free -g host-RAM
check (standing OOM-class rule): 94/221, stable. babysit.toml tiny
boundary updated (@9500 rung, rate steady, ENDPOINT IMMINENT block).
No post — two in-band rungs are record-only; the Δ_capacity endpoint
post (next tick) carries the morning's story. Queue validate OK:
depth 0 pickable WITH stated depth_reason (lit pause). run_work_next
left unarmed — the ~05:1x–05:3xZ tick chain owns tiny10k
post-processing (panel_v2 → Δ_capacity read). Body + footer rolled
per last-2 (04:25 block kept, 04:13 block + note → 08-10 archive).*

**Next**: tiny10k endpoint ~05:1xZ → chained panel_v2 → Δ_capacity
read @10k (vs banked F@10k 9.4157) — pre-kill was 0.05 UNDER at
@9000, resumed path 0.15 above at the re-run, 9.50@9500 in between:
the lean is genuinely open, |Δ|≤0.3 "prior confirmed" vs "tiny
wins", the @10k paired CI95 decides. er_60k: compute the @9000
matched Δ next tick (needs the box-side 40k curve), rungs
record-only to endpoint ~08-11 ~12:00Z; @7500-class transient
recurrence upgrades to a posted fact. No lit refills until the
owner re-enables.*

*Updated 2026-08-10 04:25–04:3xZ (real `date -u` at write: 04:27) —
tick (babysit): **tiny10k rate RECOVERED — ~22 st/min steady**, back
in the pre-kill 21.7–23.6 band (log `s_per_step` 2.67–2.70; babysit
window 20.9 st/min over 8840→9080). Last tick's 13.3 st/min was a
post-resume transient (worker/cache warm-up), NOT a steady
input-bound state — the workers-6 config is fine, the ride-don't-
restart call cost nothing. **Endpoint moves back to ~05:1xZ**
(projection back to ~8.2/15 GPU-h). Host RAM 93/221 used, 128
available — slight growth vs 84 last tick; record-only watch, ~40
min of run left, no OOM risk at this margin. Loss 0.13x in-band,
vram 36.6 alloc peak unchanged.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~8,720, probe … 7.65@8000 → 8.29@8500 (latest; @9000 eval ~10 min
out — next tick's fact), 27.9 st/min, util 65–100% ×4, vram ~71.7
×4 vs 77 bar, projection 22.4/155 GPU-h; endpoint ~08-11 ~12:00Z.
`fontaine-tiny10k-r8750` LIVE local — step ~9,120/10,000 at ~22
st/min, rung @9500 next (~04:4xZ), then the @10000 primary read vs
banked F@10k 9.4157; endpoint ~05:1xZ + chained panel eval.

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 0 (both live; tiny window valid again
post-reset — 20.9 st/min over 11.5 min). Rate cross-checked by hand:
75 s grep window misfired (matched the resume banner), so read the
log's own `s_per_step` field directly — 2.67–2.70 s/step steady =
~22 st/min, transient-recovery confirmed. free -g host-RAM check
(standing OOM-class rule): 93/221, 128 available, watch noted.
babysit.toml boundary updated (RATE RECOVERED block, endpoint back
to ~05:1xZ). No post — a rate recovery that un-slips an endpoint is
record-only good news; the Δ_capacity endpoint post carries it.
Queue validate OK: depth 0 pickable WITH stated depth_reason (lit
pause). run_work_next left unarmed — the ~05:1x–05:3xZ tick chain
owns tiny10k post-processing (panel_v2 → Δ_capacity read). Body +
footer rolled per last-2 (04:13 + 04:03 blocks, 04:03 note → 08-10
archive).

**Next**: tiny10k @9500 rung ~04:4xZ (record-only, wobble band
~0.25), endpoint ~05:1xZ → chained panel_v2 → Δ_capacity read @10k
(vs banked F@10k 9.4157) — pre-kill was 0.05 UNDER at @9000, the
resumed path re-ran @9000 at 9.56 (0.15 above), the tiny-wins lean
is genuinely open; |Δ|≤0.3 "prior confirmed" vs "tiny wins", the
@10k paired CI95 decides. er_60k @9000 rung imminent, record-only
to endpoint ~08-11 ~12:00Z; @7500-class transient recurrence
upgrades to a posted fact. No lit refills until the owner
re-enables.*

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


Session 2026-08-10 04:25–04:3xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~22.4/155 projection, tiny10k ~8.2/15): recovery tick —
tiny10k rate back to ~22 st/min (log s_per_step 2.67–2.70; babysit
window 20.9), last tick's 13.3 was a post-resume transient, not
input starvation; endpoint back ~05:4x → ~05:1xZ, ride call
vindicated at zero cost. Host RAM 93/221 with 128 available (mild
growth vs 84, ~40 min run left — record-only watch). er_60k no new
rung (8.29@8500 latest, @9000 ~10 min out), step ~8,720 @27.9
st/min. No steering. Queue depth 0 pickable with stated reason (lit
pause). run_work_next unarmed — the ~05:1x–05:3xZ tick chain owns
tiny10k post-processing.

Session 2026-08-10 04:35–05:0xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides 24.1/155 projection, tiny10k ~8.4/15): double-rung tick
— §6 hold caught tiny10k 9.5045@9500 (record-only, between the
9.56@9000 re-run and pre-kill 9.37 run-best) and er_60k 7.82@9000
(descent resumed off 8.29@8500, second-best of the run; matched Δ
vs 40k computes next tick — curve banked box-side only). tiny ~22.7
st/min steady, step ~9,640, endpoint ~05:1xZ imminent; host RAM
94/221 stable. Watcher lesson recorded: the er log path is box-only,
grep it via babysit/ssh, never locally. No steering. Queue depth 0
pickable with stated reason (lit pause). run_work_next unarmed — the
~05:1x–05:3xZ tick chain owns tiny10k endpoint + panel_v2 +
Δ_capacity read.
