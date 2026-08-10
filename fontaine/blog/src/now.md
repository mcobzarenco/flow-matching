# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 04:56–05:1xZ (real `date -u` at write: 05:09) —
tick (babysit): **tiny10k COMPLETE — step 10,000 hit 05:06Z, caught
in-session (§6 hold). Final probe 9.3469@10000 vs banked F@10k
9.4157 → probe-level Δ_capacity −0.069**, deep inside the |Δ|≤0.3
"prior confirmed" band: at the fully matched read, width alone does
not separate tiny (h256/d12) from F. The resumed path converged
back onto the pre-kill curve (9.37@9000 pre-kill → 9.56/9.50 wobble
→ 9.35@10000) — the OOM cost ~310 replayed steps and ~30 min,
nothing else. Checkpoint `step_010000` saved (async, 0.6 s);
**chained panel_v2 @10000 LIVE** in-unit, pre-reg args verbatim
(k4l2 plan sha-verified, heun30/draws1/stable, npz + HTML report) —
the paired per-frame CI95 vs F's banked panel npz is the PRIMARY
read. F's npz verified BOX-SIDE ONLY
(`eval__fontaine_molmo2_flow_frozen_10k_ddp4__step_010000__panel_v2…​.npz`)
— scp before pairing. Endpoint posted in-channel 05:07Z;
**run_work_next ARMED** — the chained work session owns the panel
readout, Δ chart, follow-up post, ledger row, and the step_010000
weights-only upload to fontaine-checkpoints.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~9,520, probe … 7.65@8000 → 8.29@8500 → 7.82@9000 → **7.92@9500**
(in-band wobble), 22.5–26.7 st/min, vram ~71.7 ×4 vs 77 bar,
projection 24.4/155 GPU-h; endpoint ~08-11 ~12:00Z.
`fontaine-tiny10k-r8750` — **train COMPLETE 05:06Z (~8.7/15
GPU-h incl. OOM replay)**; chained panel_v2 eval LIVE (CPU
dataset-load phase at write, GPU engages when sampling starts).

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 0 (both live; er @9500 rung surfaced).
free -g host-RAM check: 98/221, 122 available — mild growth, run
ended before it mattered. §6 hold: 10-min until-loop watcher caught
the endpoint in-session — wandb summary, probe line, async save,
and panel launch all verified. Endpoint post in-channel 05:07Z
(one cosmetic mangle: zsh command-substitution ate the backticked
checkpoint name — post otherwise clean, no correction sent; lesson:
single-quote Discord post strings). babysit.toml tiny entry flipped
to panel phase (eval log, gpu_mem_min 0 during CPU load, boundary +
box-side F-npz path pinned, PRUNE-at-completion note).
run_work_next ARMED with queue depth 0 stated-reason OK (lit
pause) — the work chain is post-processing, not queue-driven. Body
+ footer rolled per last-2 (04:35 block kept, 04:25 block + note →
08-10 archive).

**Next**: chained work session (immediately after this tick):
babysit the panel eval to completion, scp F's box-side npz, compute
the paired per-frame CI95 Δ_capacity (tiny minus F), build the Δ
chart (dark-mode, eval-report scheme), follow-up post + blog +
ledger row, upload step_010000 weights-only to fontaine-checkpoints,
then prune the tiny babysit entry. er_60k: rungs record-only to
endpoint ~08-11 ~12:00Z; @9000 matched Δ vs 40k still owed (needs
the box-side 40k curve); @7500-class transient recurrence upgrades
to a posted fact. No lit refills until the owner re-enables.*

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
call — no endpoint, no chained evals)**; local tiny10k 08-09 20:1xZ
→ 08-10 05:06Z train COMPLETE **~8.7/15 GPU-h incl. OOM replay** +
chained panel_v2 eval live). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).


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

Session 2026-08-10 04:56–05:1xZ (tick, babysit; tiny10k train
COMPLETE at ~8.7/15 GPU-h incl. OOM replay; er_60k rides 24.4/155):
endpoint tick — §6 hold caught step 10,000 at 05:06Z in-session:
final probe 9.3469@10000 vs banked F@10k 9.4157 → probe-level
Δ_capacity −0.069, prior-confirmed band (|Δ|≤0.3); resumed path
converged back onto the pre-kill curve. Checkpoint step_010000
saved async; chained panel_v2 launched in-unit (pre-reg args
verbatim, sha-verified plan). F's panel npz confirmed box-side only
— path pinned in babysit.toml for the scp. Endpoint posted
in-channel 05:07Z. er_60k 7.92@9500 in-band wobble, record-only.
No steering. Queue depth 0 with stated reason (lit pause).
run_work_next ARMED — the chained work session owns panel readout →
paired CI95 → chart → post → ledger → checkpoint upload.
