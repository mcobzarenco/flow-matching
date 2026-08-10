# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 09:14–09:3xZ (real `date -u` at write: 09:21) —
tick (babysit): **owner steering 08:29Z executed — er_60k @15000
checkpoint → hub → local panel_v2 → HTML report.** Weights-only
upload (backbone/expert/prompt safetensors + config, ~10.5G)
launched 09:18Z on the box as transient unit `hf-up-er15k` →
`fontaine-checkpoints/fontaine_molmo2_er_60k_ddp4/step_015000`;
**run_work_next ARMED** — the chained work session watches the
upload, downloads locally, runs panel_v2 k4l2 `--report` on
step_015000 (local GPU free since 05:45Z), posts the HTML link
in-channel. Ack + plan + ETA (~2–3 h) posted 09:17Z. **CREDITS
OUTAGE**: ticks 08:28 / 08:42 / 08:52 / 09:03 all died on an
out-of-credits 429 (the 08:42 harness alert); 09:14Z is the first
surviving session — the run was never at risk, owner told
in-channel. Two rungs banked from the outage window: 7.3267@15500
(Δ +0.45 vs 6.8736) and 7.0094@16000 (Δ +0.37 vs 6.6439) — fifth
and sixth positive legs in a row, running mean ≈ +0.21, still
inside the ~±0.8 wobble; endpoint panel decides.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
16,380 at poll, probe … 6.8543@13000 → 7.15 → 7.37 → 7.42 →
6.9230@15000 → 7.3267@15500 → **7.0094@16000**, 27.1 f/min window,
vram ~71.7 ×4 vs 77 bar, ~41.7/155 GPU-h; endpoint ~08-11 ~12:00Z.
Box also carries upload unit `hf-up-er15k` (CPU/network only).
Local GPU free — reserved for the owner-requested step_015000
panel_v2 (chained session).

**Steering**: **08:29Z owner**: "copy the 15k checkpoint … to the
hub, then download it on your local machine, run the eval panel
and post html report link here" — acknowledged 09:17Z (delay =
credits outage, explained in the ack), execution in flight per
Status. Lit pause unchanged.

**Done**: babysit exit 0 (liveness 8 procs, util 58–97% at snapshot,
window 27.1 f/min healthy). Diagnosed the exit-1 harness alert:
four ticks 08:28–09:03 killed by out-of-credits 429 (log tails all
show `api_error_status: 429`), not auth — first surviving session
09:14Z. @15500 + @16000 matched-Δ legs banked record-only
(baseline values pulled from the verified ar_40k box log). Hub
upload unit launched + verified active ~6 s. babysit.toml
rung-state block refreshed (new legs + upload-in-flight + outage
note). Queue validate OK depth 0 pickable (lit pause + owner-gated
tail), 7 open. **run_work_next ARMED** (owner-requested panel
pipeline is the work item).

**Next** (pipeline ran ahead of plan in-tick: upload DONE 09:18:58Z
in 43 s [hub commit cb05e71, 4 files verified], local download DONE
09:23:27Z in 16 s [9.1G], **panel_v2 LIVE** on the local H100 as
unit `eval-er15k-panel` since 09:3xZ — verbatim er_60k endpoint
eval at 1 rank, ~1–1.5 h wall; progress posted 09:3xZ): chained
work session: (1) first-poll util check on the eval, (2) watch to
rc=0, (3) frozen reads vs banked 40k endpoint (6.0079) +
60k-continuation (5.8602) npz, (4) post HTML report link
in-channel + reports page per standing rule, (5) clean up the
ad-hoc helpers (~/hf_up_er15k.py on box, ~/hf_dl_er15k.py,
~/eval_er15k_panel.sh). er_60k rides to endpoint
~08-11 ~12:00Z → chained panel_v2 → paired CI95 vs banked 40k
(6.0079) + 60k-continuation (5.8602). Rungs record-only; kill
lines unchanged. Next save boundary @20000 ~11:3xZ. Watch credits:
if 429s recur, sessions die again — resetsAt stamp says Aug 15
22:00Z, so current headroom is whatever the owner topped up. No
lit refills until re-enabled.*

*Updated 2026-08-10 08:08–08:3xZ (real `date -u` at write: 08:28) —
tick (babysit): **SAVE BOUNDARY @15000 caught — er_60k 6.9230,
second sub-7** (just off the 6.8543@13000 run-best); nine straight
rungs (7.54 / 7.59 / 7.37 / 7.40 / 6.85 / 7.15 / 7.37 / 7.42 /
6.92) under the pre-plateau 7.65@8000 mark. Async save green:
`captured in 21.7s; gather+write continue in background`, util back
at 100% after the pause. Matched Δ vs 40k (shared seed, box-side
log) extends the record-only table: **@15000 +0.19 (6.9230 vs
6.7311)** — full table @9000→@15000: −0.44 / +0.53 / +0.77 / +0.80
/ −0.43 / +0.39 / −0.19 / −0.50 / −0.24 / +0.17 / +0.47 / +0.73 /
+0.19. Fourth positive leg in a row but back off the +0.73 upper
edge; running mean ≈ +0.17 on the ~±0.8 wobble — endpoint panel
(~08-11 ~12:00Z) decides. **Morning results post landed
in-channel 08:2xZ** (the pre-declared post moment — sub-7.65 band
held). Rung caught with a ~13-min §6 hold (ssh until-loop keyed on
the eval_chunk_mae jsonl line — fired first try; the corrected
~08:2xZ ETA was right).*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
15,000 saved + posted, probe … 7.65@8000 → 7.95@10500 → 7.54@11000
→ 7.59@11500 → 7.37@12000 → 7.40@12500 → 6.8543@13000 →
7.1503@13500 → 7.3734@14000 → 7.4229@14500 → **6.9230@15000**,
27.0 f/min window, vram ~71.7 ×4 vs 77 bar, ~38/155 GPU-h;
endpoint ~08-11 ~12:00Z. Local GPU free (next local launch needs a
fresh pre-reg).

**Steering**: none — `read` empty, history ×5 all our own posts,
no new reactions (lit-pause exchange still the last owner
message).

**Done**: babysit exit 0 (liveness 8 procs, 4× GPU engaged, util
94–100%, window 27.0 f/min healthy). §6 hold ~13 min for the
@15000 boundary; baseline identity re-verified (ar_40k
@13000–@14500 all match banked legs) and the @15000 leg banked;
save-boundary fact captured (21.7 s async capture, util 100%
after). **Posted** the morning results post (ladder + full Δ table
+ save fact + health + endpoint plan). babysit.toml rung-state
block refreshed (@15000 boundary + @20000 ETA ~11:3xZ). Queue
validate OK: depth 0 pickable with stated reason (lit pause +
owner-gated tail), 7 open. run_work_next NOT armed — CPU-side
queue empty, box busy, local idle-by-design (charter §5 exit).

**Next**: er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-continuation
(5.8602) panels. Rungs record-only; @7500-class transient
recurrence upgrades to a posted fact; kill lines unchanged (NaN,
probe-vs-@2500 by 10k — PASSED, probe>25 ×3). Next save boundary
@20000 ~11:3xZ (rungs every ~18.5 min stay in-band unless the Δ
table breaks ±0.8). Next local launch owner-gated:
named-not-preregistered candidates T2 depth rung + tiny decode
microbench (#16); fjoint finalize waits on owner go (~08-12). No
lit refills until re-enabled.*

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
chained panel_v2 eval COMPLETE 08-10 05:45Z (+~0.6 GPU-h, **~9.3/15
total, rung closed**)). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).





Session 2026-08-10 08:08–08:3xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~38/155, sole live run): single-run tick with a ~13-min
§6 hold — **SAVE BOUNDARY @15000 caught: er_60k 6.9230, second
sub-7** (just off the 6.8543@13000 run-best); nine straight rungs
(7.54/7.59/7.37/7.40/6.85/7.15/7.37/7.42/6.92) under the
pre-plateau 7.65@8000 mark. Async save green (captured 21.7 s,
util 100% after the pause). Matched-Δ table vs 40k extended
record-only (@15000 +0.19, 6.9230 vs 6.7311 — fourth positive leg
in a row but back off the +0.73 upper edge; running mean ≈ +0.17
on the ~±0.8 wobble; endpoint panel decides). Baseline log
identity re-verified (ar_40k @13000–@14500 match all banked legs).
Watcher: single until-loop keyed on the eval_chunk_mae jsonl line
fired first try — the corrected ~08:2xZ ETA was right. **Morning
results post landed in-channel 08:2xZ** (pre-declared post moment,
sub-7.65 band held). Next save boundary @20000 ~11:3xZ. No
steering (read empty, history ×5 unchanged). Queue depth 0
pickable with stated reason (lit pause + owner-gated tail);
run_work_next NOT armed — CPU queue empty, local GPU
idle-by-design, plain §5 exit.

Session 2026-08-10 09:14–09:3xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~41.7/155, sole live run): owner-steering tick.
**08:29Z owner request executed**: er_60k step_015000 weights-only
→ fontaine-checkpoints via box transient unit `hf-up-er15k` —
upload DONE in 43 s, local download DONE in 16 s, panel_v2 eval
LIVE on the local H100 (unit `eval-er15k-panel`); run_work_next
ARMED — chained work session watches to rc=0 and posts the HTML
link. Ack 09:17Z + progress post 09:3xZ. **Credits
outage diagnosed**: ticks 08:28/08:42/08:52/09:03 all died on
out-of-credits 429 (the 08:42 harness alert); 09:14Z first
surviving session, run never at risk, owner told. Rungs banked
from the gap: 7.3267@15500 (Δ +0.45) / 7.0094@16000 (Δ +0.37) —
fifth/sixth positive legs, running mean ≈ +0.21, in-band,
record-only. babysit exit 0 (27.1 f/min, vram 71.7 ×4). Queue
depth 0 pickable with stated reason; next save boundary @20000
~11:3xZ.
