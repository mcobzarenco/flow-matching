# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 11:35–12:0xZ (work session) — **onerig leg-2 CPU
tail CLOSED (panel guard PASS 28.81 vs 58.14; truth-fit 27.26 —
the 28/100 grasper and the 1/100 convict sit ~0.2 apart on the
panel: grasping lives in sim100, not panel MAE) — and the GRPO R2
parity verdict came back PASS with PERFECT parity, so the frozen
A3.4 relaunch fired mechanically 11:59:20Z.***

**Status**: `grpo-r2` live on the H100 (launched 11:59:20Z on the
registered PASS branch, unit `grpo-r2`, A3.4 frozen argv verbatim:
base step_002000_v2, 8×8 T=1.0, surface B, lr 1e-6, kl_beta 1.0,
kl_stop 0.06, seed-base 2000, wave0 knockaway re-base, mixed-abort
0.20): step 0/10, wave-0 rollouts collecting; first poll 12:02Z util
100%, 28.8 GiB VRAM, RAM 162G available. Pace anchor ~0.98
GPU-h/step → endpoint **~22:0xZ**; gates ≤15 GPU-h / ≤75 GiB; lane
~19.6/20 vs the A4 gate — zero slack. Wave-0 `mixed_groups_frac`
(in-loop abort <0.20, predicted ~0.44) reads off the first train
row at the next babysit checkpoint.

**Steering**: none — inbox empty at boot and every poll (11:35/
11:55Z); the owner 👍 on the 11:26Z policy-server reply stays the
last owner signal.

**Done**: onerig-endpoint-close FULLY CLOSED (commit 9ff74ee): panel
guard PASS (28.81 vs disc-1000 raw 58.14, Δ −29.34 CI95 [−30.03,
−29.38], n=15,056; wrist_roll −46.9 / wrist_flex −5.0 mechanism
receipts; oracle re-run green); truthfit rewear native 28.81 →
truth-fit 27.26 (seam +1.55; ladder 27.26 ≈ convicted 27.44 ≈ disc
27.40 ≈ released 27.14, all at/above the 25.15 null); ladder
restamped for the onerig cell (chart grew --label/--title, pdnorm
defaults byte-stable, oracle +1); `onerigendpoint` report preset →
flow_unseen100.html (anchors 9/11/1/44, paired + ladder + seam
embeds, 4-clip gallery); 6 artifacts + gallery on fontaine-reports
all curl-200; reports.md section. Parity item closed: verdict PASS
~11:58Z (both paths 2/20, same seeds 207/214, interacted 20/20 both;
Δ 0 / 0.0), relaunch fired + babysit registry rolled parity→loop.
Posts 1539966194436673556 (tail close) + 1539967344539996224
(PASS + launch). check.py green ×2.

**Next**: babysit owns the loop (~30-min checkpoints via ticks;
first train row = the mixed_groups_frac read). Step-10 endpoint
~22:0xZ → `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt` (refuses while grpo-r2 is
alive). `queue_cli.py next` → `prereg-draft-demos-plus-clean` (CPU,
any work window; GPU launch only after this lane's window closes).
Queue depth 1 open with stated reason — refill decision at the R2
boundary. `run_work_next` armed.*

*Updated 2026-08-20 11:08–11:2xZ (tick) — **endpoint battery
COMPLETE 11:17:49Z (leg 2 k4l2 panel json+html clean; native
bijou@3000 panel MAE 28.81 — read withheld for the truthfit rewear +
panel guard in the chained session); ckpt bank VERIFIED on HF; the
GPU window rolled straight into the R2 parity read 11:18:20Z — a
31-second handoff, no idle gap.***

**Status**: `grpo-r2-parity` live on the H100 (launched 11:18:20Z in
the freed window per queue priority; unit `grpo-r2-parity`, ~0.7
GPU-h → verdict ~12:0xZ): seeds 200–219 greedy through BOTH serving
paths — loop stack `--joint-frame rig`, then BijouPolicy
`--serve-head ar` — chaining `grpo_r2_parity_verdict.py`. Registered
A5 rule: PASS iff |ΔSuccesses| ≤ 2 AND |ΔInteractedFrac| ≤ 0.30 (the
convicted mode read 0.00 vs ~0.59). On PASS the frozen A3.4 relaunch
fires mechanically (~14.9 GPU-h; lane ~19.6/20 — zero slack); on
FAIL the lane parks, no override. Battery unit exited clean (4h30m
CPU, 54.6G mem peak); registry rolled battery→parity.

**Steering**: owner 11:20:27Z — "Where is the onerig 3k checkpoint?
How can I run the policy server with it?" Replied in-channel
11:26:34Z (local `~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_
pdnorm_onerig/step_003000`, HF bank folder, and the
`bijou.policy_server --port 8144` command incl. the bfloat16 flag +
the note that the parity read leaves headroom to serve alongside) —
acked, inbox clear; owner 👍 on the reply by 11:34Z
(acknowledgement). Signal: the owner intends to rig-serve the onerig
3k ckpt — consistent with it being the first grasping mixed
checkpoint. Held conversational ~7 min after the reply, no
follow-up.

**Done**: babysit poll exit 0 (leg 2 healthy mid-scoring at 11:08;
watched in-session to the 11:17:49Z boundary — the babysit
bare-count 99 was leg-1 residue, the journal was the real progress
read). Ckpt bank verify closed: 6 weights-only files live under
`grasp_sft_v2_joint_pdnorm_onerig_step3000` (DONE 10:52:39Z).
Parity launch + registry roll + Discord post 11:19Z. Disk 107G free
(+10G, pruner). Queue validate green depth 3 (16 open).
`run_work_next` stays armed — the chained work session takes the
leg-2 CPU tail (panel guard vs disc-1000 npz, truthfit rewear,
ladder restamp, onerig HTML report) + the demos+clean pre-reg draft,
and catches the parity verdict ~12:0xZ.

**Next**: parity verdict ~12:0xZ → on PASS `./launch_grpo_r2.sh
launch` fires mechanically (no GO ask) + babysit entry + announce;
on FAIL park the lane + postmortem post. At the R2 endpoint the
boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

*Updated 2026-08-20 07:30–11:0xZ (work session) — **ONERIG VERDICT:
sim100 28/100 — MIX-EXONERATED through the frozen grid. The
two-dataset mix (demos + v2 ×4, clean dropped) beats its demos-only
control +17 (CI95 [8, 26], McNemar p = 0.0009) and the convicted
three-way cell +27 (CI95 [19, 36], p = 1.5e-8). One recipe delta —
dropping 13.6k clean frames, 0.65% of the corpus — turned 1/100 into
28/100. Rig data at ~6% share HELPS grasping once clean is out.***

**Status**: battery leg 2 (k4l2 panel npz) live on the H100 (started
~10:51Z, ~0.5 GPU-h, unit `fontaine-onerig-endpoint-battery`);
checkpoint bank upload live (unit `fontaine-onerig-ckpt-bank`,
weights-only ~12 GiB → `fontaine-checkpoints/`
`grasp_sft_v2_joint_pdnorm_onerig_step3000`). Train COMPLETE 3000/3000
(~13.4 GPU-h, loss 0.3299, vram 62.21/71, zero crossings); final
probe 4.5266@3000 held the run low — the curve ended improving, the
opposite shape of the convicted cell. Honest cell total ~17.0 vs the
17 gate: a ~48-min idle gap (my own watch-loop cmdline matched the
run pgrep and deadlocked the battery wait — the 08-19 class incident
reproduced, note sharpened in the registry) ate the margin; no extra
legs.

**Steering**: none — inbox empty all session (reads 07:30/08:38/
09:11/09:40/10:09Z); the three recorded 👍s unchanged.

**Done**: onerig-endpoint-close primary verdict banked (battery
script staged + armed pre-endpoint 028c94c; registry rolled
train→battery 5e73979; verdict + paired reads + queue/now/blog this
commit). Posts: train-complete 08:38Z, mid-leg signal 09:11Z (7/25
early-terms), VERDICT 10:51Z (id 1539950050740801616). Paired jsons
banked (`analysis__sim100_paired_onerig3000_vs_{disc1000,pdnorm3000}
.json`). Queue: verdict noted on onerig-endpoint-close (leg-2 CPU
tail remains), refill `prereg-draft-demos-plus-clean` queued (the
poison-pinning cell) — validate green, depth 3.

**Next**: leg 2 lands ~11:2xZ → chained session takes the CPU tail
(panel guard vs disc-1000 npz, truthfit rewear, ladder restamp,
onerig HTML report, bank verify) + the demos+clean pre-reg draft;
`run_work_next` armed. Then `queue_cli.py next` →
grpo-r2-parity-read-and-relaunch owns the first free GPU window
(`./launch_grpo_r2.sh parity`, A5 gate, no GO ask); at the R2
endpoint the boundary is `./launch_grpo_r2.sh boundary
outputs/sim/grpo_r2/loop/step_0010.pt`.*

## Utilization footer

Session 2026-08-20 12:08–12:1xZ (tick; `grpo-r2` riding, ~0.15
GPU-h elapsed of ~14.9 projected / lane ~19.6/20): **babysit exit 0
— 9 min post-launch at the 12:08Z poll, step 0/10, wave-0 rollouts
mid-collection (journal live at seed 210 replans; eval/baseline seed
range — the ≥2000 pin binds train rows only, none written yet); util
100%, 27.0 GiB VRAM, unit mem 29.2G, no gate crossings; the babysit
"last sample 08-19" stamp is the stale step-0 eval row in the reused
loop/train.jsonl, meta.json fresh 11:59:20Z — first real train row
(the mixed_groups_frac abort read) due ~12:5xZ, next sessions take
it; Discord fully quiet (read + inbox empty, no new reactions);
queue validate green; run_work_next already armed 12:05 (CPU item
`prereg-draft-demos-plus-clean` pending)** — queue green depth 1
(stated reason, 14 open).

Session 2026-08-20 11:35–12:0xZ (work session; exploit; ~0.66 GPU-h
parity read closed in-window + grpo-r2 launched ~14.9 GPU-h): **onerig
leg-2 CPU tail closed (guard PASS 28.81 vs 58.14 Δ −29.34; truth-fit
27.26 ≈ convicted 27.44 — the panel doesn't separate the grasper from
the convict; ladder restamped, onerigendpoint report + 6 artifacts
curl-200; commit 9ff74ee); parity verdict PASS ~11:58Z (perfect: 2/20
both paths, same seeds, Δ 0/0.0) → frozen A3.4 relaunch fired
11:59:20Z (unit grpo-r2, endpoint ~22:0xZ, lane ~19.6/20 zero slack);
registry parity→loop; first poll util 100%; posts ×2; queue rolled
(both GPU items closed done, depth 1 + stated reason);
run_work_next armed** — queue green, 14 open.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
