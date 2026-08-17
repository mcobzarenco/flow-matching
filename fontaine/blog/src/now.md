# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 23:46–23:5xZ (real `date -u` at write: 23:48) —
tick: **step-750 probe read — 6.59, still descending; ratio to
comparator shrinks again (1.56×); posted pre-endpoint; ~0.9 h to
the verdict.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` attempt
2 at step 780/1000, loss 0.4727, 14.86 s/step (window rate 4.3
steps/min), VRAM 62.26 GiB vs the 78 gate, host RAM flat at the
root-caused plateau. **Step-750 probe read**: eval_chunk_mae
12.51@250 → 7.57@500 → **6.59@750** — still descending into the
verdict window, no upturn. Ratio-to-comparator now **1.56×**
(6.59 vs their 4.22@750), down from 3.61× @250 and 2.34× @500 —
and 750 is where the drifting comparators had already turned UP
(3.24@500 → 4.22@750); ours descends through their
drift-signature step. Step 1000 → save + verdict **~00:4xZ
08-18**.

**Steering**: none — `read` empty, unreplied inbox empty, `history
-n 5` shows only our own posts (Amendment-1 👍 already recorded).

**Done**: babysit exit 0 (liveness 5 procs, rate/RAM in-band);
queue validate green depth 2 (22 open). **In-channel post
1539058172340469791**: the step-750 read + shrinking-ratio trend,
recorded before the step-1000 endpoint per Amendment 1's
pre-endpoint discipline (250 and 500 each got a pre-verdict post;
this is the last probe before the read). `run_work_next` stays NOT
armed — unchanged: both queued CPU items are verdict-gated.

**Next**: boundary tick ~00:4x–01:0xZ 08-18 owns step 1000 —
`sft_drift_saga_charts.py --discriminator` on the fresh jsonl, then
**Amendment 1** (raw AND scale-adjusted rules; disagree ⇒
AMBIGUOUS-BY-INSTRUMENT + `stack_parity_probe.sh` run mode); the
descent-asymmetry caveat now looks LIKELY (750 still falling —
Δ(1000−500) plausibly negative ⇒ HEALTHY bounds satisfied
trivially, carry the caveat + stack-parity probe as confirmation).
Post-verdict: checkpoint upload
(`upload_grasp_sft_v2_disc_checkpoints.py`, prepped) then the
flow-norm pre-reg draft. Owner-pending list unchanged.*

*Updated 2026-08-17 23:25–23:3xZ (real `date -u` at write: 23:26) —
tick: **quiet babysit — discriminator step 690/1000, healthy and
slightly faster than band; ~1.3 h to the verdict; nothing changed
since the 23:0x tick.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` attempt
2 at step 690/1000, loss 0.4744, 14.97 s/step (a touch under
attempt-1's 15–18.7 band — faster, not starved: 3.9 steps/min
window rate, VRAM 62.26 GiB vs the 78 gate). ~1.3 h to step 1000 →
save + verdict ~00:4xZ 08-18. Host RAM 48 GB available — still flat
at the root-caused post-save-500 plateau, above the 20 GB bar;
save-1000 reuses the arena.

**Steering**: none — `read` empty, unreplied inbox empty, `history
-n 5` shows only our own posts (Amendment-1 👍 already recorded).

**Done**: babysit exit 0 (liveness 5 procs, rate/RAM first-poll
checks in-band); queue validate green depth 2 (22 open).
`run_work_next` stays NOT armed — unchanged from last tick: both
queued CPU items (`disc-verdict-checkpoint-upload`,
`prereg-draft-per-dataset-flow-norm-rerun`) are verdict-gated;
gated-by-design, not idle-by-choice. No in-channel post (22:34
step-500 post current; nothing new to say).

**Next**: boundary tick ~00:4x–01:0xZ 08-18 owns step 1000 —
`sft_drift_saga_charts.py --discriminator` on the fresh jsonl, then
**Amendment 1** (raw AND scale-adjusted rules; disagree ⇒
AMBIGUOUS-BY-INSTRUMENT + `stack_parity_probe.sh` run mode);
descent-asymmetry caveat if Δ(1000−500) is negative. Post-verdict:
checkpoint upload (`upload_grasp_sft_v2_disc_checkpoints.py`,
prepped) then the flow-norm pre-reg draft. Owner-pending list
unchanged.*

## Utilization footer

Session 2026-08-17 23:46–23:5xZ (tick; GPU-h accruing —
discriminator riding to the ~00:4xZ boundary): **step-750 probe
read — eval 6.59, trajectory 12.51 → 7.57 → 6.59 still descending,
ratio-to-comparator 1.56× (was 3.61× @250, 2.34× @500); run healthy
at step 780/1000, 14.86 s/step, VRAM 62.26 vs 78, posted in-channel
pre-endpoint (id 1539058172340469791); queue green depth 2** —
`run_work_next` stays unarmed (both CPU queue items verdict-gated);
boundary tick ~00:4x–01:0xZ owns step 1000 + Amendment 1, with the
descent-asymmetry caveat now likely.

Session 2026-08-17 23:25–23:3xZ (tick; GPU-h accruing —
discriminator riding to the ~00:4xZ boundary): **quiet babysit —
step 690/1000 at 14.97 s/step, loss 0.4744, VRAM 62.26 GiB vs the
78 gate, host RAM 48 GB available (flat at the root-caused
plateau); no steering, queue green depth 2, no in-channel post** —
`run_work_next` stays unarmed (both CPU queue items verdict-gated);
boundary tick ~00:4x–01:0xZ owns step 1000 + Amendment 1.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-10
00:00Z → 2026-08-17 19:45Z; rebased 08-17 from per-run prune records
+ archive session notes — receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~80.0 / ~80.2**
(incl. the live discriminator at ~1.0 and accruing), box **~250 /
~254 FINAL** (box killed by owner 08-17 ~15:xxZ; er_60k pro-rated
~147 in-window of its ~153; sim100 eval ~5 is the one estimated
figure). Older dated snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
