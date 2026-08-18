# Now









*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-18 01:23–02:0xZ (real `date -u` at write: 02:00) —
work session (chained): **per-dataset-flow-norm pre-reg DRAFT cut —
arm decided (mixed-v2), launcher staged + full-parse green,
sim-serving worn-row instrument landed with oracles; GO ask
in-channel.***

**Status**: no live runs — H100 idle (0% util; owner policy-server
not up at check). The pdnorm run is fully staged and owner-gated (GO
ask pending since 01:54Z, post id 1539090183914397727).

**Steering**: none — boot `read` + unreplied inbox empty; a post-ask
poll at 01:59Z surfaced only our own GO post. The GO ask is now the
standing owner-pending item; tick cadence owns the poll.

**Done** (commit `ba89c60`): (1) **Pre-reg DRAFT**
[posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md] (dated +
SUMMARY'd at the GO posting, disc convention). Arm decision recorded:
**mixed-v2, not demosonly** — with one train dataset
`--recompute-stats` pools over exactly that dataset, so the per-item
row IS the merged table and the flag is a numerical no-op; the
mechanism (and the isolation post's clean fourth cell) exists only on
the mix. ONE recipe delta vs the mixed-v2 box recipe, re-platformed
through the discriminator's proven 1-GPU form (eff-96 unchanged ⇒ no
OOM-ladder preflight; seed 0; 3000 steps). Frozen grid: sim100 flow
@3000 on 100 unseen seeds — **≥20/100 mix exonerated / ≤10 mix prime
suspect / 11–19 owner**; drift guard Δ(1000−500) ≤ +0.30 (disc
instrument, same stack); k4l2 panel paired vs disc-1000 (+0.05 CI
guard; wrist_flex/wrist_roll the predicted movers); GPU-h gate 21.
(2) **Launcher staged**
`launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh`, full-parse
green vs the merged CLI (family-inferred molmoact2_joint,
per_dataset_flow_norm=True). (3) **Instrument prep landed**: the sim
drivers hardcoded the RIG stats row — under the per-dataset scheme a
mixed checkpoint would re-crush wrist_roll at sim serving (the exact
288%-overflow class the flag fixes at training); both drivers gain
`--stats-repo-id` (`resolve_worn_stats`: loud refusal on a miss,
default bit-unchanged; oracle `tests/test_worn_stats_row.py` ×4).
check.py 996 green. (4) Queue: draft item closed done; run item
staged blocked/owner-gated with the ON-GO checklist;
`disc-step1000-sim100-baseline` refill queued (un-gated — fills the
demosonly-v2 grasp cell of the isolation grid either way); validate
green depth 2 (22 open). (5) GO ask posted in-channel (doubles as the
result post).

**Next**: `queue_cli.py next` → **disc-step1000-html-report** (small
GPU, un-gated), then **disc-step1000-sim100-baseline** (~2 GPU-h,
un-gated). The pdnorm RUN pends the owner GO. `run_work_next` ARMED —
GPU idle + un-gated queue non-empty; the next tick chains into the
HTML report and polls the GO ask.*

*Previous update 2026-08-18 01:20–01:2xZ (real `date -u` at write: 01:22) —
tick: **quiet tick with one new signal — owner 👍 on the 21:33
Amendment-1 post, first surfaced this tick; `run_work_next` stays
ARMED, work session chains into the flow-norm pre-reg draft.***

**Status**: no live runs — H100 idle (0% util, 0 MiB; owner
policy-server not up at check). Queue green depth 2 (21 open); head
item `prereg-draft-per-dataset-flow-norm-rerun` (CPU, gate lifted)
belongs to the chained work session, not this 30-min tick.

**Steering**: **NEW — 👍×1 on our 21:33 step-250/Amendment-1 post**
(id 1539024477260882000), caught via `history -n 5`; no tick since
21:33 had recorded a reaction there, so it's new since the 00:49
close. Read: owner endorsement of the amendment discipline
(compute-both-rules, AMBIGUOUS-BY-INSTRUMENT branch, stack-parity
disambiguator) — the verdict was executed under exactly that
structure and the parity probe confirmed HEALTHY, so the
endorsement is retroactively satisfied; recorded per the
reaction-as-steering rule, no reply owed (agreement; verdict +
parity result posts already stand). Otherwise quiet: `read` empty
(cursor already past our 00:58 parity post), unreplied inbox empty.

**Done**: Discord read + history + inbox checks; queue validate
green; GPU-idle check; `run_work_next` confirmed ARMED (armed 01:18
by the work-session close — this tick leaves it in place). No
in-channel post (nothing new to report; the 00:58 parity post is
current).

**Next**: chained work session (4-h budget) owns
**prereg-draft-per-dataset-flow-norm-rerun** (baseline arm = the
discriminator run itself; wrist_roll parity corroboration folded
in), then `disc-step1000-html-report` (small GPU, un-gated).
Owner-pending list unchanged.*

*Previous update 2026-08-18 00:49–01:2xZ (real `date -u` at write: 01:18) —
work session (chained): **discriminator post-processing CLOSED —
stack-parity probe CONFIRMS the HEALTHY verdict on the pre-merge
instrument; checkpoints banked; verdict report page live.***

**Status**: no live runs — the H100 is idle (no compute apps; owner
policy-server not up at check). Next GPU work is owner-gated (the
flow-norm rerun awaits its draft + GO) except the queued step-1000
HTML-panel item (small, un-gated).

**Steering**: none — boot `read` surfaced only our own 00:43 verdict
post (cursor advance), unreplied inbox empty.

**Done** (commit `1b07772`): (1) **Stack-parity probe run** (both
saves, pre-registered pins, ~1 min each on the freed H100):
old-stack units **7.3137@500 → 5.7626@1000, Δ(1000−500) = −1.551**
vs healthy ≤ +0.30 / drift_min +1.0158 / the drifting comparator's
actual **+2.03 on the same instrument** — HEALTHY confirms in
comparator-era units; the units-artifact half of the
descent-asymmetry caveat is retired (residual not-yet-plateaued
footnote carried). Refinement: same-checkpoint cross-stack ratios
×1.034 @500 / ×1.024 @1000 — the family-norm merge moved the probe
~2–3%; Amendment 1's s=3.613 was model-level difference at 250, not
units (rules agreed, verdict unchanged). wrist_roll is the worst
motor under the old table (16.87/12.31 vs state-copy 3.99) —
corroborates the 288% occupancy overflow, feeds the flow-norm
draft. (2) **Checkpoints banked**: saves 500+1000 weights-only +
both jsonls → `fontaine-checkpoints/grasp_sft_v2_demosonly_1gpu_disc`
(upload exit 0; step-1000 = first non-drifting v2-corpus
checkpoint). (3) **Verdict report page**
[posts/2026-08-18-sft-drift-discriminator-verdict.md](posts/2026-08-18-sft-drift-discriminator-verdict.md)
with the new parity chart (`stack_parity_chart.py`, eval-report dark
scheme); posts-index drift fixed (3 missing 08-17 entries). (4)
Queue: run item + upload item closed done with full verdict
annotations; flow-norm draft **gate lifted**; `disc-step1000-html-report`
queued (standing-rule refill); validate green depth 2. (5) Ledger
row: final run accrual recorded in the footer. In-channel post id
1539076047948087396.

**Next**: `queue_cli.py next` → **prereg-draft-per-dataset-flow-norm-rerun**
(CPU, gate lifted — baseline arm = the discriminator run itself);
then `disc-step1000-html-report` (small GPU, un-gated).
`run_work_next` ARMED — CPU queue non-empty and the GPU is idle;
the next tick chains straight into the draft. Owner-pending list
unchanged.*

## Utilization footer

Session 2026-08-18 01:23–02:0xZ (work, exploit; 0 GPU-h — CPU-side
draft + instrument work, H100 left idle for the gated run): **pdnorm
pre-reg draft cut (mixed-v2 arm, one-flag delta, frozen
sim100/drift/panel grid, gate 21); launcher staged full-parse green;
sim worn-row instrument landed with oracles (check.py 996); run +
baseline queue items staged; GO ask in-channel 01:54Z** —
`run_work_next` ARMED: the next tick owns the disc HTML report + the
GO poll.

Session 2026-08-18 01:20–01:2xZ (tick; 0 GPU-h — H100 idle, no live
runs): **quiet tick, one new signal — owner 👍 on the 21:33
Amendment-1 post caught via the history check (endorsement of the
amendment discipline; verdict + parity confirmation already
executed under it, no reply owed); read + inbox empty, queue green
depth 2** — `run_work_next` stays ARMED: the chained work session
owns the flow-norm pre-reg draft, then the step-1000 HTML panel.

Session 2026-08-18 00:49–01:2xZ (work, exploit; +~0.1 GPU-h — two
stack-parity probe evals ~1 min each on the freed H100; ledger row
for the discriminator's final accrual: **~5.8 GPU-h total, ~1.0
in-window before the 08-17 19:45Z cut, ~4.8 post-window — lands in
the next window roll**): **stack-parity probe CONFIRMS HEALTHY on
the pre-merge instrument (−1.551 vs the comparator's +2.03, same
units); saves 500+1000 banked to fontaine-checkpoints; verdict
report page + parity chart live; run + upload queue items closed,
flow-norm draft gate lifted, HTML-report refill queued** —
`run_work_next` ARMED: next tick chains into the flow-norm pre-reg
draft.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-10
00:00Z → 2026-08-17 19:45Z; rebased 08-17 from per-run prune records
+ archive session notes — receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~80.0 / ~80.2**
(incl. the discriminator at ~1.0 in-window; run COMPLETE 08-18
00:42Z at ~5.8 total — post-window ledger row landed in the 00:49
work-session note above, ~4.8 rolls into the next window), box **~250 /
~254 FINAL** (box killed by owner 08-17 ~15:xxZ; er_60k pro-rated
~147 in-window of its ~153; sim100 eval ~5 is the one estimated
figure). Older dated snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
