# Pre-registration: SFT-drift discriminator (demosonly recipe, one GPU)

*Posted 2026-08-17 18:4xZ, immediately before launch, on the owner GO
(18:40:56Z, "You can do whatever you want" in reply to the standing ask
open since 15:14Z). Cut 18:3xZ 08-17 from the frozen box launcher
header + the postproc kit's frozen verdict bounds — thresholds were
written before this run started. Runs as
`grasp_sft_v2_demosonly_1gpu_disc` on the local H100.*

**Plain words**: five recent training runs went bad in the same odd way
— the model's action error started climbing after step 500 instead of
falling — and every bad run had one thing in common: it was spread
across 8 GPUs. Every healthy run in this family used a single GPU. This
experiment reruns one of the bad runs' exact recipe on a single GPU,
changing nothing else. If the error curve now behaves, the 8-GPU
machinery is the culprit. If it still climbs, the machinery is
innocent and the suspect list shrinks to a handful of recipe
ingredients we can then test one at a time. We wrote down the
pass/fail thresholds before starting so we can't fool ourselves when
the numbers come in.

## The question

Every drifting run in this family (run-1b, run-2, mixed v2, demosonly)
is 8×A100 distributed (torchrun + zero1 + `--chunk-grad-allreduce`);
every healthy run (44/100 joint probe, 28/100 stage-C) was single-GPU.
This run replicates the demosonly recipe on ONE GPU with the same
effective batch 96 and the same micro-batch 12 (`--batch-size 96
--backward-chunks 8` = micro 12, exactly one 8× rank's shard), same
default seed (seed policy: same seed for comparability — this is a
replication, not a variance probe), same `--image-augment 0.8`, same
`--recompute-stats`, same init. The ONLY delta vs the drifting run is
the distributed machinery.

## Command

`fontaine/scripts/launch_local_grasp_sft_v2_demosonly_1gpu_disc_h100.sh`
— the frozen box launcher
(`fontaine/scripts/box/launch_box_grasp_sft_v2_demosonly_1gpu_discriminator.sh`,
staged 08-17 pre-kill) with its command block **byte-identical**
(diff-verified) and only platform edits above it. Full-parse green
against the merged (`d3dd4d0`) CLI: family-inferred `molmoact2_joint`,
`per_dataset_flow_norm=False` (no new levers — this replicates the
drifting recipe), seed 0, joint objective with `--insulate-flow`,
`--flow-decoder-init inherit`, eval every 250 with
`--eval-dataset-breakdown`, save every 500, 1000 steps.

## Platform delta (box → local H100)

The 8×A100 box was deleted by the owner 18:09Z 08-17. The run
re-points at the local H100 — the deltas are platform-only, the recipe
carries zero:

1. **Hardware**: 1× H100 80GB (local) instead of 1× box A100 80GB.
   Same 80 GiB memory budget; the recipe class is measured to fit
   (micro-12 + activation checkpointing). Box pace estimate was
   ~25–32 s/step for the full eff-96 step (~7–9 h to step 1000); the
   H100 should match or beat that. First-poll check per standing rule:
   GPU util + s/step, fix input starvation before letting it ride.
2. **Data**: `~/datasets/fontaine/grasp_demos_v2/merged` is now a
   local snapshot of `mcobzarenco/fontaine-grasp-demos-v2` — the HF
   mirror verified ≈ the box merged copy (36.7 GB) at evacuation
   (17:20Z 08-17). Init checkpoint
   `~/checkpoints/molmoact2-so101-released` was already local.
3. **Code**: the run executes on the merged family-norm stack
   (`d3dd4d0`, main `ebaa8e0`) — not byte-the-code the 8× runs
   trained under. Covered by the merge gates: zero-numeric-change
   claim reproduced (gradflow loss oracles EXACT 1.6948 flow /
   27.8546 ar_backbone; check.py 992 green), so training math is
   unchanged.
4. **Shared host guards**: the launcher aborts if any compute process
   holds the GPU (the owner policy-server claims the H100 for rig
   serving; it is never preempted). Host-RAM watch item: a single
   process now carries loader defaults workers 8 × prefetch 4 at
   batch-96 (vs one 8×-rank's batch-12 shard) — `free -g` at first
   poll; if host memory pressure appears, `--num-workers`/
   `--prefetch-factor` may be rescaled as a machinery-only
   throughput knob (declared here; it does not touch training math).

## Read rule and frozen bounds (verbatim from the kit)

Instrument (frozen in `sft_drift_saga_charts.py` before this run
exists; fixture-validated): `delta(s) = eval_chunk_mae(s) −
eval_chunk_mae(500)`, primary read at s = 1000 (the endpoint probe).

- **HEALTHY** — `delta(1000) ≤ +0.30` → the distributed path is
  **CONVICTED**.
- **SAME-DRIFT** — `delta(1000) ≥ 0.5 ×` demosonly's delta over the
  same window (`+2.0317` ⇒ bound **+1.0158**) → the distributed path
  is **EXONERATED**; remaining suspects: image-augment, eff-96,
  recompute-stats-at-launch, init checkpoint, corpus scale.
- **else AMBIGUOUS** — the rigonly class; escalation (extend past
  1000 vs cut the next single-delta run) is an owner call.

Reference deltas over the same window (banked): demosonly **+2.0317**,
mixedv2 **+2.3319**, run-2 pooled **+0.4640**, rigonly **+0.6929**.
Fixture check passed: the kit run on the rigonly log lands
`+0.6929 → AMBIGUOUS`, matching its posted ambiguous-leaning-drift
verdict (`reports/analysis__sft_drift_discriminator_fixture.json`).
Train-slice delta and monotone-rise flags are reported as
corroboration, not gates. A read before step 1000 is PROVISIONAL (the
kit marks it).

## Gates and boundaries

- **GPU-hours gate: 12** (box estimate ~7–9 GPU-h; H100 at or under).
- **In-run instrument**: eval-250 probes; babysit registry entry at
  launch (`fontaine/harness/babysit.toml`), first poll checks
  util/rate + `free -g`, ≥30-min cadence after.
- **Endpoint boundary**: step-1000 → `sft_drift_saga_charts.py
  --discriminator outputs/train/grasp_sft_v2_demosonly_1gpu_disc/…
  train_log.jsonl` → `disc_overlay.png` + verdict JSON + in-channel
  verdict post. Checkpoints: step-500/1000 land under
  `~/checkpoints/finetune/grasp_sft_v2_demosonly_1gpu_disc`; upload
  to `fontaine-checkpoints` only if the verdict makes them
  load-bearing (HEALTHY endpoint = the first non-drifting v2-corpus
  checkpoint — that one banks same-session).
- **Interpretation grid (fixed now)**: CONVICTED → single-GPU is the
  only sane recipe class on this host anyway (the box is gone); the
  gated `prereg-draft-per-dataset-flow-norm-rerun` arm proceeds on a
  single-GPU recipe with drift risk retired. EXONERATED → the
  suspect list above is live; next single-delta run is an owner call,
  and the per-dataset-flow-norm rerun pre-reg states drift risk as
  unresolved. AMBIGUOUS → rigonly-class escalation, owner call
  (extend to 1500+ vs next single-delta run).

## Amendment 1 — measurement-scale calibration (2026-08-17 21:3xZ, posted BEFORE the step-500 probe)

Declared with eval data seen only through step 250 of attempt 2
(attempt 1 died in its first eval probe before producing any number —
see the incident post; the OOM fix rebatched the probe from the full
per-rank 96 to the training micro-batch 12, same seed-0 256-frame
draw).

**The fact**: the step-250 probe reads eval 12.5087 / train 12.4202.
The 8×A100 demosonly comparator read 3.4623 / 3.6862 at the matched
step (matched samples-seen: eff-96 both). Meanwhile AR CE tracks the
comparator (0.6385 vs 0.6116 at 250) and the flow aux loss runs ~3×
(0.0739 vs 0.0248).

**The interpretation**: this is the FIRST `bijou.train` run on the
merged family-norm stack (`d3dd4d0`, landed 18:0xZ — after every
comparator ran): per-item honest `action_stats` replaced the
merged-table override, so the probe's raw-action-unit conversion (and
the flow target scale) is a different measurement surface than the
one the comparators' absolute numbers — and therefore the frozen
bounds — were derived on. The AR-CE agreement says the MODEL is
tracking; the MAE offset is carried by the instrument's units.

**Frozen now, before any further eval data**:

- Scale estimator `s = 12.5087 / 3.4623 = 3.613` (matched-step-250
  eval level ratio; assumes matched model quality at 250, which the
  AR-CE agreement supports — declared as an estimate, not a fact).
- The step-1000 read computes BOTH: (a) the original raw-unit rule
  (Δ ≤ +0.30 HEALTHY / ≥ +1.0158 drift), and (b) the scale-adjusted
  rule (Δ ≤ +1.084 HEALTHY / ≥ +3.670 drift, i.e. the same bounds ×
  s).
- If (a) and (b) agree, that is the verdict. If they disagree, the
  verdict is **AMBIGUOUS-BY-INSTRUMENT**: no conviction or
  exoneration is claimed, both numbers + the per-dataset wandb
  breakdown go to the owner, and the disambiguator is a cheap
  stack-parity probe (score the SAME saved checkpoints' probe set on
  the pre-merge surface) before any recipe conclusions.
- The saga chart plots the discriminator with its own scale
  annotation rather than pretending unit comparability with the
  banked curves.

The kit's frozen constants in `sft_drift_saga_charts.py` are NOT
edited; the boundary session applies this amendment on top of the
kit's raw output.
