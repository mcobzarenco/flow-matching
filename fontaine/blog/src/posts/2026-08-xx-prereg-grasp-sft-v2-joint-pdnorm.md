# Pre-registration: grasp-SFT v2 mixed rerun with per-dataset flow normalization

*DRAFT cut 2026-08-18 01:xxZ (work session, queue item
`prereg-draft-per-dataset-flow-norm-rerun`); posting + launch are
owner-gated as always. Runs as `grasp_sft_v2_joint_1gpu_pdnorm` on the
local H100 via
`fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh`
(staged, full-parse green vs the merged CLI: family-inferred
`molmoact2_joint`, `per_dataset_flow_norm=True`, seed 0). Follows the
[isolation verdict](2026-08-17-sft-v1-flow-isolation.md) (its recipe
recommendation), the
[discriminator verdict](2026-08-18-sft-drift-discriminator-verdict.md)
(its interpretation grid: single-GPU recipe class, drift risk retired),
and the [v2 pre-reg](2026-08-17-prereg-grasp-sft-v2-joint.md) (whose
grid named this cell next).*

**Plain words.** Our best small model grasps the boat 44 times out of
100; every retrain on the bigger mixed corpus (simulated demos plus
real robot recordings) almost never grasps. The isolation work found
the likely culprit: before training, action values get squashed into a
standard range through one shared table, and mixing datasets makes
that table fit the simulated demos badly — always distorting a wrist
joint, exactly where a grasp lives or dies. We built a fix — each
dataset squashes through its own table — and separately proved the
other prime suspect (the 8-GPU training machinery) guilty of an
unrelated disease, so this run uses the now-proven single-GPU setup.
This experiment reruns the mixed-corpus recipe with exactly one
change: the per-dataset tables. If the model now grasps, the mix is
vindicated and the fix becomes the recipe; if it still fails, the mix
itself (sim and real data interfering) becomes the prime suspect. The
pass/fail counts are written down below, before the run starts.

## The question

The [occupancy analysis](2026-08-17-sft-v1-flow-isolation.md)
quantified how a shared normalization table breaks mixed training:
run-2's pooled table crushed wrist_flex to **0.24×** gradient weight
(48.9% occupancy); run-1b's rig-lineage table overflowed wrist_roll at
**288%** (targets clipping, serving capped at ~±66° of a ±157°
motion). The one run whose table fits its own data grasps 44/100. The
[stack-parity probe](2026-08-18-sft-drift-discriminator-verdict.md)
added an independent signature: under the old rig-lineage table our
demos-trained model's worst motor by far is wrist_roll (16.87@500 /
12.31@1000 vs state-copy's 3.99) — the same channel, from a different
instrument.

`--per-dataset-flow-norm` (enabler `6a6a0aa`, family-level port
`d3dd4d0`, oracle suite `tests/test_per_dataset_flow_norm.py`) makes
flow targets normalize under each item's OWN dataset q01/q99 row —
sim supervision through a sim-fit window, rig supervision through the
rig's — while CE/state tables stay merged. Serving reads the recorded
scheme (`q01q99_per_dataset`) at load and denormalizes under the row
the item wears.

**Why the mixed cell, not a demosonly one-flag run.** The queue item
left the arm open (demosonly or mixed). Demosonly is settled by
inspection: with a single train dataset, `--recompute-stats` pools
over exactly that dataset, so the item's own row *is* the merged table
and the flag is a numerical no-op (`flow_normalize_targets` applies
the same q01/q99 map either way). The mechanism the flag fixes only
exists on a mix — and the mixed cell is the decision-relevant one:
per the isolation post, **v2-mixed with a sim-fit table is the clean
fourth cell**: if it grasps, the data mix is exonerated for free and
we have our first grasping mixed-corpus model; if it doesn't, the mix
becomes the prime suspect. Either way the run is the isolation — no
extra GPU-hours spent on it.

## Command

`fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh`
— the mixed-v2 box recipe
(`fontaine/scripts/box/launch_box_grasp_sft_v2_joint_8xa100.sh`) with
exactly **one recipe delta**: `--per-dataset-flow-norm`. The platform
form is the discriminator run's proven single-GPU shape, carried
verbatim: eff-batch 96 = micro-12 × 8 backward chunks,
`--activation-checkpointing --offload-optim` (measured 62.26/78 GiB on
this host — eff-batch unchanged at 96, so the 08-08 OOM-ladder
preflight condition is not triggered), same 3-dataset mix
(`grasp_demos_v2/merged` + `so101_pick_place_v2` +
`so101_pick_place_clean`, `--dataset-repeat 'so101_pick_place*=4'`),
same `--recompute-stats`, same joint objective with `--insulate-flow`,
same lrs / `--image-augment 0.8` / holdout 0.1 / eval-250 with
`--eval-dataset-breakdown` / save-500, same default seed 0 (seed
policy: same seed for comparability), **3000 steps** (the registered
mixed-v2 length). The distributed machinery (`torchrun + zero1 +
chunk-grad-allreduce`) is deleted — that path was
[convicted](2026-08-18-sft-drift-discriminator-verdict.md) 00:42Z.

Fit smoke (`STEPS=20 SMOKE=1`) before the full run, per house rules.
Compute-app abort guard for the owner policy-server, as on the
discriminator launcher.

## Baseline arms and anchors

- **Baseline run**: `grasp_sft_v2_demosonly_1gpu_disc` — same
  platform, same single-GPU form, `per_dataset_flow_norm=False`;
  saves 500/1000 + full eval jsonl banked at
  `fontaine-checkpoints/grasp_sft_v2_demosonly_1gpu_disc`. Its demos
  holdout is **the same episode set** this run holds out (the split is
  a pure function of `(repo_id, episodes, fraction, split_seed)`, all
  four identical for the demos dataset), so the demos-slice breakdown
  curve is directly comparable.
- **Grasp anchors** (sim rollouts, unseen seeds): probe
  `joint_corrected@2000` **44/100**; run-2 **5/100** (box) / 0/20
  (local); run-1b **0/20**. The broken class sits at ~5%.
- **Baseline cell MEASURED** (04:19Z 08-18, before GO — the
  demosonly-v2 leg this draft queued un-gated):
  `grasp_sft_v2_demosonly_1gpu_disc/step_001000` reads **11/100**
  (mean progress 2.04 cm, 64/100 moved >0.5 cm, 0 strikes; 7 of the
  11 success seeds are probe-success seeds;
  [report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_v2_demosonly_1gpu_disc__step_001000__flow_unseen100.html)).
  Sits at the top edge of the broken class's CI (5/100 → ~2–11) and
  far below the probe band — the healthy-training + honest-stats
  demosonly cell does NOT restore probe-level grasping, so the mix
  is not the only suspect for the grasp gap. **Calibration note,
  recorded pre-launch**: the baseline itself lands inside this
  draft's 11–19 ambiguous band, i.e. the ≥20 exoneration bar asks
  the mixed run to BEAT its demosonly control roughly twofold. The
  absolute bands stay frozen; a paired per-seed read vs this
  baseline's 100 episodes will be recorded alongside them (owner
  flagged in-channel with the GO ask still open). **Instrument
  frozen pre-data** (05:5xZ 08-18): `sim100_paired_read.py` —
  success-count delta with seed-0 10k-resample bootstrap CI95,
  discordant-seed table with exact two-sided McNemar p, paired
  progress delta CI; oracle `tests/test_sim100_paired_read.py`,
  retro-validated on probe(44) vs this baseline(11): +33 successes
  CI95 [22, 44], 37-vs-4 discordant, progress +3.57 cm [2.66, 4.46]
  ([banked read](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim100_paired_probe_vs_disc1000.json)).
- **Drift anchors**: mixed-v2 8× rose **+2.33** over the 500→1000
  window (killed @~1150); the discriminator fell **−1.67** over the
  same window on this exact platform and instrument.

## Reads and frozen decision grid

**Primary — grasp competence.** At step 3000: sim100 flow leg, 100
unseen seeds, on this host (`sim.rollout_sim`, episode 30 s,
execute-horizon 30, euler-10, bfloat16 decoder — the v1/v2 sim100
protocol). Decision bounds, fixed now:

- **≥ 20/100 grasps** → the table was the lever and the **mix is
  exonerated**: first working mixed-corpus checkpoint; banks
  same-session with an HTML panel; per-dataset flow norm becomes the
  house recipe for mixed corpora.
- **≤ 10/100** → table fit, machinery gone, and it still fails: the
  **mix itself is the prime suspect** (sim/real interference at the
  flow head). Next isolation is an owner call; the draft names
  demos + one-rig-dataset as the cheapest next cell.
- **11–19** → ambiguous band: per-channel MAE, per-slice breakdown,
  and rollout videos go to the owner before any recipe claim.

(For calibration: 5/100 has a 95% CI of roughly 2–11; 44/100 roughly
34–54. The bands are chosen to separate the broken class from the
healthy one with no overlap.)

**Serving-row rule (frozen, instrument prep landed this session).**
Under the per-dataset scheme a served chunk denormalizes under the row
the item WEARS, so the sim eval must wear the sim demos' row — the
sequential and parallel sim drivers previously hardcoded the rig row
(`so101_pick_place_v2`) with a merged fallback, which would have
re-introduced the exact wrist_roll window crush at serving. Both
drivers now take `--stats-repo-id` (explicit row, loud refusal on a
miss; default behavior bit-unchanged; oracle
`tests/test_worn_stats_row.py`). The sim100 legs here run
`--stats-repo-id grasp_demos_v2/merged`. Rig serving wears the rig
row — that asymmetry is the scheme working as designed, not a
confound. The k4l2 panel needs no override: post-`d3dd4d0` eval items
wear their own dataset rows honestly.

**Secondary — drift guard.** In-train eval probe,
`Δeval(1000−500) ≤ +0.30` (the discriminator's raw-units rule; same
merged-stack instrument, directly comparable — the parity probe
measured the two surfaces within ×1.03). A failure here would be NEW
information (mix-specific drift on a single GPU): the grasp read still
stands, but the endpoint choice re-opens to the best-grasping save and
the drift becomes its own investigation item.

**Tertiary — panel guard, paired at endpoint.** k4l2 panel
(`panel_v2` instrument) at step 3000, paired vs the discriminator's
banked step-1000 on the shared frames: mixing real rig data plus the
fix should not leave real-data MAE worse than the demosonly baseline
by > +0.05 with CI excluding 0 (the house guard convention). Per-motor
deltas recorded — wrist_flex and wrist_roll are the channels the
mechanism predicts should move.

**Panel baseline MEASURED** (04:57Z 08-18, before GO — protocol
pinned in `eval_disc1000_k4l2_panel.sh`, the endpoint leg copies it):
disc-1000 reads **58.14** on the panel vs state-copy 8.37 (0% win) —
the demosonly checkpoint is catastrophically out-of-distribution on
community data despite beating state-copy on its own demos holdout
(5.76). Two consistent mechanisms, deliberately NOT adjudicated
pre-launch (instrument-audit item queued): weight-level forgetting
after 1000 narrow steps, and/or serving through the demos-recomputed
table's windows on community-range states. **Calibration note**: at
baseline 58.14 the +0.05 guard above is near-vacuous as framed — it
still catches "mixed worse than demosonly on real data", but any
plausible endpoint clears it; the informative panel comparison at
endpoint will be vs state-copy and vs the pre-SFT released
checkpoint's panel row, recorded alongside the frozen guard.

Curve-level record: demos-slice breakdown vs the discriminator curve
(same holdout episodes); rig-slice curves recorded as the first
per-dataset-normalized rig numbers. Token-leg sim100 and further
slices may run as unregistered corroboration; only the reads above are
gated.

## Gates and boundaries

- **GPU-hours gate: 21** — train ~13 (3000 steps at the
  discriminator's measured ~15.1 s/step) + sim100 pair ~4 (this run's
  endpoint + the baseline's step-1000 leg, which fills the demosonly-v2
  grasp cell of the isolation grid) + panel + probes.
- **In-run instrument**: eval-250 probes; babysit registry entry at
  launch (`fontaine/harness/babysit.toml`); first poll checks GPU
  util/rate + `free -g` (the mix adds rig-video decode to the batch-96
  loader buffers; worker/prefetch rescale is declared as a
  machinery-only knob, per the discriminator convention).
- **Boundaries**: step-1000 drift-guard read (PROVISIONAL for the
  grasp question); step-3000 endpoint → sim100 + panel + verdict post.
- **Checkpoint policy**: saves land under
  `~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm`; the
  endpoint banks to `fontaine-checkpoints` same-session if any gated
  read makes it load-bearing (a grasping mixed checkpoint certainly
  is), weights-only + logs, with the standing HTML report.
