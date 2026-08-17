# Pre-registration: grasp-SFT v2 joint run

*2026-08-17 09:4xZ, posted in-channel before launch (owner go 09:08:27Z
"start train run v2", recipe locked 09:23:42Z: identical
hyperparameters, NO per-dataset norm). Runs as
`grasp_sft_v2_joint_8xa100` on the A100 box.*

**Plain words**: we retrain the robot policy exactly the way we trained
it yesterday — same model, same knobs, same seed — but on the new
v2 demonstration dataset (smoother expert, real-looking camera
bracket, refitted wrist camera). Yesterday's run learned to read its
own action tokens but its flow head forgot how to grasp in sim; the
open question this run answers is whether better demonstrations fix
that, with everything else held fixed.

## Command (run-2 script verbatim, two deltas)

`fontaine/scripts/box/launch_box_grasp_sft_v2_joint_8xa100.sh` — a
copy of run 2's `launch_box_grasp_sft_v1_joint_8xa100.sh` with exactly
two changes:

1. `--train-data ~/datasets/fontaine/grasp_demos_v2/merged` (was
   `grasp_demos_v1/merged`) — the only data delta.
2. Run/save/log names `grasp_sft_v2_joint_8xa100`.

Everything else byte-identical: `--objective joint --joint-ce-weight
1.0 --insulate-flow --recompute-stats --flow-decoder-init inherit
--image-augment 0.8 --decoder-lr 5e-5 --backbone-text-lr 1e-5 --steps
3000 --batch-size 12` (eff-96 over 8 ranks), `--backward-chunks 2
--chunk-grad-allreduce --zero1 --activation-checkpointing
--holdout-episodes 0.1 --eval-every 250 --eval-samples 256
--eval-dataset-breakdown --save-every 500`, same
`--dataset-repeat 'mcobzarenco/so101_pick_place*=4'`, same
`--init-from ~/checkpoints/molmoact2-so101-released` (the owner's
re-converted release), same default seed (seed policy: same seed for
comparability — this is the same run on better data, not a variance
probe). Fit smoke first per house rules (`STEPS=20 SMOKE=1`); the
recipe class is run 2's measured config (micro-12 + act-ckpt fits in
80 GiB).

**Normalization (the locked call)**: `--recompute-stats` pooled exact
q01/q99 over THIS run's train split (so the v2 demos' occupancy moves
the table), one merged table for CE, state and flow — the owner's
09:23Z decision. `--per-dataset-flow-norm` (landed `6a6a0aa`, default
off) is deliberately NOT used: the released checkpoint trained flow on
a wide mixture under one shared table and works. It stays banked as
the ready lever if v2 also collapses on the sim slice.

## Anchors (all banked before launch)

- **Run-2 endpoint, flow head**: sim100 5/100 (box) / sim20 0/20
  (local) — the regression under investigation.
- **Run-2 endpoint, token head (b779ba4-fixed decode)**: 3/20
  (seeds 100–119); the full-100 number lands today as eval-chain
  leg 3.
- **Joint-corrected probe**: 44/100 — the competence bar this recipe
  class has hit on v1-era data.
- **Run-2 step-500 flow sim100**: eval-chain leg 1, boundary ~10:1xZ
  today — lands before this run's own step 500 and dates the v1
  collapse (broken-from-start vs degraded-from-competence).
- **Watch item (real slice)**: run-2's v2-rig generalization gap
  (train 8.15 vs eval 15.77 flow-eval at 8% share) — the
  `--eval-dataset-breakdown` curves are the instrument.

## Gates and boundaries

- **GPU-hours gate: 40** (run 2 spent ~31, 21:14Z→01:08Z wall at
  ~3.9 s/step; same steps, same batch — expect ≈ the same).
- **In-run instrument**: eval-250 breakdown; the sim-slice flow-eval
  curve vs run-2's same curve is the primary in-flight read.
- **Endpoint boundary**: step-3000 checkpoint → convert → sim100 flow
  (primary, vs run-2's 5/100 and the 44/100 probe) + token leg (vs
  leg-3's number); checkpoint uploads to fontaine-checkpoints;
  HTML report per the standing rule.
- **Interpretation grid (fixed now)**: flow ≥ the probe band ⇒ v1
  demos were the lever, mix exonerated; flow ≈ run-2's 5/100 ⇒ data
  wasn't the lever — the banked per-dataset-norm cell becomes the
  next registered arm; token ↑ while flow flat ⇒ the heads are
  decoupling under insulation, isolation continues.
