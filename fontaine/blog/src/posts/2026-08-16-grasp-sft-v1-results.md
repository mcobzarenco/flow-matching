# Grasp-SFT v1: the 5,000-demo joint run — endpoint results

*2026-08-16→17. Run 2 of the recipe: `grasp_sft_v1_joint_8xa100`
RESTARTED with `--recompute-stats` on the owner's 20:51Z order
(launched 21:14:48Z 08-16, unit `grasp-sft-v1c`), complete 01:07Z
08-17 at step 3000. Run 1b — same command on the remap-only release
table — was killed at step ~1900 by the same order after its real-data
eval slice rose monotonically (16.0→18.4) while sim fell: the remapped
table leaves wrist_roll/wrist_flex saturating away from ground truth,
so per-channel normalization was recomputed from the corpus itself
(receipt: wrist_roll ±65.6/43.5 → ±157.2, wrist_flex 4.9/93.4 →
−52.4/95.0, lift lows −96.1 → −124.8, descending orientation
preserved). Anchors: the route-C joint probe checkpoint — trained on
~10× fewer demos — scored
[44/100 unseen (flow)](2026-08-16-amendment-grasp-sft-route-c-joint.md)
and the token-head competence bar is ≥20/100.*

**Plain words.** Two days ago we generated 5,000 fresh demonstrations
of the pick-and-place task with our scripted expert (about 16× more
than the 313 the previous model learned from). This page records the
model trained on that full corpus: a "joint" fine-tune, meaning the
same network simultaneously learns two ways of producing actions — a
fast continuous head (flow) and a token-by-token head that speaks the
base model's discrete action language. The first attempt at this run
was stopped two-thirds through and restarted when we found the model
was, in effect, reading mis-calibrated instruments: the table that
normalizes joint angles clipped two wrist joints well short of their
real range, and the real-robot slice of the eval was getting *worse*
as training went on. The restart recomputed that table from the data
itself. The question the run answers: does 16× more demonstration
data move the 44/100 success rate of the smaller-data model?
⟨FINALIZE: one-sentence answer⟩

## The run

- **Recipe** (route C, the only measured joint config): `bijou.train
  --objective joint --joint-ce-weight 1.0 --insulate-flow
  --recompute-stats`, init from the owner's re-converted
  `molmoact2-so101-released`, image-augment 0.8, eff. batch 96 (12×8
  ranks), activation checkpointing + ZeRO-1, 3000 steps, eval every
  250 with `--eval-dataset-breakdown`, async save every 500. Same
  seed as run 1 (registered comparability policy).
- **Data**: 3 datasets, 4,551 train episodes / 1.49M frames
  ([grasp_demos_v1](https://huggingface.co/datasets/mcobzarenco/fontaine-grasp-demos-v1)
  ~91% + `pick_place_v2` ×4 repeat 7.9% + `pick_place_clean` ×4 repeat
  0.8%), 506 episodes held out.
- **History**: launch 1a (17:49Z 08-16) died at its *first eval* — the
  ported `molmo_flow` decoder returns CPU actions and the joint family
  is the first to route the in-train eval through it; one-line fix
  (`2d6a2b3`), relaunch 18:21Z. Run 1b then trained to ~1900 before
  the owner-ordered kill above (~17.5 GPU-h spent; saves archived as
  `_run1_remaponly`). Run 2 = `main` merge `3a12c86`
  (`--recompute-stats`) + 20-step smoke with per-joint receipt before
  relaunch.
- **Cost**: run 2 **~31 GPU-h** (21:14:48Z→01:07:43Z wall on 8×A100,
  eval pauses included) vs the 40 GPU-h babysit gate; ~3.9 s/step
  steady, VRAM ~64.5 GiB/rank, zero tracebacks end to end.
- Curves: [wandb `grasp_sft_v1_joint_8xa100` run `cgo3by9j`](https://wandb.ai/aristotle1337/fontaine).

## Training curves

![component losses](../img/grasp_sft_v1/loss_curves.png)

Both heads learn without fighting: the flow MSE (insulated from the CE
gradient) and the action-token CE fall together; windowed loss ended
at 0.32–0.34 and grad norms stayed ~1.6 throughout. (Absolute MAE
below is NOT comparable to run 1's curve — the recomputed table
rescales the metric; per-dataset *trends* are the comparison.)

![eval MAE, pooled + per-dataset](../img/grasp_sft_v1/eval_mae.png)

Held-out chunk MAE was **not monotone — it oscillated all run**: 4.05
@250 → 4.54 → 3.74 → 5.00 → 5.09 → **3.62 @1500** (best) → **6.64
@1750** (worst) → 5.49 → 6.36 → 5.48 → 5.52 → **5.41 @3000**. Most of
the drop happened by step 250; after that the curve swung in a
widening band with train_mae tracking eval at every point (no
train/eval divergence on the pooled read — the swing is the model
moving, not the probe).

| dataset | eval MAE @3000 | train MAE @3000 |
|---|---|---|
| grasp_demos_v1 (sim, 91%) | 5.06 | 4.99 |
| pick_place_v2 (real, 8%) | **15.77** | **8.15** |
| pick_place_clean (real, 1%) | — (no eval slice) | 5.33 |

The split is the story: the sim slice sits tight (train≈eval≈5), but
the real-robot slice shows a **2× train/eval gap** — the model fits
the real *training* frames (8.15) and does not generalize to held-out
real episodes (15.77, vs 10.0 at step 250). At an 8% mix share, more
sim demos did not buy real-data generalization; run 1b's coverage
diagnosis (kill signature: v2 rising monotone) became, under the
corrected table, a *flat-to-bouncy* v2 slice that never improved past
its step-250 reading.

## The competence read: sim100 vs the probe anchors

![sim100 strips](../img/grasp_sft_v1/sim_strip.png)

Protocol identical to the route-C probe legs (euler-10 flow on unseen
seeds 0–99; `--serve-head ar` greedy for the token head), sharded 4×25
across the box's GPUs — exact, because every rollout stochastic
stream is keyed by the (seed, replan, draw) triple, invariant to batch
composition.

- **Flow, unseen**: ⟨FINALIZE⟩/100 vs the probe checkpoint's 44/100
  (and base 9, corrupt-table 28).
- **Token, unseen**: ⟨FINALIZE⟩/100 vs the R2 competence bar ≥20/100.

⟨FINALIZE: verdict paragraph — did 16× data move the needle⟩

## Artifacts

- Checkpoint (weights-only, standing rule):
  [`grasp_sft_v1_joint_step3000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/grasp_sft_v1_joint_step3000)
  (byte-verified against the box post-upload)
- Banked reads: `reports/analysis__grasp_sft_v1_endpoint.json`,
  regenerable via `fontaine/scripts/grasp_sft_v1_endpoint_report.py`
- ⟨FINALIZE: HTML eval report link⟩

## What's next

⟨FINALIZE: next pointer — queue state, owner-gated items⟩
