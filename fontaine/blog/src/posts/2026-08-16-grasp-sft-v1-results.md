# Grasp-SFT v1: the 5,000-demo joint run — endpoint results

*2026-08-16, drafted ~20:0xZ during the run's ride, finalized ⟨FINALIZE:
close stamp⟩. Run: `grasp_sft_v1_joint_8xa100` (owner GO 17:07:47Z on
the 16:39Z consolidated command, eff-96 delta; ASAP 17:29Z), 3000 steps
on the 8×A100 box, launched 18:21:14Z after a 12-minute
crash-fix-relaunch cycle (`2d6a2b3`). Anchors: the route-C joint probe
checkpoint — trained on ~10× fewer demos — scored
[44/100 unseen (flow)](2026-08-16-amendment-grasp-sft-route-c-joint.md)
and the token-head competence bar is ≥20/100.*

**Plain words.** Yesterday we generated 5,000 fresh demonstrations of
the pick-and-place task with our scripted expert (about 16× more than
the 313 the previous model learned from) and published them as a public
dataset. This page records the first model trained on that full corpus:
a "joint" fine-tune, meaning the same network simultaneously learns two
different ways of producing actions — a fast continuous head (flow) and
a token-by-token head that speaks the base model's discrete action
language. The previous, smaller run of this recipe completed the task
44 times out of 100 on scenarios it had never seen. The question this
run answers: does 16× more demonstration data move that number?
⟨FINALIZE: one-sentence answer⟩

## The run

- **Recipe** (route C, the only measured joint config): `bijou.train
  --objective joint --joint-ce-weight 1.0 --insulate-flow`, init from
  the owner's re-converted `molmoact2-so101-released` (remap-stats
  v21-to-v30), image-augment 0.8, eff. batch 96 (12×8 ranks),
  activation checkpointing + ZeRO-1, 3000 steps, eval every 250 with
  `--eval-dataset-breakdown`, async save every 500.
- **Data**: 3 datasets, 4,551 train episodes / 1.49M frames
  ([grasp_demos_v1](https://huggingface.co/datasets/mcobzarenco/fontaine-grasp-demos-v1)
  ~91% + `pick_place_v2` ×4 repeat 7.9% + `pick_place_clean` ×4 repeat
  0.8%), 506 episodes held out.
- **Incident**: launch 1 (17:49:48Z) died at its *first eval* — the
  ported `molmo_flow` decoder returns CPU actions and the joint family
  is the first to route the in-train eval through it. One-line fix
  (`2d6a2b3`, score on the eval device), relaunch 18:21:14Z, ~2.5 GPU-h
  lost. The relaunch passed the eval that killed launch 1, first try.
- **Cost**: ⟨FINALIZE: GPU-h⟩ vs the 40 GPU-h babysit gate; 3.9 s/step
  steady, VRAM ~64.5 GiB/rank, zero tracebacks.
- Curves: [wandb `grasp_sft_v1_joint_8xa100`](https://wandb.ai/aristotle1337/fontaine).

## Training curves

![component losses](../img/grasp_sft_v1/loss_curves.png)

Both heads learn without fighting: the flow MSE (insulated from the CE
gradient) and the action-token CE fall together. ⟨FINALIZE: final loss
sentence⟩

![eval MAE, pooled + per-dataset](../img/grasp_sft_v1/eval_mae.png)

Held-out chunk MAE ⟨FINALIZE: monotone? final value vs 14.53 at
250⟩. The per-dataset breakdown (`--eval-dataset-breakdown`, landed
`d642f7b`) splits the pooled number: the grasp corpus dominates
(~91% of eval frames) and tracks the pooled curve; `pick_place_v2` —
real-robot episodes, 8% share — reads noisier off a ~20-frame slice of
the 256-sample probe. ⟨FINALIZE: table + what the split says⟩

| dataset | eval MAE @3000 | train MAE @3000 |
|---|---|---|
| ⟨FINALIZE⟩ | | |

## The competence read: sim100 vs the probe anchors

![sim100 strips](../img/grasp_sft_v1/sim_strip.png)

Protocol identical to the route-C probe legs (euler-10 flow on unseen
seeds 0–99; `--serve-head ar` greedy for the token head), sharded 4×25
across the box's idle GPUs — exact, because every rollout stochastic
stream is keyed by the (seed, replan, draw) triple, invariant to batch
composition.

- **Flow, unseen**: ⟨FINALIZE⟩/100 vs the probe checkpoint's 44/100
  (and base 9, corrupt-table 28).
- **Token, unseen**: ⟨FINALIZE⟩/100 vs the R2 competence bar ≥20/100.

⟨FINALIZE: verdict paragraph — did 16× data move the needle; kept-split
if informative⟩

## Artifacts

- Checkpoint (weights-only, standing rule):
  [`grasp_sft_v1_joint_step3000`](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/grasp_sft_v1_joint_step3000)
- Banked reads: `reports/analysis__grasp_sft_v1_endpoint.json`,
  regenerable via `fontaine/scripts/grasp_sft_v1_endpoint_report.py`
- ⟨FINALIZE: HTML eval report link if rendered⟩

## What's next

⟨FINALIZE: next pointer — queue state, owner-gated items⟩
