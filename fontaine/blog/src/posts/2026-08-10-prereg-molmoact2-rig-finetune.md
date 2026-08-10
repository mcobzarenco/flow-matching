# Pre-registration: MolmoAct2 rig fine-tune, rung 1 (action-expert-only, local H100)

*Posted 2026-08-10 ~17:0xZ from owner GO 15:24:16Z ("Yes, I want a
runnable runbook and I want you to go ahead and do a fine-tune on the
local GPU"). Param sheet posted in-channel at the same time; the
owner-agreed protocol is **objection window, silence = launch** —
this pre-reg is immutable once the preflight + smoke oracles pass and
the launch fires. The companion runnable runbook (including the
rollout-server side for the rig machine) lands as its own page this
session.*

## What and why

Fine-tune the released **`allenai/MolmoAct2-SO100_101`** policy on
the owner's two SO-101 rig repos, on the free local H100, using
AllenAI's own training stack (`experiments/launch_scripts/
train_lerobot.py` in the cloned `~/molmoact2` repo) — nothing of ours
in the model path.

Motivating evidence (today's out-of-band panel eval,
[results](2026-08-10-molmoact2-oob-results.md)): the released
fine-tune is competent on repos inside its 1,220-repo training
mixture (beats state-copy by −0.75 there) and ~2× worse than
state-copy outside it (16.97 clean vs 8.32). The owner's rig repos
are exactly the "outside" case; a rig fine-tune closes precisely the
gap the eval measured. Their README recommends warm-starting from
SO100_101 for SO-101 embodiments.

## Data

| repo (local, `/home/ubuntu/datasets/`) | codebase | eps | frames | fps | cams |
|---|---|---|---|---|---|
| `mcobzarenco/so101_pick_place_clean` | **v3.0** | 7 | 3,399 | 30 | front, wrist |
| `mcobzarenco/so101_pick_place_v2` | **v3.0** | 50 | 32,679 | 30 | front, wrist |

57 episodes / 36,078 frames, 6-dim state+action (`so_follower`),
both repos already carry q01/q99 in `meta/stats.json` (measured —
no augmentation step needed). Trained on in full; **no held-out
episodes** — offline reads on these frames are convention/sanity
checks, not generalization claims; the real eval is rig rollouts.
(Owner may object and ask for a 3-episode holdout in the window.)

## Mixture registration (their API, patch in `~/molmoact2`)

New builder `so101_rig` in `experiments/launch_scripts/
data_mixtures.py`: **tag `so100_so101_molmoact2` reused verbatim**
from their pretrain SO-100/101 tag (same `setup_type`/`control_mode`
prompt text the trunk was trained with; `normalize_gripper=True`,
`action_horizon=30`, `n_action_steps=30`, `action_key=action`,
`state_keys=[observation.state]`), repo list = the two rig repos
only, `camera_keys=[observation.images.front, observation.images.wrist]`
(their SO-100 tag leaves camera keys empty, which their wrapper
rejects under default `random_camera_order=none` — pinning the rig's
two cameras is the validated path). Their `_collect_tagged_stats`
computes per-tag stats **from the mixture's repos only** → rig-only
q01/q99 normalization falls out of the registration itself; no
separate stats run.

## The v2.1/v3.0 joint-convention decision (owner thread 15:48Z)

**Decision: stay in v3.0 end-to-end, measured, no data conversion.**
Superseding my 15:50Z lean toward converting rig data v3.0→v2.1:
the documented conversion is exact only if the rig was calibrated
with exactly lerobot 0.5.1 — an assumption we cannot verify from
here, and a silent-error channel into training data. The v3.0
path is self-consistent by construction:

- Training consumes stored v3.0 values; per-joint rig-only q01/q99
  normalization absorbs any per-joint offset/scale between the
  model's native (v2.1-recorded) space and the rig's v3.0 space.
- The **only** residual risk is a per-joint sign mirror (a sign flip
  survives quantile normalization). That is *measured before
  launch*, not assumed: preflight P3 below runs the released
  checkpoint zero-shot on rig frames and requires positive per-joint
  motion correlation.
- Rollouts (runbook §rollout): the fine-tuned checkpoint then
  expects v3.0-space observations and emits v3.0-space actions — the
  server must **disable** the community `inference.py` default
  v2.1 conversion (offsets 0 / signs +1) and use the rig-tag stats.
  Requirement collapses to "rig calibration unchanged between
  recording and rollout," with no dependence on which lerobot
  version calibrated it. Belt-and-braces: step-0 continuity oracle +
  no-execute dry-run + command clamp before any motion.

## Recipe (their README "Action-Expert-Only Fine-Tuning", deltas marked)

Rung 1 = trunk frozen: `--ft_vlm=false --ft_action_expert=true
--ft_embedding=none --lora_enable=false` (VLM, ViT, connector,
embeddings, LM head frozen; flow-matching action expert trains).
`--action_expert_learning_rate=5e-5` (their value), warmup 200,
multimodal cosine to 0.1×, AdamW(0.9, 0.95) wd 0, grad-clip 1,
amp_bf16, `--packing=false --dynamic_seq_len=true` (their fine-tune
examples; compile off). Δ from their example: single GPU
(`--nproc-per-node=1`), `--global_batch_size=64` (their fine-tune
value) with `--device_batch_size=8` (micro-accum 8), **`--max_duration=2000`**
steps ≈ 3.5 epochs over 36k frames (short schedule per owner spec —
their 50k default is for datasets 100× larger),
`--save_interval=500`, keep all rungs, `--num_workers=12
--prefetch_factor=4 --pin_memory=true` (host RAM 221G free ≫ worker
buffers; rescale-on-first-poll rule applies), `--norm_mode=q01_q99`
(default), `state_format=discrete` / `action_format=continuous`
(defaults), img_aug full (their default). Seeds: their defaults
(data 50189 / train 6198) — fresh run, no comparability constraint.
Checkpoint: `allenai/MolmoAct2-SO100_101` (HF cache warm from
today's eval). Save folder
`~/molmoact2/checkpoints/finetune/fontaine_so101_rig_ae_r1`, unit
`fontaine-molmoact2-rig-ft` via `systemd-run --user`, log
`~/logs/molmoact2_rig_ft.log`, wandb project `fontaine`.

## Preflight oracles (all must pass before launch; hard abort)

- **P1 env**: fresh `~/molmoact2/.venv` imports olmo+lerobot 0.5.2 +
  torch 2.10 (done 16:5xZ); their built-in torchcodec preflight runs
  at plan build.
- **P2 stats**: dump the trainer's resolved `LEROBOT_STATS_BY_TAG`;
  q01/q99 must equal the count-weighted combination of the two rig
  repos' `stats.json` and differ from the released checkpoint's
  SO-100 community stats (proves rig-only normalization);
  `repo_to_tag` maps exactly the two rig repos.
- **P3 convention (GPU, ~10 min)**: released checkpoint zero-shot on
  ~240 evenly-strided rig frames (batch-1, the ported panel
  predictor path). Per joint: corr(predicted 30-step motion, truth
  motion) **> 0** and |signed step-0 offset| ≪ joint range. Any
  strongly negative joint ⇒ sign mirror ⇒ **launch holds**, finding
  posted, conversion path re-opened with the owner. Side product:
  zero-shot MAE + state-copy MAE on rig frames = the anchors the
  fine-tune must beat.
  **Amendment 1 (16:4xZ, posted in-channel inside the objection
  window, before launch)**: P3 ran and the half-span offset line
  fired on joint1 (+79.0). The added diagnostic (err~truth corr,
  pred/truth std) classifies it as zero-shot posture-collapse, not a
  convention offset: pred0_std 2.0 vs truth0_std 44.8, err~truth
  −0.999, and the measured mechanism is that 97% of rig joint1
  states fall outside the released checkpoint's state-normalization
  range ([43.7, 185.3] theirs vs [−103, +67] rig). The offset line
  is reclassified **record-only** (it detects the off-distribution
  weakness the fine-tune exists to fix); the launch-blocking P3
  criterion is the sign gate only — **all six joints positive
  (+0.12…+0.45), PASS**. The finding also confirms the affine
  per-joint gap between their native space and rig v3.0 (absorbed
  by rig-only q01/q99) and hardens the rollout rule: the server
  must use rig-tag stats, never the released ones.
- **P4 smoke train**: their smoke recipe (20 steps, bs 1, dynamic
  seq len) on `so101_rig`: finite decreasing loss, checkpoint
  writes, measured f/min recalibrates the wall-time projection —
  if projected train > gate, launch holds and the sheet is amended.

## Gate and kill lines

- **Gate: ≤ 12 GPU-h total** (train + smoke + preflight + post-run
  offline reads). Expectation: decode-bound; 128k sample-frames at
  ≥300 f/min ⇒ ~4–7 h wall.
- Kill: NaN/inf loss; sustained < 150 f/min across 3 consecutive
  ~30-min polls after step 100 (input starvation unfixed = wrong
  recipe for this box); vram alloc > 78 GiB; train loss not below
  its step-50 value by step 1000.
- First-poll (standing rule): util + f/min + vram + host RAM within
  ~15 min of launch; fix starvation before letting it ride.

## Expectations (pre-registered)

1. Train loss falls materially below its warm-start value within the
   first epoch (~570 steps) — the released policy is far off-manifold
   on rig data (13.87 matched-window vs snapflow 3.90).
2. Offline matched-window MAE on rig frames (contaminated-by-
   construction, labeled as such): fine-tuned rungs **beat both**
   the zero-shot anchor and state-copy on the same frames. Failing
   to beat state-copy on its own training frames = rung falsified.
3. No convention pathology: step-0 continuity holds on every rung
   (predicted step-0 near current state, no per-joint sign flips).

Non-goals of rung 1: no VLM/LoRA unfreezing (that is rung 2, only
if rung 1 plateaus above the anchors), no claim about rig task
success — that is the owner's rollout to run with the runbook's
dry-run gate.

## Post-run (chained, CPU/GPU-light)

`convert_molmoact2_to_hf.py` on the best rung → offline reads +
step-0 continuity via the ported predictor → HTML report on
fontaine-reports → in-channel numbers → checkpoint uploaded to
`fontaine-checkpoints` (standing rule). Babysit entry registered at
launch; boundaries ride with the box's er_60k schedule.
