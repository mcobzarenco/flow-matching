# MolmoAct2 on our panel: out-of-band eval plan (deep implementation read)

*2026-08-10 · owner steering 10:50Z/11:06Z: "I'd like to evaluate
molmo2act on our panel … deeply read their implementation … does
their model predict only 1 sec of actions at whatever the fps of the
dataset is? … How does normalisation of actions work? … post an
in-depth plan of how we'll eval our panel out of band."*

**In plain words:** MolmoAct2 is AllenAI's robot-control model built
on the *same* Molmo2-ER backbone our current er_60k run trains from.
It reads camera images plus a text description of the task and the
robot's current joint positions, and predicts the next second of
joint movements. We want to score it on the same 25,800-frame test
panel we use for our own models — without retraining or modifying
anything — to see how a heavily-resourced open model compares on our
exact data. This post is the result of a deep read of their code and
the concrete plan for doing that comparison fairly. The two headline
subtleties: their model predicts a **shorter time window** than ours
(1.0 s vs 1.67 s), so the fair comparison re-scores our models on
their window; and **31% of our panel's frames come from datasets
MolmoAct2 trained on**, so we report clean and contaminated splits
separately.

Repo read: `github.com/allenai/molmoact2` (cloned locally,
`lerobot` submodule initialized). Everything below is from the code,
with file:line receipts; nothing is from the paper or docs on faith.
Companion piece: the paper-level
[MolmoAct2 deep dive](2026-08-09-molmoact2-deep-dive.md) from
yesterday's owner request covers the system's story, data, and
benchmark results; this post is the code-level ground truth needed
to actually run it on our panel.

## 1. What the model is

- **Backbone**: Molmo2-ER (Qwen3-4B-class LLM + SigLIP-class ViT;
  the system totals ~5B with the expert) — the exact HF checkpoint
  our er_60k run warm-starts from. Their VLA =
  our trunk + robot-state prompt injection + a flow-matching action
  expert. This makes the comparison unusually clean architecturally.
- **Action expert**: DiT-style adaLN-Zero blocks, width 768, 8
  heads, one expert block per LLM layer (36 for this backbone),
  conditioned by **cross-attending into each LLM layer's KV cache**
  (not just final hidden states)
  (`experiments/launch_scripts/lerobot_utils/hf.py:328`,
  `experiments/olmo/hf_model/modeling_molmoact2.py:2870`).
- **Objective/solver**: rectified flow (`xt = (1-t)·noise + t·x`,
  target `x − noise`), inference = **fixed 10-step uniform Euler**,
  t: 0→1 (`modeling_molmoact2.py:3141,3260`; default
  `flow_matching_num_steps=10`).
- **Two output modes**: continuous (flow expert, what all shipped
  servers use) and discrete (autoregressive FAST tokens) — selected
  exclusively, never cascaded (`modeling_molmoact2.py:4483`). We
  eval the continuous mode.
- The relevant checkpoint for our panel:
  **`allenai/MolmoAct2-SO100_101`** with norm tag
  **`so100_so101_molmoact2`** — fine-tuned on 1,220 community
  SO-100/101 LeRobot repos (`data_constants.py:782,2896`).

## 2. The horizon question — owner's hunch confirmed

The horizon is a **fixed frame count per embodiment tag**, stored in
the checkpoint's `norm_stats.json` metadata, consumed as
`action_indices = [0 … H-1]` at the dataset's native fps
(`experiments/olmo/data/lerobot_wrapper.py:1186`):

| tag | horizon | control mode |
|---|---|---|
| `so100_so101_molmoact2` | **30** | absolute joint pose |
| `yam_dual_molmoact2` | 30 | absolute joint pose |
| `franka_droid` | 15 | absolute joint pose |
| `widowx_bridge` / `google_robot_fractal` | 5 / 3 | delta EEF |

(`experiments/launch_scripts/data_mixtures.py:201-292`.) Nothing in
the code says "1 second" — but SO-100 data is 30 fps, DROID 15 Hz,
bridge 5 Hz, RT-1 3 Hz, so every tag works out to **≈1.0 s of future
actions**. For our panel (30 fps): their prediction is
**30 steps = 1.0 s; our panel chunk is 50 steps = 1.67 s.**

**Consequence for MAE.** Later chunk steps are strictly harder
(uncertainty grows with lead time — our own per-step error curves
rise through the chunk), so pooled-over-50 vs pooled-over-30 is not
a fair fight in either direction. The fix is free: our banked panel
npzs store per-step errors for all 50 steps, so we **re-pool our own
banked predictions over steps 0–29 only** — a pure-CPU read, no
GPU re-eval of our checkpoints needed. Primary comparison:
**matched-window (first 30 steps ≈ 1 s) chunk MAE, both models, same
frames, paired per-frame**. Our full-50 numbers stay as our internal
anchors but are never quoted against theirs.

## 3. How normalization works

Everything (state and action) is **q01/q99 quantile normalization to
[−1,1], per dimension, clipped**:

```
x_norm = clip( 2·(x − q01)/(q99 − q01) − 1 , −1, 1 )
```

(`experiments/olmo/data/robot_processing.py:72-101`; training flag
`--norm_mode q01_q99` is the default, `train_lerobot.py:296`.)
Inference **un-normalizes the expert's output** back to raw dataset
units with the inverse map (`modeling_molmoact2.py:4696`). The stats
ship inside the checkpoint as `norm_stats.json`, keyed by norm tag;
per-tag stats are a **count-weighted merge across all training
repos** of the tag (a cross-repo average of per-repo quantiles, not
a true global quantile — `lerobot_utils/stats.py:96-102`).

Notable details:

- **SO-100/101 is the only tag that normalizes the gripper** —
  `normalize_gripper=True` (`data_mixtures.py:222`). Every other tag
  passes gripper dims through raw. All 6 of our dims are normalized.
- Feature names are canonicalized (`main_`/`left_`/`right_` prefixes
  stripped) and must match
  `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
  gripper]` exactly (`stats.py:159-167`) — the same joint layout as
  our panel's 6 dims.
- **Model outputs arrive in raw dataset units** (LeRobot SO-100
  convention: joints in the −100..100 range or degrees, gripper
  0–100). Our panel truth is stored in the same LeRobot-native units
  our loaders emit, so no unit conversion should be needed — but
  this is verified, not assumed (step 5 of the plan).
- Robot **state is not a tensor input**: after normalization it is
  discretized into 256 bins and inlined in the prompt as
  `<state_start><state_N>…<state_end>` tokens, one per dim
  (`modeling_molmoact2.py:1271-1287`). The released expert refuses
  continuous state embeddings outright (`:720`).

## 4. Preprocessing and prompt (what we must reproduce exactly)

- Images: **exact squash-resize to 378×378** (aspect ratio not
  preserved, no letterboxing), SigLIP `x·2−1` normalization, 196
  pooled patch tokens per image; robot models all train with
  `--crop_mode=resize` — no multi-crop tiling
  (`image_processing_molmoact2.py:398`, `experiments/README.md:333`).
- Cameras: **identity is positional only** — no role tags, just
  `Image 1<|image|> Image 2<|image|>` ordinal prefixes (bare
  `<|image|>` for a single camera). The SO-100/101 tag deliberately
  declares no camera keys and was trained with per-episode
  **randomized camera order** over 1,220 heterogeneous community
  repos → the policy is camera-count- and order-agnostic
  (`lerobot_wrapper.py:2313-2346`). Our panel's 1–2 cam rows are
  both in-distribution; we pass cameras in the dataset's own key
  order.
- The verbatim prompt (`modeling_molmoact2.py:1312-1351`):

```
{images}<|im_start|>user
The task is to {task}. The setup is <setup_start>single so100/so101
robotic arm in molmoact2<setup_end>. The current state of the robot
is <state_start><state_…><state_end>. The expected control mode is
<control_start>absolute joint pose<control_end>. Given these, what
action should the robot take to complete the task?<|im_end|>
<|im_start|>assistant
<action_output>
```

- Task strings are normalized: whitespace-collapsed, prefix-stripped
  ("Instruction: …"), multi-sentence joined with "; ",
  **lowercased** (`:1290-1309`). We feed each panel frame's own
  LeRobot task annotation through their normalizer (it's applied
  inside `predict_action` by default).
- In continuous mode **nothing is sampled from the LM head** — one
  VLM prefill builds the KV cache, the expert integrates 10 Euler
  steps, done. Deterministic given the initial noise draw (we seed
  the generator; optionally mean-of-N draws later, matching our
  draws protocol).

## 5. Contamination (measured, not hypothetical)

Our panel is built from community SO-100/101 LeRobot repos — the
same pool AllenAI harvested. Measured against their fine-tune
mixture list (`SO100_SO101_MOLMOACT2`, 1,220 repos after their own
440-repo filter):

- **245 of our 878 panel repos are in their fine-tune mixture.**
- That is **31.0% of panel frames and 31.0% of core (pooled) frames**
  (7,996/25,800 and 5,332/17,204).

Their training saw *entire episodes* of those repos (their split ≠
our holdout split), so for those rows MolmoAct2 is being tested on
its own training distribution — likely on literally-seen frames.
Every read therefore lands **three ways: pooled / clean-633-repos /
contaminated-245-repos**, with the clean split as the honest
headline. (Caveat noted in the report: "clean" means absent from
their SO-100/101 *fine-tune* list; their pre-training mixture uses
the same tag lists at lower sampling weight, so clean-split numbers
still carry an asterisk vs a truly unseen benchmark.)

## 6. The eval plan (pre-registration sketch)

Out-of-band throughout: **no bijou.eval changes, record-only, no
gate on our runs, nothing repoints.** GPU: local H100, free after
the 15k-panel eval. Est. ~2–5 GPU-h (gate ≤ 8 — batch-1 prefill of
a ~5B VLM × 25,800 frames dominates; CUDA-graph capture for the
expert loop is ~2× and we enable it).

1. **Predictor script** (`fontaine/scripts/molmoact2_panel_predict.py`):
   iterate the exact panel rows (same plan json + holdout split seed
   0); for each frame load images + raw state + task string from the
   LeRobot datasets; call the HF checkpoint's own
   `model.predict_action(processor=…, images=…, task=…, state=…,
   norm_tag="so100_so101_molmoact2",
   inference_action_mode="continuous", enable_depth_reasoning=False,
   num_steps=10, seeded generator)` — their code does their
   preprocessing/prompt/normalization end-to-end; we adapt nothing
   ourselves. Returns `(30, 6)` un-normalized actions per frame.
2. **npz contract**: write `pred:molmoact2-so100@release` rows
   `(25800, 50, 6)` — steps 0–29 filled, steps 30–49 NaN + a
   30-step validity note in the analysis json; identity columns
   copied verbatim from a banked npz (hard-abort oracle if the
   dataset row iteration diverges from `index`).
3. **Matched-window reads instrument**
   (`molmoact2_panel_reads.py`, oracle-gated like the er15k one):
   all MAE pooling restricted to steps 0–29 ∩ valid, for **both**
   their rows and our banked 40k / 60k-cont / er15k (and later
   er-60k endpoint) rows; paired per-frame Δ + seeded bootstrap
   CI95; each read × {pooled, clean, contaminated}. State-copy rows
   re-pooled over the same window as the shared floor.
4. **Smoke before sweep**: 500 stratified frames (≥1 per 50 repos,
   both camera counts) + worst-frame gallery + scale sanity: their
   unnorm output must land in the truth's numeric range per dim —
   a wrong unit/sign shows up as state-copy-scale MAE instantly.
   Only then the full 25,800 sweep (systemd unit, `--report`-style
   HTML at the end).
5. **Verifications folded in**: (a) truth-units check vs their
   unnorm range (step 4); (b) their `n_obs_steps` config gotcha —
   the HF config class defaults `n_obs_steps=30` while training used
   1; if the shipped `config.json` lacks the key, chunk slicing
   starts at index 29 (`configuration_molmoact2.py:377` vs
   `convert_molmoact2_to_hf.py:442`) — we assert
   `config.n_obs_steps == 1` at load; (c) kwarg drift — newer
   snapshots take `inference_action_mode`, the old DROID server used
   `action_mode`; we match the snapshot's signature at runtime.
6. **Decision line (frozen)**: this is a reference point, not a
   gate. Whatever the deltas, our runs' kill lines and the er_60k
   endpoint protocol are untouched. What it *informs*: whether
   VLA-style state-in-prompt + flow expert on the identical trunk
   is worth a pre-registered arm of our own (their expert recipe on
   our data pipeline), and how far our from-scratch decoder is from
   a 1,220-repo fine-tune.

Open item for the owner: none blocking — checkpoint is public
(Apache 2.0, ~22 GB, bf16 fits the H100 easily). If you'd rather
score a *base* MolmoAct2 (not the SO-100 fine-tune) as a second
arm, say so and it becomes a +1 sweep with the same harness
(`norm_tag` still required — base ships the same tag set).

*Full pre-registration (frames, seeds, abort oracles, exact read
list) lands as its own post before any GPU minute is spent, per
charter. The deep-read receipts above are the paper trail.*
