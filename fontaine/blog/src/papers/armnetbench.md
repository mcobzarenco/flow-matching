# ArmnetBench v0.1: someone built the eval farm we don't have — and released the failures

*Read 2026-08-09 (lit slice `lit-radar-0817`, priority 2 — priority 1
MolmoAct2 was satisfied by the [owner deep
dive](../posts/2026-08-09-molmoact2-deep-dive.md)). Paper:
[2607.24481](https://arxiv.org/abs/2607.24481) — "ArmnetBench v0.1:
Parallel Real-World Evaluation of Manipulation Policies on a
Low-Cost Arm Farm" (Selvaraj, Uttini, Kuosmanen; armnet.dev,
2026-07-27, CC BY 4.0).*

**The paper in plain words.** Evaluating robot policies is the
bottleneck nobody enjoys: someone has to stand at a real robot,
reset the scene, run the policy, and write down whether it worked —
hundreds of times per claim. This group built three cheap SO-101
robot stations (about $360–480 each, mostly 3D-printed) that run
evaluations semi-automatically: an operator spends ~10 seconds per
episode resetting objects and clicking a score, and three stations
run in parallel. They pushed seven well-known policies — π0.5, π0,
GR00T, Diffusion Policy, ACT, MolmoAct 2, SmolVLA — through twelve
tasks, each policy fine-tuned on the same 50 demonstrations per
task, and human-scored 2,518 rollouts. π0.5 won at 47.6% success;
over half of all rollouts failed. Then they released everything —
video, trajectories, and the success/failure label for every
episode — in the standard LeRobot format. The scoreboard is noisy,
but the release is something genuinely rare: thousands of *labeled
failures* on exactly the arm we train for.

## What it contributes

- **The farm itself**: 3 co-located SO-101 cells (2 single-arm, 1
  bimanual), $359/$477 per cell (Table 2) — SO-101 + 3 cameras
  (top/front pan-tilt RPi cams at 1024×576, wrist U20CAM 720p) +
  Raspberry Pi + networked power plug, recording at 20 fps.
  Operator cost ~10 s active time per rollout (reset + 3-way
  score), 3 operators supervising 3 cells.
- **A shared-budget leaderboard**: 7 policies × 12 tasks (8
  single-arm, 4 bimanual), every policy fine-tuned per task on the
  **same 50 teleop demos** — the contest is data-efficiency at a
  fixed demo budget, not peak capability.
- **The release** (the durable contribution): two LeRobot-v3.0
  datasets under Apache 2.0 —
  [single-arm](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101)
  (2,499 episodes: 915 success / 52 suboptimal / 1,532 failure) and
  [bimanual](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_bimanual_so101)
  (1,219 episodes: 409/54/756) — full video + float32[6] (or [12])
  joint trajectories, with per-episode `success_class`,
  `policy_type`, `policy_repo_id` fields. **2,288 labeled failure
  episodes on a bone-stock SO-101 config**, directly loadable with
  `LeRobotDataset`.

## The experiments it ran

Core benchmark: 2,518 human-scored policy rollouts + 600 reference
demos = 3,118 core episodes (~30 rollouts per task–policy cell;
3,718 episodes released once non-core extras are counted). Label
taxonomy successful / suboptimal / failure, but "suboptimal" was
used for only 3.5% of rollouts — in practice the labels are nearly
binary. Pooled leaderboard (Table 5): **π0.5 47.6%** > π0 35.1 >
GR00T N1.7 29.4 > Diffusion Policy 26.7 > ACT 19.2 > **MolmoAct 2
18.9** > SmolVLA 15.0. Overall 56% of rollouts failed; no policy
ever succeeded at cable_clip; best task–policy cells hit 60–86%.
Rankings flip by embodiment (π0.5: 45.4% single-arm vs 52.1%
bimanual) and per-task variance is wild (tool_removal: Diffusion
63%, ACT 27%, SmolVLA 3%). **What the paper does NOT contain: any
offline-metric-vs-real-success correlation study** — no validation
loss, no MAE, no sim comparison. It is the real-rollout half of the
calibration question, published without the offline half.

## What transfers to us

- **The failure corpus is the prize (#16, #6).** Labeled failure
  rollouts on our exact embodiment (same 6-dim joint schema, 20
  fps, LeRobot v3.0 — schema-identical to community_curated_v0) are
  data we cannot collect without a rig and would never get from
  success-biased hub uploads. Uses that survive scrutiny: failure-
  detector *evaluation* corpus for the #6 slot (held-out policy
  rollouts with ground-truth outcome labels, spanning 7 policy
  families including flow-based π0/π0.5 and the Molmo-trunk
  MolmoAct 2), reward-model or quality-conditioned training signal
  (#16's RL pole needs exactly this: failures with terminal
  labels — the LWD lesson that success-only corpora collapse the
  advantage signal).
- **The #9 calibration study is now *enabled but blocked*.** The
  clean version — run our offline probes on their evaluated
  checkpoints, correlate against their measured success rates —
  needs the 84 task–policy checkpoints, which the paper claims are
  released but which are **not on the Hub** (org has zero public
  models as of today). Until they appear, the fallback is weaker:
  test whether trajectory-space similarity-to-demo metrics separate
  their success from failure episodes.
- **A Molmo-trunk caution flag (#17):** MolmoAct 2 ranked 6/7 at
  18.9% under the 50-demo fine-tune budget — Molmo-trunk VLAs are
  not automatically strong in low-data per-task adaptation, at
  least under their undisclosed recipe. (See the confound below
  before quoting this number.)

## What doesn't transfer

- **Absolute rates and rankings.** Training recipes are
  undisclosed and self-acknowledged as recipe-dependent; n≈30 per
  task–policy cell puts ±15–18-pt confidence intervals on per-task
  numbers, so adjacent pooled ranks (ACT 19.2 vs MolmoAct 2 18.9)
  are indistinguishable. Each task ran on exactly one cell, so
  task-vs-cell effects are confounded.
- **Visual detectors trained on their footage**: fixed pan-tilt
  camera geometry differs from typical community setups —
  trajectory-space signals will travel better than pixels.
- **The MolmoAct 2 number specifically**: cell-3's front camera was
  misaligned for every policy *except* MolmoAct 2, and the bimanual
  right-wrist camera was blurry for all *except* MolmoAct 2 — it
  was evaluated under slightly different camera conditions than
  the other six. Direction of the bias is unclear; quote 18.9%
  only with this asterisk.

## Hook corrections

The banked one-liner ("7 policies x 12 tasks, 3,118 human-labeled
episodes success/suboptimal/failure RELEASED") was wrong in two
ways and right in the one that matters: (1) only **2,518** episodes
are human-scored rollouts — the other 600 are demos "successful by
construction," never scored (and the actual release is 3,718 once
extra non-core rollouts are counted); (2) the **checkpoint release
is claimed but undelivered** — the cleanest downstream study is
blocked on it; (3) "rare labeled failure-rollout data on our exact
embodiment" holds up fully — 2,288 failures, Apache 2.0, LeRobot
v3.0 native.

## Which idea/arm it fed

#16 (`rig-transfer-benchmark`) — the RL-pole's missing ingredient
(labeled failures with terminal outcomes on SO-101) now exists as a
public artifact; banked as the designated calibration/eval corpus.
#6 (`aux-attribution` failure-detection slot) — the detector eval
corpus: any #6 candidate (FoMo-FD-style world model, SAFECAST-style
probes) can now be *scored* against ground-truth labeled rollouts
without our own rig time. #9 — the offline↔real calibration study
is specified and waiting on their checkpoint release (watch item).
No gate changes.
