# Qwen-RobotManip: the 38,100-hour corpus is one-third data, two-thirds re-render — but the curation pipeline is worth stealing

*Read 2026-08-09 (lit slice `lit-radar-0818`, priority 3). Paper:
[2606.17846](https://arxiv.org/abs/2606.17846) — "Qwen-RobotManip
Technical Report: Alignment Unlocks Scale for Robotic Manipulation
Foundation Models" (Yuan, Liang, Chen et al., Qwen team / Alibaba;
2026-06-16, v2 06-17; CC BY 4.0; 44 pp).*

**The paper in plain words.** The Qwen team built a robot-control
model on top of their 4B vision-language model and asked: can we
train it on a huge pile of hours the way language models are trained
on huge piles of text? Robot data is messy — every lab logs
different arms, different cameras, different conventions — so most
of the report is plumbing: filters that throw away recordings where
the logged arm motion doesn't match physics or the video, a common
numeric format so 15 different robots look alike to the model, and
a trick that takes videos of *human hands* doing chores and
re-renders a robot arm into the same scene, turning ~1,900 hours of
human video into ~24,800 hours of pretend robot data. The total
"38,100 hours" is therefore mostly that multiplication, not
collected robot experience. The model beats π-0.5 clearly, but
only on deliberately-hard out-of-distribution tests — on standard
benchmarks, models with no pretraining at all do just as well.
Nothing is released: no weights, no data, no code.

## What it contributes

- **A ~38,100h pretraining corpus from open sources only** — no
  proprietary teleop. Composition (Table 1): robot datasets
  11,420h (single-arm 3,808h: OXE ~600, DROID ~500, RoboMIND
  ~1,400, InternData-A1 ~3,600 *sim*, misc ~700; dual-arm 6,744h:
  AgiBotWorld-Beta ~2,400, RH20T ~1,100, Galaxea ~500, RoboCOIN
  ~430, etc.; mobile/humanoid 868h), human egocentric 1,933h
  (EgoDex 732, VITRA 247, EgoVerse 954), and **24,808h synthesized
  human-to-robot data** — the 1,933 human hours re-rendered across
  15 platforms (Panda, UR5e/UR10e, xArm7, ARX-L5, WidowX, AgileX
  ALOHA, …), a 12.8× multiplier. Unique data ~13,350h.
- **A five-stage state-action curation pipeline**: (1) jerk/spike
  detection via cascaded median + Savitzky-Golay residuals; (2)
  state-action directional-agreement on lag-aligned first diffs,
  episode dropped below DA 0.6–0.7 — "81% of episodes in the
  RoboMIND UR-type data failed this check"; (3) quantile band
  outlier removal; (4) forward-kinematics consistency via
  Pinocchio, used to *correct* TCP/frame conventions; (5)
  base-frame rotation alignment. Plus three cross-modal checks:
  VLM instruction-consistency voting, URDF-render-vs-SAM3-mask
  overlap, and visual-defect filtering.
- **Three-way "alignment"**: an 80-dim canonical state-action
  vector with per-dim masking (per arm: 7 joints + 9 EE pose + 1
  gripper + 12 hand); camera-frame delta-pose actions so visually
  similar motions are numerically close across embodiments;
  in-context adaptation reading intra-episode history as an
  implicit embodiment identifier.
- **Human-to-robot synthesis**: hand keypoints → virtual finger
  (0.7·index + 0.3·middle) → gripper pose/width; SAM3 masks +
  ProPainter inpainting remove the arm; base-pose grid search
  maximizes IK feasibility; MuJoCo render composited via Depth
  Anything v3 occlusion. Per-source speed alignment (EgoDex
  subsampled to 60%, EgoVerse 45%, VITRA 25%).

## The experiments it ran

- Architecture: Qwen3.5-4B VLM trunk, **fully joint-trained**
  (grads from flow-matching + lambda=0.1 next-token VLM loss;
  9:1 robot:VL data mix, ~28M VL samples incl. embodied CoT and
  2D-trajectory prediction, to prevent VLM forgetting). Action
  expert: flow-matching DiT, 10 blocks × d=768 × 12 heads
  (~100M-order by config; count not stated), cross-attn to VLM
  hidden states alternating visual/language tokens per block.
  4 Euler steps + real-time chunking at inference. K_repeat=8
  flow draws per sample during training.
- Headline OOD results (success %): RoboCasa365 35.9 vs π-0.5
  16.9 / GR00T-N1.5 23.9 / RLDX-1 33.2; LIBERO-Plus 89.0 (91.4
  with in-context variant) vs π-0.5 84.4; EBench 45.6 vs 27.1;
  RoboTwin-Clean2Rand-Hard 62.6 vs 47.9; RoboTwin-IF 72.2 vs
  49.6; RoboTwin-XE zero-shot cross-embodiment 23.9 vs 7.5.
  1st on RoboChallenge Table30-v1, "20% relative improvement."
- Key negative result: on *standard* LIBERO/RoboTwin, models
  without large-scale pretraining match or exceed pretrained
  ones — in-domain benchmarks fail to measure pretraining.
- Real-robot validation on AgileX ALOHA, Franka, UR, ARX
  (in-domain, OOD, few-shot, zero-shot cross-embodiment);
  per-platform trial counts/success rates not in the main tables
  we could extract.
- **What it did NOT run**: no ablation of the curation pipeline,
  no ablation of the 24,808h synthesis data, no data-scaling
  curves, no frozen-vs-unfrozen trunk comparison. The title's
  causal claim ("alignment unlocks scale") is never isolated.

## What transfers to us

- **The curation pipeline is a reference pipeline for our 229h
  corpus, fully offline** (no rollouts — fits our frozen-panel
  regime): jerk filter, DA check, quantile bands, FK consistency
  all run on logged state/action alone. The 81% RoboMIND-UR
  exclusion rate says community corpora can be *mostly* broken
  proprioception; our SO-100/101 community data is that class.
- **URDF-render-vs-mask check** is mechanizable for SO-101 (URDF
  in LeRobot) and catches video/state desync invisible to MAE.
- Their benchmark-saturation evidence independently corroborates
  VLM4VLA: in-domain metrics don't rank pretraining quality.
  Standing caution for panel-MAE-only evaluation.
- VL co-training recipe (9:1 mix, lambda=0.1 aux LM loss) is a
  concrete number pair if we ever unfreeze the trunk.

## What does NOT transfer

- **Nothing is public.** GitHub README verbatim: "There is
  currently no plan to release the model weights for
  Qwen-RobotManip or Qwen-RobotNav." No data, no code either;
  the repo is docs-only (131 stars). All numbers unverifiable.
- Human-to-robot synthesis needs calibrated intrinsics/extrinsics
  + SAM3/ProPainter/MuJoCo/Depth-Anything stack; camera-frame
  delta actions are "more sensitive to calibration errors" —
  poor fit for uncalibrated hobby SO-101 rigs.
- Their joint-trained-trunk choice carries no evidence against
  our frozen-trunk result: they never ran the frozen arm.
- Scale conversion: "166× our scale" is really ~58× unique data,
  ~34× real-robot teleop (~7,800h after removing sim + synth).

## Which idea it feeds

- **#9 (data levers)**: adopt stages 1–3 of their filter (jerk
  residuals, DA<0.6–0.7 episode drop, quantile bands) as a
  candidate cheap pass over our 229h; measure panel MAE trained
  with vs without excluded episodes. Add URDF-render mask-overlap
  as a stretch check. Log that the pipeline is *unablated* —
  reference, not evidence.
- **#17 (new trunks)**: third attachment pole recorded —
  cross-attn to hidden states with per-block visual/language
  alternation, ~1:40 expert:trunk ratio (vs our ~1:11 residual
  taps). Joint-training + aux VLM loss (lambda=0.1, 9:1 mix) is
  the priced anti-forgetting recipe if we ever unfreeze. Their
  standard-benchmark-saturation result strengthens the case for
  adding at least one OOD-style probe next to panel MAE.
