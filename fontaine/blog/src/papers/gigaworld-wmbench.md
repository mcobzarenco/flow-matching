# GigaWorld-1 / WMBench: the first public report card for world-model evaluators — but its 324K "rollouts" are graded videos, and real-robot policy ranking is never touched

*Read 2026-08-10 (lit slice `lit-radar-0821`, priority 4). Paper:
[2607.02642](https://arxiv.org/abs/2607.02642) — "GigaWorld-1: A
Roadmap to Build World Models for Robot Policy Evaluation" (GigaWorld
Team, GigaAI — 26 named authors incl. Jiwen Lu; arXiv cs.RO, v1
2026-07-02; CC BY 4.0; [project
page](https://open-gigaai.github.io/giga-world-1/) verified live).
Artifacts verified 2026-08-10: **code released** —
[github.com/open-gigaai/giga-world-1](https://github.com/open-gigaai/giga-world-1)
Apache-2.0, 1,120 stars, pushed 2026-07-12 (stage-1 training, DMD2
distillation training, i2v/t2v inference, LeRobot-style data
pipeline); **weights released** — HF
[open-gigaai/Giga-World-1](https://huggingface.co/open-gigaai/Giga-World-1)
stage-1 Nano 1.3B + Pro 5B (distilled stage-2 weights "coming soon");
toy dataset + CVPR-2026 challenge dataset on HF; WMBench itself
**"partially open-sourced"** per the repo's own status table (15
metrics + leaderboard + VLM judging live; RL post-training and the
acceleration stack "coming soon").*

**The paper in plain words.** A world model is a pretend robot run:
show it a photo of the scene and a sequence of arm motions, and it
dreams the video of what happens next. People want to use these
dreams to grade robot policies without touching a real robot. This
paper asks the question one step earlier, the one our rollout-free
eval read left open: before you trust the dream to grade anything,
who grades the dream? Their answer is a report card. Take thousands
of real recorded robot episodes where the true ending is on film,
feed each pretend-world the same first frame and the exact same
recorded motions, and check whether the dreamed movie ends the way
reality did. They collected 324,000 such dreamed segments — from over
100 teams' models submitted to a competition they hosted — and paid
three human annotators per video to grade each on a four-point scale
whose ordering is itself a thesis: a blurry movie with the *right
ending* outranks a gorgeous movie with the *wrong one*. Then they
asked which cheap automatic scores predict the human grade. The
answer is tidy: the object staying itself and the camera geometry
staying sane predict it strongly; a rock-steady, pretty background
*anti*-predicts it, because the easiest way for a video model to look
stable is to freeze and do nothing. Models degrade over 40-second
horizons unless given explicit memory. They distill all this into
their own model, top their own leaderboard — and, unusual for this
class, actually ship it: weights, training code, and a data pipeline
that reads the same LeRobot format our corpus is stored in. The
catch: nowhere in 324,000 videos does anyone check that a world model
with a good report card *ranks policies* the way a real robot would.
The dreams replay recorded human motions; no policy ever drives. The
one published measurement of that gap (PolaRiS grading Ctrl-World)
found a fluent video model mis-ranking policies badly. For us the
report card is the useful part — it runs on data we already have —
but passing it is a necessary condition, not the certificate.

## What it contributes

- **WMBench, a benchmark for world models *as evaluators*.** Corpus:
  2,989 paired real trajectories across 8 manipulation task families
  (rigid + deformable; humanoid, dual-arm, single-arm platforms),
  roughly 1:1 teleoperated demos vs rollouts from their own GigaBrain
  policy checkpoints; 82,470 s train / 7,200 s test,
  episode-disjoint. "Paired" means: real first frame + real action
  sequence + real outcome video, so a world model can be interrogated
  by replay and its generation compared against what actually
  happened. ~15 automatic metrics in three groups — frame/
  representation fidelity (image quality, JEPA similarity, subject
  consistency), geometry/semantics/interaction (geometry accuracy,
  perspectivity, trajectory accuracy, instruction following), and
  motion/long-horizon (flow score, PSNR/FID/FVD over time).
- **WMES (World Model as Evaluator Score)**, the human ground truth:
  a 0–3 ordinal per rollout — 3 = accurate outcome + high fidelity,
  2 = accurate outcome + degraded visuals, 1 = *wrong* outcome
  despite high fidelity, 0 = wrong + degraded. Note the scale
  hard-codes outcome-over-prettiness: accurate-but-ugly (2) outranks
  pretty-but-wrong (1) *by definition*.
- **A validated VLM judge**: Qwen3-VL-8B-Instruct + LoRA (r=16),
  measured against human WMES on 5,000+ videos — **87.80% exact
  agreement, 99.16% adjacent, Spearman 0.7574, MAE 0.13**. This is
  the thing RoboWorld's GPT-4o judge never had: a
  human-agreement number, on an open-weights model you can run.
- **The 7×4 study**: seven video-model backbones (the tables name
  SVD, CogVideoX, LTX-Video, Wan 2.1, Wan 2.2, Cosmos-Predict 2.5,
  plus GigaWorld-1's own Wan-based instantiation; the exact seventh
  slot is fuzzy in the HTML — challenge-submission variants blur the
  count) crossed with four action-conditioning interfaces: none,
  cross-attention, ControlNet-style spatial, channel-concatenated
  control maps.
- **GigaWorld-1 itself**: Wan-backbone Nano (1.3B) / Pro (5B) trained
  on ~12,980 h (1,298 h physics/internet video + 5,377 h open-source
  robot incl. Open X/AgiBot + 2,411 h egocentric human + 3,894 h
  Giga-collected), with spatially-aligned action control (EE-pose
  maps + ray maps, channel-concatenated), hierarchical history with
  first-frame anchor, relative RoPE for long horizons, and a DMD2
  4–6-step distillation + SageAttention + sequence parallelism giving
  a claimed 35.93× inference speedup. Repo inference: 10 FPS, 33 s
  rollouts.

## The experiments they actually ran

What "324K+ simulated rollouts" actually are, precisely: they hosted
the CVPR 2026 World Model Challenge, and from 100+ teams' submitted
models they sampled **324,000 generated rollout segments**, chained
20–30 segments into complete 20–40 s closed-loop episodes, and had
**three independent human annotators (plus senior spot-checks)**
grade every one on WMES. They are world-model *outputs* under
replayed/benchmark action sequences, human-judged against the paired
real executions. They are not policy evaluations, and no real robot
ran anything new for them.

The findings, with the numbers that carry them:

- **Which metrics predict evaluator quality** (Spearman vs WMES
  across models): Subject Consistency **ρ = 0.88**, Perspectivity
  **0.86**, Instruction Following 0.84, visual-fidelity group 0.78,
  geometry group 0.71 — while Background Consistency is **ρ = −0.45**
  and photometric stability −0.42. The degenerate-stability result is
  the sharpest thing in the paper: static-scene metrics reward
  freezing, the exact failure mode that makes a world model useless
  as an evaluator.
- **Long-horizon degradation** (Table 4): Wan 2.1 without memory
  decays PSNR 14.46 → 13.37 and FVD 197 → 321 across a 40 s rollout;
  with their hierarchical memory, 19.82 → 17.41 and FVD 35 → 98.
  Long-horizon assessment, not single-clip quality, is where models
  separate.
- **Action interface matters more than backbone polish**: Trajectory
  Accuracy across the four conditioning schemes: none 0.1576,
  cross-attention 0.1620 (barely better than nothing — the standard
  choice is nearly inert), ControlNet-style 0.2566,
  channel-concatenated spatial maps **0.3528**. Actions must be
  *pixel-aligned* to be obeyed.
- **Data mix**: adding broad physical-interaction video to their
  robot data lifts the composite from 0.5654 to 0.6144; robot-only
  scaling gives less. Transferable physical priors beat model scale
  in their pretraining ablations.
- **Their own model**: +14.9% on "evaluator-alignment metrics" over
  competitive baselines (composite; the paper's own leaderboard).
  Training on 32 H20s; the repo recommends 8×H20/A100 and says
  consumer GPUs work with ZeRO/offload at reduced settings.

And the number that is *not* there: the paper defines the evaluator
target as Corr(S_real(π), S_wm(π)) over policies — and never reports
it. **No correlation between world-model policy scores and real-robot
policy success is computed anywhere.** Ground truth throughout is
human/VLM judgment of generated videos against paired recorded
executions. The "eval" in "evaluator score" is graded by humans
watching dreams, not by robots succeeding.

## What transfers to us — and what doesn't

- **The grading protocol runs on our data class.** WMBench's ground
  truth is exactly what we own: recorded real trajectories with
  actions and filmed outcomes. Our 229 h / 38.6k-episode LeRobot
  corpus supports the same interrogation — hold out episodes, replay
  their actions through a candidate world model from the real first
  frame, compare dreamed vs real video, judge outcome match. Their
  data pipeline literally ingests LeRobot-format datasets (Qwen3-VL
  captions + Depth Anything V2 preprocessing). No rig required. This
  is the first substrate in this whole literature whose *pre-trust
  screen* costs us zero real rollouts.
- **The judge is open and validated.** Qwen3-VL-8B + LoRA with
  measured 87.8% human agreement is a different trust class from
  RoboWorld's unvalidated closed GPT-4o rubric — and 8B-scale is
  runnable on our hardware.
- **What the screen cannot buy.** Replay-grading certifies the model
  on *demonstrator* actions. A policy being evaluated emits
  counterfactual actions — off-distribution exactly when the policy
  is bad, which is when eval matters. WMBench never closes that loop,
  and PolaRiS's measurement (Ctrl-World, MMRV 0.22, "heavy
  hallucinations during object interaction" causing mis-rankings)
  shows a fluent generator can still mis-grade policies. The
  rollout-free-eval bottom line stands: the *policy-ranking*
  certificate is still priced in real rollouts we don't have.
- **Our embodiment is absent.** No SO-100/SO-101, no community
  teleop anywhere in WMBench (their platforms are
  humanoid/dual-arm/single-arm at Giga scale; training mixes Open X
  and AgiBot). Their own Finding on robot-specific data — improves
  embodiment fidelity but sharpens trade-offs — says fine-tuning on
  our corpus would be mandatory, at unknown cost to the pretrained
  physical priors that their ablations say do the work.
- **Compute is nontrivial but not absurd.** Nano is 1.3B and stage-1
  weights are downloadable; fine-tune + replay-screen on our corpus
  is a multi-GPU project, not a fleet. The 35.93× fast path
  (distilled weights, acceleration stack) is the part still "coming
  soon."

## Hook corrections

Banked hook: *"7 video world models × 4 action reps, 324K+ simulated
rollouts: long-horizon action-faithful consistency matters more than
visual realism for eval alignment — frames the sim-grading question
the rollout-free-eval page opened; read with its Ctrl-World artifact
hook."*

1. **Right:** 7 models × 4 action representations confirmed (seventh
   slot fuzzy — challenge variants); 324K+ confirmed; the
   action-faithfulness-over-realism direction is real and *measured*,
   not just asserted — Subject Consistency/Perspectivity ρ = 0.88/
   0.86 vs Background Consistency ρ = −0.45 as WMES predictors, plus
   the 40 s degradation tables and the 0.16→0.35 Trajectory Accuracy
   jump from pixel-aligned action injection. It does frame the
   sim-grading question, and pairing with Ctrl-World was the right
   call.
2. **Wrong — "simulated rollouts":** they are world-model-*generated*
   video segments from 100+ challenge submissions under replayed
   action sequences, human-annotated on WMES. No policy drives; no
   new real-robot execution is compared. Reading them as policy
   evaluations (the natural reading) is false.
3. **Wrong — "eval alignment":** alignment here means agreement with
   *human judgment of the generated videos*, anchored to paired
   recorded executions. The real target, Corr(real policy success,
   world-model policy score), is defined in the paper and never
   reported. Also, part of the headline ordering is definitional: the
   WMES scale ranks accurate-but-degraded above pretty-but-wrong by
   construction.
4. **Missed — the biggest news:** the hook priced this as another
   framing paper; it is the first **released** full stack in the
   class — Apache-2.0 training/inference code, Nano/Pro stage-1
   weights, LeRobot-compatible pipeline, open validated VLM judge —
   with the caveats that WMBench itself is "partially open-sourced"
   and distilled weights are pending.
5. **Correction to the companion hook:** Ctrl-World is no longer "the
   only RELEASED artifact in the world-model-eval class" — it now
   shares that class with a bigger, more permissively licensed one.

## What it feeds

- **Idea #16 (rig benchmark / eval-substrate menu) — the
  world-model tier changes state.** The rollout-free-eval page banked
  tier 3 as "world-model eval — not actionable (no artifact, and
  uncalibratable without real rollouts we lack)." Half of that
  verdict is now dead and half is confirmed stronger.
  **No-artifact: dead.** GigaWorld-1 Nano/Pro weights + Apache-2.0
  training code are live (verified 2026-08-10), Ctrl-World's MIT code
  + ~8G DROID checkpoint are live; the RoboWorld-style "reimplement
  from scratch" objection no longer holds. **Uncalibratable: refined,
  not removed.** WMBench contributes the missing middle rung: a
  pre-trust *replay screen* — replay held-out corpus actions, compare
  generations to real video, judge outcomes with an open VLM whose
  human agreement is measured — that costs zero real rollouts and
  runs on our LeRobot-format corpus as-is. The menu entry becomes:
  world-model eval = artifact available + replay screen computable
  now + policy-ranking calibration still priced in rig-day rollouts
  (the banked "every certificate was bought with real rollouts"
  verdict survives intact — WMBench just showed how much trust you
  can buy *before* paying it, and Ctrl-World's MMRV 0.22 shows the
  screen alone is not enough). Design constants to carry: grade with
  outcome-first ordinal scales, never static-stability metrics
  (ρ = −0.45 degenerate); demand pixel-aligned action conditioning in
  any candidate model; test at 20–40 s horizons where models actually
  separate. No GPU arm now — this stays a rig-era design note plus a
  cheap future screen, per the owner park.
- **Ctrl-World triage verdict (2510.10125, abs + repo only):** id
  resolves; ICLR 2026 (Guo, Shi, Chen, Finn — Stanford/Tsinghua);
  code **actually live** at
  [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)
  (MIT, 546 stars, pushed 2026-04-08) with a DROID-trained ~8G
  checkpoint on HF (`yjguo/Ctrl-World`), replay/keyboard/π0.5
  policy-in-the-loop scripts, DROID training pipeline. SVD-backbone —
  the weakest family class in WMBench's long-horizon tables, and the
  exact model PolaRiS measured mis-ranking policies. Released ≠
  reliable; deep-read only if we ever execute the world-model tier.
- **2511.11520 triage:** resolves to "Scalable Policy Evaluation with
  Video World Models" (Tseng, Gu, Zhang, Mao, Liu, Shkurti, Yen-Chen
  Lin — NVIDIA-adjacent list; v1 2025-11-14, CC BY-NC-ND; no code
  link on abs). Topic confirmed; keep as a spare hook, artifact
  signal absent.
- **New ids worth triage:** [2511.19861](https://arxiv.org/abs/2511.19861)
  "GigaWorld-0: World Models as Data Engine to Empower Embodied AI"
  (same team, Nov 2025 — the *training-data-generation* predecessor;
  relevant to #9 synthetic-data levers, and the repo's own README
  badges half-point at it). Unverified adjacent: the VLAW
  world-model-post-training paper referenced in Ctrl-World's README
  (no id captured), and the GigaBrain policy tech report behind
  WMBench's rollout data.
