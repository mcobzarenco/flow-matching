# Quality over Quantity: influence curation anchored to 10 held-out demos, not rollouts — the offline point on the ATHENA axis, proven only where half the data is deliberately broken

*Read 2026-08-10 (lit slice `lit-radar-0821`, priority 1). Paper:
[2603.09056](https://arxiv.org/abs/2603.09056) — "Quality over
Quantity: Demonstration Curation via Influence Functions for
Data-Centric Robot Learning" (Haeone Lee, Taywon Min, Junsu Kim,
Sinjae Kang, Fangchen Liu, Lerrel Pinto, Kimin Lee; KAIST + UC
Berkeley + NYU; arXiv cs.RO, submitted 2026-03-10; accepted to ICRA
2026, 8 pages; arXiv nonexclusive distribution license. **No code,
no project page** — none linked in the paper, none found by search
as of 2026-08-10.)*

**The paper in plain words.** When you teach a robot by showing it
recordings of a task, some recordings help and some quietly poison
the lesson — a fumbled grasp, a detour, a demonstration of the
wrong thing entirely. This paper's question: can you find the bad
ones automatically, without ever running the robot? Their answer
uses "influence functions," a classical statistics tool that asks,
for each training example, *if I nudged the model using this
example, would it get better or worse at a small set of examples I
trust?* The trusted set is the whole trick: just 10–20 held-out
demonstrations that a human has verified are good. Score every
frame of every training recording by how well its learning signal
(its gradient) points in the same direction as the trusted set's,
and two refinements make the score usable. First, compare each
frame against the *single most similar* trusted moment rather than
the average of all of them — a good grasp frame should be credited
for matching a trusted grasp, not diluted by trusted frames of
reaching and carrying. Second, don't keep or discard individual
frames; average the score over each full recording and keep or
drop recordings whole, which preserves the connected arc of a
demonstration. On a simulated can-in-bin task where they salted
the training set with 50% deliberate failures, the method fishes
out the good half almost perfectly (99.4% of what it keeps is
good) and the retrained policy jumps to 99.2% success. On a real
arm with 40% bad demos mixed in, retraining on the curated subset
hits 86.7% versus 36.7% for training on everything. The catch, and
it's a real one: every headline number comes from datasets where
the authors *injected* the failures, in single tasks, at
200–500-recording scale, and the one test on naturally messy data
(DROID) only checks whether the scores rank recordings correctly —
they never retrain a policy on it. For us the appeal is exactly
what it doesn't need: no robot rollouts, just held-out
demonstrations — which is the shape of evaluation we already have.

## What it contributes

- **QoQ-score: Hessian-free influence with max-over-validation
  matching.** Start from the standard influence function
  `I(x_i) = −∇L(D_val)ᵀ H⁻¹ ∇L(x_i)`, drop the Hessian and
  normalize gradients (the TracIn first-order recipe), giving
  cosine similarity between per-sample gradients. Then the paper's
  first novelty: score each training state-action pair against the
  **maximum** over validation pairs,
  `QoQ(s,a) = max_{(s',a')∈D_val} g(s',a')ᵀ g(s,a)` with
  `g = ∇_θ log π_θtr(a|s) / ‖·‖`, instead of summing over the
  whole validation set. Their own framing of the difference: with
  a success-only validation set, the summed version *is* the CUPID
  score — the max is claimed to cut noise from irrelevant
  validation transitions (a grasp frame shouldn't be judged
  against reaching frames).
- **Trajectory-level aggregation:** mean of QoQ-scores over each
  trajectory, then select top-N whole trajectories. Rationale:
  frame-level selection piles up redundant high-scoring moments
  (everyone's grasp frames) and shreds state coverage.
- **Practical cost tricks:** gradients computed only on a subset
  of layers (vision encoders excluded — their ablation shows
  action-head-only gradients *match or beat* full-model
  gradients), plus OPORP one-permutation-one-random-projection
  gradient compression. No wall-clock or GPU-hour numbers are
  given anywhere — a real gap for a method whose selling point is
  practicality.
- **Pipeline shape:** train a reference policy on *all* data →
  compute gradients at that single final checkpoint → score →
  select top-N trajectories → retrain from scratch on the subset.
  So curation costs one full training run plus a scoring pass
  before it saves you anything.
- **A rollout variant exists but is optional:** for the cabinet
  task they build the validation set from 20 policy rollouts (5
  success, 15 fail) and give failed rollouts *negative* weight —
  the bridge case toward CUPID/ATHENA territory. Everywhere else,
  validation = 10–20 human-verified held-out successful demos.
  This is the load-bearing fact for us.

## The experiments they actually ran

- **Sim (Robomimic can task, Franka; small transformer policy +
  ResNet encoder):** training set = 100 successful + **100
  deliberately failed** trajectories; validation = 10 successful.
  Curation accuracy (fraction of kept data that is truly from
  successful trajectories): **QoQ 99.4%±0.3** vs Behavior
  Retrieval 67.8, Flow Retrieval 56.9, all-data base rate 55.4.
  Policy success after retraining: **99.2%** vs Flow Retrieval
  76.0, Behavior Retrieval 60.0. Means over 5 runs.
- **Real (Franka FR3, GR00T N1 fine-tuned with LoRA, 20k steps):**
  banana grasping with 60 successful + **40 injected failures**;
  validation = 10 successful. Curation accuracy 83.6%±0.8;
  success **86.7%±8.9** vs Behavior Retrieval 56.7±17.6.
  Multi-object pick-and-place: **93.3%** vs all-data 36.7 and
  Behavior Retrieval 20.0. Cabinet (rollout-validation variant):
  curated beats all-data (figure-only, no table number). Means
  over 3 runs — n is small and the ±s show it.
- **DROID (the only naturally-contaminated data):** 200 pen/pencil
  pick-and-place trajectories (133 success / 67 fail as found in
  the wild), 20 held-out successes as validation, GR00T N1 50k
  steps. QoQ gets the best curation accuracy (~75–80% by figure
  read — well below the 99.4% of the injected-failure sim).
  **No policy is retrained and evaluated on the DROID selection**
  — the in-the-wild evidence stops at ranking accuracy.
- **Ablations (the good part):** max-vs-mean validation scoring —
  mean drops both accuracy and success (and since mean ≈ CUPID's
  score by their own equivalence, this doubles as the only, proxy,
  comparison to influence-function kin). Trajectory-level vs
  per-frame selection — trajectory wins clearly.
  Gradient-layer choice: action-head-only 83.6%±1.3 vs full model
  82.1%±1.2 — the cheap version is the best version. Selection
  consistency across seeds (Kendall's W): QoQ 0.77 vs Behavior
  Retrieval 0.33.
- **The budget lands on a confession:** sweeping how many
  trajectories to keep (banana task): keep 10 → 36.7%, keep 20 →
  63.3%, keep 40 → 60.0, keep **60 → 86.7%**, all 100 → 36.7%.
  The peak is exactly at the number of true successes in the pot —
  the method ranks well but you still have to know (or sweep)
  where to cut, and cutting too deep costs coverage.
- **What's missing:** no comparison against *any* prior
  influence-function method run as an actual baseline (CUPID,
  DataMIL, DemInf are discussed, never head-to-head; ATHENA is
  not cited); no compute costs; no multi-task or cross-embodiment
  experiment (assumed shared embodiment, acknowledged); no test
  where contamination is subtle rather than binary
  success/failure; largest dataset touched is 200 trajectories.
  Every policy-level win is measured against contamination the
  authors put there themselves.

## What transfers to us — and what doesn't

- **The anchor is the right shape for our regime.** This is the
  first influence-curation paper we've read whose validation
  signal is *held-out demonstrations, not rollouts*. ATHENA needs
  success/failure rollouts of a trained policy; we have none. QoQ
  needs a trusted held-out set — which is structurally what our
  panel already is. Gradient cosine against panel chunks is
  computable today on the 40k/60k expert checkpoints, expert-only
  gradients (their own ablation blesses exactly that restriction —
  and our trunk is frozen anyway, so "action-head-only" is forced
  and free).
- **But our panel is not their validation set.** Theirs: 10–20
  *verified-successful, single-task, same-scene* demos. Ours: an
  unlabeled heterogeneous held-out slice of community episodes —
  held out for distribution match, never human-verified for
  quality. QoQ against our panel answers "which training episodes
  point the same way as typical held-out episodes," not "which
  are good." If junk modes (lag, desync, sloppy teleop) are
  equally present in the panel, influence will happily keep them.
  Mitigation: the max-over-validation design is actually the one
  thing that makes a heterogeneous panel usable at all (each
  training chunk is matched to its most similar panel chunk, not
  the panel average) — but a spot-verified "clean panel" subset
  would be the honest anchor.
- **Log-likelihood gradients don't exist for a flow head — and
  didn't for theirs either.** `∇ log π` is BC-NLL notation; their
  real-robot policy is GR00T N1, a flow-matching model, so in
  practice the score is the training-loss gradient. For us:
  per-chunk flow-matching MSE gradient with shared (t, ε) draws.
  Same surrogate ATHENA needed, minus the Hessian machinery.
- **Contamination regime mismatch is the big unknown.**
  Their gains are proven at 40–50% injected binary failures.
  community_curated_v0 is pre-curated community data — our
  contamination is real but subtler (Qwen-RobotManip's DA-check
  found 81% broken proprioception in RoboMIND-UR; ours is that
  *class* of corpus but already once-filtered). The DROID result
  (~75–80% ranking accuracy, no retrain) is the closest analog to
  our setting and it is the paper's weakest.
- **Scale is untested.** 200–500 trajectories vs our ~52.5k
  episodes / 229 h. Scoring cost scales linearly and is bounded
  (one backprop pass with expert-only grads + compression;
  frames subsampled per episode) — our estimate is
  single-GPU-day class, but the paper gives zero cost data to
  check against.

## Hook corrections

Banked hook: *"influence functions with max-over-validation
scoring + trajectory-level aggregation rank demos; consistent
sim+real gains over prior selection — a principled per-episode
weighting computable against our held-out panel; the #9 curation
lever."*

1. **Mechanism: verified, and both pieces are ablated.**
   Max-over-validation and trajectory-mean aggregation are exactly
   the two contributions, and each is shown to beat its
   alternative (mean scoring, per-frame selection). Hook right.
2. **"Consistent sim+real gains over prior selection" needs a
   loud asterisk.** The baselines beaten are *similarity-retrieval*
   methods (Behavior Retrieval, Flow Retrieval) and all-data —
   never another influence-function method. CUPID/DataMIL/DemInf
   appear only in related work; the mean-scoring ablation is a
   proxy CUPID at best. And every policy-level gain is on datasets
   with 40–50% *author-injected* failures; the one naturally-dirty
   dataset (DROID) gets ranking accuracy only (~75–80%), no
   retrained policy. "Consistent" is true within a regime much
   dirtier and much smaller than ours.
3. **"Computable against our held-out panel": directionally
   right — the hook's most important claim survives.** Validation
   is offline held-out demos (10–20 of them), not rollouts; the
   rollout variant is optional. Two unpriced costs the hook
   skipped: it needs a *trained reference policy* first (we have
   checkpoints, so cheap for us), and their validation demos are
   verified-good while our panel is unverified — the anchor needs
   a spot-checked clean subset to mean "quality."
4. **"Per-episode weighting" is a slight overread:** the paper
   does hard top-N *selection*, not soft weighting; the budget
   sweep shows selection size matters a lot (36.7%→86.7% across
   cuts) and the paper gives no principled way to pick it. Scores
   could be used as sampling weights, but that arm is ours, not
   theirs.
5. Not in the hook but radar-relevant: **no code, no project
   page** — reimplementation from equations (which are simple:
   normalized grad dot products) is the only path.

## What it feeds

- **Idea #9 (data levers) — this is now the middle pole of the
  curation axis, and the axis is resolved.** Qwen-RobotManip =
  zero-GPU offline state-action heuristics, no model in the loop.
  ATHENA = influence anchored to policy rollouts we don't have,
  at π-0 scale, no code. **QoQ sits exactly between: model-based
  and principled like ATHENA, but anchored offline like Qwen** —
  and it's the only one of the three whose anchor (held-out
  demos) we already possess. ATHENA's real-robot lesson (length
  heuristic < random) said heuristic gates need an
  influence-shaped check; QoQ is the cheapest influence shape
  that runs in our regime.
- **The concrete cheapest arm:** (0) *Zero-GPU-ish gate first:*
  spot-verify ~20 panel episodes as a clean anchor subset;
  (1) scoring pass — existing trained expert checkpoint, per-chunk
  flow-loss gradients on expert params only (last layers if
  needed), OPORP-compress, max-cosine vs clean-anchor chunk
  gradients, mean per episode → one score per corpus episode
  (single-GPU-day class, our estimate); (2) *sanity gate before
  any training:* inspect the ranking — bottom-decile episodes
  should be visibly worse on spot-check and should correlate with
  the Qwen stage-1–3 offline flags (jerk/DA/quantile); a
  degenerate or uncorrelated ranking kills the lever for free;
  (3) only then one paired 40k arm — top-~70% episodes vs
  same-size random subset, same seed policy, panel chunk-MAE with
  CI95 as the gate. Their budget-sensitivity result says do NOT
  cut deep on the first arm; 70% retention is the conservative
  read of their sweep.
- **Standing caveat to log with the arm:** any gain we see is
  evidence for influence curation *in a low-contamination,
  heterogeneous, multi-task regime the paper never tested* —
  a null here would not falsify QoQ, and a win here would be a
  result the paper doesn't actually contain.
- **Radar candidates surfaced (unverified ids):** CUPID
  [2506.19121] (the rollout-anchored ancestor both this and
  ATHENA build on — released project page exists); DataMIL
  [~2505.09603] datamodels for robot IL; 2604.23000
  smoothness-driven data-quality metrics (offline, heuristic
  pole); 2510.18137 "Quality Over Quantity: Curating
  Contact-Based Robot Datasets" (same-title collision,
  contact-focused curation).
