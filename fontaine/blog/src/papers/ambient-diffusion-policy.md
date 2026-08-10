# Ambient Diffusion Policy: keep the bad demos but ban them from mid-range noise levels — the +33% is tower height on 20 trials, and you must hand the method the good/bad split yourself

*Read 2026-08-10 (lit slice `lit-radar-0822`, priority 1). Paper:
[2606.12365](https://arxiv.org/abs/2606.12365) — "Ambient Diffusion
Policy: Imitation Learning from Suboptimal Data in Robotics" (Adam
Wei, Nicholas Pfaff, Thomas Cohn, Arif Kerem Dayı, Constantinos
Daskalakis, Giannis Daras, Russ Tedrake; MIT, Pfaff/Cohn equal
contribution; arXiv v1 2026-06-10, 14 pp main + 52 pp total, arXiv
perpetual non-exclusive license — no venue listed on the abstract
page; the radar's "RSS demos spotlight" tag is unverified. Project
page [ambient-diffusion-policy.github.io](https://ambient-diffusion-policy.github.io/)
fetched HTTP 200: "Code coming soon", "YouTube coming soon" — no
GitHub repo exists yet (web search confirms none); only extras are a
slides.com deck, HTTP 200.)*

**The paper in plain words.** Every robot lab has a small pile of
demonstrations it trusts and a much bigger pile it doesn't — shaky
recordings, data from other labs, other tasks, other robots. The
standard choices are to throw the bad pile away (wasteful) or mix it
all together (the robot picks up the bad habits). This paper offers a
third option built on how these "denoising" policies are trained: the
network repeatedly takes a real motion, buries it under a chosen
amount of random static, and learns to dig the motion back out. A key
quirk is that heavy static erases fine detail first — under a lot of
static, only the broad sweep of a motion survives, and under a little
static only the fine detail is still at stake. So a jerky demo of the
right overall behaviour is still perfectly good teaching material at
high static (the jerkiness is buried anyway), and a smooth demo of
the *wrong* behaviour is still useful at low static (only local
finesse is being learned there). The recipe: measure, per data
source, how much static it takes before good and bad data become
indistinguishable — a small classifier network does this
automatically — then let bad data teach only outside the forbidden
middle band. It is a one-line change to the training data sampler.
The results are strong: a maze policy hits 99.5% success while
staying twice as smooth as naive mixing; a block-sorting policy
trained mostly on wrong-way-around demos scores 93.3% where mixing
collapses to 22.7%; and on a real robot fed 48 public datasets of
wildly mixed quality, it cleans tables about 12 points better than
mixing and stacks towers a third taller. The catches: the headline
"+33%" is tower height, not success rate, from 20 trials; the method
never finds the bad data for you — you must supply the good/bad
partition up front; and all evidence is real-robot or simulator
rollouts, with theory proven only for Gaussian toy models.

## What it contributes

- **Time-banded data admission.** Two datasets: trusted
  D_p ~ p, suboptimal D_q ~ q. Standard Diffusion Policy trains
  denoisers h_θ(A_t, O, t) on A_t = A_0 + σ(t)Z, minimizing
  E‖h_θ(A_t,O,t) − A_0‖². Ambient keeps the loss and changes only
  the sampler: D_p samples are admissible at all t; D_q samples only
  at t ∈ [0, t_max) ∪ (t_min, T]. Inference is unchanged. Paper's own
  framing: "a single change to Diffusion Policy's data sampler."
- **High-t end (t > t_min): contraction through noise.** Noise
  erases p-vs-q differences; past t_min the noisy marginals p_t ≈ q_t
  and D_q supervision is unbiased for the *global* plan.
- **Low-t end (t < t_max): locality.** At low noise the optimal
  denoiser is nearly local (each action depends on nearby actions in
  the chunk), so D_q whose *local* primitives are fine but whose
  global/semantic content is wrong can still teach fine motor detail.
- **Spectral power law as the load-bearing property.** Empirically,
  action-chunk power spectral densities follow S(f) = C|f|^(−α)
  (α > 1) across OXE (2.4M episodes, 70+ datasets, resampled 10 Hz,
  horizon 100) and every dataset they tested: teleop vs scripted,
  absolute vs delta, EEF vs joint space. Theorem 1 (zero-mean
  stationary Gaussians agreeing below f*, power-law tail above): once
  σ_t² ≥ C(f*)^(−α), d_TV(p_t, q_t) ≤ √2·C / (σ_t²·√((2α−1)(f*)^(2α−1)))
  — much faster contraction than the generic
  d_TV(p,q)·D/(2σ_t) bound. Theorem 2 bounds the optimal denoiser's
  error when masking actions > L steps away by αN‖x‖_∞/(8L), i.e.
  power law ⇒ locality.
- **Classifier annotation of t_min (Phase 1).** Train c_φ(A_t, t) on
  noisy action chunks only (no images needed) to tell p_t from q_t;
  set t_min = inf{t : E_q[c_φ*(A_t,t)] > 0.5 − τ}, τ = 0.05 (Eq. 5).
  Theorem 4 shows a well-trained classifier certifies p_t ≈ q_t
  beyond t_min. Alternative: brute hyperparameter sweep (needs
  rollouts). t_max was mostly hand-set or swept; the Daras et al.
  classifier variant assigns t_max = 0 to nearly all OXE datasets.
- Presented in variance-exploding notation; the implementation is
  the original variance-preserving Diffusion Policy (U-Net, ResNet18
  encoder, cosine schedule, T = 100). Appendix B gives the VP↔VE
  change of variables; the loss footnote notes the denoising loss
  "or a reparametrization of it, as in flow matching."

## The experiments they actually ran

- **2D maze (noise mismatch), 1000 trials.** D_p = 50 smooth GCS
  trajectories, D_q = 5,000 jittery RRT. Success/smoothness (mean sq.
  accel, successful rollouts only): filtering 57.5%/31.9; co-training
  (α* = 0.019) 99.4%/62.2; Ambient (σ_tmin* = 0.074) 99.5%/31.0 —
  co-training's success at half its jerk. Ablation: training D_q
  *only* at t > t_min and D_p only below matched it (99.0%/29.9),
  direct evidence for the hierarchy.
- **7-DoF neural motion planning, 1000 trials.** Filtering 46.0%/3.9;
  co-training (α* = 0.091) 59.9%/42.7; Ambient (σ_tmin* = 0.025)
  65.9%/31.4.
- **Planar pushing sim-to-real proxy, 200 trials** (Wei et al.
  sim-and-target datasets: 50 teleop target demos vs 2,000 planner
  trajectories). Filtering 56.5%; co-training 84.5%; Ambient with
  per-dataset t_min (sweep) 87.0%; per-datapoint t_min (classifier)
  93.5%; + locality (σ_tmax = 0.025) 92.0%. Finer annotation
  granularity is worth +6.5 points.
- **Block sorting (task mismatch), ~800 blocks.** D_p = 50 correct-
  sorting demos, D_q = 200 *opposite*-sorting; t_min = 0, sweep
  t_max. Logic/Motion/Success: filtering 98.6/61.9/61.0; D_q-only
  3.0/83.0/2.5; co-training (α* = 0.9) 26.0/87.2/22.7; Ambient
  locality (σ_tmax* = 0.46) 98.2/95.0/93.3. Exclusive-band ablation
  again matches (97.9/93.8).
- **Finetuning comparison.** Finetuning an Ambient base beats
  finetuning a co-trained base; the un-finetuned Ambient policy often
  beats the finetuned co-trained one; finetuning Ambient doesn't
  always help. On OXE, finetuning produced no statistically
  significant change.
- **Real-robot OXE scaling** (VR-teleop D_p; D_q = Magic Soup++, 27
  OXE datasets, or Custom OXE, 48 datasets; t_min via 48 per-dataset
  classifiers; t_max hand-set at 10/5, explicitly *not* swept —
  "a full hyperparameter sweep on real hardware would have been
  prohibitively time-consuming"). Table cleaning, 50 demos: filtering
  68.2%; co-training +2–3 points and *plateaus* MS++ → COXE; best
  Ambient beats co-training by 12% and keeps improving with more
  suboptimal data. 150 demos: filtering rises to 80.1%; Ambient still
  up to +10% over co-training; locality no longer helps. Tower
  building, 35 demos, 20 trials/policy: towers up to 84% taller than
  filtering, 33% taller than co-training. Qualitative: 25–40% fewer
  grasps per object than co-training.
- **Re-weighting ablation.** Without dataset re-weighting Ambient
  degrades ≤ 9%; unweighted co-trained policies were "too dangerous
  to evaluate" on hardware.
- **Not covered:** no ablation of τ, classifier architecture, or
  noise schedule; t_max never classifier-annotated at scale;
  observation-space shift unsolved (both Appendix J attempts —
  observation noising, classifier-free guidance — didn't help);
  theory Gaussian-only; "best Ambient" on hardware implies some
  model selection via real rollouts; 20-trial real-robot samples are
  small; no comparison against influence-function or scored-curation
  baselines (Hejna et al. cited, not run).

## What transfers to us — and what doesn't

- **The lever shape is exactly what the radar guessed:** a
  training-recipe change on the noise/flow *time* axis, orthogonal
  to episode-level curation. Bad data isn't dropped, it's banned
  from the middle band. Composes with, does not replace, a QoQ-style
  influence pass — in fact it *needs* a partition, and an influence
  or heuristic pass is an approved way to make one ("our framework
  applies equally well to any other definition, including heuristic
  measures or human labels").
- **Flow-time correspondence: transfers in principle, with one
  precision.** Everything the argument uses depends only on the
  corruption marginals: A_t = A_0 + σ_t Z with Gaussian Z and
  monotonically increasing noise scale. Our rectified-flow
  interpolant x_t = (1−t)A_0 + tZ has marginals equal (up to a
  1/(1−t) rescale, exactly the paper's Appendix B VP↔VE change of
  variables) to variance-exploding noising with σ̃(t) = t/(1−t) —
  monotone, spanning [0, ∞). So the band-mask ports directly as a
  data-sampler mask on flow time t ∈ [0,1], and the classifier
  annotation ports by comparing noisy chunks at matched σ̃. The
  precision: thresholds live in *noise-scale* space (the paper
  reports σ_tmin, not raw t) — map σ_tmin* through σ̃(t), never copy
  a t value across schedules. Heun decode is untouched; inference
  identical.
- **Positions-only is fine.** The power law was verified in joint
  space and EEF space, teleop and scripted; the classifier eats
  action chunks only — no force/current channel needed, unlike the
  FACTR2-style contact gates.
- **The mismatch: their evidence is rollouts, our panel is
  chunk-MAE.** Their gains show up as success rate and smoothness.
  A policy that correctly *stops imitating* jitter in bad datasets
  can score worse chunk-MAE against those datasets' own jittery
  ground truth — the same eval confound already flagged for the
  velocity-debias lever. Any arm must be read on a trusted-subset
  MAE split, not corpus-wide MAE.
- **Their D_p is task-specific (35–150 demos on the target rig); we
  have no target rig data yet.** Our nearest anchor is the QoQ
  trusted held-out set (10–20 demos). The paper itself flags the
  generalist-pretraining case as future work needing "a more
  principled understanding of data quality" — we'd be in exactly
  that untested regime.
- **Cheapest first arm (offline, no policy training committed):**
  Phase-1 only. (1) Compute action-chunk PSDs on community_curated
  _v0 — does the power law hold on our corpus, and with what α?
  (2) Train small chunk classifiers (trusted anchor vs a handful of
  best/worst datasets) and read off the σ_tmin distribution: if most
  datasets get σ_tmin ≈ 0, the lever has nothing to grip; if it
  spreads, a single flow-head retrain with the band-masked sampler
  becomes a justified arm, judged on trusted-subset chunk-MAE.

## Hook corrections

- **"Useful signal only at high/low diffusion times via a spectral
  power law" — broadly confirmed, direction sharpened.** The method
  admits D_q at t ∈ [0, t_max) ∪ (t_min, T]. But the two ends serve
  different failure modes: high-t (past t_min) is for *local/high-
  frequency* corruption like jitter; low-t (below t_max) is for
  *globally/semantically* wrong data with good local primitives. The
  power law is the empirical property (OXE, 2.4M episodes) that
  makes contraction fast (Thm 1) and denoisers local (Thm 2) — but
  theorems hold for zero-mean stationary Gaussians only.
- **"+33% over naive co-training" — number real, metric not what it
  sounds like.** The 33% is *tower height* ("up to 84% and 33%
  taller than the data filtering and co-training baselines"), 20
  trials/policy. Table-cleaning success gains are +12% (50 demos)
  and up to +10% (150 demos) over co-training; Fig. 9's caption says
  "up to 15% on table cleaning and 84% on tower building" vs both
  baselines. No task shows a 33-point success-rate gap on OXE.
- **"Purely offline" — half right.** The *annotation* can be fully
  offline (classifier on noisy action chunks; 48 classifiers for
  COXE; no rollouts). But the alternative annotation is a rollout
  sweep, the paper admits the classifier "does not always outperform
  a (costly) hyperparameter sweep", and every reported result is a
  rollout metric with some on-hardware model selection ("best
  Ambient"). No offline proxy metric appears anywhere in the paper.
- **"How is 'suboptimal' designated?" — user-supplied partition,
  dataset provenance throughout.** No automatic quality scoring, no
  per-sample labels: GCS vs RRT, real vs sim, correct vs opposite
  sorting, my-50-demos vs all-of-OXE. "Ambient Diffusion Policy
  requires the user to partition their data into D_p and D_q" — but
  the partition definition is explicitly arbitrary, so it COMPOSES
  with an influence-curation pass rather than requiring labels.
- **"Does it transfer to rectified flow?" — untested in the paper,
  but nothing blocks it.** Flow matching appears once, as a footnote
  calling the FM loss a reparametrization of denoising. The argument
  needs only additive-Gaussian corruption marginals with monotone
  noise scale — satisfied by the rectified-flow path with
  σ̃(t) = t/(1−t). Port thresholds in σ-space, not raw t.

## What it feeds

- **#9 data levers (main).** New banked lever: flow-time band-mask —
  restrict which flow times unlabeled-quality datasets supervise,
  with σ_tmin annotated offline by per-dataset chunk classifiers
  anchored to the QoQ trusted set; composes with the influence pass
  (which can *define* the D_p/D_q split it needs).
- **#9 eval-confound flag (reinforced).** Third independent case
  (after velocity-debias and chunk-MAE smoothing) where "learns to
  ignore bad data" can *hurt* corpus-wide chunk-MAE — trusted-subset
  MAE split moves from nice-to-have toward prerequisite.
- **Corpus characterization rider.** The PSD power-law check on
  community_curated_v0 is a cheap CPU work item with standalone
  value: it tests this paper's core empirical premise on our own
  880 datasets before any training arm is bought.
