# Is Diversity All You Need?: the "expert diversity" that hurts is velocity spread — worth 2.5x data to remove, and never actually ablated by operator

*Read 2026-08-09/10 (lit slice `lit-radar-0820`, priority 3). Paper:
[2507.06219](https://arxiv.org/abs/2507.06219) — "Is Diversity All You
Need for Scalable Robotic Manipulation?" (Modi Shi, Li Chen, Jin Chen,
Yuxiang Lu, Chiming Liu, Guanghui Ren, Ping Luo, Di Huang, Maoqing
Yao, Hongyang Li; OpenDriveLab / AgiBot; arXiv v1 2025-07-08, v2
2026-06-04; license nonexclusive-distrib/1.0; code pointer
[github.com/OpenDriveLab/AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World)
verified live — but it ships base GO-1 / GO-1 Air only; **no GO-1-Pro
checkpoint and no velocity-debias code released**).*

**The paper in plain words.** When you collect robot demonstrations,
you can spend your budget three ways: on many different tasks, on many
different robots, or on many different human demonstrators. This paper
asks which kinds of variety actually help. Two answers are comforting:
a grab-bag of many tasks beats a curated pile focused on the tasks you
care about (even when the curated pile contains *more* examples of the
skills you'll be tested on), and pre-training on one robot transfers
to other robots about as well as pre-training on twenty-two. The third
answer is the interesting one: variety among the humans is partly
poison. Different people drive the arm along the *same path* at
*different speeds*, so the same camera image gets labeled with
conflicting "what happens in the next two seconds" answers. The fix is
almost embarrassingly simple: learn a small model of how fast the
robot is expected to move, then stretch or squash each training
snippet in time so every demonstration moves at the expected speed —
keeping genuinely different strategies (go left vs go right) while
erasing the fast-person/slow-person noise. On four real household
tasks this time-rescaling was worth a 15% relative score gain, the
same boost as collecting 2.5x more pre-training data. The caveats:
the "diversity hurts" story was never tested by actually varying the
number of demonstrators, the gain was measured on one model family,
and the recipe's code was never released.

## What it contributes

- **A three-axis empirical study of data diversity** (task,
  embodiment, expert) run at real scale: pre-training pools sampled
  from AgiBot World Beta (1M+ bimanual trajectories, single AgiBot G1
  embodiment, professional teleoperators) and OXE (2.4M trajectories,
  22 embodiments), evaluated in ManiSkill and RoboTwin sim and on
  real AgiBot G1 and AgileX Cobot Magic (Piper arm) robots.
- **Task diversity > per-task depth.** At identical 10% data volume,
  episode-based sampling (uniform over all tasks, maximal variety)
  beats task-based sampling (10% of tasks hand-picked for downstream
  relevance) by +0.10 average score across four real tasks — despite
  the episode-sampled pool containing *fewer* episodes of the target
  atomic skills (59.2% vs 71.1% coverage). Biggest wins: Make
  Sandwich +0.26, Pour Water +0.14.
- **A pre-training scaling law on real tasks:** GO-1 average score
  0.28 (no pre-training) → 0.47 (100K demos) → 0.53 (250K) → 0.58
  (1M); fitting optimality gap (1 − normalized score) vs demo count
  gives y = 1.24·x^−0.08, r = −0.99. The −0.08 exponent is shallow:
  each doubling of data buys little, which is what makes the
  debiasing result valuable in data-equivalent terms.
- **Embodiment diversity is optional.** RDT pre-trained on
  single-embodiment AgiBot World (RDT-AWB) matches RDT-OXE on
  ManiSkill at 250 fine-tuning demos and pulls ahead as fine-tuning
  data grows (125/250/500/1000 demos/task sweep); on the real
  never-seen AgileX Piper rig, RDT-AWB 0.45 vs RDT-OXE 0.40 average
  over four tasks. Quality/consistency of one embodiment beat the
  22-embodiment zoo.
- **Velocity-ambiguity diagnosis and a debiasing recipe.** Expert
  diversity = "distributional variations ... arising from different
  teleoperators' habits, skill levels, and inherent randomness." The
  paper splits demonstration multimodality into *spatial* (different
  strategies — keep: "meaningful task strategies that should be
  retained") and *velocity* (same path, different speed — remove:
  "undesirable noise that complicates training"). Fix: (1) train a
  velocity model VM(o_t) (frozen SigLIP encoder + MLP head, MSE loss
  against realized chunk velocity, output normalized to [0,1]);
  (2) during policy training, per sample, search chunk length
  L ∈ [0.5T, 1.5T] minimizing |VM(o_t) − v(a_{t:t+L})| and
  interpolate a_{t:t+L} back to T steps. Training-time only; nothing
  changes at inference.

## The experiments it actually ran

- **Policy classes — be precise here.** The debiasing result was
  measured on **GO-1 only**: InternVL2.5-2B VLM trunk + latent action
  planner + an action expert trained with a **diffusion objective**
  (verified in the GO-1 paper, 2503.06669 — "an action expert that
  utilizes a diffusion objective to model the continuous distribution
  of low-level actions", 30-step chunks). So the harm was measured on
  a multimodality-native head, *not* a unimodal regressor. RDT
  (diffusion DiT) appears only in the embodiment section and was
  never debias-tested; no autoregressive-decode policy appears
  anywhere.
- **The debiasing table (Table II, four real tasks on AgiBot G1,
  rubric scoring 0/0.5/1 per step, ~10 trials per scenario):**
  biased pre + biased FT 0.46; debiased pre + biased FT 0.49 (+6.5%);
  debiased pre + debiased FT 0.53 (+15%). Per task: Pour Water
  0.20→0.32 (+60%), Fold Shorts 0.30→0.37, Make Sandwich 0.67→0.73,
  Wipe Table 0.66→0.70. No error bars reported.
- **The "2.5x data" arithmetic:** debiased-at-100K scores 0.53, the
  same as the *measured* biased 250K point on the scaling curve —
  so it is an interpolation between measured scaling points, not a
  direct paired data-doubling arm, and 250K/100K = 2.5. A companion
  claim: GO-1-Pro reaches GO-1's fine-tuned performance with 50% of
  the fine-tuning data.
- **What was NOT run:** no controlled single-operator vs
  multi-operator comparison (demonstrator count is never varied); no
  velocity-distribution statistics for AgiBot World (no histograms,
  no operator counts — the mechanism figure is a Push-T cartoon); no
  debiasing on RDT or any second policy class; sim evidence for the
  task-diversity claim comes from RoboTwin/ManiSkill side
  experiments, the headline numbers are real-robot rubric scores.
- **Authors' own limitations:** the rescaling "cannot be applied to
  dynamic tasks such as ping-pong where the varying velocities are
  crucial"; pauses and suboptimal segments in demos remain open
  ("could cause robots to enter infinite loops").

## Corrections to our banked hook

The hook survives in outline, needs three sharpenings:

1. **"Expert diversity actively HURTS" is an inference, not a
   measurement.** No experiment varies demonstrator count or
   compares single- vs multi-operator pools. The evidence is: a
   velocity-debiasing intervention helps, therefore velocity spread
   was hurting. And the paper explicitly protects *spatial* operator
   diversity as beneficial. Honest restatement: *velocity*
   multimodality — one component of expert diversity — costs ~15%
   in their setup; the rest of the axis was not isolated.
2. **"+15% ≈ 2.5x data" verified but scoped:** 0.46→0.53 average
   rubric score, four real tasks, GO-1 only, ~10 trials/scenario, no
   error bars; the 2.5x is read off their own scaling curve between
   measured points, not a paired arm. Debiasing pre-training alone
   is only +6.5% — the fine-tuning half matters.
3. **The debias recipe is NOT released.** The linked repo has base
   GO-1/GO-1 Air training + checkpoints (CC BY-NC-SA 4.0) but no
   velocity model, no rescaling code, no GO-1-Pro. The recipe is
   simple enough to reimplement from the paper (Section V-A), but
   nothing is runnable off the shelf.

One upgrade the hook missed: the harm was demonstrated **on a
diffusion action expert**. The comfortable prior that
multimodality-native heads are immune to velocity ambiguity is
exactly what this paper's setup contradicts.

## What transfers to us — and what doesn't

- **Our corpus is the far end of their axis.** AgiBot World is one
  embodiment, one camera rig, professional "skilled teleoperators"
  under verification protocols — and velocity multimodality was
  *still* worth 15%. community_curated_v0 is 880 datasets / hundreds
  of hobbyist operators / heterogeneous rigs, speeds, and
  calibrations — plausibly a much wider velocity spread.
  Directionally this says the effect should be bigger for us;
  the magnitude does not port (different policy, scale, eval), and
  in our corpus operator is confounded with rig, scene, and task, so
  any read must stratify by dataset.
- **Does the mechanism even apply to expressive heads? Partly — and
  that is the finding.** GO-1's diffusion expert can represent the
  multimodal chunk distribution, yet debiasing still paid. Candidate
  mechanisms that survive expressiveness: probability mass and model
  capacity spent on nuisance speed modes instead of spatial
  strategy; open-loop chunk execution splicing incompatible speeds
  at chunk boundaries; and (GO-1-specific) the latent-action
  supervision inheriting the same ambiguity. Our flow-matching heads
  sit in the same class as their diffusion expert, so "we model
  multimodality natively" is not a free pass. AR decode with
  temperature is untested by the paper — no evidence either way.
- **For us velocity spread is ALSO an eval confound, which the paper
  never faces.** Their eval is real-robot success rubrics — rollout
  success does not care how fast the policy moves. Our panel is
  offline per-frame chunk-MAE against a single ground-truth chunk:
  velocity multimodality inflates the irreducible MAE floor for
  *any* policy, and a velocity-normalized model would be scored
  against un-normalized held-out chunks and lose by construction.
  Any debias arm on our stack must transform the probe targets
  consistently or the panel will misread the result.
- **The two side findings push on our data backlog directly.** The
  task-diversity result warns against curating the corpus toward
  rig-relevant tasks (their task-based sampling lost even with more
  target-skill episodes) — relevant to every #9 filtering lever. The
  embodiment result (consistent single-embodiment ≥ 22-embodiment
  zoo for RDT) argues our SO-family-only corpus is not the handicap,
  and demotes the Bridge V2 cross-embodiment pilot ranked in the
  08-09 trajectory-datasets survey.
- **Scale gap, stated plainly:** their pre-training pools are
  100K–1M trajectories; our whole corpus is 38.6k episodes / 229 h.
  Their −0.08-exponent scaling law and 2.5x-equivalence arithmetic
  live at a scale we cannot reach by collection — which cuts both
  ways: data-equivalent gains from debiasing are worth relatively
  *more* to us, and their absolute numbers mean nothing for us.

## Which idea it feeds

**Idea #9 (data levers) — one concrete zero-GPU instrument, then a
gated screen.** Cheapest falsification chain on our corpus:

1. **Operator-speed census (zero GPU, instrument-only, joins the
   wrap-census / continuity-screen family):** per-episode mean and
   profile of |Δq| from joint deltas (6-dim @ 30 fps — the fields
   already parsed by the continuity screen), aggregated per dataset.
   Deliverables: distribution of per-dataset median speed;
   within-dataset vs cross-dataset variance ratio (the paper's
   implicit claim is cross-operator spread dominates); flag of
   heavy-spread datasets. If cross-dataset velocity spread is small,
   the whole hook dies cheaply on our corpus.
2. **Free correlation read on banked panels:** per-dataset probe MAE
   (existing eval npz breakdowns) vs per-dataset velocity dispersion
   from the census, controlling for task class. A positive
   correlation is the cheapest evidence that velocity spread is
   costing us panel points; a null at wide spread would say our
   heads/eval absorb it — a real falsification of transfer.
3. **Only if 1+2 read positive — a screen-rung debias arm:** we do
   not need their velocity model offline: the realized future is
   available at training time, so rescale chunks toward a
   *per-dataset* canonical speed (preserving within-episode speed
   profile — a global constant would destroy legitimate
   phase-dependent speed, which their conditional VM deliberately
   preserves). Non-negotiable design constraint from the eval-
   confound bullet above: the held-out probe targets get the same
   transform, or the arm is unreadable.

No pre-reg queued from this read alone: step 1 is below the
screen-rung bar and belongs as a work item; steps 2–3 are gated on
its output.
