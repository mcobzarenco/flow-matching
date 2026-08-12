# Sim-as-eval: what a simulator is allowed to claim about a real policy

**Papers:** SIMPLER
([2405.05941](https://arxiv.org/abs/2405.05941)) · AutoEval
([2503.24278](https://arxiv.org/abs/2503.24278)) · SureSim
([2510.04354](https://arxiv.org/abs/2510.04354)) · Beyond Binary
Success ([2603.13616](https://arxiv.org/abs/2603.13616)) · A Practical
Recipe Towards Improving Sim-and-Real Correlation for VLA Evaluation
([2606.10366](https://arxiv.org/abs/2606.10366)) · REALM
([2512.19562](https://arxiv.org/abs/2512.19562)) · periphery: WorldEval
([2505.19017](https://arxiv.org/abs/2505.19017)), Veo world-simulator
eval ([2512.10675](https://arxiv.org/abs/2512.10675)), sim-to-real
benchmarking position paper
([2508.11117](https://arxiv.org/abs/2508.11117)).
**Read:** 2026-08-11, sim lit lane (owner directive 17:07Z re-opened
the lit pause for sim topics). **Fed:** the
`sim-policy-eval-100seeds` protocol pre-reg (this page is its design
citation), idea #16 (the proxy question), and the
[offline-validation](offline-validation.md) page's proxy-audit frame
(SIMPLER/AutoEval get their full treatment here).

**The question in plain words.** We are about to start scoring our
policies by running them in a home-made simulation of our robot arm
and measuring how far they push a toy boat toward a disk. The obvious
worry: the simulator is not reality — its grip physics are imperfect
(we just measured exactly how), its pictures are renders, its motors
are idealized. So what is a sim number actually *worth*? A line of
papers from 2024 to 2026 has measured precisely this: build a
simulated mirror of a real robot setup, run the same policies in
both, and check whether the simulator puts policies in the right
order. The short answer: yes, a deliberately imperfect simulator can
rank policies almost perfectly — *if* you fix the motor model against
real trajectories, paste real camera backgrounds into the render, and
validate the mirror once against a handful of real trials. And there
is a sharp caution: the ranking holds for policy families the mirror
was validated on, and can fail catastrophically (0% in sim, 94% in
reality) for a new one.

## SIMPLER: the founding measurement

SIMPLER (2405.05941, CoRL 2024) built simulated mirrors of two real
setups — the Google Robot RT-1 lab cell and a WidowX + BridgeData V2
bench — and made a point of *not* chasing full digital-twin fidelity.
They attack exactly two gaps:

- **Control gap → system identification.** Fit the arm's PD
  controller stiffness and damping per joint against a small set of
  real trajectories, minimizing a summed translation error
  (mean end-effector ‖Δx‖) plus rotation error (arcsin of the
  Frobenius gap between rotation matrices), via three rounds of
  simulated annealing with shrinking search ranges.
- **Visual gap → green-screening, not photorealism.** "Visual
  matching": composite the sim-rendered arm and objects over *real
  photographs* of the actual bench from the fixed camera, and project
  real object textures onto the sim assets. The laborious
  alternative — "variant aggregation," averaging over many sim scene
  variants — was also built, and lost.

They scored fidelity with two metrics. **Pearson r** between sim and
real success rates, and — the keeper — **MMRV, mean maximum rank
violation**: for every pair of policies the sim mis-orders, charge
the *real-world* success-rate margin between them, then average each
policy's worst such charge. A rank flip between two near-tied
policies costs almost nothing; a flip across a large real gap is the
failure that matters.

The numbers, on six policies spanning weak to strong (RT-1 at three
training stages, RT-1-X, RT-2-X, Octo): visual matching averaged
**MMRV 0.056 / Pearson 0.924** across the Google-robot tasks, beating
variant aggregation (0.143 / 0.778) on both metrics. On the WidowX
tasks, all policies ranked correctly for all but one task. Real-side
sample sizes were small — 24 to 75 trials per task, ~1,500 sim
episodes total. Stated limits: rigid objects only, fixed cameras
only, no shadows from the composite, heavy manual asset curation for
articulated objects (the drawer task is exactly where variant
aggregation collapsed to r = 0.486).

## AutoEval: the caution about new policy families

AutoEval (2503.24278) automated *real* evaluation instead — a
fine-tuned VLM success classifier plus a learned reset policy on a
WidowX cell, matching human-run evals at Pearson 0.942 / MMRV 0.015
with ~850 episodes per 24 h and roughly three human interventions a
day. We have no fleet, so the infrastructure isn't the lesson. The
lesson is their measurement of SIMPLER-style sim on a policy family
it wasn't validated on: **Open-π0 scored 0/50 in the simulated sink
task and 47/50 on the real one.** A twin validated on policies A–D
can be confidently, catastrophically wrong about policy E. Sim-real
fidelity is a property of the (simulator, policy-family) pair, not of
the simulator.

## SureSim: how to use a biased simulator honestly

SureSim (2510.04354) is the statistically principled synthesis:
treat sim as a *biased predictor* of real performance, run a large
sim eval plus a small paired real+sim sample, and use
prediction-powered inference to rectify the bias — yielding
non-asymptotic confidence intervals on the real quantity. On
diffusion policies and fine-tuned π₀ they report ~20–25% less real
hardware burden at equal interval width. This is the eventual shape
of our own pipeline once rig rollouts exist: the 100-seed sim panel
does the heavy lifting, a small real batch anchors it, and the claim
is a debiased CI, not a raw sim number.

## Continuous progress beats binary success at small n

Beyond Binary Success (2603.13616) is the direct precedent for the
owner's metric choice. Using safe anytime-valid inference for
sequential policy comparison, they show **competing policies separate
faster on fine-grained task-progress metrics than on binary
success** — up to 70% fewer trials than fixed-batch testing and up to
50% fewer than binary-outcome sequential methods, on both sim and
real data. AutoEval independently lists binary-only scoring as a
limitation of its own system. Notably, none of this lineage shows
continuous metrics correlate better *with real outcomes* — the
demonstrated win is statistical power at small n, which is exactly
the regime of a 100-seed panel. Our boat→disk distance-reduction
primary read is well-precedented; success rate should ride along as a
secondary column so MMRV/Pearson stay computable against any future
real panel.

## The 2026 meta-lesson: simulator choice changes conclusions

The Practical Recipe paper (2606.10366) ran the head-to-head:
three simulators (REALM on Isaac, VLA-Arena on MuJoCo, SIMPLER on
SAPIEN) against the same real DROID evals of π₀/π₀-FAST/π₀.5/GR00T
policies. Ranking fidelity diverged sharply by simulator — Spearman
**0.700 / 0.575 / 0.400** and MMRV **0.030 / 0.060 / 0.128**
respectively — so which simulator you ask changes which policy wins,
and they attribute the spread to *simulator-level fidelity*, not
task mismatch. Two more results worth keeping: fine-tuning the
policy on just 10 sim demos per task improved alignment (Spearman
0.700 → 0.875) while 20 demos *degraded* it, and only REALM
preserved the real-world ordering of perturbation severities. REALM
(2512.19562) itself — 15 perturbation factors, 3,500+ objects,
π₀/GR00T-class policies — positions sim explicitly as a
*weakness-finding* proxy rather than a real-world equivalent. And the field's far end is
already evaluating policies inside generative video models instead of
physics engines (WorldEval 2505.19017; Gemini's Veo evaluator
2512.10675, validated against 1,600+ real trials) — not replicable at
our scale, but a signal that "the evaluator is a model too" is now a
mainstream position. The RSS'25 position paper (2508.11117) distills
the checklist we should cite in the protocol: high visual fidelity
where it's cheap, systematic perturbation ramps, and *explicit*
quantification of sim-real alignment rather than assumed transfer.

## What transfers to us

- **SysID before seeds.** SIMPLER's single most transferable move is
  fitting the controller against real trajectories. We have 229 h of
  real SO-101 data on the hub and a sim whose servo model is
  menagerie-default. Fitting actuator gain/damping (and the
  home-pose offsets the sim review flagged) against a few real
  episodes, with SIMPLER's translation+rotation loss, is a
  CPU-cheap, pre-registerable step — and our servos are noisy
  low-cost STS3215s, which makes the control gap *larger* than on
  the lab arms SIMPLER tuned. Their ablations put numbers on the
  priority order — controller gains first-order, friction values
  second-order — detailed on the
  [contact-fidelity page](sim-contact-fidelity.md).
- **The validation experiment is checkpoints, not policy families.**
  SIMPLER's six policies spanned weak-to-strong partly by using one
  model at three training stages. We can mirror that for free: run
  the 100-seed panel on early/mid/endpoint checkpoints of the same
  trunk (e.g. er_60k at 15k/35k/60k, whose panel-MAE ordering is
  banked) and check the sim orders them the same way. Target bands
  from the lineage: MMRV ≲ 0.06, r ≳ 0.9 — with no real rollouts
  yet, ordering-vs-panel-MAE is the available half of the check.
- **Distance-reduction primary, success secondary** — precedented
  (2603.13616), and our sim review found `success()` has latching
  quirks anyway; the continuous read is also robust to the
  reset-settle displacement once initial distance is read
  post-settle.
- **The AutoEval caution lands directly on us.** Our first sim
  panel scores our own trunk family; the moment we score a
  *different* stack (MolmoAct2, a flow variant with different
  chunking), the twin's fidelity claim resets to zero until
  spot-checked. Write this into the protocol as a standing caveat.
- **SureSim names our end-state.** When owner-rig rollouts exist,
  the sim panel + small paired real batch → debiased CI is the
  claim structure; parked with #16's real half.

## What doesn't transfer

- **Green-screen visual matching, for now.** Our policies consume a
  top camera and a *wrist* camera; SIMPLER's compositing works for
  fixed cameras only. A real-background composite for the top view
  is plausible later; the wrist view will stay fully rendered, and
  the sim review already verified the render path is deterministic —
  the visual-gap *magnitude* is simply unmeasured until we have real
  rollouts to compare against.
- **Fleet-scale anything.** AutoEval's reset policies, RoboDojo-class
  real cells ([2607.04434](https://arxiv.org/abs/2607.04434)),
  1,600-trial validation sets — the shapes are instructive, the
  scale is not ours.
- **Generative-video evaluators** — noted as field direction, not
  tooling we touch this year.
