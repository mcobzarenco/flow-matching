# PhAIL: time-to-success CDFs + macro-KS resolve close policy pairs at ~25–30 episodes/cell where binary metrics need 600+ — and the human anchor does none of the statistical work

*Read 2026-08-10 (lit slice `lit-radar-0822`, priority 4). Paper:
[2605.29710](https://arxiv.org/abs/2605.29710) — "PhAIL: A Real-Robot
VLA Benchmark and Distributional Methodology" (Sergey Arkhangelskiy;
Positronic Robotics; arXiv preprint, 22 pp, 10 figures, 8 tables,
cs.RO, submitted 2026-05-28; arXiv nonexclusive-distrib/1.0 license.
Artifacts VERIFIED by fetch: [phail.ai](https://phail.ai) HTTP 200 —
live leaderboard site ("Four leading models. One commercial task.
Production metrics."), Run Explorer at /episodes (Rerun-SDK viewer for
auditing annotations against raw video), model-submission page at
/eval, /consortium program, and the v1.0 dataset download (~180 MB
metadata-only tier; ~990 episodes with synchronized video, telemetry,
event annotations, plus the 449-episode / ~13 h fine-tuning demo set).
[github.com/Positronic-Robotics/phail-paper](https://github.com/Positronic-Robotics/phail-paper)
HTTP 200 — full analysis pipeline (`build/stats.py`: Kaplan–Meier,
RMST, bootstrap; `fig_*.py`/`tab_*.py` regenerate every paper figure
via `make figures-paper`), 461 manually-reviewed JSON annotation
sidecars, croissant.json dataset metadata, LaTeX source; 70 commits,
0 stars (fresh). Their robot stack
[github.com/Positronic-Robotics/positronic](https://github.com/Positronic-Robotics/positronic)
also HTTP 200. This is a genuinely complete release: data + eval code
+ stats code + audit tooling.)*

**The paper in plain words.** When people compare two robot policies
they usually run each one 10–30 times, count successes, and report
two percentages — and at that sample size a 5-point gap between two
percentages is statistical noise, so most published rankings of close
policies are unresolvable coin flips. This paper's move is to stop
throwing away the clock: instead of recording "succeeded or not," it
records *when* each item was successfully placed, treats runs that
time out as "still going when we stopped watching" (the same
censoring trick medical survival studies use for patients who outlive
the trial), and treats hard failures — item flung off the table,
safety stop — as successes that arrive at time infinity. Every
policy then gets a full curve of completion-probability-over-time
per object, and two policies are compared by asking whether their
curves are distinguishable anywhere along the time axis, not just at
one arbitrary cutoff. A curve contains far more information than a
bit, so differences resolve with far fewer trials: two of their three
close model pairs separate reliably at 25–30 episodes per cell, where
they estimate a binary success-rate test would need 600–1500 paired
rollouts — roughly 30× more. Separately, they collect ~400 human
teleoperation episodes on the identical fixture and report each
model's headline number as a percentage of human throughput, so "13.8%"
means the best VLA moves items at about one-seventh the pace of a
person driving the same arm. The catch, which the paper is honest
about: the human number is for interpretation and cross-lab
comparability, not statistical power; the closest model pair still
does not resolve even with the fancy test; and a trivial change no
human would notice — swapping which side the camera and outbound tote
sit on — shifts one model's completion rate by 22 points, more than
the gap between the top two models, so all the statistics are
worthless without blinded same-session randomization of which policy
runs next.

## What it contributes

**Primitive: time-to-success CDF with operational failure semantics.**
Each placement operation yields a pair (T, E): T = time to successful
placement, E = event indicator. The CDF F(t) = P(T ≤ t) is estimated
per (model, object) cell by the Kaplan–Meier product-limit estimator.
Censoring/failure taxonomy:
- Success: (T, 1) at the observed placement time.
- Hard failure (item lost outside workspace, dropped-and-uncollected
  at episode end, safety stop): absorbed as a (T=∞, E=1)
  "ghost event" — the operation terminated, at infinite time. This is
  what makes the CDF a *joint* speed-and-completion object: F(t)
  plateaus below 1 forever for a policy that loses items.
- Timeout (incomplete at episode end): right-censored (T_tail, 0)
  with T_tail = episode duration minus last placement time. Episodes
  with zero successes contribute a single right-censored observation.
Timeout censoring is heavy in practice: 64–89% of episodes per model.

**Headline scalar: Human-Relative Throughput (HRT).** Per cell,
HRT(m, o) = RMST_Human,o(τ) / RMST_m,o(τ), τ = 240 s, where
RMST is restricted-mean survival time, the integral from 0 to τ of
(1 − F(t)) dt. Macro-averaged over objects with equal weights;
dimensionless, reported as a percentage of human pace. Stated
purposes (Sec 3.2): operator-practice grounding (a UPH-equivalent
against a same-fixture human reference, "so embodiment confounds
cancel"), cross-deployment comparability ("different operators,
different rooms, different reference pacing" reduces to comparing
ratios), and inheritance of the joint speed+completion property via
T=∞ inflating RMST_model. Explicitly NOT claimed: variance
reduction or any role in the significance machinery.

**Uncertainty: episode-clustered bootstrap.** 95% CIs from n_boot =
1000 resamples that resample *whole episodes*, not operations —
necessary because operations within an episode are correlated
(intra-episode ρ in [0.66, 0.71]).

**Significance: macro-averaged two-sample KS.** Per object,
D_o = sup_t |F_o^(a)(t) − F_o^(b)(t)|; the test statistic is the
macro average D̄ = (1/J) Σ_o D_o over J = 4 objects (so
per-object discrepancies at different timepoints all count, rather
than one pooled CDF washing them out). P-values come from a
pooled-resample episode-clustered bootstrap under H0 — not the
textbook KS null distribution, which would be invalidated by
censoring and clustering.

**Protocol recommendations** (Sec 4): a blind scheduler picks which
model runs next; the operator doesn't know which policy is active and
intervenes only for safety stops; spatial configuration (external
camera side, outbound tote side) is randomized and logged, balanced
per model; operator logs per-episode item counts as the success
source of truth; per-item timestamps are annotated post-hoc from
synchronized video (telemetry-based detector proposes, human reviews
on disagreement). The paper's own summary: "Blinded same-session
randomized rotation is the single protocol recommendation that does
the most work."

## The experiments they actually ran

**Rig and task.** Franka Research 3 + Robotiq 2F-85 (DROID-style
fixture), dual RGB (over-shoulder external + wrist). One task
primitive: bin-to-bin order picking, inbound tote to outbound tote.
Four trained objects spanning failure modes: wooden spoons (rigid,
elongated), towels (deformable), scissors (articulated, metallic),
batteries (small, rigid). Mean ~4.4 placement operations per episode;
per-item time budget 30 s (~10× the human pace of ~2.7 s/item); RMST
horizon τ = 240 s. Fine-tuning set: 449 episodes / ~13 h (spoons
167, towels 112, scissors 83, batteries 87).

**Policies.** Four public VLAs fine-tuned on that set: OpenPI
π0.5 (3B, FAST tokens), GR00T N1.6 (3B, Cosmos-Reason VLM +
diffusion head), ACT (CVAE chunking transformer), SmolVLA (450M,
LeRobot). Plus the human teleop reference on the same fixture via
their open-source positronic framework.

**Sample sizes.** ~995 episodes total: human 396; OpenPI 165; GR00T
165; ACT 151; SmolVLA 118 — i.e. 26–46 episodes per (model, object)
cell, "N ≈ 35 per cell," which the paper notes is already 2–3× the
field-median eval size. NB: N counts *episodes* (multi-operation,
cluster-correlated), not single-attempt trials.

**Headline table** (τ = 240 s): Human RMST 10.5 s [10.3, 10.8],
HRT 100%. OpenPI 77.7 s [69.2, 87.0], HRT 13.8% [12.2, 15.7]. GR00T
77.2 s [69.0, 86.4], HRT 13.3% [12.0, 15.2]. ACT 100.9 s [85.8,
117.6], HRT 10.5% [9.2, 13.2]. SmolVLA 165.8 s [147.0, 185.6], HRT
6.4% [5.7, 7.5]. Best VLA ~7× slower per operation than human.

**Power experiment** (Sec 5.2, the load-bearing result). For each
model pair and each N in {5, 10, 15, 20, 25, 30} episodes per cell:
300 outer subsampling trials (N episodes per cell, with replacement),
each running a 200-rep inner episode-clustered bootstrap for a
two-sided p-value at alpha 0.05; detection rate = fraction of trials
rejecting; power target 0.8; KS run at τ = 120 s here to cut
censoring. Results: macro-KS hits 80% detection on GR00T vs ACT at
N = 25 and on OpenPI vs ACT at N = 30; the closest pair, OpenPI vs
GR00T, does NOT resolve (detection 0.63 at N = 30). Binary
thresholds F(30s), F(60s) and RMST-as-scalar "fall well short of 0.8
on every close pair within budget." Their binary-baseline sizing: a
stratified-McNemar test on a 5 pp paired difference needs 600–1500
paired rollouts per cell — the "roughly 30×" sample-efficiency claim.

**Aggregation disagreement** (Sec 5.3). Same data, opposite top-1:
macro-AUC-vs-human ranks ACT 0.134 [0.108, 0.162] above OpenPI 0.100
and GR00T 0.095, while RMST/HRT ranks OpenPI/GR00T above ACT. Their
conclusion: no single scalar suffices; the headline-scalar choice is
a disclosed methodological commitment.

**Sensitivity and label audits.** Appendix G: same-side vs
opposite-side camera/tote swap shifts completion rate by +6.1 pp
(OpenPI), +22.2 pp (GR00T), +4.0 pp (ACT), +2.6 pp (SmolVLA) — the
GR00T shift exceeds the GR00T–OpenPI gap under study. Label
robustness: telemetry-proposed placement events disagreed with the
operator-logged success count in ~42% of episodes, triggering manual
review (single, non-blinded reviewer; timestamps only, <1 s timing
uncertainty; counts stay operator-truth); reanalysis on the
manually-reviewed cohort alone (N ≈ 420) preserves the full ranking
with HRT shifts of 2–5 pp, no CI crossings.

**Not tested:** any second embodiment or camera rig, any task beyond
the single pick-and-place primitive, held-out objects, autonomous
(non-teleop) human baselines, cross-lab replication of the same
protocol, or any drift/re-anchoring of the human reference over time.

## What transfers to us — and what doesn't

**The human-anchor question — answered, and it's a no.** The N≤30
resolution claim is carried entirely by model-vs-model statistics:
the KS statistic D_o = sup_t |F_o^(a) − F_o^(b)| contains no human
term, and the power experiment never touches the human data. What
does the statistical work is (a) using the whole event-time
distribution instead of one bit per trial, (b) macro-averaging KS
across objects so per-object gaps at different timepoints add rather
than cancel, and (c) episode-clustered bootstrap keeping the
inference honest. The human anchor only normalizes the *headline
scalar* (HRT) for interpretability and cross-lab comparability; drop
it and you lose a readable number, not the ability to resolve pairs.
For #16 that means the anchor is not a dealbreaker OR a requirement:
one cheap teleop block on a rig day buys the HRT-style readout, and
skipping it costs nothing statistically.

**What the SO-101 rig-day protocol would borrow** (design inputs, not
commitments): per-event time-stamping instead of binary outcomes
(synchronized video + telemetry-proposed timestamps, human-confirmed
— this slots directly onto our existing exteroceptive success-label
audit; their 42% telemetry/operator disagreement rate independently
corroborates our 32–48% telemetry false-positive finding that
telemetry alone can't be the label source); Kaplan–Meier + RMST at a
declared horizon with hard-failure-as-T=∞ semantics (a clean
answer to our partial-success/timeout bookkeeping question); the
macro-averaged KS with cluster-bootstrap p-values as the pairwise
comparison test; and above all blinded same-session randomized
rotation with spatial-nuisance logging — their 22 pp
camera-side effect is the loudest possible warning for a rig whose
camera mounting is improvised. Their entire stats pipeline is open
(`build/stats.py`), so the test can be lifted rather than re-derived.

**Caveats before rewriting the ≥50-trials/cell input.** (1) Their N
counts multi-operation *episodes*: 30 episodes × ~4.4 placements is
~130 timed events per cell (correlated, ρ ~0.7) — on a
single-attempt SO-101 task, 30 trials buys 30 events, so the honest
reading is "25–45 episode-equivalents," not "30 trials beats 50."
(2) Power was measured for *their* effect sizes on *their* gaps; the
closest pair still failed at N=30. (3) Time-to-success needs tasks
where time varies meaningfully; our tolerance-ladder tasks qualify,
pure binary-precision cells may not. (4) Franka FR3 + Robotiq is a
stiff, repeatable platform; SO-101 servo noise inflates within-cell
variance, pushing required N up, not down. (5) Their 64–89% timeout
censoring shows the method survives hard tasks, but our 20–80%
success-band placement remains the cheaper way to keep information
per trial high. Net: the CDF machinery likely *reduces* the trials
needed per resolvable comparison versus 50 binary trials, but the
defensible #16 position is to keep ≥50 single-attempt trials/cell
as the budget and treat KS-on-CDFs as the analysis that may let some
comparisons close early — not to pre-shrink the budget to 30.

## Hook corrections

- "Time-to-success CDFs, survival-analysis style" — CONFIRMED and
  stronger than flagged: literal Kaplan–Meier estimation, right-
  censored timeouts, RMST functionals, plus a non-obvious
  hard-failure-as-(T=∞) ghost-event convention.
- "HRT metric" — CONFIRMED: HRT(m,o) = RMST_Human,o(240s) /
  RMST_m,o(240s), macro-averaged, best model 13.8% (~7× slower than
  human). It is the headline scalar only.
- "Bootstrap confidence intervals" — CONFIRMED with a detail that
  matters: episode-clustered (whole-episode resampling, ρ 0.66–0.71
  intra-episode correlation), n_boot = 1000, 95%.
- "Per-object Kolmogorov–Smirnov tests" — PARTLY WRONG: not separate
  per-object tests. Per-object KS *statistics* are macro-averaged
  into one D̄ and a single p-value comes from a pooled-resample
  episode-clustered bootstrap (textbook KS null is invalid here).
- "Claims resolution at N ≤ 30 rollouts per cell" — CONFIRMED but
  narrower than the radar phrasing: 2 of 3 close pairs (GR00T–ACT at
  N=25, OpenPI–ACT at N=30, 80% detection, alpha .05); the closest
  pair OpenPI–GR00T does NOT resolve (0.63 at N=30). And "rollout" =
  a ~4.4-operation episode, not a single attempt.
- "Is the human anchor load-bearing for the resolution claim?" — NO.
  KS is purely model-vs-model; the anchor buys interpretability and
  cross-lab ratio comparability, zero statistical power. The radar's
  worry dissolves; the anchor is optional garnish we'd collect anyway.
- "Franka? objects? censoring?" — Franka FR3 + Robotiq 2F-85 (DROID
  fixture), single bin-picking primitive, 4 trained objects, timeouts
  right-censored (64–89% of episodes), hard failures as T=∞,
  no partial credit — per-item events do the graded-credit work.

## What it feeds

- **#16 (SO-101 rig-transfer bench)** — primary: adopt time-stamped
  per-event outcomes + KM/RMST + macro-KS-with-cluster-bootstrap as
  the analysis layer over the ≥50-trials budget; add blinded
  same-session policy rotation and camera/fixture-side logging as
  hard protocol requirements (22 pp nuisance effect); one teleop
  anchor block per rig day for an HRT-style headline (optional,
  non-load-bearing); lift `build/stats.py` rather than reimplement.
- **#16 (label audit thread)** — their 42% telemetry/operator
  disagreement independently replicates our 32–48% telemetry
  false-positive finding; video-confirmed timestamps with telemetry
  as proposer-not-truth is the convergent design.
- **#9 (zero-GPU instruments)** — timestamps come from synchronized
  video + gripper telemetry post-hoc: reaction-time and
  time-to-first-action accounting drops out of the same event stream
  for free, no extra sensors.
