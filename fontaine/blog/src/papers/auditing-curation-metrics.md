# Action-only demo scorers can't see a wrong action — but the "state" that rescues them is object pose, not proprioception, in a 3-seed NumPy toy

*Read 2026-08-10 (lit slice `lit-radar-0822`, priority 3 — companion of
[What Demonstration Curation Metrics Do to Your Policy](what-curation-metrics-do.md)
([2606.10229](https://arxiv.org/abs/2606.10229)), read the same session).
Paper: [2606.05588](https://arxiv.org/abs/2606.05588) — "Auditing
Demonstration Curation Metrics: Action-Only Scorers Fail on the
Structural Defects That Degrade Imitation Policies" (Aarav Bedi, sole
author; University of California, Berkeley; arXiv preprint 2026-06-04,
cs.RO/cs.LG, 5 pages, 3 figures, 4 tables; CC BY 4.0. Code VERIFIED:
[github.com/aaravbedi/scorer-fail-on-structural-defects](https://github.com/aaravbedi/scorer-fail-on-structural-defects)
fetched HTTP 200 — 12 commits, simulator/bakeoff/selftest scripts plus
a `test_libero_env.py`, README is title-only; arXiv abs page fetched
HTTP 200. Companion relationship verified from both abs pages: same
sole author, 2606.10229 submitted four days later (2026-06-08) and
moves the same seven-metric audit and the same early-release defect
onto contact-rich LIBERO — it is literally the "next step" this
paper's limitations section asks for; 05588 does not cite it.)*

**The paper in plain words.** When you teach a robot by copying human
or scripted demonstrations, some demonstrations are bad, and people
have proposed automatic scorers that grade each demonstration so the
bad ones can be thrown away before training. The catch is that every
scorer was validated on different data with different protocols, so
nobody knows which ones actually find the demonstrations that hurt
the trained robot. This paper builds a deliberately tiny, fully
controlled world — a pick-and-carry task in a NumPy simulator styled
on a single-arm ALOHA rig — where the author can inject known flaws
into known episodes and check two separate things: does a scorer flag
the flawed episodes, and does training on what the scorer keeps
actually make the robot better? The flaws come in two families. One
family is "messy but right": shaky correlated noise, tremor, chopped
or looping trajectories layered on otherwise correct behavior. The
other is "clean but wrong": the gripper opens too early mid-carry, so
the demo confidently does the wrong thing at the key moment. Result:
the messy family is easy — a generic outlier detector spots it almost
perfectly and filtering restores nearly all lost performance. The
wrong-action family is invisible to every scorer that looks only at
the motion statistics of the commands; two such scorers actually rank
the flawed episodes as *better* than clean ones, because noise looks
like expressive variety. Only scorers that look at where the arm and
object actually went detect the wrong action, and even the best of
those recovers only about a third of the lost success rate. The moral
the author draws: judge a curation method by the policy it produces,
not by the defects it flags — the two are decoupled.

## What it contributes

- A controlled, label-known curation testbed: clean demos from a
  privileged scripted controller; defects injected with recorded type
  at a 40% contamination rate; metrics see states+actions only,
  enforced at the type level so no scorer can peek at the label.
- A two-axis audit of seven published-signal-style metrics: detection
  AUROC against hidden labels, AND downstream BC success on the
  top-scoring kept fraction, with data volume held fixed (every
  condition trains on exactly N=150 demos) so differences are quality,
  not quantity.
- The regime split as the headline finding: subtle perturbations are
  detectable (isolation forest 0.968 AUROC) and recoverable (71.3% vs
  72.0% oracle); the structural error is invisible to all action-only
  metrics (best: 0.543, chance) and only ~1/3-recoverable even by the
  best state-side metric.
- The decoupling result: comparable detection can yield policies
  ~11 points apart (kNN 48.0% vs trajectory alignment 37.3% at AUROC
  0.86 vs 0.76) — detection accuracy is the wrong headline number for
  a curation method.
- Full release of simulator, injectors, all seven metric
  implementations, and the pipeline reproducing every table/figure.

## The experiments they actually ran

**Testbed.** NumPy pick-and-place simulator modeled on single-arm
ALOHA. Action: 7-D (end-effector translation, rotation, gripper).
Observation: 11-D — EE position and orientation, gripper state, a
*noisy object-position estimate* (Gaussian σ = 0.03 m per step,
explicitly "standing in for imperfect perception"), and a normalized
time index. Clean demos from a phase-based scripted controller with
privileged state; defective demos are clean episodes passed through
an injector. Policy: three-layer BC MLP, 50 fresh eval rollouts,
3 seeds (42, 0, 7).

**Defect taxonomy.** Two regimes, each at 40% contamination:

- Subtle (four injectors): *action noise* — temporally correlated
  AR(1) noise on actions ("shaky teleoperation rather than random
  garbage"); *tremor* — high-frequency sinusoid on actions;
  *truncation* — episode cut to roughly half length; *detour* — a
  reversed mid-episode segment spliced back in so the trajectory
  loops. No numeric injector parameters (AR coefficient, tremor Hz,
  amplitudes) appear in the paper text — they live only in the code.
- Structural (one injector): *early release* — gripper commanded open
  during the carry phase, object dropped partway to goal. "Not noise;
  a systematic wrong action in a specific part of the state space."

**Metrics and their feature spaces.** All seven consume low-dim
states and actions only — no images, no embeddings, no language, no
force/torque anywhere in the audit:

- Action-only: *smoothness* (spectral arc length of the speed
  profile), *entropy* (std of the action sequence), *length*
  (trajectory length), *isolation forest* and *ensemble* (both on a
  vector of action-derived summary features).
- State-trajectory-aware: *kNN* (distance to nearest neighbors in a
  trajectory-level feature space including state-trajectory
  summaries), *trajectory alignment* (agreement of a demo's state
  trajectory with the dataset's aggregate behavior). "State" here is
  the 11-D observation — which includes the noisy *object* position.

**Detection AUROC** (3 seeds; subtle regime pooled across the four
injectors — no per-injector breakdown is reported):

- Subtle: isolation forest 0.968±0.004, kNN 0.871±0.003, length
  0.625±0.000, traj-align 0.528±0.044, smoothness 0.478±0.005,
  ensemble 0.002±0.001, entropy 0.000±0.000.
- Structural: kNN 0.863±0.025, traj-align 0.762±0.025, isolation
  forest 0.543±0.031, length 0.500±0.000, entropy 0.469±0.028,
  ensemble 0.291±0.028, smoothness 0.244±0.020.

**Downstream success** (N=150 kept, 50 rollouts, 3 seeds):

- Subtle: oracle 72.0±4.9, isolation forest 71.3±10.9, smoothness
  70.0±9.1, length 56.7±15.5, ensemble 56.0±16.1, contaminated
  baseline 55.3±0.9, entropy 41.3±20.2. Seed SDs run to ±20 points;
  the author flags subtle-regime downstream numbers as "suggestive."
  kNN and trajectory alignment do NOT appear in this table.
- Structural: oracle 62.7±4.7, kNN 48.0±4.3, length 38.7±5.0,
  traj-align 37.3±5.0, contaminated 36.0±4.3, isolation forest
  35.3±2.5, ensemble 32.0±5.7, entropy 30.7±3.4, smoothness 27.3±6.8.
  A 27-point oracle gap; kNN recovers about a third of it; the
  below-baseline action-only results are called "a consistent trend
  rather than a sharp effect" (within 3-seed variance).

**Mechanism offered.** BC averages over demos: zero-mean wobble
washes out with data, so even rough curation suffices; early release
"plants a specific wrong action in a specific part of the state
space, and averaging never removes it" — its signature lives in
*where the arm went*, exactly what action statistics discard.

**Not tested / by-design limits.** One structural defect, not a
taxonomy; defect families deliberately chosen "to straddle the
action/state divide," so action-only blindness is "in part by
design" (the author says so). No pixels; no real teleop data; one
task, one embodiment, one policy class; 3 seeds; no ablation of the
state metrics with object-position dims removed (i.e., robot-proprio
only) — the single ablation we most needed; no filtering-budget
sensitivity; no per-injector subtle AUROCs.

## What transfers to us — and what doesn't

The radar framed our positions-only corpus as "the failing feature
space." Half right. Precisely:

- **The rescue signal is object state, not proprioception.** Their
  "state" metrics read a state vector *containing object position*
  (noisy, σ = 3 cm). Joint positions alone are also "state," but
  the paper never ablates whether kNN/alignment still detect early
  release without the object dims — plausibly the drop signature IS
  the object diverging from the hand path. So a positions-only corpus
  can *run* their state metrics but may not get their detection.
- **RGB partially rescues us.** Our frames have images; their object
  estimate explicitly stands in for imperfect perception, and their
  results survive 3 cm noise — roughly off-the-shelf visual pose/
  tracking error. Lifting object state from RGB (or using visual
  trajectory embeddings as the "state summary") is the concrete path
  to their kNN-class scorer on our corpus. Caveat they never face:
  kNN and alignment compare against "the dataset's aggregate
  behavior" — meaningless across 880 heterogeneous teleop datasets;
  any port must run per-task/per-dataset cohorts.
- **Velocity census** (action statistics): maps onto their entropy/
  smoothness family. Blind to early-release-class defects and, worse,
  variance-based scoring is the inversion case — shaky-but-correct
  demos get *rewarded*. Fine as a diversity/coverage lever; do not
  promote it to a quality gate.
- **Kinematic continuity screen**: their subtle regime is its home
  turf — isolation-forest-style multivariate outliering hit 0.968
  AUROC and recovered the full gap. Splice/truncation/tremor-class
  artifacts are the winnable regime; this arm is *supported*, not
  undermined. It contributes nothing on wrong-action defects (0.543
  = chance for the best of that family).
- **Δq_d = action−state gate**: consumes actions AND proprio, so not
  "action-only" — but its blind spot is sharper: an early release is
  a wrong command that the plant *tracks faithfully*, so the
  command−state residual is small by construction. It detects bad
  tracking/contact, not bad intent. Their defect class sails through.
- **Influence-function scoring** (policy gradients): outside their
  audited feature spaces entirely, and per our 0821 QoQ read it is
  the offline pole with gains on injected failures. Their meta-lesson
  binds it anyway: report the curated-policy delta, never the
  detection AUROC alone.
- **Cheapest decisive probe** (their code is released and fetches
  200): rerun their bakeoff with the object-position dims deleted
  from the state metrics' view — a NumPy sim, an MLP, 3 seeds. If
  kNN's 0.863 structural AUROC collapses without object state, then
  positions-only state metrics are dead for wrong-action defects and
  RGB-derived object state becomes the required ingredient; if it
  holds, proprio-only trajectory scoring is live for us. No new
  commitment implied — noting it as the experiment that settles the
  radar's framing.

## Hook corrections

- **"Action-only scorers blind to structural defects" — CONFIRMED,
  with two narrowings.** There is no structural defect *taxonomy*:
  exactly one structural defect (early gripper release during carry)
  vs four subtle ones (AR(1) noise, tremor, truncation, detour — the
  radar's list of three missed detour). Best action-only structural
  AUROC 0.543 (chance); and the paper itself concedes the blindness
  is "in part by design" since the defect was built to live in state.
- **"Two metrics actively PREFER defective episodes" — CONFIRMED;
  the two are entropy and ensemble, and the regime matters.** Named
  in Fig. 1: "entropy and ensemble are inverted (below chance) on
  one or both." Sharpest on the *subtle* regime: entropy AUROC
  0.000±0.000, ensemble 0.002±0.001 — near-perfect inversion, because
  correlated noise and tremor inflate action variance so defective
  demos score "more exploratory, hence higher quality." Entropy-
  curated training lands at 41.3% vs 55.3% uncurated. On the
  structural regime the below-chance metrics are smoothness (0.244)
  and ensemble (0.291), with smoothness worst downstream (27.3% vs
  36.0% baseline) — a trend within 3-seed variance.
- **"Do state metrics need VISUAL state?" — CORRECTED, both ways.**
  Neither visual nor purely proprioceptive: they consume a low-dim
  state vector whose load-bearing extra ingredient is a noisy
  *object-position estimate* (σ = 0.03 m, a stand-in for
  perception). No images or embeddings anywhere. Whether robot-side
  positions alone suffice is untested (no ablation) — so "a
  positions-only corpus can run them" is true mechanically and
  unverified in effect; RGB-derived object state is our route to
  what they actually measured.
- **"Feature spaces consumed" — CONFIRMED.** Actions-only for five
  metrics (smoothness, entropy, length, isolation forest, ensemble);
  states+actions (low-dim, incl. object pose) for kNN and trajectory
  alignment. No language, no vision, no force/torque in the audit.

## What it feeds

- **#9 data levers** — velocity census stays a coverage lever, never
  a quality gate: variance scoring is the documented inversion case.
- **#9 data levers** — kinematic continuity screen validated for the
  subtle/artifact regime (0.968 AUROC, full gap recovered); scope it
  to artifacts, not intent errors.
- **#9 data levers** — Δq_d gate blind-spot noted: well-tracked wrong
  commands produce small residuals; wrong-action defects need a
  state-trajectory (object-aware, RGB-derived) scorer we do not have.
- **#9 data levers** — one settling probe on their released testbed:
  object-dims-ablated kNN decides whether proprio-only trajectory
  scoring can ever catch wrong-action defects.
- **Lit thread** — pairs with the companion page on
  [2606.10229](what-curation-metrics-do.md): same author, same
  metrics and defect, LIBERO-scale; that paper adds the episode-
  length confound (5 of 7 metrics exploit length as a label proxy),
  which this toy's length-AUROC 0.500/0.625 already hints at.
