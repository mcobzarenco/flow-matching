# The best defect detector yields the worst policy (AUROC 0.804 → 13.3% success) — but it's one defect, one task, 80 demos, 3 seeds

*Read 2026-08-10 (lit slice `lit-radar-0822`, priority 2 — companion
of [Auditing Demonstration Curation Metrics](auditing-curation-metrics.md)
([2606.05588](https://arxiv.org/abs/2606.05588)), read the same
session). Paper: [2606.10229](https://arxiv.org/abs/2606.10229) —
"What Demonstration Curation Metrics Do to Your Policy" (Aarav Bedi;
affiliation not listed on the abs page — the companion paper lists UC
Berkeley; arXiv preprint, submitted 2026-06-08, 6 pages / 1 figure /
2 tables, cs.RO + cs.LG; CC BY 4.0. Code IS released: the paper
links https://github.com/aaravbedi/structural-defect-curation —
fetched, HTTP 200; the repo contains the metric implementations
(`methods/curation_metrics.py`), the LIBERO testbed, and a 13-stage
eval pipeline (`run_curation_eval.py`); the README as fetched does
not cite either arXiv ID. arXiv abs/pdf both 200.)*

**The paper in plain words.** When you collect robot demonstrations
at scale, some of them are bad — the operator fumbled, the object
slipped — and the standard fix is to run a "quality metric" over
every episode, keep the top-scoring ones, and train on those. The
obvious way to check whether such a metric works is to ask how well
it separates bad episodes from good ones, summarized as a detection
score (AUROC). This paper builds a small, fully controlled testbed
to ask a sharper question: does being good at *finding* bad episodes
mean the *policy trained on your filtered data* is better? The
author takes a simulated pick-and-place task (grasp a bowl, carry it
to a plate), makes 16 clean scripted demonstrations and 64 broken
ones where the gripper opens too early and drops the bowl mid-lift,
mixes them into one 80-episode pile, and runs seven curation metrics
over it. Each metric keeps its top 75% of episodes; a small network
is cloned on each kept set and rolled out in the simulator. The
result is a clean dissociation: the metric that detects defects best
(a hand-built gripper-timing score, AUROC 0.804) trains the *worst*
policy — 13.3% success, barely above the 3.3% you get with no
filtering at all — because it doesn't just discard broken episodes,
it preferentially keeps the *least obviously broken* broken ones. A
metric with much weaker detection (trajectory alignment, AUROC
0.638) trains a policy at 90.0%, nearly matching the 93.3% oracle
trained on ground-truth clean data. Across all seven metrics,
detection rank and policy rank are essentially unrelated (Spearman
−0.14). The second finding is just as useful: before a controlled
truncation step, five of the seven metrics scored near-perfect
detection for a boring reason — broken episodes never finish, so
they run to the 500-step time limit while clean ones end near 325
steps, and any metric that accumulates over time inherits episode
length as a free label. The catch: all of this is measured on
exactly one task, one injected defect type, 80 demonstrations, and
three seeds, and the author says so plainly.

## What it contributes

1. **A detection-vs-downstream dissociation, cleanly instrumented.**
   Seven curation metrics are scored on the same contaminated set on
   both axes: defect-detection AUROC and success rate of the
   behavior-cloned policy trained on their top-75% retained subset.
   The orderings disagree almost completely (Spearman rho = −0.14
   across the seven metrics). Best detector (gripper timing, 0.804)
   → worst policy (13.3%); second-worst detector among the useful
   ones (trajectory alignment, 0.638) → 90.0%, versus 93.3% oracle.

2. **A named failure mechanism, not just a correlation.** Gripper
   timing fails because its score makes "fine-grained distinctions
   within the defective class": among broken episodes it top-ranks
   the ones that release *latest*, so the curated set is stuffed
   with near-miss failures while some clean episodes are discarded.
   Isolation forest fails differently: after the drop, the arm moves
   with near-zero gripper actuation, which looks *less* anomalous in
   action-summary space than a real grasp — so it keeps the broken
   episodes (3.3% success, exactly the no-curation baseline).

3. **The episode-length confound, with a control.** Defective
   episodes hit the 500-step time limit; successful ones end ~325
   steps. Before controlling for this, five of seven metrics (and a
   pure length baseline) sat at or near AUROC 1.000. Truncating all
   episodes to T=324 — valid because 91% of defective episodes
   release before step 324 (mean release t=199, range 171–343) —
   collapses them: e.g. isolation forest 1.000→0.440, smoothness
   0.979→0.447. The headline AUROCs are the *length-controlled*
   ones.

4. **Two explicit recommendations**: "evaluate curation metrics by
   the policy they produce, not by their detection AUROC" and
   "control for episode length before computing any curation
   metric."

5. **A released testbed** (verified live, see header) so the whole
   table is reproducible.

## The experiments they actually ran

**Testbed.** LIBERO benchmark (robosuite-based, contact-rich
physics). One task: pick-and-place — grasp a bowl from a randomized
start position, carry to a fixed plate. Phased scripted controller
(approach/pre-grasp, descent, grasp/lift, transport/release).
Observations are 26-D low-dim state (end-effector position, gripper
state, bowl position, a 6-D phase one-hot, among the listed
components); actions are 4-D (3-D end-effector delta + gripper).
Testbed validation: with phase conditioning, a policy trained on 47
clean demos hits 90% over 10 rollouts before any defect injection.

**Defect taxonomy: one type.** Early gripper release — the scripted
defective policy opens the gripper at a random timestep between 30%
and 70% of the LIFT phase, dropping the bowl before transport.
Clean demos: 100% success. Defective: 0%. Dataset: 80 episodes
total, 16 clean + 64 defective = **80% contamination** (the author
notes real pipelines are more like 20–40%).

**The seven metrics.** (1) Smoothness: SPARC spectral arc length of
the action speed profile. (2) Entropy: negative std of the action
sequence. (3) Gripper timing: normalized timestep of first gripper
opening after closure — the defect-targeted, hand-built metric.
(4) Isolation forest on per-dimension action summary features
(mean/std/max/min/RMS), fit on clean demos. (5) Ensemble: 0.5 ×
smoothness + 0.5 × gripper timing. (6) kNN (k=5) distance in a
state/action summary feature space. (7) Trajectory alignment:
cosine similarity of a demo's mean state trajectory to the clean-set
mean.

**Protocol.** Each metric keeps its top 75% (60 of 80). BC policy:
2-hidden-layer MLP, 256 units, tanh output; Adam, lr 1e-3, weight
decay 1e-4. Three seeds (42, 0, 7) × 30 rollouts each = 90 rollouts
per cell.

**Main table (length-controlled AUROC / success %, mean±std over 3
seeds):**

| Metric               | AUROC | Success (%) | Gap closed |
|----------------------|-------|-------------|------------|
| Oracle (16 clean)    |   —   | 93.3 ± 0.0  | 100%       |
| Ensemble             | 0.761 | 91.1 ± 1.6  | 97%        |
| Trajectory alignment | 0.638 | 90.0 ± 0.0  | 96%        |
| Entropy              | 0.280 | 77.8 ± 12.6 | 82%        |
| Smoothness           | 0.447 | 63.3 ± 30.7 | 67%        |
| kNN                  | 0.712 | 58.9 ± 41.7 | 62%        |
| Gripper timing       | 0.804 | 13.3 ± 16.6 | 11%        |
| Isolation forest     | 0.440 |  3.3 ± 0.0  | 0%         |
| Contaminated (all 80)|   —   |  3.3 ± 0.0  | 0%         |

**Length table (AUROC raw → truncated to T=324):** length baseline
1.000→0.500; ensemble 1.000→0.761; isolation forest 1.000→0.440;
kNN 1.000→0.712; trajectory alignment 1.000→0.638; smoothness
0.979→0.447; gripper timing 0.957→0.804; entropy 0.000→0.280 (the
raw 0.000 — perfectly *inverted* separation — is left undiscussed).

**What was NOT tested** (mostly self-declared limitations): only
one defect type and one task — "whether the AUROC-downstream
decoupling holds across defect types is an empirical question we
have not answered"; only 80% contamination (ordering "may change" at
realistic 20–40%); the oracle's 16 clean demos were collected under
different random seeds than the main clean set, so the 93.3% ceiling
carries a small distributional caveat; 3 seeds gives "wide
confidence intervals for high-variance metrics" (kNN ±41.7pp is
close to uninformative); no retention-threshold sweep in the paper
(though the repo ships a `sweep_contamination_boundary.py`); no
real-robot data; no learned/influence-style metrics — all seven are
cheap statistical scorers, so QoQ-style influence functions are
outside the tested set.

## What transfers to us — and what doesn't

**The core warning transfers; the numbers do not.** 80 scripted
demos, a 2×256 MLP on 26-D state, and one injected defect are
nowhere near our regime (~18.7M frames, 880 heterogeneous teleop
datasets, frozen Molmo2-4B trunk + flow action expert, quality
unlabeled). Treat every number above as an existence proof, not an
effect size: it is now *demonstrated* that a curation metric can be
the best available defect detector and still poison the retained
set, via within-defective-class ranking. That failure mode needs no
simulator to be plausible for us.

**The length confound hits us twice, as the radar said — but note
what they actually showed.** They showed length leaks into
*detection AUROC* when failure correlates with timeout; they did
not show length leaking into an offline *training-loss* eval,
because they don't have one — their downstream axis is rollouts.
The transposition to our world is still direct:

- **(a) #9 curation arms.** Any arm whose score accumulates or
  averages over an episode (velocity/speed census, kinematic
  continuity screen, isolation-style outlier scores, arguably the
  influence pass if per-episode influence is summed over frames) can
  rank episodes by length or speed composition rather than quality.
  Their five collapsing metrics are exactly the summary-statistic
  family our census/screen arms belong to.
- **(b) the chunk-MAE panel itself.** Our panel pools over frames,
  so a curation arm that shifts the *length/speed composition* of
  training data can move panel MAE without changing policy quality
  — the same spurious channel, one level up. The paper's rollout
  evaluation is precisely the escape hatch we don't have.

**Cheapest equivalent control in an offline-MAE world** (sketch, no
commitment): (1) their truncation control transposes to
*length-stratified or length-partialled scoring* — before trusting
any #9 arm's ranking, report its Spearman correlation with episode
length (and mean speed), and/or recompute scores on
length-normalized segments; (2) add a **rank-by-length null arm**:
if a curation arm's retained set doesn't beat "keep the top-75% by
episode length" on the panel, its signal is presumptively a length
proxy; (3) for the panel, report per-episode-mean MAE (equal
episode weighting) and length/speed-stratified cells alongside the
pooled per-frame number, so composition shifts are visible instead
of silent. All three are CPU-cheap and rollout-free.

**What doesn't transfer.** The specific metric rankings (trajectory
alignment winning here is an artifact of one defect producing a
large state-trajectory deviation); the 80% contamination regime;
the oracle comparison (we have no ground-truth clean labels, only
held-out trusted demos); and their recommendation #1 taken
literally — "evaluate by the policy produced" means rollouts, which
our programme deliberately doesn't run. For us the honest reading
is: rollout-free curation evaluation inherits *both* of this
paper's failure modes at once, so the length controls above are the
minimum, not a nicety.

## Hook corrections

- **Best detector → worst policy (0.804 → 13.3%): CONFIRMED.**
  Gripper timing, AUROC 0.804 (highest of the seven,
  length-controlled), success 13.3 ± 16.6% vs 3.3% unfiltered.
  Nuance the hook missed: 0.804 is the *post-truncation* AUROC; on
  raw data several metrics sat at 1.000, so "best detector" is only
  well-defined after the length control.
- **Weak detector ≈ oracle (0.638, 90.0 vs 93.3): CONFIRMED.**
  Trajectory alignment, 90.0 ± 0.0% vs oracle 93.3 ± 0.0%. The hook
  under-sells the table: the ensemble (0.761) does even better at
  91.1 ± 1.6%, so "weak detector wins" is really "detection rank is
  uninformative" (rho = −0.14), not "weakness helps."
- **5 of 7 metrics exploit length: CONFIRMED as stated in the
  paper**, but the five are never named in one sentence; from Table
  I the five with near-perfect raw AUROC (≥ 0.979) that collapse
  under truncation are smoothness, ensemble, isolation forest, kNN,
  and trajectory alignment. Gripper timing only partially depends
  on length (0.957 → 0.804); entropy is *inversely* length-driven
  (raw AUROC 0.000). One subtlety: in this testbed length is a
  *true* failure correlate (defective = timeout), so raw detection
  was genuinely perfect — the objection is that it's a trivial
  proxy that won't survive settings where length and quality
  decouple.
- **Testbed released: CONFIRMED AND VERIFIED.** URL is in the
  paper; https://github.com/aaravbedi/structural-defect-curation
  returned HTTP 200 and contains the metric implementations,
  testbed, and eval pipeline. (README as fetched cites neither
  arXiv ID; no license file surfaced in the fetch.)
- **What the dissociation is measured on: the hook's suspicion was
  right — it is one cell.** One task (LIBERO bowl pick-and-place),
  one defect type (early gripper release, injected at 30–70% of
  LIFT), 80 demos at 80% contamination, top-75% retention, 3 seeds
  × 30 rollouts. The *dissociation* is a pattern across the seven-
  metric table within that single setting; its generality across
  defect types is explicitly declared open. 13.3 ± 16.6% over three
  seeds also means individual seeds varied widely.
- **Relationship to companion 2606.05588:** same sole author,
  posted four days earlier; the companion audits the same seven-
  metric family across *two* defect categories (subtle
  perturbations vs structural errors) and finds action-only scorers
  blind to structural defects; 10229 is the narrow deep-dive on one
  structural defect plus the length confound. The 10229 HTML text
  contains no citation of 05588 — the pairing is ours, not the
  paper's.

## What it feeds

- **#9 data levers (primary):** every banked arm gets a
  length/speed-correlation report before its ranking is trusted;
  add a rank-by-length null arm as the beat-this baseline.
- **#9 influence pass (QoQ):** check whether per-episode influence
  aggregation is length-dependent before hard top-N selection.
- **Eval methodology (chunk-MAE panel):** per-episode-mean and
  length/speed-stratified panel variants alongside pooled
  per-frame MAE, so curation-induced composition shifts are
  visible.
- **Lit thread:** pair with
  [2606.05588](auditing-curation-metrics.md)'s structural-defect
  blindness result; both testbeds are released and could seed a
  defect-injection sandbox if #9 ever needs a controlled positive
  control.
