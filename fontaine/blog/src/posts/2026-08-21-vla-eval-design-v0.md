# The VLA eval design, v0 — verdicts, guards, and non-instruments

*2026-08-21. Design doc, v0 — no new experiments; every number below
is a banked read with its receipt already published. This turns the
[probe-decoupling standing rule](2026-08-21-probe-decoupling-note.md)
into the eval architecture the north-star rig VLA
([idea #16](../ideas/16-rig-transfer-benchmark.md)) needs decided up
front, and consolidates the bench-design inputs the lit radar has
been banking into #16 for three weeks. CPU work item, written while
the gripper-carrier endpoint battery runs.*

**Plain words**: when we eventually put a policy on the owner's real
arm, we need to have decided *in advance* which measurements are
allowed to answer "is this model better at the task?" and which are
only allowed to say "something is broken". This document fixes that
assignment. The rule, learned the hard way twice this month: only
actually running the policy — in simulation today, on the rig later
— may answer the first question. Every cheaper measurement is a
smoke detector: valuable when it fires, meaningless when it is
silent. We now have two banked cases where a policy that had
catastrophically stopped grasping scored *at the healthy level* on
every cheap instrument at once.

## The organizing rule: every instrument gets exactly one role

Three roles, assigned per instrument at pre-registration time, never
renegotiated mid-run:

- **Verdict** — may answer "did the model get better at the task?".
  Rollout-based only. Pre-registration gates must be stated in
  rollout numbers.
- **Guard** — may trigger *hygiene* actions (abort, investigate,
  block a launch) via an explicit threshold against a banked anchor.
  Record-only for everything else. **Silence from a guard proves
  nothing about the verdict.**
- **Non-instrument** — may never gate a verdict, however green it
  looks. Listed explicitly, with the banked case that convicted it.

The forcing exhibit (full analysis in the
[decoupling note](2026-08-21-probe-decoupling-note.md)): four
training cells differing only in mix composition. The rollout column
spans **28×** (1 → 28 grasps/100, paired per-seed reads
CI-excluding-zero at every rung); both offline columns are flat
*and interleaved* — the collapsed cells do not even sort to one end.

![Panel and probe columns for the four cells vs their sim100
rows](../img/probe-decoupling/decoupling_columns.png)

## 1. Verdict instruments — rollouts only

### Tier 1, today: sim100

The working verdict instrument: 100 unseen seeds, closed-loop, grasp
success counted per seed (euler-10, 30 s episodes,
execute-horizon 30). What makes it a *verdict* instrument is the
protocol wrapped around it, not the simulator:

- **Paired per-seed reads** (`sim100_paired_read.py`): every claim
  is a McNemar read on per-seed discordant pairs, CI excluding zero
  required. Same seeds across arms — the variance that matters is
  between policies, not between draws.
- **Frozen verdict grid, posted before launch**: e.g. the live
  gripper-carrier cell's ≥20 / ≤10 / 11–19 grid with each branch's
  follow-up pre-committed. The grid is written when we are ignorant;
  the numbers land into it.
- **Banked anchors**: democlean 8/100, onerig 28/100, control
  11/100, convicted 1/100 — new cells read against these, paired.
- **Cost**: ~2.5 GPU-h per endpoint. The cost asymmetry that once
  justified probe-based verdicts is gone; a wrong verdict costs a
  17 GPU-h cell.

Honest limit: sim100 is itself a proxy with a visual gap, and its
substrate is part of the pre-registration — renderer amendments
(shadow v4, fitted lens) are paired-gated and owner-called precisely
because changing the eval mid-lineage silently re-prices every
anchor. Its authority is strongest *relative* (A/B on paired seeds),
and the sim-vs-rig calibration is explicitly unmeasured until the
rig bench exists.

### Tier 2, the cheap relative screen: Squint-class twin rollouts

The [Squint substrate](../papers/squint.md) (lit `0819`; preflighted
GO 2026-08-14) gives deterministic-seed, success-predicate rollouts
of our exact arm class on trivial compute (~1.35 s per 50-step
episode at the CPU floor; the real cost is policy inference). Its
role in this design:

- **Relative A/B screens only** — deltas between our checkpoints
  with the (large, wrong) domain gap held constant across arms. The
  in-domain BC baseline at 41.9% sim success is the permanent
  warning label on any absolute number.
- **Ground-truth-labeled rollout generation** for calibrating
  failure detectors (idea #6) — sim success predicates are free
  labels at unlimited volume.
- **Gate before first verdict-adjacent use**: its own
  pre-registration plus a sim-adaptation sanity arm, so "policy is
  bad" separates from "renderer is alien". Until then it screens,
  it does not judge.

### Tier 3, the rig: protocol sketch

What v1 grows into a real pre-registration when the owner's better
rig dataset and rig time exist. Every constant below is a banked
design input, not an invention:

- **Pairing and blinding.** Paired designs with same-session
  **blinded policy rotation** and spatial-nuisance logging as hard
  requirements: PhAIL measured a camera/tote side swap moving one
  model 22 pp — larger than the model gap under study. On the rig,
  "same seed" becomes "same scripted reset + logged nuisances".
- **Budget.** ≥50 single-attempt trials per cell (n=20 gives
  ±22 pp); pilot every task into the 20–80% success band before it
  counts (2 of 4 tasks in the SO-101 VLA benchmark were wasted on
  ceiling/floor); precision tasks built as one task × 2–3 tolerance
  levels with the fitted precision ceiling *c* as the headline.
- **Analysis.** McNemar on paired binary outcomes (our standing
  read) plus PhAIL's machinery for early closes: per-event
  time-to-success CDFs (Kaplan–Meier, timeouts right-censored, hard
  failures at T=∞) + macro-KS with episode-clustered bootstrap —
  resolves close pairs at 25–30 episodes/cell where binary tests
  need 600–1500. Continuous progress as the primary metric separates
  policies at up to 70% fewer trials than binary success. Publish
  the full CDF panel: macro-AUC and RMST are banked ranking the same
  three models in *opposite* order.
- **Labels.** Telemetry-style success flags run 32–48%
  false-positive even in clean sim (independently replicated by
  PhAIL's 42% telemetry/operator disagreement) — every cell gets an
  exteroceptive audit; a final-frame scene read is the cheapest
  sufficient form. Annotation protocol (multi-label vs primary
  failure) pre-registered.
- **Secondary instruments, record-only at first.** TTFA
  (time-to-first-action; the chosen execution horizon will dominate
  decode latency), completion time (hesitation — Legato's −20%
  lives where frame-level smoothness reads see nothing), jerk (p95
  acceleration + velocity zero-crossing rate), boundary-overlap
  RMSE. One human-teleop block per rig day for the readable headline
  scalar — zero statistical cost to skip.
- **Retention rider.** Any rig fine-tune carries 229h-corpus replay
  at ρ ∈ [0.02, 0.2] and a held-out retention read — naive rig-only
  fine-tuning is banked wiping prior competence within a few
  thousand steps (BWT −81), and weight-space adaptation's forgetting
  cost is measured (0.88 vs −0.94 held-out retention).

## 2. Guard instruments — anchored thresholds, hygiene only

A guard needs three things to stay on the roster: a banked anchor,
an explicit threshold, and a *named failure class it has actually
caught*. Current roster:

| guard | failure class | threshold vs anchor | banked catch |
|---|---|---|---|
| k4l2 panel + guard read | normalization/wear bugs | worse than banked npz by > +0.05, CI-excl-0 | raw-table wear read 58.14 vs 27.40 re-worn (~30 deg class signal); pdnorm table fix verified Δ −28.96 |
| in-train probe (eval-250) | training divergence | convicted-class elevation (plateau → peak) ⇒ investigate | three-way mix fighting the demos: 5.45/5.47 → 6.83 → 6.17, flagged mid-run for free |
| truth-fit rewear | estimator seams | native−truthfit gap bounds any panel claim | seam measured 1.55–1.91 deg — *larger than the 1.17-deg between-cell spread* being read |

Guards gate hygiene: they can abort a run, block a launch, or force
an investigation. They cannot clear a cell, and they cannot convict
one — the convicted three-way cell's probe elevation was a true
positive, but the democlean cell showed that the *same collapse
class* can track the healthy curve to the digit. An alarm that fires
on some poisons and not others, with no way to know which kind you
have, is a drift alarm — not a verdict.

## 3. Non-instruments — what may never gate a verdict

Each entry carries the case that convicted it:

1. **Panel chunk-MAE rankings.** The between-cell spread (1.17 deg)
   sits *below the instrument's own estimator seam* (1.55–1.91 deg).
   The 8/100 collapsed cell read within ~1 point of the healthy
   class; the 1/100 convict sorted *between* the 11/100 and 28/100
   cells. Rank-reads from this column are below the noise floor by
   construction, not by bad luck.
2. **Probe silence and probe endpoint levels.** The democlean cell's
   probe fell monotonically to 4.6848 — closing at the healthy
   onerig level (4.5266) — while grasping sat at 8/100 the whole
   run. A green curve is compatible with collapse; that is a
   measured fact, twice.
3. **The general class: any action-prediction score on
   demo-distribution states.** The miss is structural, not a tuning
   problem — collapse lives in a small systematic error that only
   compounds on the policy's *own* states in closed loop, and
   chunk-MAE dilutes a one-channel contact-frame error below the
   estimator seam. Externally replicated: raw validation MSE at
   Spearman −0.61 vs rollout success with in-family sign *flips*
   (CI-MSE, 27 checkpoints), independently reproduced by PolaRiS.
4. **Telemetry-only success flags.** 32–48% false-positive among
   flagged successes with scripted policies in clean sim; PhAIL's
   42% telemetry/operator disagreement replicates it on real
   hardware. Flags propose; exteroceptive reads decide.
5. **Uncalibrated sim absolute numbers and world-model scores.**
   Every rollout-free certificate banked so far was purchased with
   real rollouts (PolaRiS: per-checkpoint co-training against 20
   real rollouts/policy/env); the world-model tier's
   real-policy-ranking correlation is defined and never computed —
   screen ≠ certificate (Ctrl-World MMRV 0.22 is the proof).
6. **Single-scalar aggregations that hide the CDF.** Macro-AUC and
   RMST rank the same three models in opposite order on the same
   data. The full panel is the claim; the scalar is garnish.

## What v0 leaves open (the v1 slots)

- **Sim-vs-rig calibration is the standing unknown.** Rollout-vs-
  offline closed; rollout-vs-*rig* stays open until the bench runs.
  The riders that make it cheap are already banked for rig day: a
  2–5 min workspace scan (unlocks the PolaRiS scan-to-sim route
  retroactively), the 46-column servo-register log (NeuralActuator's
  virtual force sensor, our exact arm), one teleop block.
- **The Squint tier's pre-registration** — sim-adaptation sanity
  arm, which camera/overlay variant, and what a twin success rate is
  allowed to claim. Tier decision pends the wrist-transfer screen's
  outcome.
- **The verdict-grid grammar on the rig.** The sim cells' frozen
  ≥X / ≤Y / ambiguous-band grids have worked; v1 should state the
  rig equivalent in KS-on-CDF terms (what effect size + p closes a
  branch early, what falls to the ambiguous band's no-claim rule).
- **Precision ceiling *c*** — adopted as a rig-phase headline
  instrument; not computable pre-rig.

*Related: [probe-decoupling note](2026-08-21-probe-decoupling-note.md)
(the standing rule + both banked misses),
[Squint](../papers/squint.md) (the twin substrate),
[PhAIL](../papers/phail.md) (the statistical protocol),
[silent failures](../papers/silent-failure-observability.md) (label
audits), [rollout-free eval](../papers/rollout-free-eval.md) (why
certificates cost rollouts),
[idea #16](../ideas/16-rig-transfer-benchmark.md) (the ledger this
consolidates).*
