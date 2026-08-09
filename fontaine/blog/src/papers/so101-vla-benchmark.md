# SO-101 VLA benchmark: 320 real rollouts, a leaky taxonomy — and the eval episodes are quietly sitting on the Hub, unlabeled

*Read 2026-08-09 (lit slice `lit-radar-0819`, priority hook "SO-101
failure benchmark"). Paper:
[2606.08881](https://arxiv.org/abs/2606.08881) — "Benchmarking
Vision-Language-Action Models on SO-101: Failure and Recovery
Analysis" (Yi Yu, Xinchuan Qiu; Hiroshima University; arXiv cs.RO/cs.AI,
submitted 2026-06-07, v2 2026-06-11; arXiv nonexclusive-distrib
license; 13 pages, 2 authors, ACM template with the "Woodstock, NY"
placeholder still in it — read as an unreviewed student-lab preprint,
which it survives better than most).*

**The paper in plain words.** Take the cheapest serious robot arm in
the community — the SO-101, the same one on our bench — collect 100
teleoperated demonstrations for each of four tabletop tasks, fine-tune
four policies on them (π0.5, SmolVLA, Wall-X, and an ACT baseline),
and run every policy 20 times per task on the physical arm: 320 real
rollouts. Instead of only counting successes, watch *how* each run
fails: did the gripper lose the object, did the arm loop the same
motion forever, did the policy carry on as if a failed grasp had
succeeded, or did it just miss a tight insertion? And when something
went wrong mid-episode, did the policy ever fix it by itself? The
headline: the big pretrained policies win (π0.5 56.25% average
success vs ACT's 33.75%), almost every failed episode involves
low-level execution trouble, and the ability to recover from a
mid-episode mistake separates the policies far more sharply (π0.5
recovers 30.77% of the time, SmolVLA 3.23%) than success rate does.
The fine print: 20 episodes per cell is very noisy, the failure
labels contradict the paper's own labeling rule, and the released
artifact is the *training* demos — though we found the evaluation
rollouts themselves uploaded to the same HuggingFace account,
unlabeled and uncited.

## What it contributes

- **A real-SO-101, all-real-hardware benchmark.** Four tasks — Pen
  Transfer (control fidelity), Selective Color Sorting (language
  grounding under distractors), Multi-Object Packing (long-horizon
  consistency), Precision Pen Placement (tight-tolerance insertion) —
  each with 100 in-house teleop demos, each policy fine-tuned per its
  own recipe, 20 hardware trials per model-task pair, 320 episodes
  total. Genuinely SO-101 (LeRobot `so101_follower`), genuinely real:
  no sim anywhere in the pipeline.
- **A four-way failure taxonomy** applied to every failed episode:
  Grasp Instability, Repetition Loop, State Mismatch (policy keeps
  executing its plan after reality diverged — e.g., transporting
  nothing after a failed grasp), Precision Misalignment.
- **A semantic/execution split**: State Mismatch is called
  "semantic", the average of Grasp Instability and Repetition Loop is
  called "execution", Precision Misalignment is excluded. This split
  is where the abstract's "execution instability is the dominant
  failure source" lives.
- **A recovery metric**: recoveries / recovery opportunities, where
  an opportunity is any mid-episode failure event. No annotation
  protocol, annotator count, or inter-rater agreement is reported for
  either the failure labels or the opportunity counts — all of it is
  human judgment by (presumably) the two authors.

## The experiments it ran

- **Success rates (Table 3, n=20 per cell).** Average: π0.5
  **56.25%**, Wall-X **51.25%**, ACT **33.75%**, SmolVLA **32.5%**.
  Per task: Pen Transfer 70–95% (near ceiling for everyone), Color
  Sorting 0–10% (near floor for everyone), Packing 10–55% (the only
  task that separates the field), Placement 45–80%. Note SmolVLA —
  the LeRobot-ecosystem model closest to our stack — loses to plain
  ACT on average; the paper itself says the gap is within the
  20-episode noise. At n=20 a 50% success estimate carries a ±22pp
  CI95 half-width; no interval or test appears anywhere in the paper.
- **Failure incidence (Table 4, over failed episodes).** ACT: Grasp
  94.34%, Repetition 92.45%, State Mismatch 98.11%, Precision
  15.09%. π0.5: 91.43 / 91.43 / **45.71** / 14.29%. Wall-X: 100 /
  100 / 61.54 / 0%. SmolVLA: 92.59 / 94.44 / 70.37 / 7.41%. The
  clean monotone story: state mismatch drops as VLM pretraining gets
  stronger (98 → 70 → 62 → 46%), while grasp/repetition stay pinned
  above 90% for everyone.
- **Recovery (Table 5).** π0.5 30.77% (8/26 opportunities), Wall-X
  20.51% (8/39), ACT 6.45% (2/31), SmolVLA 3.23% (1/31). We
  recomputed all four from the episode-level Table 6 counts and they
  check out exactly — the paper's internal arithmetic is solid even
  where its definitions aren't.
- **Episode-level table (Table 6)** gives per-model-per-task counts
  of successes, each failure mode, opportunities, and recoveries —
  the most reusable numbers in the paper.

## Corrections — what the banked hook got wrong

Our hook said "real-world failure taxonomy + recovery analysis on the
SO-101, execution instability dominant." Clause by clause:

- **Hardware: CONFIRMED.** Real SO-101, single arm, single lab, all
  four tasks. Not SO-100, not sim.
- **Policies: π0.5, not π0** (plus SmolVLA, Wall-X, ACT; no GR00T).
  And they are fine-tuned on **100 in-house demos per task** (400
  demos, ~86 minutes of data total) — *not* community-data
  generalists. Every success rate here is a small-data
  per-task-specialist number; it says nothing about policies trained
  the way ours are.
- **"Single primary failure mode" is false in their own tables.**
  Section 2.5.2 says each failed trial gets one primary mode; Table 4
  rows sum to ~300%, and Table 6 makes it explicit (ACT Color
  Sorting: 20 failures carrying 19 + 19 + 20 + 6 = 64 labels). The
  taxonomy is multi-label incidence, not a distribution — so "X% of
  failures are grasp instability" reads very differently than the
  text implies.
- **"Execution instability dominant" is their claim (abstract,
  verbatim) but it is weaker than it sounds.** With grasp and
  repetition labels pinned at 91–100% of failed episodes for every
  policy, the execution category is nearly saturated — a label that
  fires on essentially every failure discriminates nothing. And for
  ACT the "semantic" category (98.11%) actually *exceeds* the
  execution average (93.4%), so the cleanest true statement is:
  execution labels saturate for everyone, and semantic failure is
  what *varies* — falling with VLM pretraining strength.
- **The split itself is leaky.** State Mismatch's own definition —
  "attempting object transport after a failed grasp" — is a
  *downstream consequence of an execution failure*. Their
  semantic/execution decomposition double-counts single causal
  chains; treat the two axes as correlated labels, not disjoint
  causes.
- **Scale: 320 rollouts, 4 tasks, 4 policies, 20 episodes per cell.**
  An order of magnitude below ArmnetBench's 2,288 before we even ask
  about labels.

## Release audit: better than the paper admits, worse than we hoped

- The paper cites exactly one artifact: the **400 training demos**
  ([HF collection](https://huggingface.co/collections/Qiu-Xinchuan/400-so-101-vla-evaluate-dataset),
  resolves, 4 LeRobot-v3 datasets, 100 episodes each, Apache 2.0,
  3 cameras at 30 fps). Useful as clean single-rig SO-101 teleop, not
  as a calibration corpus.
- **Undocumented find:** the same account hosts 20 `rollout_*`
  datasets — `rollout_{pi05,smolvla,act,wallx}_{task}` for **all 16
  model-task pairs** (plus a duplicate and three `new_test` extras),
  20 episodes each, LeRobot v3, three cameras (top/front/right),
  6-D joint-space actions + state at 30 fps, pen-task episodes
  time-capped at exactly 600 frames (20 s). These look like the
  actual 320 evaluation episodes behind Tables 3–6. Nothing in the
  paper mentions them.
- **But: zero labels.** The rollout schemas contain no
  success/failure/recovery fields; episode metadata is just the task
  string. Per-episode ground truth exists only as aggregate counts in
  Table 6. Licensing on the rollout sets is inconsistent (some
  Apache 2.0, some untagged). No code, no evaluation scripts, no
  annotation guidelines.
- **Verdict on "second calibration corpus": PARTIAL.** The raw
  material is real and LeRobot-native — 320 on-policy SO-101 failure
  and success episodes from four architectures including two
  flow-matching VLAs — but we would have to label it ourselves from
  the videos (feasible: ~2–3 hours of footage; Table 6 pins the
  per-cell totals we should recover, and saturated cells like ACT
  Color Sorting are nearly self-labeling). It is a labeling
  afternoon away from being a small second corpus, not a drop-in one.

## What transfers to us

- **The state-mismatch gradient is a probe-target argument.** The one
  failure class that varies across policies (98% → 46%) is exactly
  the "policy's internal state has diverged from reality" class —
  the thing a hidden-state probe on trunk residuals should see if it
  sees anything. The saturated classes (grasp, repetition) are
  detectable *without* hidden states: repetition loops from action
  periodicity, grasp instability from gripper/proprio signals. That
  hands us both the target and the cheap baselines our #6 gate must
  beat.
- **On-policy SO-101 failure footage from a flow-matching VLA**
  (π0.5, SmolVLA rollouts) is the closest publicly available thing
  to what our own policy's failures will look like — useful probe
  training/eval fodder once labeled.
- **Bench-design lessons for #16, mostly negative examples:** two of
  their four tasks are wasted on ceiling (Pen Transfer 70–95%) or
  floor (Color Sorting 0–10%); only mid-band Packing (10–55%)
  discriminates. n=20 per cell (±22pp) is too small to rank policies
  — their own SmolVLA-vs-ACT caveat concedes it. Success criteria in
  their Table 2 are one vague line each ("manipulation errors").
  Concrete #16 takeaways: pilot tasks into the 20–80% band before
  committing trials, n≥50 per cell or paired designs, pre-registered
  multi-label vs primary-label failure annotation, and an
  operationalized recovery-opportunity definition (theirs is
  unspecified human judgment).

## What does NOT transfer

- **No offline↔rollout calibration signal.** No offline metric of any
  kind appears — no action MAE, no validation loss, nothing to
  correlate with the 320 rollout outcomes. The read our panel
  programme most wants is simply not in this paper (and with
  per-task-specialist policies it would not have mapped onto our
  generalist setting anyway).
- **The success-rate numbers.** 100-demo per-task specialists on one
  rig; not evidence about community-data generalists like ours.
- **The taxonomy as ground truth.** Unspecified annotation protocol,
  multi-label saturation, and a leaky semantic/execution split mean
  we should re-derive labels under our own scheme if we use the
  rollouts — not inherit theirs.

## Which idea it feeds

- **Idea #6 (hidden-state failure-detection gate).** Feed: (i) the
  probe's marginal value should be *claimed on the state-mismatch
  class specifically* — periodicity and gripper-signal baselines
  plausibly cover the saturated execution classes for free; (ii) a
  candidate second corpus exists: the 16 `rollout_*` datasets (320
  episodes) pending a self-labeling pass, sitting next to
  ArmnetBench's 2,288 — but it enters the gate only after we label
  it.
- **Idea #16 (rig-transfer benchmark).** Feed: difficulty-band task
  selection (pilot to 20–80%), ≥50 trials per cell, pre-registered
  annotation scheme, operationalized recovery events. Their four-task
  execution-dimension framing (control fidelity / grounding /
  temporal consistency / precision) is a reasonable starting axis
  set; their statistical treatment is the anti-pattern.
