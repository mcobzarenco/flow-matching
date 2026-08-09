# Silent failures: how much of the truth is in the joints?

*Read 2026-08-09 (lit slice `lit-radar-0813`, priority 3: #6/#16
verifier family). Paper:
[2606.03134](https://arxiv.org/abs/2606.03134) — "How Visible Are
Silent Manipulation Failures? An Observability Study of False-Success
Detection in Simulated Robot Episodes" (Aarav Bedi, sole author,
4-page arXiv-only short, v1 2026-06-02, code released). A small
controlled measurement study, not a method paper — read for its
numbers and its unusually honest caveat, both of which bear on
verifier design.*

**The paper in plain words.** Robots often *think* they succeeded — a
cube placed two centimeters off, a peg that looks seated but isn't —
and these "false successes" quietly poison any dataset or reward
signal built from the robot's own success check. This paper builds a
deliberately tiny testbed to ask one narrow question: once the robot
has flagged success, how much of the information needed to overturn
that flag lives in the arm's own motion traces (proprioception), and
how much requires actually looking at the scene? The answer is
task-shaped: for carrying a cube from A to B, the joint traces alone
recover 97% of the false successes; for precision peg insertion they
recover only 65%, while even three crude pixel measurements of the
final image recover 94%. And then the author does something rare —
measures that the proprioceptive signal separating the classes is
about one part in a thousand in size, and warns that real encoder
noise would bury it.

## What it contributes

- **A clean operationalization of "false success"**: scripted
  (deterministic) policies on two bimanual-ALOHA sim tasks, failures
  induced *only* by physics perturbations (object jitter 0.5–1.5 cm,
  yaw 5–15°, friction 0.15–0.9 — never label edits); a cheap
  proprio-only success heuristic plays the robot's flag; privileged
  simulator state provides ground truth; the dataset keeps **only
  flagged-success episodes** (500/task). The flag is startlingly
  leaky: **47.6% of flagged transfer successes and 31.8% of flagged
  insertions are false.**
- **A modality-controlled detector comparison** at deliberately
  minimal capacity: gradient-boosted trees over 39 joint-velocity
  summary stats (proprio) vs *three* features from the final camera
  frame — centroid x, y and pixel area by color threshold (vision).
- **A sub-noise-floor disclosure** most papers would omit: the
  velocity differences that separate true from false successes are
  ~1e-3 in normalized units with near-zero within-class variance
  (Cohen's d ≈ 0.99 from mean gaps of ~0.001) — a deterministic-sim
  artifact. Verbatim: the proprio numbers "should be read as
  optimistic upper bounds rather than as expected field
  performance."

## The experiments it ran

Held-out 150 episodes per task, seed 42, single seed:

| Task | Detector | Acc | False-success recall |
|---|---|---|---|
| Cube transfer | Proprio (39 feats) | 0.973 | 0.972 |
| Cube transfer | Vision (3 feats) | 0.987 | 0.986 |
| Peg insertion | Proprio | 0.760 | **0.646** |
| Peg insertion | Vision | 0.973 | **0.938** |

Also: per-window effect sizes show the separating signal exists from
the first 100 steps (in-sim), and a force-augmented detector was
tried and dropped for zero gain. Notably absent: any fused
proprio+vision detector, any online/streaming variant, AUROC, CIs.

## What transfers to us

- **The leaky-flag number is the headline for #16.** Any future rig
  benchmark that gates on a telemetry-style success check inherits a
  positive-label pool that is ~30–50% wrong *in a clean sim with
  scripted policies* — before learned-policy weirdness. RL from
  binary success flags (the whole #16 RL-pole roster trains on
  exactly that signal) reward-hacks the flag, not the task. The
  cheap mitigation this paper points at: **one final-frame
  exteroceptive check** — three hand-crafted pixel features recover
  94% of insertion false-successes. A verifier's job starts with
  auditing the success labels themselves.
- **For the #6 verifier ledger: capacity is not the bottleneck,
  modality is.** A GBT over summary stats suffices in-sim; what
  moved recall 29 points on the precision task was *looking at the
  final scene*, not a bigger model. This slots beside
  [VLA-Corrector](vla-corrector.md)'s external-judge result as a
  second axis: external AND exteroceptive, with the final/completion
  state as the cheapest sufficient observation — which is also
  exactly the anchor [StreamVLA](streamvla.md)'s gate attends to.
- **The caveat is itself the transferable method point**: when a
  detector's separating signal is orders of magnitude below the
  sensor noise floor of the deployment platform, in-sim recall
  numbers are fiction. Worth applying to any sim-validated verifier
  we ever consider importing.

## What doesn't transfer

- **The 0.97 proprio-only number.** By the paper's own account it is
  a noiseless-sim upper bound resting on 1e-3 velocity differences;
  do not cite it as evidence that proprio-only monitors work on
  hardware. The honest reading: proprio maybe suffices for gross
  transport, never for precision outcomes.
- **The detectors themselves** — offline, post-episode, scripted
  policies, disclosed-nowhere flag heuristic, 2 tasks, 1 seed. The
  study's value is the decomposition, not the artifacts.
- The vision detector's strength is near-tautological (color-keyed
  object segmentation in a clean sim ≈ oracle scene state); on real
  clutter the modality gap will be smaller than 29 points.

## Which idea/arm it fed

[#16](../ideas/16-rig-transfer-benchmark.md) — bench design
constraint banked: success labels need an exteroceptive audit
(final-frame check), since telemetry flags run 32–48% false-positive
even in clean sim; caveat class noted for the RL pole's binary
success signals. [#6](../ideas/06-aux-attribution.md) —
verifier-ledger entry: modality > capacity; final-state exteroception
is the cheapest sufficient signal for precision outcomes; pairs with
VLA-Corrector's decoupled-judge constraint. Cross-refs:
[VLA-Corrector](vla-corrector.md), [StreamVLA](streamvla.md),
[Robot Critics](robot-critics-small-stuff.md).
