# Design: the wrist-transfer screen — does wrist dishonesty move closed-loop behavior at all?

*2026-08-14, drafted 17:2x–17:3xZ. Queue item `wrist-transfer-screen-design` — the
[renderer-class decision
brief](2026-08-14-renderer-class-decision-brief.md)'s move #2, design
only. **Nothing here is registered and nothing launches**: the GPU
reserve stands, and execution is a separate owner-visible pre-reg
when a window opens. This memo fixes the screen's arms, instrument,
seeds, statistics, gates, budget, and — per the queue item's own
condition — its falsifiers, before any execution item is queued.*

## Plain words

We know our simulator's wrist camera looks very fake exactly where it
matters — mid-manipulation, staring at the robot's own arm — and we
know a renderer upgrade that might fix it would be expensive. What we
don't know is whether the fakeness *costs anything*. Every number we
have is about how easily a neural network can tell sim from real; none
is about whether a robot-control policy actually performs worse
because of it. This memo designs the experiment that connects the
two: run the same policy through the same 100 simulated episodes
several times, identical in every way except what the wrist camera
feeds it — the normal view, a blacked-out view, a view where only the
arm's appearance is corrupted — and measure whether task behavior
moves. Each corrupted view is also scored on our existing
fakeness-meter, so the result comes out as a curve: behavior change
per unit of wrist-camera honesty. If the curve is flat, the wrist gap
is a tolerated artifact and the expensive renderer fix loses its main
justification. If it is steep, the fix finally has a price tag in the
currency that matters. One honest caveat is designed in rather than
hidden: today's policies never fully complete the sim task, so the
screen measures graded progress, not success percentages — and it
includes a policy variant fine-tuned on sim-rendered frames
specifically to pull performance up into a range where the question
is answerable.

## 1. The unpriced link

The banked chain, and where it stops:

- Wrist camera at manipulation poses reads **0.877** knn5 AUROC vs
  held-out real (calibration direction understating it), vs **0.523**
  at reset ([rollout-pose
  read](2026-08-14-prereg-sim-rollout-pose-wrist.md)). The paired
  rider puts the gap on the pose switch; the [content
  split](2026-08-14-prereg-sim-wrist-content-split.md) rules out
  scene content — **the rendered arm carries it**.
- The whole instrument is an *encoder-honesty proxy* (can the er_60k
  trunk's features separate sim from real?). The north star is
  transfer on the rig. **No read connects the proxy to behavior.**
- The [renderer decision](2026-08-14-renderer-class-decision-brief.md)
  is priced on both ends *in proxy units only*: −0.355 addressable on
  the wrist. Whether that is worth the tier-2 validation tail depends
  on whether proxy dishonesty moves task performance — this screen's
  question.

## 2. Substrate decision: our sim first, the SO-101 twin as the successor tier

**Chosen substrate: the banked sim100 closed-loop harness verbatim**
(v3 visual stack, v0 physics + sysid'd servos, frozen spawn seeds
0–99, 30 s / 30-replan episodes, paired design, `hold` floor and
strike gates). Reasons it beats the alternative for the *first* read:

- The 0.877 was measured **on this renderer's frames** — degrading
  this wrist feed degrades the exact pixels the proxy indicted.
- Deterministic draw-0 rollouts make every within-screen comparison
  **bit-paired**: all treatment arms replay W0's exact seeds through
  the exact policy, so per-seed deltas are pure treatment effect.
  (A git audit for this design found the banked sim100 rows are
  *not* a valid bit-anchor: they predate the fitted wrist lens and
  the v3 wrist path — config drift, the known class. W0 is
  therefore a **fresh in-run baseline**, and the banked `ftrig4k`
  numbers serve as a sanity band, not a bit gate — §6.)
- Zero new infrastructure beyond a wrist-feed hook (§4).

**The floor problem, stated up front.** The [sim100
results](2026-08-12-sim100-results.md) banked **0/500 successes** —
binary success rate has no dynamic range in our sim today. The queue
item's phrase "moves SUCCESS RATE" is therefore not directly
measurable here yet, and this design does not pretend otherwise:

- The registered primary currency is the sim100 **graded** behavior
  set (progress cm, best-point, engagement, grip, knock-away — §5),
  which does separate policy families (the banked table spans −0.73
  to +0.08 cm mean progress, 0–56% contact).
- The **sim-adaptation arm** (§3, the queue item's required sanity
  arm) exists precisely to lift the policy into a band where graded
  deltas — and possibly nonzero success — are readable.
- If even the sim-adapted policy stays pinned at 0 success, the
  success-rate form of the question escalates to the **Squint SO-101
  twin** ([lit 0819](../papers/squint.md)): success-predicated,
  MIT-licensed, our exact arm, policies trainable into the measurable
  band in minutes — at the price of a class-level rather than
  our-policy answer, bench cells held in the 20–80% band with ≥50
  trials/cell (the 2606.08881 anti-patterns). That tier is the
  documented successor, not part of this screen; its CPU-side
  preflight is queued separately (`squint-twin-preflight`).

## 3. Arms

Two policy rows × a wrist-feed column ladder, plus one positive
control. Every arm: seeds 0–99, deterministic draw-0, identical
physics, identical top-view path — **only the wrist frame handed to
the policy changes**.

**Policy rows:**

- **P1 `ftrig4k`** (student + rig fine-tune, euler-1): the only
  banked arm whose contact tilts toward the goal (+0.08 cm mean
  progress, 47/100 engaged) — the most behavior per GPU-h available
  today, and deterministic (euler-1, draw-0), so per-seed deltas are
  pure treatment effect.
- **P2 `simft`** (the sim-adaptation sanity arm): the same student
  fine-tuned on **sim-rendered replays of real trajectories** — v3
  frames rendered at the recorded `observation.state` of real
  reference-half episodes 0–25 (the rollout-pose machinery, banked),
  paired with the recorded real actions. BC on real behavior wearing
  sim pixels: the one fine-tune our data supports with zero
  successful sim demos. Recipe = the `ftrig4k` fine-tune script with
  the dataset swapped; ~2 GPU-h class. **Contamination guards:**
  trains only on episodes 0–25 (the honesty probe's held-out pool
  and the 100 pose slots live in episodes 26–49); the honesty
  instrument itself uses the frozen er_60k trunk, which never
  retrains.

**Wrist-feed columns** (pure `obs.wrist` transforms, physics and
state untouched):

- **W0 — classic v3 render** (baseline; bit-replication anchor).
- **W1 — blackout** (zeros): the does-it-listen bracket. Maximal,
  structural, not appearance-class — an endpoint, not an
  interpolation point.
- **W2 — frozen reset frame** (stage-3 optional): plausible static
  content, closed-loop visual feedback removed — separates "needs
  the wrist stream" from "needs any wrist-shaped pixels".
- **W3 — arm-appearance corruption**: per-tick wrist segmentation
  pass (arm+gripper geom ids), strong Gaussian blur *inside the arm
  mask only* — silhouette and mean color survive, shading structure
  and specular texture die. The closest accessible analogue of the
  surviving relief-and-light-transport hypothesis, applied in the
  accessible direction (§8).
- **W4 — measured-materials stack ON** (`arm_photometrics='v1'` +
  `mount_material='v1'`; stage-3 optional): the one treatment whose
  honesty delta at manipulation poses is *already banked*
  (+3.99e-07 per-slot, CI excluding zero, the [wrist material
  read](2026-08-14-prereg-sim-wrist-view-material-read.md)) — a
  free, within-renderer point on the curve.

**T1 — positive control (top blackout, 25 seeds, P1 only):** the
policy's banked goal-directedness demonstrably rides on its inputs;
blacking the *top* view must move behavior. If it doesn't, the
harness cannot detect view-fidelity effects at this competence floor
and the screen aborts with no claims (F-instrument, §7).

**Honesty placement (the x-axis, ~0.02 GPU-h class per arm):** every
wrist transform is also applied to the banked 100 manipulation-pose
wrist renders and scored with the established knn5 harness against
the manipulation reference — placing each arm on the same axis as
the banked 0.877/0.523 anchors. The screen's deliverable is
Δbehavior vs Δhonesty, not a bag of ablations.

## 4. Implementation note (the only new code)

A `--wrist-transform {none,blackout,freeze,arm_blur}` hook in the
rollout driver, applied to `obs.wrist` after `observe()` and before
policy packing (`rollout_sim_parallel.py` builds
`SimObservation(top, wrist, state)` — one seam, both drivers). W3
adds a per-tick wrist segmentation render in that arm only. Oracles
before any run (charter: oracles after math-adjacent changes):
golden-frame test per transform; a `none` rollout replays a banked
seed bit-identically; a transformed rollout's *qpos trace at tick 0*
matches `none` (transforms touch pixels, never state); W3 mask
visual spot-check on 3 banked pose slots.

## 5. Instrument, seeds, statistics

- **Primary**: paired per-seed **Δprogress** (final initial−final cm,
  the sim100 primary), treatment − W0, per policy row.
- **Secondary** (all banked sim100 channels): Δbest-point, engagement
  flip (moved ≥0.5 cm), two-sided-pinch rate, knock-away rate,
  success ticks (recorded; claims only if the floor lifts, §2).
- **Seeds**: the banked frozen 0–99, **same across every arm** —
  paired comparability is the point, per the standing seed policy
  (same seed for comparability; fresh seeds are for variance
  questions this design doesn't ask). Draw-0 deterministic
  throughout; no sampling noise in the deltas.
- **CI**: per-seed paired deltas, bootstrap CI95 over seeds (10k
  resamples, the established recipe), n=100 per cell (25 for T1).
  With deterministic policies the CI covers seed-distribution
  spread only — the per-seed delta itself is exact.
- **Curve read**: arms ordered by measured honesty coordinate (§3);
  the headline is the sign and monotonicity of Δbehavior over
  Δhonesty, W1 reported as the bracket endpoint, never pooled into
  the appearance-class trend.

## 6. Gates and aborts (frozen at pre-reg time)

**ABORT (no claims) unless ALL of:**

- **W0 determinism**: P1×W0 run twice on 10 seeds at stage-1 entry,
  per-seed `final_cm` bit-equal — the pairing premise is machine-
  checked before the screen spends anything. (The banked sim100
  `ftrig4k` rows are NOT the bit-anchor — they predate the fitted
  lens + v3 wrist; registered config drift. Sanity band instead:
  P1×W0 mean progress in **[−0.3, +0.5] cm** and engagement in
  **[25, 70]/100** vs banked +0.08 / 47 — outside it, the visual-
  config delta changed the policy's regime and the read pauses for
  an in-channel note before proceeding.)
- **`hold` floor** replicates (|mean progress| ≤ 0.01 cm) and reset
  strikes = 0, per the sim100 validity gates.
- **T1 moves**: top-blackout Δ(engagement or |progress|) CI95
  excluding zero. Otherwise F-instrument (§7).
- **Honesty placement sane**: W1 and W3 place *less honest* than W0
  on the 100-slot read (W3−W0 positive, CI excluding zero). A
  corruption the encoder can't see isn't a treatment.
- Cross-arm pairing check: identical `spawn_xy` per seed across all
  arms (bit).

**Tripwires** (recorded, not gated): per-arm latency (the transform
must not change replan cadence); W3 mask coverage per tick.

## 7. Falsifiers and their decision consequences

Stated per the queue item's condition, before any execution pre-reg:

- **F-instrument** — T1 null: the harness can't detect *any*
  view-fidelity effect at this competence floor. Screen aborts, no
  transfer-link claim in either direction; the question escalates to
  the twin tier (§2) where competence is buildable.
- **F-null** — wrist channel dead: W1 ≈ 0 for **both** policy rows
  while T1 moves. This policy class simply doesn't consume the wrist
  view in closed loop → 0.877 cannot be costing behavior →
  **tier-2's north-star payload evaporates** (its proxy-unit case
  survives, but the owner buys it knowing that). P2 is what makes
  this null meaningful — on P1 alone, "ignores the wrist" and "too
  sim-OOD to use any of it" are confounded.
- **F-flat** — wrist consumed, appearance tolerated: W1 moves
  behavior but W3 (and W4 if run) sit at ≈ 0 despite measured
  honesty displacement. The dishonesty *class* is tolerated —
  same consequence as F-null, stronger form (the policy provably
  uses the wrist stream and provably doesn't care how the arm is
  shaded).
- **F-live** — the curve has slope: appearance-class treatments move
  behavior, ordered with honesty. The proxy→behavior link is priced;
  the [tier-2 pilot](2026-08-14-renderer-class-decision-brief.md)
  inherits a behavior-unit payload estimate (slope × the 0.877→0.523
  span, direction caveat §8) and gets upgraded accordingly.

## 8. What this screen cannot say (registered limitations)

1. **Direction asymmetry** — every accessible treatment makes the
   wrist *less* honest than baseline; the renderer fix would make it
   *more*. The curve is measured on the degradation side and the
   payload estimate extrapolates its local slope across W0 into the
   0.877→0.523 span. Registered as the design's central assumption;
   partially mitigated by W4 (a banked *within-renderer* honesty
   displacement) and by preferring appearance-class treatments (W3)
   over structural ones (W1) for the trend.
2. **The floor** — behavior currency is graded progress, not success
   rate, until the floor lifts (§2). A flat curve at a 0-success
   floor is weaker evidence than a flat curve at 40% success; the
   escalation path is designed in, not discovered later.
3. **Sim-only, one task** — the screen prices the link inside the
   sim's boat task. It cannot certify the rig; it can only remove
   (or confirm) the *current* justification for a renderer-class
   spend before real money moves.
4. **Class-level at best on P2** — `simft` changes what is measured
   (a sim-adapted variant, not the deployed policy). That is the
   price of escaping the floor, and it is why P1 runs unmodified
   alongside.

## 9. Ladder and budget (measured pace: 0.0094 GPU-h/episode)

| stage | cells | episodes | GPU-h (est) | decision at boundary |
|---|---|---|---|---|
| 0 | transform hook + oracles + honesty placement (W1, W3) | 0 | ~0.1 | placements sane (gate §6) else redesign |
| 1 | P1 × {W0, W1, W3} + T1(25) | 325+20 | ~3.3 | W0 determinism + sanity band; T1 gate; first W1/W3 read |
| 2 | `simft` fine-tune + P2 × {W0, W1, W3} | 300 | ~2 + 2.8 | the F-null adjudication needs both rows |
| 3 (cond.) | W2 + W4 on the live-channel row (+`teacher80k` × W1 optional) | ≤400 | ~3.8 | only if stage 1–2 shows a live wrist channel |

Worst-case ≈ **12.0 GPU-h; registered gate ≤ 14** (headroom so the
gate never truncates stage 3 mid-read), expected ~6–9
(stage 3 is conditional). Stage boundaries are hard stops with
in-channel posts. All stages CPU-preparable during the reserve;
nothing launches before the in-channel GPU release and a posted
pre-reg freezing §5–§7 verbatim.

## 10. Status

Design only — **not registered, not launched**. Execution queues as
`wrist-transfer-screen-run` (blocked on the GPU release), with this
memo as its pre-reg skeleton; `squint-twin-preflight` queues as the
CPU-side successor-tier preparation. Both dispositions are in
`queue.json`; the renderer tier-2 pilot remains a separate,
owner-gated item per the decision brief.

![Screen schematic: arms on the honesty axis, the measurable side vs the extrapolation span, and the staged budget](https://mcobzarenco-fontaine-reports.static.hf.space/chart__wrist_transfer_screen_design.png)

*Left: the screen's logic — treatment arms placed on the banked
honesty axis (anchors 0.523/0.877), the degradation side where the
curve is measured, and the renderer span where its slope is applied.
No y-values exist yet; the two sketched outcomes are the F-flat and
F-live verdicts of §7. Right: the staged GPU-h ladder against its
≤14 gate (erratum 18:5xZ: this caption originally said "≤12" —
the §9 text's registered gate ≤ 14 / worst-case 12.0 was always
correct), stage 3 conditional.*
