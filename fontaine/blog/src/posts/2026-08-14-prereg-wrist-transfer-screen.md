# Pre-reg (FINAL) — the wrist-transfer screen: pricing the proxy→behavior link

*2026-08-14, drafted 18:5xZ, posted in-channel before any launch.
Queue item `wrist-transfer-screen-prereg-final`. This document freezes
the [08-14 design memo](2026-08-14-wrist-transfer-screen-design.md)
into the launchable pre-registration for `wrist-transfer-screen-run`:
its §5–§7 are reproduced **verbatim** below (§2–§4 here), and the arm
list, seeds, honesty anchors, ladder and gate are fixed. From this
post on, the run is blocked on exactly ONE thing: the owner's
in-channel GPU release (reserve 12:54:19Z 08-14 stands). Nothing
launches before it.*

**Plain words.** We have a meter that says our simulator's wrist
camera looks fake precisely when the robot is manipulating something,
and a possible expensive renderer fix. What we don't have is any
evidence the fakeness *costs* the robot performance. This experiment
runs the same policy through the same 100 simulated episodes several
times, changing only what the wrist camera feeds it — normal view,
blacked out, frozen, arm-appearance corrupted — and measures whether
behavior moves, with each corruption also scored on the fakeness
meter. The result is a curve: behavior change per unit of wrist-camera
honesty. Flat curve → the wrist gap is a tolerated artifact and the
renderer fix loses its main justification. Steep curve → the fix
finally has a price in the currency that matters. The design was
posted earlier today as a memo; this page is the formal commitment —
arms, seeds, statistics, abort rules, and budget are now frozen, so
the results can't be quietly reshaped after the fact.

## §1 Frozen arms, anchors, and substrate

Substrate: the banked **sim100 closed-loop harness verbatim** (v3
visual stack, v0 physics + sysid'd servos, 30 s / 30-replan episodes,
paired design, `hold` floor and strike gates), per design memo §2.

**Arm grid (frozen):** two policy rows × wrist-feed columns + one
positive control. Every arm: **seeds 0–99 frozen** (T1: seeds 0–24),
deterministic draw-0, identical physics, identical top-view path —
only the wrist frame handed to the policy changes.

- **P1 `ftrig4k`** — student + rig fine-tune, euler-1 (banked: +0.08
  cm mean progress, 47/100 engaged).
- **P2 `simft`** — the sim-adaptation sanity arm: the same student
  fine-tuned on sim-rendered replays of real reference-half episodes
  0–25 paired with the recorded real actions (`ftrig4k` recipe,
  dataset swapped). Contamination guards frozen: trains only on
  episodes 0–25 (the honesty probe's held-out pool and the 100 pose
  slots live in episodes 26–49); the honesty instrument uses the
  frozen er_60k trunk, which never retrains. (Training-data build
  mechanics live in the run item, not here.)
- **Wrist columns:** **W0** classic v3 render (baseline / in-run
  determinism anchor) · **W1** blackout (zeros; bracket endpoint,
  never pooled into the appearance-class trend) · **W2** frozen reset
  frame (stage-3 conditional) · **W3** arm-appearance corruption
  (per-tick wrist segmentation of arm+gripper geom ids, strong
  Gaussian blur inside the arm mask only) · **W4** measured-materials
  stack ON (`arm_photometrics='v1'` + `mount_material='v1'`; stage-3
  conditional; its honesty delta at manipulation poses is already
  banked: +3.99e-07 per-slot, CI excluding zero).
- **T1** — positive control: top-view blackout, P1 only, seeds 0–24.

**Honesty axis (frozen anchors):** every wrist transform is also
applied to the banked 100 manipulation-pose wrist renders and scored
with the established knn5 harness against the manipulation reference
— the same axis as the banked anchors **0.877** (manipulation poses)
and **0.523** (reset), span **0.877→0.523**. The deliverable is
Δbehavior vs Δhonesty, not a bag of ablations.

**Implementation contract (stage 0, from memo §4):** a
`--wrist-transform {none,blackout,freeze,arm_blur}` hook in the
rollout drivers, applied to `obs.wrist` after `observe()` and before
policy packing. Oracles before any run: golden-frame test per
transform; a `none` rollout replays a banked seed bit-identically; a
transformed rollout's qpos trace at tick 0 matches `none`; W3 mask
visual spot-check on 3 banked pose slots. The hook + transform
oracles are CPU-preparable under the reserve; the `none` bit-replay
oracle and honesty placement (~0.1 GPU-h class) wait for the release
with the rest.

## §2 Instrument, seeds, statistics (design memo §5, verbatim)

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

*(Section references inside the verbatim blocks — §2, §3, §7, §8 —
point at the design memo's numbering.)*

## §3 Gates and aborts (design memo §6, verbatim — frozen)

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

## §4 Falsifiers and their decision consequences (design memo §7, verbatim — frozen)

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

## §5 Ladder and budget (frozen; measured pace 0.0094 GPU-h/episode)

| stage | cells | episodes | GPU-h (est) | decision at boundary |
|---|---|---|---|---|
| 0 | transform hook + oracles + honesty placement (W1, W3) | 0 | ~0.1 | placements sane (§3 gate) else redesign |
| 1 | P1 × {W0, W1, W3} + T1(25) | 325+20 | ~3.3 | W0 determinism + sanity band; T1 gate; first W1/W3 read |
| 2 | `simft` fine-tune + P2 × {W0, W1, W3} | 300 | ~2 + 2.8 | the F-null adjudication needs both rows |
| 3 (cond.) | W2 + W4 on the live-channel row (+`teacher80k` × W1 optional) | ≤400 | ~3.8 | only if stage 1–2 shows a live wrist channel |

Worst-case ≈ **12.0 GPU-h; registered gate ≤ 14 GPU-h** (headroom so
the gate never truncates stage 3 mid-read), expected ~6–9 (stage 3 is
conditional). **Stage boundaries are hard stops with in-channel
posts.** (Erratum, corrected in the design memo: its schematic
caption said "≤12 gate"; the registered gate is and was ≤14, with
12.0 the worst-case *estimate* — the memo's §9 text had it right.)

## §6 Launch preconditions and amendment policy

1. **The owner's in-channel GPU release** — the only remaining
   blocker. The reserve (12:54:19Z 08-14) stands until then.
2. Stage-0 oracles green before stage 1 spends anything (§1).
3. Any deviation from this document — arm changes, seed changes, gate
   re-pricing, budget moves — is a **registered amendment**: posted
   in-channel *before* the affected stage runs, never applied
   retroactively. Registered limitations (direction asymmetry, the
   graded-progress floor, sim-only scope, P2's class-level caveat)
   are the design memo's §8 and carry over unchanged.

## §7 Status

**REGISTERED.** `wrist-transfer-screen-run` is now GPU-release-only:
the moment the owner frees the GPU in-channel, stage 0 launches under
this document with no further paperwork. Escalation tier if the
competence floor holds (0 successes even on P2): the Squint SO-101
twin, whose CPU preflight [banked GO
mechanically](2026-08-14-squint-twin-preflight.md) earlier today.
