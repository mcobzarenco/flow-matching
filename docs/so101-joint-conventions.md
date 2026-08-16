# SO-100/101 joint conventions — the two orthogonal axes

Every SO-10x joint value in this project — dataset frames, checkpoint
normalization tables, sim seams, rollout commands — lives at a point in
a TWO-AXIS convention space. Confusing the axes (or assuming a single
"v3 format") is a recurring footgun: it produces tables that clamp
whole joints to their edges, models that move an arm in the wrong
direction, and sims that silently evaluate garbage. Read this before
touching normalization stats, converting a released checkpoint, or
wiring a new data source.

Primary source: the LeRobot backward-compatibility page
(<https://huggingface.co/docs/lerobot/en/backwardcomp>, the PR #777
section). The transforms below are quoted from it and verified against
real tables (§4).

## 1. Axis one — calibration convention (v2.1 ↔ v3.0)

Where each joint's ZERO sits and which direction is POSITIVE.

| | old (pre-PR #777, "v2.1 era") | new (v3.0) |
|---|---|---|
| zero position | arm fully extended horizontally | middle of each joint's range |
| boundary handling | ±180° wrap-around safeguards | none needed (mid-range zero) |

The official per-joint migration (old → new, expressed in degrees):

| Joint | Transform | Kind |
|---|---|---|
| `shoulder_lift` | `new = −(old − 90)` = `90 − old` | **sign flip + 90° shift** |
| `elbow_flex` | `new = old − 90` | **90° shift only** |
| every other joint (pan, wrist_flex, wrist_roll, gripper) | identity | unchanged |

Only TWO joints differ between calibrations — and they differ
DIFFERENTLY. The shift (elbow) is cheap to fix or adapt through; the
sign flip (shoulder_lift) is the expensive axis: it inverts what
"positive motion" means, so a model trained under one calibration
moves that joint the wrong way under the other, and no range
re-centering repairs direction.

## 2. Axis two — representation (`use_degrees`)

The UNITS values are expressed in, WITHIN whichever calibration:

| `use_degrees` | body joints | gripper |
|---|---|---|
| `True` | degrees | degrees |
| `False` | −100…100 (percent of calibrated range, mid-zero) | 0…100 |

This is a per-robot-config knob
(`lerobot/robots/so_follower/config_so_follower.py`;
`MotorNormMode.DEGREES` vs `RANGE_M100_100`), orthogonal to axis one:
degrees-vs-±100 is a pure per-joint linear rescale inside the same
calibration. The official v2.1-replay migration applies the axis-one
transforms AND sets `--robot.use_degrees=true` — two independent
decisions.

**Do not call ±100 "the v3 convention."** v3.0 names the calibration;
±100 names a representation available under it. A dataset can be
v3.0-calibrated and recorded in degrees — ours all are.

## 3. Where our artifacts sit

| Artifact | Calibration | Representation |
|---|---|---|
| rig `so101_pick_place_v2` / `_clean` | v3.0 | degrees |
| sim `demos_v0` / grasp shards | v3.0 (calibrated against the rig) | degrees |
| MolmoAct2 released checkpoint's recorded stats table | **v2.1** | degrees |
| lerobot controller at rollout | v3.0 | per `use_degrees` (ours: degrees) |

The mixture is one consistent space — no conversion layer inside
training. The only convention boundary in the project is the released
MolmoAct2 table (§4).

## 4. Fingerprints — recognizing a convention from the numbers

Verified values; use them to classify any new table or dataset:

| Joint | v2.1° (released table q01→q99) | mapped to v3.0° | v3.0° (our data) |
|---|---|---|---|
| `shoulder_pan` | −42.1 → 48.6 | −42.1 → 48.6 (identity) | −25 → 41 |
| `shoulder_lift` | 45.2 → 186.1, mean 125.8 | **+44.8 → −96.1** (descending — the flip) | −103 → +29 |
| `elbow_flex` | 35.4 → 173.6 | −54.6 → 83.6 | same family |
| `wrist_flex` | 4.9 → 93.4 | identity | — |
| `wrist_roll` | −65.6 → 43.5 | identity | — |
| `gripper` | −0.3 → 44.8 | identity | 1.9 → 42.8 |

Quick reads: a lift mean near +125 with a 45→186 range = v2.1 degrees.
A lift range near −100→+30 = v3.0 degrees. Everything inside ±100 with
mid-range zeros AND a 0…100 gripper = the ±100 representation. A lift
range that overlaps none of these = ask, don't guess.

## 5. The MolmoAct2 released checkpoint, concretely

Normalization on the molmoact2 families is DECODER-OWNED: one global
q01/q99 clamp table from checkpoint `metadata.stats`, applied to state
and action targets alike (`bijou/fast/molmoact2.py`). There is no
per-dataset collate normalization on this path (that is the
gemma4-family design), so the checkpoint's table is the single site
where conventions meet — and the single site to fix.

The released table is v2.1 degrees. Consumed directly against v3.0
data: `elbow_flex` and `shoulder_lift` frames clamp to the table edge
almost everywhere (measured: the ranges barely overlap), and lift's
learned direction is INVERTED. Two repair strategies exist:

1. **Recompute the table from the target mixture** (the corrected-table
   pattern): fixes clamping; SFT adapts to the re-centered space. But
   it cannot repair the sign flip — the model's learned lift direction
   stays inverted relative to the data, which SFT must unlearn and
   which breaks zero-shot use outright.
2. **Remap the released table through the official transforms**
   (conversion-time, the preferred fix): quantile normalization is
   affine, so writing `q01′ = A⁻¹(q01)`, `q99′ = A⁻¹(q99)` per joint
   reproduces the model's normalized stream EXACTLY on v3.0 data —
   flips included. The flipped joint lands as a DESCENDING pair
   (q01′ > q99′), which the normalization math carries transparently:
   the clamp happens in normalized space
   (`normalize_q01q99(...).clamp(−1, 1)`), never as a raw-space
   min/max, so a negative denominator simply composes. Zero adaptation
   at init; zero-shot coherent.

Machinery: the per-joint affine family and its gated fit live in
`bijou/eval/molmo_norm.py` (`AffineMap`, `fit_convention_map`) and
`sim/convmap.py` (the sim seam + `--convmap-override joint=sign,offset`
syntax). Fits are MEASURED against table pairs, never assumed —
because v3.0 units depend on per-robot calibration there is no
universal closed-form map. Pre-registered expectation for the released
table vs our mixture: scale 1.0 on every joint, `shoulder_lift`
(−1, +90), `elbow_flex` (+1, −90), identity elsewhere; any deviation
is a tripwire (wrong tag, wrong table, `use_degrees` mismatch), not a
value to accept.

## 6. The four mechanisms — and why they never stack

Four places handle convention differences; each is a COMPLETE fix for
its scope, so applying two to the same deployment double-remaps:

| Mechanism | Scope | Kind |
|---|---|---|
| conversion-time table remap (planned `--remap-stats-to`) | the checkpoint itself, permanently | exact affine, flips included |
| `--joint-frame v30-to-v21` (rollout; `JointFrameTransform` in `rollout_safety.py` owns the literals, test-pinned both directions) | one physical deployment, at the robot boundary | the official transforms, applied to state in / chunks out |
| `sim/convmap.py` seam (+ `--convmap-override`) | release-checkpoint-in-sim reads | fitted discrete map around an unmodified checkpoint (off-contract, `_convmap`-tagged) |
| `bijou.eval --molmo-norm` pdnorm/convmap | offline eval reads only | per-dataset affine wraps; `_pdnorm` additionally rescales spans (quantile equating), which is a DIFFERENT thing from a convention fix |

Rule: a checkpoint whose table was remapped at conversion deploys with
`--joint-frame rig` and needs no sim seam or eval wrap; an unremapped
v2.1-table checkpoint picks exactly ONE mechanism per context. The
rollout envelope gate catches a missing remap (state lands outside the
table's box); nothing automatic catches a DOUBLE remap on
shoulder_lift's self-inverse map (90 − (90 − x) = x — identity again!)
while elbow shifts 180° — a double remap is a half-broken arm, so the
exclusivity is a rule, not a runtime check.

## 7. Rules

- **Classify before you mix.** Every new dataset, table, or checkpoint
  gets placed on both axes (use §4's fingerprints) before it enters a
  mixture, a conversion, or a sim seam.
- **One convention per mixture.** Never rely on per-dataset stats to
  absorb a calibration difference — on molmoact2 there are no
  per-dataset stats, and on any family a sign flip survives
  normalization.
- **The sign flip is the expensive axis.** Offsets re-center; flips
  invert learned semantics. Any plan that "fixes stats" must say what
  it does about `shoulder_lift`'s direction explicitly.
- **Convention maps are provenance**: fitted maps and overrides ride
  the artifacts they produced (rows JSON, `stats_note`,
  `converted_from`) verbatim.
- Unrelated but adjacent: the same LeRobot page's PR #1452 section
  (policies' embedded-normalization migration) does NOT apply here —
  our normalization was never embedded lerobot-style; it lives in
  checkpoint metadata by design.
