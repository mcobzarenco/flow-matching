# Pre-registration (DRAFT) — sim spawn-v2: randomize the disk and the boat

*2026-08-16, ~10:3xZ. Owner steering 09:16Z: "Both the disc and boat
should be placed randomly." Status: **DRAFT** — the CPU slice of queue
item `sim-spawn-v2-randomization`. Finalization (frozen constants from
the measured envelope + objection window) comes before any sim change
lands or any GPU stage launches; the owner's priority call (asked
in-channel 09:50Z: does this outrank the token-legs report?) sequences
the GPU slices.*

**Plain words.** Today's simulator always puts the wooden disk in the
same spot and drops the toy boat in a small patch right in front of
it, so every episode looks alike: boat on the same side, ~9–10 cm
from the goal. That was a deliberate prototype choice — the patch is
where the arm is strong and where the boat can't land on the parked
gripper — but it means a policy could pass our evals while only ever
having seen one corner of the problem. Spawn-v2 places *both* objects
randomly: the disk anywhere the arm can comfortably work, the boat
anywhere in a ring around the disk. "Comfortably work" is not a
hand-drawn box: during the scripted-expert work we measured exactly
where the arm's shoulder servo runs out of static torque and where
its inverse kinematics can put the jaws, and those measurements — not
guesses — define the allowed region. Everything trained or evaluated
so far keeps its numbers under the old protocol (frozen, labeled
spawn-v1); new demos and the new headline eval move to spawn-v2.

## §1 Spawn-v1, exactly (the thing being replaced)

`sim/so101_sim.py`: disk **fixed** at (0.22, 0.11) m, radius 0.04 m
(success = boat base inside the disk radius, upright, still, not
held). Boat: `SPAWN_X (0.195, 0.27) × SPAWN_Y (-0.005, 0.04)`,
uniform yaw — a 7.5 × 4.5 cm band in front of the disk, mean
boat→disk distance ~9.5 cm. The band's documented rationale: (a) the
comfortable-reach envelope around the menagerie pickup keyframe
(~0.22 m forward); (b) jaw clearance — spawns nearer than x ≈ 0.17
landed the boat ON the parked jaw tips (x ≈ 0.155) for ~4% of seeds;
(c) layout match to the original rig recordings. All three carry into
v2 as *constraints*, not as a fixed band.

## §2 Spawn-v2 design

Two placements per episode, both from the episode's spawn RNG stream:

1. **Disk**: uniform over the **measured workspace region** W — the
   set of table positions where the scripted expert's IK solves a
   grasp-height jaw-pad pose with residual < 1 mm AND the sysid'd
   shoulder servo's static gravity moment at that pose stays under a
   registered fraction of its force limit (the stage-A torque wall,
   `SERVO_SYSID` + the nullspace posture machinery, used here as an
   *instrument*). W is precomputed once by the reachability probe
   (§3) and frozen as an explicit polar-grid mask constant — the
   sampler draws from the mask, so the envelope is inspectable data,
   not a runtime solver call.
2. **Boat**: uniform over the **full annulus** around the drawn disk
   center — r ∈ [r_min, r_max], θ ∈ [0, 2π), uniform yaw — rejection
   sampled against: (a) boat pose itself inside W (the arm must be
   able to grasp *and* the disk to receive); (b) min separation:
   r_min ≥ disk radius 0.04 + hull half-length 0.03 + margin (boat
   must not start touching the disk); (c) parked-jaw clearance: hull
   keep-out around the settled home pose's jaw tips (the measured ~4%
   failure of v1); (d) table bounds with hull margin. r_max is a
   drawn-distance cap so episode length stays in the phase-clock
   budget the scripted expert and evals assume — pinned at
   finalization from the measured expert traverse envelope.

**Determinism + protocol versioning.** Disk draws add draws to the
spawn stream, so v2 seeds are NOT stream-compatible with v1 — by
design, as a *registered protocol break*: `spawn_version` becomes an
explicit sim parameter; `"v1"` reproduces today's draw order
bit-identically (oracle-guarded), `"v2"` is the new protocol. Every
banked v1 read stays frozen and labeled; no v1 number is ever
compared against a v2 number in a verdict.

**Rejection sampling is bounded**: the sampler refuses (loud error,
not a silent retry-forever) if acceptance over the first N draws
falls below a registered floor — a degenerate mask should fail the
oracle suite, not stall an eval.

## §3 Instrument: the reachability probe (CPU, this queue item's slice)

`sim/spawn_v2.py` (sampler + mask constants) +
`fontaine/scripts/spawn_v2_reachability_probe.py` (the instrument):
sweep a polar grid over the table's reachable quadrant, run the
stage-A IK (jaw-pad-midpoint space, wrist locked to the P4 pitch, 4
free dofs, nullspace posture pull) at grasp height, record IK
residual + static shoulder moment fraction; emit the W mask + a chart
(the measured envelope is itself a finding — the same torque wall the
learned policies face). Oracles (all CPU): v1 bit-compat under
`spawn_version="v1"`; v2 determinism (seed → identical placements);
constraint invariants on 10k draws (separation, keep-out, bounds,
in-W); acceptance-floor refusal on a degenerate mask.

### §3.1 Instrument v0 — first measured fields (added ~10:3xZ, same session)

The probe ran (CPU, ~2–4 min; `reports/analysis__spawn_v2_reachability_v0.json`):
1 cm Cartesian grid, 2196 cells, the stage-A grasp solve at
GRASP_Z = 0.014 m with the pan pre-swung to each cell's bearing and a
radially-facing hull. The instrument took two iterations to get honest,
and the interim readings are kept here because they shaped the design:

- **v0** used `ExpertPlanner.solve_grasp` as-is and read 425 cells
  under the 1 mm bar — but arranged in ring-bands. On that speckled
  mask the sampler's measured tail hit **194 of the 200-draw refusal
  bar** (mean 7.2 attempts over 2000 episodes): edge disks saw a
  mostly-rejected annulus. A morphological clean then collapsed the
  bands to 66 cells and the sampler *correctly refused outright* —
  the loud-refusal design did its job and flagged the instrument.
- **Root cause**: `solve_ik` stops at its 2 mm SITE tolerance, so
  whether a cell's *pad* residual lands under 1 mm was stopping luck,
  not reachability. The rings were a solver artifact, not arm physics.
- **v1** re-solves with an instrument-grade tolerance (0.2 mm, doubled
  iteration budget, local to the probe — stage-A behavior untouched).

![Spawn-v2 reachability probe: IK-residual field and static shoulder-moment field](../img/spawn_v2/reachability_v0.png)

The v1 facts the finalization constants will be cut from:

1. **1105 cells sit inside the 1 mm residual bar**, and the
   one-pass neighbor clean + largest-component step leaves a **solid
   977-cell region (~977 cm² vs the v1 band's 34 cm² — ~29×)**,
   containing the v1 band and disk comfortably. On the cleaned mask
   the sampler measures mean 2.4 boat attempts, p99 10, **max 35** of
   the 200-draw refusal bar over 5000 episodes — the tail is gone.
2. **Static torque does not bind**: over the whole reachable field the
   shoulder's static gravity moment peaks at 0.25 of the 3.478
   force limit — the nullspace posture pull keeps solves out of the
   straight-arm poses that saturated the servo in stage A. The torque
   bound in W is a backstop, not the working constraint; the residual
   bar does the work.
3. The sampler (`sim/spawn_v2.py`, 7 CPU oracles) is standalone —
   `SO101Sim.reset` is untouched until this pre-reg finalizes.

## §4 Registered consequences (the expensive part, priced)

| stage | what | cost class |
|---|---|---|
| A′ | scripted-expert validation re-run under spawn-v2, fresh held seeds, gate ≥70% (v1 ladder's bar) | ~0.2–0.4 GPU-h |
| B′ | demo re-collection under spawn-v2 (target per the stage-B recipe) | ~4 GPU-h |
| C′ | SFT on spawn-v2 demos (route per owner — joint class measured 5.7 GPU-h) | ~4–6 GPU-h |
| D′ | spawn-v2 eval, seeds 0–99 — becomes the **new primary** read | ~1.3–2 GPU-h |

The current joint checkpoint's 44/100 (and the whole route-C chain)
stays the frozen **band-protocol** read. The scripted expert's pan-arc
traverse was designed around a fixed disk bearing; A′ is a genuine
re-validation, not a formality — if it fails its gate, the expert
gets ONE registered robustness amendment (the v1 ladder's precedent)
before any F-verdict.

## §5 What finalization must pin — with proposed values (measured, not yet frozen)

From the v1 instrument, the freeze candidates now have numbers
(`sim/spawn_v2.py` DRAFT constants, sources commented at each):

| constant | proposed | source |
|---|---|---|
| W residual bar | 1 mm under the tight-tol solve (0.2 mm / 120 iters) | §3.1 v1 |
| W torque bound | 0.5 of forcerange (backstop; measured max 0.25) | §3.1 |
| mask clean | one ≥5-of-8-neighbors pass + largest 4-connected component | §3.1 (one pass only — iterating erodes any finite region) |
| r_min | 0.08 m (disk 0.04 + hull 0.03 + 0.01 margin) | v1 band arithmetic |
| r_max | 0.19 m (~2× the v1 mean start distance) | phase-clock budget |
| jaw keep-out | 0.04 m around (0.155, 0) | v1's measured ~4% failure |
| refusal bar | 200 draws (measured max on cleaned mask: 35) | §3.1 v1 |

Still genuinely open: A′ seed band + gate arithmetic; whether B′–D′
inherit the stage-B/C frozen recipes verbatim or re-open any knob
(default: verbatim). Finalization = a registered post freezing this
table + the objection window, after the owner's two calls: priority
vs the token-legs report (asked 09:50Z), and the C′ route choice.

## §6 FINALIZED (2026-08-16, same day)

The §5 table is **frozen at its proposed values, unchanged**. Chain of
record: owner approved the v1-dataset protocol built on this table
12:21:03Z ("agree with v1 with just the boat upright in the annulus");
the objection window was set in-channel 12:22:14Z as "flag it before
the box lands"; the A100 box landed 12:25:56Z with no objection —
window closed. What landed with the freeze:

- **W is committed data**: `sim/spawn_v2_mask.json` — the 977-cell
  cleaned mask from the §3.1 v1 instrument read, loaded by
  `WorkspaceMask.frozen()` with the cell count pinned (a drifted asset
  refuses at sim construction).
- **Integration**: `SO101Sim(spawn_version="v2")` — disk uniform over
  W (a static geom moved on the model each reset; `success()` and the
  scripted expert read the live `disk_center`), boat via the annulus
  sampler, all on the spawn stream in the sampler's pinned draw order.
  `spawn_version="v1"` (the default) is **bit-identical to the
  pre-change code** — verified qpos-digest-equal against the pre-change
  tree at reset, and the existing appearance/spawn-stream oracles all
  pass unchanged (check.py 950).
- **Measured, not assumed**: a disk drawn at the worst case — directly
  on the parked-jaw keep-out center (0.155, 0) — makes **zero contacts**
  with the homed arm (the parked gripper sits ~10 cm up; the disk is
  12 mm tall), so the boat keep-out needs no disk twin.
- **Sequencing note**: A′ (expert re-validation under spawn-v2) merges
  into the v1 dataset generation itself — the sharded collection
  measures the expert's spawn-v2 success rate on thousands of seeds as
  it generates; the first-shards read is the go/no-go telemetry.

## §7 A′ FAILED on v2 as frozen → registered amendment v2.1 (same day)

A′ ran immediately (the A100 box landed) and **failed hard: 19.8%
expert success on 600 unrendered spawn-v2 seeds** (first rendered
smoke agreed: 3/22). This is the §4 registered risk realized — and the
instrument, not the expert, is the culprit:

- **Failure geometry**: success is a cliff in *boat* distance from
  base — 48.3% below r_base 0.26 m (the v1 band's radius), 6.3% at
  0.26–0.34, 0.8% beyond. Bearing doesn't matter (0–4% across ±70° at
  far radius). Failed episodes loop jam-flip → recover → approach:
  the measured pads never come within the 3.5 cm jam threshold of the
  solve target, which the expert misreads as a mechanical jam.
  (`reports/analysis__spawn_v2_expert_probe.json`, phase traces
  included; droop-clip A/B at ±8 cm: no effect, 19.2%.)
- **The instrument was wrong about torque**: §3.1's static-moment
  field read ≤0.25 of forcerange across W. Direct measurement
  (`spawn_v2_hold_probe.py`: solve the expert's own grasp IK, teleport
  onto the solution, hold under physics) shows the sysid'd
  shoulder-lift servo **saturated (force fraction 1.00)** holding
  extended poses, with steady-state pad sag growing from ~3 mm at
  r_base 0.20 to ~20 mm at 0.36. The 1 mm IK-residual bar measured
  *kinematic* reachability; the actuator cannot statically serve the
  outer half of W. This is real arm physics (the same servo the rig
  runs), not a sim artifact.
- **Amendment v2.1** (`spawn_version="v2.1"`, the ONE registered
  robustness amendment): keep the annulus geometry, full ±180° yaw,
  uniform-in-W draws — constrain both placements to the **measured
  competence bands**: boat r_base ∈ [0.16, 0.27], disk r_base ∈
  [0.18, 0.32]. Cut from the 600-seed field (68.2% on the post-hoc
  joint band, n=110); the v2.1 sampler measures **56.0% end-to-end on
  400 fresh seeds** (edges of the bands are weaker than the interior —
  the numbers above are the honest sampler-weighted rate). v2 as
  frozen stays oracle-pinned and unused; v1 bit-compat untouched.
- **Coverage vs v1 remains a step change**: disk anywhere in a
  56 cm² annular band across all bearings (v1: one fixed point), boat
  in the full annulus around it at 8–19 cm separation with free yaw
  (v1: a 34 cm² patch in front of one disk).
- The far-radius region is not lost, just deferred: reaching it needs
  either a stronger shoulder (hardware) or a non-prehensile/regrasp
  strategy (the side-spawn righting probe's territory — same skill
  family, queued).
