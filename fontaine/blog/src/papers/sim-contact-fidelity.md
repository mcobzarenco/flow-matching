# Making the grasp honest: contact fidelity for a MuJoCo SO-101 twin

**Sources:** MuJoCo
[modeling](https://mujoco.readthedocs.io/en/stable/modeling.html) /
[computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
docs + [changelog](https://mujoco.readthedocs.io/en/stable/changelog.html)
(fetch-verified 2026-08-11) · CoACD
([2205.02961](https://arxiv.org/abs/2205.02961)) · IPC-GraspSim
([2111.01391](https://arxiv.org/abs/2111.01391)) · BAM extended
friction models ([2410.08650](https://arxiv.org/abs/2410.08650),
[code](https://github.com/Rhoban/bam)) · SIMPLER's physics ablations
([2405.05941](https://arxiv.org/abs/2405.05941)) · real-to-sim
capture: PolaRiS ([2512.16881](https://arxiv.org/abs/2512.16881)),
GSWorld ([2510.20813](https://arxiv.org/abs/2510.20813)), soft-body
GS twins ([2511.04665](https://arxiv.org/abs/2511.04665)),
Real-is-Sim ([2504.03597](https://arxiv.org/abs/2504.03597)).
**Read:** 2026-08-11, sim lit lane. **Fed:** the fix list ahead of
the `sim-policy-eval-100seeds` pre-reg — every one of the
[sim review](../posts/2026-08-11-sim-review-findings.md)'s four
findings turns out to have a documented mechanism and a named fix.

**The problem in plain words.** Our simulator review found that the
virtual boat is wrapped in an invisible 4-millimeter force field
(a by-product of how curved shapes get approximated for collision
checking), that the gripper's settings silently cancel the boat's
carefully tuned friction at the exact moment of grasping (so the
boat spins in the jaws), and that the arm's motors fight themselves
in the home position. Before fixing any of that, we read what the
simulator's own documentation and the surrounding literature say.
The satisfying answer: all three problems are *known*, documented
failure modes with standard fixes — we were not fighting mysterious
physics, we were holding the tools wrong. The more surprising
answer: the biggest lever for making a sim *rank policies
correctly* is not contact physics at all — it's whether the
simulated motors respond like the real ones, which one paper
measured directly and we can fix with a few recorded real
trajectories. Exact friction values, the thing we were most worried
about, turn out to be nearly irrelevant for ranking.

## Finding 3 (phantom margin): we capped the wrong knob

CoACD's intended fidelity control is the **concavity threshold**
(`-t`, default 0.05, sensible grasping range 0.01–0.02), not the
hull cap. Our conversion asked for `-t 0.05` but capped hulls at 16,
forcing CoACD to stop *before reaching its concavity target* — the
measured 0.149 achieved concavity and the p99 3.8 mm phantom margin
are exactly this failure mode. The fix is to drive by threshold
(and preprocessing resolution `-pr`), let the hull count float, and
accept the runtime cost — MuJoCo's docs themselves recommend CoACD
by name for this workflow, and `obj2mjcf` (the tool behind
menagerie's own assets) uses CoACD as its backend. V-HACD is the
wrong tool here: its known weakness is *filling concavities*, the
exact property a graspable hull must keep.

Two escape hatches if threshold-driven CoACD stays too coarse:

- **Native SDF collisions** (MuJoCo ≥3.3.5 needs no plugin): a
  voxelized signed-distance field generated *at model compile time*
  from the unmodified mesh — no convexification, so the phantom
  margin vanishes by construction. Costs: the docs' most expensive
  collider class, and contact-finding by gradient descent can miss
  contacts if `sdf_initpoints` is low. Bonus that matters to us: the
  benchy is CC BY-ND, so our CoACD hulls are un-committable derived
  assets rebuilt per machine (the repro hazard the sim review
  flagged) — an SDF is generated at load from the original STL, so
  the repo ships nothing derived. That would close the
  per-machine-asset hole and the phantom margin in one move, if the
  per-step cost is acceptable at our ~27 ms/tick budget.
- **`maxhullvert` quality jumped in 3.10.0** (June 2026, Qhull Q9) —
  relevant if we coarsen hulls for speed later.

## Finding 4 (spin in the jaws): documented behavior, three fixes

The priority rule is spec, not bug: *"if one of the two geoms has
higher priority, its friction coefficients are used"* — wholesale,
not element-wise. Our gripper's `priority=1` therefore replaces the
benchy's condim-6 friction triple (torsional 0.05) with its own
(torsional 5e-3) at every gripper↔boat contact. Fixes, in
increasing order of control: set the full friction triple on the
priority geom; drop priority and rely on element-wise max; or —
cleanest — declare an explicit `<contact><pair>` for the jaw–boat
seam, which overrides both geoms and is also the only place the
separate `solreffriction` works.

The docs' *"Preventing slip"* section reads like a checklist written
for our probe results: condim ≥ 4 for torsional friction ("prevents
rotation around the normal" — our 6.9° in-grip spin), **elliptic
cones + impratio ≈ 10 + Newton solver** for slip suppression, keep
`multiccd` on for flat-on-flat jaw contact, armature/implicitfast
against vibration. Worth knowing exists: MuJoCo 3.11.0 (July 2026)
added a per-geom/pair **adhesion** attribute — "a physical
stabilizer for grasping" — a legitimate, documented stabilizer if
honest friction still under-holds, at the cost of physics realism.

The measured anchor for "compliance dominates": IPC-GraspSim
(2111.01391) benchmarked grasp-outcome prediction against 2,000
physical grasps — compliant, intersection-free jaw contact hit
F1 = 0.85 and beat both analytic models and Isaac Gym. In MuJoCo
terms: condim 4 + softened fingertip solref/solimp is the analog;
rigid perfect jaws are the least faithful option, not the most.

## Findings 1–2 (home pose, servo saturation): sysid is the first-order term

Two independent sources pin our servo model as the thing to fix
*first*:

- **BAM** (2410.08650) identified extended friction models for
  hobby-class servos from a pendulum bench — CMA-ES over 2–11
  params, converging in ~5 min — improving trajectory MAE 1.5–2.9×
  over the standard Coulomb-viscous model, and the repo **ships an
  identified Feetech STS3215 model** (our exact servo) with MuJoCo
  integration. The cheap version of this — replay a few real SO-101
  episodes through the sim and fit gains — is SIMPLER's sysid
  recipe, and it needs no new hardware bench.
- **SIMPLER's ablation** (Table II): degrading controller sysid
  from control-loss 0.131 to 0.432 moved eval-ranking MMRV from
  0.031 to 0.100 — a 3× fidelity loss from controller gains alone.
  Meanwhile their Table X varied *contact* parameters (object
  friction 16×, density, gripper friction) and rankings barely
  moved (MMRV ≈ 0.055 across nearly all settings). **Controller
  gains are first-order for eval fidelity; friction coefficients
  are second-order.** One caveat cuts back our way: their tasks
  never stressed in-hand torsion — a *qualitatively* wrong contact
  regime (zero torsional friction at the seam, phantom margins) is
  a different failure class than an imprecise coefficient, so
  Findings 3–4 still need their fixes; we just shouldn't *tune*
  friction values beyond qualitative correctness.

This lands on a live discrepancy the
[landscape page](so101-sim-landscape.md) surfaced: our menagerie
model runs `kp 998.22 / forcerange ±2.94` where TheRobotStudio's
upstream publishes `kp 17.8 / forcerange ±3.35` for the same servo —
and ±2.94 is precisely the saturation ceiling the review measured in
the jammed home pose. Fitting gains against real episodes (we have
229 h of rig data) resolves the 56× disagreement empirically instead
of by trusting either XML.

## What we deliberately don't need

- **Gaussian-splat capture pipelines** (PolaRiS: 2–5 min phone scan
  → eval-grade twin; GSWorld; Real-is-Sim's 60 Hz-synchronized
  twin): the field's answer to *visual* fidelity plus quick asset
  capture. Our visual gap is real but unmeasured until real
  rollouts exist, and our scene is three rigid objects on a table —
  scanning tooling solves a problem we don't have yet. Notably,
  none of this line reports contact-*geometry* error metrics (our
  p99 phantom-margin number has no published analogue) — contact
  quality is asserted via end-task correlation only.
- **Eval-time physics randomization**: no published result shows
  ensembling over physics parameters improves rank correlation with
  real; SIMPLER's Table X suggests rankings are already stable
  across wide physics ranges (so the ensemble adds variance, not
  signal), and the 2026 cross-simulator recipe paper attributes
  proxy reliability to simulator-level fidelity, not parameter
  sweeps. Single fixed physics + fixed seeds stays the right
  design.
- **Grasp-quality instrumentation exists if wanted**: MuJoCo 3.3.5's
  `contact`/`insidesite` sensors give clean per-contact readouts —
  nicer than our probe scripts' manual mining if the fix-verification
  probes grow.

## The fix list this page feeds

In pre-reg order for `sim-policy-eval-100seeds`:

1. **Servo/controller sysid** against real episodes (BAM STS3215
   params as the informed prior; resolves the kp discrepancy and
   should un-jam the home pose with Finding 1's mount fix).
2. **Home pose + spawn-after-settle** (Findings 1–2, already
   specified in the review) — re-verify 0 reset strikes over the
   registered seed list.
3. **Jaw–boat `<contact><pair>`** with condim 4+, elliptic cones,
   impratio ~10, Newton (Finding 4); re-run the pinch probe,
   compare spin/tilt.
4. **Threshold-driven CoACD** (`-t` 0.01–0.02, uncapped hulls) or
   the SDF experiment (also fixes the CC-BY-ND per-machine asset
   hazard); re-run the phantom-volume probe.
5. Friction *values*: leave at qualitative correctness — SIMPLER
   Table X is the citation for not tuning them.
