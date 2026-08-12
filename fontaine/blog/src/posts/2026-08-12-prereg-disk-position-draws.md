# DRAFT pre-reg: disk-position draws — the target moves, the success zone moves with it

*Drafted 2026-08-12 ~11:0xZ (work session; real `date -u` at write:
10:52). **STATUS: DRAFT — not registered.** This is the (c) leg the
content-diversity pre-reg
[scoped out as task semantics](2026-08-12-prereg-sim-content-diversity.md)
— the disk is the task target, so moving it is an eval-protocol
change, not an appearance draw. It holds for owner sign-off (the
queue item pins it to the rerun call); nothing is implemented or
launched from this document. Evidence base:
[content-diversity close](2026-08-12-sim-content-diversity-results.md)
(the record-only disk fact) ·
[sim100 v0 close](2026-08-12-sim100-results.md) ·
[spot20 v3 results](2026-08-12-sim-spot20-v3-results.md).*

## Plain words

Our simulator glues the wooden target disk to one spot on the table,
episode after episode. The real rig doesn't: measuring the 26 real
reference episodes, the disk sits somewhere different almost every
time — the operator nudges it around a region about 20 by 29
centimeters. Worse, the one spot the sim glues it to turns out to be
a place the real disk *never actually sat* — it's just outside the
measured range. This plan makes the sim draw the disk's position
fresh every episode from the real measured spread, with the boat
spawned in front of the disk wherever it lands and the
success zone following it. Beyond realism, that upgrade turns the
eval into a question the pinned disk can never ask: does the policy
*look* for the disk, or does it just drive to the memorized spot?
A policy that only memorized will dump the boat where the disk used
to be; a policy that sees will track it. The plan sits in draft
until the owner signs it off — and it deliberately stays out of the
v3 rerun, which must measure one change (visuals) at a time.

## The measured baseline (banked, bank_manifest.json)

- Disk detected in **21/26** A-episodes (`disk_record_only`,
  per-object centroid-bias-calibrated camera model, same pipeline
  the v3 clutter draws use). The 5 absences are detection misses /
  occlusions — a diskless place-task is undefined, so **presence is
  pinned at 1** in this protocol.
- Absolute world frame (follower base = origin): x **0.083–0.288**,
  y **−0.193–0.097**, median **(0.199, −0.048)**; radius from the
  follower base 0.107–0.288 m, median 0.217.
- The sim pins the disk at **(0.22, 0.11)** — radius 0.246, and
  **y = 0.11 is outside the entire measured y range** (max 0.097).
  The pinned eval measures a target placement the real rig never
  exhibited.
- Frame alignment is trusted on the mouse precedent: its measured
  absolute box (x 0.485–0.555, y −0.233..−0.075) brackets its sim
  canonical (0.50, −0.085), so the calibrated world frame IS the sim
  frame. The laptop needed delta-mode only because its real center
  sits past the frame edge; the disk has 21 clean in-frame
  detections and no such excuse.

## Design — six decisions, stated so they can be objected to

**D1 — draw mode: ABSOLUTE, not delta-about-canonical.** Uniform in
the measured box (x 0.083–0.288, y −0.193–0.097) intersected with
the validity region (D4), rejection-redrawn until valid. Absolute is
the fidelity-first choice given the frame alignment above;
delta-about-canonical would re-center the real spread on a pinned
position the real disk never occupied, dragging ~38% of the
empirical draws beyond even the farthest radius the real rig ever
demonstrated (0.288 m). One draw per episode, presence 1.

**D2 — success geometry follows the drawn disk.** Mechanism: the
reset writes `model.geom_pos["disk"]` and updates
`self.disk_center` (today read once at init, `so101_sim.py:238`).
`success()`, `benchy_disk_distance()`, and the initial/min/final cm
metrics all key off `disk_center`, so `progress_final` stays
"distance recovered toward the (drawn) disk" with no metric change.
Note the disk — unlike the contype-0 clutter stand-ins — has real
collision (the boat physically rests on it at success). Moving it is
a physics change, and that is the point.

**D3 — benchy spawn goes DISK-RELATIVE.** The current absolute
spawn box was hand-placed for the pinned disk (mean boat→disk gap
~9.5 cm, boat "in front of" the disk at −y). Keep exactly that
relationship: spawn = drawn disk + delta, delta-x ∈ [−0.025, 0.05],
delta-y ∈ [−0.115, −0.07], yaw uniform — the current box expressed
relative to the disk. This preserves the `initial_cm` scale
(episodes stay ~9.5 cm tasks) and the task difficulty distribution;
an absolute spawn box under a wandering disk would smear initial
distance over ~2–25 cm and make per-seed numbers incomparable in a
useless way (difficulty, not ability, would drive the spread).

**D4 — joint validity clamp, by rejection redraw.** A drawn (disk,
spawn) pair is valid iff: disk radius from base ∈ [0.12, 0.28]
(brackets 20/21 real draws; the far real outlier at 0.288 is
marginal); disk edge ≥ 7 cm from the parked jaw tip (0.155, 0.01)
(the settle would strike it); every spawn-box corner at x ≥ 0.17,
y ≥ −0.19, radius ≤ 0.30 (jaw clearance, leader-arm clearance — the
leader mounts at y=−0.25 and the raw delta box would push spawns to
y=−0.31 under the most down-table real draws — and reach). Invalid →
redraw both. **Constants are proposals**, finalized by the
policy-free 1000-seed reset sweep at implementation (gates: reset
strikes = 0 on every seed, settle displacement nominal); the sweep
also **reports the truncation fraction** — how much of the measured
real support the clamp cuts — so nothing is silently capped.

**D5 — comparability: banked rows DIE under this protocol.** The
choice, stated plainly: with disk-relative spawn, the same seed no
longer produces the same boat world pose as banked sim100/spot20
rows — and the frames differ regardless (the disk is elsewhere).
**No number produced under disk draws may be compared to a banked
pinned-disk row, paired or otherwise.** This is protocol v2 of the
sim eval ("sim100-D"); any registered run under it re-runs its own
`hold` floor and baselines from scratch. What pairing survives is
the within-run kind — and fully: same seed → same disk + same spawn
across arms AND render styles, so arm-vs-arm per-seed deltas keep
their full power. The alternative (absolute spawn box, banked
spawn-stream bit-match preserved) was rejected: the banked
comparison it preserves is vacuous anyway (different disk → different
initial_cm → `progress_final` not comparable), and it costs the
difficulty-controlled task (D3).

**D6 — stream discipline.** Both draws live on the SPAWN stream
(seeded by `seed`, untouched by `appearance_seed`): disk draw first,
then the relative spawn draws, then yaw. The appearance stream is
not consumed → the protocol is orthogonal to render style (v0/v2/v3
all see the same drawn geometry for a given seed). A `disk_draws`
flag (default off) guards the whole path: off = today's behavior,
bit-identical, oracle-pinned.

## The grounding probe (the payoff read)

The pinned-disk eval cannot distinguish "drives to the memorized
spot" from "sees the disk" — the two policies act identically. Under
draws they diverge, and the divergence is registered as a
diagnostic: per seed, compare the boat's final distance to the
**drawn** disk against its distance to the **canonical** location.
Summary read: across seeds, the regression slope of final-boat
displacement on disk-draw offset — slope ≈ 1 is a tracker, slope ≈ 0
with finals clustered at canonical is a memorizer. Priors to be
registered per-arm at the run-specific pre-reg (not here), but the
shape expected from spot20: teacher80k (the one confirmed visual
responder) is the candidate tracker; er60k's reach-over-and-miss
fingerprint predicts slope ≈ 0.

## Implementation plan + oracles (the follow-up item, CPU-only)

Implementation is NOT this item; when signed off it lands with:

1. Guard oracle: `disk_draws=False` → settled qpos, spawn xy, and
   frames bit-identical to today across seeds (extends
   `tests/test_sim_appearance.py`).
2. Pairing oracle: same seed → identical (disk xy, spawn xy, settled
   qpos) across appearance seeds AND render styles.
3. Geometry oracle: teleport boat onto the drawn disk → `success()`
   true; onto the canonical location (disk drawn elsewhere) →
   false; `benchy_disk_distance` tracks the drawn center.
4. Determinism oracle: rejection redraws are a pure function of
   seed — same seed, same draw count, same result, any process.
5. Render oracle: the drawn disk appears at its drawn position in
   BOTH cameras under v2/v3 composites (it is a V2_DYNAMIC_STATIC —
   rendered, never plate content; the plates median it out).
6. The 1000-seed policy-free reset sweep (D4 gates + truncation
   report).

## Cost and sequencing

- Implementation + oracles + sweep: ~1 CPU session, no GPU.
- Any registered eval under sim100-D prices exactly like the v3
  rerun: parallel-oracle GREEN → ~2–3 GPU-h for 4 arms × 100 seeds;
  sequential fallback ~6–9 h wall. Owner-gated, separate pre-reg.
- **Sequencing (registered intent): AFTER the v3 rerun reads out.**
  The rerun isolates the visual overhaul at n=100; stacking a task
  change into it would confound both reads. Disk draws are the
  *next* protocol step, with the rerun's v3 numbers as its pinned
  baseline context.

## Owner decision points

1. Sign off the protocol change at all (it retires banked-row
   comparability for future runs — D5).
2. D3's disk-relative spawn (the alternative preserves banked
   spawn bit-match but was rejected above — overrule if the
   banked-row link matters more than difficulty control).
3. Whether the pinned-disk protocol stays available as a registered
   variant (the `disk_draws` flag keeps it one boolean away) or is
   deprecated for anything but reproduction runs.

## Finalization checklist (at sign-off)

1. Owner call on the three points above.
2. Implementation session lands the six oracles + sweep; truncation
   fraction and final clamp constants appended HERE with a dated
   edit note.
3. Run-specific pre-reg (arms, per-arm grounding-probe priors,
   GPU gate) posted with param sheet in-channel + objection window.
