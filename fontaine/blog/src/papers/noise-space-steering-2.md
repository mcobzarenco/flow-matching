# Noise-space steering II: executing through the channel, teaching through it

*Read 2026-08-07 (standing lit slice, the GPU-busy window before the
tsens dT read; closes the two radar hooks banked on the
[first ladder page](noise-space-steering.md)). Two sources, one
theme extension — the same frozen-decoder noise channel, pointed at
two problems the first page didn't cover: PAINT — Start Right,
Arrive Right: Asynchronous Execution via Initial Noise Selection
([2606.19774](https://arxiv.org/abs/2606.19774), June 2026) uses it
for chunk-boundary *consistency*, and UniSteer — Unified Noise
Steering for Efficient Human-Guided VLA Adaptation
([2605.10821](https://arxiv.org/abs/2605.10821), May 2026) uses it
for *human-in-the-loop adaptation*. Fed:
[#22 async staleness](../ideas/22-async-staleness.md) (a new,
cheaper first arm),
[#16 rig transfer](../ideas/16-rig-transfer-benchmark.md) (rig
lever #3), and
[#1 noise-draw ensembling](../ideas/01-noise-draw-ensembling.md) (a
probeable locality property, no gate change).*

## Where these sit on the ladder

The [first page](noise-space-steering.md) mapped the *adaptation*
ladder: who chooses ε, optimized against what reward or reference.
These two papers answer different questions with the same channel.
PAINT asks: forget improving the policy — can noise choice alone
make an **unmodified** policy behave coherently under deployment
latency? UniSteer asks: when a human is available to correct the
robot live, is noise space the right place to put those corrections?
Both keep every weight frozen, both ship "a vector, not a
checkpoint" — the ticket screen's economics, at two more points in
the design space.

## PAINT: the async problem is a noise-selection problem (2606.19774)

The setup is exactly our #22 regime: an action-chunk policy replans
while the previous chunk executes, so by the time the new chunk
arrives, d actions of the old one have already run. Naive switching
splices two independently-sampled chunks and jerks at the seam; RTC
(the [async zoo page](2026-08-07-async-chunk-execution.md)) fixes
this by *guiding the sampler* — freeze the committed prefix, inpaint
the rest, with a vector-Jacobian product per denoising step.

PAINT's reframe: don't steer the trajectory, **pick the starting
noise that makes steering unnecessary**. Three moves, all
gradient-free: (1) run one naive forward pass to get a draft chunk;
(2) build a target endpoint whose first d actions are the
already-executed prefix and whose tail is the draft's, then run the
flow ODE *backwards* (backward Euler, N steps) to recover the noise
that would have produced it; (3) the **repainting rule** — keep only
the *prefix* of that inverted noise, splice back the original random
noise for the suffix, and integrate forward once more. The prefix of
ε anchors the chunk to what the robot already did; the fresh suffix
keeps the policy's own diversity for everything not yet committed.
Cost ≈ 3N velocity-field calls (naive pass + inversion + final
pass), no gradients, no retraining, no policy modification.

Results: on GR00T-N1.5 (H=16, N=4) and π₀ (H=50, N=10 — our chunk
length), across 12 Kinetix force-control environments and six real
tasks on three embodiments (single-arm, ALOHA bimanual, humanoid).
Headline shape: PAINT **matches or beats RTC without its gradient
machinery** — real-task success 0.85 vs 0.75 (Toy in Drawer), 0.79
vs 0.76 (Towel Flinging), ties elsewhere, with uniformly lower
boundary-inconsistency error; on Kinetix it is the most robust
method as injected delay grows to d=4, precisely where the async
survey showed inference-time RTC collapsing. It also *composes*
with training-time methods: on top of TT-RTC it cuts consistency
error 0.11 → 0.08 at d=4. The ablation ladder over inversion
schemes says inversion quality is what buys robustness — backward
Euler at N steps is the sweet spot; one-step inversion leaves a
visible mismatch.

Honest caveats, theirs: the whole method leans on a **locality
assumption** — perturbing the prefix of ε mostly moves the prefix
of the decoded chunk, which optimal-transport flow matching
encourages but nothing enforces; real-world evaluation used a
single natural delay (d≈3); and an executed prefix knocked
off-manifold by a disturbance inverts to a poor ε (LP-DS's warning
surfacing again, from the execution side).

## UniSteer: human corrections become noise supervision (2605.10821)

The first page left a gap between FRS (demos → noise, no rewards)
and DSRL (rewards → noise, no demos). UniSteer occupies it: a
lightweight noise actor ψ(z|s) (a three-layer MLP) in front of a
frozen π₀ decoder, trained by **both** signals at once — live human
corrective actions *and* task reward — because both are converted
into the same noise coordinates.

The conversion is the mechanically interesting part. A human
correction is an action, not a noise; to supervise the actor you
must invert the decoder. Their **per-step fixed-point inversion**
exploits the Euler structure: each denoising update x_{k+1} = x_k +
Δt·v(x_k) inverts by iterating x = y − Δt·v(x, t_k) to its fixed
point (contractive whenever Δt·L < 1), recursing back through all
K=10 steps. At M=16 iterations per step it recovers a noise target
in ~0.1 s with reconstruction loss ~0.002 — against 50+ seconds and
worse accuracy for optimization-through-the-decoder. Their Table 2
makes it a three-way argument for the noise-space route: fixed-point
inversion 8/8 task success, optimization-based 4/8, direct
action-space supervision of a residual policy 7/8 at 2× the wall
clock.

Training is **SFT-then-RL on the same actor**: supervised regression
to inverted human corrections first (establishes an exploration
baseline), then Q-learning refinement. The ordering ablation is a
directional prior worth banking on its own: SFT→RL 95% average,
SFT-only 82.5%, RL→SFT 80%, RL-only 60% — corrections first, reward
polish second, and reversing the order *overwrites* what RL learned.

Results: four real tasks on an AgileX Piper, π₀ warm-started with 30
demos. Average success 20% → **90% in ~66 minutes** of adaptation,
vs DSRL 55% (pure noise-RL, no human) and DAgger 60% (human
corrections in *action* space). Two details matter more than the
headline: DAgger needed ~8 pure human trajectories per round where
UniSteer needed ~1 (corrections are absorbed as supervision, not
replacement data); and on out-of-distribution object positions
UniSteer holds 100% on three tasks where **DSRL drops to 0–25%** —
supervision-in-noise-space generalizes where reward-only noise-RL
overfits its training positions. Their stated limitations: inverted
noise targets sit slightly off the N(0, I) prior (LP-DS's
off-manifold drift, acknowledged but not trust-regioned), one robot,
four tasks, and a human who can actually teleoperate corrections.

## What transfers to us, and what doesn't

**#22 gets a cheaper first arm.** The async page's conclusion was
uncomfortable: naive switching degrades at our mean-of-10 staleness
(576 ms ≈ 18 ticks, chunk 50), inference-time RTC collapses there
too, and the named fallback was a TT-RTC fine-tune (~25% of base
training, weak at chunk 50) or an A2C2 residual head. PAINT
re-orders that list: training-free, gradient-free, ~3× the decode
cost of one chunk, demonstrated on a chunk-50 π₀. It slots ahead of
both training-time arms if the #22 screen ever shows a real
staleness cost — and it plausibly *composes with batched draws*:
the inverted prefix noise is shared across all 10 draws, each draw
keeps its own fresh suffix, and the mean inherits the anchored
prefix (our design note, not the paper's — it would need its own
oracle). #22 stays parked on #16 regardless; this changes the arm
order, not the gate.

**#16 gains rig lever #3, and an ordering prior.** The rig-time
menu is now: ship a searched ticket (zero extra machinery), distill
10 demos via FRS/DSBC, or — with a teleop pedal — UniSteer's
corrections-as-noise-supervision at ~1 h per task. The OOD result is
the strongest evidence yet for the *supervised* rungs of the ladder
over the pure-RL one at small budgets, and SFT-then-RL is the banked
schedule if a noise actor ever gets built. The criterion caveat from
the first page applies verbatim: every number above is rollout
success with a human or environment in the loop; none of it is
panel MAE.

**#1: a probeable property, not a gate change.** PAINT's locality
assumption — prefix of ε controls prefix of the chunk under OT flow
matching — is a claim about *our* teacher's geometry, and our
batched-draws machinery can test it for free: fix a draw, perturb
only ε[:d], measure where the decoded chunk moves. If our head is
locality-respecting, that's mechanistic support for reading ticket
structure at all (a [50, 6] ticket is implicitly a claim that noise
coordinates map to chunk coordinates); if it isn't, R4's per-dataset
matrix gets an extra grain of salt. Record-only diagnostic if ever
run; the ticket screen's R1/R2 verdicts remain the only gate on the
ladder.

**The inversion catalogue is now three deep.** FRS integrates the
whole ODE backwards from a reference endpoint; PAINT backward-Eulers
a constructed endpoint and keeps only the prefix; UniSteer
fixed-point-inverts one Euler step at a time. UniSteer's is the only
one with a head-to-head against optimization (8/8 vs 4/8, 500×
faster) — if an inversion primitive ever lands in `bijou.eval`, the
per-step fixed-point form is the numbers-backed default. All three
are flow-only: the molmo2 AR trunk stays exempt from this entire
family.

## Which idea it fed

**#22**: PAINT banked as the new first arm (training-free beats
both named training-time arms on cost, matches RTC on quality at
chunk 50); the screen and its #16 gate are unchanged. **#16**:
UniSteer banked as rig lever #3 (corrections + optional reward,
frozen trunk, ~66 min/task, needs teleop) with the SFT-then-RL
schedule prior. **#1**: locality probe noted as a free record-only
diagnostic; no new arm, no gate change — stage 1 of the ticket
screen tonight is unaffected.

*Radar hooks from this read: none banked — both hooks from the
first ladder page are now closed.*
