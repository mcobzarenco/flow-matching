# Async execution II: shrink the delay, smooth the seam, or train for the lag

*Lit slice 2026-08-09 (work session 11:56Z). Cluster page: FASTER
([2603.19199](https://arxiv.org/abs/2603.19199), v3 May 2026) +
ABPolicy ([2602.23901](https://arxiv.org/abs/2602.23901)) + DEFLECT
([2605.19294](https://arxiv.org/abs/2605.19294), v2 June 2026). The
follow-up to [the RTC + method-zoo page](2026-08-07-async-chunk-execution.md)
and [PAINT](noise-space-steering-2.md). Fed #22 (arm menu re-ranked),
#16 (latency accounting), #12 (a fourth pole on the one-step axis).*

## The papers in plain words

A robot policy that plans 50 actions at a time has a built-in lag
problem: by the time the newly computed plan starts executing, the
world has moved on from the camera frame it was computed from. The
three papers here attack that lag in three different places, and
reading them together is the point:

- **FASTER** attacks the *cause*: it makes the very first action of
  the plan come out of the sampler almost immediately (one denoising
  step instead of ten), streams actions to the robot as each one
  finishes rather than waiting for the whole plan, and keeps refining
  the later actions meanwhile. The robot starts reacting ~1.3–3×
  sooner on the same hardware — enough to play table tennis.
- **ABPolicy** attacks the *symptom you can feel*: jerky motion. It
  has the policy output a handful of spline control points instead of
  40 raw actions (smooth by construction), and stitches consecutive
  plans together by re-fitting the new spline so it starts exactly
  where the executed motion left off.
- **DEFLECT** attacks the *residual error*: it accepts the lag will
  exist and post-trains the policy to compensate. It builds "good
  action vs. bad action" pairs automatically — the same frozen policy
  queried once with the observation the robot will *actually* have
  acted in and once with the stale one — and preference-tunes the
  policy toward the former, gaining most where runtime tricks like
  RTC have already collapsed.

None of these were run at our scale of staleness (our expensive
decode is ~18 control ticks old; they test up to ~7), but the
division of labor — reduce the delay, smooth the seam, train for
what's left — is the map this whole literature was missing.

## FASTER: reaction time is a schedule problem

**Contribution.** The paper first re-derives what "responsiveness"
even is for a chunked policy: an external event lands at a uniformly
random point in the execute-then-replan cycle, so reaction time is
distributed U(Δt_infer, Δt_infer + Δt_exec) with mean
Δt_infer + ½·Δt_exec. Two consequences fall out. Going synchronous →
asynchronous only shaves ½·Δt_infer off the mean — much less than
async advocacy implies. And the metric that actually matters is
**time-to-first-action (TTFA)** — the analogue of time-to-first-token
for LLM serving.

The mechanism is the **Horizon-Aware Schedule (HAS)**: instead of
denoising all 50 actions of the chunk in lockstep over N flow steps,
each action index i gets its own "hit time" u_i = (1 − (i/(H−1))^α)·u₀
at which it finishes. Setting u₀ = (N−1)/N makes action 0 fully
denoised after **one** step of N while the tail of the chunk keeps
refining over the remaining steps. Two supporting pieces: a
**streaming** client-server path that dispatches each action the
moment its local schedule hits τ=0 (network latency for action k is
masked by the execution of actions <k), and **early stopping** once
everything inside the execution horizon is finalized. Training is a
mixed-schedule fine-tune — HAS with probability p, the original
constant schedule otherwise — no architecture change; their pilot
study motivates it by showing near-term actions have straighter
(easier) denoising paths than distant ones.

**Experiments.** π0.5 (chunk 50) and X-VLA on RTX 4090 and
consumer-grade RTX 4060 at 30 Hz control. TTFA: π0.5/4090 80.0 →
62.1 ms (1.29×); X-VLA/4090 113.7 → 44.8 ms (**2.54×**, and the
feasible execution horizon s_min drops 4 → 2); X-VLA/4060 3.09×. On
X-VLA the improvement is distribution-dominant: FASTER's *worst-case*
reaction time beats naive async's *best case*. Real tasks: table
tennis (visibly earlier racket positioning, better completion scores
than training-time RTC on both GPUs), pick-beverage, dual-arm towel
folding.

**Caveats they own.** Accelerated sampling of the early actions can
cost some prediction quality (Appendix G); the analysis assumes the
client-server split; highly stochastic environments untested.

## ABPolicy: make smoothness a property of the action space

**Contribution.** Flow matching over **cubic B-spline control
points** — 8 continuous control points reconstruct a 40-step chunk at
C² continuity, with least-squares fitting of ground-truth chunks to
control points as the label transform (reconstruction error 3.1e-4 /
50.7 dB SNR, vs 1.0e-3 for a DCT basis and 2.0e-3 for BEAST-style
discretized bins). Within-chunk jitter is gone *by construction*.
Chunk boundaries are handled by **bidirectional action prediction**
(the chunk spans 8 past + 32 future steps, so the model predicts
through the seam, not from it) plus **continuity-constrained
refitting**: when a new chunk arrives mid-execution, only its leading
control points are re-optimized to anchor onto the actions already
executed; the rest stay as predicted. Async inference then just
works — a ~90 ms replan runs in a parallel thread while the previous
spline executes, and refitting splices the update in.

**Experiments.** Small scale but real: a 6-DoF AgileX Piper, DiT
head on frozen DINOv2, 100–200 demos/task, seven tasks of which
three are genuinely dynamic (objects on a rotating platform). Async
beats sync where the world moves — stack-block 55 vs 30%, hang-cup
60 vs 40% — and matches it on static tasks while finishing 14%
faster (no stop-and-go). Smoothness deltas are large: 95th-percentile
acceleration −57%, velocity zero-crossings −29%. Ablations:
bidirectional prediction alone lifts static stacking 60 → 85% and
cuts boundary jitter 23%; refitting cuts it 46%.

**Caveats.** Single embodiment, small policy, no VLA-scale trunk, no
comparison against the RTC/BID family — this is a representation
paper, not a delay-robustness paper.

## DEFLECT: the delay becomes a preference label

**Contribution.** The staleness problem generates its own supervision.
Roll a trajectory; at time t the deployed policy would see
observation o_t but act at t+d. Query a **frozen reference copy** of
the policy twice with the *same* sampling noise: once conditioned on
the execution-time observation (→ preferred chunk A⁺), once on the
stale one (→ rejected A⁻). Then fine-tune with flow-matching DPO —
−log σ(−β[(L_θ⁺−L_ref⁺) − (L_θ⁻−L_ref⁻)]) with the flow loss as the
likelihood surrogate — **scoring both chunks under the stale input
the policy will actually receive at deployment**, plus an SFT anchor
term on expert data. That last detail is the paper: the policy learns
to emit execution-aligned actions *from* stale observations. No human
labels, no reward model, no architecture change; on π0.5 only the
~693M action expert trains.

**Experiments.** Bases are VLASH (the predicted-future-state method
from [the method zoo](2026-08-07-async-chunk-execution.md)) and π0.5,
on Kinetix (12 tasks), LIBERO (4 suites), and 3 real bimanual tasks.
Delays d ∈ {0..4} ticks in training, evaluated out to d=7 (~233 ms at
30 Hz). Headline: at d=5–7, where **RTC and BID have collapsed to
≤5%** (insufficient chunk overlap left to reconcile), DEFLECT gives
73.5% vs VLASH's 67.1% (+6.4 pp; +8.0 over preference-tuning with
clean observations). LIBERO gains grow monotonically with delay
(+0.2 pp at d=1 → +4.6 pp at d=7). Real robot: conveyor tasks +10
and +6.7 pp, whack-a-mole 13.6 vs 10.4 moles/30 s. The ablations are
unusually honest: dropping the SFT anchor collapses Kinetix to 12.3%
(mode collapse off the expert manifold); training on d ∈ {1,2} only
still transfers +3.7 pp to d=7; and Appendix L measures that the
cosine-restart schedule *alone* contributes +2.6 pp of the headline —
the delay-specific mechanism nets **+1.6 to +2.3 pp** at high delay.

**Caveats.** The preference signal vanishes at low delay (the two
observations are nearly identical); flow heads only (AR likelihoods
untested); gains ride on top of VLASH's state-forward conditioning,
which needed oracle state in the survey's LIBERO setting.

## What transfers to us, what doesn't

- **The map is now three-axis.** The method zoo + PAINT gave us
  *bridging* options for a fixed delay. FASTER says the delay itself
  is partly a scheduling artifact: with HAS + streaming, the
  first-action latency of a 10-step flow decode approaches the
  1-step decode's — which directly attacks the premise of #22's
  screen ("mean-of-10 costs 18 ticks"). If HAS composes with batched
  draws — and nothing structural says it doesn't: the schedule is
  per-action-index, so it tiles across the draw dimension exactly
  like the [draws-major batching](../ideas/01-noise-draw-ensembling.md);
  the per-index mean is computable as each index finalizes across
  all 10 draws — then the 18-tick figure could drop toward ~2–4
  ticks *before any bridging method is bought*. That re-orders the
  #22 arm menu again: **measure naive-switch cost → try
  HAS-on-decode (fine-tune, no architecture change) → PAINT → A2C2
  residual → TT-RTC/DEFLECT-style post-training.**
- **DEFLECT validates the regime table from the other side.** The
  survey predicted runtime reconcilers die at high delay-in-ticks on
  long chunks; DEFLECT *measures* RTC/BID at ≤5% for d≥5 and shows a
  training-time fix that still works there. But d=7 is their
  ceiling and our mean-of-10 sits at ~18 — outside everyone's tested
  range. The honest read stands: nothing published covers our
  worst-case regime; FASTER-style delay reduction is worth more than
  better bridging.
- **Effect-size sobriety.** DEFLECT's restart-corrected net (+1.6 to
  +2.3 pp) is the right number to carry, not the +6.4 headline — the
  same bundling-hides-sign lesson as our perf pass-1. Banked as a
  caution on any future preference-tuning arm.
- **ABPolicy's spline space is a #5-adjacent observation, not an
  arm.** Continuous control points beat discretized bins (BEAST) and
  DCT at reconstruction — but our action head regresses raw chunks
  and our panel MAE is computed in raw action space; switching
  representation would touch everything for a smoothness gain our
  offline panel cannot see (same closed-loop-only visibility as all
  of #22). Its refitting trick is the cheap part worth remembering
  if rig rollouts ever show boundary jerk. Its jerk metrics (95th-pct
  acceleration, zero-crossing rate) are a ready-made instrument for
  the #16 rollout eval — same family as the SDN jerk read on
  [noise-space III](noise-space-steering-3.md), where our flow
  draws were already uniformly smooth (which predicts ABPolicy's
  within-chunk win would be small for us; the *boundary* term is the
  open one).
- **TTFA accounting transfers immediately to #16.** E[reaction] =
  Δt_infer + ½·Δt_exec means the execution horizon we choose on the
  rig will likely dominate our decode latency; the
  [HyperVLA 4 ms pole](hypervla-hypernetwork-inference.md) and this
  page's streaming numbers bracket the design space. Free design
  input for the rig bench, no measurement needed now.
- **Doesn't transfer:** every success-rate ranking here (Kinetix MLPs,
  chunk 16–40, single-arm Piper ≠ our chunk-50 VLA); VLASH-stacked
  gains (oracle-state caveat unchanged); table-tennis-class dynamism
  (the owner rig's tasks are quasi-static manipulation — reactivity
  buys less there, another reason the #22 screen must *measure*
  naive-switch cost before buying anything).

## Where it lands

- **#22** record + arm order updated: HAS-on-decode slots in as the
  new second rung (after the naive-switch measurement, before PAINT);
  DEFLECT joins the training-time tier with the restart-corrected
  effect size attached; the parked-until-#16 gate is unchanged.
- **#16**: TTFA/E[reaction] accounting + ABPolicy's jerk instruments
  banked as rollout-eval design inputs.
- **#12**: FASTER is a fourth pole on the one-step axis — *one-step
  for the head of the chunk, many-step for the tail* — orthogonal to
  SnapFlow's distill-everything; relevant only if we ever revisit
  multi-step decode for quality (mean-of-10 is exactly that case).
- Radar hooks banked unread: RDT2 2602.03310 (UMI scaling; rode the
  queue but the cluster used the session), Spatial Forcing
  2510.12276 (VEGA's baseline, 3.8× training-accel claim).
