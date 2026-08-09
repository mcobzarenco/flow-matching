# Legato: train the flow to expect the splice instead of clamping it at inference

*Read 2026-08-09 (lit slice `lit-radar-0817`, priority 4 cluster
with [Reflex](reflex.md) — train-time vs inference-time complements
for chunk transitions). Paper:
[2602.12978](https://arxiv.org/abs/2602.12978) — "Legato: Learning
Native Continuation for Action Chunking Flow Policies" (Liu et
al., v2 2026-05-17, RSS 2026). [Project
page](https://lyfeng001.github.io/Legato/); real-robot code not
released (only a Kinetix sim repo).*

**The paper in plain words.** Chunked robot policies plan a couple
of seconds of motion at a time, and the seams between chunks are
where arms hesitate or jerk: the new chunk doesn't quite agree with
the tail of the old one. The standard fix (RTC, real-time chunking)
forces agreement at execution time — it clamps the start of the new
chunk to the old chunk's tail while the flow model denoises. But
the model was never trained with anyone clamping its outputs, so
it's being steered in a way it never learned to expect. Legato
moves that steering into training: the flow objective is reshaped
so the model *learns* to generate under a "continue from these
known actions" schedule, with the schedule itself given as an
input. On a real dual-arm robot this mostly doesn't change how
smooth the motion is frame-to-frame — what it changes is
hesitation: tasks complete ~20% faster than under RTC, at equal or
better task scores.

## What it contributes

- **A guidance-aware flow objective**: replace the pure-noise
  source with a schedule-shaped mixture ε_eff = (1−ω)⊙ε + ω⊙A
  (ω ∈ [0,1]^H per action timestep), and rescale the velocity
  target accordingly — the per-step guidance that RTC applies at
  inference is internalized into the learned field, so training
  dynamics and inference dynamics match.
- **Schedule conditioning**: ω is appended to the expert's input,
  and the schedule is randomized during training (delay d ~
  U[0,10], ramp r ~ U[0,50]) — one model serves any deployment
  latency without retraining. Removing this conditioning
  measurably degrades boundary overlap (Table IV).
- No architecture change, no extra data, no cross-chunk pairs —
  the "known actions" during training are ground truth from the
  same chunk.

## The experiments it ran

Fine-tuned from the *same* π0.5 checkpoint as the RTC baseline,
identical data and hyperparameters — a properly matched
comparison. Real dual-arm platform (7-DoF ×2, 3 cameras), five
tasks, 30–50 trials each, chunk H=60 at 30 Hz, N=5 denoising
steps. Versus RTC: **completion time −19% to −23%** on every task
(e.g. stack bowls 52.9→42.7 s, pour 95.1→75.7 s); task scores
(0–10 rubric) all improve, several within error bars; boundary
overlap RMSE roughly halves on 3 of 5 tasks; frequency-domain
smoothness (NSPARC) is nearly flat (<1% on 4 of 5 tasks) while
jerk-based NLDLJ improves clearly (pour 2.85→1.65). Robust across
injected delays d ∈ {6,8,10} (~200–333 ms), and replicates on π0
(92.9→88.3 s).

## What transfers to us

- **It lives entirely in the expert's objective** — trunk-agnostic,
  so it drops onto our frozen-Molmo2 + flow-expert stack as an
  expert fine-tune (modified FM target + ω conditioning). No trunk
  surgery.
- **The right sequencing for our stack**: RTC-style inpainting
  first — inference-only, zero training cost, banked machinery from
  the async-chunk-execution read — measure boundary jerk/hesitation
  on the rig; pay Legato's fine-tune only if RTC's
  train/inference mismatch shows up as a measured problem. Legato
  is the *upgrade path*, not the first move.
- **Its real gain is a wall-clock metric**: ~20% completion time =
  less hesitation and multimodal dithering at seams. That is an
  on-rig metric — our offline chunk-MAE panels would barely see
  it. Files under the rig-day eval design (#16's benchmark), not
  the current panel ladder.

## What doesn't transfer

- **It is an objective change, and that has bookkeeping teeth**:
  a Legato-trained expert's losses/probes are NOT on the same
  scale as standard-FM runs (the velocity target is rescaled) —
  it can never be a retrofit inside a matched-recipe comparison;
  it must be its own arm.
- **The denoising step count is baked in at training** (their
  stated limitation, N=5) — incompatible with #19-style ODE-step
  sweeps at inference on the same checkpoint.
- **It replaces RTC rather than composing with it** (same
  functional slot: the chunk-transition mechanism). It composes
  fine with Reflex's serving layer underneath.

## Hook corrections

The banked one-liner ("native chunk-continuation training, ~10%
smoother vs RTC") mis-sells the result in both directions:
frequency-domain smoothness is nearly *flat* (the ~10% is a blend
across three heterogeneous smoothness metrics, carried by jerk and
overlap-RMSE) — while the **completion-time gain (~20%) is the
actual headline** and the one-liner omitted it entirely.

## Which idea/arm it fed

#22 (`async-staleness`) — the chunk-transition menu is now a
two-rung ladder with measured spacing: RTC (free, inference-only)
→ Legato (fine-tune, −20% completion time, schedule-conditioned
delay robustness); adopt RTC first, Legato gated on measured
boundary artifacts at rig time. #16 — completion time and
boundary-overlap RMSE join the rig-benchmark candidate metric set
(offline panels are blind to the seam behavior Legato fixes). #19
cross-note: Legato's fixed-N training couples the expert to one
solver budget — a conflict to remember if draws machinery ever
meets a Legato arm. No gate changes.
