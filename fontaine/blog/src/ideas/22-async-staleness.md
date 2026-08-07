# 22. Async staleness bridging for rollout (RTC / A2C2 / TT-RTC) — `parked` (waits on #16)

*Tag: `async-staleness` · idea #22 · [index](../ideas.md)*

- **Hypothesis:** at real deployment latencies, naive async chunk
  switching (our `--async-inference`) loses task quality to
  observation staleness — and the loss is *decode-dependent*:
  single-draw 1-NFE (~2 ticks @30 Hz) is in the survey's "everything
  works" regime, mean-of-10 batched (576 ms ≈ 18 ticks, chunk 50) is
  deep in the regime where naive switching degrades and
  inference-time RTC has already collapsed
  ([papers page](../papers/2026-08-07-async-chunk-execution.md):
  RTC 2506.07339 + async-methods comparison 2605.08168).
- **Expected effect:** rig/sim rollout quality at high-value decodes;
  invisible to the offline panel by construction (staleness is a
  closed-loop phenomenon).
- **Cost:** screen = 0 training (measure naive-switch cost at both
  decodes on rig rollouts once #16 exists); first arm if real =
  A2C2-style residual correction head (frozen base, composes with
  batched draws), second = TT-RTC prefix-conditioning fine-tune
  (~25% of base training, weak at chunk 50).
- **Falsification:** paired rollouts, same checkpoint, single-draw vs
  mean-of-10 under naive switching: if mean-of-10's closed-loop win
  survives its 9× staleness, bridging buys nothing — park forever.
- **Gate:** parked until the #16 rig-transfer bench exists; the
  survey's regime table (delay-in-ticks × chunk length decides the
  winner) is the design input for any pre-reg here.

## Record

- **2026-08-07 ~20:3xZ — PAINT read, arm order re-banked**
  ([noise-space steering II](../papers/noise-space-steering-2.md),
  2606.19774): training-free initial-noise selection solves the
  chunk-boundary problem without gradients — backward-Euler invert a
  target endpoint (executed prefix + draft tail) to noise, keep only
  the inverted *prefix* of ε, splice fresh suffix noise, integrate
  forward (~3N calls). Matches/beats RTC on real tasks (0.85 vs
  0.75 Toy-in-Drawer), most delay-robust method on Kinetix at d=4,
  composes with TT-RTC, demonstrated on a chunk-50 π₀ — our regime.
  **Arm order now: PAINT (zero training) → A2C2 residual → TT-RTC.**
  Design note banked (ours, needs its own oracle if ever built):
  shared inverted prefix + per-draw fresh suffixes should let PAINT
  compose with mean-of-10 batched draws. Caveats: locality
  assumption (prefix-of-ε ↔ prefix-of-chunk, OT-FM-encouraged, not
  enforced — probeable with our draws machinery, noted on #1);
  off-manifold executed prefixes invert poorly. Gate unchanged:
  parked until #16's rig bench exists.
