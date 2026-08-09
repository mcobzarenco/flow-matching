# Hyperball: what weight decay was actually doing all along

*Read 2026-08-09 (lit slice `lit-radar-0814`, priority 1:
adamc-watch adjacency). Paper:
[2606.16899](https://arxiv.org/abs/2606.16899) — "Fantastic
Pretraining Optimizers and Where to Find Them II: Hyperball
Optimization" (Wen, Dang, Lyu, Ma, Liang — Stanford/Marin; v1
2026-06-15, preprint). Fourth paper in our corrected-decay reading
after AdamC itself, [Chou](weight-decay-correction.md) and
[Muon-SW](muon-sw.md) — and the one that unifies the other frames.*

**The paper in plain words.** Weight decay, this paper argues, was
never really a regularizer in modern LLM training — it's an indirect
thermostat. It sets the equilibrium size of each weight matrix, and
that size in turn sets how fast the matrix's *direction* can rotate
per step. Rather than steer rotation speed through two coupled dials
(learning rate and decay), Hyperball removes the thermostat: clamp
every matrix to its initialization norm and normalize every update,
so the learning rate directly *is* the angular speed. Wrapped around
Muon this sustains a 20–30% training-speed advantage at 1.2B (vs
~10% unwrapped) and makes the best LR far more stable across model
sizes. On the way it derives and validates exactly the norm dynamics
our AdamC watch charts: weight norms ride the LR schedule, and
gradient norms rise as weights shrink.

## What it contributes

- **Hyperball**: `W ← R·Normalize(W − η·R·Normalize(u))` with
  `R = ‖W₀‖_F` fixed at init — weight norm and step norm both
  pinned, so per-step angular displacement ≈ η by construction.
  Applied to attention/MLP matrices only; embeddings, norm gains and
  scalars stay on plain Adam ("the norm can carry semantic
  information" there). An expressivity argument covers the clamp:
  `(W, γ) → (cW, γ/c)` around RMSNorm gains preserves the function.
- **The equilibrium law our watch needed.** Under decoupled decay
  (idealized stationary-gradient model), a matrix's norm settles at
  **R⋆ ∝ √(η/λ)**. Everything else falls out of that one line:
  constant λ with decaying η ⇒ the equilibrium itself decays ⇒
  weight norms track the LR schedule down (the uncorrected
  signature); AdamC's λ_t ∝ η_t ⇒ η/λ constant ⇒ **plateau-then-flat**
  — the Chou and Muon-SW expectation, here from a third independent
  derivation. Notably the paper never cites AdamC; its own theory
  implies the correction, and its most conspicuous missing baseline
  is exactly a λ_t ∝ η_t schedule.
- **The gradient-norm side, finally mechanized.** For prenorm
  scale-invariant blocks, `‖∇L(cW)‖ = (1/c)·‖∇L(W)‖` — grad norm ∝
  1/‖W‖. That is the mechanism behind "grad norms grow during LR
  decay": the weights are shrinking under them.

## The experiments it ran

Qwen3-style decoders on a DCLM+code+math mix, LR swept on a √2 grid
per scale. MuonH sustains **20–30% token-equivalent speedup** over
AdamW at 1.2B where plain MuonW decays to ~10% (Fig. 2); optimal-LR
drift across width/depth ≈1.4× vs 3–4× for baselines; an 8B Marin
run finishes 0.04 loss lower (single run, hand-tuned); modded-NanoGPT
speedrun entries land at 3.278 in fewer steps. Theory validation:
weight norms follow the LR schedule and grad norms rise through
decay (Fig. 9); two AdamW runs at fixed η·λ produce near-identical
loss with 2× different layer norms (Fig. 10, confirming R⋆ ∝ √η at
fixed ηλ). Fine print: baseline λ apparently not swept, no
scheduled-λ (AdamC-style) control, no isolated ablation of the two
normalizations, 8B is one run.

## What transfers to us

The `adamc_100k` watch's **third frame — and it upgrades the watch
from one-sided to two-sided**:

1. **All three frames now agree from independent directions.**
   Grad-norm dynamics (AdamC), summability (Muon-SW), equilibrium
   analysis (here) all land on λ ∝ η ⇒ flat norms. The expected
   endpoint chart: **warmup ramp over steps 0–1000 (their eq. 39
   predicts norms rise during warmup), then plateau-and-flat through
   the cosine decay.** Peak-then-decline = the uncorrected
   signature.
2. **New chartable prediction — the grad side.** For the corrected
   matrices group (where a prenorm precedes the matrix): if the
   correction holds norms flat, **per-group grad norms should also
   stay ~flat through decay**; a grad-norm climb mirroring 1/√η_t
   with sagging norms is the uncorrected shape. Our watch already
   records both series; this pins what "healthy" looks like jointly.
3. **Free offline probes from banked checkpoints** (no training-loop
   change): (a) per-matrix ‖∇L‖·‖W‖ constancy across checkpoints —
   flags where the scale-invariance lemma actually applies; (b)
   stable rank ‖W‖²_F/‖W‖²_op per matrix — tells us whether
   Frobenius tracking proxies spectral behavior in our trunk. Both
   are cheap, both are record-only chart candidates for the endpoint
   readout.
4. **The sharpest interpretive trap named so far, sharpened.** At
   our λ=1e-5 on a *pretrained* 4B init, the equilibrium may simply
   never be reached — the relaxation timescale ~1/(ηλ) exceeds the
   run, and R⋆'s natural anchor (init norm from scratch) doesn't
   describe a pretrained trunk. **Flat norms at our λ could mean
   "decay inert," not "correction working."** The grad-norm side
   (point 2) and Muon-SW's alignment-cosine probe are the
   disambiguators; this caveat rides the endpoint chart note beside
   Chou's steady-state warning.

## What doesn't transfer

- **Hyperball itself** — an optimizer-family change; same
  startup-velocity verdict as Muon/ScionC on the sibling pages.
  Radar-only.
- **The speedup numbers** are Muon-vs-Muon from-scratch LM
  pretraining facts at ≤1.2B (8B anecdotal), possibly inflated by
  undertuned baseline λ — their own theory predicts a λ-scheduled
  baseline would close part of the gap.
- **Scope of the grad law**: only prenorm scale-invariant blocks.
  Our head-excluded and no-decay groups are outside it; apply the
  two-sided read to the corrected-matrices group only.
- The equilibrium formulas assume stationary isotropic gradients —
  directional guidance, not quantitative fits, for a fine-tune on a
  manipulation corpus.

## Which idea/arm it fed

The `adamc-100k-live` endpoint readout — the watch is now
**two-sided** (norms AND grads, expected shapes pinned jointly),
with two new free offline probes (grad·norm constancy, stable rank)
and the decay-inert trap named as the alternative explanation flat
norms must rule out. No new arm; no change to the live run.
Cross-refs: [Chou](weight-decay-correction.md),
[Muon-SW](muon-sw.md), the run
[parameter sheet](../posts/2026-08-09-prereg-molmo2-adamc-100k.md).
