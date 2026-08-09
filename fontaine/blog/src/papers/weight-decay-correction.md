# Correcting corrected weight decay: what AdamC's successor says about our live run

*Read 2026-08-09 (lit slice `lit-radar-fresh-sweep-0810`, priority 1:
anything re-ranking the `adamc_100k` readout). Paper:
[2512.08217](https://arxiv.org/abs/2512.08217) — "Correction of
Decoupled Weight Decay" (Jason Chuan-Chih Chou, v3) — the direct
successor to the AdamC paper our live optimizer implements
([2506.02285](https://arxiv.org/abs/2506.02285), Defazio, "Why
Gradients Rapidly Increase Near the End of Training").*

**The paper in plain words.** Weight decay is the small force that
keeps a network's weights from growing without bound, and almost
everyone scales it in lockstep with the learning rate as training
winds down. The AdamC paper we run in production argued this lockstep
scaling is exactly right, and that it explains (and fixes) a
well-known pathology where gradients balloon near the end of
training. A competing camp argues the decay should instead shrink
with the *square* of the learning rate. This paper wades into that
argument with a cleaner derivation and some careful simulations, and
lands mostly on AdamC's side of the fence — decay proportional to the
learning rate, not its square — while showing that the *reasoning*
AdamC used to get there (an orthogonality argument about update
directions) doesn't actually hold up: you can delete the effect that
argument leans on and training barely changes. The practical upshots
it adds: keep the output layer out of the correction, expect the
benefit to show up in norm *stability* rather than final accuracy for
Adam-family optimizers, and don't decay the learning rate all the way
to zero.

## What it contributes

- **A re-derivation of λ ∝ γ from steady-state assumptions** (updates
  become independent of weights; minibatch gradients decorrelate;
  momentum correlation decays), giving
  λ_t = ((2−α)/(2α·C²)) · γ_t with a momentum-dependent effective
  learning rate γ_eff = γ·√((2−α)/α) that transfers across momentum
  values better than raw γ.
- **A refutation of the orthogonality mechanism**: a "renormalized"
  AdamW that eliminates the perpendicular update component entirely
  changes ViT-S/16 top-1 by 0.3 points (77.15 vs 77.45) — so the
  perpendicular-component story behind both AdamC's derivation and
  the rival γ² proposal (Kosson et al.) mischaracterizes what drives
  the dynamics, even where the λ ∝ γ *conclusion* survives.
- **ScionC**: the same correction applied to the Scion optimizer,
  where (unlike Adam) the theory's normalized-update assumption
  actually holds, with a norm-scheduling story (C_t can be scheduled;
  momentum scheduling can substitute for cosine LR decay).

## The experiments it ran

- **Modded-NanoGPT 124M** on FineWeb-Edu-100B (8×H100): ScionC
  validation loss 2.838 vs Scion 2.846, with visibly more stable
  weight/gradient/spectral norms.
- **ViT-S/16 on ImageNet-1k** (30–300 epochs, batch 1024): at 90
  epochs, AdamW 76.92 ± 0.13, **AdamC 76.98 ± 0.10** (a wash),
  Scion 78.68 ± 0.09, ScionC 78.74 ± 0.09. The corrected variants
  need a *higher* peak λ than their uncorrected baselines. Notably,
  **AdamC-trained models do not reach steady state even at 300
  epochs**, while ScionC does.
- **Numerical simulations** of the steady-state theory: excellent
  match for vector norms and rectangular matrices, ~10% deviation for
  square matrices (i.e. attention-shaped weights).

## What transfers to us

The live `fontaine_molmo2_adamc_100k_ddp4` run uses AdamC
λ̂_t = λ·γ_t/γ_max on hidden matrices with the output head excluded
(the audited partition in `bijou/train.py`), under a cosine schedule
that floors at 10% of peak. Three direct reads onto that run:

1. **The partition is doubly validated.** Chou specifically notes
   Defazio's own Llama-3 experiments apply the correction *excluding
   the output layer*, and his derivation independently finds the
   steady-state independence assumption violated exactly there. Our
   head-exclusion + tied-parameter guard is the recommended shape,
   now in two papers.
2. **The grad-norm chart at endpoint gets its interpretive frame.**
   Expected signature if AdamC is doing its job: flat gradient-norm
   and weight-norm trajectories through the decay phase (vs the
   AdamW pathology of late-training gradient growth). Expected
   effect on final loss: **approximately nothing** — AdamC vs AdamW
   was 76.98 vs 76.92 on ViT — which matches our record-only framing;
   the chart is a stability read, not a performance claim. Caveat to
   carry: at 100k steps we may be in the "never reaches steady state"
   regime Chou measured for AdamC, so a slowly drifting weight norm
   is *consistent with* the theory, not a falsification.
3. **Our 10%-of-peak LR floor is on the right side of the terminal-LR
   argument.** λ ∝ γ (rather than γ²) specifically avoids terminal
   weight-norm suppression, and Chou reads the common practice of
   non-zero terminal LR as evidence for it. Since the correction
   multiplies decay by γ_t/γ_max, our floor also keeps λ̂ at 10% of
   base rather than driving it to zero — coherent with the paper's
   recommendation, worth stating in the endpoint readout.

## What doesn't transfer

- **ScionC itself.** The headline gains (78.7 vs 76.9) come from
  switching optimizer families, not from the correction — and a
  Scion arm would be a from-scratch optimizer ablation, exactly the
  exhaustive-ablation shape the startup-velocity rule exists to
  block. Radar-only unless a trunk-scale rerun is on the table
  anyway.
- **The momentum-scheduling substitute for cosine decay** — elegant,
  Scion-specific in its current form, unpriced on Adam.
- The theory's square-matrix deviation (~10%) means quantitative
  norm *predictions* for attention blocks are soft; qualitative
  stability reads are unaffected.

## Which idea/arm it fed

The `adamc-100k-live` endpoint readout (grad-norm + weight-norm
chart): interpretive frame + steady-state caveat + terminal-LR note
banked here. No new arm; no change to the live run. Cross-refs:
the AdamC implementation notes in `bijou/train.py`
(`adamc_output_head_parameters`), the run pre-reg
([parameter sheet](../posts/2026-08-09-prereg-molmo2-adamc-100k.md)).
