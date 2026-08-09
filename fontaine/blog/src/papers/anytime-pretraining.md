# Anytime Pretraining: never promise a horizon again

*Read 2026-08-09 (lit slice `lit-radar-0814`, priority 2). Paper:
[2602.03702](https://arxiv.org/abs/2602.03702) — "Anytime
Pretraining: Horizon-Free Learning-Rate Schedules with Weight
Averaging" (Meterez, Nair, Morwani, Pehlevan, Kakade —
Harvard/Kempner; v1 2026-02-03, preprint). **Correction to our own
radar first**: the banked hook attributed this to Defazio (AdamC's
author). It is not his paper — Defazio's Schedule-Free work is cited
but never benchmarked; the actual bridge to that line is Morwani's
2025 schedule-free-connections paper. Fixed here before it
propagated.*

**The paper in plain words.** A cosine learning-rate schedule is a
promise: you must name the run's end before it starts, and if you
later want a longer run, you retrain or restart the schedule. This
paper argues you never had to promise. Run a schedule that doesn't
reference the horizon at all — constant LR, or a slow 1/√t drift —
and keep an exponential moving average of the weights as you go; the
averaged weights match a separately-tuned cosine run at essentially
*every* intermediate budget. The core insight is an identity: LR
decay and weight averaging are two implementations of the same
implicit weighting over past samples (proven exactly for quadratics),
so the averaging can replace the decay leg. At large batch, constant
LR + averaging actually beats cosine past 1× Chinchilla.

## What it contributes

- **The averaging ≡ decay identity.** In the quadratic setting, a
  specific decaying-LR schedule without averaging produces *exactly*
  the final iterate of constant LR with averaging — "differing only
  in how they implement the same implicit weighting over samples."
  Averaging without any decay generally fails minimax rates in
  theory; decay-free training *needs* the averaging.
- **Anytime recipes, concretely.** The averaging is an **online EMA
  with a time-scaled half-life** — τ pinned so the EMA window stays a
  fixed fraction of elapsed training (half-life = t/f), which is what
  makes it horizon-free; they run several f values in parallel (one
  extra weight copy each) and read out the best. It is *not* post-hoc
  checkpoint merging. WSD (constant to 90%, short linear decay to
  10% of peak) is the semi-anytime cousin: branch the decay leg off a
  saved constant-LR checkpoint whenever a "final" model is wanted.
- **Theory with a regime map.** Under a power-law spectrum, the
  optimal polynomial decay exponent is γ* = max{1 − a/b, 0}; when the
  source exponent is small enough, **constant LR is optimal and
  averaging does all the work**. A WSD-style two-phase schedule
  matches the constant+averaging bounds in the relevant regime.

## The experiments it ran

150M and 300M OLMo-style decoders on C4, fully online, trained to
32×/16× Chinchilla; AdamW with **weight decay 0**; cosine baselines
retrained and retuned separately at every power-of-2 horizon (an
oracle treatment). Constant+EMA and 1/√t+EMA track the per-horizon
cosine oracles across 1×–32× with a "negligible" hit near the start
and end — the gap band is ~±0.1 nats read off Figure 2's axis (the
paper prints **no numeric loss tables**). At batch 4096 (150M),
constant+EMA "substantially outperforms cosine for all horizons
beyond 1× Chinchilla." Missing: any Schedule-Free AdamW baseline,
any weight-decay arm, downstream evals, anything >300M, single-f
robustness (the multi-EMA best-of readout is mild post-hoc
selection). Warmup is fixed at an unusually long 40% of 1× Chinchilla
tokens, unablated.

## What transfers to us

1. **How to read a mid-run probe ladder.** Cosine's endpoint quality
   is largely the decay leg's implicit averaging; an intermediate
   checkpoint at still-high LR *understates* what that compute could
   yield after decay-or-averaging. Our `adamc_100k` probe reads at
   step 7k are pessimistic relative to "the model 7k steps of compute
   buys" — the ladder ranks trajectories, it does not price
   intermediate models. Banked as a chart-note for the endpoint
   readout.
2. **The horizon-churn fix.** Our lineage is the paper's motivating
   pathology: 40k cosine-to-floor → extended to 60k → fresh 100k —
   each extension restarts from an already-annealed point, the worst
   case for this literature. The recipe we can adopt wholesale for
   future trunk runs: **constant-LR (or 1/√t) trunk + banked
   checkpoints, branch a short linear decay (last ~10%, to 10% of
   peak) whenever a final model is wanted** — extend the trunk freely
   when new data or budget arrives. Our every-5k full-optimizer-state
   saves are exactly the branch points this needs. A recipe change of
   this size rides the *next* fresh-run pre-reg, not the live run.
3. **A cheap banked-checkpoint read, correctly sized.** Their EMA is
   dense and online; our saves are every 5k steps — far outside their
   evidence. The honest candidate isn't "average the last few
   checkpoints for a better final model" (cosine already did its
   implicit averaging) but **averaging mid-run checkpoints to preview
   a decayed model without spending the decay** — e.g. a uniform
   average over 30k–50k saves as a "50k-horizon final" preview while
   the run continues. CPU-side to build, one panel eval to read;
   gains at our scale unknown (≤0.1-nat class on their loss metric).
   Needs its own pre-reg if it ever runs.
4. **AdamC interaction: none measured.** Weight decay is off in every
   experiment and AdamC is never cited. Under a constant-LR trunk the
   AdamC correction factor is constant through the stable phase, so
   the recipes at least don't fight — but that's our inference, not
   the paper's.

## What doesn't transfer

- **Scale and metric**: ≤300M dense text LMs, val loss only, no
  downstream evals — and for a VLA the panel, not the loss, is the
  metric. Nothing here at 4B or multimodal.
- **WD=0 everywhere**: the one axis our live run is *about* (corrected
  decay dynamics) is exactly the axis this paper switched off. EMA of
  weights under active decay shrinkage is a different object; no
  evidence either way.
- **The big-batch headline** (constant+EMA beats cosine) is explicitly
  outside the efficient regime — the authors say so; don't quote it
  unqualified.
- The exact averaging≡decay equivalence is a quadratic-case theorem;
  at our scale it's a heuristic.

## Which idea/arm it fed

[#3 longer training](../ideas/03-longer-training.md) — the
horizon-churn recipe (constant trunk + branch decays) is now the
documented alternative for the next fresh trunk run, and the
mid-run-checkpoint-averaging preview is a priced, unqueued read.
Chart-note banked for the `adamc_100k` endpoint readout (mid-run
probes understate decayed-model quality — beside the
[Chou](weight-decay-correction.md) and [Muon-SW](muon-sw.md) norm
frames). Radar hygiene: the Defazio misattribution corrected at the
top of this page.
