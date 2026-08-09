# Scale buys cycles, not immunity — and the WD half of this hook is someone else's paper

*Read 2026-08-09 (lit slice `lit-radar-0818`, priority 4). Paper:
[2606.24752](https://arxiv.org/abs/2606.24752) — "Can Scale Save Us
From Plasticity Loss in Large Language Models?" (J. Fernando
Hernandez-Garcia, Tomás Figliolia, Beren Millidge; Zyphra; submitted
2026-06-23; license: standard arXiv non-exclusive, not CC).*

**The paper in plain words.** When a neural network is trained on one
thing after another for a long time, it slowly gets worse at picking
up anything new — this is "plasticity loss." The authors ask whether
simply making language models bigger fixes the problem. They train
small-to-medium GPT-style models by cycling through eight languages
over and over, and every so often they check how quickly a copy of
the model can learn a ninth, held-out language (Vietnamese). Every
model size eventually gets slower at learning Vietnamese; bigger
models just take longer to start degrading. Fitting a curve to when
degradation starts, they conclude that making models bigger delays
the disease but does not cure it. They also find the degradation
happens even when the eight languages are mixed together with no
task switches at all, and that none of the popular cheap "health
metrics" (dead neurons, weight size, attention collapse) reliably
predicts when the degradation starts.

## What it contributes

- First systematic *scaling* study of plasticity loss in GPT-style
  LMs: 8 sizes, 5M / 12M / 27M / 39M / 53M / 83M / 106M / 314M
  non-embedding params (head dim 64 fixed, d_model/L ≈ 80, tied
  embeddings, pre-LN, GeLU).
- Operational metric: plasticity = ability to improve on a probe
  distribution under a **fixed adaptation budget** — AUC of
  validation loss while adapting 5B tokens to held-out Vietnamese
  (chosen for low transfer from the training languages). Rising AUC
  across cycles = plasticity loss.
- Onset law: onset task-instance T = 1.3e-5 · P^0.8269 (P =
  non-embedding params), fit on *measured* onsets — e.g. 5M degrades
  "immediately after the first cycle," 83M after cycle 7, 314M at
  task instance 118 (~cycle 15). Sublinear exponent < 1 is the whole
  headline: "Scale alone cannot save us from plasticity loss."
- Plasticity loss appears **without task switches**: stationary
  8-language mixture also degrades (5M/12M/27M tested; the 27M
  crosses into net degradation at ~475B tokens stationary vs ~320B
  continual).
- Negative instrument result: parameter-norm growth, dormant units,
  and attention entropy all fail to track onset ("None of the three
  metrics perfectly tracks the onset or severity"; "we do not yet
  manage to find a 'smoking gun'").

## The experiments it ran

- Data: CulturaX; 8-language cycle (En, Zh, Fr, Ja, Es, De, Pt, Ru),
  5B tokens per task instance; up to 48 cycles = 384 task instances
  ≈ 1.92T tokens for the longest continual runs. Probe: Vietnamese,
  5B-token adaptation on a model copy, 1B-token validation.
- Optimizer: AdamW (0.9, 0.95), **weight decay 0.1 fixed
  everywhere** — never swept, never intervened on. Batch 0.5M
  tokens; LR power-law interpolated 3e-3 (5M) → 1e-3 (314M);
  constant LR after 5% linear warmup; optimizer state reset and
  warmup restarted at each task boundary.
- Onset detection (App. B): minimum of a window-3 moving average of
  the probe-AUC-vs-cycle curve; Table IV lists measured onsets per
  size (314M → 118).
- Correlates, with their failures: 53M reached >95% dormant units in
  layer 8 and 106M ~80% in layer 10 (ε=0.01), but the 12M lost
  plasticity with *no* dormancy growth after cycle 5; parameter
  magnitude grew for the 53M through cycle 7 while the probe was
  still *improving*, and *fell* for the 106M over cycles 8–20 while
  it was *deteriorating*; collapsed-attention-head counts trended
  opposite directions in the 53M vs 106M.
- No mitigation experiments. Resets (Continual Backprop, ReDo,
  Self-Normalized Resets), shrink-and-perturb, and higher WD are
  *discussed as future directions only*.
- No gradient-norm analysis anywhere in the paper.

## What transfers to us

- **The instrument lesson, not the phenomenon.** The only measure
  that tracked plasticity was behavioral: fixed-budget adaptation
  speed (loss AUC) on a held-out distribution. That is exactly the
  shape of our probe-MAE-under-budget instruments, and it argues for
  keeping them behavioral for second-phase questions (rig fine-tune
  episodes) rather than trusting cheap network-health proxies.
- **Do not build watch/kill lines on dormant units, param norms, or
  attention entropy.** Their own in-domain data shows each proxy
  giving false positives and false negatives on onset. Feeds #17's
  instrument-selection notes directly.
- **A frame for repeated rig fine-tunes.** Each future fine-tune of
  the trunk on new rig data is one "task instance" in their terms.
  Extrapolated onset for a ~3.5B non-embedding trunk is ~1,000 task
  instances (~5T tokens of repeated shifts) — enormous headroom, but
  if we ever chain many sequential fine-tunes, tracking adaptation
  AUC across episodes is the right early-warning, per this paper.
- Weak color for the AdamC motivation: parameter norms grew for many
  cycles despite WD 0.1 under constant LR — consistent with the
  "AdamW regularization strength drifts over training" story, though
  the paper never analyzes gradients or intervenes.

## What does NOT transfer

- **Scale claim stops at 314M.** Our trunk is 4B — 12× beyond their
  largest measured point; anything at our scale rests on a power-law
  extrapolation the authors themselves hedge ("likely to be
  insufficient"). LLM next-token prediction only: no vision, no
  action heads, no fine-tuning regime.
- **Nothing about weight decay is this paper's evidence.** WD=0.1
  fixed throughout; the "higher WD improves plasticity despite worse
  pretrain loss" line in the banked hook is their *citation of Han
  et al. 2602.11137* — the paper we already read
  ([weight-decay-plasticity](weight-decay-plasticity.md)). It
  neither licenses nor forbids anything at our λ=1e-5, and it is
  not an independent vote for the WD-plasticity link.
- **Onset requires exposure we will never see in the live run.**
  314M onset needed ~590B tokens of repeated distribution shifts;
  our 100k-step fine-tune is task instance ~1 at low-single-digit-B
  tokens. No observable prediction for the adamc run — no
  grad-norm statement exists in the paper at all.
- Stationary-degradation result is real but starts at ≥320B tokens
  even for a 27M model; irrelevant at our budgets.
- Optimizer resets + warmup restarts at every task boundary are a
  confound for mapping "task instance" onto our continuous runs.

## Which idea it feeds

- **#17 (optimizer/trunk-recipe radar):** record-only for the AdamC
  grad-norm watch (paper is silent on gradients); bank the negative
  instrument result — behavioral fixed-budget probes beat cheap
  health proxies for plasticity; note the hook's WD clause as a
  duplicate citation of 2602.11137, not new evidence.
- Secondary: future-rig fine-tune planning — adaptation-speed AUC
  across sequential fine-tune episodes is the early-warning metric
  if the trunk ever accumulates many continual-learning steps.
