# LAFM: don't search the noise — learn a library of priors

*Read 2026-08-07 (same-session lit slice, the session that landed the
golden-ticket instrument). Source: Flowing With Purpose — Latent
Action Guided Flow Matching Policies for Robotic Manipulation
([2606.23420](https://arxiv.org/abs/2606.23420), June 2026). Fed:
[#1 noise-draw ensembling](../ideas/01-noise-draw-ensembling.md) —
it sits exactly one rung above the golden-ticket screen on the
"structure in the noise" ladder, and it names the rung above itself
too (DSRL, RL on the noise space).*

## The one-sentence contribution

Standard flow matching integrates every action chunk from the same
isotropic Gaussian; LAFM replaces that single prior with a **library
of K learned Gaussians indexed by discrete motion primitives**, picks
one per observation with a small classifier, and integrates from
there — the claim being that robot action spaces are fragmented and
heteroscedastic, so making the *starting point* mode-aware shortens
and disentangles the transport paths the vector field has to learn.

## How it actually works

Three pieces, all trained together on top of an ordinary flow policy
(110M params, DiT decoder — no VLA-scale trunk):

1. **A latent action model (LAM)** — a spatial-temporal transformer,
   pre-trained to compress the visual dynamics between consecutive
   frames into a discrete codebook (C = 512). This is the indexing
   signal: "what kind of motion is happening here", learned from
   pixels, no action labels needed at that stage.
2. **The prior library** — two embedding tables (means and
   log-variances, one row per code), initialized at N(0, I) and
   trained with the flow loss plus a KL leash back to N(0, I). Crank
   the leash to infinity and the method collapses to vanilla flow
   matching — a clean knob between "one shared prior" and "K
   specialized priors".
3. **A mode classifier** on the policy encoder predicts the code for
   the current observation; a cross-entropy term supervises it
   against the LAM's code.

At inference the chain is **deterministic in the mode and stochastic
within it**: argmax the classifier, sample ε from that mode's
Gaussian, Euler-integrate. There is a modest theory note (their
Proposition 1): if a prior's mean sits on the action mode's mean and
its variance is tighter than identity, expected transport distance
drops under perfect mode coupling — the formal version of "start
closer to where you're going".

## What they measured

- **LIBERO-90**: 93.0 ± 0.2% success vs 82.6 (their FM baseline),
  86.2 (ACT), 92.1 (π₀ at 3.3B — thirty times their parameter
  count). The 4-suite LIBERO average is 96.5 ± 0.3%.
- **Real Franka, 4 tasks, 50 demos each**: 86.7% vs 63.3% (FM) and
  71.7% (π₀). Real-robot rollouts, not offline metrics.
- **Ablations that matter for us**: the codebook sweep peaks at
  256–512 and *degrades at both ends* (too few priors = no
  specialization, too many = redundancy); the LAM's pre-training
  corpus barely matters (Fractal vs Droid vs mixed all ~92 on
  LIBERO-90 — motion diversity suffices); and the killer control —
  predicting latent actions as an auxiliary task WITHOUT the
  structured priors (their FM+LAC row) gets 85.4, far short of 93.0.
  The gain is in the priors themselves, not in the extra
  supervision signal.

## What transfers to us, and what doesn't

**The framing transfers cleanly.** Our golden-ticket screen
([pre-reg](../posts/2026-08-07-prereg-golden-ticket-screen.md)) asks
whether ONE searched constant noise vector beats fresh Gaussian
draws — the K = 1, search-not-learn corner of exactly the design
space LAFM lives in. The paper's own ladder reads: one shared prior
(vanilla FM) → one searched point (Golden Ticket) → K learned
mode-priors (LAFM) → RL directly on the noise space (DSRL, which
LAFM cites as the optimization-based sibling; not yet read here).
Our R4 per-dataset read is a cheap probe of the same
heteroscedasticity claim: if datasets disagree on the argmin ticket
with real margins, that is LAFM's "fragmented action space" showing
up in our data, and the escalation amendment has a literature-backed
shape — per-mode or per-dataset priors, not just a bigger ticket
search.

**The magnitude does not transfer.** Their +10.4 points on LIBERO-90
is against *their own* 110M FM baseline in a success-rate metric on
suites with strong task structure; our panel is pooled chunk MAE
over 878 heterogeneous datasets, where the banked prior says noise
effects are small (σ_draw 0.024 panel-scale) and the shared-ticket
regime regressed in the Golden Ticket paper itself. Nothing in LAFM
predicts our stage-1 kill line flips.

**Two structural caveats before anyone proposes an LAFM rung.**
First, LAFM is a *training-time* change — the priors co-train with
the vector field. Bolting learned priors onto our frozen teacher is
NOT the paper's method; the honest analogue for a frozen policy is
their cited DSRL direction (optimize the noise for a frozen policy),
which is the golden-ticket screen's own escalation path, or a
re-train — a much bigger ask than the eval-side screens #1 runs on.
Second, their LAM limitation is real for us: it indexes on visual
dynamics and admits it can latch onto "moving backgrounds or severe
camera shifts" — our community-curated corpus is exactly the messy
multi-rig regime where that failure mode lives.

## Which idea it fed

**#1**, two ways: (a) it upgrades the interpretation of the screen's
R4 task-locality read — per-dataset argmin disagreement is now
evidence *for the LAFM world*, not just a paper-replication
footnote; (b) it pins the escalation ladder above the ticket screen
in the ideas file: searched ticket → per-dataset tickets → learned
mode-priors (training-time, needs a retrain decision) → RL on noise
(DSRL — worth its own read if stage 1 CONFIRMs). No new arm
pre-registered from this read alone: the screen's own R1/R2 verdicts
decide whether any of that ladder is worth paying for.
