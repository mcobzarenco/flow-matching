# The state shortcut — proprioception as a crutch, and what we measured when we kicked it away

**Papers:** Adapt Your Body
([2506.23944](https://arxiv.org/abs/2506.23944), withdrawn), the
state-free policy ([2509.18644](https://arxiv.org/abs/2509.18644)),
ReViP ([2601.16667](https://arxiv.org/abs/2601.16667)), GAP
([2602.12032](https://arxiv.org/abs/2602.12032), ICLR 2026),
ThinkProprio ([2602.06575](https://arxiv.org/abs/2602.06575)), and
Cloak ([2606.22836](https://arxiv.org/abs/2606.22836)). Banked
across the 2026-08-06 lit slices; re-read at full-text depth for
this page. **Fed:** #9 (state dropout — pre-registered, run, and
*falsified at p=0.8* on our stack) and #11 (the grounding gap /
reliance probe, where the mechanism was *confirmed*). This is the
rare theme where we can score the literature against our own
completed experiments — and where the retroactive deep read
caught our banked citation being wrong in a way that matters.

## The theme

Imitation-trained manipulation policies get two views of the world:
cameras, and the robot's own joint state. State is low-dimensional,
noiseless, and — because expert demos are smooth — an excellent
predictor of the *next* action from the *current* one. That makes
it a classic shortcut feature: fast to learn, great on the training
distribution, and a liability the moment reality diverges from the
demo manifold. The papers here all orbit one question: how much of
a policy's competence is visual understanding, and how much is
dead-reckoning off proprioception? Their answers range from
"remove state entirely" to "rebalance it adaptively" — and they
disagree in an instructive way.

## What we ran — both sides of the story

This literature fed two completed experiments here, which came back
with a spliced verdict.

**The mechanism is real (reliance probe, #11).** We froze a masked
evaluation (`--mask-state` substitutes the dataset state mean) and
asked whether B — the arm with the *better* intact first-action
error — was buying it with heavier state reliance. It was: D =
Δ_first(B) − Δ_first(A-s0) = **+0.702**, CI95 [0.498, 0.916], 14×
the pre-registered threshold, the chunk secondary agreeing
(+0.389). Exactly the causal-confusion story: the better
leaderboard number was partly a deeper draw on the proprioceptive
crutch.

**The naive fix did not pay (state dropout, #9).** The
supposedly literature-endorsed lever — zero-masking state with
p=0.8 — went into a pre-registered 40k arm (`--state-dropout 0.8`).
Outcome: paired Δchunk **+2.64** MAE [2.55, 2.74], far outside the
±0.15 band — a clear COSTS verdict. But the *mechanism* worked
exactly as advertised: the dropout-trained model barely notices
state masking (first_mae 8.56 intact → 8.11 masked) while the
baseline collapses under the same mask (3.94 → 24.08). Our
one-sentence summary: **dropout kills the shortcut without teaching
the replacement.** A p=0.3 screen stays queued; nothing was
adopted.

The deep read then re-framed both results — see the first paper.

## 1. Adapt Your Body — our citation was wrong, and the paper is withdrawn (2506.23944, v1)

The correction first: **p=0.8 zero-masking is not this paper's
method.** It is their *baseline* — "Random Dropout: randomly mask
proprioception input to zeros with probability 80%" — which their
actual method beats. The method (NADA) is two-pass: train a policy,
roll it out, measure a time-conditioned Wasserstein distance
between expert and rollout state distributions, pick the **noise**
scale σ\* that minimizes it, and retrain with Gaussian-noised
proprioception. On Robomimic/MimicGen (9 sim tasks, 3 seeds) NADA
posts e.g. Square 56.7 vs dropout's 44.8, StackThree 61.4 vs 52.7 —
best on 6/9 tasks — with the optimal σ visibly task-dependent
(0.6 to 1.2 on their grid). Their per-dimension analysis fingers
*velocity* terms as the most harmful proprio components. And then
the kicker: **v2 of this paper is a withdrawal notice** — the only
content version is v1, reason unstated.

So our #9 arm C was, strictly, a test of this paper's *straw man*
at its quoted strength — and the straw man cost us +2.64 MAE, the
same direction their own table points (masking is dominated by
calibrated noise). The honest bank update: the input-corruption
family is still live, but its literature-backed form is *noise
scaled to the deployment gap*, not heavy zero-masking, and its
flagship citation is withdrawn evidence. The queued p=0.3 screen
survives on our own branch rule, not on this paper's authority.

## 2. The state-free policy — amputation works, but only with two accomplices (2509.18644, v2)

The maximalist answer: drop state entirely. Confirmed as banked —
but the full text shows the vision-only headline rests on two
specific enablers, not on deletion alone: (a) **relative
end-effector actions** (displacements from the current pose), and
(b) "full task observation" — dual **wide-angle wrist cameras**
(~120°×120°). With both, real-robot spatial generalization goes
from ~0 to 0.98 (height) / 0.58 (horizontal) on Pick Pen, and the
pattern is architecture-agnostic (π0, ACT, Diffusion Policy all
flip from ~0 to strong). Without the action-space change,
*everything scores exactly zero* — relative-joint, absolute-joint,
absolute-EE all dead on generalization. Their rescue attempts for
state-based policies (noise aug 0.633, diverse data 0.117, LoRA 0)
all lose to state-free 0.983; an overhead camera actively *hurts*
extreme generalization. In-domain, state-based and state-free tie
(LIBERO 0.938 vs 0.945).

For us: the cleanest external statement that state removal is a
*system* choice, not a flag. Our arm C removed the crutch (their
diagnosis held) but had neither accomplice — our action space is
absolute joint-style chunks and our cameras are what the corpus
gives us — and their own ablation table predicts exactly what we
measured: zeros without the enablers.

## 3. ReViP — "false completion" and the case for modulating instead (2601.16667, v3)

The diagnostic paper. "False completion": a policy acts as if the
goal is achieved because its *internal progress estimate* (carried
by proprioception and habit) says so, ignoring visual evidence —
their motivating real-robot study shows a state-enabled π0 with 70%
success still false-completes on 46/50 perturbed trials; naive
state masking cuts false completion to 17/50 **but drops success to
40%** — the same modulate-don't-amputate lesson as our arm C, from
an independent lab. Their fix keeps state and rebalances: an
external VLM (Qwen2.5-VL-3B) extracts progress-aware visual cues,
which FiLM-modulate the vision-language prefix before action
generation. On their 8-task perturbation benchmark (objects dropped
mid-episode, distractors swapped, scenes relaid): π0 36% → ReViP
59% (+23; the banked "+26" is ReViP\* with a 72B VLM — a precision
fix to our note). Real-robot 62% → 88%. Cost: 44.6 → 62.4 ms per
step, and a second model in the loop.

For us: the mechanism story matches our probe result almost
line-for-line (better intact numbers riding on state reliance), and
it names the failure mode our offline panel can't see — false
completion is a *rollout* phenomenon. Banked as the heavier
modulation-side arm behind dropout in #9's original design; the
deep read keeps it there but upgrades its evidence class:
real-robot confirmed, ablated across VLM backbones.

## 4. GAP — the training-dynamics cause, with the receipts (2602.12032, v1, ICLR 2026)

The why. GAP runs temporally controlled interventions — splice the
vision-proprio policy's actions into a vision-only rollout for
10-step windows — and finds degradation concentrates in
**motion-transition phases** (locate/reorient moments: −7 to −14%),
nowhere else. Cause: proprio's concise signal gives faster early
loss reduction, winning the modality competition and suppressing
visual-branch gradients precisely where vision matters. The fix is
optimizer-side: segment the demo into motion-consistent phases
(change-point detection on proprio deltas, LSTM smoothing into a
transition probability ρ), then *shrink the proprio branch's
update* by λ(1−ρ) during the first half of training only. Verified
numbers: assembly 74.6 (concat) → 94.2, threading 33.2 → 53.0;
real dual-arm lift-lid-and-pour 5/20 → 15/20; on a fine-tuned Octo
it averages +17% over the vision-proprio baseline — though "works
on VLAs" rests on that one VLA. A fixed-probability masking
baseline (RDT-style) loses to GAP everywhere it's reported.

For us: still the named follow-on if input-side corruption
plateaus — now with the sharper framing that our arm C tested the
family GAP's own baseline column shows is dominated. GAP also made
a testable side prediction we banked: the grounding gap should
concentrate in motion-transition frames, readable for free from the
probe npz by conditioning Δ_first on progress-within-episode.

## 5. ThinkProprio — state as a lens, not a crutch (2602.06575, v2)

Our thin note ("proprio as text tokens at the prompt input") was
directionally right but missed the actual contribution. Yes, each
proprio scalar is binned to 256 levels and mapped into the
*pretrained vocabulary embedding table* (no learned projection,
Florence-2 backbone). But tokenized proprio alone buys ~nothing
(CALVIN 4.45 vs baseline 4.44). The headline mechanism is
**state-grounded visual token selection**: language and proprio
tokens form two guidance branches that select ~12% of visual
patches before the VLM runs. Both branches matter — language-only
selection scores 3.40, proprio-only 3.12, together 4.52. The
load-bearing ablation for the placement question we care about
(#11): MLP-encoded proprio injected at the VLM *hurts* (4.15 vs
4.44 with none) while vocab-token proprio doesn't — evidence that
*how* state enters is as important as whether. CALVIN ABC→D 4.52
(FLOWER 4.44), LIBERO 97.7% avg, real UR3 88.9% vs FLOWER's 80.7%
at 22 ms vs 52 ms per step.

For us: the most constructive paper of the set — state used to
direct visual attention rather than to predict actions. It inverts
the shortcut: proprioception decides *where to look*, vision
decides *what to do*. Banked into #11's conditioning-placement
discussion (our soft-state-token enters late; their table says late
+ learned-projection is the worst quadrant they measured).

## 6. Cloak — masking on the vision side (2606.22836, v1)

The other masking axis: hide the *end-effector* from the model,
visually, so the policy can't bind to embodiment appearance —
zero-shot cross-embodiment transfer follows. Mechanism (sharper
than our note): not inpainting — the EE mask is rasterized
geometrically from the robot model + state + camera parameters and
applied as a **ViT attention mask**, with training-time
augmentations (rolled-image fill, capsule attachment, disk removal)
so the silhouette can't be memorized; a generative model never
touches the observations. Trained on DROID (Franka), zero-shot to a
UMI gripper, a YAM arm, and a five-fingered Sharpa hand: task
*progression* rate (their metric — stage completion, not binary
success) 85.1/86.3/81.8 vs ~55–70 for π0.5-droid with tip-pose
retargeting. Source-embodiment performance is uncompromised (88.0
vs 89.3, within error). Limits they admit: two-fingertip skills
only, no rich contact.

For us: not actionable at the panel stage (we have no
cross-embodiment axis yet), but it's the standing answer to a
north-star question — when the owner rig arrives with a different
arm than the corpus embodiments, visual EE masking is the
zero-shot lever with real evidence, and it composes with anything
above (it's vision-side; the state-side debate is orthogonal).

## What transfers, what doesn't, and what it fed

**Transfers:** the diagnosis, fully — measured here twice (probe
D = +0.702; baseline collapse 3.94 → 24.08 under masking). And the
cross-paper consensus the retroactive read surfaced: **modulate,
don't amputate** — ReViP's masking study (success 70% → 40%), GAP's
dominated masking baseline, and Adapt Your Body's own tables all
point the same way, and our arm C is an independent fourth
datapoint. Amputation works only in the state-free paper's full
system, with relative EE actions and wide-FOV wrist cameras doing
half the work.

**Doesn't transfer:** the p=0.8 recipe as literature-endorsed
practice — that was a baseline in a withdrawn paper, mis-banked by
our skim as the method. Success-rate margins on scripted
perturbation suites don't map to our offline chunk-MAE panel
(false completion is invisible to it). ThinkProprio's selector
assumes a token-budget architecture we don't run today.

**Fed:** #9 — arm C's verdict re-framed (we falsified the family's
weakest member; calibrated noise à la NADA and GAP-style gradient
scaling are the surviving levers, p=0.3 screen still queued on our
own branch rule). #11 — ThinkProprio's placement ablation joins
the conditioning-placement evidence; GAP's motion-transition
prediction stays a free conditional read on the probe npz. North
star — Cloak banked as the cross-embodiment lever for the rig.
And one bank hygiene note: the ideas.md hook for 2506.23944 now
carries the withdrawn flag and the baseline-not-method correction.
