# The one-step menu — three objective families for killing the solver, and the one we didn't need

**Papers:** One-Step Flow Policy
([2603.12480](https://arxiv.org/abs/2603.12480)), MeanFlow-based
one-step VLA ([2603.01469](https://arxiv.org/abs/2603.01469)),
Let It Be Simple ([2606.05737](https://arxiv.org/abs/2606.05737)),
and GoldenStart ([2603.14245](https://arxiv.org/abs/2603.14245),
screened out). Banked 2026-08-06 as the fallback/follow-on menu
while our [SnapFlow](snapflow.md) distillation was still training;
re-read at full-text depth for this page. **Fed:** #12 — the
distillation leg. SnapFlow *hit*, so this menu never fired as a
fallback; it survives as the map of what else the one-step design
space contains — and the full-text re-read turned up one paper that
changes how we should think about our own win.

## The theme

A flow-matching action expert decodes by integrating an ODE — every
action chunk costs 10–60 network evaluations. The one-step
literature asks how to collapse that to one ("1-NFE"), and by 2026
there are at least three genuinely different answers on the table:
*distill it* (teach the model its own long jump — SnapFlow, OFP),
*change the objective* (learn an average-velocity field that never
needed the solver — MeanFlow), or *deny the problem exists* (argue
the task is so strongly conditioned that plain flow matching with
the right timestep schedule is already one-step-capable — Let It Be
Simple). We banked the menu while betting on the first family; the
interesting part in hindsight is how much the third family explains
about why the bet was safe.

## Where our result sits

Our SnapFlow replication ([page](snapflow.md),
[results](../posts/2026-08-06-snapflow-results.md)) made the menu
question concrete: the 1-NFE student scored **5.6036** chunk-MAE on
the full panel — beating not just its own Heun-30 teacher (6.6232)
but the AR anchor (5.8026) — at one expert eval, mean-of-10 at
5.3675. The fallback branch these papers were banked for never
opened. What the menu still owes us is context: why did
consistency-style self-distillation work first try, and what would
we reach for if we wanted more?

## 1. One-Step Flow Policy — the from-scratch cousin (2603.12480, v1)

OFP is the closest recipe to SnapFlow's, with the teacher removed:
a **from-scratch** self-distillation that never has a pretrained
flow model to start from. It learns an interval-averaged velocity
field and distills against an EMA copy of *itself*, with a
three-part loss: standard flow matching, a self-consistency term
whose sampling interval contracts over training, and a CFG-style
"self-guidance" term on the EMA teacher that sharpens one-step
predictions. The warm start our notes flagged is real but
inference-time only: the previous chunk's unexecuted suffix (last
action repeated) is noise-blended into the starting point,
exploiting temporal correlation between consecutive chunks to lower
the transport cost.

Numbers, verified: **71.6% ± 4.1 average at NFE=1 over 56
simulated manipulation tasks** (the 3D-pointcloud track: Adroit,
DexArt, 49 MetaWorld tasks) vs 66.4% for DP3 at NFE=100 — quoted
speedups 183× vs DP3. Integrated into π0.5 on four RoboTwin 2.0
tasks it averages 94.7% at one step and the paper claims this
exceeds the 10-step baseline — but the baseline number itself never
appears in the main text, so the margin is unverifiable; the
key ablation numbers likewise live in an appendix. Sim-only, by
their own admission.

For us: the reserve recipe if a future flow lineage has no
checkpoint worth distilling (SnapFlow needs one; OFP doesn't), and
its warm-start trick is a free idea for #1's noise structure — a
*structured* initial point instead of fresh Gaussian noise is
exactly the kind of thing our `sample_actions(noise=...)` hook can
test offline.

## 2. MeanFlow one-step VLA — a different objective, with a catch (2603.01469, v1)

The one entry that isn't distillation at all: replace the
instantaneous velocity field with a MeanFlow **average-velocity**
objective, so the network directly models the noise-to-data
mapping — "without pre-training, distillation, or additional
consistency heuristics," eliminating exactly the consistency
constraint whose drift we watched during the SnapFlow run. Built on
a SmolVLM-2 backbone, evaluated **real-robot only**: an SO-101 arm,
three tasks, 100 demos each.

The full-text read corrected our banked hook. The 8.7× speedup vs
SmolVLA is confirmed — but it is bought with accuracy: **78% average
vs SmolVLA's 84.5%**, losing on two of three tasks (stacking 64% vs
81.5%, which they attribute to precision demands). And the paper's
own NFE sweep shows 49% at NFE=1 in one configuration while the
chunk-size ablation shows 84.25% in another, an unreconciled
config-sensitivity the abstract doesn't mention. Our one-line note
("8.7× vs SmolVLA") was the marketing read; the honest read is
*faster and worse*.

For us: still the right shape for a paired follow-up **if**
consistency-style distillation ever misses — a genuinely different
objective family, not a SnapFlow re-tune. But our student got its
speedup with a panel *win*, not a panel loss, so the bar this paper
sets is one we already clear.

## 3. Let It Be Simple — the paper that reframes the whole menu (2606.05737, v2)

The provocative one, and the deep read paid. Claim: VLA generation
is **image-to-text-like, not text-to-image-like** — the condition
(images, language, state) is rich and the target (an action chunk)
is compact, so the "irreducible velocity loss" — the uncertainty
about the target that remains *after seeing the condition* — is
small, and one-step decoding needs no distillation at all. Their
entire mechanism is a timestep schedule biased toward high noise
(t near 0): with α=4, plain flow matching hits **95.6% on
LIBERO-Long at one step** vs 70.2% for the uniform schedule; +5.4
mean success points over ten-step across 18 LIBERO-Plus recipes;
real-robot spot checks (bimanual, 5 trials/task) where one-step
*beats* ten-step (Tower of Hanoi 100% vs 50%).

Two caveats the abstract hides, both load-bearing. First, the
schedule is a specialization, not a free win: the same α=4 model
scores **63.4% at ten steps** — high-noise training trades away the
multi-step regime. Second, their condition-weakening ablation pins
the effect chiefly on **state**: remove proprioception and one-step
success collapses to ~0% across every LIBERO suite, while removing
the image costs far less (and they admit residual no-image
performance may reflect static-layout dataset bias). "Strong
conditioning" in their result is substantially *state*
conditioning — the very shortcut our
[state-probe work](state-shortcut.md) measured. Their horizon
ablation also shows the one-step advantage shrinking as chunk
length grows, consistent with their own theory (bigger target, more
residual uncertainty).

For us this paper does two things. It predicts our SnapFlow result:
if the conditional action distribution is nearly deterministic
given the context, a one-step map is learnable — by distillation or
by schedule — and our student's *draw-spread collapse* (the
distillation compiled the mean and discarded the distribution) is
exactly what "small irreducible velocity loss" looks like from the
inside. And it leaves a zero-training probe on the table that our
notes already flagged: score the **teacher** at 1-NFE. If the
teacher's one-step number is already decent, most of the "one-step
gap" was schedule/objective slack, not something distillation had
to build; the banked instrument (`--target-time zero`) can read
this from existing checkpoints whenever the question becomes
decision-relevant.

## 4. GoldenStart — screened out, and the screen-out holds (2603.14245, v1)

One-step distillation of a flow policy with a **Q-guided prior** (a
conditional VAE trained to emit initial noise whose decoded action
scores highest under a learned critic) plus entropy-regularized
distillation. The full-text check sharpens our screen-out reason:
rollouts are only needed for its offline-to-online phase (a purely
offline mode exists: OGBench 47.1 avg vs FQL's 38.5), but **the
critic is always required** — and a critic needs reward-labeled
data, which our demonstration corpus doesn't have. The binding
constraint is Q-functions, not rollouts; the verdict (not our
setting) stands. Sim-only. Worth one sentence of memory anyway: its
"advantage noise selection" is a third data point — with Golden
Ticket and OFP's warm start — that the field increasingly treats
**initial noise as a controllable input**, which is #1's whole
program.

## What transfers, what doesn't, and what it fed

**Transfers:** the design map itself. Family (1) is banked and
replicated (SnapFlow, and OFP as the teacher-free reserve); family
(2) is a real alternative objective but currently posts
speed-for-accuracy trades we don't need to make; family (3) costs
nothing but a sampling schedule and gives us both a mechanism story
for our own win and a free diagnostic (teacher-at-1-NFE). The
noise-as-input thread (OFP warm start, GoldenStart's learned
priors) feeds #1's Golden-Ticket program directly.

**Doesn't transfer:** every success-rate margin here is
LIBERO/MetaWorld/small-real-robot; none of these papers scores a
distribution the way our panel does, so claims like "matches
ten-step" hide exactly the effect our replication surfaced
(mean-compilation, draw collapse). MeanFlow-VLA's headline speedup
is not a model for what "adopt" should mean here. Let It Be
Simple's state-carried conditioning is a warning label, not a
recipe — on our stack the state shortcut is measured and its
naive removal measured to cost (+2.64 MAE at p=0.8 dropout).

**Fed:** #12 — SnapFlow stays the adopted leg; OFP is the named
reserve; a MeanFlow arm remains the "different objective" branch if
consistency distillation ever misses; teacher-at-1-NFE banked as
the cheap schedule-vs-distillation decomposition read. #1 — two
more entries in the structured-noise column. #11/#9 — one more
external datapoint that state is the dominant condition channel in
this model class.
