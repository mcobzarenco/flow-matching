# The attachment frontier — expert memory, leashed trunks, and world-action models

**Sources:** AR-VLA
([2603.10126](https://arxiv.org/abs/2603.10126), RSS 2026, read at
v2), Anchor-Align
([2607.13429](https://arxiv.org/abs/2607.13429) — "representation
anchoring" in our bank), and NVIDIA's
[world-action-models post](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
(the π0.7 source). Banked across the 2026-08-05/07 lit slices;
re-read at full depth for this page. **Fed:** #4 (two new priors on
the seam screen — one for the K premise, one third recipe), #17
(the history-aware-expert direction, now with its headline
ablation), #6 (a caution flag on the text-subgoal probe's ceiling).

## The theme

Our stage-2 question is *how to attach an action expert to a
pretrained trunk* — frozen (F) vs KI-joint (K). These three sources
each stretch that question along a different axis: AR-VLA gives the
**expert** its own memory across observations; Anchor-Align trains
the **trunk** but leashes every layer to its frozen self; the WAM
tier replaces the trunk's *pretraining* with video-dynamics models.
None changes the live screen; all three shape what the escalation
map looks like after it reads out.

## 1. AR-VLA — the expert that remembers (2603.10126)

**What it contributes.** Every VLA we run resets the action expert
at each chunk: fresh prefix, no cross-observation state. AR-VLA
builds the expert as a standalone AR transformer with a **hybrid KV
cache**: a rolling FIFO buffer of proprio/action KV pairs that
persists across observations (test-time length 20), plus a
single-slot vision-language buffer that is **replaced wholesale**
each time a new frame is processed. The two async streams stay
coherent through Dynamic Temporal Re-anchoring: action tokens take
sequential RoPE indices on the causal timeline, VL tokens take the
fixed index of their capture step, and RoPE's shift invariance
makes only the relative staleness matter. Actions are continuous
per-timestep regressions (not flow, not diffusion, not discrete
tokens), decoded one step at a time at 5 Hz.

**What they ran.** SimplerEnv WidowX: **61.5%** average vs CogACT
52.1, π0.5* 51.0, π0-FAST* 49.0 (starred = reproductions); 89%
zero-shot on a real WidowX from BridgeV2; specialist variant beats
ACT on ALOHA cube transfer (97.3 vs 86.0 scripted) though Diffusion
Policy keeps PushT. Lowest action jerk of all baselines and a
28.9 ms expert pass (vs 84.3 ms for a flow expert). The ablations
are the substance:

- **History length 1 → 20 is worth +25 points** (36.5 → 61.5;
  40 slightly worse). A length-1 context is structurally our
  chunk-reset design — this is the cleanest published number for
  what cross-observation expert memory buys.
- **History masking is load-bearing, not a trick:** train with no
  masking and success is **0.0%** (while validation error is the
  *lowest* — the model over-trusts its own history and collapses
  in closed loop); rate 0.6 is the peak at 61.5%.
- Static positional embeddings instead of DTR: 3.1% — the
  re-anchoring is what makes refreshable prefixes + persistent
  cache coherent.

**Caveats.** The flagship "memory" tasks (goals that become
unobservable mid-episode) are figure-only — no table values; the
quantitative memory evidence is the history-length ablation, which
is a context-length effect, not a beyond-occlusion proof. And the
authors name the failure mode themselves: an OOD action gets
written into the cache *as history* and drags the policy further
OOD — memory compounds errors as readily as it fixes them.

**What transfers.** Two things, one per axis. On the expert axis
this is the banked #17 history-aware direction with its headline
number attached — a named escalation if the attach screen leaves
headroom. On the trunk axis, a sharpening of our bank: AR-VLA is
**not** a third topology there — it freezes the VLM and blocks
action gradients explicitly, writing that AR gradients degrade the
trunk "similar to flow-matching experts," necessitating knowledge
insulation. An AR expert team independently converging on the K
premise is evidence for the K arm's motivation, from outside the
flow family.

## 2. Anchor-Align — train the trunk, leash it to its frozen self (2607.13429)

**What it contributes.** Between F (frozen trunk: nothing
forgotten, nothing adapted) and K (stop-grad joint training) there
is a third recipe: fine-tune the trunk *with a distillation leash*.
Anchor-Align adds two losses to standard BC. **Anchoring**: keep a
frozen copy of the pretrained VLM and penalize the trainee's
hidden-state drift from it (vision and text tokens, every decoder
layer, per-layer Frobenius norm). **Language-action alignment**:
label each chunk with a programmatic 6-way motion direction (up /
down / left / right / forward / backward), and train the trunk to
predict that word through its **frozen pretrained LM head** — so
the language space itself is forced to encode intended motion. Our
bank had only the first half; the second is co-equal (worth +4.9 /
+3.5 on their two suites alone) and drives their flashiest result —
after alignment training, a policy told to grab the pink mug grabs
it 100% of the time where the BC baseline grabs the
training-biased green one 90% of trials.

**What they ran.** LIBERO-PRO (semantic OOD): 71.9 mean vs 61.0
naive-BC, and — the row that matters for #4 — **43.8 for a
Co-training+KI baseline and 43.1 for full-freeze**: on their
benchmark, both of our screen's endpoints lose to the leash by ~28
points. On position-swap every baseline sits at 0.0–2.3% and
Anchor-Align is the only method off the floor (22.6%). Real xArm7:
28.3 → 54.2 and (on a flow-head architecture) 36.7 → 60.0.
Representation preservation is measured directly: naive BC loses
**94% of GQA VQA accuracy within 10k fine-tuning steps**;
anchoring retains ~70%. Shuffled-label controls collapse to
baseline, so the alignment gain is not generic regularization.

**Caveats.** Primary backbone is 0.5B — small; no layer-subset
ablation (is every-layer anchoring necessary?); λ values live in a
truncated appendix; the KI baseline is their reimplementation on
their benchmark, not π0.5's recipe at π0.5's scale. Grasp-and-drop
errors slightly *increase* (12 → 13) — more grasps survive to the
drop stage.

**What transfers.** A named third recipe for the seam screen's
escalation map: if K wins Δ_seam but shows the named cost
(VQA/semantic drift), anchoring is the published repair — and for
us the teacher is *free*, since the frozen copy is exactly our F
trunk. The cheap probe to steal regardless: measure VQA retention
on the Molmo2 trunk before/after stage-2 training — the 94%-in-10k
figure says drift is fast when it happens.

## 3. The world-action-model tier — π0.7 and the video prior (NVIDIA post)

**What it says.** The tier above VLA trunks: policies built on
pretrained *video* models, which already encode language→
visual-change dynamics. Two families — inverse dynamics (generate
future video, recover actions: LingBot-VA on Wan 2.2-5B, 16k hours
of robot pretraining, two-way video↔action conditioning) and joint
denoising (DreamZero: one DiT from Wan 2.1-14B denoises video and
actions together; **RoboArena 1750 vs π0.5's 1622** from DROID-only
action tuning). Being-H0.7 bridges latently (V-JEPA2.1-encoded
future observations as a posterior plan, dropped at test time).
And **π0.7**: high-level policy emits a subtask, a BAGEL world
model renders it as a **subgoal image**, the action expert
conditions on observation + subgoal — reported necessary for
dataset-bias-breaking tasks where no-subgoal variants fail, and
faster to train because action prediction becomes near-inverse-
dynamics. All banked systems and numbers verified (one date
touch-up: GR-1 is 2023/ICLR-24). The costs the post is honest
about: WAM inference runs 590–800 ms per chunk vs ~190 ms for
π0.5, and the full video-pretrain stacks sit at ~10× the compute
of a VLA fine-tune.

**The addition worth banking:** **Fast-WAM** — a representation-
only WAM that skips test-time video generation, runs 3–4× faster,
and reportedly matches LingBot-VA in sim *without* the 16k-hour
robot pretraining. That is the strongest single argument that the
**video prior, not test-time generation, carries the value** — and
it points at the reachable version for us: a video-capable trunk
(Molmo2 — already ours) conditioning the expert on predictive
features rather than rendered frames.

**What transfers — and a flag for #6.** The reachable-scale ladder
stands as banked: text-subgoal probe → subgoal-image conditioning →
video-prior features. But π0.7 is also a caution for the
self-subgoal probe's ceiling: they found text subtasks
*insufficient* for the bias-breaking tasks and needed rendered
images. If our rung-(a) text probe reads null, that is consistent
with the field's experience, not evidence the hierarchy thesis is
dead — the pre-registered read should say so in advance rather
than after.
