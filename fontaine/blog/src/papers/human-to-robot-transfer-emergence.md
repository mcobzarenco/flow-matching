# H2R emergence: human video nearly doubles generalization — but only on top of diverse robot pretraining; a base VLM gets nothing

*Read 2026-08-09/10 (lit slice `lit-radar-0820`, priority 4). Paper:
[2512.22414](https://arxiv.org/abs/2512.22414) — "Emergence of Human
to Robot Transfer in Vision-Language-Action Models" (Simar Kareer,
Karl Pertsch, James Darpinian, Judy Hoffman, Danfei Xu, Sergey
Levine, Chelsea Finn, Suraj Nair; Physical Intelligence + Georgia
Tech; arXiv cs.RO/cs.AI, submitted 2025-12-27; license
nonexclusive-distrib/1.0; NO code or data release — project page
[pi.website/research/human_to_robot](https://www.pi.website/research/human_to_robot)
verified live (videos + hiring blurb only), arXiv abs and ar5iv HTML
verified live).*

**The paper in plain words.** Everyone wants robot policies to learn
from videos of people, because people are cheap and robots are not.
The usual obstacle is that a video of a hand is not a robot
trajectory — someone has to engineer a mapping. Physical
Intelligence's answer here is: stop engineering the mapping and make
the *pretrained model* absorb the difference. They strap a head
camera and two wrist cameras onto human data collectors, recover
6-DoF hand motion with SLAM and hand-keypoint tracking so the human
video looks like robot data with pseudo-actions, and then simply
co-train π0.5 on a 50-50 mix of human and robot data with the same
losses for both. The punchline is not the recipe — it is when the
recipe works. Starting from the bare vision-language model, human
data does nothing. Starting from a π0.5 pretrained on its full
diversity of robot scenes, tasks, and embodiments, 3-5 hours of
human video per task nearly doubles success on generalization
settings the robot never saw (spice-rack task 32% to 71% in unseen
homes). Transfer from people, they argue, is an emergent property of
diverse robot pretraining: diverse pretraining pushes human and
robot data into a shared, embodiment-agnostic representation, and
only then can gradients from human video move robot behavior. For
us the paper is mostly a gate: it is the first controlled evidence
about *when* the human-video lever pays, and by its own control
condition, a stack like ours — VLM trunk, no large-scale diverse
robot-action pretraining inside the trunk — sits in the corner
where the lever measurably pays nothing.

## What it contributes

- **A deliberately plain co-training recipe (π0.5+ego).** Human
  data is made robot-shaped: head-mounted high-res camera plus two
  time-synchronized wrist cameras; visual SLAM gives 6-DoF head
  motion; 17 3D keypoints per hand give a pseudo end-effector
  (palm + finger keypoints → 6-DoF relative EE transformations);
  dense language subtask annotations. Human action space is 18-dim
  (2×6+6) vs the robot's 16-dim; gripper open/close is learned from
  robot data only (no hand-openness estimate). Both embodiments get
  identical objectives — FAST discrete action tokens, flow-matching
  continuous actions, and high-level subtask prediction — with no
  alignment module, no domain adversary, no retargeting network.
- **The emergence result.** Fine-tune (robot + human) vs
  (robot-only) from checkpoints pretrained on 0% (base VLM
  initialization only), 25%, 50%, 75%, 100% of "the full diversity
  of scene-task combinations" in their target-embodiment corpus
  (ARX + mobile ARX), plus 100%+X-emb (the full π0.5 mixture with
  its many non-target embodiments). The human-data lift is ~zero at
  0-25% and large at 75-100%: "While with no or little pre-training,
  VLAs cannot benefit from human data co-training (0%, 25%), VLAs
  pre-trained on diverse data see significant gains (75%, 100%)."
- **A mechanism sketch.** t-SNE of mean-pooled final-layer VLM
  embeddings: human and robot data form disjoint clusters under
  weak pretraining and increasingly overlap as pretraining diversity
  grows — "embodiment agnostic representations emerge with
  pre-training scale." (Visualization only; no quantitative
  alignment metric, no linear probe.)
- **Human data ≈ another embodiment.** A 400-demo (7.45 h) UR5
  bussing corpus transfers to the ARX robots with similar magnitude
  to the human video — both beat robot-only, both lose to
  target-robot data. Human video slots into the cross-embodiment
  transfer picture rather than being a special modality.

## The experiments it actually ran

- **Four tasks, 14 h of purpose-collected human data total:**
  bussing (3 h), spice organization (3 h), dresser tidying (3 h),
  egg sorting (5 h). This is task-matched, pseudo-action-labeled,
  in-house collection with a bespoke rig — not internet video, not
  Ego4D-scale, not unlabeled.
- **Generalization present only in the human data:** scene transfer
  (spice, dresser in unseen apartments), object transfer (bussing
  with novel object categories), task/semantic transfer (egg
  sorting — a concept absent from robot data). Fine-tuning mixes
  human data for the generalization task 50-50 with robot data for
  the *nearest-neighbor* robot task. 20-40 evaluations per
  experiment, error bars 1 SE.
- **Headline numbers (robot-only → +human, on top of full
  pretraining):** spice 32% → 71%, dresser 25% → 50%, egg sorting
  57% → 78%, bussing 53 → 63 correct placements — the abstract's
  "nearly double" is the spice/dresser pair.
- **Human vs target-robot data (Fig 9):** on egg sorting and
  dresser, human data is nearly as effective as equivalent
  target-robot data; on bussing, robot data wins clearly (65 vs 25
  in the figure's gain comparison — figure-read, not a text table).
- **Diversity sweep detail lives in figures.** The per-level
  numbers (Fig 8/13) are plotted, not tabulated; the per-task
  scaling caption states "a clear upward trend in the efficacy of
  finetuning with human data as pretrained diversity increases."
- **Ablations:** wrist cameras help where manipulation is contact-
  rich (bussing, dresser), matter little for spice/eggs; both the
  high-level (subtask prediction) and low-level (action) transfer
  channels contribute, neither alone matches the combination.

## Corrections to our banked hook

Our hook — "co-training pays off only ABOVE a pretraining
scene/task/embodiment diversity threshold" — needs four edits.

1. **The swept axis is scene-task diversity of target-embodiment
   robot data.** Embodiment diversity is not swept; it is one extra
   endpoint (100%+X-emb = full π0.5 mixture). The clean claim is
   about scenes and tasks on the robots you will deploy on.
2. **"Threshold" is partly our compression.** The paper's own
   scaling caption describes a monotone "clear upward trend"; the
   text brackets a transition somewhere between 25% and 75% of
   *their* corpus. No absolute threshold is derived anywhere.
3. **There are no absolute units to place ourselves in.** The
   x-axis is fractions of an undisclosed corpus — the paper never
   states the hours, scene count, or task count of the pretraining
   mixture (that scale lives in the π0.5 paper, not this one). "Our
   229 h in their units" is unanswerable from this paper's text;
   only the qualitative corner mapping survives (below).
4. **Diversity is confounded with quantity.** The 25/50/75%
   subsets are fractions of scene-task combinations, and nothing we
   could extract holds total hours fixed while varying diversity —
   so "diversity threshold" could equally be a data-scale
   threshold. The paper's framing chooses diversity; the design
   does not isolate it.

One more scope correction that matters for angle A: the human data
here is **pseudo-action-labeled, task-matched, rig-collected
demonstration video** (SLAM head pose + hand keypoints + subtask
labels). This is evidence about co-training on human demos that
already look like robot data — it neither tests nor licenses
latent-action pretraining on unlabeled/internet video (the
CLAP/Motus mechanism), which is a different bet.

## What transfers to us and what doesn't

- **The 0% condition is the load-bearing control for us, and it is
  bad news.** 0% = base VLM initialization only — a fully
  video/image-pretrained VLM, like our Molmo2-4B trunk — and it
  gains ~nothing from human co-training. VLM-scale visual
  pretraining does not substitute for robot-action pretraining
  diversity on their axis. Whatever we inherit from Molmo2's video
  pretraining, it is not the thing this paper says unlocks
  human-to-robot transfer.
- **Our corner mapping:** their diversity axis lives *inside the
  trunk* — π0.5 pretrains the whole model on the robot mixture,
  and the mechanism story is that those weights hold aligned
  human/robot representations which co-training gradients exploit.
  Our 229 h corpus trains attachment heads (and at most low-LR text
  layers) on top of a trunk that has seen zero robot action data
  (Molmo2) or a 3.3M-sample embodied specialization (Molmo2-ER, the
  live er_60k arm). We sit at or near their measured no-transfer
  corner: 0%-diversity trunk, frozen, single target embodiment.
- **What would move us up their axis costs us nothing extra:**
  trunk swaps to embodied-pretrained variants (Molmo2-ER today;
  whatever AI2 ships next) are exactly "buy robot-pretraining
  diversity inside the trunk with someone else's compute." If this
  paper's mechanism is right, embodied trunk pretraining is the
  *precondition* for the human-video lever, which sequences our
  fronts: trunk-embodiment first (already live), human video later.
- **What does not transfer:** the collection rig. 14 h of
  head+wrist-cam, SLAM-tracked, hand-keypoint-annotated human data
  is a Physical Intelligence in-house pipeline; nothing is
  released. Reproducing pseudo-action human data for the owner's
  SO-101 would be its own engineering project — and the paper's own
  result says it would pay ~nothing on our current trunk.
- **Also honest:** their gains are measured with an unfrozen 3B-4B
  class model fine-tuned end-to-end at PI compute, tasks are
  household mobile/static manipulation on ARX arms, and evals are
  20-40 trials/experiment — real-robot small-n with 1 SE bars.

## What it fed

- **Idea #9 (data levers) — the human-video lever gets its gate.**
  The 08-09 trajectory-datasets survey flagged human video as one
  of the few unbounded data levers beyond our ~855 in-scope hub
  hours. Verdict from the first controlled evidence: **parked, not
  dead** — the one measured recipe pays off only on top of diverse
  robot-action pretraining inside the trunk, which we do not have,
  and pays ~zero from a bare VLM init, which is where we are.
  Reopening condition (pre-registerable): a trunk with substantial
  embodied pretraining (ER-class or better) in our stack, or
  external evidence of human-video gains at ≤~250 h single-
  embodiment robot data on a frozen/lightly-tuned trunk.
- **Idea #17 (new trunks) — the pretraining axis gains a second
  production datapoint.** MolmoAct2's Molmo2→Molmo2-ER +6.0
  LIBERO-Long said embodied trunk pretraining pays for action
  decoding; this paper says it is also what makes cross-embodiment
  and human data usable at all (and that VLM-benchmark inheritance
  ≈ their 0% condition — consistent with VLM4VLA's "benchmark
  cards don't predict VLA rank"). Strengthens the rationale of the
  live er_60k ER-trunk run beyond its own panel delta: ER-class
  trunks are the cheap way up this paper's x-axis.
- **Gate verdict on the angle-A spares (CLAP 2601.04061, Motus
  2512.13030, LingBot-VA 2.0): gated off for execution; no
  deep-read arm justified now.** This paper gates the co-training
  form of the lever directly. The latent-action form (CLAP/Motus)
  is mechanistically distinct and not refuted here — but its
  pretraining runs are far beyond our budget, our corpus headroom
  argument doesn't bite until the trunk precondition is met, and
  under startup velocity a deep read that cannot change a near-term
  launch is a spare, not a slice item. Cheap standing screen
  instead: skim any latent-action paper's abstract for gains
  claimed at low robot-data scale on a frozen VLM trunk — that
  specific claim, if credible, contradicts this paper's corner
  mapping and would justify promotion to a deep read.
