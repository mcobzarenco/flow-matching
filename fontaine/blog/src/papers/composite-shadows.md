# The composite's missing shadow: what the paste-the-robot papers do about light

**Papers:** ConCent ([2606.30268](https://arxiv.org/abs/2606.30268))
· ReBot ([2503.14526](https://arxiv.org/abs/2503.14526)) · Re³Sim
([2502.08645](https://arxiv.org/abs/2502.08645)) · GreenAug
([2407.07868](https://arxiv.org/abs/2407.07868)).
**Read:** 2026-08-12, sim lit lane (`lit-sim-improvement-levers`,
owner-called 09:23Z). **Fed:** the contact-shadow axis for the v3
composites (new idea, below) and the queued `sim-wrist-compositing`
design; sits alongside [sim-as-eval](sim-as-eval.md) (SIMPLER's
visual matching) and
[sim-contact-fidelity](sim-contact-fidelity.md) (the physics side).

**The problem in plain words.** Our evaluation pictures are collages:
a real photograph of the empty bench (the "clean plate") with a
computer-rendered robot arm pasted on top. The collage trick is the
single best thing we did for visual realism — the background is
*exactly* real because it is a photograph. But a pasted arm breaks
one law of physics that every real frame obeys: it casts no shadow.
A real arm hovering over the table darkens the wood under it; our
rendered arm floats above an eternally sunlit plank. This slice read
the four papers closest to our collage pipeline to see what they do
about light and shadow. The finding is odd and useful: **almost
nobody handles shadows, nobody measures them, and the one paper that
does anything treats shadows as noise to randomize rather than
signal to match.** Meanwhile the one careful ablation in the set
says the *rendered foreground's* realism barely matters at all —
which both vindicates our collage strategy and tells us where not to
spend effort.

## ConCent: the only explicit shadow recipe

ConCent (2606.30268) is a real-to-sim-to-real system that learns a
precision insertion task (a 40 mm block into a 42 mm hole) from one
demonstration, by extracting the *contact event sequence* from the
demo and training RL in sim against it. Its rendering stack for the
distillation dataset is a collage like ours, but fancier: 3D
Gaussian splats for objects, a flow-matching generative model for
robot and background. And that stack has our exact defect — "object
shadows cannot be captured" because objects render separately from
the scene.

Their fix is the one concrete shadow recipe in this slice: **sample
a virtual light source at random, compute each object's shadow
projection onto the ground plane, and draw it into the composite** —
applied both when generating training data and at inference. Note
what this is: not shadow *matching* (estimating the real room's
light and casting the shadow it would cast) but shadow
*randomization* — teach the policy that a dark blob under an object
is uninformative. They report 80% task success (16/20) but no
ablation isolating the shadow layer, so its individual contribution
is unmeasured.

The transferable geometry is trivially cheap: a shadow is a
projection of the object silhouette from a point (or direction) onto
a plane, darkened and blurred. For a MuJoCo scene we hold every mesh
and pose, so a ConCent-style shadow pass over the clean plate is
pure arithmetic — no renderer changes.

## ReBot: ships the collage with no shadow handling at all

ReBot (2503.14526) is the closest published pipeline to our v3
composites, run in the opposite direction — it manufactures
*training* videos rather than eval frames. Real robot trajectories
are replayed in sim over diversified objects, then the sim arm and
object are merged onto real backgrounds. The background prep is
GroundedSAM2 to segment the real robot and object, ProPainter to
inpaint them away — the same clean-plate-by-inpainting move as our
`make_clean_plates.py`.

On light and shadow the paper is silent: no relighting, no color
harmonization, no shadow synthesis — extract pixels, merge, done.
And the numbers still move: fine-tuning on ReBot videos lifts Octo
+7.2 / +19.9 points (in/out-of-domain) and OpenVLA +21.8 / +9.4 on
SimplerEnv WidowX tasks, and +17 / +20 points on a real Franka. So a
shadow-free, harmonization-free collage is demonstrably good enough
to *train* against at these competence levels. What ReBot cannot
tell us — because they never measured it — is whether the missing
shadow costs anything when the collage is the *eval* and the
question is whether a real-trained policy behaves identically on it.

## Re³Sim: the ablation that says foreground realism is not the lever

Re³Sim (2502.08645) builds the whole background as a 3D Gaussian
splat reconstruction and composites *rendered foreground objects*
into it by ground-truth-depth Z-buffering — our collage inverted
(their background is the reconstruction, ours is a photograph; both
paste a rendered arm). No shadow projection onto the background
here either, and the authors list lighting estimation as an open
limitation.

The valuable part is Table VI, the only controlled visual-fidelity
ablation in this slice: swap the foreground objects' *mesh*
rendering for photoreal 3D Gaussian splats and task success does not
move — 0.70→0.70 and 0.75→0.75 on their two tasks. Zero-shot
sim-to-real still exceeds 58% average from ~10 minutes of sim data.
**The rendered foreground's photorealism was not the binding
constraint; the scene around it was.** That is exactly the bet our
composites make (spend everything on a perfectly real background,
render the arm plainly), and it is a warning against gold-plating
the arm render — the marginal axis worth probing is the arm's
*interaction* with the real background (shadow, occlusion edges),
not its surface appearance.

## GreenAug: the counterpoint — for training, stop matching and randomize

GreenAug (2407.07868) collects demos against a literal green screen
and chroma-keys backgrounds in: random textures (GreenAug-Rand),
generative scenes (-Gen), or masked-out black (-Mask). The striking
result is that **random textures beat the realistic generative
backgrounds** for generalization to novel scenes — the crude
augmentation wins.

Read next to SIMPLER (see [sim-as-eval](sim-as-eval.md)), this
completes a clean division of labor: when the collage is *training
data*, background realism is optional and randomization is the point
(GreenAug, ReBot); when the collage is an *evaluation mirror*,
matching is everything and partial matching is worse than none
(SIMPLER Table III — the constraint that already governs our wrist
decision). Shadows inherit the same split: a training pipeline
should randomize them (as ConCent does); an eval mirror should cast
the *one correct* shadow or provably show the encoder does not care.

## What transfers, what doesn't

**Transfers — a probe-gated contact-shadow pass for the v3
composites (new idea).** The v1 study's scene pass was our biggest
single visual win (top-cam 5-NN AUROC 0.835→0.786 on content alone),
and the remaining top-cam gap plausibly includes the shadow channel:
every real frame darkens the table under the arm; no composite frame
does. The ConCent recipe adapted to matching rather than
randomization: estimate the room's dominant light once from the
clean plates themselves (the direction of shadows already baked into
the *static* scene — bench legs, boxes — is visible in the plate),
project the arm+boat silhouette onto the table plane, multiply-darken
with a soft edge, one strength parameter. Gate it exactly like every
other render-style change: reset-render probe, top-cam 5-NN AUROC
must drop from 0.773, wrist unaffected (its composite path is
separate). If the encoder doesn't care, the axis dies for ~0.02
GPU-h — cheap either way. Fed to `ideas.md` as the
`composite-contact-shadows` hook.

**Transfers — the eval/training split as a standing design rule.**
Randomize shadows in anything we ever *train* on composites
(GRPO-in-sim included, if the live probe opens that door); match or
omit-and-measure shadows in anything that *scores* a policy.

**Doesn't transfer.** ConCent's generative robot rendering and
contact-sequence RL (single rigid insertion, one demo — different
problem); ReBot's trajectory-replay data engine (we have a
task-capable sim, not a data famine on this axis); Re³Sim's full-GS
background reconstruction — our photograph *is* the ground truth
their reconstruction approximates. And note none of these papers
measure the shadow axis; our AUROC probe can, which makes the cheap
experiment above worth its half hour.
