# The lens is a fittable parameter: fisheye choices, scale overfitting, and rendering through the real distortion

**Paper:** Rethinking Camera Choice: An Empirical Study on Fisheye
Camera Properties in Robotic Manipulation
([2603.02139](https://arxiv.org/abs/2603.02139),
[site](https://robo-fisheye.github.io/)).
**Read:** 2026-08-12, sim lit lane (`lit-sim-improvement-levers`,
owner-called 09:23Z). **Fed:** the camera-parameter-fitting lever —
two concrete upgrades to the v1 wrist render path
(`sim/so101_sim.py::_init_fisheye`) and a training-side augmentation
hook; context for the wrist-periphery story in the
[v1 render-style record](../posts/2026-08-12-prereg-sim-wrist-periphery.md).

**The problem in plain words.** Our robot's cameras are cheap
130-degree wide-angle modules — straight table planks bow visibly in
every real photo. The simulator fakes this: it renders a normal
(pinhole) image with a 72-degree view and then warps it outward with
a textbook "equidistant fisheye" formula, tuned so the center of the
image matches and the edges bend roughly like the real lens. That
warp was a guess with two known sins: the *textbook formula* is not
the real lens's actual curve, and the *72-degree source* physically
cannot contain things the real 130-degree lens sees at the edges of
its frame. This paper is the first careful study we've found of
exactly these choices — which lens properties help manipulation,
what happens when the training lens and the deployment lens differ,
and how to render arbitrary lenses honestly in our exact simulator
(MuJoCo). Its sharpest finding: policies secretly use *how many
pixels wide an object is* to judge distance, so getting the lens
model wrong shifts every distance the policy perceives — a bug our
appearance-based image probe would never see.

## What the study did

Wrist-camera manipulation only (deliberately no third-person cams,
"consistent with prior UMI-like works" — note our top cam is outside
their scope). A Flexiv Rizon 4 arm, six tasks (Pick Cup, Fold Towel,
Hang Chinese Knot, ...), pinhole vs fisheye across simulation (90°
vs 235° FoV) and real hardware (60° vs 180°), plus a set of
different real fisheye lenses up to 220° for transfer tests.

The simulation infrastructure is the directly reusable piece: a
**two-stage projection pipeline in MuJoCo**. Stage 1 renders six
pinhole cameras along the cardinal directions into a cubemap and
stitches an equirectangular panorama; stage 2 resamples the panorama
through *any parametric lens model* into the final fisheye frame
(implementation via OmniCV-Lib). Because the source is a full
panorama, any FoV and any distortion profile is renderable exactly —
lens parameters become simulation knobs.

## The findings

**Fisheye wins at the wrist, when there is texture to see.** On real
Pick Cup across 8 scenes, fisheye scores 0.988 (normalized) vs
pinhole's 0.181. Wide FoV improves spatial localization (1.73 cm
translation error on their proprioception-prediction probe) — but
the gain is contingent on *feature-rich* environments (+0.39 in
rich vs +0.18 in plain scenes): the extra periphery only helps if it
contains usable texture. With 8+ diverse training scenes, fisheye
policies exceed 95% real-world success on their suite.

**Scale overfitting is the transfer killer.** Policies trained on
one lens "overfit to the absolute pixel scale of objects to
determine distance." Swap the lens at deployment and performance
collapses — a baseline policy moved to a 220° lens scores 0.0025.
The distortion profile *is* part of the policy's metric ruler.

**Random Scale Augmentation (RSA) is the antidote.** Per training
image, sample a scale factor s ~ U(0.7, 1.3) and zoom in/out (pad
with black when zooming out). This forces distance estimation onto
relative spatial relationships rather than absolute pixel size:
the 220° transfer score rises 0.0025 → 0.60, and on a deliberate
scale-mismatch stress test (s = 1.30) RSA holds 1.000 vs the
baseline's 0.650.

## What transfers, what doesn't

**Transfers — replace the 72° warp with the cubemap pipeline (fixes
a real limitation we already hit).** Our `_apply_fisheye` warps a
single 72°-fovy pinhole source, so output pixels can only show rays
the source contains (diagonal ~50°); the real modules see ~130°
before the 4:3 crop. The wrist-periphery re-tune (AUROC 0.900→0.548)
worked by *re-aiming the camera* so the frame stops needing rays we
cannot render — a workaround for exactly the constraint the cubemap
approach removes. Six renders per camera per frame instead of one
(eval-render cost only, not physics), and any lens model becomes
available, which enables the second transfer:

**Transfers — fit the real lens instead of assuming ideal
equidistant.** One checkerboard session with the actual rig modules
(standard Kannala–Brandt or Scaramuzza polynomial fit; both MATLAB
and OpenCV tooling are routine) gives the true θ→r curve, and the
stage-2 resampler renders through it. The pre-reg for any such
change is already implicit in the scale-overfitting finding: the
metric that must move is not only appearance (5-NN AUROC) but
*perceived scale* — a wrong distortion curve rescales objects at
every off-center radius, and a policy that keys on pixel scale reads
that as distance error. Cheap validity check without new hardware
captures: compare bowing curvature of the table-plank edges,
sim-rendered vs the 150 pinned real reference frames, as a direct
θ→r residual readout. Fed to `ideas.md` as the
`fit-real-lens-model` hook.

**Transfers — RSA, two ways.** (a) Training-side: if we ever
fine-tune or GRPO-train on sim or rig frames, RSA is a
one-transform, zero-risk augmentation with a measured 240× transfer
delta behind it in the wrist-cam regime. (b) Eval-side sensitivity
knob: rendering eval frames at a few fixed scale offsets (their
s = 1.30 stress test) measures how much of a checkpoint's sim score
rides on absolute pixel scale — if the score is flat, our lens-model
guess is not load-bearing for that policy; if it swings, lens
fitting is urgent, not cosmetic.

**Doesn't transfer.** Their camera *choice* question is settled
hardware for us (the rig has the modules it has); the pinhole-vs-
fisheye comparisons and the feature-rich-scene recommendation
describe data collection we are not redoing. Their setup is
wrist-only — our top cam is a fixed third-person view outside the
study's scope, and its composite path (real clean plate) already
sidesteps most lens error at the periphery since *the photograph is
taken through the real lens*; the rendered-arm overlay is the only
part that carries our synthetic distortion. The wrist, fully
rendered, carries it across the whole frame — which is why both
upgrades above aim there first.
