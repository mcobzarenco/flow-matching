# Hy-Embodied-0.5-VLA: what a full robot-learning stack looks like

*Read 2026-08-09 (standing lit slice, attach_F train window — arm F
of the #4 screen running on the box while this was read). Paper:
Hy-Embodied-0.5-VLA, [2606.14409](https://arxiv.org/abs/2606.14409),
June 2026.*

**The paper in plain words.** Most robot-learning papers show one
piece: a better model, or a better way to collect data, or a trick
for running faster on the robot. This one is a company showing the
whole assembly line — how they collect ten thousand hours of
demonstrations *without robots* (people wear instrumented grippers
and a motion-capture system records their hands), how they pre-train
a vision-language-action model on that, how they adapt it to real
robot arms with a few hundred teleoperated demos, how they then
*keep improving it past imitation* by letting an operator catch the
robot mid-mistake and rewind (the failure and the correction become
a preference pair), and how they run it smoothly at 50 Hz by
stitching each new action chunk onto the tail of the old one with a
spline. The end state is 94–99% success on four bimanual
manipulation tasks. No single component is unprecedented; the value
is seeing the pieces chosen, sized, and wired together, with numbers
at each seam.

## What it contributes

- **The stack, end to end**: 10K hours of egocentric human (UMI-style)
  data → continued pre-training of a 4B VLM+flow-expert → SFT on
  ~300 demos/task (18 h teleop) → preference-RL post-training
  (their "FlowPRO") → async 50 Hz deployment. Each stage published
  with its recipe.
- **FlowPRO / RPRO — RL on a flow policy without a reward model.**
  The flow-matching loss itself becomes an implicit reward:
  `r(s,a) = (β/2)(ℓ_ref(s,a) − ℓ_θ(s,a))` — the policy is "rewarded"
  for assigning lower flow loss to good actions than its frozen
  reference does. Preference pairs come from an
  **intervention-and-rollback** teleop loop: operator catches a
  failure, the system rewinds, records failure trajectory + corrective
  trajectory, and interpolation densifies them into per-state
  (s, a_win, a_lose) tuples. The loss = contrastive term + proximal
  anchor to the reference + an SFT term; when a_win = a_lose the
  contrastive gradient cancels exactly, so plain SFT data routes
  through the same loss safely.
- **Deployment mechanics for chunked policies**: an async
  producer-consumer loop (inference thread + servo thread) hides
  backbone latency, and a **latency-aware cubic Bézier stitch**
  discards the stale prefix of each new chunk and joins it
  C¹-continuously onto the executing one, using the executed
  trajectory's tangent and the new chunk's predicted tangent.
- **Architecture**: "Mixture-of-Transformers" backbone — vision and
  language keep *separate* QKV/FFN parameters and interact only
  through shared self-attention; a separate 370M flow expert
  (~11:1 backbone:expert); H=50 action chunks (10 Hz pretraining,
  50 Hz on-robot); a K=6-frame memory encoder that reuses the image
  encoder with factorized temporal attention (no new parameters).

## The experiments

- **RoboTwin 2.0 sim, 50 tasks**: 90.9% clean / 90.1% randomized —
  +25/+32 pts over π0, +8/+13 over π0.5, statistical tie with the
  best competing stack. UMI pre-training and the memory encoder each
  ablate to only ~2–3 pts — the headline gap over π-family is the
  *whole stack*, not one component.
- **Real robots** (Dobot bimanual, JAKA arm, Astribot humanoid):
  Track A adapts with 300 demos/task; Track B transfers to new
  embodiments from UMI data with **zero target-robot teleop**
  (SE(3) delta-chunks + IK make the action interface
  embodiment-agnostic).
- **FlowPRO, 3 rounds on 4 bimanual tasks**: 94–99% success —
  **+6 to +12 pts over DAgger** with the same interventions, +3–5
  over an advantage-conditioned regression baseline (their π0.6
  stand-in) — and *faster executions* (Bottle 16 s vs DAgger's 27 s:
  preference pairs penalize dithering, positive-only imitation
  cannot).

## What transfers to us, and what doesn't

- **The rig path (#16) — this is the most complete published
  blueprint for our north star.** Three pieces bank directly:
  (1) **FlowPRO's implicit-reward trick is decoder-agnostic for us**
  — it needs only our flow loss and a frozen reference copy; the
  label source (intervention + rollback preference pairs) is the
  same 5–20-interventions-per-task teleop currency as
  [[flowdagger-latent-dagger]]. The two are now the poles of the
  post-SFT menu: FlowDAgger fixes the *noise* and proves weight
  edits destroy held-out skills (SFT −0.94 retention); FlowPRO
  edits *weights* but anchors them proximally to the reference —
  and **never measures retention**, so FlowDAgger's critique
  stands unanswered against it. If we ever run this menu on the
  rig, retention on held-out tasks is the first read to demand.
  (2) The **Bézier chunk-stitch + async loop** is a deployment
  lever at *exactly our chunk length* (H=50, theirs; chunk_size 50,
  ours) — the latency-aware stale-prefix drop is the missing piece
  in our decode-cost story, which so far only measures per-chunk
  cost ([leaderboard microbench](../posts/2026-08-07-prereg-leaderboard-decode-microbench.md)),
  not chunk-boundary continuity.
  (3) The data lesson cuts both ways: robot-free human data works,
  but their collection rig (mocap cage, instrumented grippers,
  sub-mm tracking) is heavy infrastructure — the part of the recipe
  a single-rig owner cannot copy; 300 teleop demos/task for SFT is
  the copyable number.
- **The seam ledger (#4).** One more entry for the joint pole:
  everything trainable at continued pre-training (VLM + random-init
  expert, no stop-grad, no KI mention) — but from an
  *embodiment-pretrained* VLM, which is [[apt-expert-pretraining]]'s
  named condition for joint training being safe. Also a sizing data
  point: 370M expert on a 4B trunk (~11:1) vs our ~0.2B-class
  expert on 4B — same regime, corroborates the expert-scale prior.
  Nothing here re-ranks F vs K before our readout; the screen's
  measurement stands.
- **What doesn't transfer.** The MoT backbone would mean a new trunk
  (#17 keeps it on the survey list, nothing more); the memory
  encoder buys ~2 pts in *their* multi-step kitchen tasks and needs
  K=6 frame history our single-frame panel doesn't model; sim
  numbers vs π-family are data-mismatched (their 10K-hour corpus vs
  π's public weights) — the +25-over-π0 headline is a stack
  comparison, not a controlled ablation.

## Fed

- **#16 (rig-transfer):** FlowPRO banked as the weight-space pole of
  the post-SFT adaptation menu (vs FlowDAgger's frozen pole), with
  the retention-unmeasured caveat loud; Bézier chunk-stitching
  banked as the deployment-side lever for chunk-boundary continuity
  at our H=50.
- **#4 (stage-2 attachment):** joint-pole ledger entry (pretrained
  VLM + random expert, no insulation) under APT's interpretation;
  expert-size ratio corroboration.
- Cross-links: [[flowdagger-latent-dagger]] (the opposing
  adaptation philosophy, same intervention currency),
  [[apt-expert-pretraining]] (why their joint recipe is safe),
  [[attachment-frontier]] (expert sizing).
