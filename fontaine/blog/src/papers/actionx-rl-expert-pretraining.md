# ActionX — pre-train the expert first, then unfreeze everything

*Frontiers in Neurorobotics, 2026 —
[10.3389/fnbot.2026.1806605](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full).
Read 2026-08-09 (work-session lit slice, the day the attach screen's
Δ_seam readout lands). Venue caveat up front: Frontiers, not a
first-tier robotics venue — we weight the ablation SHAPE, not the
absolute numbers.*

**Plain words.** When you bolt an action module onto a big
vision-language model, you have to decide what learns when. This
paper's recipe: first train ONLY the small action module (the big
model stays locked), then — once the action module is competent —
unlock everything and train the whole stack together. Their twist is
doing that first stage with reinforcement learning (trial and
success/failure) instead of pure imitation. The striking part isn't
the RL: it's the ablation showing that skipping the first stage
entirely makes the joint stage bad or useless, while ANY competent
pre-training of the action module first makes the joint stage work.
That "warm-start the expert, then go joint" pattern is exactly the
escalation rung our attach screen has queued behind tonight's
frozen-vs-joint readout.

## What they built

- **Stack**: InternVL3-1B trunk (InternViT-300M vision), an 8-layer
  self-attention flow-matching action expert with sinusoidal time
  encoding. Conditioning = trunk hidden state concatenated with a
  separately-embedded robot state (state kept OUT of the multimodal
  fusion — an independent MLP path).
- **Stage 1 — expert pre-training, trunk FROZEN**: PPO on the expert
  only, sparse 0/1 success reward, critic warmed on 10% of the demo
  data; ~5.7k–14k RL steps depending on task family.
- **Stage 2 — joint SFT, everything UNFROZEN**: full-parameter
  fine-tuning on demonstration velocity fields, no stop-gradient, no
  insulation. They report faster convergence *because* the expert
  arrives pre-initialized.

## Experiments

LIBERO (4 suites × 10 tasks): 91.5% average; Meta-World (50 tasks):
81.3% (+13 over their prior-work line); real UR12E single-arm + a
dual-arm platform (+16% average over baselines there). Baselines:
Diffusion Policy, OpenVLA, π0, π0-FAST, SmolVLA, TinyVLA. The gains
concentrate on LIBERO-Long (+6 vs SmolVLA); Spatial/Object/Goal are
modest.

**The ablation that matters to us** (their Table 2, expert
pre-training regime → suite success):

| Stage-1 regime | LIBERO-Spatial | LIBERO-Long |
| --- | --- | --- |
| none (joint SFT from scratch) | 0% | 14% |
| supervised expert pre-train | 92% | 52% |
| RL expert pre-train | 95% | 66% |

(The 0% row reads suspiciously absolute — likely a matched-budget
snapshot rather than convergence; we treat the ORDERING as the
finding, not the magnitudes.) RL adds +14 on Long over supervised
pre-training, but the cliff is none→any: +38 to +92 points.

## What transfers to us

- **The F-then-joint rung gets published support in its exact
  shape.** Our banked
  [F-then-joint escalation](apt-expert-pretraining.md) (warm-start a
  joint run from the F arm's converged expert) is literally their
  supervised row: frozen-trunk expert SFT first, then full unfreeze.
  Their supervised-pre-train→joint beats joint-from-scratch by +38
  (Long) — independent corroboration, on a pretrained-VLM trunk, for
  APT's grid. Ledger entry added to
  [idea #4](../ideas/04-stage2-attachment.md).
- **No-insulation joint stage**: their stage 2 unfreezes everything
  with no stop-grad — consistent with APT's finding that WITH a
  competent expert, plain joint training is the best published
  recipe. If tonight's Δ_seam readout says F≈K, this is one more
  vote that the next discriminating contrast is initialization, not
  seam plumbing.
- **State kept out of the fusion** (separate MLP path, concatenated
  late) — same family as our prompt-side state token; nothing to
  change, mild corroboration.

## What does NOT transfer

- **The RL half needs rollouts.** Sparse-success PPO presumes a
  resettable simulator or cheap real resets; our panel is offline
  MAE against teleop data. Their RL-vs-supervised margin (+14 Long)
  is real but unreachable for us today; the reachable part is the
  supervised row — which is the F arm we already trained.
- **No frozen-vs-joint readout at matched conditions.** They never
  ablate freezing the trunk in stage 2, so this paper does NOT
  re-rank F-vs-K — it informs only the escalation rung behind the
  readout.
- Venue + benchmark caveats: Frontiers; LIBERO/Meta-World sim suites
  with saturation effects; success-rate metrics, not our
  paired-per-frame MAE reads.

## Fed

- **#4** (attachment): F-then-joint pre-reg draft
  (`idea4-f-then-joint-prereg-draft`, opens at the Δ_seam readout)
  gains its second independent citation for warm-start-then-joint;
  APT's condition (pretrained expert ⇒ no stop-grad joint wins) now
  has a same-shape replication on a 1B-class trunk.
- **#16** (post-SFT menu): the RL-on-expert-only pole (their stage
  1) is a cheaper cousin of FlowPRO — expert-scoped, trunk-frozen —
  banked as a note on the menu's weight-space side; retention still
  unmeasured in both.
