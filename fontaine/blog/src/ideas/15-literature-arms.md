# 15. Literature-sourced arms — standing

*Tag: `lit-arms` · idea #15 · [index](../ideas.md)*

The arXiv radar (VLA/robot learning, flow matching, action
tokenization, data curation) feeds this list; every borrowed idea
cites its source in the pre-registration; every "novel" idea gets a
search first. Local canon: π0, π0.5, SmolVLA, FAST
(arXiv:2501.09747).

- **π0.5 canon deep-read DONE (2026-08-07,
  [post](../posts/2026-08-07-pi05-deep-read.md))** — π0.5
  (arXiv:2504.16054) + KI (arXiv:2505.23705), read against the live
  stage-2/Molmo2 question. Findings banked into #4 (two named
  attachment arms), #5 (FAST-vs-naive ablation), #6 (Implicit-HL
  replication + self-subgoal rung-(a) probe), #16 (Fig. 8 external
  anchor). Convention flag: π0.5's τ=1 is DATA; ours is NOISE.

- **IVRA (arXiv:2601.16207, lit slice 2026-08-06 16:2xZ)** —
  training-free, inference-side: VLAs flatten patches to 1D and lose
  2D spatial cues; IVRA injects vision-encoder patch-affinity signals
  into ONE LM layer ("where instance-level features reside"), no
  retraining, +4.2% on VIMA low-data / consistent LIBERO gains across
  LLaRA/OpenVLA/FLOWER. Fits #11's diagnosis exactly (acuity probe:
  position info sharpest at tower output, degraded through LM layers).
  Our analogue: bias trunk attention over soft tokens with
  tower-output affinities at eval — rung (a), zero training, panel
  first_mae is the readout. Cheapest falsification: single-layer
  injection on the flow teacher, panel-v2 first_mae vs banked
  2.0720 ctrl. Worth a probe if arm A's img280 read leaves grounding
  headroom on the table (interacts: more tokens vs better-used
  tokens are the same front, opposite ends). **Papers-page re-read
  2026-08-07 ([page](../papers/grounding-conditioning.md)):
  mechanism is token-feature mixing (affinity-weighted pooling +
  convex blend λ≈0.2–0.3), not attention editing; best at layer
  20/32 (~62% depth), LM-input injection catastrophic; banked
  "LIBERO gains across LLaRA/OpenVLA/FLOWER" corrected — LLaRA is
  never on LIBERO (its results are VIMA low-data +4.2 and a small
  real study).**

- **Lit slice 2026-08-07 ~07:5xZ (session slice, two banked):**
  (a) **TapSampling (arXiv:2605.25547)** — a FOURTH selection-rung
  flavor for #19: inference-time sampling with a TASK-PROGRESS
  verifier (learned progress-understanding score filters sampled
  action candidates per step). Joins MG-Select (verifier-free KL) /
  VLA-ATTC (trained pairwise critic) / CoVer (contrastive
  instruction-alignment verifier) in the banked flavor list — all
  wait behind the oracle best-of-10 ceiling read
  (`idea19-selection-ceiling-read-script`): if the ceiling is small,
  every one of these is dead on our panel; no selector gets built
  before that number exists. (b) **AR-VLA (arXiv:2603.10126, RSS
  2026)** — a standalone AR action EXPERT generating actions as a
  continuous causal sequence over refreshable vision-language
  prefixes, with its own persistent history — a third attachment
  topology beside our F (frozen-trunk flow expert) and K (KI-joint):
  the expert keeps cross-observation memory instead of resetting per
  chunk. Not actionable while the #4 seam screen is the live
  question (its F/K verdict comes first); banked to #17 as the
  history-aware-expert direction if the attach screen leaves
  headroom. Also seen: "representation anchoring" (arXiv:2607.13429,
  frozen-copy distillation to keep OOD generalization during
  finetuning) — context for the K named-cost branch's repair space,
  recorded here only; AEGIS stays the single pre-registered
  escalation (frozen decision rule untouched). **Papers-page
  re-read 2026-08-07 ([page](../papers/attachment-frontier.md)) —
  both sharpened:** AR-VLA is NOT a third *trunk* topology — it
  freezes the VLM + stop-grads explicitly ("AR gradients degrade
  the VLM like flow gradients"), independent outside-the-flow-family
  support for the K premise; its memory number is history length
  1→20 = +25 pts (36.5→61.5), with the tax that no-masking training
  collapses to 0% and OOD actions feed back through the cache.
  Anchor-Align (2607.13429) is half-banked — a co-equal
  language-action alignment loss (6-way direction words through the
  frozen LM head) drives its pink-mug result; on its benchmark the
  leash beats BOTH a Co-training+KI baseline (71.9 vs 43.8) and
  full-freeze (43.1); cheap probe to steal: VQA-retention on the
  Molmo2 trunk before/after stage-2 (naive BC loses 94% GQA in 10k
  steps, anchoring keeps ~70%).

- **Lit slice 2026-08-07 ~08:2xZ (session slice, two banked):**
  (a) **Look Before You Leap (arXiv:2607.03751)** — a FIFTH
  selection-rung flavor for #19: MCTS explores a FROZEN VLA's output
  distribution offline, then distills the search into a Q-value
  action evaluator used at test time (trained-critic family beside
  VLA-ATTC, but the critic's labels come from tree search over the
  policy's own distribution, no human/reward labels). Same gate as
  the other four: waits behind the best-of-10 ceiling number — the
  script for that is now landed, so the flavor list has its
  adjudicator ready at the ~08-08 endpoint dump. (b) **DVAC
  (arXiv:2606.03847)** — training-free, orthogonal to selection:
  variance of the clean-action estimate over the FINAL DENOISING
  STEPS of a flow/diffusion policy decides when to replan (execute
  the stable low-variance prefix, replan before high-variance
  tails); π0.5-based flow policies, LIBERO 94.75→98.00 with 43%
  fewer replans. Banked to #1 as the inference-time cousin of our
  dispersion machinery — our panel is offline chunk-MAE (replan
  timing is invisible to it), so this is a ROLLOUT-phase lever for
  the rig/sim stage; note the ceiling read's dispersion-vs-gain
  quartiles are exactly the offline precursor (if oracle gain
  concentrates in high-dispersion frames, both selection and
  DVAC-style commit-gating draw from the same signal).
