# 4. Stage-2 follow-ups (flow expert on AR trunk) — `screening` (attachment seam screen PRE-REGISTERED 2026-08-07 ~05:1xZ)

*Tag: `seam-screen` · idea #4 · [index](../ideas.md)*

Inherited questions from mainline §8.11 (banked: 6.57 in-run / 6.62
panel @80k, 2.2× smaller expert): more/deeper export streams (AR
adaptation lives in all 35 layers; the expert reads {4,9,14} —
untested headroom), expert width h512/h1536 on the better features, a
second-generation AR trunk re-measured through the stage-2 lens.
Cost: one screen-rung run per arm. Falsification: paired screens at
matched steps.

- **Deep read 2026-08-07 00:2x–00:5xZ
  ([post](../posts/2026-08-07-pi05-deep-read.md)): π0.5
  (arXiv:2504.16054) + Knowledge Insulation (arXiv:2505.23705) are
  the production version of this recipe, and the two dials where
  theirs differs from ours are now named external arms:**
  1. **All-layer reads** — their expert attends per-layer to the
     full backbone KV stack; ours cross-attends to 3 exported
     streams ({4,9,14} of 35). Production-scale evidence for the
     deep end of the already-flagged export-streams headroom.
  2. **Trunk kept adapting under stop-grad** — KI's backbone
     continues CE-on-FAST *during* expert training, with
     stop-gradient on the expert→backbone attention seam
     (expert queries attend to `sg(K_b)`, `sg(V_b)`); naive joint
     training collapses language following (~75%→~5-10%) and is
     7.5× slower to converge; with stop-grad the CE/flow loss
     balance needs no tuning (α=1 vs π0.5's tuned α=10). Our
     sequential freeze is "extreme KI" — KI's frozen-backbone-0%
     result does NOT indict it (their backbone was action-naive,
     ours is action-pretrained), but a joint arm (trunk CE
     continuing + stop-grad seam) vs the frozen baseline is a
     screen-rung question with a banked anchor.
  Natural venue: the Molmo2 trunk endpoint (~08-08), where stage-2
  attachment becomes a live decision. π0.5 post-training also keeps
  the discrete head alive beside the flow head — our decoder kinds
  share the seam, so a both-heads arm is config, not surgery.
- **Lit slice 2026-08-07 03:2xZ — a THIRD independent group ships
  the KI-joint recipe:** [LabVLA](https://arxiv.org/html/2606.13578)
  (lab-bench manipulation, Qwen3-VL-4B trunk) uses exactly the
  two-stage shape we're deciding on: FAST-token pretraining makes
  the backbone action-aware first, then an 18-layer DiT flow expert
  attaches under stop-grad with **the backbone's FAST/annotation CE
  kept active during expert training** (their words: flow loss
  updates projection+DiT only; token losses still train the VLM).
  No isolating ablation published — adoption evidence, not
  measurement — but the KI-joint arm now has π0.5/KI + LabVLA
  behind it vs our sequential-freeze default. Also a second 4B-scale
  trunk data point for the Molmo2-4B port's size class. Their expert
  reads a projected "detached prefix slice", not all-layer KV —
  arms 1 and 2 remain independently testable.
- **Lit slice 2026-08-07 04:4xZ — the seam question now has a
  three-way map, all sides published:**
  [AEGIS](https://arxiv.org/abs/2604.16067) (2604.16067) names the
  mechanism the KI/stop-grad camp is defending against —
  "cross-modal gradient asymmetry": high-magnitude continuous
  flow/MSE gradients from the action expert overwrite a CE-trained
  VQA manifold; their middle path is layer-wise orthogonal gradient
  projection + a Wasserstein-2 anchor to pre-trained activations
  (claim: sheds <1% of gradient energy yet stops the activation
  drift; abstract carries no task-success table). A trained-repair
  alternative to the stop-grad seam if the KI-joint arm ever shows
  the expert starving for backbone adaptation — bank, don't build.
  [Wall-OSS-0.5](https://arxiv.org/abs/2605.30877) (2605.30877)
  ships the opposite corner: gradient-bridged co-training where the
  DISCRETE action head routes "VLM-native" CE gradients into the
  backbone while flow matching is the deployment-time interface —
  which is structurally OUR recipe (FAST-CE trunk + flow expert),
  argued from the multimodal-preservation side. Net for the
  attachment decision at the molmo2 endpoint: the frozen-vs-KI-joint
  screen (arm 2) stays the right first measurement, and both
  escalation directions now have named citations if it lands either
  way. **Papers-page re-read 2026-08-07 (~09:2xZ,
  [page](../papers/seam-debate.md)) surfaced the load-bearing
  ablation our skim missed: Wall-OSS's 5-task from-scratch seam
  comparison has STOP-GRAD WORST (co-train 57.0% > flow-only 36.6%
  > stop-grad 31.9%) — from-scratch/action-naive regime, so it does
  not indict KI-in-posttraining, but it is the strongest published
  counter-evidence to stop-grad-as-free-lunch; AEGIS re-read:
  preservation-only, NO closed-loop success table at all.**
- **Lit slice 2026-08-07 ~11:2xZ (pre-endpoint, the last look before
  the stage-2 decision) — APT
  ([2606.12366](https://arxiv.org/abs/2606.12366),
  [page](../papers/apt-expert-pretraining.md)): the seam damage is an
  INITIALIZATION problem.** Random-init experts learn the
  language-imbalance shortcut and their noisy early gradients are
  what wreck the trunk; pretrain the expert first (language-masked,
  frozen trunk — structurally our F arm) and the best published
  recipe then unfreezes EVERYTHING with no stop-grad at all
  (pick-place grid: expert-pretrain+joint 98/84/92/58 vs KI+pretrain
  96/74/90/62 vs π0.5's 84/70/86/50; "stop-gradient is not a
  necessary condition"). Net for the screen: NOTHING changes
  pre-readout (K's seam is corroborated for the random-init regime
  we are actually in); the escalation map gains a named rung —
  **F-then-joint**, warm-starting a joint run from the F
  checkpoint's expert (free Stage-1 capital) — and an F≈K tie now
  has a published interpretation (two working guards; the next
  contrast is initialization, not the seam). **Siblings read same
  session (~11:5xZ, [page](../papers/vla-initialization.md)):**
  VLM4VLA (2601.03309) — frozen VISION ENCODER is the published
  frozen-trunk failure mode (Qwen2.5VL-7B Calvin 4.057→2.823,
  Paligemma 3.506→0.495; word-embedding freeze free; VQA scores
  poorly predict VLA rank) → F-arm caveat softened for us (our
  trunk is embodiment-adapted BEFORE the freeze) + the diagnostic
  if F loses: look at vision-limited frames first; feeds #17 a
  trunk-selection criterion (probe vision adaptability, not VQA
  benches). 2605.25802 — LoRA > full-FT for VLA init ("overly
  reshaping the pretrained representation weakens initialization");
  reconciles with APT: what matters is what SHAPED the gradients
  that move the trunk, not whether it moves.
- **Attachment seam screen PRE-REGISTERED 2026-08-07 ~05:1xZ
  ([post](../posts/2026-08-07-prereg-molmo2-attach-screen.md))**: two
  arms at matched 10k steps / eff-48 on the molmo2 40k endpoint
  trunk — F (hard-frozen, our default) vs K (KI-joint: phase-1 CE
  continuing verbatim + stop-grad seam, α=1 fixed); naive joint NOT
  run (KI measured the collapse). Surface held constant across arms:
  residual taps, molmo2 rule pinned (12 taps, stride 3, layers
  2,5,…,35, expert depth 12, h1024) — the depth-of-reads dial (arm
  1) stays open, NOT measured by this screen. Primary read Δ_seam =
  panel_v2 heun30/draws1/stable K−F paired CI; trunk-drift
  diagnostic (K's greedy AR panel vs the 40k endpoint number, band
  0.3) is the language-following analog; frozen decision rule (ties
  → frozen default stands). Instrument to land oracle-gated first:
  molmo2 residual exports + guard lift, seam stop-grad flag, joint
  CE+flow objective; #20 activation checkpointing is a hard K
  prerequisite. Cost est. 50–60 GPU-h, ceiling 70 with matched 5k
  downshift. Opens after the endpoint + #19 box obligations + the
  attachment-decision item (owner steer window).
- **Instrument LANDED 2026-08-07 ~06:0xZ, oracle-gated (this
  commit)**: all three pre-reg parts. (1) `Molmo2Encoder` residual
  exports (the trunk-side tap protocol existed since WP1 — the
  encoder/config/loading wiring was the gap; queue-title audit paid
  off again): `residual_exports` on encoder + `Molmo2PromptConfig`,
  `molmo2_residual_taps` pins the rule (stride 3, last tap = final
  layer; 36 ⇒ 12 taps at 2,5,…,35), `molmo2_residual_expert_config`
  mirrors trunk geometry (GQA 8 kv-heads × head_dim 128, plain RoPE
  θ=5e6), guard lifted for `--decoder flow --conditioning-streams
  residual`, checkpoint load/save round-trips. (2) `--seam-stop-grad`
  detaches taps pre-adapter in `BijouModel.encode`. (3) `--joint-ce`:
  Molmo2ARDecoder rider on the model, CE sums inside autocast + fp32
  flow outside, THREE-normalizer chunked-backward form, rider tables
  at decoder-lr, saved as `joint_ce.safetensors` (+ config section),
  rider continues from the endpoint's `expert.safetensors` under
  `--backbone-init-from` (pre-reg AMENDMENT appended: "decoder
  fresh" = the flow expert; a fresh-table CE branch contradicts
  "continuing verbatim"). 13 new oracles in
  `tests/test_molmo2_residual.py` — taps byte-match trunk hidden
  states, cache bit-identical with/without taps, K/V contract +
  padding invariance, stop-grad zero-vs-nonzero with naive-joint
  negative control, and BOTH α-edges bitwise through the real
  `BijouTrainStep` (flow half ≡ F-arm step; trunk grads ≡ phase-1 CE
  step). check.py 423 passed. Remaining before launch: #20
  activation checkpointing (hard K prerequisite), F/K launch
  scripts + the joint-checkpoint AR-view materializer for read 4.

- **Launch prep LANDED 2026-08-07 ~06:1xZ (work session):** both arm
  launchers exist (`launch_box_fontaine_molmo2_attach_{F,K}_10k_ddp4.sh`
  — sequential F-first, sha256-pinned plans, chained panel_v2 evals;
  K chains `materialize_joint_ar_view.py` + the greedy k4l2 drift
  panel for read 4, and a `K_MEM_READY` guard refuses a blind K
  launch before #20 + the smoke ladder). The 70 GPU-h cost gate is
  mechanized (`attach_rate_gate.py`, median-s/step projection with
  the batch's extra term; 5k-downshift marker BOTH launchers honor —
  matched, never one arm alone). Probe-kill bars pinned in
  babysit.toml prepared entries: 12.6394@5000, 11.6356@7500,
  10.1652@10000 (phase-1 curve + 3.0; the @10000 value from the K1
  crossing, green at 7.1652 vs ≤12.0944). AR-view materializer
  oracle-gated against the REAL `save_checkpoint` write side (rider
  bitwise, adapted trunk, taps stripped, greedy decode via
  `from_checkpoint` on the tiny fixture). 10 new oracles; check.py
  433. Remaining before launch: #20 + the K smoke memory ladder.

- **K smoke-ladder SCRIPT LANDED 2026-08-07 ~06:5xZ (work session):**
  `smoke_attach_k_ddp4.sh` — the exact K recipe verbatim (warm-start
  from the 40k endpoint, `--joint-ce --seam-stop-grad
  --activation-checkpointing`, zero1 + chunked backward), 150 steps
  per rung with eval@100 + save@100 so the probe-decode and
  joint-checkpoint-writer memory shapes are exercised, not just the
  bare step. Ladder B12c6 → B8c4 → B6c3 (chunk microbatch pinned at
  2); pass = rc 0 AND max `vram_alloc_peak_gib` over the rung's jsonl
  ≤ 71.0 (torch alloc peak, the gate babysit enforces — NOT
  nvidia-smi reserved, the phase-1 ladder lesson); green writes the
  `fontaine/harness/state/k_mem_ready` record and echoes the exact
  `K_MEM_READY=1 BATCH= BACKWARD_CHUNKS=` launch line; a sub-B12
  green is echoed as a MATCHED DOWNSHIFT for BOTH arms (the ladder
  must run before F, not just before K); all-red = no marker, owner
  steer. Runs on the box after the endpoint (~08-08), post the #19
  box obligations. Remaining before launch: run the ladder green.

- **Δ_seam frozen-read script LANDED 2026-08-07 ~07:2xZ (work
  session):** `attach_seam_results.py` — the pre-reg's reads 1–5 as
  one command, ready before any arm data. Read 1 Δ_seam paired
  per-frame chunk CI (K − F, panel-v2 core, seeded bootstrap 10k —
  `box_batch_results.py` pooling verbatim, `arch_batch_results.py`
  paired-read/LORO machinery reused via sibling import); read 2 the
  frozen decision rule with ALL branches coded (KI-joint adopt /
  frozen-default-stands + Wall-OSS reading / K-wins-with-named-cost
  → AEGIS escalation + owner steer / partial-pending-drift); read 3
  context anchors quoted-not-deciding, with the state-copy execution
  oracle pinned pre-data ("decisively" = ≥ 1.0 chunk-MAE below the
  same-npz state-copy — VOID outranks every seam verdict); read 4
  trunk drift from the K ar_view + endpoint k4l2 JSONs, band 0.3
  inclusive, strict plan/frame-count semantics guard; read 5
  first_mae mirror + per-step curves (record-only). Defaults wired
  to the launchers' exact output names incl. the 5k-downshift stems
  (`--steps 5000`). Oracle-gated pre-data: v2 anchors 6.7151/1.9453
  + state-copy 11.7639 reproduced through this file's own pooling;
  degenerate K:=F → exact zeros → frozen-default; synthetic ×0.95 /
  ×1.05 / ×3.0 error effects fire adopt / falsified / VOID; drift
  band edge inclusive; misaligned index + wrong-plan JSONs hard
  abort. check.py 437 passed. Remaining: the screen itself (~08-08).

- **Lit slice 2026-08-07 ~08:3xZ — a scale-transfer caveat on
  reading Δ_seam:** Encoder Winners Do Not Reliably Transfer Across
  VLA Backbone Scale ([2606.14153](https://arxiv.org/abs/2606.14153))
  — frozen-backbone grafting diagnostic (swap the component, freeze
  the rest, one trainable projector): component RANKINGS flip with
  backbone scale (SigLIP wins at SmolVLA-450M, DINOv2-small at
  pi0.5-3.3B; 40 grafting runs, two LIBERO suites), and the wrapper
  itself has opposing effects across backbones. Two takes for the
  screen: (i) methodological validation — cheap frozen-graft screens
  as a pre-commit diagnostic is exactly the F-arm's role; (ii) the
  caveat to write into the read — the F-vs-K verdict is a
  molmo2-at-this-scale fact, not a family-wide law; re-screen, don't
  extrapolate, if the trunk or its scale changes (#17 trunks).

- **Lit slice 2026-08-07 ~09:1xZ — the recipe has independent
  adopters, and the frozen arm has an RL future:** (i) LabVLA
  ([2606.13578](https://arxiv.org/abs/2606.13578)) trains a lab-robot
  VLA with EXACTLY our staging — FAST-token pretraining makes the
  Qwen3-VL-4B backbone action-aware first, THEN flow-matching
  posttraining attaches a DiT expert under knowledge insulation;
  tops LabUtopia ID + OOD among their baselines. Independent
  adoption of the stage-1-AR → stage-2-KI-attach ordering the seam
  screen measures — the K arm is the field's incumbent, which is
  what makes Δ_seam the right pre-commit read. (ii) Q-VGM
  ([2606.08015](https://arxiv.org/abs/2606.08015)) does OFFLINE RL
  on a flow-matching VLA with the backbone FROZEN and only the flow
  expert updated — Q-gradient ascent on clean-action estimates
  converted to residual velocity targets (no backprop through the
  denoising chain); LIBERO 79.0% → 92.5% from a few-shot SFT start.
  Meaning for the screen: the F-arm configuration (frozen trunk +
  flow expert) is not a dead end even if Δ_seam favors K — it is
  the exact substrate the field fine-tunes with offline RL, so
  "frozen default stands" keeps an escalation path that KI-joint
  would complicate (RL updates into a live trunk). Record-only
  prior; no new arm until the screen's own verdict lands.

- **Lit 2026-08-09 ([VLAFlow](../papers/vla-training-objectives.md),
  2607.01586)**: controlled 4-recipe bake-off on one π0-style
  skeleton (5,000 h) — **stop-gradient cost ~26 pts** LIBERO-Plus
  (anti-KI, same side as APT), and the frozen-VLM trade-off measured
  both ways (VL generalization kept 74.9 vs 68.8, embodiment
  adaptation lost: WidowX 54.4). Joins the interpretation ladder for
  the F/K readout; screen unchanged. Caveat carried: their expert
  co-pretrains from scratch — our random-init K expert is exactly
  APT's damage regime, so the screen still adjudicates.
