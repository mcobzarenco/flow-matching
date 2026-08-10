# 4. Stage-2 follow-ups (flow expert on AR trunk) — `decided` (frozen default ADOPTED 2026-08-09; arm 1 depth-of-reads + F-then-joint rung stay open)

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

- **Lit 2026-08-09
  ([FlowDAgger](../papers/flowdagger-latent-dagger.md),
  2607.08877)**: steer-window context note, weighed only at F≈K —
  the frozen-capital *aftermarket* now has a measured retention
  number (latent-space adaptation keeps held-out skills at 0.88
  where SFT collapses to −0.94 delta). Not seam evidence (their
  bases are complete post-attach VLAs); it prices what an intact
  frozen policy composes with after deployment (correction loops,
  guidance, steering — with [Q-guided
  flow](../papers/qguided-flow-critic.md) from the inference side).

- **Lit 2026-08-09 ([Hy-Embodied stack](../papers/hy-embodied-stack.md),
  2606.14409, read during the live F arm)**: one more joint-pole
  ledger entry — 4B MoT trunk + random-init 370M flow expert,
  everything trainable, no stop-grad, no insulation — but from an
  *embodiment-pretrained* VLM, i.e. exactly
  [APT](../papers/apt-expert-pretraining.md)'s named condition for
  joint being safe; doesn't re-rank F vs K for our
  generic-VLM + random-expert regime. Expert sizing corroboration:
  ~11:1 trunk:expert (370M on 4B), same regime as ours. RoboTwin
  stack-vs-stack numbers (+25 over π0) are data-mismatched — not
  seam evidence.

- **Lit 2026-08-09
  ([ActionX](../papers/actionx-rl-expert-pretraining.md), Frontiers
  Neurorobotics 1806605, read during the live K arm)**: the
  **F-then-joint rung's second independent citation, in its exact
  shape** — supervised expert pre-training on a FROZEN trunk, then
  full joint unfreeze (no stop-grad), beats joint-from-scratch by
  +38 pts LIBERO-Long (52 vs 14; their RL variant adds +14 more but
  needs rollouts we don't have). No frozen-vs-joint ablation at
  matched conditions, so it does NOT re-rank F vs K — it prices the
  escalation rung behind tonight's Δ_seam readout. Venue caveat
  loud: Frontiers; ablation ORDERING trusted, magnitudes not (their
  no-pretrain Spatial row reads 0%, likely matched-budget snapshot).
- **Lit 2026-08-09 later session
  ([RDT2 page](../papers/rdt2-umi-scaling.md), 2602.03310)**: a
  production-scale vote for the F-shaped ordering, hours before the
  Δ_seam read — RDT2's 7B recipe is discrete-AR pretraining first
  ("avoided damaging discrete VLM knowledge", their ablation), then
  a 400M flow expert trained on a **frozen** backbone, then 1-step
  distillation; **no joint stage at all** in the main recipe. Filed
  as F-pole ledger context only — the frozen read decides on our own
  numbers, unchanged; a K win would now have to explain away both
  APT's diagnosis and a shipped 10k-hour stack that never unfroze.
- **Lit 2026-08-09 12:3xZ session
  ([Qwen-VLA page](../papers/qwen-vla-early-fusion.md),
  2605.30280)**: F-then-joint production vote #2, also filed
  pre-Δ_seam — Qwen-VLA's Stage I trains its 1.15B expert with the
  trunk FROZEN (text-to-action warm-start) before any joint stage;
  exactly the APT/ActionX escalation shape our f-then-joint rung
  names. Disanalogy carried: their Stage I is language-only, not a
  full-recipe F arm. With RDT2 that's two production stacks this
  week whose first move protects the trunk from a random-init
  expert; ledger context only, the frozen read still decides on our
  own numbers.
- **DECISION 2026-08-09 ~13:5xZ
  ([memo](../posts/2026-08-09-molmo2-stage2-attachment-decision.md)):
  the frozen default STANDS — sequential hard-freeze is the stage-2
  attachment recipe for the Molmo2 trunk class; KI-joint
  closed-unmeasured (not falsified).** The screen ended with one
  complete arm (owner killed K at ~4160 on cost, 12:38Z). Basis: F
  valid (panel 9.4157 vs state-copy 11.7639, −2.35 ≥ the 1.0
  decisive bar; cond-sens 0.828); 8 matched in-run probe evals show
  no K advantage (K−F mean +0.208, K ahead 2/8) with K's CE branch
  healthy throughout (2.6–2.8 vs phase-1 tail ~3.68 — trunk fine,
  still not paying); measured cost 4.11× (3.782 vs 0.920 s/step);
  production frozen-first votes (RDT2, Qwen-VLA). Wall-OSS reading
  recorded: phase-1 CE already routed the action gradients. Binds:
  full-length stage-2 on this trunk attaches FROZEN on the pinned
  surface (taps 12@stride3, h1024×12), pre-reg cites the memo;
  scale caveat carried (re-screen on trunk/scale change). Residuals
  priced: Δ_seam@3750 from retained checkpoints ~2.5 GPU-h (own
  pre-reg, rescue-only); F-then-joint draft must argue vs the 4×
  joint-step cost; AEGIS repair unbanked-never (its trigger outcome
  can no longer occur); arm 1 depth-of-reads untouched.
- **F-then-joint rung PRE-REG DRAFT posted 2026-08-09 ~14:2xZ
  ([draft](../posts/2026-08-09-prereg-fjoint-rung.md)) — the
  escalation the memo unblocked, drafted in the adamc_100k shadow.**
  Design: both arms warm-start from the banked F@10k endpoint
  (APT's Stage-1 capital, already paid for); **J** = trunk unfrozen,
  NO stop-grad, CE rider continuing (the APT best-row analog) vs
  **F2** = frozen continuation control (exists because F's probe was
  still falling — beats crediting the joint phase with plain extra
  training). Matched +5k eff-48, fresh shared shuffle seed 2;
  primary Δ_joint = J@+5k − F2@+5k paired CI; conditional 10k
  extension only on a negative CI; adoption bar −0.3 at +10k; drift
  band 0.3 vs 60k 5.8602 inherited as read 4. Cost committed ~32
  GPU-h (ceiling 35; extension → global 70), J's rate anchored on
  K's MEASURED 3.782 s/step. The 4×-cost burden answered up front:
  the rung prices a bounded final phase, not a lineage. Code audit
  found `--init-from` does the warm-start nearly free; instrument
  gaps named = composite materializer (F@10k + phase-1 rider
  tables), a narrowly-scoped escape for the naive-joint guard
  (refusal stays for random-init), AR-view compat check, J-config
  memory smoke. DRAFT status: finalizes on instrument + owner go +
  execution queue item; venue opens ~08-12 post-adamc-endpoint, and
  the rung-vs-attach sequencing question goes to the owner at
  finalization.
- **Instrument LANDED 2026-08-09 ~15:0xZ (finalization condition 1
  of 3 satisfied).** `materialize_fjoint_init.py` builds the
  composite warm start (F expert/prompt/trunk bytes verbatim +
  phase-1 FAST tables as the rider; refuses byte-differing trunks —
  the wrong-phase-1 trap); `--joint-unfrozen-seam` is the guard
  escape, warm-start-only (requires `--init-from`, refusal
  verbatim-preserved for fresh runs, launch banner says `seam
  UNFROZEN (flow grads enter the trunk)`); AR-view drift-read compat
  verified against J-written checkpoints on the fixture family.
  12 new oracles in `tests/test_fjoint_init.py`, `check.py` 596
  green. Remaining before launch: owner go at the sequencing
  decision + the box memory smoke (§4) at launch time.

- **2026-08-09 fresh sweep — the fjoint joint phase as a
  *conditional escalation*
  ([Z-1 page](../papers/z1-selective-joint-rl.md), 2606.31846):**
  fourth same-shape vote for frozen-first (after LP-FT, APT,
  ActionX), and the first to make the JOINT half conditional in
  production: Z-1's GRPO on a flow VLA trains the action expert on a
  frozen PaliGemma by default, unfreezing the trunk per-task only on
  training-stage diagnostics (SFT success level, early expert-only
  progress, rollout failure modes), configuration frozen before
  final eval. Maps onto the fjoint rung's conditional-extension
  clause — a prior for making the joint phase trigger-gated rather
  than scheduled. Evidence thin (one task shown, no final-number
  decomposition); ledger prior, no gate change.

**2026-08-09 — lit `0815`: the seam question's capacity axis,
measured ([Decoupled Action Expert page](../papers/decoupled-action-expert.md),
2511.12101):** pretrain a generic denoiser on observation-free
forward-kinematics data, freeze it, retrain only the conditioning
pathway — a 5M MLP matches (and beats) Diffusion Policy's 244M
U-Net: LIBERO avg 84.7 vs 79.3, and the frozen-backbone version
keeps 84.2. The capacity prior for the fjoint read: the pure
denoising job fits in ~5M params, so the F arm is not
expert-capacity-starved — if J beats F2, read it as
trunk-representation adaptation, not expert relief; if J≈F2, this
paper is the null's mechanism. The conditioning-mechanism ablation
is the load-bearing datum for us: cross-attention conditioning
collapses under backbone freezing (DP-T 76.4→5.9 LIBERO, −41.5
avg) while modulation (FiLM/AdaLN) survives within ~0–8 pts — task
knowledge migrates into whatever pathway is trainable, so the
banked F@10k expert (cross-attn to residual taps) is
task/trunk-entangled capital, not a portable module. Framing
caveats carried loudly: the testbed is Diffusion Policy — no VLM
trunk, no flow matching, VLA validation explicitly deferred; the
freeze direction is *inverted* vs our seam (they freeze the action
backbone and train conditioning); and "5M matches 244M" is partly
plain capacity (DP-MLP beats DP-C end-to-end too, 84.7 vs 79.3).
Capacity datum only; silent on frozen-vs-joint.

**2026-08-09 — lit `0816`: a third pole for the attachment
frontier, and a sixth production frozen vote
([VLA-GSE page](../papers/vla-gse.md), 2605.06175 +
[LWD page](../papers/learning-while-deploying.md), 2605.00416):**
VLA-GSE initializes a tiny adapter-MoE from the frozen backbone's
own SVD spectrum (leading components → always-on shared expert,
disjoint residual blocks → 7 routed rank-2 experts, 2.51% params)
and beats full finetune on LIBERO-Plus perturbation robustness
(81.2 vs 74.9) while retaining LoRA-grade VLM knowledge — an
anti-unfreeze datum from the PEFT direction, though it never tests
our sequential-converged-F-then-brief-joint design. The ablation
isolates the mechanism: Gaussian-init same-architecture lands at
60.9, *below plain LoRA's* 69.2 — the spectral init carries the
gain, the MoE plumbing is worth ~4–6 pts (PiSSA already at 74.5).
Frontier now has three poles: pure-frozen F (ours), spectral-init
trunk adapters (F-cost, trunk never moves), brief joint unfreeze
(fjoint, ~32 GPU-h). Cheapest decisive probe if this pole ever
opens: PiSSA-vs-LoRA-vs-nothing on the tap layers. Hook
corrections logged: "zero-shot" = held-out *perturbations* of
trained tasks; "insulation-by-construction" oversells — retention
is empirical and LoRA-grade (51.1 vs LoRA 51.8 MMMU). And LWD's
production datum for the ledger: 16-robot fleet RL updates **only
the flow expert on a frozen trunk** even mid-RL with every
incentive to adapt — the sixth production frozen-first vote. The
fjoint pre-reg's frozen reads are untouched.

**2026-08-09 — MolmoAct2 deep dive
([post](../posts/2026-08-09-molmoact2-deep-dive.md), 2605.02881):
the seam ledger's most relevant production entry.** Their staging
IS our debate: post-train the 621M flow expert **with knowledge
insulation** (KV conditioning detached — F's philosophy), then at
finetune **drop insulation and unfreeze everything** — measured:
expert-only finetune 93.05 vs full-FT 97.20 (**+4.15, the
strongest joint-pole vote banked**), insulation-at-finetune a wash
(97.05 vs 97.20), discrete co-training rider +0.25, LoRA −0.95.
Caveats before this re-ranks anything: their "expert-only" starts
from a jointly post-trained system, not a converged frozen-trunk
expert like F; LIBERO is at ceiling; and it's the same benchmark
family every ledger entry leans on. Net: doesn't overturn the
frozen memo, but it predicts fjoint > F2 and raises the rung's
expected value. Also: per-layer KV cross-attention beats
final-hidden-state 95.9 vs 94.0 — the deep-read direction of our
12-tap surface, priced at ~+2 at ceiling.

**2026-08-09 — lit `0819` ([CL triangle](../papers/cl-triangle.md)):**
two free riders for the fjoint rung. (1) A zero-cost drift
instrument: per-layer weight-delta effective rank + nuclear norm vs
the pretrained trunk (the papers' cleanest full-FT-vs-LoRA statistic:
324.7±465.0 / 4.31 vs 27.5±5.7 / 0.48) — computable from checkpoints
we already save. (2) A LoRA-joint candidate first rung: LoRA-32 on
the trunk preserves geometry at ~1.4% params and is proven
insufficient alone under SFT but cheap insurance under a joint phase.
Downside bound banked: "forgotten" competence recovers in <10% of
original training steps — a bad joint phase is a recoverable
experiment, not a lost trunk.

**2026-08-10 — T1 tiny-expert capacity rung READ OUT
([results](../posts/2026-08-10-tiny-expert-results.md),
[pre-reg](../posts/2026-08-09-prereg-tiny-expert-40k.md)): the
Decoupled-Action-Expert width prior CONFIRMED at the pinned band on
our stack.** h256/d12 (86.8M expert, 4.2× smaller total — the
identical tap/adapter surface is a fixed cost) vs F h1024/d12
(367.5M), same frozen 60k trunk, fully step- and batch-matched at
10k. Primary paired read on 15,056 panel-v2 core frames:
**Δ_capacity@10k = +0.188 [CI95 +0.155, +0.221]** — inside |Δ| ≤ 0.3
("prior confirmed"), far from the ≥ +1.0 capacity-binds line, but
the CI excludes zero: width buys a real, small +2.0% margin,
concentrated late-horizon (per-step Δ grows +0.106 → +0.374 across
the 50-step chunk). State-copy execution oracle byte-green across
machines. Consequences for this idea: the fjoint expert does not
need h1024 to hold the frozen-trunk score — expert sizing is now a
cost knob, not a risk knob, and the cheap-expert pole (#16 rig
inference) has a measured price tag. Probe-vs-panel sign flip logged
(probe had tiny −0.069 UNDER F; the panel flips it to +0.188 over) —
small-sample probes kill runs, panels make claims.
