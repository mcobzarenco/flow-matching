# 17. New trunks / new architectures — standing owner mandate (2026-08-05 17:24Z)

*Tag: `new-trunks` · idea #17 · [index](../ideas.md)*

"The current repo should just be a starting point for what I've
tried." A ranked exploration front, fed by the literature slice;
every candidate enters at the screen rung with a pre-reg and counts
toward the exploration budget.

- **Lit feed 2026-08-07 ([page](../papers/vla-initialization.md)):
  trunk-screening criterion from VLM4VLA (2601.03309)** — general
  VQA-bench scores are POOR predictors of VLA rank (Kosmos-2 1.7B
  beats 30B-class models on SimplerEnv; no model dominates across
  suites); the load-bearing component is the VISION pathway
  (frozen-encoder collapse 4.057→2.823 Calvin, worth +29 pts when
  action supervision reaches it). Screening a candidate trunk =
  probe its vision-pathway adaptability on OUR data, not its
  benchmark card. Also: all 7 of their embodied-VQA co-training
  mixes UNDERPERFORMED plain baselines — do not import aux-data
  recipes on faith (compare against our own #6 measurements).

- **Vision-unfreeze rung pre-reg DRAFTED 2026-08-07, AMENDED to the
  warm-start two-arm design same day (owner steering 18:02Z)
  ([draft post](../posts/2026-08-07-prereg-molmo2-vision-unfreeze.md);
  owner question 17:04Z + the
  [vision-encoder-freeze](../papers/vision-encoder-freeze.md) slice)**
  — both arms `--init-from` the 40k endpoint checkpoint
  (frozen-continue control vs thawed-continue `--backbone-vision-lr
  2e-6`; full-FT tower, never LoRA-on-SigLIP), 3k steps each at the
  40k tail LRs (decoder 1e-5 / text 2e-6), seed 1 both arms →
  identical batches, fresh AdamW symmetric (`--resume` mechanically
  excluded: the extra vision param group breaks
  `optimizer.load_state_dict`). Primary = **thawed@3k − frozen@3k
  paired per-frame Δ** (CI95, null band 0.07; critical-frame
  re-pool robustness). ~15 GPU-h train vs ~27 from-scratch, and a
  win directly upgrades the deployment artifact. Declared caveats:
  late low-LR thaw may understate from-scratch unfreeze (tie ≠
  "unfreezing doesn't help"); panel can't see the MAPS-style OOD
  tax. Memory ladder unchanged (67.07/71 + ~3–4 GiB tower adder →
  chunks 6→12 → decoder activation-ckpt; matched downshift
  excluded). DRAFT status: execution blocked on finalization
  amendment (launcher byte-audit + memory smoke + endpoint probe
  quote) + owner go, window post-attach-screen (~08-09+).
- **OWNER PICK (2026-08-05 21:57Z): E4B screen confirmed as the next
  pre-reg** — AR-100k on the freed 4×H100, **matched parameters with
  the E2B AR-100k run** (verified from the recipe: `--batch-size 12`
  /GPU on DDP4 = effective 48, decoder-lr 1e-4, backbone-text-lr
  2e-5, grad-clip 100, seed 0, same aux/condition/dropout flags; if
  E4B OOMs at 12/GPU, grad-accumulate to the same effective 48 —
  stated in the pre-reg, batch semantics never change
  mid-comparison). **Gates = the MAE curve over time, not the
  endpoint**: matched-cadence in-run probes vs the banked E2B curve
  + mid-run panel evals (~25k/50k) with pre-registered bands so a
  losing rung dies early. **Owner 21:58Z: image (embedding) budget
  is the follow-on ablation arm** on the winning trunk — one
  variable per rung, trunk first, then image token budget (pairs
  with #11's grounding read). **Pre-reg POSTED 2026-08-05 ~22:4xZ**
  ([post](../posts/2026-08-05-prereg-e4b-screen.md)): verbatim mainline
  recipe + `--backbone google/gemma-4-e4b-it` (AR path verified
  fully config-driven), eff-48 with a pre-registered
  chunked-backward fallback ladder (bijou.train has no grad-accum
  today — impl + oracles is a pre-launch CPU item if the memory
  smoke says B12 doesn't fit; E2B B12 peaked 77.5 GiB, E4B text
  ~2.2× params), probe-curve gates @30k/50k (±0.5 floor) + mid-run
  panels @25k/50k on the local GPU, endpoint adopt rule bound to
  tonight's E5 σ_seed via finalization amendment. Launch blocked on:
  box free + e4b checkpoint download (not in box cache) + parity
  spot-check + memory smoke + amendment.
- **NO-LAUNCH (2026-08-06 ~05:4xZ)** — the pre-registered memory
  ladder exhausted: all four rungs (B12 direct / 2×6 / 3×4 / 4×3
  chunked backward) OOM'd on 80 GB before completing one optimizer
  step; Adam state (~31.8 GiB) never even allocated ⇒ steady-state
  needs ~≥110 GiB/rank under the matched recipe
  ([finding post](../posts/2026-08-06-e4b-no-launch.md), Amendment 2 of
  the pre-reg). **Feasibility negative, not a scale answer** — the
  probe/panel gates never ran; the attribution question stays open.
  Follow-on = owner decision (options posted): ZeRO-1 re-entry as a
  NEW pre-reg vs redirect to Molmo2-4B (rank 2) / #11 grounding
  arms. E4B's zero-port-cost premise is dead; Molmo2-4B competes on
  closer-to-even terms now.
- **Owner steering 2026-08-06 11:44Z — multi-GPU architecture run
  requested** (new trunk / full residuals / bigger images, "really
  just examples") → **architecture batch #1 pre-registered ~12:2xZ**
  ([pre-reg](../posts/2026-08-06-prereg-arch-batch-1.md), filed under
  #11 — the owner's examples are the grounding front): bigger-images
  + full-residual arms first (same trunk, clean attribution); the
  trunk swap (Molmo2-4B) is its own follow-on pre-reg, promoted to
  next-in-line if both arms null. E4B ZeRO-1 re-entry queued behind
  the architecture run (owner 11:44Z: E4B paused).
- **Arm B implementation LANDED 2026-08-06 ~12:4xZ** (the F1 critical
  path: the smoke needs BOTH configs, so arm A could not launch
  before this existed): `--conditioning-streams residual` in
  bijou.train — encoder exports raw post-layer hidden states
  (res0..res14), the flow expert projects them through learned
  per-layer adapters (RMSNorm + K/V proj + k_norm/v_norm + RoPE,
  mirroring `TextAttention.project_kv` exactly, so the streams are
  contract-identical to K/V exports and the blocks are untouched).
  Adapters live DECODER-side and attach OUTSIDE the no-grad prefix
  encode — trainable under the frozen trunk. Real-config count
  23.62M params ✓ (pre-reg said ≈23.6M). All five pre-launch oracles
  green as CPU tests (`tests/test_residual_streams.py`, 11 tests):
  stream contract + padding-orientation invariance, trunk
  bitwise-frozen through an optimizer step, grads reach all
  adapters, checkpoint round-trip with no flags, K/V path untouched
  (state-dict keys + banked loss oracles). `check.py` 285 green.
  Still owed at the arm-C-boundary code sync: SnapFlow stage-0
  re-verify on the box + F1 two-config smoke.
- **Molmo2-4B port plan POSTED 2026-08-06 ~14:1xZ** (owner-promoted
  12:03Z, background work independent of the batch verdict;
  [plan](../posts/2026-08-06-molmo2-port-plan.md), distilled primary-
  source doc `docs/molmo2.md` per §6 post-cutoff rule). Key design
  calls: **residual-only conditioning** (arm B's path — learned
  adapters keep the expert contract at kv1×512 regardless of
  Qwen3's GQA 32:8; no KVCache/layer-type/`project_kv` port
  needed) and **15-of-36-layer mount** (fractional depth 0.417 vs
  E2B's 15/35 = 0.429 — expert depth and res0..res14 schedule
  carry over unchanged). Five WPs: WP0 seam refactor (the
  `docs/plan.md` encoder ABC, oracle-guarded, lands alone) → WP1
  Qwen3 decoder port + HF parity (shared with InternVL3.5/Qwen3-VL
  — one port, three trunks) → WP2 SigLIP tower/connector → WP3
  ChatML collator (turn-close probe + state-slot splice re-proved)
  → WP4–5 exports/schema/audit. Phase 1 = flow on the **raw frozen
  prefix** (no AR port, no vocab surgery; the AR-adaptation −2.7
  confound ships with any claim). Mounted footprint ~2.3B ≈
  4.7 GiB bf16; est. 4–6 CPU work sessions, GPU only for parity
  bursts + the memory smoke. First run gets its own pre-reg after
  the §4 oracle suite is green.
- **External prior (lit slice 2026-08-05 ~22:5xZ):**
  [2606.31382](https://arxiv.org/pdf/2606.31382) (VLM-to-VLA
  parameter redundancy) reports **bigger VLM backbones do NOT
  consistently improve action performance after adaptation** (their
  ablations; skim-depth read via a fast-model summary — re-read
  before citing numbers). Direction: strengthens the E4B screen's
  kill branch as a live outcome, not a formality — and raises the
  prior on #11 (grounding/adaptation quality, not trunk scale, as
  the binding limit). The screen runs regardless: our recipe, our
  corpus, pre-registered either way. **Papers-page re-read
  2026-08-07 ([page](../papers/data-and-trunks.md)) — banked claim
  CORRECTED: 2606.31382 makes no backbone-scale claim at all** (it
  is a pruning-as-diagnostic study: divergence-ranked, no-recovery
  pruning removes 12–30% of params at 85–96% retention); the
  bigger-isn't-better claim belongs to VLM4VLA, which it merely
  cites and which we already carry via the ICLR-26 survey — cite
  VLM4VLA for the kill-branch prior, not this paper.
- **Lit slice 2026-08-07 ~04:0xZ — world-action models on the
  radar** (via the
  [NVIDIA WAM post](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/);
  skim-depth, re-read before citing numbers): the emerging tier
  above VLA trunks conditions action decoding on video-model
  dynamics — UniPi '23 → GR-1 '24 → DreamZero '26 (monolithic
  video+action denoising; RoboArena 1750 vs π0.5's 1622),
  LingBot-VA (Wan 2.2-5B inverse dynamics), Being-H0.7 (latent
  VLA↔WAM bridge); π0.7 itself now renders subgoal IMAGES from a
  BAGEL world model between the HL policy and the action expert.
  Direction: a video-capable trunk (Molmo2 — already ours) plus a
  subgoal-image conditioning arm is the reachable-scale version of
  this thesis; ties the #6 explicit-HL ladder to the trunk front.
  No action until the Molmo2 40k endpoint + stage-2 decision land.
  **Papers-page re-read 2026-08-07
  ([page](../papers/attachment-frontier.md)): all named systems +
  numbers verified; two additions** — **Fast-WAM**
  (representation-only, skips test-time video generation, 3–4×
  faster, reportedly matches LingBot-VA *without* the 16k-h robot
  pretrain) is the strongest evidence the video *prior* not the
  generation carries the value → the reachable Molmo2 version is
  predictive-feature conditioning, not rendered frames; and a #6
  flag — π0.7 found TEXT subtasks insufficient for its
  bias-breaking tasks (needed rendered subgoal images), so a null
  on our rung-(a) text probe is consistent with the field, not
  fatal to the hierarchy thesis (state this in the pre-registered
  read).

**Ranked 2026-08-05 by the [trunk survey](../posts/2026-08-05-trunk-survey.md)**
(paper + fetched-config deep-reads, owner method): **1. Gemma 4 E4B**
(zero-cost in-family rung) → **2. Molmo2-4B** (best-in-tier quality,
video+spatio-temporal grounding, Apache) → **3. InternVL3.5-4B**
(same Qwen3-4B decoder as Molmo2 — one port serves both; only modern
4B with a true base ckpt ⇒ the idea #10 vehicle) → **4. V-JEPA 2.1
ViT-L augmentation arm** (the dynamics bet; 2-AC's <62h-robot-video
→ zero-shot Franka is the strongest external evidence for the
north-star thesis) → **5. Qwen3-VL-4B in reserve** (most
reimplementation surface, no base ckpt). Screened out: Ministral 3
3B (no video), SmolVLM2 (older gen; but SmolVLA = external
validation of our trunk+flow-expert protocol), Cosmos-Reason1-7B
(second round iff E4B says scale is the lever), all MoEs (budget +
export-stream semantics). Original slate below, kept for scope:

- **Trunk swaps at reachable scale:** E2B → stronger open VLM
  families (Qwen-VL, larger Gemma-4 variants — E4B/12B) through the
  existing stage-2 trunk-swap protocol; also the queued base-vs-IT
  swap (idea #10) as
  the cheapest member of this family.
- **Video/dynamics-pretrained encoders** (V-JEPA-style) vs
  image-language pretraining — the grounding probes (idea #11) say
  the visual stack is the bottleneck; dynamics-predictive
  pretraining is the structurally different bet.
- **Tokenizer-free continuous action heads end-to-end** — remove
  FAST; the flow expert reads the trunk at full depth rather than
  export streams {4,9,14} (subsumes the idea #4 stream question).
- **Small world-model / latent-dynamics trunk** trained on the
  community corpus, policy as readout — cross-embodiment by
  construction; the speculative end.
- **Consistency-distilled 1–2-step deployment decoders** (pairs with
  ideas #1 and #12) — the deployment-latency leg of the rig goal.
