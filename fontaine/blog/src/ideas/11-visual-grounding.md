# 11. Visual grounding arms — `queued`, the open front

Re-anchor probe: error is frame-dependent level mis-estimation;
acuity probe: the text stack's use of visual tokens is the
bottleneck. Arms: trunk shaping, schedules, vision-side aux tasks —
chartered on the community panel; `first_mae` is the
grounding-sensitive column (2.143 vs copy 2.620 — headroom).
High-variance; counts toward the ≥20% exploration budget.

- **Lit slice 2026-08-06 ~02:5xZ — mechanism story named: state-
  dominant bias.** [ReViP](https://arxiv.org/abs/2601.16667)
  diagnoses "false completion" in VLAs as modality imbalance —
  policies over-rely on internal state progression and under-use
  visual evidence (their fix: a progress-aware observer that FiLM-
  modulates the vision/proprioception coupling; +26% over π0 on
  their perturbation suite; abstract-depth read). The causal-
  confusion line ([2506.23944](https://arxiv.org/abs/2506.23944),
  [2509.18644](https://arxiv.org/abs/2509.18644)) says the same:
  proprioception is the shortcut, vision is what generalizes. This
  is a candidate mechanism for BOTH our standing grounding gap
  (first_mae barely ahead of state-copy) AND B's pending aux-off
  flag (first_mae 3.5009 WORSE than copy 2.6202 @40k — consistent
  with aux-off models leaning harder on the state shortcut; paired
  per-frame reads pending ~04Z decide nothing until then).
- **State-reliance probe rung (a) — PRE-REGISTERED 2026-08-06
  ~03:1xZ, instrument landed**
  ([pre-reg](../posts/2026-08-06-prereg-state-reliance-probe.md)):
  `bijou.eval --mask-state` substitutes the dataset state mean (soft
  state token collates to exactly zero; `_state-masked` name suffix;
  report/npz record it; parse guards; `tests/test_mask_state.py`).
  Frozen subset `plans/holdout_curated_v0_k4l2_stateprobe_q4.json`
  (every 4th core row, 4,301 frames — intact side pools from banked
  npzs, zero intact evals). Primary read D = Δ_first(B) −
  Δ_first(A-s0), supported iff CI excludes 0 and D ≥ 0.05. 4 masked
  runs ≈ 1.7 GPU-h, blocked on A-s0's ~04Z npz; first quiet GPU
  window. Supported ⇒ #9 state-dropout gets its own pre-reg;
  ReViP-style modulation stays the heavier arm behind it.
- **Rung (a) RESULT 2026-08-06 06:1xZ — SUPPORTED**
  ([results](../posts/2026-08-06-state-probe-results.md), instrument
  `fontaine/scripts/state_probe_results.py`, report
  `reports/analysis__state_probe_q4.json`): **D = Δ_first(B) −
  Δ_first(A-s0) = +0.702, CI95 [0.498, 0.916]** — 14× the 0.05
  threshold; chunk secondary agrees (+0.389 [0.106, 0.674]). B's
  better intact first_mae is bought with heavier state reliance.
  All three banked expectations came true (Δ_chunk +15.3–16.4 on
  every arm; no masked arm beats intact state-copy first; D > 0).
  Absolute Δs stay descriptive (OOD masking — masked levels ~2×
  worse than state-copy). Branch rule fired: #9 state-DROPOUT
  promoted to its own pre-reg. The grounding gap keeps re-anchor +
  acuity live for the residual intact-state gap.
- **ARCHITECTURE BATCH #1 PRE-REGISTERED 2026-08-06 ~12:2xZ (owner
  steering 11:44Z: multi-GPU run on fundamental architecture
  changes)** ([pre-reg](../posts/2026-08-06-prereg-arch-batch-1.md)):
  paired arms on the stage-2 family, DDP3 on box GPUs 1–3, panel-v2 +
  stable keying, 40k eff-96 — **arm A `--max-soft-tokens 280`**
  (2× visual tokens/camera, the acuity lever; Amendment 2, owner
  12:59Z: 480p sources make 560's marginal tokens the most
  interpolated — 560 demoted to a follow-on rung contingent on a
  positive 280 read) and **arm B full-residual conditioning**
  (res0..res14 hidden-state streams with learned K/V projections
  replace kv4/9/14; ~23.6M params; impl + 5 oracles landed 12:2xZ)
  vs **control := teacher@40k** (Amendment 1; arm 0 dropped).
  Adopt-lever iff paired Δchunk ≤ −0.15 CI-excl-0; grounding read
  Δfirst ≤ −0.10. Both-null branch promotes the Molmo2-4B trunk
  swap. **Results instrument `arch_batch_results.py` banked before
  any data 13:4xZ (5th oracle-before-data application): 5 oracles
  green incl. v2 anchors + K1 gate vs the teacher's banked probe
  curve.** Explore class.
- **Lit slice 2026-08-06 ~13:3xZ — independent support for the
  early-layers story (arm B context, banked before its data):**
  [SmolVLA](https://learnopencv.com/smolvla-lerobot-vision-language-action-model/)
  conditions its action expert on features from ~L/2 of the VLM
  (not the last layer), and
  [FLOWER](https://arxiv.org/html/2509.04996v1) prunes up to 50% of
  the deep LLM layers outright and reallocates the capacity to the
  diffusion head — both consistent with our acuity probe (position
  info sharpest at the vision-tower output, degrading through the
  LM stack). Read for arm B: if full-residual res0..res14 nulls,
  the cheap follow-on is an EARLY-ONLY schedule (res0..res7, or
  vision-tower output as a direct stream) rather than more layers —
  the literature's winning configs concentrate conditioning at or
  below mid-stack. Also
  [SCALE](https://arxiv.org/pdf/2602.04208) (self-uncertainty
  conditioned adaptive looking) as arm-A-adjacent: token budget
  spent adaptively rather than uniformly; parked unless arm A
  reads positive. Trunk-swap caveat from the
  [ICLR 2026 VLA survey](https://mbreuss.github.io/blog_post_iclr_26_vla.html):
  VLM4VLA finds downstream VLA performance has NO correlation with
  the VLM's standard-benchmark scores — the Molmo2-4B port's case
  must rest on its vision-tower/grounding properties (pointing-
  pretrained, our acuity story), not on benchmark superiority;
  frame the port plan's success criteria accordingly.
  (Abstract-depth reads.)
- **Papers-page re-read 2026-08-07
  ([page](../papers/grounding-conditioning.md)) — the slice above
  sharpened three ways:** FLOWER's "up to 50% pruning" is the
  encoder-decoder config only — **decoder-only optimum is 30%
  dropped and 50% hurts** (72.1/70.7 → 66.4/62.5), with the
  conditioning tap at ~70% depth, *above* mid-stack; SmolVLA's own
  Table 8 shows the **full stack slightly beats the L/2 cut**
  (80.3 vs 78.5 — the cut is for compute, weaker evidence than
  banked); SCALE's banked mechanism was wrong — **no token budget
  involved**, it is uncertainty-gated sampling + vision-encoder
  attention *temperatures* (training-free, +5.8 OpenVLA-LIBERO,
  AR-path-only, directly pluggable on our FAST decode). Corrected
  arm-B read: if full-residual nulls, the follow-on is ONE tap at
  60–70% depth (near kv14), not maximally-early streams — early
  fusion collapses in FLOWER's own ablation.
- **Lit radar 2026-08-06 ~03:2xZ — the mechanism gets a training-
  dynamics CAUSE: [GAP](https://arxiv.org/abs/2602.12032) (ICLR
  2026)** shows proprioception dominates because it offers *faster
  loss reduction early in training*, suppressing visual learning
  specifically during motion-transition phases (target
  localization); their fix adaptively shrinks proprioceptive
  gradients during those phases (phase detection via proprio state
  estimation; sim+real, single+dual arm, works on VLAs).
  Consequences for us: (1) if the state-reliance probe supports the
  mechanism, the #9 train-time arm has TWO candidate levers —
  state DROPOUT (input-side, cheap, the current pick) vs GAP-style
  phase-guided gradient scaling (optimizer-side, no input
  corruption); dropout stays first (simpler, matches
  [2506.23944]'s p=0.8 masking evidence), GAP banked as the
  follow-on if dropout helps but plateaus. (2) GAP predicts the
  grounding gap concentrates in motion-transition frames —
  testable for free in the probe's npz by conditioning Δ_first on
  progress-within-episode; noted for the probe's discussion
  section, not its frozen reads. (Abstract-depth read.)
