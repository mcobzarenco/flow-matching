# 5. FAST tokenizer v3 — `queued`

*Tag: `tokenizer-v3` · idea #5 · [index](../ideas.md)*

- **Hypothesis:** refitting on curated-v0's exact quantiles removes
  the ~1.94%-of-chunks clip rate; small but real MAE effect on
  clipped chunks.
- **Cost:** CPU-only fit (~32 min measured for v2); token metrics
  RESET (never cross tokenizer versions) — coordinate with run seams.
- **Falsification:** paired arms (same seed/data/arch, only the
  artifact differs — the v1-vs-v2 precedent); recon error + clip rate
  in the fit report before any training touches it.
- **Lit radar (2026-08-06 20:5xZ):** FASTer (arXiv:2512.04952)
  replaces DCT+BPE with a learned VQ tokenizer ("FASTerVQ" — action
  chunks encoded as single-channel images, global spatio-temporal
  dependencies) + block-wise AR decoding; claims better token
  utilization/reconstruction and SOTA-beating speed+success vs
  FAST-style AR VLAs. If v3's quantile refit leaves clip/recon
  headroom on curated-v0, a learned-VQ arm is the natural rung after
  it (same paired-arm falsification; token metrics reset applies
  either way). **Papers-page re-read 2026-08-07
  ([page](../papers/action-tokenization.md)): scope corrected —
  FASTer's speed win is vs AR-FAST not diffusion (WBC 237 vs π0's
  225 ms), 2.2 pts of its LIBERO headline come from block-decoding
  + action expert not the tokenizer, and the tokenizer gain shrinks
  to +1.3 on a well-tuned FAST baseline. New cheap gate BEFORE any
  VQ arm: compute our v3 fit's vocab utilization / max-token-freq /
  unigram entropy vs FAST-on-Bridge's pathology (48% / 9.6% / 0.69)
  — near entropy 0.9 the arm dies pre-birth. FAST itself confirmed
  (quantile-norm spec = our v3 target; decode latency ~750 ms vs
  ~100 ms diffusion is the binding deployment axis, which #12's
  1-NFE student already sidesteps).**
- **Deep read 2026-08-07 ([post](../posts/2026-08-07-pi05-deep-read.md)):
  KI (arXiv:2505.23705) measured FAST vs naive per-dim binning as
  the backbone's discrete training signal: ~95% vs ~85% table-
  bussing success — external support for the token-quality premise
  behind the v3 refit.

**2026-08-09 — RVQ banked as the priced-first alternative
([RDT2 page](../papers/rdt2-umi-scaling.md), 2602.03310):** at
matched quantization error, residual VQ (temporal-CNN encoder, m
codebook depths onto 1024 reserved vocab entries) uses ~⅓ the tokens
of FAST or uniform binning; their codebook-collapse mitigations
(low-dim codes, cosine similarity, EMA, dead-entry restart) are the
practical recipe. If v3 ever reopens, price RVQ before another FAST
refit.

**2026-08-09 — discretization datapoint from the perception side
([QDepth-VLA page](../papers/qdepth-vla.md), 2510.14836):** CE over
VQ depth codes beats pixelwise depth *regression* by +3.9 avg on
Simpler-WidowX (−14.6 on the worst task) when the supervision is
noisy monocular pseudo-depth — the same
discretize-and-predict-distributions argument FAST makes on the
action side, with mean-collapse under noise as the stated mechanism.
Strengthens the token-quality premise; no v3 status change.

**2026-08-09 — first measured order-preservation datum
([DFM-VLA page](../papers/dfm-vla.md), 2603.26320):** MAAT tokenizer
— uniform 2,001-value grid (no BPE compression) + triplet-margin
embeddings forced to preserve numeric order — is worth +4.4 pp on
LIBERO-Plus for a *refinement* decoder that moves through token
neighborhoods. The trade vs FAST is explicit: compression vs metric
structure. For a pure-AR consumer nothing refines through
neighborhoods, so the answer may differ — "ablate embedding metric
structure" banked as a free rider on any v3 refit. No status change.

**2026-08-09 — lit `0814`: a clean external null for the learned-VQ
escalation ([X-Tokenizer page](../papers/x-tokenizer.md),
2606.14752):** the strongest learned action tokenizer published to
date (2.4M-trajectory pretrain, 17 embodiments) *loses to FAST in
FAST's own job twice* — plain residual-VQ-no-aux is worse than FAST
on control (69.1 vs ~73.0 progress) and the full tokenizer
reconstructs 17% worse (ℓ1 0.01693 vs 0.01446); its real wins
(+13.5% rel. VQA, +8.25 long-horizon) come only in an
auxiliary-supervision role where the tokens are never executed, and
need a frozen 7B VLM teacher. The entropy/utilization gate stands,
now with a measured datum behind it. Two free riders banked for the
v3 refit: their 0.1%/99.9% quantile normalization is exactly the
queued v3 move (independent confirmation), and their
**WER-under-noise probe** (same noisy chunks → token edit distance;
FAST blows up 3× at σ=0.008 via BPE re-segmentation cascades) is a
cheap CPU diagnostic to run on v2-vs-v3. No status change.

**2026-08-09 — lit `0819`
([Action-space design](../papers/action-space-design.md),
2602.23408):** action parameterization is a normalization decision
in disguise — step-wise and chunk-wise delta are bijective
reparameterizations of the same targets that differ ~10pp at rollout
purely through what the decode map does to prediction noise
(step-wise amplifies O(k) in chunk length, (2k+1)/π factor). Judge
any v3 encode map by its decode-time noise amplification, not its
encode-side statistics.
