# 9. Data levers — `screening` (state-dropout arm C ANSWERED 2026-08-06: COSTS, adopt nothing; p=0.3 screen is the sanctioned follow-up)

*Tag: `data-levers` · idea #9 · [index](../ideas.md)*

`--trim-leading-idle` (~6.7% of frames), state-noise augmentation,
judge-score-weighted sampling (never yet run). Each is a cheap paired
arm at the screen rung. Any derived corpus ships with the leakage
check (charter §2) before training touches it.

- **Lit slice 2026-08-06 ~02:5xZ — state-noise sharpened to state
  DROPOUT:** the shortcut-learning literature's standard lever is
  random state *masking*, not noise —
  [Adapt Your Body](https://arxiv.org/abs/2506.23944) masks
  proprioception to zeros with p=0.8 and reports it effective
  against proprioception-shortcut overfitting;
  [2509.18644](https://arxiv.org/abs/2509.18644) goes further
  (state-FREE policy, relative EE actions, vision-only) and reports
  better spatial generalization. If the #11 reliance probe shows
  heavy state reliance, the paired arm here is `--state-dropout p`
  (train-time masking, eval unchanged) — config-only surface, screen
  rung. **PROMOTED 2026-08-06 06:1xZ: the #11 probe came back
  SUPPORTED (D = +0.702 [0.498, 0.916],
  [results](../posts/2026-08-06-state-probe-results.md)) — the branch
  rule fires and state-dropout is owed its own pre-reg** (design
  notes: p per 2506.23944's 0.8 vs a lower screen value is the one
  free parameter; the probe's masked-eval instrument doubles as the
  reliance readout for the trained arm; GAP-style phase-guided
  gradient scaling stays the follow-on if dropout helps but
  plateaus). **PRE-REGISTERED 2026-08-06 ~08:1xZ
  ([pre-reg](../posts/2026-08-06-prereg-state-dropout-40k.md)): arm C =
  A-s0 recipe + `--state-dropout 0.8`, seed 0, 40k, box GPU 0 —
  paired vs A-s0's banked npz, band 0.15; chained masked-subset
  reliance eval; `--state-dropout` landed with the pre-reg (shared
  `mask_state_item` primitive with the eval probe, p=0 bitwise-inert,
  oracles green).**
- **Results instrument banked BEFORE the data (2026-08-06 ~09:0xZ,
  box-batch pattern): `fontaine/scripts/statedrop_results.py`** — all
  three frozen reads + the E3 probe gate + the verdict assembly
  (adopt-default / hardening-lever / mechanism-inert-kill / p=0.3
  branch / falsified) encoded and oracled against the banked A-s0
  panel npz: anchors 7.7966/3.9422 + state-copy 11.7848/2.6202 +
  subset state-copy first 2.4316 all reproduce through its pooling;
  degenerate zeros; synthetic COSTS/HELPS/inert/strong known-effect
  cases; misaligned-index abort. 4 CPU tests under `check.py` (244
  green). Arm C's ~12:3x–12:4xZ boundary read is now
  zero-improvisation: defaults point at the chained eval's output
  names; pass `--probe-final` from the train log's last in-run probe.
- **CORRECTION (papers-page deep read 2026-08-07,
  [page](../papers/state-shortcut.md)): the p=0.8 zero-masking recipe
  was mis-banked** — in 2506.23944 it is the *Random Dropout
  baseline*, not the method (NADA = Wasserstein-calibrated Gaussian
  state NOISE, which beats p=0.8 masking on 6/9 tasks), and the
  paper was **withdrawn** (v2 is a withdrawal notice). Cross-paper
  consensus (ReViP masking study, GAP's dominated masking baseline,
  our own arm C +2.64): *modulate, don't amputate*. The queued
  p=0.3 screen survives on our own branch rule only; if the family
  is revisited, calibrated noise (NADA-style) and GAP-style
  gradient scaling are the literature-backed levers, and full
  amputation needs the state-free paper's enablers (relative EE
  actions + wide-FOV wrist cams) we don't have.
- **Lit check at pre-reg time (2026-08-06 08:1xZ, skim-depth — re-read
  before citing numbers):** the masking lever keeps accumulating
  neighbors: [ThinkProprio, 2602.06575](https://arxiv.org/abs/2602.06575)
  goes the OPPOSITE direction (proprioception as text tokens fused at
  the prompt input rather than late conditioning — relevant to our
  soft-state-token placement question, #11 discussion);
  [Cloak, 2606.22836](https://arxiv.org/pdf/2606.22836) masks the
  END-EFFECTOR VISUALLY for zero-shot cross-embodiment — a different
  masking axis (vision-side, not state-side) that would matter for
  the rig-transfer north star if arm C's mechanism reads clean. (Skim-depth, same pass:
  [2602.09722](https://arxiv.org/abs/2602.09722) "Rethinking VLA
  scaling" — pooling heterogeneous robot data induces negative
  transfer; selective mixture + regularization beat full pooling.
  Directionally supports judge-score-weighted sampling and the
  census's fork findings; re-read before citing numbers. The
  [data-engine survey](https://arxiv.org/abs/2604.23001) frames
  dedup/contamination checks as THE underexamined bottleneck — our
  #18.7 census is exactly this; no new action.)
- **Papers-page re-read 2026-08-07
  ([page](../papers/data-and-trunks.md)) — BOTH banked claims above
  corrected:** 2602.09722's negative transfer is −2.2 to −5.9 pts
  and **frozen-VLM-only** (unfrozen trunk ≈ stable across mixtures);
  no "selective mixture" method exists in the paper, and its
  regularization finding is the *inverse* of banked (dropout +
  curricula don't help; end-to-end on the full pool is their best).
  The 2604.23001 survey contains **zero dedup/contamination
  content** — we projected our census onto it; honest citation is
  that the field's own data survey *omits* the leakage axis our
  #18.7 census covers. #9's sampling lever keeps its motivation
  from our fork census alone.
- **VISTA continuity-screen hook CLOSED 2026-08-09
  ([results post](../posts/2026-08-09-corpus-continuity-screen.md),
  qualified null at zero GPU):** all 52,507 corpus episodes scored
  with VISTA's three-regime per-tick continuity, rig-calibrated
  (oracle-gated `corpus_continuity_screen.py`). Teleport-class tail
  = 123 episodes (0.23%), dominated by the two repos the
  [wrap census](../posts/2026-08-05-wrap-census.md) already caught
  (kevin510 wrap seam 40/40; willnorris counts-units 41/42); the 42
  genuinely new sub-300° dropout episodes (30 repos, 0.08%) sit an
  order of magnitude under the census's own effect-size kill line
  for curation arms, so no pre-reg queued. Zero overlap with LORO
  influential repos. The instrument survives as a standing intake
  filter for any future curated_v1 / new community data.
- **Arm C RESULTS (2026-08-06 ~19:0xZ,
  [post](../posts/2026-08-06-statedrop-results.md)): mechanism
  WORKED, actions PAID — adopt nothing.** Paired per-frame Δchunk vs
  A-s0 = **+2.64** [2.55, 2.74], C wins only 23.9% of frames (pooled
  C 10.5024/8.5606 vs A-s0 7.7966/3.9422) — far beyond the ±0.15
  band; the reliance read confirms the mechanism (masked-vs-intact
  gap nearly closed). Verdict branch: mechanism-works-actions-pay →
  the sanctioned follow-up is a p=0.3 screen (own branch rule; see
  the correction bullet above — *modulate, don't amputate* is the
  cross-paper consensus).

- **Owner-directed dataset survey 2026-08-09
  ([post](../posts/2026-08-09-trajectory-datasets-survey.md)) — the
  corpus-growth lever quantified:** a live hub sweep found **855
  in-scope hours** (6-dim @ 30 fps SO-family) vs our 229 — ~300 h of
  it new since the community_dataset_v3 crawl era, uncurated. Named
  shortcut: AI2's
  [MolmoAct2-SO100_101](https://huggingface.co/datasets/allenai/MolmoAct2-SO100_101-Dataset)
  curation (1,220 repos / 38k eps / ~184 h, Apache-2.0, with
  re-annotated instructions as a downloadable manifest) — diff its
  source list against our 981, port its relabels. New scope hazard:
  2026's hub volume is contaminated with sim-generated LeRobot
  uploads (one MuJoCo repo = 11k episodes), so a re-crawl needs a
  real-vs-sim provenance filter the v0 pipeline never needed.
  Cross-embodiment/UMI/sim options ranked in the post (Bridge V2
  pilot > UMI cup → FastUMI-100K > sim-as-augmentation-only). A
  corpus-delta re-crawl is the survey's #1 recommendation — needs
  its own work item + judge budget before any training touches it.

- **2026-08-09 MolmoAct2 deep dive
  ([post](../posts/2026-08-09-molmoact2-deep-dive.md)): the
  survey's #1 recommendation is now mechanized.** The
  MolmoAct2-SO100_101 release is an **annotations manifest**:
  `repo_list.json` names all 1,222 kept repos verbatim (1,660
  candidates, 438 rejected by structural → eval-style → license →
  TOPReward quality gates), plus per-repo re-annotated instruction
  parquets (Qwen3.5-27B; SO-100/101 unique instructions 707 →
  16,205). So: (a) corpus intersection with community_curated_v0
  is a set operation, not a re-crawl; (b) the instruction port
  joins directly onto our copies (verify per-repo episode counts —
  re-uploads shift indices); (c) membership in their list = a free
  external quality signal on our 229 h. Their 183.6 h is SO-100 +
  SO-101 combined vs our SO-101-only 229 h — neither is a superset.
  Owner-decision item; not queued.

- **2026-08-09 lit `0817` — the offline↔real calibration study is
  now *specified and blocked on one artifact*
  ([ArmnetBench](../papers/armnetbench.md), 2607.24481):** the farm
  measured real success rates for 7 policies × 12 SO-101 tasks
  (2,518 human-scored rollouts) but ran **zero**
  offline-metric-vs-rollout correlation itself — and the clean
  version of that study (run our probes on their evaluated
  checkpoints, correlate against their measured rates) is blocked
  because the paper claims all 84 task–policy checkpoints are
  released while the HF org has zero public models. WATCH ITEM:
  if/when the checkpoints land, this is the cheapest calibration
  read the panel programme has ever been offered. Fallback
  meanwhile: trajectory-similarity metrics on their released
  rollouts vs their labels (weaker — no policy internals).

- **Lit `0818` 2026-08-09 — the curation axis splits into a
  blocked-principled pole and a runnable-heuristic pole
  ([ATHENA](../papers/athena.md) 2606.16208 +
  [Qwen-RobotManip](../papers/qwen-robotmanip.md) 2606.17846):**
  ATHENA validates influence-function curation at π-0's 3.3B scale
  (Kronecker gradient projection + low-rank Hessian, 313× vs their
  own dense baseline; square-flow surrogate built for flow heads
  like ours) — but the score is **rollout-anchored** (R∈{1,−1} over
  eval rollouts, CUPID recipe scaled up), corpora tiny (9.3h sim /
  6.9h real vs the hooked "billion-scale"), and code unreleased
  (dead link) → parked as an "offline-ATHENA" design note gated on
  any rollout/proxy success signal. Two live warnings: their
  demo-length heuristic Oracle landed BELOW random on real tasks
  (+8pp for ATHENA over full data, 47.3% Oracle vs 50.0% random) —
  naive heuristic gates on the 229h need a sanity check; and
  cross-model transfer (π-0-scored subsets work for π-0.5) licenses
  proxy-policy scoring. Qwen-RobotManip supplies the runnable pole:
  a 5-stage state-action filter that is **fully offline** (jerk
  residuals, state-action directional-agreement with DA<0.6–0.7
  episode drop, quantile bands, FK consistency, base-frame fixes) —
  their DA check excluded **81% of RoboMIND UR episodes** as broken
  proprioception, exactly the hazard class of community SO-100/101
  data. Cheapest concrete arm: DA + jerk pass over our 229h, panel
  MAE with vs without excluded episodes. Caveat both: Qwen's
  pipeline is unablated (reference, not evidence), and their 38,100h
  is ~65% re-rendered human video (~7,800h real teleop, not "166×
  our scale").

- **Lit `0820` 2026-08-09 — three data-lever reads in one slice
  ([FACTR 2](../papers/factr2-torque-estimation.md) 2606.12406 +
  [Is Diversity All You Need](../papers/is-diversity-all-you-need.md)
  2507.06219 + [H2R emergence](../papers/human-to-robot-transfer-emergence.md)
  2512.22414):** (1) The weighted-sampling slot gets a
  literature-backed sibling — phase-weighted BC sampling by
  estimated contact proximity (FIRST: pre-contact upsampling 0.818
  vs contact-only 0.670, both torque-conditioned; the +17% headline
  bundles torque-as-observation with re-sampling and sampling-only
  is never ablated). NEXT proper needs 100 Hz motor current we
  don't log, but its own input ablation crowns Δq_d = commanded −
  measured position = `action − observation.state`, present in
  every corpus episode. Cheapest gate (zero GPU): offline contact
  segmentation from tracking-error residuals, validated against
  gripper-close commands as weak grasp labels. (2) Velocity
  multimodality costs ~15% (≈2.5× pre-train data) even on a
  diffusion action expert — "expert diversity hurts" is inferred
  from the debias gain, never operator-ablated; recipe unreleased
  but trivial (chunk time-rescale toward a canonical speed).
  Falsification chain on our corpus: zero-GPU operator-speed census
  (per-dataset |Δq| stats) → free correlation of per-dataset panel
  MAE vs velocity dispersion on banked npz → only if both read, a
  speed-normalization screen rung with probe targets transformed
  identically (velocity spread inflates our chunk-MAE floor by
  construction — an eval confound their rollout evals never face).
  Two levers repriced by the same paper: episode-sampling beat
  task-relevant curation (+0.10 despite fewer target-skill
  episodes) — warns against rig-relevance filtering; and
  single-embodiment RDT-AWB ≥ 22-embodiment RDT-OXE demotes the
  survey's Bridge V2 cross-embodiment pilot. (3) The human-video
  lever gets its gate: the one measured co-training recipe
  (π0.5+ego, 14 h pseudo-action ego video) nearly doubles
  generalization (spice 32→71) but ONLY atop diverse
  target-embodiment robot pretraining; base-VLM init gains ~zero —
  we sit at the measured no-transfer corner. Parked, not dead;
  reopening condition = an ER-class embodied trunk in our stack
  (the live er_60k is exactly this) or external evidence of
  human-video gains at ≤~250 h single-embodiment scale. Caveats
  banked: no absolute threshold units published; diversity
  confounded with hours; rig-collected pseudo-labeled demos mean
  latent-action-on-unlabeled-video is untested, not refuted.
