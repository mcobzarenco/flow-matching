# 9. Data levers — `screening` (state-dropout arm C ANSWERED 2026-08-06: COSTS, adopt nothing; p=0.3 screen is the sanctioned follow-up)

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
