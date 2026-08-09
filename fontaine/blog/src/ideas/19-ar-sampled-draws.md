# 19. AR sampled-draws eval (mean-of-samples) — `screening` (AR-100k draws10_t1 READ OUT 2026-08-07, all expectations met; tsens rung next; molmo2 arm waits on its endpoint)

*Tag: `ar-draws` · idea #19 · [index](../ideas.md)*

Greedy decode is the AR family's single-draw voice; the flow family's
deployment read is mean-of-10 draws. The owner's fairness point
(2026-08-06 19:15Z): when we quote flow mean-of-N, the AR models get
temperature-sampled draws-N + mean-of-samples too — both sides get
the same instrument or neither.

**Status 2026-08-07.** Instrument landed + gated 2026-08-06
(`78c9f56`: `--ar-temperature T --sample-draws N`, Gumbel-max over
the grammar-masked softmax, action block only — aux value lines stay
greedy; draws share one prefill via by-reference cache
snapshot/restore; `stable_sample_rng` keying domain-separated from
flow noise; `_drawsN_tT` provenance). Pre-registered the same night
([pre-reg](../posts/2026-08-06-prereg-ar-sampled-draws.md), T=1.0
pinned untuned, frozen reads Δ_AR vs 5.8026 / fairness vs −1.258 /
family vs 5.365, q4 cost fallback). **AR-100k arm running now**
(`draws10_t1`, local GPU, boundary ~13Z 2026-08-07 → frozen reads).
**Frozen-read script landed 2026-08-07 ~07:5xZ**
(`draws10_t1_results.py`): the pre-reg's reads 1–5 as one command,
oracle-gated on every branch before any draws data existed —
degenerate self-pair → exact zeros, synthetic ×0.95/×1.005/×1.05/
×0.75/×0.90 error effects land on the E1+E2 / null / E4-falsifier /
E2-not-met / E3-overtake branches with magnitude checks, and 11
hard-abort guards (state-copy byte-match, ar_temperature/plan/count
semantics, `_draws10_t1` provenance, checkpoint pairing, report
reproduction |d| < 5e-3). The q4 cost-fallback path is coded (index
join + re-pool, flagged subset_mode) and the molmo2 arm reuses the
same command with explicit paths at its endpoint.
**Molmo2 arm oracle-complete 2026-08-07 ~04:3xZ**: the pre-reg's
mechanics were oracle-pinned on the gemma trunk only while the
molmo2 arm runs the shared suffix decode over a different cache
(`Molmo2KVCache`) — `tests/test_molmo2_ar_sampling.py` now pins the
trunk-specific halves on the molmo2 fixture (T→0 greedy recovery,
draw determinism/distinctness, snapshot/restore prefill-sharing
bit-exactness, the append-only-cache contract directly, the
`ar_predict_sampled` dispatch). **Launcher prep landed 2026-08-07
~04:4xZ (`6c3cc3b`)**: `eval_box_molmo2_endpoint_draws10_t1.sh` makes
the endpoint read one command — greedy arm re-run only if the training
launcher's chained eval didn't land, then the draws arm 4-GPU sharded,
with the pre-registered first-~200-frames cost gate MECHANIZED
(`draws_rate_gate.py`, 10 oracles: rank-0-shard parsing, GPU-h
projection, strict >24 threshold, timeout-with-partial-progress still
decides) and the q4 fallback kill+relaunch automated; babysit.toml
entry prepared (commented, fill `started_utc` at launch). Remaining:
execute at the endpoint (~2026-08-08).

**Lit-sourced escalation rung (banked 2026-08-07 ~04:4xZ, NOT
pre-registered).** If the frozen mean-of-draws reads land small,
selection-over-draws is the named next rung, two flavors from the
radar: (a) MG-Select ([2510.05681](https://arxiv.org/abs/2510.05681))
— verifier-free BEST-of-N: pick the draw whose action-token
distribution maximizes KL(conditional ‖ condition-MASKED reference),
where the reference needs a model trained with condition dropout —
**AR-100k already is** (state dropout 0.5, subgoal dropout 0.5) and
`--mask-state` already computes the masked context, so this is a
zero-training read over a `--dump-draws` npz + per-draw logit
retention; (b) VLA-ATTC
([2605.01194](https://arxiv.org/abs/2605.01194)) — a trained relative
action critic ranks candidates (pairwise), with uncertainty-gated
test-time compute; the trained-critic alternative if (a)'s
verifier-free score is noise. Both papers frame greedy as the
precision bottleneck — the OPPOSITE of our expectation 2
(greedy ≈ posterior mean); the draws10 primary read adjudicates
between these two priors on our own panel before any escalation.

- **Lit slice 2026-08-07 ~06:1xZ:** a THIRD selection flavor with a
  scaling claim — CoVer
  ([2602.12281](https://arxiv.org/abs/2602.12281)): scaling test-time
  VERIFICATION beats scaling policy pre-training for VLA-instruction
  alignment (contrastive verifier over rephrased-instruction ×
  candidate-action pairs; +22% ID / +13% OOD SIMPLER, +45% real) —
  independent adoption evidence that the field's compute is moving to
  the selection side; RoboMonkey
  ([2506.17811](https://arxiv.org/abs/2506.17811)) is the same bet
  with a VLM verifier + majority voting. Cheapest next read named:
  **oracle best-of-10 ceiling** — per-frame best draw vs ground truth
  from a per-draw stack bounds what ANY selector (verifier-free or
  trained) could buy on our panel before we build one. Retention
  gap found and fixed: the live AR-100k draws10 run dumps only the
  pooled predictions (per-draw reads there need a re-run — accepted,
  mean-of-samples is the registered read), but the molmo2 endpoint
  draws launcher now carries `--dump-draws` (added pre-launch,
  data-retention only, ~310 MB) so its selection-rung reads come free
  from the ~08-08 compute. The ceiling read is now a queue item
  (2026-08-07 ~06:5xZ, `idea19-selection-ceiling-read-script`, CPU:
  audit `draws_fairness.py`'s existing best-of-N against the AR
  draws-npz contract first, extend only the delta, oracle-gate on
  synthetic per-draw fixtures; exploratory read, NOT pre-registered
  — escalation to any actual selector needs its own pre-reg).

- **Ceiling read script LANDED 2026-08-07 ~08:1xZ**
  (`selection_ceiling_results.py`): the audit found
  `draws_fairness.py`'s best-of-N is flow-probe-hardwired (panel
  joins + flow anchors), so the delta is a standalone sibling — the
  exact order-statistic best-of-K ladder for K = 1..10 (sorted
  per-frame draw MAEs weighted C(N−i, K−1)/C(N, K); no Monte Carlo),
  greedy/ensemble headroom with a paired CI on the oracle gain,
  first_mae mirrors, and selector diagnostics (argmin-draw
  uniformity, dispersion-conditioned gain quartiles — where a
  selector would buy most). Oracle PASS pre-data: ladder ==
  brute-force enumeration over all K-subsets; degenerate draws=1
  reproduces the 5.8026/2.1431 anchor through the ceiling path;
  planted best-draw pattern recovered exactly with magnitude checks;
  5 abort guards (sample_draws mismatch, non-extending policy,
  draws=1 real mode, misaligned index, pooled-npz mean drift).
  Defaults = the endpoint launcher's exact `_draws.npz` stems; runs
  the moment the ~08-08 dump lands. Follow-on queued
  (`idea19-endpoint-fairness-es-read`): the energy-score delta —
  the strictly-proper-scoring-rule AR-vs-flow comparison from the
  same npz, record-only.

- **Launcher landed 2026-08-07 ~08:3xZ** (`0cb8cf8`,
  `eval_ar100k_tsens_q4_draws10.sh`): the pre-registered RECORD-ONLY
  T-sensitivity rung is one command — 3 sequential q4 rungs
  T ∈ {0.5, 0.7, 1.3}, draws 10, with the pre-reg's "run ONLY if the
  primary lands inside the gate" clause mechanized (full-panel report
  + registered semantics + elapsed GPU-h from the babysit
  `started_utc` ≤ 24.0; five abort branches oracle-checked).
  `--dump-draws` retention per the endpoint precedent, so
  dispersion-vs-T and the per-T ceiling come free later. Follow-on
  queued (`idea19-tsens-dt-read`): the dT table — a T-parameterized
  sibling loader (the frozen-read script hard-pins T = 1.0 by
  design), no decision branches.

- **dT-table read script LANDED 2026-08-07 ~10:0xZ**
  (`tsens_dt_results.py`): the T-parameterized sibling loader the
  tsens launcher's follow-up named — registered T set
  {0.5, 0.7, 1.0, 1.3} only, one record-only dT table (pooled
  chunk/first per T on the same frozen q4 rows; the T=1.0 row
  re-pooled from the full-panel primary npz via the `join_rows`
  subset join), NO decision branches per the pre-reg sensitivity
  clause. Oracle PASS pre-data: a synthetic T=1.0 rung fixture
  reproduces the primary's q4 re-pool EXACTLY; ×0.93/×0.98/×1.07
  rungs land at exactly factor × re-pool; 11 guard aborts
  (unregistered T, wrong plan/draws/ar_temperature, policy+stem tag
  mismatch, rung-row disagreement, full-panel-as-rung, state-copy
  drift, checkpoint mismatch, report drift). Defaults = the
  launcher's exact stems; the read is one command once the rungs
  land (which gate on the primary landing inside 24 GPU-h).

- **Energy-score read script LANDED 2026-08-07 ~09:0xZ**
  (`energy_score_results.py`): the strictly-proper-scoring-rule
  AR-vs-flow comparison from banked data — endpoint draws ES vs the
  paired greedy arm as the degenerate N=1 baseline (interaction term
  zero by definition; ES gain + paired per-frame CI), plus the
  flow-side comparison via index-join to the banked drawsprobe_s7
  stack (2,458 rows × 10 draws): both families get the SAME
  instrument on identical frames — N-draw ES, matched truth terms,
  paired per-frame ES delta. Audit honored: mean/best/dispersion stay
  in `selection_ceiling_results.py`; this file is ES only
  (`draws_fairness.energy_score` reused verbatim). Oracle PASS
  pre-data: degenerate draws=1 → interaction exactly 0 + ES == direct
  RMS-L2 (< 1e-12); the banked read-4 numbers
  (5.930763/9.882476/3.951713/8.769585) reproduced EXACTLY through
  this file's own join + pooling; N=2 hand fixture exact; residual
  ×3 homogeneity; 5 abort guards. Defaults = the endpoint launcher's
  exact stems; the tsens q4 dumps run via explicit paths
  (extending-policy guard is T-agnostic).

- **Lit slice 2026-08-07 ~08:3xZ — a SIXTH selection flavor, the
  cheapest trained one:** What Frozen VLAs Already Know About Success
  ([2605.28527](https://arxiv.org/abs/2605.28527)) — LINEAR probes on
  frozen VLA features (OpenVLA, pi0.5) recover value-like success
  structure their imitation objective never asked for (~92% pairwise
  success-ordering on LIBERO-Goal, beating progress/time-to-go/
  proprioception baselines), and the probe used as a selector over
  sampled action prefixes lifts push-plate success 26.7% → 44.3%
  (gains not universal, costs inference compute). Slots between
  MG-Select (verifier-free) and VLA-ATTC/CoVer (trained critics): a
  one-linear-layer trained selector over representations we already
  compute. Same gate as flavors 1–5: the oracle best-of-10 ceiling
  read decides if ANY of this is worth building on our panel. Also
  independent evidence for the #6/#17 prior that frozen trunks carry
  more task structure than their action head uses. **Papers-page
  re-read 2026-08-07 ([page](../papers/test-time-selection.md))
  caveat the hook missed: the 26.7→44.3 selector result is NOT
  probe-only — each candidate prefix is rolled out in the simulator
  from a snapshot before the probe breaks ties (probe adds value on
  top of rollout screening; probing R²/pairwise numbers stand
  clean).**
- **draws10_t1 READOUT (2026-08-07 ~12:2xZ, frozen reads via
  `draws10_t1_results.py` →
  `reports/analysis__draws10_t1_ar100k_k4l2.json`): ALL THREE
  PRE-REGISTERED EXPECTATIONS MET.** E1: Δ_AR (draws10 − greedy) =
  **−0.14505**, CI95 [−0.182, −0.109], excludes zero. E2: |Δ_AR| is
  ~9× smaller than the flow draws gain 1.258 — the pre-registered
  mean-collapse shape (greedy AR decode already sits near the
  predictive mean). E3: draws10_t1 5.6515 does not overtake the flow
  draws10 band 5.365. Falsifier (Δ_AR > +0.1) NOT tripped; oracles
  clean (row pairing full byte-match, both report arms reproduced
  |d| < 5e-3). Cost ~12.7 GPU-h — inside the 24 GPU-h gate by ~2×,
  so the q4 fallback stays closed. Next: the T-sensitivity q4 rung
  (`eval_ar100k_tsens_q4_draws10.sh`, record-only), then the molmo2
  arm at its endpoint.
- **Lit slice 2026-08-07 ([decode-temperature
  page](../papers/decode-temperature.md), 5 sources): a directional
  prior for the dT read, written down BEFORE the rungs land** — the
  multimodal-failure analysis (2605.22493: deterministic beats every
  generative variant on near-unimodal tasks; coverage ≠ success) plus
  MARS (2605.29766: stochasticity pays only in genuinely diverse
  phases) predict a near-flat dT table with mild asymmetry against
  T=1.3 on our unimodal-dominated panel. Also banked: BOKBO
  (2605.30660) measures policy-internal confidence correlating poorly
  with violations — the SECOND independent strike (after the rollout
  caveat) that any post-ceiling selector needs trained scoring, not a
  free confidence readout; and 2603.20538 gives the q-token + CE
  trunk its sample-complexity-optimality citation. Hook parked, not
  armed: DDVLA's temperature *schedule* (decay 1→0 beats both fixed
  extremes, 97.4 vs 96.4/96.2 LIBERO-Goal) is an unexplored axis in
  AR-VLA decoding — nothing opens unless the dT table shows real
  sensitivity.
- **T-SENSITIVITY dT TABLE BANKED (2026-08-07 23:09Z, record-only
  per the pre-reg sensitivity clause —
  `reports/analysis__tsens_dt_ar100k_q4.json` via
  `tsens_dt_results.py`, all guards green, T=1.0 re-pooled from the
  full-panel primary npz onto the same 4301 q4 rows):** chunk MAE
  6.5004 / 6.5668 / 6.7812 / 7.1843 at T = 0.5 / 0.7 / 1.0 / 1.3
  (dChunk −0.2808 / −0.2144 / 0 / +0.4032; first-MAE mirrors
  −0.1710 / −0.1357 / 0 / +0.3654). Against the banked
  decode-temperature prior ("near-flat with mild asymmetry against
  T=1.3"): the T=1.3 asymmetry CONFIRMED (+0.40, the largest entry);
  the low side is not flat but mildly **monotone toward T=0.5** —
  mean-of-10 at cooler temperatures sits closer to the greedy mean,
  the same mean-collapse shape as the primary read (draws add noise
  that averaging removes; cooling adds less to remove). Total spread
  ~0.68 on a 6.78 base over T ∈ [0.5, 1.3]. Per pre-reg this is
  quoted as a dT diagnostic, never a headline, never a license to
  re-pick T — **primary stays T=1.0**. DDVLA's temperature-schedule
  hook stays parked: the table is its recorded input, and the
  primary read already bounds draws-at-any-T as a small-effect axis
  for this AR family. Rungs cost ~7.2 GPU-h ≤ 12 gate (t0.5 ~2.4 h,
  t0.7 ~2.3 h, t1.3 ~2.4 h). Follow-up read item still open:
  T-guard delta via the q4 subset join. Local GPU confirmed free
  23:09Z (transient unit exited, 0 MiB) — the selfsubgoal probe
  window is open.
- **2026-08-09 — seventh selection flavor, with the first direct
  evidence on WHY external critics fail best-of-K ([ForesightFlow
  page](../papers/foresightflow-self-scored-bestofk.md),
  2606.04968, deep-read same session the sweep banked it):** the
  same flow network generates a per-step success-potential track
  beside each action chunk (~1K extra params); best-of-K picks the
  highest-mean track. **The load-bearing table is the K-sweep**:
  IDQL's separate 500M critic ranks the policy's own candidates at
  chance (39.0 → 38.4 SR from K=1→5) while the jointly-generated
  scorer climbs +5.0 — selector *shape* beats selector *size*, and
  that's the third strike on post-hoc probe selectors against our
  banked best-of-10 ceiling. Training = decoupled
  advantage-weighted FM (weights on action velocities only —
  coupled weighting lets an overconfident scorer mask its own
  corrective gradient). Carried as a directional prior, not a
  plan: needs stage labels + mixed-quality rollouts, which the
  offline panel doesn't have. Bonus instrument: 1-NFE endpoint
  preview ranks candidates at Kendall τ ≈ 0.83 vs full
  integration (~97% of the selection gain) — score-before-
  integrate is available to any future selector rung.
- **Production dT sighting (2026-08-09,
  [Qwen-VLA page](../papers/qwen-vla-early-fusion.md)):** RL
  rollouts sample at τ=1.0, deployment sharpens to **τ=0.6** — a
  production stack independently landing on the cool side, the
  same direction as our record-only dT table's mild monotone
  toward T=0.5. Banked beside the table as a sighting, not
  evidence (their mechanism is exploration-vs-exploitation, not
  mean-collapse); primary stays T=1.0 per pre-reg.
- **2026-08-09 ~14:3xZ — trained-critic pole placed and parked
  ([Robot Critics page](../papers/robot-critics-small-stuff.md),
  2606.21572):** fine-tuned VLM critic (pairwise success/failure
  supervision from policy rollouts) + action-conditioned video
  outcome model → +11% real / +5.9% sim. Consistent with RoVer and
  the free-scorer arc: learning the judge is what makes judging
  work. Parked at its stated price — needs rollout labels (#16)
  and a video model, and our ceiling reads cap the payoff on our
  decodes (AR small, flow ~null). No rank change.

- **External prior on the head's trained shape (2026-08-09,
  [FAN page](../papers/fan-feasible-action-neighborhood.md),
  2604.01570): the training-side push toward exactly the
  unimodal-around-the-mode distribution our decode reads measured.**
  FAN regularizes a discrete-token VLA head toward a Gaussian bump
  centered on its own argmax (CVPR26; OOD gains +5–6 pts on
  OpenVLA/-OFT). Read for #19: a head trained this way should
  *widen* the greedy-vs-sampled gap (more mass adjacent to the
  mode → sampling averages over near-mode neighbors), i.e. an
  external vote that mean-collapse is a property of well-trained
  manipulation heads, not a defect of ours. Record-only; no rank
  change, family decodes stand.

- **Lit `0816` 2026-08-09 — the cheap-draws hook died on
  verification ([ActionCache page](../papers/actioncache.md),
  2607.06370):** the banked "changes #19's cheap-draws cost model"
  claim is corrected — ActionCache's retrieval is top-1 and returns
  ONE deterministic cached chunk (it *collapses* the draw
  distribution rather than amortizing N draws), and it accelerates
  only the flow head while the trunk — our dominant cost — runs
  every tick to produce the cache key. Our draws economics are
  unchanged (the 1191 ms 10-draw figure is the AR decoder anyway).
  A top-k retrieval variant would be a cheap-diverse-draws
  mechanism, but that is our extrapolation, not the paper.
  Record-only; family decodes stand.

- **2026-08-09 lit `0817` — the draws cost model splits cleanly,
  and the mean-collapse asymmetry gets a weak external rhyme
  ([Reflex](../papers/reflex.md) 2607.14695 + [Compression
  Gap](../papers/compression-gap.md) 2604.03191):** Reflex's
  timestep-invariance observation (trunk KV exactly valid across
  all ODE steps — true of our frozen trunk by construction) means
  K draws share ONE trunk prefill: marginal draw cost = expert-only
  FLOPs. Combined with ActionCache's per-decision anchor (~102 ms,
  VLM ≈ 22 ms), draw economics are better than the 0816 correction
  alone implied — the fixed trunk cost is per-decision, not
  per-draw. Compression Gap files as *consistent-with*, never
  *predicted-by*: continuous heads convert encoder upgrades to
  +21–26 pts where an 80-bit FSQ codebook passes +4–10 — but tiny
  non-VLA single-seed models, frozen encoders, mechanism asserted
  not measured, and our AR head's ~1,800-bit budget plausibly
  escapes the bound entirely; with a WEAK encoder discrete wins by
  17 pts. Watch note for the adamc k4l2 readout: flow-vs-AR
  divergence under unfrozen vision may cite it as a rhyme only.
