# DRAFT — structure for the field-conditioning + subgoal meta-report

*Drafting cell for queue item `fieldcond-subgoal-meta-report` (owner
steering 08-08 13:21Z). Written 2026-08-08 19:4xZ while the seating
arm decoded; composition proper opens after `fieldgen-accuracy-eval`
closes (post-60k-chain) so the fields-panel numbers land in §3.
This file is the skeleton + artifact map — every number cited below
is already banked; slots marked ⏳ wait on a named run. NOT in
SUMMARY.md; delete when the report page lands.*

**Title candidate:** "Conditioning on words: what the subgoal
channel actually buys" (constraint: must not say "visual report";
charts are default treatment, not the headline).

**Narrative arc** (one sentence): the aux text head is load-bearing
(+0.462), the subgoal *slot* is worth −0.29 when fed truth, the
model can't yet feed itself (−0.018 at 3× cost), the failure is
located (single-frame phase estimation, not the channel, not
aliasing), and the open ladder (draws selection b′ → learned
verifier / history conditioning) is priced with published numbers.

---

## §1 — The channel is load-bearing: aux attribution (root, #6)

- Claim: aux-off costs **+0.462** [0.387, 0.537] panel chunk MAE
  (7.5× replicate threshold); σ_seed 0.038; arm B first_mae twist
  hands off to #11 (one sentence + link, not retold here).
- Artifacts: `analysis__box_batch_40k_k4l2.json`,
  posts/2026-08-06-box-batch-results.md.
- Chart: small dumbbell/bar — arms A-s0/s1/s2 vs B (new render, dark
  theme, from the banked json).

## §2 — What the slot is worth when fed truth: rung (a)

- Claims: Δ_oracle **−0.290** [−0.331, −0.225], **6× late-horizon**
  (last-10 −0.480 vs first-10 −0.081); Δ_self **−0.018** (zero at 3×
  cost); channel read narr−self **+0.043** → generation, not
  channel.
- Artifacts: `analysis__selfsubgoal_ar100k_k4l2.json` (has 50-step
  horizon curves per arm), posts/2026-08-08-selfsubgoal-results.md.
- Charts: (a) per-arm horizon curves (oracle/self/narr/baseline) —
  THE central figure of the report; (b) Δ by horizon decile bar.

## §3 — What the model knows about fields: accuracy-by-field

- Banked half (AR-100k): holding acc 0.807, progress MAE 0.062,
  event presence 0.878, visible slot-set 0.319; narration pass
  costs +0.054 chunk MAE. The 2f4d575 bug note (molmo2 aux silently
  fieldless) belongs here as an integrity aside.
- ⏳ molmo2@60k fields panel (opens at the 60k chain close) — the
  side-by-side table AR-100k vs molmo2-60k is the section's chart
  (grouped bar per field + the two narration-cost deltas).
- Artifacts: posts/2026-08-08-prereg-accuracy-by-field.md (banked
  numbers inline), `eval__..._panel_k4l2.html` field block;
  ⏳ fieldgen-accuracy-eval outputs.

## §4 — The ambiguous frames the owner asked for (frame-mining)

- The mined pairs ARE the owner's requested exemplars: near-identical
  embeddings, divergent continuations (start-vs-end aliasing etc.).
  Reuse pair_01/07 + 2–3 more from `img/framemining/pair_NN.png`
  (already in the eval-report per-joint layout) with tightened
  captions; contact sheet linked, not inlined.
- The honest twist (do not bury): the subgoal gain does NOT
  concentrate there — flagged-vs-rest Δ_oracle −0.003
  [−0.205, +0.176], ρ −0.01. Story: uniform prior/guidance, not
  disambiguation rescue. But the instrument is real: alias score ↔
  baseline error ρ 0.41, flagged frames +29% baseline MAE.
- Per-frame subgoal-delta vs alias-score chart: reuse
  `concentration_deciles.svg` (already dark).
- Artifacts: `analysis__framemining_ar100k_k4l2.json`, flagged npz,
  posts/2026-08-08-framemining-aliased-frames.md,
  papers/observation-aliasing.md.

## §5 — Closing the gap: the selection ladder (b → b′ → above)

- Rung (b): closed at table cost — 11.5% T=1 truncation derailment,
  bar (a) 20/60; Δ_bon/Δ_ceil never measured. One table excerpt
  (2–3 rows: a clean draw, a truncated derailment) as a figure.
- Rung (b′): eligible-list rule; stage 1 free (0/60 pick changes,
  bars pass, gate OPEN). ✅ LANDED 08-09: E6 FALSIFIED → NO-SCORER —
  Δ_bon +0.142 [+0.027,+0.260] (SC ANTI-selects, worse than bare
  baseline), bon−self +0.210 CI clear; ceiling ALIVE Δ_ceil −0.250
  [−0.353,−0.148], late-horizon −0.464 (the rung-(a) slot
  signature); width structurally fine (eligible 8.06/9, 0
  fallback). The delta chart already exists
  (img/fieldcond/subgoal_cleandraws_deltas.svg) — embed as the
  section figure. Results post + analysis json banked
  (2026-08-09-subgoal-draws-cleanlist-results.md,
  analysis__subgoal_draws_cleanlist_q4_ar100k_k4l2.json).
- Escalation map, priced from the lit pages (one compact table):
  scorer-is-the-gap → RoVer-style 40M chunk-scored PRM
  (papers/rover-learned-verifier.md); phase-is-the-gap →
  history/logit probes (papers/progress-from-logits.md); width →
  VINE K=4 peak; verifier-noise framing from ELASTIC
  (papers/elastic-adaptive-compute.md).
- Artifacts: `analysis__subgoal_draws_stage1_table.json`,
  `analysis__subgoal_draws_cleanlist_stage1.json`, both pre-reg
  posts, stage-1 close post.

## §6 — Open questions (each pre-named, none pre-judged)

- Subgoal-swap sensitivity read (wrong-episode subgoal, fixed
  frame): closes the presence(−0.29) / channel(+0.043) / CONTENT
  triangle for ~1 panel pass — flagged in ideas/06 as the
  meta-report candidate; needs its own pre-reg.
- ✅ (b′) routed NO-SCORER 08-09 → the escalation choice is now
  RoVer-style supervised (4,298 in-domain picked-vs-oracle pairs
  dumped by the run itself) vs set-joint label-free
  (papers/label-free-selection-signals.md: uPRM batch-joint
  principle + the masked-conditioning sketch); physics-side
  (jerkpick) already priced OUT alone (8% of the gap, banked
  08-08). Each needs its own pre-reg.
- History-conditioned phase estimation (TOPReward-style prefix
  logit) as the rung above — cite, don't run.

---

**Chart inventory** (`fieldcond_meta_report_charts.py`, banked
jsons only): §1 arms bar + §2 horizon curves/delta panel RENDERED
08-08 20:1xZ → `img/fieldcond/aux_arms.svg`,
`img/fieldcond/selfsubgoal_horizon.svg` (dark theme, eyeballed).
Still to render: §3 fields grouped bar (⏳ needs 60k half).
**Reused as-is**: §4 pair figures + deciles, §5 table excerpt
(typeset, not screenshot).
**Page location**: posts/ (dated), linked from ideas #6 and the
papers cross-refs; index hook in ideas.md.
