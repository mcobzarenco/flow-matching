# Ideas

The backlog. Every entry: hypothesis, expected effect, cost, cheapest
falsification. Seeded 2026-08-05 from charter §8 (which distills the
mainline ledger, `docs/architecture.md` §7–8); ordering ≈ expected
information × cheapness. Status tags: `queued` / `screening` /
`running` / `confirmed` / `falsified` / `parked`.

## 1. Inference-time noise-draw ensembling — `queued`, natural first

- **Hypothesis:** mean-of-N noise draws through the flow expert cuts
  panel MAE substantially in the unconstrained class (mainline
  measured 5.30°→2.88° on motion frames for a ft'd model,
  mean-of-10); the stage-2 flow-on-AR-trunk lineage (6.623 panel)
  should benefit similarly.
- **Expected effect:** large on the flow lineage's panel number;
  unconstrained-class only (charter §2) until distilled.
- **Cost:** ~20 lines eval-side + one eval burst per N in {1,5,10}.
  Zero training.
- **Falsification:** paired panel eval, same checkpoint, draws
  stated. Check unimodality of draws first (averaging multi-modal
  draws is wrong): per-frame draw spread on a few hundred panel
  frames. If mean-of-10 does not beat single-draw beyond the panel's
  pairing noise, kill.
- **Open sub-question:** an AR-family analogue (temperature/nucleus
  chunk ensembles, chunk-level medians) — separate screen.
- **Instrument note (2026-08-05):** upstream already ships
  `bijou.eval --sample-draws N` (a16e65a) — verify its semantics
  (independent draws? mean-of-N in action space?) before writing any
  code; the pre-reg's eval-side work may reduce to a flag.
- **Paired-analysis prediction (2026-08-05 ~20:15Z, before the
  draws-10 numbers land):** flow's deficit vs AR is a monotone
  horizon-divergence (crossover at step 2; +1.2 by step 40 —
  [post](posts/2026-08-05-flow-vs-ar-paired.md)). If per-draw
  spread grows along the horizon, mean-of-N should close the
  *late-horizon* deficit preferentially: chunk_mae moves a lot,
  first_mae barely. Score the draws-10 run per-step, not just
  pooled.

## 2. Throughput: bucketed batching + torch.compile on the frozen prefix — `screening` (2a landed 2026-08-05; GPU A/B conditional)

- **Hypothesis:** length-bucketed batching + `torch.compile` of the
  prefix encode (79% of step time) buys ≥20% step-time on 1×H100 —
  compounding interest on every later run.
- **2a LANDED (2026-08-05, [post](posts/2026-08-05-bucketing-impl-sim.md)):**
  `--bucket-by-length` (default OFF) — `LengthBucketedBatchSampler`,
  camera-count keys, oracle-gated (3 CPU oracles bit-exact, gradflow
  green, 6 unit tests). **Sim finding: under the current recipe
  (`--camera-counts 1 2`) padding inflation is only +5.09% → ceiling
  ~3.6% step-time — below the <5% deprioritize line ⇒ NO GPU screen
  for current lineages.** Full-corpus census (3–4-cam datasets in):
  +32.55% → −23.8% padded tokens, ~19% ceiling. Conditional pre-reg
  in the post: first widened-selection run family runs the 1k-step
  A/B before adopting; paired arms must share the flag.
- **Cost remaining:** 2b (compile) — real implementation vs the
  blocker map below; decoupled from bucketing under narrow census
  (shape variance is text-jitter ⇒ pad-to-fixed-length).
- **Falsification (2b):** measured s/step and samples/s on identical
  configs, before/after, on THIS box. If <10% combined, bank the
  numbers and deprioritize.
- **Implementation notes (deep-dive 2026-08-05):** compile blockers
  on the prefix path: `pooled[valid_mask]` dynamic shape
  (vision.py:606), host syncs + `masked_scatter` (masks.py:132,
  model.py:196-204), KVCache `torch.cat` mutation, dense additive
  masks. No prefix attention takes the flash path today (sliding =
  always-masked, global head_dim 512 > fused cap). Bucketing is a
  compile prerequisite. Bonus levers: skip K/V writes for
  non-exported layers when `retain_cache=False`; cache frozen-run
  probe prefix encodes (bit-identical across evals).

## 3. Longer training on the best recipe — `queued`

- **Hypothesis:** rcond-100k was still improving at 100k
  (75k→100k bought 0.05–0.3); an extension banks a cheap win.
- **Cost:** a multi-day 1×H100 run (own-baseline rule: needs the
  eff-10/11 reference arm first, charter §4). Resume traps: fresh
  `--seed`, `--steps` = new TOTAL, cosine re-heat semantics.
- **Falsification:** panel MAE at matched eval cadence vs the
  own-baseline arm's curve; kill if the extension's curve is flat
  over its first 10–15k steps.

## 4. Stage-2 follow-ups (flow expert on AR trunk) — `queued`

Inherited questions from mainline §8.11 (banked: 6.57 in-run / 6.62
panel @80k, 2.2× smaller expert): more/deeper export streams (AR
adaptation lives in all 35 layers; the expert reads {4,9,14} —
untested headroom), expert width h512/h1536 on the better features, a
second-generation AR trunk re-measured through the stage-2 lens.
Cost: one screen-rung run per arm. Falsification: paired screens at
matched steps.

## 5. FAST tokenizer v3 — `queued`

- **Hypothesis:** refitting on curated-v0's exact quantiles removes
  the ~1.94%-of-chunks clip rate; small but real MAE effect on
  clipped chunks.
- **Cost:** CPU-only fit (~32 min measured for v2); token metrics
  RESET (never cross tokenizer versions) — coordinate with run seams.
- **Falsification:** paired arms (same seed/data/arch, only the
  artifact differs — the v1-vs-v2 precedent); recon error + clip rate
  in the fit report before any training touches it.

## 6. Aux attribution arms — `running` (paired 40k, launched 2026-08-05)

The still-owed paired aux-on vs aux-off arms (does aux supervision
shape the representation, separate from "does narrating help" — the
100k run answered only the latter). Pre-registered mainline
expectation: within probe noise (±0.3). Promoted to arm B of the
paired 40k run after the wrap census killed unwrap-at-load:
[pre-reg](posts/2026-08-05-prereg-paired-auxoff-40k.md). Primary read:
paired per-frame panel chunk_mae A@40k vs B@40k. **Executing on the
4×H100 box since 17:12Z** (parallel arms + 2 control seed replicates
for the noise floor, with a pre-registered decision rule:
[box batch pre-reg](posts/2026-08-05-prereg-box-batch-4xh100.md)).

## 7. Stream-schedule re-test — `queued`

0-0-16 vs 4-4-8 vs shallow-heavy (8-4-4) at scale: the acuity probe
(shallow stream carries sharpest position) and streams0016's rig hint
pull opposite directions — measure. Config-diff cheap per arm; enters
at the short-run screen rung.

## 8. Shortlist/output-vocab head for ar_backbone — `queued`

The 262k-vocab CE softmax is the VRAM headroom eater; a shortlist
head raises feasible batch on 1×H100 (mainline queued it as the
structural fix after the B12 OOM). Cost: real code + an equivalence
check (loss oracle moves → loud re-baseline). Payoff multiplies every
future ar_backbone run on this box. **Design concretized (deep-dive
2026-08-05):** chunked/fused linear-CE (logsumexp vs `lm_head.weight`
+ the 1026-row patch; elementwise softcap fuses) — never materialize
the `[B·S, 262k]` fp32 logits (~1 GiB at B10,
`ar_backbone.py:743-748`). Decode-side: action-phase argmax over
block columns only is exact (grammar mask + monotone softcap).

## 9. Data levers — `queued`

`--trim-leading-idle` (~6.7% of frames), state-noise augmentation,
judge-score-weighted sampling (never yet run). Each is a cheap paired
arm at the screen rung. Any derived corpus ships with the leakage
check (charter §2) before training touches it.

## 10. E2B base-vs-IT swap — `queued`

Pre-registered mainline prediction ±0.2 MAE; backbone-swap arm, tests
whether instruct tuning matters at our instruction distribution.
Verify the -pt checkpoint ships the vision tower first.

## 11. Visual grounding arms — `queued`, the open front

Re-anchor probe: error is frame-dependent level mis-estimation;
acuity probe: the text stack's use of visual tokens is the
bottleneck. Arms: trunk shaping, schedules, vision-side aux tasks —
chartered on the community panel; `first_mae` is the
grounding-sensitive column (2.143 vs copy 2.620 — headroom).
High-variance; counts toward the ≥20% exploration budget.

## 12. Solver/Heun-gap work — `queued`, re-opened by a surprise

The h1536 adaRMS Heun-gap collapse did NOT transfer to h1024-on-AR-trunk
(measured −0.28 at 10→30, first_mae −0.46): sampler quality is back on
the table for the best flow lineage. Arms: step-count sweeps, solver
variants, consistency/distillation toward 1–2-step deployment decodes
(the distillation leg pairs with idea 1).

- **Scoring note (2026-08-05 paired analysis):** flow's deficit is
  ~all late-horizon (crossover step 2, monotone to +1.2 @40 —
  [post](posts/2026-08-05-flow-vs-ar-paired.md)). Score solver arms
  per-step: a solver fixing only late-horizon costs nothing at
  first_mae; pooled-only scoring would misread it.

## 13. Sign-convention detection & repair (owner hypothesis) — `screening`

- **Hypothesis:** a small set of community repos encodes joint angles
  with flipped sign conventions (esp. wrist_roll on mirrored wrist-cam
  mounts); training on them injects contradictory supervision.
- **Status:** stage 1 (CPU screen over the panel npz) done 2026-08-05
  — 9 candidate (repo, dim) cells, three pathologies separated by
  per-frame classification; cleanest mirror lead
  kantine/domotic_dishTidyUp_anomaly wrist_flex (median frame corr
  −0.75). Instrument: `probes/probe_sign_convention_stage1.py`;
  [results post](posts/2026-08-05-sign-convention-stage1.md).
- **Next / falsification:** stage-2 optical-flow probe (flow curl vs
  wrist-velocity sign) on the mirror-signature candidates —
  pre-registered before running; catches internally-consistent
  mirror repos stage 1 structurally misses. If confirmed, the repair
  arm (flip-corrected derived corpus + leakage check) is a cheap
  paired fine-tune. Awaiting owner steer on stage-2 scope.

## 14. ±180° wraparound census & shortest-arc error — `confirmed`/banked (measured 2026-08-05)

- **Hypothesis:** truth chunks wrapping the ±180° boundary inject
  ~360°-scale discontinuities into BOTH raw-degree training targets
  and MAE; a wrap census may explain a measurable slice of panel MAE.
- **Measured** ([write-up](posts/2026-08-05-wrap-census.md),
  instrument `probes/probe_wrap_census.py`, anchors in-probe): panel —
  16/17,204 wrap frames (0.093%, under the 0.1% gate) carrying 0.0720
  of the 5.8026 pooled chunk_mae (1.24%; shortest-arc re-score
  5.7498). Corpus — 81/42,872 episodes (0.19%) across 23 repos;
  kevin510 systemically corrupted (40/40 eps), willnorris/bbox-2 a
  separate state-stream glitch. wrist_roll dominates (204 action
  jumps), matching the SO101 calibration story (lerobot#1255, PR#777,
  fixed in 0.6.0).
- **Consequences:** unwrap-at-load training arm killed (0.19% cannot
  move a 40k pair); shortest-arc metric proposal → owner sign-off
  (moves every anchor); kevin510 + willnorris/bbox-2 flagged for any
  future curated-v1 exclusion list.

## 16. Few-shot rig-transfer benchmark — `queued`, **the north star** (owner 2026-08-05 17:20–17:23Z)

- **Goal statement (owner):** "build a VLA for my rig… prove transfer
  so you can fine-tune a task on a new SO101 arm with tens of
  examples." Community-panel MAE is the proxy; **the
  sample-efficiency curve is the product metric.**
- **Design sketch (pre-reg to write after the box batch lands):**
  fine-tune the best lineage on N ∈ {10, 25, 50} episodes of a
  held-out rig task; measure panel-style MAE on that task's holdout
  (and eventually rollout success) vs N. Protocol precedent: the
  owner's ft-rig lineage (4–5k-step fine-tunes, `run_ft_rig*.sh` on
  the second box, both AR and flow variants).
- **Dependencies:** tonight's aux-off answer + seed-noise floor pick
  the trunk and set the minimum detectable effect for paired ft
  comparisons; sign/calibration hygiene (ideas #13, #14) bites
  hardest on a new arm — keep them warm.
- **Falsification:** the curve itself — if MAE at N=50 is no better
  than zero-shot, transfer is not proven and the pretraining recipe
  (not the ft protocol) is the suspect.
- Reweights the whole list: rig-transfer relevance now outranks
  community-panel micro-optimization at equal cost.
- **Metric note (2026-08-05 paired analysis):** the pre-reg must fix
  the deployment replan interval k and quote first-k pooled MAE next
  to chunk_mae — the flow-vs-AR ranking *flips* at k≤3 vs k≥5
  ([post](posts/2026-08-05-flow-vs-ar-paired.md)); chunk_mae alone
  is the most AR-favorable point on that axis.

## 17. New trunks / new architectures — standing owner mandate (2026-08-05 17:24Z)

"The current repo should just be a starting point for what I've
tried." A ranked exploration front, fed by the literature slice;
every candidate enters at the screen rung with a pre-reg and counts
toward the exploration budget.

**Ranked 2026-08-05 by the [trunk survey](posts/2026-08-05-trunk-survey.md)**
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

## 18. Instrument & infra hardening — `screening` (item 1 done 2026-08-05)

The [bijou deep-dive](posts/2026-08-05-bijou-deep-dive.md)'s fix
queue, in leverage order (details + file:line in the post):

1. ~~Hardening pass~~ **DONE 2026-08-05 ~20:55Z**
   ([post](posts/2026-08-05-hardening-pass.md)): aux-prompt-hash →
   probe/eval selection (`bijou.eval --aux-prompt-hash` new flag);
   `resolve_plan` bounds assert; `score_frame` n_valid assert;
   report JSON records full scoring semantics
   (exclude/aux_prompt_hash/sample_steps/method/draws/generate/
   condition_override/batch/world); npz gains episode_index/
   frame_index identity columns. Oracle: banked AR-100k panel
   recomputed bit-exact (12/12 cells, d=0) through the edited
   scoring path; 3 new unit tests; check.py green. NOT included:
   deep-dive finding 6b — now item 8 below.
2. Flow-noise stable-triple seeding — **implemented behind
   `--noise-key` 2026-08-05 ~21:20Z, break pre-registered**
   ([amendment](posts/2026-08-05-noise-reseed-prereg.md)): `stable`
   keys noise to blake2b(repo_id, episode, frame) via numpy
   SeedSequence (128-bit, no torch 32-bit trap, no draw stride);
   default stays `index` (byte-identical, oracle 12/12 d=0) until the
   flip executes at the first anchor boundary after the box reads —
   one flow-80k panel re-bank, decision band pre-registered off the
   draws chain's empirical σ_draw. Until then flow anchors remain
   valid only at frozen corpus composition.
3. Q3 tripwire noise fix (reuse scalar-pass noise) — before any
   conditioned flow run.
4. Resume hardening (fresh-seed enforcement + bf16-snap warning) —
   **blocks idea #3** (longer training) until done.
5. Rig-rollout safety gate (mandatory clamp, first-obs stats
   envelope assert, rollout reads `camera_kinds.json`) — **blocks
   the first physical run** (idea #16).
6. Parity extension: one padded/batched/2-camera HF comparison +
   `--require-bitwise` eager gate.
7. Duplicate-content census over curated_v0 (CPU fingerprints).
8. Leakage checker same-repo-id count/content assert (deep-dive
   finding 6b): the same-repo-id branch maps episodes identically
   with no check — a filtered-and-renumbered corpus keeping its repo
   id gets a false PASS. Cheap assert; do before any derived-corpus
   training run (ideas #9, #13 repair arm).

## 15. Literature-sourced arms — standing

The arXiv radar (VLA/robot learning, flow matching, action
tokenization, data curation) feeds this list; every borrowed idea
cites its source in the pre-registration; every "novel" idea gets a
search first. Local canon: π0, π0.5, SmolVLA, FAST
(arXiv:2501.09747).
