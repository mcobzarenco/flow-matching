# Bijou code deep-dive — ranked findings

*2026-08-05, work session ~18:50Z→. Owner ask (16:17Z): a deep review
of the bijou codebase. Deliverable: this ranked list.*

## Scope and method

The review covers all 57 Python files (~22.3k lines) of `bijou/`,
split into six subsystems, each read line-by-line by a dedicated
reviewer, findings then cross-checked against the code by me before
ranking:

1. **Training loop + model assembly** — `train.py`, `model.py`, `nn.py`
2. **Data pipeline** — `data.py`, `loading.py`, `aux_text.py`,
   `annotations.py`
3. **Decoders + FAST tokenizer** — `decoders/*`, `fast/*`
4. **Eval instrument** — `eval/*`
5. **Gemma-4 trunk** — `gemma4/*`, `encoders/*`
6. **Rollout + judge** — `rollout*.py`, `interface.py`, `judge/*`

Ranking criteria, in order: (P0) silent-wrongness risks — anything
that could corrupt a loss, a metric, or a paired comparison without
announcing itself; (P1) measurement-integrity risks; (P2) performance
on the 1×H100 boxes; (P3) structural headroom for the north star
(few-shot rig transfer) and hygiene. Style nits were out of scope.
`docs/architecture.md` was used as the intended-design reference —
a finding has to disagree with the code, not just with the doc.

Verification status is marked per finding: **[verified]** = I
reproduced the claim against the code myself this session;
**[reviewer]** = reported with file:line evidence by the subsystem
reviewer and consistent with everything I checked, not independently
re-derived. One reviewer finding was refuted on verification and is
recorded at the bottom — the base rate for that is why the marks
exist.

## Headline

**No P0 (silent-wrongness-in-current-numbers) finding survived
verification.** The measurement core held up under adversarial
reading: pooling math is exactly as documented (valid-element
weighted), the holdout split is a pure function of its flags, the
FAST round-trip is inverse-exact with conservative clipping, the Heun
solver is a correct explicit trapezoid matching the π0 convention,
the trunk is bitwise-parity-anchored against HF (eager/H100), and the
seeding chains make training and eval draws pure functions of their
flags. The current box batch and draws chain are not invalidated by
anything below.

What the review did find is a layer of **contract gaps at the
instrument's edges** — places where the fail-loud philosophy has
quiet holes that don't corrupt today's numbers but will corrupt
*some future* number silently when a precondition shifts (corpus
composition, a resume, a conditioned flow run, the first rig
deployment). Ranked by leverage:

## Tier 1 — measurement integrity (fix before they bite)

**1. Flow-policy eval noise is keyed to the corpus-relative concat
index — sealed plans pin frame identity but NOT noise identity.**
[verified] `bijou/eval/policies.py:288-295` seeds each frame's draw as
`sample_noise(seed + index)` where `index` is the global concatenated
index recomputed from the *current* corpus (`eval/plan.py:301`).
Adding/removing/growing any dataset in the eval data dir shifts the
offsets of everything after it: the plan still resolves, the same
frames score — but every flow draw changes. Across-noise-draw std is
~5.9°, orders of magnitude above the 1e-4 anchor bands, and
state-copy stays bitwise-identical, making the drift look like model
change. Flow anchors are therefore only valid at frozen corpus
composition. Fix: seed from the stable `(seed, repo_id, episode,
frame)` triple — a one-time, versioned instrument break that needs a
posted amendment with re-banked flow anchors.

**2. Resume semantics: three quiet traps around `--resume`.**
[verified] (a) Nothing restores the data-stream position: resume sets
`epoch = 0` with the same-seed shuffle (`train.py:2451-2467`), so a
same-`--seed` resume replays exactly the batches and τ/ε draws
already trained — the team's "fresh seed on resume" convention is
real but unenforced by code. (b) Live-backbone resume is not the
"lossless continuation" the comment claims: fp32 masters round-trip
through the bf16 `backbone.safetensors` snapshot
(`loading.py:816-833`), and `optimizer.pt` holds Adam *moments*, not
master weights — every resume boundary snaps the backbone to the
bf16 grid, discarding exactly the sub-bf16-resolution updates that
are the stated reason fp32 masters exist (`train.py:1975-1979`).
(c) A changed `--backbone-*-lr` on resume is silently ignored
(optimizer state restores all groups; the advisory note checks group
0 only, `train.py:2307-2319`). All three land directly on idea #3
(longer-training extension runs).

**3. The Q3 conditioning-collapse tripwire cannot fire for flow
decoders.** [verified] The pre-registered alarm ("mean |Δprediction|
≈ 0 means the label is ignored") compares the outcome-overridden
decode against the scalar pass's predictions, but passes the
*advanced* generator (`train.py:818-823`) — fresh noise. For a flow
decoder mean|Δ| has a floor at the sampling variance even if the
model is completely conditioning-blind, which is precisely the state
the tripwire was registered to catch. Exact for ar_backbone (greedy).
Fix: reuse the scalar pass's noise per item.

**4. `--aux-prompt-hash` pins training but not measurement.**
[verified] The pin reaches the train-side `select_datasets`
(`train.py:1765`) but neither the in-run probe selection
(`train.py:2360`) nor offline eval (`eval/cli.py:421`) — a pinned run
whose stamp mismatches trains a dataset as unjudged (kinds render
`unknown`, no labels) while probe/eval render full tags and
conditioning for the same dataset. Train and instrument disagree on
the prompt distribution, silently. Fix: one kwarg at two call sites.

**5. The eval report artifact does not pin scoring semantics.**
[verified] The JSON records data/split/filters/seed/plan but omits
`--sample-steps`, `--sample-method`, `--generate`, `--exclude`,
`--condition-override`, batch size, and world size
(`eval/cli.py:1004-1041`); `--condition-override` appears in **no**
artifact and doesn't rename the policy (unlike `_drawsN`), so a Q3
counterfactual JSON is indistinguishable from a deployment read.
Sharding docs say exact reproduction needs (seed, world_size,
batch_size); two of three are unrecorded. An eval is not reproducible
— for condition-override, not even interpretable — from its report
alone.

**6. Three silent holes in the fail-loud instrument.** [reviewer;
bounds-check hole verified] (a) `resolve_plan` never bounds-checks
`frame_index` (`eval/plan.py:289-301`): a truncated/re-encoded
episode maps planned frames into the *next* episode (or dataset) and
scores them without error. (b) The leakage checker's same-repo-id
branch maps episodes identically with no count/content check
(`eval/leakage.py:191-196`): a filtered-and-renumbered corpus keeping
its repo id gets a false PASS while radioactive panel content trains.
(c) `metrics.py`'s `max(divisor, 1)` guards turn an
impossible-today zero-valid frame into a perfect 0.0 score rather
than a NaN — combined with (a), mis-addressed frames could *lower*
MAE. Each fix is an assert.

**7. Trunk parity has a blind spot exactly where production lives.**
[reviewer] `verify_parity.py` compares against HF only at batch-1,
unpadded, full-depth, single-image; the production encode is
left-padded batches, logical positions, 15-layer truncation with
sliced PLE, `kv_stop_layer`, state-token splice, multi-camera. Those
equivalences are pinned by self-consistency tests only — a joint
semantic divergence from HF's padded-multimodal conventions would be
invisible. Window-boundary crossing is behind an off-by-default flag,
and the bitwise-on-eager anchor is printed but not enforced
(tolerance 2.0 on ±30 softcapped logits gates). No known divergence;
this is the largest open *coverage* hole given how much rests on the
parity claim. One padded, batched, 2-camera HF comparison closes most
of it.

## Tier 2 — the rig path (north star surface)

**8. The hardware rollout path has no absolute safety clamp.**
[verified default; lerobot branch per reviewer] `--max-relative-target`
defaults to `None` (`rollout.py:184-190`) and is the *only* limiter:
bijou sends raw predicted positions, and the vendored lerobot degrees
branch un-normalizes with no min/max against calibration
(`motors_bus.py:900-903`). A wrong `--stats-repo-id` or one bad chunk
commands arbitrary servo ticks at full speed on a default invocation.
Also un-cross-checked: the degrees convention itself (no
first-observation sanity gate against the stats). Before the first
physical run: refuse to start without a clamp + assert the first
observation lies within the stats envelope.

**9. Rollout camera kinds diverge from training when judged kinds ≠
operator names.** [reviewer] Training orders and tags cameras by
*judge-voted* kind from `meta/camera_kinds.json`; rollout derives
kind from the operator's `--camera` name (`rollout.py:239-256`). The
judge evidence code itself documents the wild case (a fixed overhead
cam named "front" voted kind "top"): such a rig fine-tunes with tag
`top` but rolls out with tag `front` — silent conditioning skew on
exactly the few-shot-transfer surface. Fix: read the rig dataset's
own `camera_kinds.json` at rollout (or `--camera-kind name=kind`).

**10. Community text is an unhardened injection surface that reaches
training prompts.** [reviewer] Episode task strings interpolate
verbatim into judge prompts with no untrusted-data framing
(`judge/claude.py:81-84`), and judge *outputs* feed back as training
inputs (suggested-instruction augmentation, subgoal conditioning). A
hostile task string can tilt curation and seed prompt text. Images
got anonymization for exactly this reason; text deserves the same.

## Tier 3 — performance (both known levers, now concretized)

**11. Idea #2 (prefix compile/bucketing) blocker map.** [reviewer]
Concrete compile blockers on the 79% path: data-dependent
`pooled[valid_mask]` (vision.py:606), host-syncing `bool(...all())` /
`int(...sum())` + `masked_scatter` (masks.py:132, model.py:196-204),
Python KVCache `torch.cat` mutation, dense additive `[B,1,S,S]`
masks. And a standing tax: **no prefix attention ever takes the flash
path** — sliding layers always carry an additive mask, global layers'
head_dim 512 exceeds the fused-kernel cap. Bucketing is a
prerequisite for compile (variable P). Also: with `retain_cache=False`
all 15 prefix layers still write K/V that only {4,9,14} consume, and
frozen-run probe evals re-encode a bit-identical prefix every eval
(~79% of probe cost is recomputing a constant — cacheable).

**12. Idea #8 (262k-vocab CE) concretized.** [reviewer]
`ar_backbone_losses` materializes a fresh `[B·S, 262144]` fp32 logits
copy (~1 GiB at B10) on top of `_patched_logits`' two full-vocab
tensors (`ar_backbone.py:743-748`); a chunked/fused linear-CE
(logsumexp against `lm_head.weight` + the 1026-row patch; the softcap
is elementwise and fuses) never materializes logits — the single
biggest VRAM lever on the ar_backbone arm. Decode-side, every
action-phase step computes all 262k logit columns to argmax over
≤1026 (grammar mask + monotone softcap ⇒ block-columns-only is
exact).

## Tier 4 — smaller integrity/hygiene items (queue fodder)

- **Normalization stats (and FAST q01/q99) are fitted on the whole
  dataset including held-out episodes** (`data.py:717-731`) —
  symmetric and small, but "held-out" claims should be worded
  accordingly (2-episode datasets: 50% of the stats data is holdout).
  [reviewer]
- **Cross-repo duplicate content is never fingerprinted**
  (`data.py:663-670` dedups exact repo ids only): community forks
  double-weight content and can place near-twins of holdout episodes
  in train. Cheap census (episode count + stats hash) before trusting
  ±0.05° holdout deltas. [reviewer]
- **Action-dim anchor is the first-discovered dataset**, even one
  later dropped by filters (`data.py:636-645`) — ordering-dependent;
  a majority-dims assert closes it. [reviewer]
- **`--dump-predictions` npz rows carry no stable frame identity**
  (concat `index` + `repo_id` only, no episode/frame columns,
  `eval/cli.py:780-791`) — paired A/B across npzs is only valid at
  byte-identical selection, and misalignment is undetectable beyond
  repo_id spot checks. Relevant to the queued flow-vs-AR per-frame
  analysis (mitigated there: same-day, same-corpus, state-copy
  summaries bitwise-equal across the two reports). [reviewer]
- **Eval-side unfetchable-frame substitution silently swaps scored
  frames** (loud on stderr, uncounted in the report). [reviewer]
- **Batched ar_backbone aux decode is batch-composition-dependent**
  (value-phase lockstep feeds batch-max terminators into row caches;
  code-documented as accepted; B=1 rollout unaffected)
  (`ar_backbone.py:487-556`); constrained-candidate forcing goes
  off-manifold the moment a candidate set has unequal token lengths
  (`:510-525`). [reviewer]
- **Judge pipeline biases**: permanent model-agnostic failure-skip
  excludes long/dense episodes from curation (3.4% truncation class,
  `sweep.py:228-261`); default materialization mixes judge models
  per-episode by `judged_at` (`materialize.py:99-115` — make
  `--model` required). [reviewer]
- **In-run probe MAE is world-size-dependent** (frame→rank→generator
  pairing, `train.py:527-528`) — 1-GPU vs 4-GPU probe curves carry an
  unbudgeted delta; registered caveat, not a fix. [reviewer]
- **No deterministic-mode toggle exists**: seeds pin draws and order,
  not kernel atomics — identical-seed replication drifts by design;
  worth knowing when a band is missed. [reviewer]
- Assorted: EVENT-negative supervision keyed to `annotation.progress`
  presence (deserves an assert); aux render caps vs decode budgets
  agree by convention only; FAST `decode` strips mid-sequence BOA
  silently; `fit_report` recon MAE excludes clip cost; audio/video
  placeholder ids embed silently as garbage rows; decoder param group
  gets blanket weight decay (norm scales included) unlike backbone
  groups; `_to_pil` floors instead of rounds. [reviewer]

## Refuted on verification

The decoders reviewer's top claim — right-padded prefixes evict real
context from sliding-window layers in batched ar_backbone decode —
is **false as stated**: the collator left-pads
(`encoders/gemma4.py:170`, `padding_side = "left"`), which is the
load-bearing, test-gated design decision (2026-08-01) that makes the
physical-index window math correct; with left padding the suffix is
physically adjacent to each row's real prefix and no real token is
evicted. The adjacent true concern (batch-composition dependence of
aux decodes) is the Tier-4 item above.

## What was checked and holds up

Load-bearing verifications across the six reviews, kept here as the
positive result they are:

- **Pooling math**: headline chunk_mae = Σ|err| over valid (step,
  motor) elements ÷ (Σ n_valid × dims), exactly as documented;
  per-dataset slices re-pool exactly; fixed accumulation order via
  index-sorted merge — consistent with observed bitwise
  reproducibility at fixed config.
- **Split determinism**: holdout is a pure function of (split_seed,
  repo_id, count, fraction), sha512-seeded, machine-independent;
  train/eval/leakage recompute identical sets from flags.
- **`--sample-draws`**: flow-only, N independent CPU-seeded draws,
  draw 0 byte-identical to the single-draw path, mean taken in raw
  degrees (affine-commutes), `_drawsN` name suffix — the running
  ensembling chain's semantics are sound.
- **Heun solver + flow objective**: correct explicit trapezoid on the
  exact-endpoint grid; τ convention, interpolant, and velocity target
  match π0/lerobot exactly; loss/predict normalization identical.
- **AR CE paths**: shift/masking/PAD-IGNORE alignment correct in both
  decoders; suffix causality holds (no future leakage); aux loss is a
  true position-weighted mean across ranks.
- **FAST round-trip**: orthonormal DCT-II with exact inverse;
  fit-clip equals encode-clip; grammar-constrained decode provably
  terminates; train and eval quantiles come from the same attached
  stats.
- **Trunk**: strict+assign weight loading (nothing can silently stay
  at init); PLE truncation slicing consistent; sliding-window/cache
  arithmetic hand-checked consistent end-to-end; 12B/MoE swaps fail
  loudly at config parse.
- **Rollout preprocessing parity**: rollout builds
  StatsAttachedDataset-shaped items through the *same* collator stack
  as training — no reimplementation seam (remaining deltas:
  camera-kind finding above; live frames skip AV1 recompression).
- **Judge integrity**: idempotent keyed sidecars, no double-count
  path, full provenance (model + prompt hash + evidence params);
  malformed verdicts are retryable failures, never clamped scores.
- **Seeding chains**: loader/worker/collator RNGs are pure functions
  of (seed, rank, worker); eval consumes no training RNG (changing
  `--eval-every` cannot perturb a paired run); grad-clip covers
  exactly the optimized set, post-allreduce.

## Consequences for the queue

Proposed new work items, in leverage order (none started — this
session was the review):

1. **Instrument-hardening pass** (CPU, small diffs, oracle-gated):
   prompt-hash kwarg to probe/eval selection; `resolve_plan` bounds
   assert; `score_frame` n_valid assert; report JSON gains
   sample_steps/method/generate/exclude/condition_override/batch/world
   fields; npz gains episode/frame columns. All additive; only the
   report-schema change touches comparability (additive keys, no
   break).
2. **Flow-noise stable-triple seeding** — a versioned instrument
   break: pre-register as sealed_v3-style amendment, re-bank flow
   anchors once, gain corpus-composition-invariant flow reads. Do at
   a natural anchor boundary (e.g. after the current box batch
   reads).
3. **Q3 noise fix** (reuse scalar-pass noise) — before the first
   conditioned flow run.
4. **Resume hardening** — enforce fresh-seed-or-warn + honest
   bf16-snap warning on live-backbone resume; required before idea
   #3 (longer training) launches.
5. **Rig-rollout safety gate** (clamp mandatory + first-obs stats
   envelope assert + rollout reads `camera_kinds.json`) — required
   before the first physical run; feeds the north-star benchmark
   (idea #16).
6. **Parity extension** — one padded/batched/2-camera HF comparison
   + `--require-bitwise` eager gate; cheap, closes finding 7.
7. **Duplicate-content census** (CPU fingerprint sweep over
   curated_v0) — before trusting fine holdout deltas.

The compile-blocker map (finding 11) folds into idea #2's
implementation notes; the chunked-CE design (finding 12) folds into
idea #8.
