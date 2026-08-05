# Pre-registration (draft): few-shot rig-transfer benchmark v0 — ideas #16, the north star

*2026-08-05 ~21:2xZ real-clock. Status: **DRAFT — binding on design,
two slots open** (init selection + noise floor, both filled by a
short finalization amendment after tonight's box reads land; charter
§4: the finalized post precedes any launch). Posted now so the design
is frozen before the numbers that fill the slots are seen — the
slots are selection *rules*, not choices deferred until after
peeking. **Amendment 1 below (same day): holdout draw mechanism +
all pre-launch instruments landed and certified.***

## Amendment 1 (2026-08-05 ~21:5xZ, before any training): holdout draw mechanism + instruments landed

**One mechanism change, posted before any model number was seen.**
The draft specified the 12-episode holdout as a bespoke uniform draw
(`numpy SeedSequence(16)`). Implementing the leakage gate revealed
that mechanism cannot feed `bijou.eval.leakage`: the checker
recomputes the radioactive set from the plan header's
`(holdout_episodes, split_seed)` through the codebase-native
`holdout_episodes()` split, and a plan whose episodes are not that
split's holdout side fails the checker's own self-check (by design —
that assert is #18.8's anti-drift tripwire). Fix: the holdout is now
the **native split at `fraction 0.212, split_seed 16`**, which lands
exactly on the pre-registered counts by per-repo rounding —
round(.212·50)=11 of `so101_pick_place_v2` + round(.212·7)=1 of
`so101_pick_place_clean` = **12 held out / 45 train**, unchanged. The
concrete episodes: v2 {1, 2, 3, 6, 11, 15, 20, 24, 25, 30, 41} +
clean {2}, frozen in `plans/rig_fewshot_v0_k4l2.json` (48 core + 24
labeled frames, k4l2). The nested-subset shuffle keeps the draft's
`SeedSequence(16)` verbatim. Everything else in this pre-reg is
untouched; the two slots stay open for the finalization amendment.

**Instruments landed with this amendment** (all CPU, oracle-gated):

- `fontaine/scripts/rig_fewshot_plan.py` — the frozen plan; frame
  draws go through `bijou.eval.plan.build_plan` itself (per-episode
  draw is pure, so filtering the full-corpus plan to the holdout IS
  the holdout plan — zero draw reimplementation).
- `fontaine/scripts/rig_fewshot_materialize.py` — the three nested
  derived corpora under `~/datasets/rig_fewshot_v0/`:
  **n10 = 6,223 / n25 = 15,881 / n45 = 29,107 frames** (n10 drew all
  ten from v2; n25 = 22 v2 + 3 clean; n45 = all 45). Parquet filtered
  + renumbered (contiguous episodes, positional metadata, offsets
  recomputed, `judgments.json` episode-remapped); videos hardlinked
  whole so pixels are **bit-identical** to source (no re-encode);
  `stats.json` recomputed exactly from the kept rows for the
  normalization-critical features. Verification in-run: per-episode
  bitwise action/state vs disk, pointer-target existence, and a
  full-set stats oracle vs both source repos' shipped stats
  (worst |Δ| 1.2e-4).
- **Leakage certs: all three subsets PASSED** through the #18.8
  provenance path (radioactive = exactly the 12 plan episodes;
  negative control with a doctored provenance FAILS loud). Loader
  smoke: lerobot opens the derived sets and decodes shifted
  mid-file episodes bit-identical to source on both cameras.
- **Hygiene gate 1 (wrap census) PASSED**: zero wrap jumps in either
  rig repo, action and state, all six dims (recording era is
  lerobot ≥0.6.0 — the calibration bug's fix — unlike kevin510).

Remaining before launch: launcher generation + the finalization
amendment (slots 1–2) after tonight's box reads.

## Question

Owner north star (2026-08-05 17:20–17:23Z): *"build a VLA for my
rig… prove transfer so you can fine-tune a task on a new SO101 arm
with tens of examples."* This benchmark is that proof's instrument:
**the sample-efficiency curve MAE(N) for N ∈ {0, 10, 25, 45} rig
episodes is the product metric.** Community-panel MAE stays the
proxy; this is the first measurement of the thing itself.

## Data and splits (fixed here)

The two owner rig repos, both fps 30 / LeRobot v3.0:
`so101_pick_place_v2` (50 eps, 32,679 frames) +
`so101_pick_place_clean` (7 eps, 3,399 frames) — 57 episodes total.

- **Benchmark holdout: 12 episodes** (~21%), drawn once, uniformly
  across both repos, with `numpy SeedSequence(16)`; the episode list
  ships in the plan file and never changes. Rationale: the charter
  flags the owner's 0.1/seed-0 (~6-ep) holdout as too coarse for
  headline claims; 12 doubles it while leaving 45 train episodes.
  Still coarse — every claim is quoted with the measured σ_ft
  (below), never bare.
- **Train subsets: nested** N10 ⊂ N25 ⊂ N45 from the 45 non-holdout
  episodes, one shuffle at the same seed (nesting makes the curve
  monotone-comparable; a fresh draw per N would confound curve shape
  with subset luck).
- **Mechanism:** `bijou.train` has no exact-N flag — the subsets are
  **materialized derived corpora** `so101_fewshot_n{10,25,45}` with
  `meta/source_provenance.json`, each passed through the leakage
  checker before training (the #18.8-hardened identity/provenance
  path — this benchmark is exactly the derived-corpus consumer that
  work unblocked). Instrument work item: a subset materializer script
  + the plan file builder (CPU, next work session).

## Eligibility rule (contamination gate)

**The init checkpoint's pretrain corpus must certifiably exclude both
rig repos** (leakage check on the pretrain corpus vs the task repos,
run and cited in the finalization amendment). Consequences today:

- `bijou_arb_rcond_100k` and all four box 40k arms qualify
  (curated_v0 has no rig sets — the owner's `run_ft_rig.sh` header
  states the fresh-domain fact for rcond-100k explicitly).
- **flow-80k does NOT qualify**: the owner's `run_ft_rig_flow.sh`
  header records the rig data in its pretrain mix from step 0. Any
  flow-lineage subject needs a rig-excluded retrain first — out of
  scope for v0, noted for the trunk-swap round (ideas #17).

## Arms — rung 1, the curve (6 GPU jobs, ~1 evening on 1×H100)

Fine-tune **[SLOT-1: init checkpoint]** with the owner-precedent
protocol, constants copied from `run_ft_rig.sh` (2026-08-04):
decoder-lr 1e-5 / backbone-text-lr 1e-5, grad-clip 10, warmup 500,
field-dropout 0.2, instruction-augment 0.5, camera-kind-dropout 0.1,
4,000 steps, eval+save every 200, seed 0. Venue: **1×H100, B10**
(eff-batch 10 vs the owner's 4×B10=40; all benchmark arms share the
topology, so within-benchmark comparisons are clean and no
cross-topology claim is made — charter §2).

| arm | train corpus | seed | note |
|-----|--------------|------|------|
| N0 | — (no ft) | — | zero-shot eval of the init |
| N10 | `so101_fewshot_n10` (~6.4k frames) | 0 | ~6 frame-epochs — memorization arc expected |
| N25 | `so101_fewshot_n25` (~16k frames) | 0 | |
| N25-s1, N25-s2 | same | 1, 2 | **σ_ft replicates** |
| N45 | `so101_fewshot_n45` (~28k frames) | 0 | ~1.4 frame-epochs |

**Checkpoint selection is part of the protocol:** each arm's score is
its **best holdout checkpoint** at the 200-step eval cadence (the
owner's ft arc — min at 250–1000 steps, memorize after — makes
final-step scoring wrong by construction). Selection on the
benchmark holdout itself is acceptable because every arm gets the
identical rule; σ_ft absorbs the selection noise.

## Metrics (fixed here)

Scored on the 12-episode holdout via a panel-style plan file
(`plans/rig_fewshot_v0_k4l2.json`, same k4l2 semantics as the
community panel), `--dump-predictions` always on:

- **Co-primary: chunk_mae AND first-4 pooled MAE** (k=4 per the
  [flow-vs-AR paired analysis](2026-08-05-flow-vs-ar-paired.md) —
  the deployment ranking flips across k, so the pre-reg fixes k
  rather than letting the read pick it; chunk_mae is kept for
  method-comparability with panel practice).
- **state-copy floor** on the same plan, quoted next to every arm.
- Per-step horizon curves from the npz dumps (secondary, for the
  crossover read).
- New frame set ⇒ **new ledger section** (`rig-fewshot-v0`); no
  numeric comparison to any community-panel number, ever.

## Decision rules (fixed here)

- **"Transfer proven at N"** iff best-checkpoint MAE(N) beats BOTH
  (a) zero-shot MAE(N0) by > 3·σ_ft and (b) the state-copy floor,
  on the co-primary metrics (both must clear).
- **Falsification (from ideas #16):** if N45 fails (a), transfer is
  not proven and the *pretraining recipe* — not the ft protocol — is
  the suspect; the result reweights ideas #17 (trunk swaps) above
  further ft-protocol work.
- **Monotonicity:** MAE(10) ≥ MAE(25) ≥ MAE(45) expected within
  noise; an inversion > 3·σ_ft is a surprise → journal + investigate
  before any further rung.
- σ_ft = stddev of {N25, N25-s1, N25-s2} best-checkpoint scores. If
  σ_ft > 0.5 (the owner's "±0.5 noisy" holdout experience), the
  headline claim degrades honestly to "the instrument cannot resolve
  the curve at this holdout size" and v1 redraws with more holdout
  episodes at the expense of N45.

## Open slots (filled by finalization amendment, not by peeking)

1. **SLOT-1 — init + aux recipe.** Rule: the project-best *eligible*
   checkpoint on the community panel at finalization time. Today that
   is `bijou_arb_rcond_100k/step_100000` (panel 5.8026). The box
   A-vs-B read decides only the ft *recipe*: aux-off within the E5
   noise floor ⇒ ft keeps the recipe-as-is aux fields (owner
   precedent); aux-off better beyond the floor ⇒ an aux-off ft
   variant becomes a rung-2 arm (v0 design unchanged).
2. **SLOT-2 — expected noise scale.** The E5 seed-noise floor from
   the box replicates is quoted in the finalization as the expected
   σ scale; the benchmark's own claims rest on the in-benchmark σ_ft
   only.

## Rung 2 (conditional, separate pre-reg)

Only if rung 1 proves transfer at any N: protocol ablation at N=25 —
**LoRA r=32 + full vision-encoder ft** vs the owner full-ft protocol
(arXiv:2607.10172: LoRA saturates at r=32; frozen vision degrades —
and the VRAM headroom converts to batch for exactly these
fine-tunes). Not designed further here.

## Hygiene gates (all CPU, all before any training)

1. **Wrap census** (`probes/probe_wrap_census.py`) on both rig repos
   — the kevin510 precedent says a systemically wrap-corrupted repo
   poisons a curve silently; gate: same 0.1%-of-frames line as #14.
2. **Leakage check** (#18.8 path) on all three derived subsets AND
   the init's pretrain corpus vs the task repos (the eligibility
   gate).
3. **Sign-convention stage-1 screen** (#13 instrument) on the rig
   npz after the first eval burst — diagnostic, not a gate, but a
   flagged (repo, dim) cell annotates the affected arm's read.

## Cost & schedule

5 fine-tunes × 4k steps ≈ 30–45 min each at the box's measured
~0.38–0.40 s/step + 6 small eval bursts (12-ep holdout ≪ the 25.8k
panel) ⇒ **≈ one evening on one H100**. Runs at the first quiet GPU
boundary after the box batch's reads + results post (charter: never
co-locate with live training). CPU prep (materializer, plan file,
hygiene probes, launcher generation) fits GPU-busy work sessions —
queued as the follow-on work items.

## What would make this pre-reg wrong

Known design risks, accepted deliberately: 12 episodes is still a
coarse instrument (mitigated: σ_ft measured in-benchmark, honest
degrade rule above); best-checkpoint selection optimizes on the
scoring holdout (mitigated: identical rule per arm, σ_ft absorbs
it); nested subsets mean subset-composition luck is shared across
the curve rather than averaged out (accepted: the curve's *shape*
is the deliverable, and nesting is what makes the shape meaningful
at this budget).
