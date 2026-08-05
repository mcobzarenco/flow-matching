# Pre-registration: E4B trunk-swap screen — matched-params AR-100k on 4×H100

*2026-08-05 ~22:4xZ. Posted before launch (charter §4). Owner pick
21:57Z: the E4B screen is the next pre-reg; the freed 4×H100
(192.222.55.210) goes here once tonight's box-batch arms + reads
land. Launch is gated on the pre-launch checklist below and on a
short finalization amendment that fills the E5 seed-noise constant
from tonight's replicate reads.*

## Question

Attribution front, question 1 (owner 21:43Z: "bigger trunk?"): does
trunk scale — Gemma 4 **E2B → E4B** (2.3B → 4.5B effective; text
decoder 35×1536 → 42×2560, ~2.2× text params) — move comm-holdout
MAE under the **identical AR recipe**? This is survey rank 1 in
ideas #17 (the zero-port-cost in-family rung): if scale at matched
compute-class doesn't pay, the front moves to grounding (#11) and
the video-trained trunks (Molmo2-4B, survey rank 2).

## Design — one run, matched parameters

The mainline E2B reference is `bijou_arb_rcond_100k_ddp4`
(docs/architecture.md § experiment reports; panel chunk_mae
**5.8026** @100k, the banked anchor). The screen re-runs its
**verbatim recipe** with exactly one science change:

- `--backbone google/gemma-4-e4b-it` (E2B default → E4B).

Verified from code this session: the `ar_backbone` path is fully
config-driven — full-depth trunk (`BackboneDepth.FULL`), FAST block
tail-anchored at `vocab_size − vocab_total` (same 262,144 vocab ⇒
same block base), no expert/stream surface involved. `bijou/gemma4/`
implements E4B (`e4b_config`, parity harness covers it).

Everything else matched to the reference: `--decoder ar_backbone`,
fast_tokenizer_v2, aux fields subgoal/holding/progress/event/visible
@ weight 0.5, `--aux-dropout 0.0 --field-dropout 0.1`, conditioning
subgoal/outcome/smoothness `--condition-dropout 0.1
--subgoal-dropout 0.5`, `--instruction-augment 0.5
--camera-kind-dropout 0.1`, `--decoder-lr 1e-4 --backbone-text-lr
2e-5 --grad-clip 100`, 100k steps, warmup 1k, **batch 12/GPU ×
DDP4 = effective 48**, workers 20, prefetch 4, `--eval-samples 256
--eval-every 500 --save-every 2500 --log-every 20`, `--seed 0
--split-seed 0 --holdout-episodes 0.1`, corpus
`community_curated_v0 @ --fps 30 --camera-counts 1 2` (box copy,
frozen; cleanup boundary still in force). Run name
`fontaine_arb_rcond_e4b_100k_ddp4`, project `fontaine`.

## The two seams, stated up front

1. **The E2B reference's own batch seam.** The reference ran eff-48
   to 20k, OOM'd (77.5 GiB at B12), and finished at B10/eff-40 —
   so E4B held at eff-48 throughout sees **~+15% samples by 100k**
   (4.80M vs 4.16M). Owner pick (21:57Z, "owner remembered 10" —
   the recipe's launch command says 12): match the *recipe*, eff-48,
   never change batch semantics mid-run. Consequence for reading
   results: gates at ≤20k are seam-free matched; post-20k the seam
   favors E4B, so a **kill** ("E4B not ahead despite ≥ samples") is
   conservative-valid, while an endpoint **adopt** carries the +15%
   caveat in the writeup.
2. **Probe corpus seam (small).** The mainline curve was measured on
   the owner's corpus copy (42,853 episodes, stamp `9b796de`); the
   box copy selects 42,872 (Δ19 episodes, 0.04%, already E1-verified
   identical across tonight's four box arms). The 256-frame in-run
   probe may therefore differ slightly in composition from the
   mainline probe. Mitigation: probe deltas are read against a ±0.5
   noise floor (observed inter-eval scatter ±0.3 late, ±0.5–0.8
   early, plus this seam); the **panel** (frozen plan
   `panel_curated_v0_k4l2`, 25.8k frames — the exact plan the 5.8026
   anchor was scored on, from this box copy, by the owner, today) is
   seam-free and is the decision instrument wherever a checkpoint
   exists.

## Memory reality and the pre-registered fallback ladder

E4B at B12/GPU will likely OOM: the E2B reference peaked 77.5/79.2
GiB at B12, and E4B's text trunk is ~2.2× the parameters (trained at
2e-5 ⇒ optimizer state scales with it). `bijou.train` today has
**no gradient accumulation** (single `loss.backward()` per step).
Pre-registered ladder, decided at the pre-launch memory smoke —
**never mid-run** (the reference's batch roulette is the lesson):

1. **B12 direct** if the smoke fits with ≥3 GiB headroom.
2. Else **chunked backward at loader batch 12**: the per-rank batch
   stays 12 (identical per-step sample composition), forward/backward
   split into equal chunks (2×6 → 3×4 → 4×3, first that fits) with
   gradient averaging — mathematically the B12 gradient up to fp
   reduction order (equal chunks ⇒ mean of chunk-means = batch
   mean), DDP `no_sync` on all but the last chunk. This is a small
   `bijou.train` change to be landed **before launch** with: the
   three CPU loss oracles bit-exact with chunking OFF, and a
   chunked-vs-unchunked gradient-equivalence test (tolerance-level,
   CPU) with chunking ON. Effective batch 48 and every LR/schedule
   constant are invariant at every rung; the chosen rung is recorded
   in the finalization amendment.
3. If even 4×3 doesn't fit: **do not launch**; post the finding
   (E4B doesn't fit this recipe on 80 GB — itself an attribution
   datum) and take the follow-on decision to the owner.

## Expectations & gates

- **E1 startup (hard gate):** selection line 878 datasets / 42,872
  episodes / dims 6/6 (box copy, identical to tonight's four arms);
  model line shows the E4B geometry (42 layers / hidden 2560) and
  decoder head sized off it; block base = 262,144 − vocab_total,
  same value as E2B. Any selection deviation ⇒ abort before step 1.
- **E2 first poll (util rule):** record s/step and peak VRAM.
  Expected 0.9–1.1 s/step at B12-equivalent (~2.2× the reference's
  0.46–0.49; chunked backward adds a little); **slowness is data,
  not a kill**. Starving util ⇒ input-pipeline fix at a safe
  boundary, logged. Wall estimate 26–31 h; the 30k decision gate
  bounds a losing run to ~9 h.
- **E3 probe curve vs the banked E2B curve** (256-frame in-run
  probe, matched cadence, ±0.5 floor). Reference points: E2B 9.43@5k,
  7.54@10k, 7.33@20k, 6.57@30k, 6.03@40k, 5.79@50k, 5.55@100k
  (best 5.29@99.5k). Pre-registered readings:
  - **@10k: record only.** No kill except divergence (probe >15
    with a falling-then-rising shape, or NaN). Bigger trunks may
    descend slower early — tonight's aux-off arm is a fresh lesson
    that early dynamics mislead.
  - **@30k: DECISION.** Kill if E4B probe > **7.07** (E2B 6.57
    + 0.5) **and** the 25k panel read (below) does not contradict
    it. At matched steps and ≥ samples, a 2.2×-text-params trunk
    showing no probe advantage by 30k means scale is not the cheap
    lever at this budget — bank the negative, free the box for
    grounding arms. If probe and panel disagree, continue to 50k.
  - **@50k: re-check.** Kill if E4B probe > 6.29 (E2B 5.79 + 0.5)
    with the same panel cross-check.
- **E4 mid-run panels (decision instrument):** checkpoints at 25k
  and 50k rsync to the local box; panel eval (k4l2 plan,
  `--dump-predictions`, 1×GPU, ~1.7 h) runs at the first quiet local
  boundary after each lands. Anchors: E2B's only panel point is
  5.8026 @100k; a probe→panel offset estimate (+0.25, from E2B's
  100k probe 5.55 vs panel 5.8026, single-pair, approximate) puts
  E2B's matched-step panel @~25–30k near **~6.8**. Readings:
  E4B@25k panel ≥ 6.9 corroborates a probe kill; E4B@50k panel
  < **5.8026** (beating E2B's *endpoint* at half the steps) is a
  strong adopt signal and gets posted immediately.
- **E5 endpoint (primary read):** E4B@100k panel chunk_mae vs
  5.8026, matched eval command (4-GPU sharded, `--dump-predictions`
  so per-frame paired analysis works). **Adopt** iff E4B beats
  5.8026 by more than max(3·σ_seed, 0.15), where σ_seed = the
  pairwise replicate panel spread from tonight's E5 noise-floor
  read (A-s0/s1/s2 @40k) — the constant is filled by the
  finalization amendment before launch, not invented here. Also
  read: first_mae (E2B 2.1431 — the grounding-sensitive column) and
  the per-repo delta distribution (coherent vs single-repo-driven).
- **E6 hygiene:** loader substitutions / value-budget fallbacks /
  cuDNN asserts counted; >2 substitutions or any assert ⇒ noted in
  the results post.

## Decision semantics (what this changes)

- **Adopt** ⇒ E4B becomes the trunk candidate: the follow-on ablation
  arm is the **image-embedding budget** on E4B (owner 21:58Z, one
  variable per rung), and stage-2 flow-expert work re-targets E4B
  (streams (5,11,17,23) — four `--stream-counts` entries; noted, out
  of scope here).
- **Kill/tie** ⇒ trunk scale is not the cheap lever; the box goes to
  #11 grounding arms and the Molmo2-4B port moves up (survey rank
  2). Either way the screen answers the owner's attribution
  question 1 with one pre-registered run.

## Pre-launch checklist (blocks launch, not this post)

1. **Box free** (all four arms + panel evals done, results post out).
2. **Checkpoint present:** `google/gemma-4-e4b-it` is **not** in the
   box HF cache (checked 22:2xZ; only e2b) — download (~16 GB).
3. **Parity spot-check on the box:** `python -m
   bijou.gemma4.verify_parity` for E4B (greedy tokens must match HF;
   the harness documents E4B ULP-tie behavior).
4. **Memory smoke:** 1×GPU, E4B, this recipe, ~50 steps at B12;
   record peak; pick the ladder rung. If rung 2: land the chunked
   backward change + oracles first (CPU work item, next GPU-busy
   window).
5. **Finalization amendment:** σ_seed from tonight's replicate
   panels, the chosen ladder rung, measured smoke peak, disk check
   (7.2T free today; 40 saves × ~35–40 GB ≈ 1.4–1.6 T fits; owner
   checkpoints untouched as ever).
6. rsync-back loop extended to the new run's log + latest two saves.

## Cost

One 4×H100 run, 26–31 h wall if it goes the distance, ~9 h if the
30k gate kills it; two 1×GPU local panel evals (~3.5 h) mid-run;
the endpoint panel (~30 min sharded). Charter §3: the run answers an
owner-ranked attribution question with a pre-registered kill that
bounds the downside.
