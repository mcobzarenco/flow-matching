# Pre-registration: state-reliance probe (masked-state panel subset)

*2026-08-06 ~03:1xZ. Immutable once posted. Ideas
[#11](../ideas.md) rung (a) — the cheapest falsification of the
state-dominant-bias mechanism named by the 02:5xZ lit slice
([ReViP](https://arxiv.org/abs/2601.16667); the causal-confusion
line [2506.23944](https://arxiv.org/abs/2506.23944),
[2509.18644](https://arxiv.org/abs/2509.18644): proprioception is
the shortcut, vision is what generalizes). Instrument landed and
gated this session; execution queued for the first quiet GPU
window.*

## Question

How much of each banked policy's panel performance rides on the
proprioceptive state input — and does aux supervision change that
reliance? The mechanism under test says aux-off models lean harder
on the state shortcut; it is a candidate explanation for BOTH the
standing grounding gap (first_mae barely ahead of state-copy) and
B's pending flag (aux-off first_mae 3.5009 WORSE than state-copy
2.6202 @40k, paired reads land ~04Z).

## Instrument (landed this session, `check.py` 221 green)

`bijou.eval --mask-state`: the bijou policy's items are rebuilt with
`observation.state := state_mean` (per-dataset), so the normalized
soft state token collates to EXACTLY zero — zero state information
at in-distribution magnitude, prompt structure untouched. Applied in
`BijouPolicy.apply_overrides`, so the narrated pass (if any) sees
identical inputs. The policy name gains `_state-masked` (the
`_drawsN` precedent: a diagnostic read must never pass as a
deployment read); report JSON, npz scalars and the report banner all
record `mask_state`. Baselines are deliberately NOT masked —
state-copy stays the intact-state reference — and truth actions are
untouched. Tests: exactly-zero collation, at-mean identity,
no-mutation (`tests/test_mask_state.py`), parse guards
(`--mask-state` without `--checkpoint`, or with `--smolvla`, dies at
the parser).

## Design — 4 masked subset evals, zero intact evals

**Subset (frozen artifact):**
`plans/holdout_curated_v0_k4l2_stateprobe_q4.json` — every 4th core
entry (positions ≡ 0 mod 4) of the frozen k4l2 plan, 4,301 rows,
labeled panel dropped (headline MAE only). sha256
`876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5`,
builder + oracle `fontaine/scripts/state_probe_subset_plan.py`. A
strict row-subset of the frozen plan ⇒ every banked full-panel npz
pools intact-side numbers over exactly these rows (the panel-v2
re-pooling precedent) — the masked runs are the only GPU work.

**Arms (one masked subset eval each):**

| arm | checkpoint | decode |
|---|---|---|
| AR-100k | `~/checkpoints/bijou-checkpoints/bijou_arb_rcond_100k_ddp4/step_100000` | greedy AR |
| flow-80k | `outputs/train/bijou_flow_artrunk_h1024_40k_ddp2/step_080000` | heun-30, draws 1, seed 0, noise-key index |
| A-s0 (aux-on) | `~/boxsync/outputs/fontaine_arb_rcond_40k_1xh100/step_040000` | greedy AR |
| B (aux-off) | `~/boxsync/outputs/fontaine_arb_rcond_auxoff_40k_1xh100/step_040000` | greedy AR |

Each run: `--sample-plan <subset> --mask-state --dump-predictions
--output-json`, panel corpus/filters verbatim from the parent plan
(holdout 0.1, split-seed 0, fps 30, camera-counts 1 2). The flow
arm's noise is identical to the banked heun-30 npz by construction:
noise-key `index` keys to corpus-relative frame indices at unchanged
corpus composition, same seed 0 — masked vs intact differ in the
state token ONLY.

**Intact side (pooled, no GPU):** per-row chunk/first MAE from the
banked npzs — AR-100k + flow-80k in `reports/`, B pulled 02:09Z,
A-s0 from its ~04Z panel npz. **The probe is blocked until A-s0's
npz is pulled local.**

## Frozen reads

Per checkpoint, per column (chunk_mae, first_mae): **Δ = masked −
intact**, paired per-row over the 4,301 subset rows, seeded
bootstrap 95% CI (seed 0, 10,000 resamples).

- **Primary: D = Δ_first(B) − Δ_first(A-s0).** The
  state-dominant-bias hypothesis predicts D > 0 (aux-off leans
  harder on state). *Supported* iff the bootstrap CI on D excludes 0
  AND D ≥ 0.05 first_mae degrees; anything smaller is *not
  supported* regardless of significance.
- Secondary: the same difference on chunk_mae; per-checkpoint
  absolute reliance (all four Δs with CIs); AR-100k vs flow-80k
  reliance compared; masked-model levels vs intact state-copy /
  state-copy-norm pooled on the subset — does vision alone still
  beat the trivial baselines?

**Execution oracles (abort on failure):** (1) each masked run's
state-copy and state-copy-norm summaries must reproduce the
pooled-from-banked values on the subset rows exactly — proves row
pairing AND that masking touched only the bijou policy; (2) report
JSON records `mask_state: true` and the policy name carries
`_state-masked`; (3) subset plan sha256 matches the frozen value.

## Stated limitation

Full masking is out-of-distribution — training never masked state,
so each Δ conflates "information lost" with "input novelty". The
PRIMARY read subtracts the common effect: B and A-s0 share corpus,
recipe, seed and architecture, differing in aux supervision only, so
D isolates the aux-linked component. Absolute Δs are quoted as
descriptive, not causal. If D is ambiguous, a shuffle-control rung
(state permuted across rows — in-distribution marginal) is the named
follow-up, not silently added.

## Branch rules

- **Supported** ⇒ ideas #9's state-DROPOUT train-time arm is
  promoted to its own pre-reg (the paired intervention; the
  literature's lever); ReViP-style modulation stays the heavier
  architecture arm behind it.
- **Not supported** ⇒ state-dominant bias is dropped as the
  explanation for B's flag; the grounding gap keeps its other
  candidate mechanisms (re-anchor, acuity — #11 main line).

## Numbered expectations (banked before data)

1. Every checkpoint degrades under masking: Δ_chunk > 0.5 on all
   four arms (state is a first-order input everywhere) —
   confidence high.
2. Masked first_mae lands above intact state-copy's subset first_mae
   for every arm (nothing fully substitutes proprioception at the
   chunk's first frame) — confidence medium.
3. D > 0 — this is the hypothesis under test, not a prediction we'd
   bet the arm on; the probe exists because either sign is
   informative.

## Cost & scheduling

4 × ~25 min (4,301 frames at ~170/min) ≈ **1.7 GPU-h** total + CPU
pooling. Venue: first quiet GPU window (local after the draws
chain + fairness probe, or box between the control evals and the
E4B launch) — never co-located with a pre-registered eval (charter
§3). Blocked on: A-s0 panel npz (~04Z). Panel convention: v1 subset
(this artifact); if the panel-v2 amendment is approved before
execution, the v2∩subset column is quoted alongside (CPU-only
re-pool, same npzs).
