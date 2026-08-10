# Pre-registration: MolmoAct2-SO100_101 out-of-band panel eval + 3-policy report

*Finalized 2026-08-10 ~13:0xZ. Owner steering 10:50Z/11:06Z, plan GO
11:59:33Z ("The molmo2act plan sounds good, let's eval the so101
checkpoint") + side-by-side report request. This finalizes the
[plan post](2026-08-10-molmoact2-oob-eval-plan.md) (deep-read
receipts live there) into an immutable pre-reg. **Record-only
throughout: nothing gates, repoints, or touches our runs.** Smoke
starts immediately per the GO; the full sweep launches after smoke
green + the in-channel objection window on this post.*

## What is being run

The released **`allenai/MolmoAct2-SO100_101`** checkpoint (bf16,
snapshot `152569fe`), scored end-to-end by **its own**
`predict_action` — their processor, squash-resize, prompt template,
q01/q99 normalization (`norm_tag=so100_so101_molmoact2`),
state-token discretization, and 10-step Euler flow expert. We adapt
nothing in the model path. Continuous mode only.

**Frames**: the exact banked `panel_curated_v0_k4l2` rows — all
25,800 (17,204 core + 8,596 labeled), identity columns copied
verbatim from the banked er_60k@15000 npz and re-verified per frame
against the live datasets (repo/episode/frame + raw action chunk,
atol 1e-6 — the frame_mining alignment oracle). Cameras passed in
the dataset's own sorted key order (their policy is camera-order
agnostic by construction); task string = the frame's LeRobot
annotation through their own normalizer; state = raw 6-dim vector.

**Determinism**: the flow expert's initial noise generator is seeded
`0 + global concat index` per frame — any subset (the smoke) is
byte-reproducible inside the full sweep; row order can never matter.
Single draw per frame (matches how their servers deploy; a
mean-of-N-draws arm would need its own pre-reg).

**Instruments** (landed before this post, oracles green):

- `fontaine/scripts/molmoact2_panel_predict.py` — predictor;
  npz contract `pred:molmoact2-so100@release` (25800, 50, 6), steps
  0–29 filled (their native 30-step = 1.0 s horizon), steps 30–49
  NaN; state-copy rows copied verbatim.
- `fontaine/scripts/molmoact2_panel_reads.py` — matched-window
  frozen reads (below); planted-delta `--oracle` + all abort
  branches green.

## Frozen reads (all matched-window: chunk steps 0–29 = 1.0 s)

Both sides pooled over steps 0–29 only (our banked npzs re-pooled —
pure CPU; later chunk steps are strictly harder, so full-50 vs
full-30 would be unfair in either direction). Each read lands
**× {pooled 17,204 / clean 11,872-core / contaminated 5,332-core}**
frames — the contamination split is AllenAI's own
`SO100_SO101_MOLMOACT2` fine-tune list intersected with our panel,
**pinned at 245/878 repos, 7,996/25,800 frames, 5,332/17,204 core
frames** (re-derived live at read time from their file; any drift is
a hard abort, not a silent re-split). Clean split is the honest
headline; even it carries the pre-training asterisk noted in the
plan post §5.

1. Matched-window chunk MAE + step-0 first MAE per arm: molmoact2,
   **snapflow 80k top-10-tickets** (`pred:bijou@80000_draws10_ticket`,
   full-50 banked 5.1847), **snapflow 80k stable-key**
   (`pred:bijou@80000`, 6.5997), **ar_40k endpoint** (6.0079),
   **ar_60k continuation** (5.8602), **er_60k@15000** (7.5283), and
   the **state-copy** floor.
2. Paired per-frame Δ (molmoact2 − arm), seeded bootstrap CI95
   (seed 0, 10,000 resamples), classified MOLMOACT2-BETTER /
   MOLMOACT2-WORSE / CI-SPANS-0 — per split.
3. Our arms' full-50 numbers recorded as secondary anchors, never
   quoted against their 30-step side.

Output: `reports/analysis__molmoact2_oob_panel_k4l2.json` + the
contaminated-repo list banked as
`reports/analysis__molmoact2_contamination_repos.json`.

## Execution oracles (each failure = hard abort)

- per-frame dataset↔npz alignment incl. action-chunk reproduction;
- `config.n_obs_steps == 1` at model load (the HF-config default of
  30 silently shifts chunk slicing to index 29);
- prediction shape exactly (30, 6) per frame;
- reads: identity + state-copy byte-match across all six npzs;
  window all-finite + tail all-NaN on the molmoact2 rows; every
  banked arm's full-50 re-pool reproduces its own report json
  (5e-3); contamination counts match the pin above.

## Smoke gate (before the sweep)

500 evenly-strided panel rows (deterministic; covers both camera
counts and ~hundreds of repos). Tripwires — any failure is rc≠0 and
**no sweep launches**:

- per-dim prediction range within truth range ± 1.5× span (unit /
  sign / normalization bug detector);
- smoke matched-window MAE < 3× state-copy's on the same rows
  (gross-harness-failure tripwire — explicitly NOT a model-quality
  gate; a genuinely-bad-but-sane checkpoint passes and gets
  reported as measured).

Plus a rate read → sweep wall-clock projection posted in-channel.

## Sweep + budget

Full 25,800 rows, local H100, systemd unit via `run_detached.sh`,
progress-checkpointed every 500 frames (resumable). **Gate ≤ 8
GPU-h total** (est. 2–5; smoke rate decides the projection). If the
projection at smoke rate exceeds the gate, stop and re-plan
in-channel before launching (options: CUDA-graph verification,
batch of the prefill, or an owner-approved gate raise) — no silent
overrun.

## Report (owner spec 11:59Z)

One HTML report in our standard eval-report format, same frames,
three policies side-by-side: **snapflow 80k** (banked; headline =
top-10-tickets since the owner asked for our best policy, stable-key
alongside) vs **MolmoAct2 SO100_101** vs **state-copy**. Summary
block on top: matched 30-step window primary (50-step secondary for
our arms), pooled + clean/contaminated splits, paired CI95, chunk +
first MAE. Per-frame sample gallery with camera thumbnails +
per-joint truth-vs-policies charts. Lands on the Space reports page
+ numbers in-channel.

## Decision line (frozen)

This is a reference point, not a gate. Whatever the deltas: our
runs' kill lines, the er_60k endpoint protocol, and every banked
anchor stay untouched. What it informs (as pre-named in the plan
post): whether a state-in-prompt + flow-expert arm on our own trunk
is worth pre-registering, and how far our from-scratch decoder is
from a 1,220-repo fine-tune on partially-seen data.

*Immutability: from this post on, any change to frames, seeds,
window, splits, tripwires, or read list is an amendment logged
in-channel before the affected stage runs.*
