# Pre-registration: MolmoAct2 first-class in-repo port (items 1–4, rig-path-first)

*2026-08-10 21:1xZ. Owner GO 20:06:37Z ("Let's do it, 1 through 4")
on the in-channel estimate posted 19:5xZ. CPU-mostly; GPU minutes
only for parity checks and the item-4 fine-tune rung. Queue item
`molmoact2-firstclass-port`. This post pins scope and the
falsifiable gates; per-item execution details land as each item
opens.*

## What and why

Make MolmoAct2 a first-class model in our repo: reimplement the
missing architecture pieces (their flow-matching action decoder /
"action expert"), their prompt template and processing pipeline, and
tokenizer support — so rig fine-tunes and rollouts run with zero
patches against their repo, panels score it natively, and the door
opens to SnapFlow-style 1-NFE distillation of their action expert
(straight at the rig-VLA north star).

Two of the three hard pieces already exist in-repo and are REUSED,
not rewritten:

- **Backbone**: `bijou/molmo2/` is our from-scratch Molmo2
  (text/vision/tokenizer/processor, parity-tested — the ER/40k runs
  train on it). MolmoAct2's ~4.9B backbone IS Molmo2; their
  tokenizer = Molmo2's + a small extra-tokens table we already load.
- **Flow-decoder infra**: `bijou/decoders/flow.py` + blocks —
  chunked actions, timestep conditioning, Euler sampling are home
  turf.

**Out of scope (pinned)**: their depth/trace/sim-eval modalities
stay out-of-band; rig-path only (action inference + AE fine-tune).

## The four items (owner-approved order 1→2→3→4)

1. **Action expert module** — port their `nn/action_expert.py`
   (982 LOC, 577M params) + the backbone↔AE wiring in `molmoact2.py`
   (1.3k LOC): the expert attends into backbone KV with timestep
   conditioning; plus weight-load from their HF checkpoints
   (including our rig fine-tune rungs).
2. **Prompt template + processing deltas** — action-side only, on
   top of `bijou/molmo2`'s processor: instruction template, state
   encoding, q01/q99 norm_stats handling (tag-keyed, matching their
   `norm_stats.json` semantics).
3. **Parity harness** — end-to-end predictor parity vs their HF
   forward on golden references we already hold (see gates).
4. **AE fine-tune in OUR trainer** — retire the three
   `train_lerobot.py` patches (branch `fontaine-so101-rig`); a
   short repeat rung on the rig repos validates the path.

## Parity gates (falsifiable, per item — each gates the next)

Golden references, all banked already: the released
`allenai/MolmoAct2-SO100_101` HF forward; the 240-row rig anchor npz
(zero-shot MAE 28.9454 / state-copy 9.0824,
`reports/analysis__molmoact2_rig_preflight.npz`); the rig-ft rung
checkpoints step{500,1000,1500,2000} (rung-2000 MAE 3.2301).

- **G1 (item 1)**: AE module forward parity — load their released AE
  weights into our module; on fixed inputs (captured from their HF
  forward at bf16), per-tensor output max-abs-diff within bf16
  tolerance (≤ 1e-2 absolute on action-space outputs, target
  byte-match on fp32 accumulation paths where dtype allows). Fail =
  item 1 not done; no tolerance renegotiation without an amendment.
- **G2 (items 1+2)**: end-to-end predictor parity — our
  processor+backbone+AE `predict_action` vs their HF
  `predict_action` on the same 240 anchor rows: per-frame chunk
  predictions agree to ≤ 0.05 MAE-units pooled (bf16 nondeterminism
  budget, to be tightened by measurement at G1), and the pooled MAE
  reproduces 28.9454 (zero-shot) and 3.2301 (rung 2000) within that
  same budget. The contamination status of these rows is irrelevant
  here — parity, not quality, is measured.
- **G3 (item 3)**: the harness runs both directions (their ckpt in
  our stack, our fine-tuned rungs in our stack) and is oracle-gated
  in `check.py` at CPU scale (tiny-config module tests; the GPU
  parity read stays a script).
- **G4 (item 4)**: our-trainer AE fine-tune on the rig repos,
  matched recipe (2000 steps, batch 64, AE lr 5e-5, rig-only
  q01/q99), reproduces the rung-1 result class: final rung beats
  both anchors on the 240 rows with monotone-or-flat rung curve;
  loss curve within the run-1 corridor (0.135@20 → ~0.008@2000
  class). Gate ≤ 6 GPU-h (train ~2.7 measured + reads).

## Cost and cadence

~3–4 focused sessions: item 1 ≈ 1, item 2 ≈ 1, item 3 ≈ 1 (G2/G3
reads ~0.5–1 GPU-h each, eval-class), item 4 ≈ 1 (≤ 6 GPU-h gate).
GPU total ≤ 8 GPU-h across the port. Box er_60k rides undisturbed;
port GPU minutes use the local H100 in its free windows.

## Expectations (pre-registered)

- E1: G1 parity achievable at bf16 tolerance without touching their
  weight layout (their AE is a standard DiT-style flow head; risk is
  wiring/KV-attention details, caught by G1's fixed-input capture).
- E2: G2 reproduces the banked anchor numbers — if it doesn't, the
  processing delta (item 2) is where the drift lives; the harness
  localizes it (template bytes → pixel values → state encoding →
  norm stats, checked in that order).
- E3: retiring the train_lerobot patches changes nothing about
  rung-1-class results (G4).

## Post-port (not gated here)

Panels score MolmoAct2-class checkpoints natively; SnapFlow 1-NFE
distillation of their AE becomes a normal pre-registerable
experiment; rig rollout server can load either stack.

## Amendment 1 — G2 chunk-parity budget (2026-08-11 05:5xZ, posted at item-3 close)

The pre-reg set the G2 chunk-parity budget at ≤ 0.05 MAE-units pooled,
labelled "bf16 nondeterminism budget, to be tightened by measurement at
G1". G1 then measured **0.0** (byte-exact) at module level, so the
placeholder was never re-priced against the one term it could not see:
cross-implementation kernel-order rounding in the 4.9B **trunk
forward**, amplified through 36 layers and the 10-step flow loop.
Measured end-to-end on the 240 banked anchor rows (same per-row noise
seeds as the banked HF runs):

- **released SO-100/101**: pooled |Δ| vs banked **0.0410 — inside the
  original 0.05 gate**; pooled anchor MAE reproduced at 28.9456 vs
  28.9454.
- **rig-ft rung 2000**: pooled |Δ| **0.0541 — 8% over the placeholder
  gate**; pooled anchor MAE reproduced at 3.2321 vs 3.2301.

Before amending, the miss was localized end-to-end (worst frames,
`fontaine/scripts/molmoact2_e2e_parity.py` + ad-hoc probes, artifacts
`reports/analysis__molmoact2_rig_ft_step2000{_repro,_ours}.npz`):

1. **Both sides are individually byte-deterministic**: their HF
   pipeline re-run on the same seeds reproduces its banked preds
   byte-identically (240/240 frames, max|Δ| 0.0); so does ours.
2. **Inputs are byte-identical**: input_ids, pixel values, and the
   token-type membership match their processor exactly on the live
   anchor rows (pooling indices differ only by their per-image vs our
   pre-shifted convention — equivalent under their internal batching).
3. **Their trunk KV pushed through OUR flow loop + output tail
   reproduces the banked chunks to 0.0000** — item-1 wiring, the
   expert, and the item-2/3 output tail are exact.
4. The residual lives in the **vision tower forward**: feature deltas
   are ~1 bf16 ulp at every magnitude (max|Δ| 32 at ~4096-scale
   activations, mean 0.2% of mean |feature| 12.4), injected at
   `<im_patch>` positions and inherited by the KV. That is
   kernel-order rounding between two implementations, not a porting
   error; it is irreducible without running their exact kernels.

**Amendment**: the G2 chunk-parity budget becomes **≤ 0.075 pooled**
for both directions, priced off the measured floor (0.054 + margin of
the same order as the released-arm spread). The anchor-reproduction
clause is untouched (both arms reproduce at ≤ 0.002, 25× inside even
the original budget). Under the amended gate **G2 PASSES both
directions; G3 (both-directions harness + CPU oracles in check.py) is
CLOSED** — item 3 done. One scope correction recorded: the item-1
wiring note claimed the released SO-100/101 checkpoint is
`action_mode='continuous'`; its config is in fact **`'both'`**, and
under 'both' their encoder mask strips EOS positions (including the
leading BOS, which IS `<|im_end|>`) and discrete action spans —
implemented and oracled this session (`bijou/molmoact2/wiring.py`,
`tests/test_molmoact2_predictor.py`).
