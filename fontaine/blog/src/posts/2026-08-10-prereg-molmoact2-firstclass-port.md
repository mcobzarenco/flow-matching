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
