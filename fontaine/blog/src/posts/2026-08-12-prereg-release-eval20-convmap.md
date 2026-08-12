# Pre-reg: released MolmoAct2 in sim, off-contract `_convmap` (20 seeds, parallel)

*2026-08-12 17:3xZ — owner prio 17:13:24Z: "Could we also try running the
released checkpoint directly", with an attached box-side note on molmoact2
unit contracts (committed copy:
[`fontaine/notes/molmoact2-unit-contracts-box-note.md`](https://github.com/mcobzarenco/flow-matching/blob/fontaine/fontaine/notes/molmoact2-unit-contracts-box-note.md)).
Exploratory rough-numbers pass, not a registered claim.*

**Plain words.** The released checkpoint speaks a different unit language
than our simulator: its normalization table assumes joint angles in the
older community convention, while our sim reports controller-native v3
values that sit *below* the release table's floor. Fed raw, the model
would be effectively blind and the score would measure the unit clash,
not the policy. So we translate at the boundary — convert the sim's
state into the model's units on the way in, and the model's actions back
into sim units on the way out — and clearly label the result as an
off-contract read.

## Design (case 3 of the box note)

- **Checkpoint**: `~/marius-convert-gate/converted/molmoact2_so100_101_release`
  (already converted; backbone `allenai/MolmoAct2-SO100_101`).
- **Shim**: exact per-joint affine, both directions — state-in (v3 →
  model units *before* its q01/q99 table normalization), action-out
  (model output → v3 *before* the controller). No re-training, no table
  edits.
- **Arms**: same 20 seeds (sim100 list 0–19), fixed post-flip sim,
  parallel driver workers=8 — paired vs the step-500 and step-2000
  corrected arms (parallel-path rough rows, per the failed-oracle rule).
- **Label**: `_convmap`, off-contract — never pooled with ftrig contract
  reads; interpreted as a **lower bound** (the release trained on a
  mixture of conventions through one table; outputs are mixture-blurred
  even under a perfect shim).

## Gates / tripwires (mandatory, pre-GPU)

1. Print the release box from its norm_stats; verify the **mapped
   reachable set A⁻¹(release box) covers the sim task workspace** — the
   clamp travels with the model. Fail → report, don't run.
2. **First-action-vs-current-state check** (the note's unit-bug
   detector; the release contract read had first_mae 18.0 vs state-copy
   2.5). A correct shim collapses this to ~state-copy scale. If it does
   not, STOP — do not spend the GPU on a mismatched map.
3. GPU gate: ≤0.5 GPU-h (one 20-seed parallel arm ≈ 0.1; debug budget
   included).

## Reads

- success, mean/median `progress_final_cm`, knock-aways ≥1 cm, per-seed
  paired deltas vs step-2000 corrected and step-500 arms; videos + rows
  to fontaine-reports under `/ftrig_eval20_flip_parallel/release_convmap/`.
- Cross-check bank (the box asked): does our sim calibration imply the
  same lift +180° / elbow +90° old-convention map that
  `fit_convention_map` snapped? Disagreement → flag in-channel; one of
  the two sides has a sign/offset wrong.
