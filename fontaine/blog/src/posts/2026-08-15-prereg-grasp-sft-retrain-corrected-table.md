# Pre-reg (DRAFT, owner-gated) — grasp-SFT retrain on the corrected table, via bijou.train

*2026-08-15, ~12:5xZ. Drafted per the morning's owner steering
(10:07Z: their `train_lerobot.py` retired, all training via
`bijou.train`; 10:10Z: stage-C killed at step 2040, step2000 probed).
Status: **DRAFT — launch is owner-gated.** Nothing in this document
starts a GPU job; the prep artifacts (corrected norm table + base
conversion) are landed and verified CPU-side.*

**Plain words.** This morning we found that the toolkit we used to
package our robot demonstrations had a real bug: it computed the
"typical range" of each joint by averaging per-episode ranges instead
of measuring the range over all frames. For the wrist-roll joint —
which flips sign between episodes — that produced a nonsense range,
and the model trained against it spent the whole run seeing ~19% of
its frames squashed against a wrong boundary. That model still tripled
the base model's success rate (9 → 28 out of 100), which makes the
obvious next question: how much better is the *same* training with the
*correct* ranges? This document freezes that experiment. It only runs
when the owner says go.

## §1 Hypothesis

**H:** the corrupt q01/q99 clamp table cost the stage-C SFT real
performance — ~19% of training frames (and the serving-time state
stream) were clamped out of the normalization box, distorting exactly
the wrist_roll branch structure the jam-recovery demos exercise.
Retraining the same data under the exact-quantile table recovers it.
The corrupt-table read (28/100 unseen) is a **floor** on what the demo
set is worth (owner-agreed framing, 12:01Z reply, 👍).

**Confound, stated honestly:** their trainer is retired, so the
retrain moves two things at once — corrupt→corrected table AND
their-stack→bijou-stack (objective `flow` = the molmo_flow expert
pathway vs their AE recipe). A corrected-table arm on their stack is
not available by standing order; if the retrain lands *below* the
corrupt-table floor, the read is "stack/objective seam, diagnose
before pricing the table" — not "the table fix hurt".

## §2 The seam this pre-reg exists to close

`bijou.train` on molmo_flow normalizes with the **source checkpoint's
baked table**, not `--train-data` stats (train.py save-checkpoint
region). A naive `--init-from` of any existing conversion inherits the
corrupt table. Prep therefore ran first (landed this session,
CPU-only):

1. **Corrected table artifact** —
   `fontaine/scripts/build_corrected_norm_stats.py` (oracle-tested,
   5 tests) projects the FIXED dataset `meta/stats.json` (exact
   quantiles over raw frames, `rewrite_quantile_stats` fix `19b7321`)
   into the molmoact2 `norm_stats.json` convention; donor metadata
   from the released base, all 20 numeric rows replaced, provenance
   sha256-pinned. Out: `~/checkpoints/norm_stats_grasp_sft_v0_corrected`.
   Measured correction: action wrist_roll q01/q99 `[35.5, 94.4]` →
   `[-157.2, 157.2]`.
2. **Corrected-base conversion** — `bijou.convert_molmoact2 --source
   allenai/MolmoAct2-SO100_101 --norm-stats-from <artifact>` →
   `~/checkpoints/converted/molmoact2_base_corrected_stats_v0`
   (588 expert tensors, expert sha `7a2d4dea…`, `converted_from`
   records the table provenance). Verified: the baked
   `normalization` block carries the corrected rows exactly.

## §3 Frozen run params (on owner go)

```
uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_sft_demos_v0 \
  --init-from  ~/checkpoints/converted/molmoact2_base_corrected_stats_v0_vla \
  --objective  flow                # molmo_flow expert pathway, trunk frozen \
  --flow-decoder-init inherit      # warm AE = stage-C's warm-start analogue \
  --steps 2000 --decoder-lr 5e-5 --batch-size 64 \
  --save-every 500 \
  --save-dir ~/checkpoints/finetune/fontaine_grasp_sft_bijou_corrected
```

*(Command amended 17:0xZ for main phase 5a — see §8; parameters
unchanged.)*

Matched to the stage-C recipe class where the stacks correspond:
AE-only (backbone frozen — no `--backbone-text-lr`), LR 5e-5, gb64,
endpoint **2000 steps** (the probed stage-C checkpoint), checkpoints
every 500. Seed: bijou default (fresh run; cross-stack seed
comparability with stage C does not exist, house same-seed policy
n/a). Launch detached via `run_detached.sh`, babysit entry at launch,
first-poll util/rate/RAM checks per standing rules.

## §4 Eval protocol + comparators (frozen)

Two-arm probe, verbatim the step2000 protocol: `rollout_sim` euler-10,
30 s episodes, execute-horizon 30, bf16 expert; **unseen 0–99** then
**train band 1000–1099**; reads via
`grasp_sft_step2000_probe_reads.py` (kept-subset split automatic).

| comparator | unseen /100 | status |
|---|---|---|
| released base (own intact table) | **9** | **primary anchor** (owner 👍 12:0xZ) |
| stage-C step2000, corrupt table | **28** | the floor to beat |
| ftrig4k / W0 | ~1 / 2 | context rows only |

## §5 Decision surface (proposed, owner may re-steer)

- **> 28 unseen**: table fix priced positive → corrected lineage
  becomes the SFT artifact for the downstream route (token-SFT →
  token-GRPO per R2 Amendment A1; GRPO re-pricing per its own
  pre-reg). Delta upload of the endpoint same-session (standing rule).
- **~28 (±3)**: SFT quality was data-limited, not table-limited —
  next lever is more/better demos, not normalization.
- **< 25**: stack/objective seam investigation BEFORE any further
  pricing (§1 confound); no cascade decisions off the number.

## §6 Budget

Train ~2.9 GPU-h at the stage-C pace analogue (5.2 s/step × 2000;
bijou pace pinned at first poll) + probe ~2.5 GPU-h = **~5.5
expected, gate ≤ 7**. Wall-clock ~4–5 h detached.

## §7 Gating

**Owner go required — nothing launches from this document.** Prep is
landed; the launch command is frozen above. On go: pre-reg flips to
FINAL (any owner edits recorded as amendments), launch at the next
free-GPU boundary, in-channel launch post.

## §8 Amendment — main phase 5a re-verify (2026-08-15 17:0xZ)

Phase 5a (`a51b172`, "bijou.train on the family CLI + the VLA
checkpoint format") landed on main after this draft and moved two
things the frozen command touched. Merged into fontaine (`351c56e`,
check.py 922 green) and re-verified end-to-end, CPU-only:

1. **Flag rename:** `--expert-init` is gone; its successor is
   `--flow-decoder-init` (`inherit` = default = the same warm-AE
   semantics this pre-reg froze; `fresh` = adaLN-Zero init). §3
   amended verbatim, parameters unchanged.
2. **Checkpoint-format break:** the new `--init-from` refuses the
   existing conversions as legacy (`bijou_config.json`). Both were
   migrated with `bijou.convert_legacy` (hard-link metadata
   re-expression, weights bit-identical by construction) and
   `validate_checkpoint` passes:
   `molmoact2_base_corrected_stats_v0_vla` and
   `molmoact2_grasp_sft_stagec_ar_step2000_vla`. Corrected wrist_roll
   rows (±157.2) verified baked through the new-format reader.
3. **Continue-from-2k arm made real** (the Q1-reply option): fresh
   `bijou.convert_molmoact2 --source
   ~/checkpoints/molmoact2-grasp-sft-stagec-ar-step2000-hf
   --norm-stats-from ~/checkpoints/norm_stats_grasp_sft_v0_corrected`
   → `~/checkpoints/converted/molmoact2_grasp_sft_stagec_ar_step2000_corrected_v1`
   (new format directly; trained expert sha `b778bbf2…` ≠ base
   `7a2d4dea…`; corrected table baked with a recorded `stats_note`).
   If the owner picks this arm, the §3 command swaps only the
   `--init-from` path to this artifact.
4. **Both arms full-parse green** against the new family CLI
   (family checkpoint-inferred as `molmoact2_flow`, `--objective
   flow` pathway); the `--image-augment p=0` bitwise oracle and the
   re-anchored gradflow probe (loss oracle 27.8546 exact) both pass
   post-merge.

Launch remains owner-gated (arm pick + route + GPU release), exactly
as §7 states.

## §9 Amendment — main phase 5c re-verify (2026-08-15 18:2xZ)

Phase 5c (`f32ae89`, "rollout + GRPO + sim on the VLA traits —
phase-5 laptop close") merged into fontaine, check.py 925 green.
Nothing in the frozen §3 command moved — **both arms re-parse green
verbatim** (family still checkpoint-inferred `molmoact2_flow`). Two
seams this pre-reg's eval protocol touches did move:

1. **Flag rename in the eval/rollout stack:** `--expert-dtype` is now
   `--flow-decoder-dtype` across `bijou.rollout`, `sim.rollout_sim`
   and `sim.rollout_sim_parallel` (same post-load cast, same
   defaults). §4's "bf16 expert" now reads "bf16 flow decoder" at
   invocation time; the protocol itself is unchanged.
2. **Metadata API:** `read_checkpoint_info(...).normalization` →
   `bijou.checkpoint.read_metadata(...).stats`. The convmap seam
   scripts were migrated upstream; verified by loading all three
   converted artifacts through the new reader (corrected q01/q99
   rows present).

Gradflow loss oracles re-anchor exact post-merge (flow 1.6948,
ar_backbone 27.8546). Launch remains owner-gated (arm pick + route +
GPU release).

## §10 Amendment — main phase 6 re-verify (2026-08-15 19:4xZ)

Phase 6 (`393163f`, "delete the old world — BijouModel, the live
legacy read path") merged into fontaine, check.py 902 green (925
minus the 23 retired `test_vla_parity` tests; the five loss oracles
remain the standing gate). Nothing in the frozen §3 command moved —
**both arms re-parse green verbatim** (family still
checkpoint-inferred `molmoact2_flow`, `--flow-decoder-init inherit`).
What did move, verified:

1. **Legacy read path deleted:** `bijou.loading` no longer reads
   `bijou_config.json` at all (`read_checkpoint_info` /
   `from_checkpoint` / `from_backbone` gone, 856→259 lines); the
   layout survives solely in `bijou.convert_legacy` (frozen format-3
   reader). Legacy dirs now refuse loudly — `SystemExit` naming the
   exact `convert_legacy` command. Smoke-verified on the real
   stage-C step2000 legacy dir: conversion rc=0,
   `validate_checkpoint` OK, output **bit-identical** (recursive
   hash sweep, 0 diffs) to the banked §8 conversion.
2. **Gradflow oracle probe reworked upstream** onto the family
   classes (`GemmaFlowVLA`/`GemmaARVLA`): both anchors reproduce
   EXACT post-merge — flow **1.6948**, ar_backbone **27.8546**, all
   partition checks PASS.
3. **Reference-trunk conversion:** `er_60k/step_060000` was still
   legacy-format and every eval/init mount now requires VLA format —
   converted this session →
   `~/checkpoints/converted/er_60k_step_060000_vla`
   (family `molmo2_ar`, backbone `allenai/Molmo2-ER`, step 60000,
   mean/std stats rows carried; `validate_checkpoint` OK). The
   sim100/OOD-probe and rig-mixture `--init-from` mounts stay
   one-command-ready.

GRPO seam 33/33 targeted post-merge (`test_grpo_loop` +
`test_molmo_flow_integration`, now on `MolmoAct2FlowVLA` /
`MolmoAct2DiscreteStack`). Launch remains owner-gated (arm pick +
route + GPU release), exactly as §7 states.
