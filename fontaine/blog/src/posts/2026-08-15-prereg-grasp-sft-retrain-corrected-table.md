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
  --init-from  ~/checkpoints/converted/molmoact2_base_corrected_stats_v0 \
  --objective  flow            # molmo_flow expert pathway, trunk frozen \
  --expert-init inherit        # warm AE = stage-C's warm-start analogue \
  --steps 2000 --decoder-lr 5e-5 --batch-size 64 \
  --save-every 500 \
  --save-dir ~/checkpoints/finetune/fontaine_grasp_sft_bijou_corrected
```

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
