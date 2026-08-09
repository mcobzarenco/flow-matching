# Ablation: 4 arms × 40k steps, single H100 each (2026-07-26)

> **Historical record** (2026-07). Two conclusions have since moved: the retained 4-4-8 stream schedule is a per-run override now (CLI default 4-4-7), and "bidirectional is catastrophic cross-rig" was later complicated — the best banked flow checkpoint is an adaRMS + bidirectional lineage (architecture.md §7).

Matched-budget architecture ablation on v1+v2+v3 community data with the
episode holdout active. Two rounds: 20k steps from scratch, then a
lossless resume to 40k (cosine re-evaluated over the 40k horizon). This
doc reports the 40k results; the 20k numbers appear as the paired
"earlier round" column.

**Shared setup.** 1036 datasets, train split 49,533 episodes / 24.27M
frames; 5,516 episodes held out (`--holdout-episodes 0.1 --split-seed 0`).
Expert 1024/8/4096/8, chunk 50, batch 64, lr 1e-4 cosine, warmup 500,
seed 42, 40k steps = 2.56M samples ≈ **10.5% of one train-split epoch**.

| arm | delta vs control | s/step |
|---|---|---|
| control | — (causal_actions, 140 soft tokens, streams 4-4-8) | ~1.0 |
| bidir | `--self-attention-mode bidirectional` | ~1.0 |
| tokens280 | `--max-soft-tokens 280` (prefix 860 vs 452) | ~1.9 |
| streams0016 | `--stream-counts 0 0 16` (all cross-attn on layer 14) | ~1.0 |

## Train loss (mean over the last 2k steps of 40k)

control **0.1501** · bidir **0.1445** · streams0016 **0.1505** ·
tokens280 **0.1501**

bidir remains ~4% lower throughout — see finding 2 of the 20k round
(easier objective: bidirectional denoising sees the whole noisy chunk, a
lower conditional-entropy target; loss is not comparable across
attention modes). All curves still declining at 40k.

## Open-loop eval (`bijou.eval`, 256 frames/side, seed 0, Heun-10)

Same 256 frames for every arm within a side (identical selection +
seed). `Δ` = paired per-frame chunk-MAE delta vs raw state-copy
(negative = better than baseline). Both rounds scored with the same
(current) evaluator — the 20k column is a re-score, matching the
original 20k numbers within 0.05 MAE.

**train episodes** (state-copy 11.10):

| arm | 20k | 40k | change |
|---|---|---|---|
| control | 11.64 (Δ+0.36) | 10.40 (Δ−0.86) | −1.24 |
| bidir | 11.82 (Δ+0.52) | 10.47 (Δ−0.78) | −1.34 |
| streams0016 | 11.59 (Δ+0.29) | 10.49 (Δ−0.79) | −1.10 |
| tokens280 | 11.58 (Δ+0.29) | 10.37 (Δ−0.90) | −1.22 |

**held-out episodes** (state-copy 10.30):

| arm | 20k | 40k | change |
|---|---|---|---|
| control | 10.52 (Δ+0.33) | **9.60 (Δ−0.56)** | −0.92 |
| bidir | 10.82 (Δ+0.61) | **9.59 (Δ−0.57)** | −1.23 |
| streams0016 | 10.86 (Δ+0.63) | 9.95 (Δ−0.25) | −0.91 |
| tokens280 | 10.39 (Δ+0.20) | 9.75 (Δ−0.41) | −0.64 |

**marius rig, held-out rig** (state-copy 9.54):

| arm | 20k | 40k | change |
|---|---|---|---|
| control | 13.80 (Δ+4.19) | 13.31 (Δ+3.69) | −0.50 |
| bidir | 17.55 (Δ+7.77) | 17.08 (Δ+7.27) | −0.48 |
| streams0016 | 14.17 (Δ+4.64) | **13.15 (Δ+3.64)** | −1.01 |
| tokens280 | 14.65 (Δ+5.01) | 14.84 (Δ+5.15) | +0.20 |

Reference on the same held-out frames: mainline
`bijou_community_v1v2v3_20k_ddp4/step_040000` scores **6.93** — but its
lineage totals ~15.4M samples (6× the arms) and it trained on *all*
episodes including these, so treat it as an optimistic
more-training-plus-contamination bound.

## Findings (40k round)

1. **The trivial baseline is beaten on held-out episodes — by every
   arm.** First honest crossing (the mainline's 6.93 is contaminated).
   Doubling steps bought 0.6–1.2 MAE on every side and the loss curves
   had not plateaued: training scale remains the dominant axis.

2. **No architecture variant separates from control where it counts.**
   bidir caught up in-distribution (9.59 ≈ 9.60) but stays
   catastrophically worse on the unseen rig (17.1 vs 13.3) — its 20k
   verdict stands: keep causal_actions. tokens280's 20k holdout edge
   *reversed* (best → third, and the only arm that got worse on the
   rig); at 1.9×/step it remains a poor trade. streams0016 is the one
   mild positive surprise: best on the held-out rig (13.15) and the
   biggest rig improvement (−1.01), hinting all-deepest cross-attention
   transfers slightly better — but it trails on holdout episodes (9.95
   vs 9.60), so it's a hint, not a winner.

3. **Cross-rig transfer is still the wall.** Episode-level
   generalization is essentially free (train Δ ≈ holdout Δ within
   ~0.3), while every arm is +3.6 or worse against state-copy on the
   unseen rig. Physical rollouts (ft-v1: task structure present, visual
   grounding weak) tell the same story from the hardware side.

4. **20k-round conclusions that survived the re-test**: causal over
   bidirectional (rig transfer), 4-4-8 default retained (streams0016's
   rig hint noted for a future re-test at larger scale), tokens280 not
   worth 1.9× while scale dominates.

## Decision

Architecture unchanged (control). All compute goes to scaling the
mainline: overnight lossless resume of
`bijou_community_v1v2v3_20k_ddp4/step_040000` → 85k total steps, DDP4
(≈+11.5M samples, doubling the lineage). Revisit streams0016's rig
advantage and the holdout-split pretrain once the scaling curve bends.

Artifacts: eval JSONs `~/eval_abl_{r1post,r2}_*.json` on the box (copies
in `outputs/abl_results/`), per-arm HTML reports (holdout + rig sides)
in `reports/`, checkpoints `outputs/train/abl_<arm>_40k/step_040000`,
wandb runs `abl-*-40k`.
