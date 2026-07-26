# Ablation: 4 arms × 20k steps, single H100 each (2026-07-26)

Matched-budget architecture ablation, from scratch, on v1+v2+v3 community
data with the episode holdout active for the first time.

**Shared setup.** 1036 datasets, train split 49,533 episodes / 24.27M
frames; 5,516 episodes held out (`--holdout-episodes 0.1 --split-seed 0`).
Expert 1024/8/4096/8, chunk 50, batch 64, lr 1e-4 cosine, warmup 500,
seed 42, 20k steps = 1.28M samples ≈ **5.3% of one train-split epoch**.

| arm | delta vs control | s/step |
|---|---|---|
| control | — (causal_actions, 140 soft tokens, streams 4-4-8) | ~1.0 |
| bidir | `--self-attention-mode bidirectional` | ~1.0 |
| tokens280 | `--max-soft-tokens 280` (prefix 860 vs 452) | ~1.9 |
| streams0016 | `--stream-counts 0 0 16` (all cross-attn on layer 14) | ~1.0 |

## Train loss (mean over steps 18k–20k)

control **0.1739** · bidir **0.1652** · streams0016 **0.1736** ·
tokens280 **0.1739**

bidir is ~5% lower throughout, from step ~500 on. streams0016 and
tokens280 are indistinguishable from control on loss.

## Open-loop eval (`bijou.eval`, 256 frames/side, seed 0, Heun-10)

Same 256 frames for every arm within a side (identical selection + seed).
`Δ` = paired per-frame chunk-MAE delta vs raw state-copy (negative =
better than baseline); `win` = fraction of frames beating it.

| side | state-copy | control | bidir | streams0016 | tokens280 |
|---|---|---|---|---|---|
| train episodes | 11.10 | 11.52 (Δ+0.24, 42%) | 11.78 (Δ+0.47, 39%) | 11.55 (Δ+0.25, 41%) | 11.59 (Δ+0.30, 41%) |
| held-out episodes | 10.30 | 10.55 (Δ+0.36, 40%) | 10.85 (Δ+0.65, 40%) | 10.86 (Δ+0.64, 41%) | 10.40 (Δ+0.21, 40%) |
| marius rig (held-out rig) | 9.54 | 13.80 (Δ+4.19, 17%) | 17.55 (Δ+7.77, 16%) | 14.17 (Δ+4.64, 14%) | 14.65 (Δ+5.01, 15%) |

Reference on the **same held-out frames**: mainline
`bijou_community_v1v2v3_20k_ddp4/step_040000` scores **6.93**
(Δ−3.14, 58% win). Caveats: its lineage totals ~15.4M samples (12× the
arms) and it trained on *all* episodes, so those frames are contaminated
for it — treat 6.93 as an optimistic bound on "same architecture, more
training".

## Findings

1. **All arms are severely undertrained; the ablation cannot rank
   architectures yet.** At 1.28M samples every arm still sits at or
   slightly below the trivial baseline, while the same control
   architecture at ~12× samples is 3+ MAE past it. Loss was still
   falling at 20k in all arms. Differences of 0.3 MAE between arms at
   this point are noise compared to the training-scale effect.

2. **bidir's lower train loss is (at least partly) an easier-objective
   artifact, not better learning.** In causal_actions mode action token
   i denoises seeing only tokens ≤ i; bidirectional sees the whole noisy
   chunk — a lower conditional-entropy target, so its loss floor is
   lower by construction. Loss values are not comparable across
   attention modes. On the like-for-like metric (sampled-trajectory
   MAE) bidir is *worse* on every side, dramatically so on the held-out
   rig (17.55 vs 13.80). Verdict: keep causal_actions; stop reading
   bidir's loss as a win.

3. **Episode-level generalization gap is tiny; rig-level gap is the
   problem.** Control's Δ vs copy: +0.24 train side → +0.36 holdout side
   (a ~0.1 gap), versus +4.19 on the unseen rig. At 5% of an epoch
   memorization is barely possible, and the split confirms within-rig
   transfer across episodes is nearly free. Everything interesting is in
   cross-rig transfer (calibration offsets, camera placement).

4. **streams0016 ≈ control; tokens280 ≈ control at matched steps, worse
   at matched compute.** Cross-attn depth placement is a wash at this
   scale — no reason to abandon 4-4-8. tokens280 posts the best held-out
   number (10.40, Δ+0.21) and best first_mae/p90 there, but is behind
   control on the other two sides and costs 1.9× per step: for the same
   wall clock, control trains ~2× the steps. Not worth it while training
   scale is the bottleneck; worth revisiting once it isn't.

## Decision

Promote **control (causal, 140 tokens, 4-4-8)** unchanged. Spend the next
budget on *training longer*, not architecture: DDP4 from scratch with the
holdout active (so eval stays honest), 40–60k steps at global batch 256
(≈40–60% of an epoch), then re-score holdout + marius rig. Re-run this
ablation only if that run's holdout Δ stalls above the baseline.

Artifacts: eval JSONs in `~/eval_abl_*.json` on the box (copies in
`outputs/abl_results/` on the laptop), checkpoints in
`outputs/train/abl_<arm>_20k/step_020000`, wandb runs `abl-*-20k`.
