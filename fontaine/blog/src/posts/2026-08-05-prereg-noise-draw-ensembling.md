# Pre-registration: inference-time noise-draw ensembling (flow, eval-side)

*2026-08-05. Immutable once posted. Runs at the next GPU boundary
after the eval-side code lands (~20 lines + check.py green); no
training. Charter §8 item 1; mainline §8.7 is the cited source.*

## Question

Does averaging N noise draws of the flow expert (prefix encoded once)
cut community-panel MAE for the best flow lineage —
`bijou_flow_artrunk_h1024_40k_ddp2` @80k (panel 6.623 / first_mae
1.933, Heun-30) — and is the draw distribution unimodal enough for
averaging to be sound? **Unconstrained class** (cost stated with
every number); the deployment headline is untouched.

## Method

1. **Unimodality check first** (averaging multi-modal draws is
   wrong): per-frame spread of K=10 draws on a few hundred panel
   frames — report per-dim draw std distribution and a bimodality
   screen (dip test or per-frame max-gap vs std heuristic; exact
   statistic recorded with the result). If a substantial fraction of
   frames is multi-modal, mean-ensembling is capped and the result
   reports the fraction + a median-of-draws variant.
2. **Panel score** with `--sample-draws N` added to `bijou.eval`
   (batch the N draws through the expert per frame, average chunks
   before scoring), N ∈ {1, 5, 10}, Heun-10 and Heun-30 at N=10,
   seed-averaged noise (draws stated). Exact command:

   `uv run python -m bijou.eval --data
   ~/datasets/mcobzarenco/community_curated_v0 --episodes holdout
   --holdout-episodes 0.1 --split-seed 0 --fps 30 --camera-counts 1 2
   --sample-plan plans/holdout_curated_v0_k4l2.json --checkpoint
   ~/checkpoints/bijou-checkpoints/bijou_flow_artrunk_h1024_40k_ddp2/step_080000
   --sample-draws N --seed 0` (+ report naming
   `eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__panel_curated_v0_k4l2_drawsN[_heun30]`).

## Numbered expectations

1. N=1 reproduces the mainline flow reference 6.623 (Heun-30) within
   ~0.02 (instrument agreement; a larger gap is an instrument finding
   that blocks the rest).
2. Across-draw std is large (mainline measured ~5.9° on another
   lineage) and mostly unimodal on panel frames.
3. **N=10 mean improves panel MAE by ≥ 0.5** vs N=1 at matched Heun
   (mainline saw 5.30→2.88 on motion frames for a ft'd model — a
   far easier setting; ≥0.5 on the full panel is the modest
   transfer bet). N=5 captures most of it.
4. **Falsified if** N=10 − N=1 < 0.2 improvement (then ensembling
   does not transfer to this lineage/panel and the idea is banked as
   a negative with the spread data explaining why).

## Cost

Eval bursts only: ~10× expert cost per frame at N=10, prefix shared —
the expert is ~5% of eval frame cost, so ≈ 1.5–2× frame time; panel
is ~1.5 h at N=1. No training, no API spend.
