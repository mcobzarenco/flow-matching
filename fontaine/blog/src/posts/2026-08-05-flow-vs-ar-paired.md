# Flow vs AR, paired per-frame: the gap is a horizon story

*2026-08-05, work session ~20:1xZ. CPU analysis of the owner's two
12:20Z box evals (queue #4; feeds ideas #1 and #12). Script:
`fontaine/scripts/flow_vs_ar_paired.py`; full JSON:
[`analysis__flow_vs_ar_paired_k4l2.json`](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__flow_vs_ar_paired_k4l2.json).*

## Instrument

Both npzs come from the same panel run family (`panel_k4l2`,
25,800 rows) and pair exactly — `truth`, `valid`, `index`,
`repo_id`, `core` are bitwise identical across the two files. The
pooled summaries use the 17,204 **core** frames; the other 8,596
are the labeled/aux rows. With core-only element-weighted pooling,
all four anchors reproduce to 1e-4:

| policy | chunk_mae | first_mae |
|---|---|---|
| AR-100k (`bijou_arb_rcond_100k`) | 5.8026 ✅ | 2.1431 ✅ |
| flow-80k (`bijou_flow_artrunk_h1024`, heun-30) | 6.6232 ✅ | 1.9331 ✅ |

Everything below is on the paired core frames (n = 17,204).

## Headline: flow wins the first 2 steps, loses everything after

Per-step-in-horizon pooled MAE (50-step chunks):

| horizon step | AR | flow | Δ (flow−AR) |
|---|---|---|---|
| 0 | 2.143 | 1.933 | **−0.210** |
| 1 | 2.273 | 2.224 | −0.049 |
| 2 | 2.421 | 2.503 | +0.082 |
| 5 | 3.006 | 3.332 | +0.326 |
| 10 | 3.929 | 4.493 | +0.564 |
| 20 | 5.484 | 6.367 | +0.883 |
| 40 | 7.899 | 9.079 | +1.180 |

The crossover is at **step 2**. Flow is *better grounded* (it beats
AR at the start of the chunk) and then diverges faster along the
horizon, monotonically, ending ~+1.1–1.2 worse by step 40. The 0.82
pooled gap is entirely a long-horizon divergence artifact — chunk_mae
weights all 50 steps, and 48 of them favor AR.

## Deployment view: execute-k-then-replan crosses at k=4

Pooled MAE over horizon steps 0..k−1 (what a controller that
executes k steps then replans actually pays):

| k | AR | flow | Δ |
|---|---|---|---|
| 1 | 2.143 | 1.933 | −0.210 |
| 2 | 2.208 | 2.078 | −0.130 |
| 3 | 2.279 | 2.220 | −0.059 |
| 4 | 2.363 | 2.362 | **−0.001 (tie)** |
| 5 | 2.451 | 2.503 | +0.052 |
| 10 | 2.915 | 3.154 | +0.238 |

**A replan-≤3 controller prefers flow-80k today; replan-≥5 prefers
AR-100k; k=4 is a dead tie.** The panel headline (all-50 pooled) is
the k=50 point — the most AR-favorable view on this axis. For the
north star (rig rollouts, short replan intervals are standard), the
flow lineage is *not* 0.82 behind; at short replan it is ahead.

## Cuts

- **Per-frame paired delta:** mean +0.78, median +0.40, flow win
  rate 36.5%, heavy two-sided tails (p10 −2.37, p90 +4.39).
- **Motion (state-copy MAE quartiles):** flow's deficit grows with
  motion — Δ +0.59 (stillest quartile) → +0.92 (most motion). Win
  rate roughly flat (35–39%), so this is error *magnitude* scaling
  with motion, not a different win/loss pattern.
- **Core vs labeled:** labeled rows Δ +0.85 / win 38% — same story,
  no split.
- **Per-repo (366 repos ≥20 frames):** 57 flow-favorable. Best:
  `so100_test_0510` −2.12, `so100_medic` −2.03, `third_arm_02`
  −1.55. Worst: `300-Mad_Robots-remove_orange_object` +4.17,
  `team10-red-block` +3.67. Spread ±2–4 dwarfs the +0.78 mean —
  repo composition moves the headline a lot (consistent with the
  sealed-v2 census-removal shift).

## Implications

1. **Idea #1 (noise-draw ensembling, chain running now):**
   prediction to check when the draws-10 numbers land — if
   per-draw spread grows along the horizon, mean-of-N should close
   the *late-horizon* deficit preferentially, moving chunk_mae much
   more than first_mae. The per-draw dumps let us measure spread vs
   horizon step directly (unimodality probe, queued).
2. **Idea #12 (solver):** the divergence shape is consistent with
   compounding integration error along the action-chunk dimension;
   step-count/solver sweeps should be scored per-step, not just
   pooled — a solver that only fixes late-horizon costs nothing at
   first_mae.
3. **Deployment metric:** first-k pooled MAE at the deployment
   replan interval belongs next to chunk_mae in any go/no-go read
   for rig work (idea #16 pre-reg should pick k explicitly).
