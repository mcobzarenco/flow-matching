# Leaderboard

*Evergreen (renamed from "Ledger", owner steering 2026-08-07 10:04Z):
the best banked score for every model family × decode config, in one
place, updated as endpoints land. Numbers only compare within one
frame set; frozen panels are immutable; flow results state their
noise draws; deployment vs unconstrained never mix
(`docs/architecture.md` §7).*

## The scoreboard — community panel v1, deployment class

All rows: `bijou.eval` on `community_curated_v0` holdout,
`plans/holdout_curated_v0_k4l2.json` — 25,800 frames scored, 17,204
core frames pooled, identical rows for every entry. Sorted by panel
MAE. Breakthrough bars (charter §2): ☆ ≤ 5.0 · ☆☆ ≤ 4.5 or
first_mae ≤ 1.6 · ☆☆☆ mainline adoption.

| # | model × decode | panel MAE ↓ | first_mae | evals/frame | eval ms/f¹ ⏱ | b=1 ms¹ ⏱ | provenance |
|---|---|---|---|---|---|---|---|
| 1 | **SnapFlow student, 1-NFE, mean-of-10** | **5.3675** | 1.5927² | 10 | 50.0 | 111.2 | [results](posts/2026-08-06-snapflow-results.md) |
| 2 | **Flow teacher @80k, Heun-30, mean-of-10** | **5.3645** | **1.4242** | 300 | 409.6 | 1245.0 | [results](posts/2026-08-06-snapflow-results.md) |
| 3 | SnapFlow student, 1-NFE, mean-of-5 | 5.3918 | 1.6056 | 5 | 50.0 | 111.2 | [results](posts/2026-08-06-snapflow-results.md) |
| 4 | SnapFlow student, 1-NFE, single draw | 5.6036 | 1.7039 | 1 | 46.9 | 100.1 | [results](posts/2026-08-06-snapflow-results.md) |
| 5 | AR-100k, draws-10 mean, T=1.0 | 5.6515 | 1.9477 | 10 (serial) | 2107.3 | 7993.0 | [readout](https://mcobzarenco-fontaine-blog.static.hf.space/reports/analysis__draws10_t1_ar100k_k4l2.json) |
| 6 | AR-100k, greedy decode (deployment anchor) | 5.8026 | 2.1431 | 1 (serial) | 247.0 | 2156.6 | [report](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.html) |
| 7 | Flow teacher @80k, Heun-30, single draw (stable-key) | 6.5997 | 1.9355 | 30 | 115.7 | 1234.0 | [rebank](posts/2026-08-06-stablekey-rebank-results.md) |
| 8 | state-copy (control) | 11.785 | 2.620 | 0 | — | — | banked, byte-matched every eval |

Ranks 1–2 are a statistical tie on chunk MAE (Δ 0.003, ~1σ_draw) at
**30× different expert compute**. The ☆☆ first-mae arm (≤ 1.6) is
**crossed** — by the teacher's mean-of-10 (1.4242, best on the
board) and the student's (1.5927). The ☆ chunk bar (≤ 5.0) is
**open**: current best 5.3675, gap 0.37.

**Row 5 landed 2026-08-07** (the `draws10_t1` boundary, all three
pre-registered expectations met): AR mean-of-10 buys −0.145 [CI95
−0.182, −0.109] — real but ~9× smaller than the flow families' draws
gain, the pre-registered mean-collapse shape (greedy AR decode
already sits near the predictive mean). **Rows still owed**: molmo2
AR 40k greedy + mean-of-10 at its endpoint (~2026-08-08). The
T-sensitivity rungs (T ∈ {0.5, 0.7, 1.3}) are **record-only by
pre-registration** and never enter the leaderboard — dT diagnostic
only.

## Reading the compute column

¹ **Both ⏱ columns are the same-harness micro-benchmark**
([pre-reg](posts/2026-08-07-prereg-leaderboard-decode-microbench.md),
[results in the main-sync post](posts/2026-08-07-main-sync-review.md),
data `reports/analysis__leaderboard_decode_microbench*.json`),
measured 2026-08-07 on the local 1×H100 on the **post-merge tree**
(batched noise-draw ensembling in): identical frames per mode across
every row, decode flags byte-matched to the banked panel stems.
`eval ms/f` = batched-eval throughput (b32/w20, N=320) — the cost of
running the panel. `b=1 ms` = single-stream latency (b1/w4, N=50) —
the deployment-facing read (#16 hook). `evals/frame` stays as the
structural column (draws × solver evals; AR decodes are token-serial
— no eval count captures them, hence "(serial)"). These replace the
earlier mtime-derived ≈ estimates and the two heterogeneous ⏱
wall-clocks; cross-row deltas are now apples-to-apples. AR singles
were measured pre-merge (the merge does not touch the AR decode
path); the flow `draws=1` pre/post control pairs reproduce to ≤0.3%.

The structural story, post-merge (batched draws): **mean-of-N now
costs single-draw latency** — student mean-of-10 111 ms vs single
100 ms; teacher mean-of-10 1,245 ms vs single 1,234 ms (was 11,284
sequential: **9.1×**). The student's 10 draws cost 11% extra latency
for a −0.24 panel gain (rows 1 vs 4); the AR family pays serially
either way (2.2 s greedy → 8.0 s draws-10 single-stream) for a
−0.145 gain — mean-of-draws is a flow-family superpower, not a
universal one (row 5's readout).

² The student's mean-of-10 first_mae (1.5927) crosses the ☆☆
first-mae bar (≤ 1.6); the teacher's 1.4242 remains the best
first-step accuracy banked.


## The instrument

**Headline metric:** community panel MAE —
`bijou.eval --sample-plan plans/holdout_curated_v0_k4l2.json` on
`community_curated_v0`, `--episodes holdout --holdout-episodes 0.1
--split-seed 0 --fps 30 --camera-counts 1 2`, deployment-class
decoding stated per row. Deterministic per checkpoint (flow rows:
stable noise keying, draws stated).

**Confirmation:** the sealed panel
(`plans/holdout_curated_v0_k4l2_sealed.json`, plan seed 1) — scored
only on claimed bests, at most ~weekly.

**Own-instrument verification (charter §10.5): DONE** — the AR-100k
baseline re-scored locally reproduces 5.8026/2.1431 exactly (banked
npz + report in `reports/`; the `draws10_t1_results.py` and
`selection_ceiling_results.py` oracles re-derive both numbers from
the raw npz on every run). Sealed-panel anchors land with the
integrity kit.

## Anchors (mainline-measured, inherited 2026-08-05)

| checkpoint | panel MAE | first_mae | notes |
|---|---|---|---|
| state-copy | 11.785 | 2.620 | on the identical frames |
| state-copy-norm | 11.736 | — | |
| **`bijou_arb_rcond_100k_ddp4` @100k** (baseline to beat) | **5.803** | 2.143 | fast path; 79% paired win rate vs copy; verified locally |
| `bijou_flow_artrunk` @80k (Heun-30) | 6.623 | 1.933 | flow-family reference, stage-2 lineage; **index keying, superseded for new quotes** |
| `bijou_flow_artrunk` @80k (Heun-30, **noise-key stable**) | **6.5997** | 1.9355 | **re-banked anchor 2026-08-06** — the quoted keying for all new flow numbers; controls bitwise, Δ vs index −0.024 ≈ 1σ_draw ([results](posts/2026-08-06-stablekey-rebank-results.md)) |

## Own-topology results — deployment class

*Frame set: k4l2 community panel v1, greedy AR, 17,204 core frames.
Topology caveat (§2): eff-10 1×H100-slice arms — cross-topology vs the
mainline anchors is directional only; paired reads within the batch are
clean.*

| run | steps | panel MAE | first_mae | notes |
|---|---|---|---|---|
| `fontaine_arb_rcond_40k_1xh100` (A-s0, aux-on control) | 40k | 7.7966 | 3.9422 | **own-topology baseline**; [results](posts/2026-08-06-box-batch-results.md) |
| `fontaine_arb_rcond_40k_1xh100_s1` | 40k | 7.8052 | 4.1118 | seed replicate |
| `fontaine_arb_rcond_40k_1xh100_s2` | 40k | 7.7355 | 3.9377 | seed replicate; σ_seed(chunk)=0.038, max pairwise Δ=0.0697 |
| `fontaine_arb_rcond_auxoff_40k_1xh100` (B) | 40k | 8.2989 | 3.5009 | aux-off: **+0.462 vs A-s0, CI [0.387, 0.537], REAL** (7.5× replicate threshold, LORO-coherent); first_mae inversion + cond-sens 1.13 vs 1.86–2.00 |

## Own-topology results — unconstrained class

*(empty — no runs yet)*
