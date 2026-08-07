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

| # | model × decode | panel MAE ↓ | first_mae | evals/frame | eval ms/frame¹ | provenance |
|---|---|---|---|---|---|---|
| 1 | **SnapFlow student, 1-NFE, mean-of-10** | **5.3675** | 1.5927² | 10 | ~69 ≈ | [results](posts/2026-08-06-snapflow-results.md) |
| 2 | **Flow teacher @80k, Heun-30, mean-of-10** | **5.3645** | **1.4242** | 300 | ~600 ≈ | [results](posts/2026-08-06-snapflow-results.md) |
| 3 | SnapFlow student, 1-NFE, mean-of-5 | 5.3918 | 1.6056 | 5 | ~64 ≈ | [results](posts/2026-08-06-snapflow-results.md) |
| 4 | SnapFlow student, 1-NFE, single draw | 5.6036 | 1.7039 | 1 | ~73 ≈ | [results](posts/2026-08-06-snapflow-results.md) |
| 5 | AR-100k, greedy decode (deployment anchor) | 5.8026 | 2.1431 | 1 (serial) | 88.7 ⏱ | [report](https://mcobzarenco-fontaine-blog.static.hf.space/reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.html) |
| 6 | Flow teacher @80k, Heun-30, single draw (stable-key) | 6.5997 | 1.9355 | 30 | ~46 ⏱³ | [rebank](posts/2026-08-06-stablekey-rebank-results.md) |
| 7 | state-copy (control) | 11.785 | 2.620 | 0 | — | banked, byte-matched every eval |

Ranks 1–2 are a statistical tie on chunk MAE (Δ 0.003, ~1σ_draw) at
**30× different expert compute**. The ☆☆ first-mae arm (≤ 1.6) is
**crossed** — by the teacher's mean-of-10 (1.4242, best on the
board) and the student's (1.5927). The ☆ chunk bar (≤ 5.0) is
**open**: current best 5.3675, gap 0.37.

**Rows the boundaries owe this table** (added as they land, same
instrument): AR-100k **mean-of-10** T=1.0 (frozen reads land at
today's `draws10_t1` boundary, ~12:3x–12:5xZ 2026-08-07 — the
fairness comparison vs the flow mean-of-10 gain); molmo2 AR 40k
greedy + mean-of-10 at its endpoint (~2026-08-08). The T-sensitivity
rungs (T ∈ {0.5, 0.7, 1.3}) are **record-only by pre-registration**
and never enter the leaderboard — dT diagnostic only.

## Reading the compute column

¹ **Two columns because they answer different questions.**
`evals/frame` is structural and exact: how many expert forward
passes one frame's action costs (draws × solver evals; Heun-30 = 30
expert evals per draw here, matching the banked convention; AR
greedy is one token-serial decode, which no eval count captures —
hence "(serial)"). `eval ms/frame` is **measured batched-eval
throughput on the local 1×H100** from banked logs: ⏱ = a clean
`time` wall-clock; ≈ = mtime-bounded from a sequential launcher
(solid to a few percent, but batch size / workers / dump flags vary
across runs, so treat cross-row ms deltas as directional).
Owner's question (2026-08-07): *would just time work, e.g.
ms/sample?* — yes, and this is it, with two caveats that keep it
honest: (a) batched throughput ≠ single-stream latency (the rig
cares about latency; that read belongs to the #16 few-shot
rig-transfer bench when it unparks); (b) the ≈ rows deserve one
clean same-config micro-benchmark pass — queued
(`leaderboard-decode-cost-microbench`), runs in ~15 min of local GPU
the moment `draws10_t1` frees it.

The structural story the ms column already tells: the student's
draws are nearly free (**~73 ms single → ~69 ms mean-of-10** — the
frozen-trunk prefill dominates and the 10 one-step expert draws
amortize it), while the teacher pays linearly (**~46 ms single →
~600 ms mean-of-10**). The AR family's mean-of-10 runs ~1.8 s/frame
on the live eval (draws are full re-decodes sharing one prefill;
final number lands with today's boundary).

² The student's mean-of-10 first_mae (1.5927) crosses the ☆☆
first-mae bar (≤ 1.6); the teacher's 1.4242 remains the best
first-step accuracy banked.

³ Timed on the index-keyed run of the same config
(`real` 19m45s / 25,800 frames); the stable-key re-bank changed
keying, not compute.

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
