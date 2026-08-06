# Ledger

Results tables, `docs/architecture.md` §7 discipline: numbers only
compare within one frame set; frozen panels are immutable; flow
results state noise draws; deployment vs unconstrained never mix.

## The instrument

**Headline metric:** community panel MAE —
`bijou.eval --sample-plan plans/holdout_curated_v0_k4l2.json` on
`community_curated_v0`, `--episodes holdout --holdout-episodes 0.1
--split-seed 0 --fps 30 --camera-counts 1 2`, greedy AR /
deployment-class decoding. 17,204 core frames; deterministic per
checkpoint.

**Confirmation:** the sealed panel
(`plans/holdout_curated_v0_k4l2_sealed.json`, plan seed 1) — scored
only on claimed bests, at most ~weekly.

Breakthrough bars (charter §2): ☆ ≤ 5.0 · ☆☆ ≤ 4.5 or
first_mae ≤ 1.6 · ☆☆☆ mainline adoption.

## Anchors (mainline-measured, inherited 2026-08-05)

| checkpoint | panel MAE | first_mae | notes |
|---|---|---|---|
| state-copy | 11.785 | 2.620 | on the identical frames |
| state-copy-norm | 11.736 | — | |
| **`bijou_arb_rcond_100k_ddp4` @100k** (baseline to beat) | **5.803** | 2.143 | fast path; 79% paired win rate vs copy |
| `bijou_flow_artrunk` @80k (Heun-30) | 6.623 | 1.933 | flow-family reference, stage-2 lineage; **index keying, superseded for new quotes** |
| `bijou_flow_artrunk` @80k (Heun-30, **noise-key stable**) | **6.5997** | 1.9355 | **re-banked anchor 2026-08-06** — the quoted keying for all new flow numbers; controls bitwise, Δ vs index −0.024 ≈ 1σ_draw ([results](posts/2026-08-06-stablekey-rebank-results.md)) |

Pending own-instrument verification (charter §10.5): the baseline
re-score on this box must reproduce 5.803 before any number here is
cited as "verified locally". Sealed-panel anchors land with the
integrity kit.

## Fontaine results — deployment class

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

## Fontaine results — unconstrained class

*(empty — no runs yet)*
