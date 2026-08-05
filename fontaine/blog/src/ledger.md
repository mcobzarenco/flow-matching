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
| `bijou_flow_artrunk` @80k (Heun-30) | 6.623 | 1.933 | flow-family reference, stage-2 lineage |

Pending own-instrument verification (charter §10.5): the baseline
re-score on this box must reproduce 5.803 before any number here is
cited as "verified locally". Sealed-panel anchors land with the
integrity kit.

## Fontaine results — deployment class

*(empty — no runs yet)*

## Fontaine results — unconstrained class

*(empty — no runs yet)*
