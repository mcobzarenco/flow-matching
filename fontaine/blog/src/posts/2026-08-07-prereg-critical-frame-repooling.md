# Pre-reg: critical-frame re-pooling screen

*2026-08-07 ~17:2xZ. Record-only robustness screen, CPU-only, over
artifacts already on disk. Queue item
`idea16-critical-frame-repooling` (the CI-MSE transfer from the
[offline-validation lit slice](../papers/offline-validation.md)).
Instrument: `fontaine/scripts/critical_frame_repooling.py` (this
pre-reg lands with it, before any critical-pool number is read).*

## Why

CI-MSE (2606.29898) measured our metric class: raw validation MSE
correlates with rollout success at Spearman **−0.61** over 27 VLA
checkpoints, and one model family got ranked *backwards*; scoring
only task-critical frames (grasp/release/decision moments, found with
a paid VLM pass) recovers −0.87. Our panel already has the labels
CI-MSE pays for — every curated episode carries judge annotations
(subgoal boundaries, sparse per-frame `holding` flags, event marks),
and every leaderboard eval dumped per-frame npz predictions. So the
cheapest possible proxy early-warning: **re-pool the existing dumps
over critical frames only and check whether any published ranking
reorders.** Rankings hold → a robustness citation for the
leaderboard. Anything reorders → the earliest available warning that
our proxy may share CI-MSE's failure mode, before any rig decision
leans on it.

## Frame-selection rule (frozen before any number is read)

A scored panel frame at 0-based within-episode index `f0` predicts a
50-step action chunk; in the judge's 1-based frame coordinates its
prediction window is **W = [f0+1, f0+50]**. The frame is **critical**
iff its episode's blessed judgment satisfies any of:

1. **Subgoal boundary in window** — some `until_frame` value `b`
   (excluding the last, which is the episode end by contract) has
   `b ∈ W`.
2. **Holding transition overlaps window** — consecutive
   judge-annotated frames `g_i < g_j` (adjacent in the sampled
   sequence) with `holding_i ≠ holding_j` bracket the transition;
   critical iff `W ∩ [g_i, g_j] ≠ ∅`.
3. **Event in window** — a judge-annotated frame `g` with non-empty
   `events` has `g ∈ W`.

No tuned parameters: the window is the prediction chunk itself.
Blessed judgment = the records matching the dataset's
`meta/judge_annotations.json` stamp (prompt hash + model), last
`judged_at` per episode winning — byte-for-byte the training-side
selection rule in `bijou.data`. Episodes with no valid blessed
judgment are **uncovered**: their frames enter neither pool and are
reported as a coverage figure.

## Rows in scope

All rows share the frozen k4l2 panel (25,800 rows, 17,204 core
pooled; identical `truth`/`valid`/`core`/`index` across dumps —
guarded). Element-pooled chunk MAE, the leaderboard convention;
`first_mae` secondary/descriptive.

| leaderboard row | dump (`reports/`) | pred key | published |
|---|---|---|---|
| student 1-NFE single draw (#4) | `...snapdistill_h1024_30k...panel_curated_v0_k4l2_1nfe_euler1_npz` | `pred:bijou@30000` | 5.6036 |
| AR-100k draws-10 mean T=1 (#5) | `...arb_rcond_100k...panel_k4l2_draws10_t1` | `pred:bijou@100000_draws10_t1` | 5.6515 |
| AR-100k greedy (#6) | `...arb_rcond_100k...panel_k4l2` | `pred:bijou@100000` | 5.8026 |
| teacher Heun-30 single, stable-key (#7) | `...flow_artrunk...panel_curated_v0_k4l2_stablekey_heun30` | `pred:bijou@80000` | 6.5997 |
| state-copy control (#8) | (from the greedy dump) | `pred:state-copy` | 11.785 |

Descriptive extras, same treatment: state-copy-norm (11.736), teacher
old-key anchor (6.6232), own-topology `A-s0/s1/s2` (7.7966 / 7.8052 /
7.7355 — the seed trio is the **empirical null scale**: its internal
critical-pool spread bounds what a meaningless reorder looks like)
and `statedrop80` (10.5024). **Not re-poolable** (stated, not
silent): rows #1–#3 (student/teacher mean-of-10, mean-of-5) dumped
JSON reports only — no per-frame npz exists; the aux-off arm's panel
npz lives only on the box. The scoreboard's re-poolable span #4–#8
still covers the cross-family ordering student < AR < teacher <
copy that the leaderboard's structural story rests on.

## Read & reorder criterion (frozen)

For each pair of scoreboard rows adjacent in published rank (4–5,
5–6, 6–7, 7–8): paired per-frame MAE delta on the **critical core
pool**, seeded frame-level bootstrap CI95 (n=10,000, seed 0 — the
`box_batch_results` machinery). **REORDER** = the critical-pool mean
delta has the opposite sign to the published gap AND its CI95
excludes 0. All 10 pairwise combinations scanned as a secondary (any
non-adjacent flip is reported the same way). The complement pool
(covered, non-critical) is reported alongside for contrast. The seed
trio's internal flips are expected and count as nothing.

## Validity gates (abort → descriptive-only, loudly)

- Overall pooled chunk MAE per row must reproduce its published
  number to 5e-4 (identity check on the pooling).
- Valid-cell-weighted recombination of critical + complement +
  uncovered must equal the overall pooled value to 1e-6 (no frame
  silently dropped).
- Coverage: ≥ 80% of core frames in covered episodes, critical core
  pool ≥ 500 frames — below either, the rule is too sparse to rank
  on; numbers land as descriptive only, no reorder verdicts.

## Record-only clause

This screen cannot change any leaderboard number — published rows
stay as banked. Outcomes: **rankings hold** → one robustness note on
the leaderboard + the ideas page. **Any confirmed reorder** → a
proxy early-warning to the owner (Discord, same session), a caveat
row on the leaderboard, and the #16 rig-transfer benchmark inherits
the critical-frame pool as a candidate scoring rule. Escalation
beyond that (e.g. re-weighting the headline metric) needs its own
pre-reg. Cost: CPU minutes; oracle (`--selftest`) green before the
real read.

---

## Results (2026-08-07 ~17:5xZ, same session — read run after the pre-reg committed)

**Every published ranking holds on the critical pool.** All 10
pairwise gaps keep their published sign with CI95 excluding 0; no
pair is even ambiguous. Validity gates all green: coverage **99.9%**
of the 17,204 core frames (12 uncovered), critical pool **11,204**
frames, every overall re-pool reproduced its published number to
4 dp, recombination exact. Data:
`reports/analysis__critical_frame_repooling.json`.

| row | published | critical | complement |
|---|---|---|---|
| student 1-NFE single | 5.6036 | 6.0327 | 4.6766 |
| AR-100k draws-10 T=1 | 5.6515 | 6.1149 | 4.6530 |
| AR-100k greedy | 5.8026 | 6.2835 | 4.7608 |
| teacher Heun-30 stable-key | 6.5997 | 7.1052 | 5.5069 |
| state-copy | 11.7847 | 12.8679 | 9.4680 |
| A-s0 / s1 / s2 (descriptive) | 7.7966 / 7.8052 / 7.7355 | 8.4725 / 8.4571 / 8.3249 | 6.35 / 6.40 / 6.47 |
| statedrop80 (descriptive) | 10.5024 | 11.3435 | 8.6952 |

Adjacent-pair paired deltas on the critical pool (CI95): student →
draws10 **+0.079** [0.025, 0.133]; draws10 → greedy **+0.165**
[0.118, 0.214]; greedy → teacher **+0.810** [0.728, 0.893]; teacher
→ state-copy **+5.683** [5.515, 5.852]. Seed-trio critical null
scale (max pairwise |Δ|) 0.1476 — about 2× its overall-pool 0.0697,
and every cross-family gap clears it by an order of magnitude except
student-vs-draws10, which is CI-clean but inside the trio scale
(that pair was already called a statistical tie's neighbor on the
board; nothing new).

**The interesting shape: the signal *strengthens* on critical
frames.** Critical frames are uniformly harder (+0.4–0.5 MAE for
models, +1.1 for state-copy), and every model's edge over state-copy
*widens* there (student +6.18 published → +6.74 critical) — the
leaderboard's separation is not carried by easy cruise frames. The
draws-10 gain over greedy also survives (+0.165 critical vs +0.151
overall). This is the opposite of the CI-MSE failure mode (easy-frame
dilution hiding critical-frame differences): our pooled MAE and a
critical-frame MAE tell the same story on every published pair.

**Honest caveats.** The rule is broad — a 50-frame window over
subgoal-boundary-dense episodes marks 65% of core frames critical,
so this is a robustness check against frame-weighting, not a
high-contrast "grasp-moments-only" metric; and CI-MSE's ultimate
referee is rollout success, which no offline re-pool can supply —
the proxy-vs-rig question stays open until #16's rig benchmark
exists. Verdict per the pre-reg: **robustness citation banked**;
leaderboard note lands with this post; no escalation.
