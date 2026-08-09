# Subgoal-draws rung (b′) FALSIFIED: the width holds better subgoals — the scorer anti-selects them

*2026-08-09 00:0x–00:2xZ. The rung-(b′) clean-list run
([pre-reg](2026-08-08-prereg-subgoal-draws-cleanlist.md)) completed
23:52Z 08-08 on the pre-registered q4 fallback subset (4,301 rows;
the rate gate fired at launch and the switch was recorded). The
frozen reads ran this session after landing the subset-join read
path the run's boundary named (the draws10/energy-score join
convention, oracle-gated with a q4-shaped slice fixture; `check.py`
538 green). Adjudication banked in
`analysis__subgoal_draws_cleanlist_q4_ar100k_k4l2.json`. **E6
fired — decisively — and the pre-registered adjudication routes
NO-SCORER.** Δ_bon/Δ_ceil, unmeasured since rung (b) closed at
table cost, are finally on the board.*

## The one number

**Read 1 (primary, frozen):** head-to-head paired (bon − self)
per-frame vs the banked rung-(a) self arm, seeded bootstrap CI95.
Falsifier: CI95 not entirely below 0 ⇒ FALSIFIED.

**Result: +0.210, CI95 [+0.113, +0.312] — entirely *above* zero.**
Conditioning on the self-certainty pick from 8 clean sampled
candidates is significantly *worse* than conditioning on the single
greedy self subgoal. Worse still, Δ_bon vs the bare baseline is
**+0.142 [+0.027, +0.260]**: the SC pick loses to running no
subgoal conditioning at all. Expectation E3 (bon lands between self
and ceiling) failed in the ugliest available way — bon is below
both.

![Four paired deltas vs the bare baseline on the q4 subset: oracle
ceiling −0.250, banked self −0.069, narrated pass 1 −0.036,
self-certainty pick +0.142](../img/fieldcond/subgoal_cleandraws_deltas.svg)

## The adjudication: NO-SCORER, with a live ceiling

The point of pairing Δ_ceil with the falsifier was to know *why* if
it fired. It fired with the ceiling alive:

| read | Δ | CI95 | verdict |
|---|---|---|---|
| Δ_bon (core, primary) | **+0.142** | [+0.027, +0.260] | SC pick loses to bare baseline |
| bon − self (head-to-head) | **+0.210** | [+0.113, +0.312] | E6 falsifier fires |
| Δ_ceil (labeled bound) | **−0.250** | [−0.353, −0.148] | width contains real gains |
| ceil − self (adjudication) | **−0.181** | [−0.281, −0.085] | clearly better texts exist |

The oracle pick (token-F1 vs the held-out true subgoal) beats the
baseline by −0.250 and the greedy self subgoal by −0.181, both CI95
clear of zero — the 8-candidate width **contains genuinely
better-phase subgoals**. Horizon decomposition says the ceiling's
gain sits exactly where the rung-(a) slot said subgoal signal
lives: last-10% steps **−0.464** vs first-10% −0.026. The
self-certainty scorer not only fails to find those texts, it
**anti-selects**: its pick's late-horizon delta is +0.055.

The clean-list filter itself did its structural job — 0 fallback
rows, eligible list mean 8.06 of 9 (min 4), 97.7% of rows with ≥ 2
unique candidate texts. Diversity is not the constraint. The
scorer is. Under the pre-reg's frozen routing: **the selection
family stays closed on scorer-free tricks; scorer-side escalations
(a learned verifier, a probe-style ranker — the
[ROVER](../papers/rover-learned-verifier.md) shape) may earn their
own pre-reg.** No prompt fishing, no post-hoc scorer promotion.

## Record-only supporting reads

**Agreement (read 3):** the SC pick's text differs from greedy on
59.8% of rows — it selects *actively*, just wrongly. The record-only
alternates barely agree with it (likelihood 40.6%, medoid 39.8%)
and agree with the oracle even less (45.6% / 44.6% on labeled
rows): no scorer in the free family tracks the ceiling.

**Free channel (record-only):** the narrated pass-1 column lands at
−0.036 [−0.143, +0.072] vs baseline — spans zero on this subset,
consistent with the known small-magnitude narration signs.

**Execution oracles (read 6, all green, each a hard abort):**
full-panel anchor reproduced 5.8026/2.1431 before the join; subset
join 4,301/25,800 with identity + state-copy byte-match on the
joined rows; every dumped live pick byte-matches the offline scorer
recompute (both arms); filter provenance in report + dump; no bare
baseline column. Baseline and banked self re-pooled onto the q4
rows pair at 6.841/6.772 — q4 pooled levels are not comparable to
full-panel numbers, all claims above are paired deltas. The pass-1
narr column differs from the banked rung-(a) narr on 3,159/4,301
rows (recorded, not adjudicated — the amendment-1
composition/device class).

## Where this leaves #6

Rung (a) put the slot's value at −0.29 (oracle text, late-horizon);
rung (b′) now shows a sampled width that *contains* that value
(−0.25 ceiling) and prices the missing piece: a scorer. Generation
is not the bottleneck at this width — phase estimation
(rung (a)'s Δ_self ≈ 0) and now selection (Δ_bon > 0) both fail on
the same axis. Anything that can rank candidate subgoals by actual
phase fit — a learned verifier head, the fields-channel probe, or
distillation from the oracle picks this run dumped (4,298 labeled
rows of picked-vs-oracle pairs, free training data) — is the named
next rung, its own pre-reg required. Cost: ~1.4 GPU-h of the ≤ 5
gate.
