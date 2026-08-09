# Rung (c) results: masked-contrast ANTI-SELECTS — the zero-training scorer family closes

*2026-08-09 ~10:2xZ. Pre-reg:
[2026-08-09-prereg-subgoal-mcselect.md](2026-08-09-prereg-subgoal-mcselect.md)
(finalized 09:2xZ, run launched 09:12:36Z, complete 10:20Z — ~1.1
GPU-h of the 4.0 gate). Frozen read:
`mcselect_results.py` → `reports/analysis__subgoal_mcselect_q4_ar100k_k4l2.json`.
Everything below is that script's output; the argmax, tie rule and
CI machinery were landed and oracle-gated BEFORE the run.*

**Plain words.** The model can write several candidate "next step"
notes to itself, and we know (from rung (b′)) that the sampled set
usually contains a genuinely better note than the one it writes by
default — but the first scorer we tried (pick the note the model is
most *confident* about) chooses badly. This rung tried the opposite
philosophy, imported from MG-Select: pick the note that most
*changes* what the model wants to do (measured against a
"no-note-at-all" reference). Verdict: that scorer chooses badly too
— in fact worse. Picking maximally-informative notes drags the
policy toward disruptive, phase-wrong instructions. Two strikes
means this whole family of free scorers is now closed for this
model; anything further has to *learn* what a good note looks like.

## The frozen reads

| read | value | bar / comparator |
| --- | --- | --- |
| **PRIMARY (mc − self), paired CI95** | **+0.31317 [+0.19962, +0.42894]** | PASS needed CI < 0; **entirely > 0 = ANTI-SELECT** |
| Δ mc vs bare (planner-less) | +0.24453 [+0.12344, +0.36786] | the mc pick is worse than no subgoal at all |
| capture fraction of ceiling | **−1.73** | banked ceiling (ceil − self) −0.181; mc collects −173% of it |
| late-horizon signature (last-10%) | **+0.385** | ceiling's slot is −0.464 territory; SC's fingerprint was +0.055 |
| SC comparator (banked) | bon − self +0.210 [+0.113, +0.312] | mc's +0.313 is the HARDER anti-select |

Agreement diagnostics: the scorer is decidedly not inert — the mc
pick differs from the greedy text on **66.1%** of rows (inert-guard
bar was 5%; 244 tie rows) — yet it agrees with the oracle pick on
only **14.4%** of labeled rows and with SC's pick on 14.5%:
chance-level against the oracle at a 9-candidate width. Execution
oracles all green pre-read (τ/sha echo, KL-finiteness ==
eligibility, state-copy byte-match on 4,301 joined rows;
pred_masked-vs-banked flip count 1207/4301 = exactly the amendment-1
composition-noise figure, a free instrument confirmation).

## What this means

**Informativeness is anti-correlated with quality here, not merely
uncorrelated.** KL(p_cond ‖ p_masked^{1/τ}) rewards the candidate
that most bends the action distribution away from the planner-less
default. On our trunk the biggest benders are disruptive
instructions — phase-wrong, over-specific, or off-task text the
policy obediently follows — and the damage lands hardest exactly in
the late-horizon slot where real subgoal signal lives (+0.385
last-10%, the mirror image of the ceiling's −0.464). Self-certainty
failed by rewarding *plausibility*; masked-contrast fails by
rewarding *impact*. The better candidates the width provably holds
(ceiling −0.250 vs bare) sit between those poles: right-phase text
that shifts the policy moderately, which neither free axis ranks
first.

**The pre-registered kill rule executes: the zero-training scorer
family CLOSES for this trunk.** Two independent scorer philosophies
(likelihood-flavored, contrast-flavored) both anti-select on the
same frozen width with the same eligibility rules. Per the pre-reg,
learned-verifier shapes
([RoVer](../papers/rover-learned-verifier.md),
[Q-guided](../papers/qguided-flow-critic.md)) now need their own
affirmative case — cost, labels, and a reason to believe a learned
critic escapes both failure axes — not a routing inheritance from
this ladder. Candidate 2 (TOPReward-shape history phase estimation)
was pre-named as the escalation only for a *flat-late-horizon* mc
failure; the observed failure is active anti-selection, not
phase-blindness, so it does not auto-open either.

**What stays alive**: the ceiling itself. Rung (b′)'s finding is
unchanged — the sampled width holds genuinely better subgoals
(−0.250 [−0.353, −0.148] vs bare). The gap is now firmly a
*scorer* gap, measured against two closed attempts. Free
exploratory follow-up banked as a queue item: a record-only
post-mortem on the banked `[N,C]` KL + `[N,C,S,D]` per-candidate
error dump — per-candidate KL-vs-quality correlation and the
oracle pick's KL-rank histogram — to see WHERE on the
informativeness axis the good candidates actually sit before anyone
prices a learned verifier.

## Cost

Run ~1.1 GPU-h (68 f/min steady over 4,301 q4 rows × ~10 decodes +
9 reference forwards each) vs the 4.0 gate and the 2–2.5 projection
— the capture-during-decode design (no logit re-forward) is what
kept the conditioned side at single-decode price.
