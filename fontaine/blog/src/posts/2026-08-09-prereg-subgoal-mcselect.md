# Pre-registration DRAFT: subgoal rung (c) — masked-contrast selection (#6)

*2026-08-09 ~05:3xZ — **DRAFT, not yet immutable.** Finalization
(immutability stamp + instrument oracles green) happens before any
launch. Entry condition MET this week, in two halves: the
[rung-(b′) read](2026-08-09-subgoal-draws-cleanlist-results.md)
routed **NO-SCORER with a live ceiling** — the 8-candidate width
holds genuinely better subgoals (Δ_ceil −0.250 [−0.353, −0.148];
ceil − self −0.181 [−0.281, −0.085]) and the self-certainty scorer
anti-selects them (+0.142 vs bare) — and the
[swap read](2026-08-09-subgoal-swap-results.md) resolved the scorer
coherence question POSITIVE (content is consumed: truth beats
plausible-wrong by +0.166 [+0.127, +0.205]). The pre-registered
routing said scorer-side escalations "may earn their own pre-reg."
This is that pre-reg, for the cheapest scorer shape on the
[pre-mapped ladder](../papers/test-time-selection.md).*

## Question

Rung (b′) proved selection is the bottleneck: better-phase subgoal
texts exist in the sampled width, the likelihood-flavored scorer
picks against them, and the oracle's gain sits exactly in the
late-horizon slot where subgoal signal lives (last-10% −0.464).
**Can a masked-contrast (MG-Select-form) scorer — zero training,
the policy's own logits — find what self-certainty anti-selects?**

## The scorer (candidate 1 of the pre-mapped ladder, alone in scope)

For each banked candidate text `c` on each row: one teacher-forced
pass-2 action forward conditioned on `c`, plus ONE subgoal-masked
reference forward per row (the planner-less path — AR-100k trained
at `--subgoal-dropout 0.5`, so the masked distribution is
well-trained; the MG-Select headline configuration's 10%-dropout
prerequisite is MET at 50%). Score
`s(c) = KL( p_cond(·|c) ‖ p_masked^{1/τ} )` averaged over the
candidate's action sequence, reference tempered **τ = 4** (their
setting, adopted verbatim — not tuned on our data). Execute the
argmax: the candidate whose conditioning is most *informative*, not
most *likely* — precisely the axis on which SC failed (its pick was
plausible and phase-wrong).

*Amended 05:5xZ same day (pre-finalization, caught while landing the
read script):* the original cost line here said "no decode loop" —
wrong for the MAE side. Every rung-(b′) comparator arm's error is
**decoded**-prediction error, so the mc arm's per-candidate errors
must come from a greedy decode under each candidate (comparability,
non-negotiable). Mechanics as amended: **C greedy decodes per row**
(one per candidate; the conditional distributions for the KL are
collected during the decode) **+ C masked teacher-forced reference
forwards** on each candidate's decoded sequence. The scorer itself
still never samples — decode is greedy, KL is the score, the argmax
lives in the read script (`mcselect_results.py`, the dump contract).

Rows and candidates are FROZEN to the banked rung-(b′) artifacts
(`eval__..._subgoalcleandraws_candidates.json`, 4,301 q4 rows × 8
clean candidates + greedy text, sha256-pinned at finalization) — the
scorer re-ranks exactly the width whose ceiling and floor are
already on the board, so every rung-(b′) number is a valid
comparator by construction.

## Frozen reads + decision rule

Primary falsifier, the E6 mirror: **paired (mc − self) per-frame,
seeded bootstrap CI95** vs the banked rung-(a) self arm on the same
rows. PASS = CI95 entirely below 0 (the mc pick beats the greedy
self subgoal); FALSIFIED otherwise. This is the content-only
contrast — both sides enjoy the swap read's ~0.11 free-format
floor, so any win is phase/content value, the only thing a scorer
is for.

Secondary (record, adjudication):
- **Capture fraction** (mc − self)/(ceil − self) against the banked
  −0.181 — how much of the live ceiling the scorer collects.
- **Late-horizon signature**: the mc pick's last-10% delta —
  mechanism check against the ceiling's −0.464 territory (SC's was
  +0.055, the anti-selection fingerprint).
- **Agreement diagnostics**: mc-vs-greedy, mc-vs-SC, mc-vs-oracle
  pick agreement (SC banked 59.8/·/45.6% as comparators).

Kill/close rules: (mc − self) CI95 entirely ABOVE 0 = the scorer
**anti-selects too** — second strike after SC, and the
zero-training scorer family CLOSES for this trunk (learned-verifier
shapes — [RoVer](../papers/rover-learned-verifier.md),
[Q-guided](../papers/qguided-flow-critic.md) — would then need
their own case, not a routing inheritance). CI spanning 0 =
FALSIFIED, record-only, family stays closed per the rung-(b′)
routing. Degenerate guard: if the mc pick agrees with greedy on
> 95% of rows the read aborts (scorer inert; no verdict either
way).

## Instrument prerequisites (before finalization)

An eval path that (a) injects a GIVEN per-row subgoal text from a
candidates file (no in-run sampling), (b) emits per-row
per-candidate conditioned action-token logprob stacks + the masked
reference stack, teacher-forced. Oracle gates: a planted-informative
synthetic fixture (one candidate constructed to sharpen the action
distribution must win the KL argmax); τ→∞ must reduce the score to
conditioned-vs-uniform-reference degeneracy check; re-scoring the
banked greedy text must reproduce the rung-(a) self arm's
conditioned loss on a spot-check subset byte-exactly.

## Cost + gates

Per the amendment above: ~9 greedy decodes + 9 reference forwards ×
4,301 rows on the local H100 — comparable single-pass decodes ran
~540 f/min, so **~2–2.5 GPU-h projected; gate ≤ 4 GPU-h** (babysit
entry at launch, q4-shaped rate check at first poll). Zero
training, zero box time. Out of scope, named for the map: candidate 2
(history-conditioned phase estimation, the TOPReward shape — M
effort, needs episode-prefix plumbing our panel doesn't have) stays
the escalation IF mc fails specifically on phase (late-horizon
signature flat while ceiling stays alive).
