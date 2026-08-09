# Pre-registration: subgoal rung (c) — masked-contrast selection (#6)

*2026-08-09 ~09:2xZ — **FINALIZED, immutable.** Instrument landed and
every named oracle is green (see the finalization block at the end);
the candidates file is sha256-pinned below. Amendments after this
stamp would be posted, dated, and never silently edited in.*

*2026-08-09 ~05:3xZ — original draft header: Finalization
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
clean candidates + greedy text; **sha256
`8175624eeb787b78cbd4363c51a35d323629ca86631c71d3ffc472067801ddad`,
pinned at finalization** — the launcher refuses any other bytes and
the read script refuses a run whose report echoes any other sha) — the
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

*Amended at finalization (09:2xZ — the third gate's comparator
corrected before any data):* rung (a)'s own amendment 1 already
FALSIFIED "byte-exact vs an npz banked at another batch composition"
as an oracle bar — greedy AR decode flips near-tie argmaxes with
kernel batch shape (measured 1207/4301 rows, mean-zero pooled). The
op-identity the gate is FOR is pinned where it is provable: (i) unit
oracles at matched composition on the real tiny decoder —
teacher-forced reference logits byte-reproduce the decode's own
captured logits over the decoded ids, and a capture-on decode is
byte-identical to capture-off (`tests/test_mcselect.py`); (ii) the
conditioned decode is the SelfSubgoalPolicy rendering path — same
collator construction, same `[generate|actions]` fast path — so
candidate-0 conditioning IS the rung-(a) op modulo composition; (iii)
the live post-run script (`mcselect_live_oracles.py`) prints the
candidate-0-vs-banked-self flip count + max |Δ| as the recorded
composition-noise diagnostic (never abort-grade, no pooled scalar)
and keeps abort-grade the composition-INdependent checks: sha/τ
echo, contract keys, KL-finiteness == eligibility, identity +
state-copy byte-match vs the banked panel rows.

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

## Finalization record (2026-08-09 ~09:2xZ, pre-launch)

The instrument landed at HEAD before this stamp, exactly the shape
the read script contracted pre-data:

- **Producer**: `--subgoal-mode mcselect` in `bijou.eval` —
  candidates-file injection (no in-run sampling), per eligible
  candidate one conditioned greedy decode with the ACTION phase's own
  logits captured (`ActionCaptureStep` — no re-forward, no numeric
  drift vs the executed decode) + one teacher-forced planner-less
  reference forward over that candidate's decoded ids against a
  snapshot/restored shared masked prefill;
  `KL(p_cond ‖ p_masked^{1/τ})` in float64 over the grammar-legal
  set, averaged over the candidate's active steps. Dump =
  `mcselect:kl` (NaN at ineligible) + `mcselect:cand_pred` +
  `mcselect:pred_masked`; report echoes `mcselect_tau` +
  `candidates_sha256`. The τ is a mandatory explicit flag — no
  silent default.
- **Oracles green**: planted-informative fixture wins the KL argmax
  with exact hand arithmetic (τ=1 identity candidate lands exactly
  0); τ→∞ collapses to `log|legal| − H(p_cond)` exactly;
  decode-vs-teacher-forced identity + capture-off byte-equality on
  the real tiny decoder; CLI flag matrix; live-oracle selftest (9
  abort branches) and the read script's own pre-data oracle both
  green (`tests/test_mcselect.py`, 15 tests, in check.py).
- **End-to-end smoke** (12 q4 rows, real checkpoint, trimmed
  plan+candidates): full pipeline rc=0, contract keys/shapes/NaN
  pattern verified, sha echoed. Measured **1.4 s/frame → ~1.7 GPU-h
  scoring projected** for 4,301 rows — inside the 2–2.5 projection,
  gate 4.0 stands. (The smoke also caught and fixed a latent
  report-stage crash that had silently cost the rung-(b′) q4 run its
  HTML — per-dataset sort keyed on the never-run bare bijou row.)
- **Launcher**: `fontaine/scripts/eval_ar100k_mcselect_q4.sh` —
  GPU-free guard, sha pins (q4 plan + candidates file), instrument
  tests + both oracle selftests re-run pre-launch, then run → live
  oracles → frozen read, each stage abort-grade before the next.
