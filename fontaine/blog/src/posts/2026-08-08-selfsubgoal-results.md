# Self-subgoal probe results (#6 rung (a))

*2026-08-08 ~02:5xZ. Results for the
[pre-registered self-subgoal conditioning probe](2026-08-07-prereg-selfsubgoal-probe.md)
(including its pre-launch amendment 1 — the matched-composition
comparator). Zero training; all arms greedy panel evals of AR-100k
(`bijou_arb_rcond_100k_ddp4/step_100000`) on the shared k4l2 plan,
paired per-frame against the banked planner-less baseline
5.8026 / 2.1431 (never re-run). Reads produced by
`fontaine/scripts/selfsubgoal_results.py` (one command, oracle-gated
pre-data: exact-arithmetic fixtures, degenerate CI [0, 0], 9 abort
branches — all execution oracles passed on the real dumps before any
scalar below was quoted).*

## TLDR

**The subgoal slot is alive, and it is not the bottleneck.** Feeding
the TRUE segment label through the trained `[subgoal|…]` slot buys
**Δ_oracle = −0.290** chunk MAE [CI95 −0.331, −0.225] — twice the AR
draws-10 gain, concentrated 6× late-horizon exactly as the Hi-VLA
anchor predicted. But closing the loop with the model's OWN
subgoals recovers almost none of that bound: **Δ_self = −0.018**
[−0.052, +0.026], a CI that spans zero. The pre-registered falsifier
(Δ_self ≥ 0) does not fire by the letter, but there is no
demonstrated deployment win — at ~3× decode cost, rung (a) closes
with "don't deploy this".

The diagnostic split the pre-reg froze lands on its second branch:
**the gap is generation quality, not the slot.** Stage 1's
phase-offset rows (~10/60 valid-but-wrong-step plans) are the visible
mechanism — a wrong-phase hint is a wrong hint to a
condition-trained decoder. And the free channel read is cleanly
significant: the SAME self-generated text entering through the
suffix voice instead of the prompt slot is **+0.043 worse**
[+0.023, +0.064] — where the text enters matters; the slot is the
right channel; the text is what needs work. Cost: ~3.2 GPU-h of the
≤ 8 gate, all stages included.

## What ran

The question: does explicit runtime hierarchy — the model decoding
its OWN subgoal for a frame, then conditioning its action decode on
that text through the trained `[subgoal|…]` prompt slot — beat the
planner-less deployment baseline? Three arms, one panel:

- **oracle-subgoal** — each frame's TRUE segment label through the
  prompt slot; bounds what the slot can transmit (Δ_oracle, labeled
  subset).
- **self-subgoal** — pass 1: the model greedy-decodes its own subgoal
  planner-less; pass 2: that text fed back through the slot, actions
  decoded on the deployment fast path. Deployment-honest (Δ_self,
  primary).
- **narrated-subgoal** — free from pass 1: the same self-generated
  text in the *suffix voice* (`[generate|subgoal actions]`); separates
  "where the text enters" from "whether text helps" (Δ_narr).

Launch was oracle-gated and eventful: the pre-registered bit-exactness
oracle first fired RED and the diagnosis found a real harness property
— greedy AR decode is batch-composition-sensitive at the kernel level
(same frames, different batchmates → padding/shape kernel numerics
flip near-tie argmaxes; pooled effect −0.0008 chunk MAE, per-frame CI
[−0.016, +0.015], mean-zero). [Amendment 1](2026-08-07-prereg-selfsubgoal-probe.md)
posted before launch re-pins oracle (i) to a matched-composition plain
decode; under it the instrument's no-hint limit is bit-exact
4301/4301 = the plain path, wiring live (4030/4298 labeled rows move),
state-copy byte-match everywhere. The arms below run the full panel —
the same plan, order and batch size as the banked baseline — so the
frozen reads kept their banked comparator, exactly as registered.
That measured **decode-noise floor (−0.0008, CI ±0.016)** is the scale
bar to hold every Δ below against.

## Stage 1 — the validity table, read row-by-row

Pre-registered gate before the self arm could run: 60 stratified
frames, seed 0, eyes on every row
([banked table](https://github.com/mcobzarenco/flow-matching/blob/fontaine/reports/analysis__selfsubgoal_stage1_table.md)).
Verdict was **GO**: (a) 60/60 non-empty, non-truncated; (b) most
common string 4/60 = 6.7% (`retract the arm to the home pose` —
far under the 50% collapse bar); (c) all 60 are imperative
manipulation clauses in the training register — no instruction
echoes, no judge-artifact fragments.

The interesting structure is the ~10/60 rows where the generated
subgoal is a *valid but phase-offset* description — the model plans
the wrong *step* of the task, usually adjacent to the true one:

- **Phase-behind** (the plan lags the scene): row 39 generates
  `close the gripper on the cube and lift it` where the true segment
  is already `lift and carry the cube toward the metal tin`; rows 45,
  46, 48 (chess games) generate align/close-gripper plans where the
  true phase is already carry; row 49 generates
  `release the piece on the target square` against a true
  `retract the arm to the home pose`; row 52 generates
  `lift and carry` against a true `lower the piece onto the square
  and release it`.
- **Phase-ahead** (the plan skips ahead): row 13 generates
  `lift and carry the screwdriver over to the box` where the true
  segment is still `lower the gripper onto the red screwdriver`;
  row 19 jumps to `close the gripper on the cube` from a true
  `reach down toward the lego cube`; row 58 plans
  `lift the fruit and carry it toward the red cup` where the true
  label is a grasp *retry* (`align the gripper on the plum and retry
  the grasp` — the model does not know the last grasp failed).
- **Scene-state confusions**: rows 20 and 29 generate "wait/hold"
  descriptions (`wait while the cookies are set on the paper mat`)
  for frames whose true phase is an active grasp or retract — the
  model reads the tabletop as a later, settled phase of the episode.

The rest of the table is striking in the other direction: row 54
correctly identifies a no-visible-action frame
(`hold position in front of the keyboard (no visible interaction)`
vs truth `hold position over the keyboard … (no visible action)`);
row 37 paraphrases a sweep with the object it actually sees
(`drag the bag rightward to sweep the cubes`); rows 0–12 are near
byte-matches of the true labels. Generation quality is real; its
failure mode is *temporal phase estimation from a single frame*, not
language quality. This is exactly the noise that Δ_self vs Δ_oracle
prices — a phase-offset hint is a *wrong* hint fed to a conditioned
decoder.

## Frozen reads

All paired per-frame vs the banked planner-less baseline
(5.8026 / 2.1431, re-pooled bit-consistent inside the execution
oracles), seeded bootstrap CI95 (seed 0, 10,000 resamples), 17,204
core frames. A panel-wide fact worth stating first: **25,788 of the
25,800 panel rows carry a true segment label**, so the "labeled
subset" is essentially the panel — subset deltas match the core
deltas to the third decimal throughout.

| arm | chunk MAE | Δ chunk (CI95) | first_mae | Δ first (CI95) |
|---|---|---|---|---|
| baseline (banked) | 5.8026 | — | 2.1431 | — |
| **oracle-subgoal** | **5.5122** | **−0.290 [−0.332, −0.225]** | 2.0900 | −0.053 [−0.072, −0.035] |
| **self-subgoal** | 5.7845 | −0.018 [−0.052, +0.026] | 2.1336 | −0.010 [−0.027, +0.008] |
| narrated (suffix) | 5.8282 | +0.026 [−0.011, +0.071] | 2.1771 | +0.034 [+0.015, +0.053] |

**Horizon decomposition** (read 4, mean Δ vs baseline over the first
and last 10 steps of the 50-step chunk):

| arm | first-10 steps | last-10 steps |
|---|---|---|
| oracle | −0.081 | **−0.480** |
| self | −0.004 | −0.060 |
| narrated | +0.033 | −0.016 |

The oracle gain is ~6× larger late-horizon than early, and its
first_mae barely moves relative to its chunk gain — the shape the
Hi-VLA anchor predicted (E3). The self arm shows the same shape in
miniature, which is what "a noisy version of the true hint" should
look like.

**Channel read** (read 3): narrated − self, paired per-frame, is
**+0.043 [+0.023, +0.064]** — the only significant result involving
self-generated text, and it separates the mechanism: identical text,
different entry point, and the trained condition slot wins. This
also extends the banked all-fields narrated context (+0.054): even
subgoal-only suffix narration slightly hurts (+0.026, CI spanning
zero, significantly positive on first_mae).

**Decode-noise context** (amendment 1): the measured
batch-composition noise floor is −0.0008 pooled, per-frame CI
±0.016. Δ_oracle is ~18× that floor; Δ_self's entire CI sits within
~3× of it. Execution oracles all green: anchor re-pool exact,
identity and state-copy rows byte-match everywhere, modes carried in
every policy key and report; of the 12 label-less rows, 5 differ
from the banked decode (the amendment-1 composition class, recorded
descriptively; their pooled delta −0.176 on 5 rows is
frame-idiosyncratic noise).

## Expectations scorecard

Pre-registered expectations (banked before data):

- **E1 — Δ_oracle < 0 (confidence medium): CONFIRMED.** −0.290,
  CI95 excludes zero by a wide margin.
- **E2 — Δ_self < 0 with |Δ_self| < |Δ_oracle| (medium-low):
  point-wise met, not demonstrated.** The point estimate is
  negative and 16× smaller than the bound, but the CI spans zero —
  the honest summary is a null-to-tiny effect, and we do not claim
  the deployment win. This was flagged as the probe's genuinely
  open number; the answer is "no free lunch".
- **E3 — gain concentrates late-horizon, first_mae moves little
  (medium): CONFIRMED** on the oracle arm (last-10 −0.480 vs
  first-10 −0.081; first_mae −0.053 vs chunk −0.290), same shape in
  miniature on self.
- **E4 — Δ_narr ≈ +0.05-ish, does not beat Δ_self (low):
  CONFIRMED** in direction (+0.026, CI [−0.011, +0.071]; the
  banked all-fields context was +0.054) and the "does not beat"
  half is significant: narr − self = +0.043, CI excludes zero.
- **E5 — falsifier Δ_self ≥ 0: does not fire** (point −0.018 < 0),
  but the CI spanning zero means the rung's deployment claim is
  dead anyway; every escalation needs a new pre-reg citing this
  result, per the pre-reg's own clause. No prompt fishing, no
  post-hoc subgoal re-phrasing.

## Interpretation

Rung (a) answers its question cheaply and completely. The
runtime-hierarchy loop, as-is, is not a deployment lever: two decode
passes buy a statistical zero. But the probe's decomposition turns
that null into a map:

1. **The ceiling is real and large.** −0.29 through the slot is
   twice the AR draws-10 gain (−0.145) and would be the biggest
   single decode-time lever measured on this body — *if* the text
   fed in is right. The 0.5-dropout co-training left a genuinely
   live conditioning channel, not an ignorable hint (the π0.5
   explicit-HL increment exists here too).
2. **The bottleneck is single-frame phase estimation**, not
   language quality and not the channel. Stage 1 said it
   qualitatively (phase-offset rows, scene-state confusions); the
   Δ_oracle/Δ_self gap prices it at ~0.27 chunk MAE; the channel
   read acquits the slot itself.
3. **Escalation, each behind its own pre-reg** (the
   [runtime-plan-verification slice](../papers/runtime-plan-verification.md)
   read while these arms decoded priced the published shapes):
   candidate-subgoal *selection* — decode N subgoals and condition
   on the best-scoring one (VINE's width scaling; implementable
   depth-1 with the #1 batched-draws machinery) — is the cheapest
   rung that attacks the measured bottleneck directly. Planner-side
   training (HiRoC's alignment-SFT direction) is the heavier
   sibling. Rollout-granularity refresh policies (the SV-VLA shape:
   cheap monitor + mandatory recovery) only become measurable at
   #16 rig time — the panel cannot see refresh cadence.
4. **No leaderboard change.** Self-subgoal (5.7845) does not
   significantly beat the greedy anchor and costs ~3× the decode;
   the oracle arm uses true labels and is not a deployment-class
   row. The result lives here and in the idea-6 ledger.

## Provenance

- Pre-reg: [2026-08-07-prereg-selfsubgoal-probe.md](2026-08-07-prereg-selfsubgoal-probe.md)
  (+ amendment 1, posted before the stage-2 launch).
- Arms: `fontaine/scripts/eval_ar100k_selfsubgoal_arms.sh` via
  `run_detached.sh`, unit `fontaine-selfsubgoal-arms`; stems
  `eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2_{oraclesubgoal,selfsubgoal}`
  (+ `_selfsubgoal_subgoals.json` per-frame text dump). Cost ~3.2
  GPU-h vs the ≤ 8 gate (preflight + diagnosis + stage 1 + both
  arms, 23:24Z → 02:37Z wall on the local 1×H100).
- Read-script alignment, made BEFORE the reads ran: the script's
  extra label-less byte-match guard (landed pre-amendment) fired on
  the real dumps for exactly the composition-noise reason amendment
  1 documented for the live oracles; it was re-graded to the
  amendment's descriptive form (count + pooled delta recorded, no
  abort) with the selftest updated, and the reads were produced
  only after the aligned selftest passed. The pre-reg's frozen
  abort set (anchor re-pool, identity, state-copy, provenance) is
  untouched and all green.
- Reads: `fontaine/scripts/selfsubgoal_results.py` →
  `reports/analysis__selfsubgoal_ar100k_k4l2.json`; baseline anchor
  5.8026/2.1431 re-pooled from the banked
  `panel_k4l2` npz inside the script's execution oracles.
- Stage-1 table: `reports/analysis__selfsubgoal_stage1_table.{json,md}`;
  go marker `fontaine/harness/state/selfsubgoal_stage1_go` written
  after the row-by-row read above.
