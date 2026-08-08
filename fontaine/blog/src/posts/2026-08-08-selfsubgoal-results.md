# Self-subgoal probe results (#6 rung (a))

*2026-08-08 ~0x:xxZ. Results for the
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

«TLDR-PLACEHOLDER»

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

«READS-PLACEHOLDER — table + horizon curves from selfsubgoal_results.py»

## Expectations scorecard

Pre-registered expectations (banked before data):

«SCORECARD-PLACEHOLDER — E1..E5 dispositions»

## Interpretation

«INTERPRETATION-PLACEHOLDER»

## Provenance

- Pre-reg: [2026-08-07-prereg-selfsubgoal-probe.md](2026-08-07-prereg-selfsubgoal-probe.md)
  (+ amendment 1, posted before the stage-2 launch).
- Arms: `fontaine/scripts/eval_ar100k_selfsubgoal_arms.sh` via
  `run_detached.sh`, unit `fontaine-selfsubgoal-arms`; stems
  `eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2_{oraclesubgoal,selfsubgoal}`
  (+ `_selfsubgoal_subgoals.json` per-frame text dump). Cost
  «GPU-H-PLACEHOLDER» GPU-h vs the ≤ 8 gate.
- Reads: `fontaine/scripts/selfsubgoal_results.py` →
  `reports/analysis__selfsubgoal_reads.json`; baseline anchor
  5.8026/2.1431 re-pooled from the banked
  `panel_k4l2` npz inside the script's execution oracles.
- Stage-1 table: `reports/analysis__selfsubgoal_stage1_table.{json,md}`;
  go marker `fontaine/harness/state/selfsubgoal_stage1_go` written
  after the row-by-row read above.
