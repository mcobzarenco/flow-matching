# Pre-registration: self-subgoal conditioning probe (#6 rung (a))

*2026-08-07 ~03:5xZ. Immutable once posted. Ideas [#6](../ideas.md) /
[#11](../ideas.md), from the
[π0.5 deep read](2026-08-07-pi05-deep-read.md) (explicit-HL is their
untested-by-us runtime increment) and the
[Hi-VLA study](https://arxiv.org/abs/2606.10267) (2606.10267:
explicit language subgoals gain most on long horizon; SELF-generated
subgoals untested there — ours is an increment, not a replication).
Zero training. The instrument does NOT exist yet: it lands
oracle-gated before launch, and if implementation forces any semantic
deviation from this post, an amendment posts before launch (the
[#19 amendment](2026-08-06-prereg-ar-sampled-draws.md) precedent).*

## Question

Does explicit runtime hierarchy — the AR model decoding its own
subgoal for a frame, then conditioning its action decode on that text
through the trained `[subgoal|…]` prompt slot — beat the planner-less
deployment baseline (AR-100k greedy panel 5.8026)? We already know
semantic co-training shapes the action representation (aux-off costs
+0.462, CI [0.387, 0.537] — the Implicit-HL replication). What has
never run is the runtime loop: generate the plan, feed it back as an
input. AR-100k is the right probe body: it trained the subgoal
condition slot (`condition_fields = [subgoal, outcome, smoothness]`)
at `subgoal_dropout = 0.5`, so the planner-less context is
well-trained AND the hint slot saw real text; its aux head generates
subgoals (owner steer 08-05 21:43Z: they generalize strikingly OOD).

## Instrument (to land, oracle-gated, before launch)

`bijou.eval` gains a subgoal-conditioning mode for ar_backbone
checkpoints (semantics frozen here; flag spelling is implementation's):

- **oracle arm** — include `SUBGOAL` in the collator's condition
  fields with no override: each frame renders its TRUE segment label
  via the existing `subgoal_text` path (`[subgoal|…]` trailing
  bracket, dropout 0); frames without a judge label render nothing
  and decode identically to baseline. Today this path is reachable
  only through a single global `--condition-override subgoal=X`; the
  instrument makes per-frame truth conditioning a first-class mode.
- **self arm** — a two-pass policy sharing one model load (the
  `NarratedBijouPolicy` pattern): pass 1 greedy-decodes the subgoal
  value line under a `[generate|subgoal actions]` request
  (planner-less prompt; its actions are retained — the
  narrated-subgoal-only arm comes free); pass 2 re-encodes with
  `item["condition_subgoal"] = <pass-1 text>` rendered in the prompt
  slot and decodes actions on the deployment fast path
  `[generate|actions]`. Pass 2 must NOT request subgoal generation —
  training's anti-copy coupling (`suppress_subgoal`) means
  condition-plus-generate never co-occurred, while condition-plus-
  fast-path is exactly the trained conditioned context.
- **provenance**: policy names carry the mode (`_selfsubgoal`,
  `_oraclesubgoal`, `_narrsubgoal` — a conditioned read must never
  pass as the deployment read, charter §2); report JSON records the
  mode; per-frame generated subgoals are retained machine-readable
  (frame identity triple → text) for the validity table and the
  results post's qualitative block.
- **oracles (abort-on-red before launch, the usual gate):** (i) the
  self arm with its generated text forced EMPTY reproduces the
  baseline decode bit-exact (the no-hint limit is the historical
  path); (ii) the oracle arm on a label-less item ≡ baseline;
  (iii) conditioned prompt bytes match the training collator's
  rendering of the same text (one rendering path, not a re-
  implementation); (iv) pass 2's generate list excludes subgoal.

## Design — two stages, gated in order

**Stage 1 — validity table (eyes before any scalar; the
never-generated-subgoal scar is the reason this stage exists and
comes first).** Generate self-subgoals for a fixed-seed (seed 0)
sample of 60 panel frames stratified across episodes/repos, table:
frame identity, instruction, TRUE segment label (or —), generated
subgoal. Read and commented in the results post, not just attached.
**Pre-registered go/no-go for stage 2's self arm:** (a) non-empty,
non-truncated text on ≥ 90% of rows; (b) no single generated string
on > 50% of rows (degenerate-collapse check); (c) qualitatively
subgoal-shaped — imperative manipulation clauses, not instruction
echoes or judge-artifact fragments (eyes, commented row-by-row where
failing). Fail → the self arm does NOT run; the failure is the
rung-(a) result for generation quality, recorded with the table. The
**oracle arm runs regardless** — it answers whether the slot is live
at all, which stage-1 failure does not touch.

**Stage 2 — scalar arms**, identical rows via the shared plan file
(`plans/holdout_curated_v0_k4l2.json`), AR-100k
(`bijou_arb_rcond_100k_ddp4/step_100000`), seed 0, k4l2 panel:

| arm | prompt slot | suffix request | source of text |
|---|---|---|---|
| baseline (banked, no re-run) | — | `[generate\|actions]` | — (5.8026 / 2.1431) |
| oracle-subgoal | `[subgoal\|…]` | `[generate\|actions]` | true segment label |
| self-subgoal | `[subgoal\|…]` | `[generate\|actions]` | pass-1 generation |
| narrated-subgoal (free, pass 1) | — | `[generate\|subgoal actions]` | its own, in-suffix |

Banked context row: the all-fields narrated arm is 5.8565 (+0.054 vs
baseline — narrating everything slightly hurts).

## Frozen reads

Paired per-row, seeded bootstrap 95% CI (seed 0, 10,000 resamples —
the draws-fairness assembly conventions). "Labeled subset" = frames
where `subgoal_text` is non-None; baseline re-pooled onto any subset
from its banked npz (the state-probe precedent), never re-run.

1. **Primary: Δ_self = chunk_mae(self-subgoal) − 5.8026** on all
   core frames (deployment-honest: a planner-less rig can always
   self-generate), with the labeled-subset value quoted beside it.
2. **Bound: Δ_oracle = chunk_mae(oracle) − baseline on the labeled
   subset.** Interpretive frame, fixed now: Δ_oracle bounds what the
   slot can transmit. Δ_oracle ≈ 0 ⇒ the slot is inert (0.5-dropout
   training made the hint ignorable) and rung (a) closes regardless
   of generation quality; Δ_oracle < 0 with Δ_self ≥ 0 ⇒ the gap is
   generation quality, not the slot.
3. **Channel read: Δ_narr (suffix voice) vs Δ_self (prompt slot)** —
   nearly the same text, different entry point; separates "where the
   text enters" from "whether text helps".
4. **Horizon decomposition**: per-step-in-horizon MAE curves for
   every arm from `--dump-predictions` npz, the
   `fontaine/scripts/flow_vs_ar_paired.py` conventions. The Hi-VLA
   anchor predicts the gain concentrates LATE-horizon (their flat
   25.30% → 67.08% gap is long-horizon-only), i.e. chunk-tail moves,
   first_mae barely.
5. first_mae mirrors of 1–3.
6. Execution oracles (abort): state-copy / state-copy-norm rows
   byte-match banked panel values; names/JSON carry the modes.

## Numbered expectations (banked before data)

1. Δ_oracle < 0 on the labeled subset (a true hint helps a
   condition-trained model) — confidence medium.
2. Δ_self < 0 but |Δ_self| < |Δ_oracle| (self text is a noisy
   version of truth) — confidence medium-low; this is the probe's
   genuinely open number.
3. The gain (if any) concentrates late-horizon; first_mae moves
   little — confidence medium (external anchor, our #1 precedent
   shows such predictions can half-fail).
4. Δ_narr ≈ +0.05-ish (suffix narration keeps slightly hurting, as
   the all-fields arm did) and does NOT beat Δ_self — confidence low
   (record-only comparison).
5. **Falsified if Δ_self ≥ 0**: explicit self-hierarchy gives
   nothing at panel granularity on this body. Recorded with the
   2-vs-oracle diagnostic split; any escalation (rollout-granularity
   refresh policies, planner-side work, subgoal-quality training)
   needs a NEW pre-reg citing this result. No prompt fishing, no
   post-hoc subgoal re-phrasing.

## Cost & scheduling

All arms are greedy panel evals (banked rate 0.081 s/frame → ~35 min
each; the self arm ~2× for its two encode+decode passes). Stage 1
≈ minutes. Pre-registered ceiling **≤ 8 GPU-h total**; if a first-200-
frame rate measurement projects past it, all arms drop to the frozen
q4 subset (4,301 rows; the #19 clause verbatim) and the switch is
recorded. Venue: local GPU, first quiet window at or after the
draws10_t1 boundary AND its frozen reads (that pre-reg's obligations
come first); never co-located with a training run's eval chain.
First-poll util+rate check per standing rule.

## Amendment 1 — oracle-(i)/(ii) comparator (2026-08-08 ~00:1xZ, posted BEFORE the stage-2 launch)

*The pre-launch adjudication run falsified an assumption inside the
oracle spec, not the oracle's semantics. Recorded here per this post's
own amendment clause; no read, arm, gate or expectation changes.*

The live oracle-(i) run (q4 subset, `--selfsubgoal-force-empty`)
was NOT bit-exact against the banked full-panel baseline npz:
1207/4,301 rows differed. Diagnosis, in order run:

- state-copy / state-copy-norm rows byte-match the banked panel
  (plan/data alignment exact); differing rows hit subgoal-labeled and
  label-less rows at the same rate (no hint-leak signature);
- the pooled effect of the differing rows is **−0.0008 chunk MAE**
  (per-frame CI [−0.016, +0.015] — mean-zero decode noise, not a
  systematic shift);
- a **plain baseline eval on the same q4 plan with zero instrument
  code involved** reproduces the same class of row flips against the
  banked npz (count quoted in the adjudication log). Mechanism:
  greedy AR decode is batch-composition-sensitive at the kernel level
  — padding/shape-dependent reduction order perturbs logits at ulp
  scale, near-tie argmaxes flip, and one flipped token cascades
  through the row. Action quantiles are per-item (verified at
  `interface.py::_stats`), so bins are composition-independent —
  kernel numerics are the only channel.

**Amended comparator (semantics unchanged: "the no-hint limit is the
plain path"):** oracle (i) is adjudicated bit-exact against a plain
baseline decode of the SAME plan at the SAME batch composition (the
`stateprobe_q4_diagbaseline` run), not against the banked full-panel
npz. Oracle (ii)'s label-less half is decode-checkable only up to the
same composition noise (label-bearing batchmates change padding), so
its abort-grade live form is the wiring check (≥ 1 labeled row moves
the decode) and the label-less byte-equality stands on the pinned CPU
prompt-byte oracle; the label-less decode count is recorded
descriptively. All composition-independent execution oracles keep
their banked comparator and abort grade: identity columns, state-copy
rows, provenance fields.

Consequence for the frozen reads: none. The stage-2 arms run the full
panel — the SAME plan, order and batch size as the banked baseline —
and the paired reads keep the banked npz as baseline, exactly as
frozen. The measured mean-zero composition noise (−0.0008 pooled,
CI ±0.016 per-frame) is quoted in the results post beside Δ_self as
the decode-noise floor context; it does not modify the read
definitions or the E5 falsifier.
