# Pre-reg (DRAFT) — GRPO R2 on the grasp-SFT policy: RL pressure on a competent base

*2026-08-15, drafted ~07:5xZ while stage C of the [grasp-SFT
bootstrap](2026-08-14-prereg-grasp-sft-bootstrap.md) trains (313-demo
set, AR primary, launched 07:29:55Z). Status: **DRAFT, conditional** —
this registration activates ONLY if stage D's frozen primary reads
**GRPO GO (≥ 20/100)**; a 5–19 read routes to the bootstrap's own
iterate-once arm first, and < 5 parks this page (F-transfer makes
visual work the binding lane, not RL). Finalization (frozen params +
objection window + HEAD re-pin) happens AFTER the stage-D verdict
banks, per Decision 11: any new GRPO run is a fresh pre-reg on the
first-class stack. Nothing here launches before that.*

**Plain words.** We stopped our last reinforcement-learning run with a
verdict worth restating: the robot was too incompetent for incentives
to matter. It succeeded so rarely (roughly 1 try in 20 at best) that
the training signal was mostly noise about *shoving*, and even after
we fixed the scoring rule so shoving never pays, the run went nowhere —
the conclusion was "teach the robot to grasp first, then come back."
That teaching is happening right now: a scripted expert generated 313
successful demonstrations, and a policy is being fine-tuned on them as
this draft is written. If that policy passes its exam (at least 20
successes on the 100 held-out test scenarios), reinforcement learning
finally has something to work with: when the robot succeeds 1 time in
5 rather than 1 in 100, every batch of attempts contains both
successes and failures to compare, which is exactly the contrast this
kind of training amplifies. This document pre-commits how that run
would work — same scoring rule, same safety wires, measured budget —
so the go/no-go decision is mechanical when the exam result lands.

## §1 Hypothesis — why R2 stops being moot

The [R1-B boundary read](2026-08-14-prereg-token-grpo-phase2-r1b.md)
closed phase 2 on surface A with "the R2 pricing discussion is moot"
— **at that policy's competence** (successes 4/3/3 of 64 per wave;
held-out 2/20 unchanged). The registered next shape was grasp-rich
SFT before RL pressure, and the [bootstrap
pre-reg](2026-08-14-prereg-grasp-sft-bootstrap.md) §4 F-live froze
the return path: ≥ 20/100 → GRPO registers fresh with
success-variance groups.

**H:** on a base policy at success rate p ≥ 0.20, GRPO's group
advantages are carried by the **success term** (+10, the largest
reward magnitude) rather than by centimeter-scale progress noise, and
gentle re-pricing measurably accumulates held-out success where R1-A/B
measured flat.

The arithmetic of the claim: a group of 8 draws on one seed is
informative when it mixes successes and failures — probability
1 − (1−p)⁸ − p⁸. At R1-B's measured competence (p ≈ 0.05) that is
~34%, and the mixed groups' contrast was a ±10 spike on top of a
reward otherwise dominated by shove/progress terms the policy could
not control. At the stage-D gate minimum (p = 0.20) it is **~83% of
groups**, each carrying a ±10 contrast between trajectories the base
policy itself produced. Same loop, same reward — the signal changes
class because the base changed class.

## §2 Run design (frozen at draft; finalization fills §6)

- **Stack (Decision 11)**: the first-class new stack —
  `bijou/grpo_replay.py` / `MolmoAct2DiscreteStack` over bijou
  checkpoints. FRESH run; the banked R1-A/B `.pt` endpoints stay
  salvage-only, nothing resumes across the retirement re-point.
  Full-width Gumbel noted: sampled streams are not comparable
  draw-for-draw with the banked port waves.
- **Base policy**: the stage-C AR endpoint **through the same two-hop
  conversion stage D evaluates** (`bijou.convert_molmoact2`, demo-set
  recomputed norm_stats riding the HF dir, no shim per bootstrap §6)
  — GRPO starts from byte-identically the policy whose 20+/100 was
  measured. Checkpoint id + conversion receipt pinned at
  finalization.
- **Reward**: `composite_reward_v2` trained-on (grasped-progress
  only, 0.5 ungrasped charge direction-blind, +10 success −2 tipped
  −5 strike), **held-out metric stays v1** — both unchanged from
  R1-B, keeping every banked pairing comparable.
- **Groups**: 8 seeds × 8 draws, temperature 1.0, advantage clip
  ±2.0 — unchanged surface-A shape.
- **Levers (draft proposal, finalization decision)**: lr **1e-6**,
  kl_beta **1.0**. Rationale to be judged at finalization: R1-B's
  3e-7 measured a greedy policy digit-identical across waves (it
  priced a null), and the escalation that motivated the ÷3.3
  re-price was the shove-funding failure mode — since retired by v2
  + the competence change. 3e-7 stays the registered fallback if
  wave telemetry runs hot. kl_stop 0.06 unchanged.
- **Tripwires**: the full R1 §7 set inherited verbatim — strikes,
  non-finite loss, spread collapse ×3, knock-away 2× ×3 (the
  in-reward fix does not retire the belt), kl_stop, competence-floor
  CI. `setback_frac` now HAS a baseline (R1-B banked 0.703 / 0.5625
  / 0.5938); promoting it from record-only to a wire is a
  finalization decision, default record-only.

## §3 Registered reads

- **PRIMARY**: held-out sim100 (frozen seeds 0–99, harness
  `success_tick`) at the boundary vs the banked stage-D base count —
  paired per-seed, exact test. Success count materially above base
  with the paired read clean → **accumulation confirmed**, the
  SFT+GRPO alternation flywheel (success-recycle into the demo set)
  prices its next rung. Flat with wires quiet → banked negative:
  competence was necessary but not sufficient at this pressure —
  a real result, not a retry license.
- **Wave-0 calibration read** (the success-variance claim's first
  live contact): fraction of groups mixed (predicted ≳ 60% at
  p ≥ 0.2 base), advantage mass attributable to the ±10 term.
  Registered bar patterned on R1-B §4: if the groups do NOT carry
  success variance at the measured base rate, that is a calibration
  FAIL and stops the run before GPU-hours burn — the group shape
  (seeds × draws) is the amendment path, not lr.
- **Behavior reads (record + judge at boundary)**: knockaway_frac /
  setback_frac / earned-vs-ungrasped decomposition on a competent
  base — R1-B's finding was that these are competence artifacts; the
  clean prediction is they FALL as success rises. If knock-away
  re-fires on a ≥ 20% base, the competence-artifact story is wrong
  somewhere and the wire analysis is the deliverable.

## §4 Falsifiers

- **F-flat** — success does not move at gentle pressure on a
  competent base: GRPO-on-this-surface banked negative twice
  (incompetent AND competent base); the 90% path re-prices toward
  more SFT rounds / demo scale instead of RL.
- **F-instability** — wires fire early on the competent base: the
  pricing (lr/β) is wrong for this base, one registered re-price
  amendment allowed (the R1 ladder's discipline).
- **F-regression** — held-out success materially BELOW base at the
  boundary: KL anchoring insufficient; stop, bank, and the
  alternation flywheel inherits the diagnosis.

## §5 Cost (measured pace)

R1-B measured ~0.98 GPU-h/step including the per-step held-out eval.
Proposal: **10 steps ≈ ~10 GPU-h, gate ≤ 12**, mid-ladder read at
wave 5, ~30-min babysit cadence, self-stops armed. This is a fresh
budget — the bootstrap's ≤ 13 gate closes with stage D and does not
carry over.

## §6 Finalization checklist (fills when stage D banks GRPO GO)

1. Stage-D verdict + the measured base success count (the PRIMARY's
   anchor) recorded here.
2. Base checkpoint id + conversion receipt (the exact bijou dir
   stage D evaluated).
3. lr decision (1e-6 primary vs 3e-7 fallback) judged against the
   stage-D read + a dry advantage-decomposition pass on wave-0 rows.
4. setback_frac: wire or record-only.
5. Rollout seed policy for training groups (fresh stream per the
   standing variance branch; eval seeds 0–99 stay the frozen
   holdout, never trained on).
6. HEAD re-pin, objection window opened in-channel, owner go or
   window expiry before launch.

## §7 Amendment A1 (pre-finalization, 2026-08-15 10:3xZ) — the head seam, and two owner decisions

Registered mid-ride after the owner's in-channel review (09:49–10:14Z
exchange); nothing here was launched, so this amends a DRAFT.

**The seam (owner-caught).** As drafted, §2 is inconsistent: stage D
measures the **flow head** (euler-10 through the converted action
expert — the only module stage C trains), but the named serving stack
(`MolmoAct2DiscreteStack`) trains the **AR/FAST token head**, whose
weights stage-C SFT never touches (`--ft_vlm=false`). Token-GRPO from
that base would start from the *released* model's discrete policy —
not the competent base this pre-reg's §1 premise requires. The
"byte-identical to the policy stage D measured" claim held for the
checkpoint dir, not for the policy the loop would actually push on.

**Owner decisions registered (2026-08-15, in-channel):**

1. **Their trainer is retired for us** (10:07Z): no further runs on
   molmoact2 `train_lerobot.py`; all training goes through
   `bijou.train` / first-class objects. The stage-C AR run (killed
   10:11Z at step 2040 on the owner's order, checkpoints 500–2000
   retained) was the last their-stack run.
2. **Direction for this pre-reg**: if token-GRPO is the tool, a
   **token-SFT arm must precede it** — a `bijou.train` run (first-class
   `ar` objective) over the same demo set, so the discrete head has
   measured competence before RL pressure. That arm would need its own
   pre-registration and its own sim eval; this R2 draft then re-bases
   on *that* endpoint (or re-scopes to whichever head the owner picks).

**Status.** The bootstrap's stage-D formalism is suspended pending the
owner's re-steer after the step2000 probe (train-vs-unseen seeds, live
at amendment time). §§1–6 above are retained verbatim as the registered
record of the draft's reasoning; activation now requires BOTH a
competent-base verdict on the head GRPO would train AND the owner's
choice of route through decision 2.

## §8 Amendment A2 (pre-finalization, 2026-08-15 14:4xZ) — the token-SFT arm is now a registered draft; re-basing spelled out

A1's decision 2 said the token-SFT arm "would need its own
pre-registration"; that document now exists — [token-SFT arm
pre-reg (DRAFT)](2026-08-15-prereg-grasp-sft-token-sft-arm.md) — and
this amendment binds the two:

1. **Re-basing.** If the owner routes token-GRPO (route B or C in the
   arm's §4), R2's **base policy** (§2 bullet 2) becomes the token-SFT
   arm's endpoint, and the **activation bar** becomes that arm's
   primary read: unseen sim100 **≥ 20/100 on the discrete head under
   greedy decode** — the same competence bar, measured on the head
   this run actually pushes. The stage-D flow-head verdict no longer
   activates this page (it measured the wrong head; A1's finding).
   §3's PRIMARY comparator becomes the token arm's banked unseen
   count.
2. **Table lineage.** The arm inits from the corrected-table base
   (verified in code this session: `bijou/fast/codec.py` normalizes
   token targets with the baked q01/q99), so R2 inherits the
   corrected lineage automatically —
   the §6.2 receipt must show it.
3. **Checkpoint receipt format.** Owner main `4fd6875` landed the
   phase-3 VLA checkpoint format after this draft was written. §6.2's
   "checkpoint id + conversion receipt" is now spelled: the pinned
   base carries a `convert_legacy`-produced (or natively written)
   VLA-format dir passing `validate_checkpoint`, whose metadata
   `stats` block + `stats_note` record the corrected-table
   provenance first-class.
4. **Unchanged.** Reward v2/v1 split, 8×8 groups at T=1.0, tripwire
   belt, lr proposal + fallback, ≤ 12 gate — all stand. §5's pace
   anchor (R1-B ~0.98 GPU-h/step) remains the estimate; it was
   measured on this same discrete stack.
