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

## §9 Amendment A3 (ACTIVE, 2026-08-19 13:2xZ) — activation from 7%: the band amendment

*Status: **ACTIVE — owner-activated 13:25:15Z 08-19** ("Re: my calls,
I agree with all your recommendations", message …407784, replying to
the 11:19Z summary …522815 rec **AMEND + ACTIVATE**). §6-style
finalization satisfied at activation: HEAD re-pin **570e53e** (the
commit carrying the A3.8 launch kit — flags, verdict instrument,
staged launcher — and this page's A3 text, frozen 12:5xZ before the
reply); objection window satisfied by the frozen page + the
in-channel summary + the owner reply. Execution per A3.8: preflight
leg 0 fires at activation (H100 idle, policy-server down, verified),
the A3.4 run fires on a PASS verdict. Drafted as an owner call under
the pre-delegation register; superseding activations fall under the
2026-08-18 no-GO-asks delegation.*

**Plain words.** The robot's token-based control passed only 7 of 100
test scenarios — below the 20 we said reinforcement learning needs.
But the autopsy of those failures showed something specific: the
policy isn't confused, it's *timid*. It reaches the boat, closes the
gripper, and then moves it at less than half the speed its
flow-based sibling manages, so time runs out. That timidity is a
known artifact of always picking the single most likely action
("greedy" decoding). Reinforcement learning as we run it doesn't use
greedy decoding — it samples, which is precisely the knob that
relieves timidity — and its reward directly pays for finishing. So
we propose to amend the entry bar and run RL from the 7% policy,
with two cheap pre-checks that abort the run early if this reasoning
is wrong.

### A3.1 What the band said, what was measured

A2.1's activation bar: unseen sim100 **≥ 20/100 greedy** on the head
GRPO trains. Measured (route-C joint `step_002000`, token head,
probe of 08-18/19): **7/100** — band 5–19 → owner decision. Frozen
anchors for every read below: **token greedy 7/100** (success seeds
34, 35, 63, 68, 71, 91, 96), **flow sibling 44/100** (same trunk,
same scenes), **token_base 0/100** (released model, SFT delta +7).

### A3.2 The case for amending (the [decode
diagnosis](https://mcobzarenco-fontaine-reports.static.hf.space/eval__grasp_sft_joint_step2000__token_unseen100.html),
frozen in `analysis__token_decode_diagnosis.json`)

1. **Not decode collapse**: 0/300 frozen episodes across all three
   legs (motion instrument; min mean motion 1.41), no stereotypy.
   The head produces coherent, scene-directed trajectories.
2. **Greedy magnitude attenuation**: funnel touch→pinch→success
   **60→22→7** vs flow **91→59→44** on the same trunk (base 7→0→0);
   conversion losses at both hops (touch→pinch 0.37 vs 0.65,
   pinch→success 0.32 vs 0.75); carry speed **0.81 vs 2.00 cm/s**
   (ratio 0.405), 2 timeouts still holding the boat (flow 0);
   reach envelope truncated — 1/14 touch at 11–13 cm spawn, **zero
   successes past 10 cm** (flow: 17 there). The failures are
   under-magnitude versions of the right behavior.
3. **Iterate-once re-buys the wrong thing**: CE owned all 2000
   route-C trunk updates (joint objective, λ=1.0) — the token head
   is not undertrained by omission. Another token-SFT pass extends
   reach; it is not a targeted fix for greedy-mean attenuation.
4. **R2's pressure lands exactly on the failure**: group rollouts
   sample at **T=1.0** (greedy never appears in training), and the
   +10 success term pays directly for completing the carry the
   greedy read leaves unfinished. Context, recorded as a rhyme and
   not evidence (different trunk, different metric): the #19 dT
   record measured AR competence monotone in T
   (6.50/6.57/6.78/7.18 at T=0.5/0.7/1.0/1.3, er60k).
5. **The §1 arithmetic at p = 0.07**: mixed-group probability
   1−(1−p)⁸−p⁸ ≈ **44%** of groups carry a ±10 contrast — half the
   ~83% at the gate minimum, but ~4 in 10 groups informative vs
   ~34% at R1-B's measured 0.05 base, and the sampled rate should
   sit *above* the greedy 7% if point 2 is right. The signal is
   plausibly there; the two gates below make that a measurement
   instead of a hope.

### A3.3 Registered gates that replace the ≥ 20 bar

- **Preflight leg 0 (F-premise, ~1.3 GPU-h)**: single-draw sampled
  T=1.0 sim100 (seeds 0–99) on the pinned base, before any
  training. Read: sampled count vs greedy 7. Materially BELOW →
  the relief premise is wrong → **abort the run**, route to
  iterate-once, bank the read. At-or-above → premise holds
  first-contact; the sampled count becomes the recorded
  training-decode competence floor.
- **Wave-0 calibration bar (from §3, re-priced)**: fraction of
  groups mixed, predicted ≈ 44% at the greedy floor. **Mixed < 20%
  → abort** before GPU-hours burn; the group shape is the
  amendment path, not lr.

### A3.4 Frozen run spec (deltas vs §2 + A2 only)

- **Base (pinned)**:
  `checkpoints/finetune/fontaine_grasp_sft_joint_corrected/step_002000_v2`
  — schema-2 VLA dir, family joint, corrected-table stats
  first-class in `metadata.json` (A2.2/A2.3 satisfied);
  HF-recoverable bitwise per the [08-19 evacuation
  audit](2026-08-19-hf-evacuation-audit-v2-fleet.md). Load seam
  verified in code this session: `MolmoAct2DiscreteStack.load` →
  `read_metadata` + `load_vla`; the joint family carries the
  format-6 discrete decoder — no conversion step needed.
- **Recipe unchanged** (§2 + A2.4): composite_reward_v2 trained-on /
  v1 held-out, 8 seeds × 8 draws T=1.0, advantage clip ±2.0,
  clip-higher [0.8, 1.28], lr **1e-6** primary / 3e-7 registered
  fallback, kl_beta 1.0 vs the frozen base anchor, kl_stop 0.06,
  option-B trainable surface (text stack fp32, vision frozen bf16).
- **Seeds (re-pinned)**: `--train-seed-base 2000` — the loop default
  1000 now collides with the stage-B collection band 1000–1099
  (demo corpus + measured train-probe band). Eval holdout 0–99
  never trained on; in-loop 20-seed eval band 200–219 stays the
  loop default, record-only trend.
- **Knockaway wire (re-pinned)**: the config default baseline
  10/120 ≈ 0.083 is an R0-A-era er60k measurement; this base's
  greedy unseen read measured 25/100 knock-aways, so the old 2×
  line would fire spuriously at wave 0. Re-pin: baseline = the
  wave-0 measured knockaway_frac, wire at 2× ×3 unchanged.
  Instrument delta: expose `--knockaway-baseline` (config field
  exists; flag does not). setback_frac stays record-only.
- **NEW registered boundary read — flow-head regression leg
  (~1.3 GPU-h)**: the trunk is shared and option B trains the text
  stack, so GRPO moves the flow head too. Flow unseen100 (euler-10)
  at the boundary vs the 44 anchor, record + judge: material
  regression is F-regression evidence even if the token side
  improves — the joint checkpoint's other head is not free.
- **PRIMARY (per §3, comparator re-based)**: boundary greedy sim100
  vs **7/100**, paired per-seed exact test (greedy stays the
  serving convention; RL pressure is sampled, the claim lands on
  the serving read). Record-only sibling: boundary sampled T=1.0
  sim100 vs preflight leg 0 (prices decode-gap movement).

### A3.5 Budget (re-priced) and kill rules

Pace anchor ~0.98 GPU-h/step (R1-B, same stack class, per-step
20-seed eval included). **10 steps ≈ ~10 GPU-h + preflight leg 0
~1.3 + boundary token sim100 ~1.3 + flow regression leg ~1.3 ≈ ~14
expected, gate ≤ 15** (supersedes §5's ≤ 12, which priced no
boundary legs). Mid-ladder read at wave 5, ~30-min babysit cadence,
detached unit + registry entry at launch. Kill rules: preflight
F-premise abort, wave-0 mixed < 20% abort, the full §2 tripwire
belt (knockaway wire re-pinned per A3.4), kl_stop 0.06, exit-3
self-stops armed.

### A3.6 Falsifiers (unchanged + one new)

F-flat, F-instability, F-regression stand as §4 wrote them —
F-regression now explicitly covers the flow-head leg. NEW
**F-premise**: preflight sampled materially below greedy 7 — the
attenuation-relief story is wrong, run aborts unstarted, the read
banks against the diagnosis.

### A3.7 Activation interface

Owner reply `2 ACTIVATE` → same-session: §6 finalization checklist
(HEAD re-pin, receipts, objection window noted as satisfied by this
frozen page + the in-channel summary), launcher staged, launch at
the next free boundary (H100 idle at draft time). `2 HOLD-BAND` →
token-SFT iterate-once arm per the registered band, this page holds.
`2 PARK` → the lane parks with the read banked.

### A3.8 Launch kit (LANDED 13:xxZ 08-19 — code staged, still not active)

Everything above is now one command per stage
(`fontaine/scripts/launch_grpo_r2.sh`, oracle-tested, full-parse
green; the A3.4 argv is spelled exactly once):

- **`parse-check`** (CPU): the frozen argv through
  `sim.grpo_loop.parse_args` with every A3.4 re-pin asserted, plus
  the babysit registry entry template (round-trip-verified against
  the babysit schema, A3.5 gates as `gpu_hours_max 15` /
  `vram_max_gib 75`).
- **`preflight`** (leg 0, ~1.3 GPU-h, detached unit): the sampled
  T=1.0 sim100 through the SAME probe driver and substrate pin as
  the greedy 7 leg (`--serve-head ar --ar-temperature 1.0`,
  `--clutter-appearance standins`), chaining into a machine-readable
  F-premise verdict (`grpo_r2_preflight_verdict.py`, provenance
  guards loud). **"Materially below" is now pinned**: exact
  one-sided binomial tail under Bin(100, 7/100) at the house 5%
  level — **ABORT at ≤ 2/100** (P(X≤2) ≈ 0.026), **PASS at ≥ 7**
  (the count becomes the recorded training-decode floor), **BAND at
  3–6** (below the anchor, not materially — a decision post owns it;
  the launcher refuses BAND without an explicit recorded override,
  and refuses ABORT unconditionally).
- **`launch`**: refuses without a PASS verdict, then fires the A3.4
  argv via `run_detached.sh`. The two in-loop gates are now
  first-class loop flags (oracle-tested, defaults unchanged — zero
  behavior change for every other run): `--wave0-mixed-abort 0.20`
  stops the run at wave 0 below the A3.3 mixed-fraction bar
  (`mixed_groups_frac` is a new heartbeat key — success/failure
  contrast per group, distinct from the z-filter's `groups_kept`),
  and `--knockaway-baseline wave0` re-bases the violence wire at
  wave 0's measured rate (the capture wave itself exempt), retiring
  the R0-era 10/120 pin for this run per A3.4.

### A4 Wave-0 substrate postmortem + relaunch (2026-08-19 17:4xZ, decided + announced per the standing no-GO-ask rule)

**What happened.** The 16:10:02Z launch aborted itself at wave 0
(17:19:34Z, ~1.2 GPU-h): `mixed_groups_frac 0.0000 < 0.20`
(the A3.3 in-loop gate). The telemetry was not "unlucky scenes" —
64/64 training episodes (seeds 2000–2007 × 8 draws, T=1.0) showed
**zero successes, zero knock-aways, zero setbacks, ungrasped
displacement ~0.0015**, and the journal's per-replan `benchy->disk`
distances were bit-frozen from replan 0 to 29 on every seed. The
step-0 in-loop eval read 0/20 on the same run. The boat was never
touched, anywhere.

**Diagnosis (convicted, two independent legs).** (1) *Code:* the
08-18 clutter-patch promotion made `patched` the production default
on the parallel driver's `WorkerConfig`; `sim.grpo_loop` builds
`WorkerConfig` without setting it, so the training waves AND the
in-loop eval rendered **patched** clutter — while the A3.1 anchors
(greedy 7/100, flow 44/100), the A3.2 decode diagnosis, and the
preflight PASS (8/100) all ran the registered **standins** substrate,
and the policy's demo corpus is stand-ins-era. The promotion sweep
pinned the eval drivers; the loop's embedded rollout path was missed.
(2) *Measurement:* the probe driver on the SAME seeds 2000–2007,
same checkpoint, same T=1.0, **standins** — 6/8 episodes interact
(progress up to 5.9 cm, min distance 3.9 cm, knock-aways present;
0/8 successes is unremarkable at p≈0.08, P=0.51). Substrate
convicted; the 2000+ seed band exonerated
(`outputs/sim/grpo_r2/wave0_diag/seeds2000_standins_t1.json`; the
aborted wave banked at `outputs/sim/grpo_r2/loop_wave0abort_patched/`).

**Reading.** The wave-0 gate read an instrument artifact (substrate
mismatch), not the registered calibration question (does success
variance appear in groups at the measured base rate?). The A3.3
calibration claim is therefore still UNREAD — the gate stays armed
at 0.20 on the relaunch. The gate itself performed exactly as
registered: it stopped a doomed 10-step run after ~1.2 GPU-h.

**Fix (mechanical, oracle-tested).** `--clutter-appearance
{patched,standins}` exposed on `sim.grpo_loop` (default `patched` —
zero behavior change for any other run), threaded through BOTH the
training-wave and in-loop-eval `run_units` seams, recorded in
`meta.json` + the per-wave rows meta; the launcher pins
`--clutter-appearance standins` in the frozen argv and parse-check
asserts it. Oracles: parse-time typo refusal + a seam test that both
wave closures forward the substrate.

**Relaunch (same A3.4 run, registered conditions restored).** The
preflight PASS stands — it was measured ON standins, which is what
the loop now renders; no re-run needed. Same frozen argv otherwise;
all gates re-arm fresh (wave-0 mixed 0.20, knockaway wave-0
self-capture, kl_stop 0.06, tripwire belt).

**Budget re-price (supersedes A3.5's ≤ 15).** Spent so far:
preflight 2.25 (ran ~1.7× the greedy leg's pace) + aborted wave ~1.2
+ diagnosis probe ~0.24 ≈ **3.7 GPU-h**. Expected remaining: 10
steps ~10 + boundary legs re-priced at measured paces (greedy ~1.3,
sampled ~2.25, flow ~1.3) ≈ ~14.9. **Total expected ~18.5, gate
≤ 20.** Kill rules unchanged.

### A5 Serving-parity postmortem + launch gate (2026-08-19 19:4xZ — the fix behind the 18:06:48Z kill)

**What the A4 relaunch showed (kill post 18:08Z).** The 17:46:56Z
relaunch — substrate genuinely `standins`, verified in meta and at
the worker seam — read 0/20 at the step-0 in-loop eval with **all 20
scenes bit-frozen** replan 0→29, while the greedy anchor leg
displaces the boat in 59/100 episodes (P(0 of 20 move) ≈ 2×10⁻⁸).
The loop's serving path is inert on the v2 base independent of
substrate. Killed at 18:06:48Z, ~0.33 GPU-h in (lane total ~4.0).

**Root cause (convicted in code, quantified).** The retirement
phase-4 re-point swapped the loop's port predictor for
`MolmoAct2DiscreteStack` but carried the port era's **hardcoded
v30→v21 joint-frame shim** (shoulder_lift mirrored, lift/elbow +90°)
over unconditionally. That shim is correct exactly for a checkpoint
whose recorded q01/q99 table is in the pre-PR#777 **v2.1** frame —
the unremapped official releases the port served (R1-B, which
interacted, ran `allenai/MolmoAct2-SO100_101`, a v2.1-table HF
layout). Every bijou-format table in play today is **v3.0-frame**
(bijou-trained like the v2 base, corrected-table recomputes, and the
conversion-remapped release alike — docs/so101-joint-conventions.md
§3–6). On the v2 base the numbers are decisive: the shim maps sim
lift ∈ [−103, +29]° to [61, 193]° against a table row of
[−110, +12]° — the normalized value clamps at **+1.0 on every
frame** (elbow likewise), so the policy is state-blind on the two
joints that matter; and the decoded chunk comes back through the
shim's inverse (lift 90−a, elbow a−90), commanding poses outside the
trained range every replan. Arm pinned off-workspace, boat never
touched, distances bit-frozen — the exact telemetry of both kills.
The doc's §6 warned nothing automatic catches a missing/double remap;
this was a missing-identity (extra) remap at a seam with no check.

**Exonerated:** the `_batch` seam flagged in the kill post (ACTION
quantiles passed as `state_stats`) is **inert** — `predict_ar` on the
ar/joint families detokenizes under the family's own
`action_quantiles` table and never reads the batch stats; prompt
state normalizes in `prompt_inputs` before `_batch` is built. Left
as-is.

**Fix (landed this session, oracle-tested).** `--joint-frame
{auto, rig, v30-to-v21}` on BOTH sim discrete drivers
(`sim.grpo_loop`, `sim.rollout_sim_parallel --molmoact2-discrete`),
resolved through one helper (`resolve_joint_frame`) that reuses
`JointFrameTransform`'s literals (the rollout CLI's vocabulary —
`rig` = identity, exactly what BijouPolicy's path applies): `auto`
fingerprints the checkpoint's state table per the conventions doc §4
(descending lift pair or far-negative lift → v3.0; lift ≥ +30 with
elbow corroboration → v2.1) and **refuses** an unclassifiable table;
an explicit mode that contradicts a classified table is refused too
— the mismatch class is now unrepresentable at this seam, for old
lineages and new. Resolved frame recorded in `meta.json`, the rows
meta (`state_units`), and the driver out-json.

**Parity oracle (CPU, in the suite).**
`tests/test_joint_frame_parity.py`: the classifier pinned on the
three real table shapes; refusal semantics pinned; the shim literals
pinned against `JointFrameTransform` both directions; and **prompt
parity** — the loop stack's `prompt_inputs` vs the BijouPolicy
collator on the same observation collate to bit-equal
`MolmoAct2Inputs` tensors on the tiny fixture. Beyond that seam the
two paths share `predict_ar` and the family's one quantile table, so
prompt parity + the identity frame map IS serving parity.

**GPU parity read (REQUIRED LAUNCH GATE, wired).**
`launch_grpo_r2.sh parity` (~0.7 GPU-h, next free GPU window —
onerig owns the H100 to ~07:0xZ 08-20): seeds 200–219 greedy on
standins through BOTH paths — the loop stack under `--joint-frame
rig` vs BijouPolicy `--serve-head ar`.
`grpo_r2_parity_verdict.py` rule, registered here: interacted :=
min_cm < initial_cm − 1e-6 or final_cm ≠ initial_cm; **PASS** iff
|Δsuccesses| ≤ 2 AND |Δinteracted_frac| ≤ 0.30 (the convicted mode
reads 0.00 vs ~0.59 — decades outside; the band absorbs decode-stack
noise, which moves seeds, not fractions). `launch` now refuses
without a PASS verdict — no override exists. The frozen argv gains
`--joint-frame rig` (parse-check asserts it); all other pins and the
A4 budget stand: spent ~4.0, parity +0.7, relaunch ~14.9 → expected
~19.6 against the **≤ 20 gate** — no re-price, but zero slack: any
further abort ends the lane at the gate.

## §10 LANE CLOSE-OUT (2026-08-20 14:1xZ) — died at the gate; what the ~6.6 GPU-h bought

**Outcome: NO primary read.** The A3.4 relaunch (11:59:20Z on the
perfect-parity PASS) died 13:59:42Z in the step-2 backward:
`torch.OutOfMemoryError` wanting +1.47 GiB at 78.26 GiB in use.
Step-1's recorded peak was 72.09 vs the ≤75 gate — the step-2 update
crossed the gate in-flight, caught by the allocator instead of a
poll. `save_every=5`, death at step 2 → no checkpoint, resume
impossible. Retry arithmetic: ~4.7 spent + ~1.9 burned + ~14.9
fresh ≈ 21.5 vs the A4 **≤20** gate; the registered zero-slack rule
("any further abort ends the lane at the gate") applied verbatim.
**Lane CLOSED, no step-10 sim100 read, no override.**

**Banked positives** (what any R3 pre-reg inherits for free):

- **Serving parity PASS, perfect** (~11:58Z): both paths 2/20, same
  seeds 207/214, interacted 20/20 both, Δ 0 / 0.0 vs tolerances
  2 / 0.30. The convicted 0.00-interacted serving mode is dead; the
  `--joint-frame rig` classifier + parity oracle are permanent
  infrastructure.
- **Wave-0 calibration read**: mixed_groups_frac 0.50 vs the <0.20
  abort bar (predicted ~0.44) — the knockaway re-base works; the
  7%-base premise (§9 A3) survives first contact.
- **Step-1 row healthy**: kl 0.0041, strikes 0, groups 8/8 kept,
  6/64 wave-0 successes, knockaway_frac 0.3281 banked as the
  violence-wire baseline.

**Death mechanism + the R3 lead** (registered, not priced):
`grpo_loop.py` sets no
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and
episode-length variance swings backward memory ~6+ GiB step-to-step
at a 72 GiB baseline. An R3, if ever re-priced, needs the alloc
conf + measured headroom in its pre-reg — or a smaller
group/batch geometry. No R3 is scheduled; the freed window went to
the demos+clean poison-pinning cell (pre-reg 2026-08-20, delegated).

Lane ledger: preflight 2.25 + patched wave ~1.2 + probe 0.24 +
relaunch-1 0.33 + parity 0.66 + A3.4 run ~1.9 ≈ **~6.6 GPU-h spent**
of the ≤20 A4 lane gate; death post 1539998329893556224.
