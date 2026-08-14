# Pre-reg (FINAL) — token-GRPO phase 2 R1-B: the reward patch + the re-priced ladder

*2026-08-14, drafted 09:5xZ, posted in-channel before launch. Executes
the owner's steering 09:16Z ("let's try your recommendation (2) then
(1)") on the R1-A tripwire boundary: option (2) — fix the reward that
FUNDS shoving — landed as code (commit `5932fb6`), and option (1) —
the lr/β re-price from the banked step_0004 — launches under it. The
owner also asked whether we define "knock away" well; the audit that
answered (in-channel 09:21Z) is §1, because its gaps shaped the
patch.*

**Plain words.** Our robot learned something ugly during training: the
scoring rule paid it for how much closer the toy boat ended to the
goal, no questions asked — so pushing the boat around with its arm
scored just as well as picking it up and carrying it. A safety wire
noticed the pushing getting worse and stopped the run. Instead of just
turning the learning rate down and hoping, we changed the scoring
rule: the robot now only earns points for moving the boat while it is
actually holding it, and gets charged for any boat movement it causes
without holding — in either direction. The simulator now tracks
whether both gripper jaws are touching the boat at every moment, which
is how we can tell holding from shoving. Then we restart training from
where the wire stopped it, gentler than before, and watch whether the
pushing habit fades and real carrying improves.

## §1 The knock-away audit (what the owner's question surfaced)

The old definition (`progress_final_cm ≤ −1.0`): endpoint-only net
displacement, used ONLY in the wave tripwire (> 2× the probe's 0.083
baseline, 3 straight fresh steps → exit 3 — what stopped R1-A at step
5). Three gaps, all now closed in code:

1. **Endpoint-only** — the per-tick `distance_cm` trace was recorded
   but unread: an episode that bats the boat 5 cm away then plows it
   back to −0.9 was invisible. → `max_setback_cm` (worst adverse
   excursion over the whole trace) is now a first-class channel;
   `setback_frac` joins the wave telemetry (record-only this run — no
   baseline exists for a wire yet).
2. **It measured the tail of a strategy the reward FUNDED** — ungrasped
   bulldozing toward the disk paid cm-for-cm; the same physics pointed
   away was "knock-away". → the v2 reward (below) removes the payment
   and adds a charge.
3. **No grasp/contact channel existed** — "moved-without-grasp" was
   undefinable from rows. → `benchy_grip_contacts()`: benchy contact
   with the follower gripper's fixed side and moving jaw, read from
   `mjData.contact`; both sides in one physics state = the two-sided
   **pinch**. Per-tick grip trace (0/1/2/3) recorded alongside the
   distance trace. Registered coarseness: the fixed side is the whole
   gripper body (a wedge against the housing + jaw counts as a pinch),
   and contacts are sampled at control-tick ends.

## §2 The reward patch (landed, oracle-pinned; `composite_reward_v2`)

```
reward = grasped_progress_cm            # progress summed ONLY over pinched ticks
       − 0.5 · ungrasped_displacement_cm  # |Δd| over unpinched ticks, direction-blind
       + 10 · success − 2 · tipped − 5 · strike   # unchanged from v1
```

- Any charge rate > 0 makes shoving strictly unprofitable; 0.5 keeps
  incidental approach nudges from dominating the base policy's group
  signal. The rate is FROZEN for this run; re-pricing it is a
  registered amendment, not a knob.
- Same endpoint, different mechanism now separates: a pure 4 cm shove
  scores −2.0 where a 4 cm carry scores +4.0 (oracle-pinned).
- Rows without a grip trace make v2 raise loudly — no silent
  reversion to the leaky reward.
- **The held-out EVAL metric stays composite_reward v1** — the outcome
  measure and its banked step-0 pairing (1.8441, 2/20) must not move
  when the training incentive does. Only the trained-on advantages
  switch.
- 13 new oracles (scripted two-sided close reads pinch in real
  physics; settled resets contact-free; query purity — no RNG draws,
  no state writes; v2-vs-v1 advantage discrimination; nan-refusal;
  parallel-driver fake extended); `check.py` 904 green.

## §3 R1-B run design (frozen)

`fontaine/scripts/launch_grpo_phase2_r1b.sh`, via `run_detached.sh`:
resume `outputs/sim/grpo_phase2_a/step_0004.pt` (the R1-A
tripwire-stop bank; also on fontaine-checkpoints weights-only) into
fresh `outputs/sim/grpo_phase2_b`; steps 5–14 (10 steps,
`--total-steps 15`). Levers vs R1-A: **lr 1e-6 → 3e-7, kl_beta 0.5 →
1.0, train_reward v1 → v2**. Unchanged: surface A (~10.5M FAST-block
rows), advantage clip ±2.0, kl_stop 0.06, eval-every 1, save-every 1,
8 seeds × 8 draws, temperature 1.0, all §7 tripwires INCLUDING the
knock-away wire (the in-reward fix does not retire the belt).
Anchor = the pristine step-0 policy from `--checkpoint`, captured
before the resume restore (loop contract, oracle-pinned).

## §4 Registered reads

- **Calibration read (step-5 wave, the patch's first live contact)**:
  the wave telemetry's `earned_progress_mean` / `ungrasped_disp_mean`
  decomposition at the resumed policy, plus groups_kept. **Registered
  bar: if ≥ 6 of 8 groups drop in EACH of the first two waves** (the
  v2 reward degenerate at this policy's competence — everything
  scores −0.5·shove with no spread) **→ self-stop via the existing
  collapse wire counts as a calibration FAIL, not a training verdict;
  re-pricing λ is the registered amendment path.** No mid-run
  re-pricing otherwise.
- **PRIMARY (unchanged ladder question)**: held-out paired Δ (v1
  metric, banked baseline) at the boundary — does gentle pressure
  ACCUMULATE over ~10 steps once shoving stops paying? CI95 entirely
  above 0 → accumulation, R2 pricing discussion. Flat with wires
  quiet → the phase-2 answer is "no accumulation on this surface at
  gentle pressure" and the ladder STOPS — banked as a real negative.
- **Behavior reads (the patch's own predictions, record + judge at
  boundary)**: knockaway_frac vs R1-A's 0.41 → 0.36 → 0.31 decay;
  setback_frac; the earned/shoved decomposition trend — the patch
  predicts shoved displacement DECAYS while earned progress holds or
  grows. If knockaway_frac instead re-fires the wire under v2, the
  shoving is not reward-driven at this surface — a finding on its
  own.
- **Tripwires unchanged** (§7 ladder): strikes, non-finite loss,
  spread collapse ×3, knock-away 2× ×3, kl_stop 0.06, competence
  floor CI < −1.0.

## §5 Cost

~0.96 GPU-h/step incl. per-step eval → 10 steps ≈ **9.6 GPU-h**
projected; ladder cum ~5.1 + 9.6 ≈ **14.7 of the 22 GPU-h gate**
(~7 headroom). Babysit registry entry at launch; ~30-min checkpoint
cadence.

## R1-B boundary read (2026-08-14 13:1xZ) — the §4 contingency IS the finding; recommended ladder verdict: STOP

R1-B self-stopped at 12:40:50Z (registered exit 3, unit rc 3): the
knock-away wire re-fired at fresh steps 5/6/7 — `knockaway_frac`
0.3281 / 0.3125 / **0.4531**, three straight above the 0.167 line,
with 0.4531 the highest wave of EITHER run. The step-7 update exited
before its save; **`step_0006.pt` is the banked endpoint** (the R1-A
pattern). Cost ~2.95 GPU-h; ladder cum ~8.1 of 22.

![R1-B boundary — the wire fired under both rewards, the v2 decomposition, and the flat held-out probe](https://mcobzarenco-fontaine-reports.static.hf.space/chart__grpo_r1b_boundary.png)

**Calibration read (§4 bar): PASS — the λ re-price amendment path is
NOT triggered.** The degenerate-reward bar was ≥ 6 of 8 groups
dropped in each of the first two waves; observed 8/8 kept in every
wave (median group std 3.27 / 3.02 / 2.14 cm). The v2 reward had
spread at this policy's competence; the run's stop is a training
verdict, not a calibration failure.

**PRIMARY (paired Δ, v1 metric, banked step-0 pairing): flat — no
accumulation measured.** At the banked endpoint the 20-episode paired
Δ vs the pristine step-0 policy (1.868) is **+0.0246, CI95 [−0.0716,
+0.1455]** (2/20 successes, unchanged). The greedy probe was
digit-identical at steps 5 and 6 (1.8926) — the same determinism
R1-A showed across its flat 1.8441 steps 1–4: at lr 3e-7 the held-out
greedy policy is measurably unchanged wave-to-wave. Neither §4
PRIMARY branch fires cleanly (the CI is not above 0; "flat with wires
quiet" requires quiet wires) — the run exits through the registered
behavior contingency instead.

**Behavior reads: the patch's prediction is FALSIFIED on the deciding
channel.** The prediction was knockaway decays (R1-A tail 0.41 → 0.36
→ 0.31) and shoved displacement decays while earned progress holds or
grows. What happened: `ungrasped_disp_mean` — the exact quantity v2
charges — DID decay monotonically (4.98 → 4.60 → 4.20 cm, −16%), but
`knockaway_frac` rose to its run max and `earned_progress_mean`
collapsed at the tripwire wave (1.19 → 1.66 → **0.58 cm**;
`reward_mean` −0.74 → −0.26 → **−1.21**). `setback_frac` banked its
first baseline: 0.703 / 0.5625 / 0.5938 (record-only, as registered).
Read together: total ungrasped contact shrank slightly while its
endpoint-adverse share GREW — the displacement redistributed rather
than retired. **§4's registered contingency is the finding: the wire
re-fired under a reward that pays nothing for shoving and charges
every ungrasped centimeter — shoving at this surface is not
reward-driven.** A policy at this pinch competence (successes 4/3/3
of 64) does not control its contact outcomes finely enough for the
incentive to reach the behavior; the shoving is a competence
artifact, not an incentive artifact.

**Recommended ladder verdict (owner adjudicates, frozen rule): STOP
phase 2 on surface A, banked as a real negative.** One run consumed
both boundary options at once — the re-price (lr ÷3.3, β ×2) and the
reward fix — and the deciding behavior got worse while the held-out
probe stayed flat across 6 banked steps of the two runs. The
remaining ~14 GPU-h of ladder headroom buys more waves of the same
physics, not a different answer; the R2 pricing discussion is moot
without accumulation. If the thread continues, the registered next
shape is a NEW pre-reg that raises pinch competence FIRST (e.g.
grasp-rich SFT before RL pressure) rather than re-pricing pressure on
a policy that can't yet grasp — and the owner has already ruled any
new run starts post-phase-4 of the molmoact2 retirement.

**Banked**: `grpo_phase2_r1b/step_0006_weights.pt` on
fontaine-checkpoints (weights-only, 2.9 GiB, with the final
train.jsonl incl. the tripwire row + meta.json) — unlike R0's
collapsed weights this endpoint is bankable: two healthy v2 updates,
anchor-KL 0.017 at the last saved step, the seed for any
owner-decided continuation. Chart
`chart__grpo_r1b_boundary.png` on fontaine-reports; babysit registry
pruned to 0 live at the stop.

---

*Post-retirement note (added 2026-08-14 ~21:4xZ, after the molmoact2
retirement completed on main `26ac1e6` — this annotates the closed
record; nothing above is changed). Three rules now bind any
continuation of this line:*

1. ***Decision 11 (fresh runs)**: any new GRPO run is a FRESH pre-reg
   on the first-class stack (`bijou/grpo_replay.py` /
   `MolmoAct2DiscreteStack` over bijou checkpoints — the port's
   HF-layout dirs + norm tags retired). The banked `.pt` endpoints
   above are **salvage-only** (weights format-compatible — the named
   trainables live on the same Molmo2Model structure), never resumed
   across the re-point.*
2. ***Masked-only decode**: the unconstrained (zeros-fallback)
   reference mode retired with the port. Any old-side comparison
   reruns only at tag `pre-molmoact2-retirement`.*
3. ***Full-width Gumbel**: the new stack draws full-width Gumbel
   vectors per step where the port drew 2048 — greedy is
   bit-identical and the masked softmax identical, but **sampled
   streams differ under the same seed**. Replay of these banked waves
   is unaffected (rows carry their bins and π_old); cross-stack
   draw-stream comparisons must not expect bit-equality.*

*Wave integrity re-verified on the new stack 2026-08-14 ~21:4xZ
(probe_grpo_replay_parity.py, local): masks bit-equal on all
1,903 + 1,904 rows of R1-A/R1-B; banked-vs-replay worst-token spreads
recorded (v1 median 5.68e-1 / p90 1.29 / max 3.92; v2 median
5.52e-1 / p90 1.58 / max 8.84 — the JPEG + policy-history-inclusive
report-only read; the loop's clipped surrogate is the consumer).*
