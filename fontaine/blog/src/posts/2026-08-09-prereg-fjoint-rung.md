# Pre-registration DRAFT: F-then-joint — is the frozen expert's capital a key to joint training? (#4)

*2026-08-09 ~14:3xZ. **DRAFT, not yet immutable.** It becomes the
binding pre-registration when three things happen, in order: (1) the
instrument section lands oracle-gated (`check.py` green), (2) the
owner OKs a box window for it (it competes with the
[adamc_100k](2026-08-09-prereg-molmo2-adamc-100k.md) endpoint's own
stage-2 work, see Scheduling), (3) the execution queue item is
created owner-visible. Edits before finalization are edits;
after, amendments per the standing rule. Ideas
[#4](../ideas/04-stage2-attachment.md); basis: the
[stage-2 attachment decision memo](2026-08-09-molmo2-stage2-attachment-decision.md)
(frozen default adopted; KI-joint closed-unmeasured at a measured
4.11× step cost), which explicitly unblocked this draft and set its
burden: **any joint-flavored escalation must argue against that
measured cost.** The argument is below, before the recipe.*

## Question

The [attachment screen](2026-08-07-prereg-molmo2-attach-screen.md)
asked whether a trunk adapting *from the start* of expert training
(KI-joint, random-init expert) beats a frozen trunk. It never asked
the question [APT](../papers/apt-expert-pretraining.md) (2606.12366)
says is the right one: **once a flow expert has been trained to
convergence against the frozen trunk, does *then* unfreezing the
trunk — flow gradients flowing in, no stop-grad — buy anything a
frozen continuation doesn't?**

APT's diagnosis: the seam damage that KI-style guards defend against
is caused by the expert's *random initialization* — an uninformed
head learns the vision-shortcut and its noisy early gradients wreck
the trunk; with a pretrained expert, their best row unfreezes
everything with no gradient stopping at all (+8..+26 pts over the
frozen row in their grid; "stop-gradient is not a necessary
condition"). [LP-FT](../papers/lpft-two-phase-schedules.md) supplies
the mechanism theorem (feature distortion is front-loaded while the
head is uninformed — align the head first, then unfreeze) and
[ActionX](../papers/actionx-rl-expert-pretraining.md) the same shape
independently (+38 pts LIBERO-Long over joint-from-scratch, Frontiers
caveat carried). Two production stacks
([RDT2](../papers/rdt2-umi-scaling.md),
[Qwen-VLA](../papers/qwen-vla-early-fusion.md)) also train the expert
against a frozen/protected trunk *first* — though RDT2 then never
unfreezes at all, which is the other way this rung can land.

We hold the exact substrate this literature names: the screen's F arm
is a converged flow expert on a hard-frozen action-pretrained trunk —
APT's Stage 1, banked and paid for. This rung spends that capital.

## Why this is worth proposing against a measured 4.11× step cost

The memo's burden, answered head-on:

1. **The 4× objection prices a *lineage*, not a *rung*.** The
   F-then-joint shape is precisely "cheap frozen phase long, joint
   phase short": if it works, the joint phase is a bounded final
   segment (here 5k steps against a 10k+60k frozen history), not a
   4×-forever recipe. LP-FT's compute-Pareto case is this shape.
2. **The rung is the cheapest remaining discriminating contrast.**
   K-from-noise answered "does trunk adaptation help an uninformed
   expert" (no, in 4k matched steps). It did NOT answer the
   initialization question — APT's grid says those are different
   regimes with opposite verdicts. One bounded rung closes the whole
   joint direction *measured* either way; without it, "closed
   unmeasured" stays a caveat on every future stage-2 pre-reg.
3. **Committed cost is capped at ~32 GPU-h** (5k matched arms +
   evals, ceiling 35), under half the screen's 70 ceiling. The
   10k extension spends only on a positive CI (see decision rule).
4. **The payoff side is not small.** If the joint phase buys even a
   0.3-class panel gain, that re-ranks the stage-2 recipe for every
   future trunk in this class — including the adamc_100k endpoint's
   full-length attachment, which per the memo currently binds to
   frozen.

## Arms — the joint bundle is the contrast, initialization is held

Both arms warm-start from the SAME parent: the screen's F endpoint
`fontaine_molmo2_flow_frozen_10k_ddp4/step_010000` (trunk = the 60k
AR endpoint verbatim — F never moved it; expert = the converged
h1024×12 flow expert, panel 9.4157, probe 9.38@10k still trending
down). Matched +5k steps, eff-48, identical data order.

| arm | trunk during +5k | expert init | seam | loss |
|---|---|---|---|---|
| **F2 — frozen continuation** (control) | hard-frozen | F@10k expert | taps, transparent | flow only |
| **J — F-then-joint** | unfrozen (`backbone-text-lr 2e-5`, phase-1 surface) | F@10k expert | taps, **no stop-grad** | CE + 1.0·flow |

- **F2 exists because F was still improving.** J@+5k vs the banked
  F@10k number would credit the joint phase with plain extra
  training. The paired read is J@+5k vs F2@+5k: same parent, same
  +5k data, the joint bundle the only difference.
- **J is the APT best-row analog, not another K.** Flow gradients
  DO enter the trunk through the taps (the thing the screen's guard
  refuses for random-init runs — see Instrument). The CE rider stays
  on, phase-1 verbatim (α=1, same aux fields/dropouts, tables
  continuing from the 60k endpoint's `expert.safetensors`), for the
  Wall-OSS reason (co-train > flow-only in their from-scratch grid)
  and because it keeps the CE-health watch and drift read
  interpretable. J's step is K's step minus stop-grad plus tap
  gradients — so K's **measured** 3.782 s/step anchors the cost
  projection (~4.0 s/step assumed; the rate gate measures).
- Changing {trunk trainable, flow-grads-in, CE rider} together is
  deliberate: the rung tests *the joint phase as a recipe*. If it
  wins, attribution gets its own pre-reg; if it loses, the bundle's
  best published form lost, which closes the direction.

## Shared recipe constants (identical across arms, NOT under test)

- **Start**: `--init-from` the F@10k checkpoint (weights only, fresh
  optimizer, step 0) — J via the materialized composite (Instrument
  §1). Warmup 500 both arms (fresh optimizer state).
- **Fresh shuffle seed, standing owner rule**: `--seed 2` (new vs
  phase-1's 0, F's 0, adamc's 1) — the SAME new seed both arms;
  matched data order is what makes the probes and the panel read
  pair. `--holdout-episodes 0.1 --split-seed 0` unchanged (the split
  is identity, not shuffle).
- **Surface**: residual taps 12 @ stride 3 (layers 2,5,…,35), expert
  h1024/12, adarms, bidirectional, `--decoder-lr 1e-4`, `--max-crops
  1` — the screen's pinned surface, verbatim.
- **Topology**: box 4×H100, eff-48 (12/rank), `--zero1
  --backward-chunks` at the K-smoke values (B12c6),
  `--activation-checkpointing` for J (K's prerequisite carries; F2
  runs without, as F did — memory, not semantics). Eval 256 @ every
  500, save every 1250 (async), both arms.
- **J's unfreeze surface**: phase-1 verbatim — `backbone-text-lr
  2e-5 --grad-clip 100`, frozen embeddings/lm_head (the molmo2
  unfreeze surface), `--prompt-generate-bracket`, same
  `--aux-fields` and dropouts.
- Run names `fontaine_molmo2_flow_fcont_5k_ddp4` /
  `fontaine_molmo2_flow_fjoint_5k_ddp4`; sequential, **F2 first**
  (cheap arm shakes out the warm-start path); `babysit.toml` entries
  at each launch, first-poll util+rate check per standing rule.

## Instrument (to land, oracle-gated, before finalization)

Semantics frozen here; flag spellings are implementation's:

1. **Composite warm-start materializer** (audit result: no train.py
   surgery needed). `--init-from F@10k --joint-ce` correctly aborts
   today — F's checkpoint carries no `joint_ce.safetensors`, and
   `--backbone-init-from` would build the expert fresh, discarding
   exactly the capital this rung spends. A small script (the
   `materialize_joint_ar_view.py` precedent, inverse direction)
   copies F@10k and adds `joint_ce.safetensors` := the 60k
   endpoint's `expert.safetensors` (phase-1 FAST tables, continuing
   not restarting) + the config section. Oracles: expert bytes ≡
   F@10k's; rider bytes ≡ the phase-1 tables; `--init-from
   --joint-ce` load round-trips strictly.
2. **Naive-joint guard escape, narrowly scoped.** The train.py guard
   refusing `--joint-ce` without `--seam-stop-grad` ("a published
   collapse (KI), refused as a run") is CORRECT for random-init and
   stays. A new opt-in flag admits the combination; the guard's
   refusal must still fire without the flag. Oracle: under the flag,
   flow-loss gradients into trunk parameters are nonzero (the
   existing negative-control oracle becomes this run-mode's positive
   contract); without it, the parser error is verbatim-preserved.
3. **Drift-read compatibility**: `materialize_joint_ar_view.py`
   accepts J's checkpoints (same file shape as K's) — verified
   against the real writer on the fixture family before launch.
4. **Memory smoke, J config exact**: one 150-step B12c6 rung
   (`smoke_attach_k_ddp4.sh` pattern), pass = rc 0 AND
   `vram_alloc_peak_gib` ≤ 71. K's 57.34 green does NOT
   automatically bind — J adds tap-gradient backward paths K
   detached. Red ⇒ matched downshift both arms, loudly echoed.

## Gates (in-run, mechanized where precedent exists)

- **vram_alloc_peak ≤ 71 GiB**, both arms (standing box rule).
- **Rate gate, measured not judged** (`attach_rate_gate.py`
  pattern, first jsonl window): projected 5k-phase batch total >
  **35 GPU-h** ⇒ kill, rung closes incomplete, owner steer (no
  downshift branch — 5k IS the short form; halving it guts the
  read).
- **Kill bars, J** (K1-style): NaN/inf ⇒ kill. Probe >
  **12.38** (= F endpoint 9.38 + 3.0) at any eval ⇒ kill at next
  save boundary — the trunk-damage backstop if APT's regime claim is
  wrong here. Probe − F2's matched-step probe > **+1.0** at any eval
  ≥ 2000 ⇒ kill (the joint phase is actively hurting; F2-first
  scheduling banks F2's curve before J needs it).
- **CE-health watch (record, not gate)**: J's `loss_aux` vs the
  screen's banked phase-1 tail anchor (~3.68) at every eval.

## Frozen reads

Panel: `plans/holdout_curated_v0_k4l2_panel_v2.json`, flow keying
heun30/draws1/stable, 4-GPU sharded, sha256-pinned — the screen's
spec verbatim. Paired per-frame from `--dump-predictions` npz, seeded
bootstrap 95% CI (seed 0, 10k resamples). `attach_seam_results.py`
machinery reused with explicit stems (J in the K slot, F2 in the F
slot — the paired-read code is arm-name-agnostic; verified on its
existing oracles before the read).

1. **Primary: Δ_joint = chunk_mae(J@+5k) − chunk_mae(F2@+5k)**,
   paired CI. This one number is the rung.
2. **Decision rule, frozen now**:
   - **Δ_joint ≥ 0 or CI includes zero** ⇒ the joint direction
     closes **measured** for this trunk class — from-noise (K) and
     from-capital (J) both null; the frozen default's standing
     becomes a measured fact, not a priced-out default. No
     extension, no refits, no α fishing.
   - **Δ_joint < 0, CI excludes zero, drift band respected** ⇒ the
     **10k extension fires**: both arms continue to +10k (same
     rates, ~+31 GPU-h, global ceiling 70), panel read repeats.
     Adoption bar at +10k: **Δ_joint ≤ −0.3 with CI excluding
     zero AND drift band held** ⇒ the full-length stage-2 recipe
     for this trunk class gains a bounded joint final phase; the
     adamc_100k endpoint attachment pre-reg cites this rung (its
     frozen phase is unchanged — this appends, the memo's binding
     is amended not overturned). Won at 5k but under the bar at
     10k ⇒ recorded, frozen default keeps, direction closed
     measured-small.
   - **J wins but breaks the drift band** ⇒ wins-with-named-cost:
     AEGIS orthogonal-projection repair un-banks as the named
     escalation; adoption waits for owner steer.
3. **Execution oracle**: both arms ≥ 1.0 chunk-MAE below state-copy
   **11.7639**, else the rung is VOID, not negative (screen read-3
   rule verbatim). Context anchors quoted-never-deciding: F@10k
   9.4157; K's probe curve; gemma lineage 6.5997 (cross-trunk,
   directional).
4. **Trunk-drift (J only)**: greedy AR panel of J's materialized
   AR-view @+5k (and @+10k if extended) vs the 60k endpoint
   **5.8602**, band **|Δ_AR| ≤ 0.3** inclusive — the screen's read
   4 inherited with its comparator.
5. first_mae mirrors of 1 and 3; per-step-in-horizon curves both
   arms (record-only).

## Numbered expectations (banked before data)

1. Both arms beat state-copy decisively and neither regresses above
   F@10k's 9.4157 — confidence high; an F2 regression voids the
   rung (warm-start or seed pathology, not a seam fact).
2. F2@+5k improves on 9.4157 by roughly 0.1–0.25 — the 60k
   continuation's slope analog, F's probe was still falling —
   confidence medium.
3. **Δ_joint is the genuinely open number.** APT/ActionX say
   negative and large; K's matched-probe null plus the Wall-OSS
   reading (phase-1 CE already routed the action gradients) say ~0.
   Banked: CI-excluding-negative at +5k gets confidence
   **medium-low** — this rung exists because the literature and our
   own trunk's evidence point opposite ways.
4. J's drift stays in band — CE co-training anchors the trunk, and
   LP-FT's mechanism says the distortion channel is disarmed once
   the head is informed — confidence medium.
5. **Falsified if Δ_joint ≥ 0 / CI spans zero**: the initialization
   escalation closes for this class, and every future stage-2
   pre-reg cites a *measured* joint null from both starting points.

## Cost & scheduling

Committed (5k phase): smoke ~0.2 + F2 ~5.1 (0.92 s/step measured) +
J ~22.2 (4.0 s/step assumed from K's measured 3.782) + 2 panel evals
~2.5 + drift AR panel ~1.7 ≈ **~32 GPU-h, ceiling 35**. Conditional
extension: ~+31 ⇒ **global ceiling 70** (spent only on a positive
5k CI). Weights: F@10k is on the box + its expert capital in
`fontaine-checkpoints` (backbone dedup'd to the 60k trunk);
execution re-verifies both before the materializer runs.

Venue: box 4×H100, opens no earlier than the adamc_100k endpoint +
its chained panel (~08-12 ~17:00Z+). **Sequencing question for the
owner at finalization**: this rung informs the adamc endpoint's own
stage-2 attachment (information-optimal order: rung first, then the
full-length attach cites its verdict), but the attach is the
deliverable — owner go decides which takes the box window. The scale
caveat (2606.14153) is carried in both directions: the verdict is a
Molmo2-4B-class fact; a different trunk class re-screens.
