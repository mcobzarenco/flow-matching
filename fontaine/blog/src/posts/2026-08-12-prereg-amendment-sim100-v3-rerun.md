# DRAFT amendment: sim100 rerun under v3 visuals — arms, re-baseline, priors, and the parallel-path contingency

*Drafted 2026-08-12 ~10:2xZ (work session; real `date -u` at write:
10:20). **STATUS: DRAFT — not registered.** This is the pre-reg
amendment the rerun item
(`sim100-v1-rerun`, owner_hold) requires before launch: it inherits
everything from
[the sim100 pre-reg](2026-08-11-prereg-sim-policy-eval-100seeds.md)
(seeds, horizon, metric, floor/validity gates, deliverables) and
changes only what is listed here. It becomes registered when the
owner unholds the rerun and the finalization steps at the bottom run
(param sheet in-channel, objection window). Evidence base:
[sim100 v0 close](2026-08-12-sim100-results.md) ·
[spot20 v3 results](2026-08-12-sim-spot20-v3-results.md) ·
[content-diversity close](2026-08-12-sim-content-diversity-results.md)
· [wrist-periphery close](2026-08-12-sim-wrist-periphery-results.md).*

## Plain words

Two days ago we ran five robot policies through 100 simulated
episodes and none ever picked up the toy boat — but the *pattern* of
failure said the simulator's fake-looking pictures, not the physics,
were the blocker. Since then the pictures were overhauled: real
photographed table backgrounds that change every episode, desk
clutter scattered the way the real operator actually scatters it,
and a wrist camera moved to where the real one sits. A 20-episode
spot-check already showed one policy — the old teacher — reacting to
the new pictures by pushing the boat the *right* way, about a
centimeter better per episode. This document is the plan, written
before spending the GPU time, for re-running the full 100 episodes
under the new pictures: which policies run, what we predict each
will do (registered now so we can't move the goalposts later), and
how each episode is compared one-to-one against its old-graphics
twin. It sits in draft until the owner green-lights the rerun.

## What carries over unchanged (from the sim100 pre-reg)

- **Seeds 0–99**, identical per arm (paired design); spawn ranges,
  30 replans × 30 executed ticks = 30 s horizon, chunk 50, per-replan
  stable noise; policy seed 0, bf16 expert, batch 1.
- **Primary metric**: `progress_final` = initial − final boat→disk
  cm (XY), settled post-reset state as initial; bootstrap CI95
  (10k resamples, seed 0); `progress_min`, success rate + median
  tick secondary.
- **Validity gates**: reset strikes = 0 on every (arm, seed);
  `hold` metric floor |mean progress_final| < 0.5 cm.
- **v0 physics** (widened limits, solver caps 50/50, CoACD benchy,
  SERVO_SYSID set) — untouched by the entire visual series;
  settled-qpos and spawn-stream bit-identity across render styles is
  oracle-pinned (`tests/test_sim_appearance.py`).
- **Success caveat, re-verified at draft time**: `success()` still
  lacks the gripper-open check its docstring claims and its
  stillness clause reads all joint velocities — success rates carry
  that asterisk; distance metrics unaffected.
- Deliverables: per-arm JSON + reads JSON on `fontaine-reports`,
  dark-mode HTML report + video gallery, results post, numbers
  in-channel.

## Change 1 — frames: `render_style="v3"` (the shipped default)

Arms run under the owner-approved v3 default (flip 07:29Z 08-12,
commit `da96d30`): per-reset draws from the 26-plate real-background
bank + measured clutter presence/pose draws (top), 72° fisheye +
re-tuned over-the-jaw pose (wrist). Execution uses the GPU
compositor path (`_TorchPost`, commit `b99be38`, owner-approved
08:12Z; ≤2/255 oracle vs the numpy reference).

**Visual re-baseline (the registered probe instrument, 100-seed
reset renders, er_60k encoder 5-NN AUROC vs real; 0.5 =
indistinguishable):**

| camera | v0 render (sim100 conditions) | v3 (reference path) | v3 (GPU path, execution) | registered line |
|---|---|---|---|---|
| top | 0.890 | 0.673 | 0.669 | ≤ 0.790 — met |
| wrist | 0.835 | 0.548 | 0.544 | ≤ 0.786 — met |

Both cameras are at or inside their registered lines; the wrist sim
frames sit *inside* the real embedding spread (k-ratio 0.97×).
Caveat carried from every probe close: this measures encoder
separability of *reset* frames; mid-episode content is unmeasured —
which is exactly what this rerun measures behaviorally.

## Change 2 — arm set (new names; output dirs must not clobber banked v0 rows)

| arm | checkpoint | decode | v0 baseline (100 seeds) | role |
|---|---|---|---|---|
| `er60k_v3` | `fontaine_molmo2_er_60k_ddp4/step_060000` | heun-10 | −0.03 cm mean, 4/100 engaged | primary reference trunk (owner goal arm) |
| `ftrig4k_v3` | `fontaine_flow_snapdistill_ftrig_4k_1xh100/step_004000` | euler-1 | +0.08 cm, 47/100 engaged, 27:20 toward:away | the only toward-tilted v0 arm; NOT spot-checked under v3 |
| `teacher80k_v3` | `bijou_flow_artrunk_h1024_40k_ddp2/step_080000` | heun-30 | −0.73 cm, 56/100 engaged, 18:38 toward:away | the confirmed visual responder (spot20) |
| `hold_v3` | none | — | −0.00 cm | metric floor under v3 rendering |

**Deltas from the queue-item text, flagged as owner decision
points:**

1. **`teacher80k` added** (the item predates spot20): it is the only
   arm with a confirmed, CI-excludes-zero behavioral response to the
   visuals (+0.97 cm [+0.16, +1.81] at n=20, direction flipped
   toward the disk). Firming that signal at n=100 is the clearest
   scientific payoff of the rerun; dropping it would leave the
   headline read at 20-seed power.
2. **`snap30k` dropped**: null at both legs (v0 sim100 −0.12 spanning
   zero; spot20 Δ +0.06 [−0.61, +0.85]). Re-running it burns an
   arm-slot to reconfirm a double null. Owner may reinstate it as a
   fifth arm at +~25% cost.
3. **The er15k/35k/55k ordering rungs stay dead** (owner amendment
   22:58Z 08-11 killed them; the fidelity-ordering validation read
   died with them and is NOT resurrected here — this rerun makes no
   sim-as-policy-meter claim).

## Change 3 — primary read: paired v3 − v0 per seed, per arm

The spot20 instrument at full power, now the registered primary:

1. **Per arm: paired per-seed Δ `progress_final` (v3 − banked v0
   row, same seed, 100 pairs)** — mean, bootstrap CI95 (10k, seed
   0), sign counts. This is the highest-power read available because
   the banked v0 rows pair bit-identically: spawn streams are
   oracle-pinned across render styles (verified 60/60 in spot20).
2. Within-v3: each policy arm − `hold_v3` paired CI95;
   `teacher80k_v3 − er60k_v3` and `ftrig4k_v3 − er60k_v3`
   record-only.
3. **Engagement split** per arm: episodes with best-point > 1 cm,
   v0 → v3, and toward:away among engaged — the axis the v0 close
   identified as the finding.
4. Success rate (owner goal: ≥ 1 success anywhere on the 100 seeds;
   baseline 0/500 across all v0 arms), with the `success()` caveat.
5. Integrity tripwires (any failure voids the affected arm):
   `spawn_xy` bit-match vs banked v0 on every (arm, seed); strikes
   0; hold floor.

## Registered priors (stated before any GPU minute)

- **`teacher80k_v3` — the confirmatory prediction**: paired Δ
  positive with CI95 excluding zero; arm mean moves from −0.73
  toward ≥ 0 (spot20 point estimate on its 20 seeds: −0.90 → +0.07).
  This is the headline registered read. Failure to confirm at n=100
  = the spot20 signal was a 20-seed fluctuation; report either way.
- **`er60k_v3` — prior null**: spot20 read Δ −0.07 [−0.33, +0.12]
  with 13/20 episodes byte-identical (it rarely touches the boat).
  What a real change would look like: engagement rising clearly
  above v0's 4/100, or a CI-excludes-zero paired Δ — either would
  mean 100-seed power caught what n=20 could not, and would be the
  bigger news since er60k's miss was the original spatial-mismatch
  fingerprint (fisheye + pose geometry moved where things appear —
  exactly the axis the v1 close named).
- **`ftrig4k_v3` — uncertain, most interesting open cell**: never
  spot-checked under v3. Its training pixels (owner rig) already
  resemble the sim scene, so its visual gap was smallest — prior is
  a Δ smaller than teacher80k's, sign positive; the registered look
  for a response is toward:away sharpening beyond 27:20 and/or a
  positive CI-excludes-zero Δ.
- **`hold_v3`**: floor holds (|mean| < 0.5 cm) — anything else is a
  reset/settle artifact and blocks all reads.
- **Success**: prior remains near-zero probability per episode —
  grasp-phase physics is still fidelity-limited (phantom collision
  margin p99 3.78 mm; gripper-priority friction override), so even
  a visually-matched policy may top out at push-the-boat. A single
  latched success would clear the owner's stated bar and headline
  the results post regardless.
- **Disk position stays pinned at (0.22, 0.11)** for this rerun:
  the measured real disk wander (8–29 cm × ±19 cm, banked in
  `bank_manifest.json`) is the queued `sim-disk-position-prereg-draft`
  item; drawing it here would break per-seed pairing against the
  banked v0 rows. Explicitly chosen, not overlooked.

## Execution path and cost (sequenced per owner 09:32Z)

On GPU release, `sim_parallel_oracle.py` runs FIRST (the
owner-sequenced item, its own
[pre-reg](2026-08-12-prereg-sim-parallel-rollouts.md)); this rerun
is the second thing on the box and inherits the outcome:

- **Path A — oracle GREEN at 2 and 8 workers**: arms run on the
  parallel path at its validated settings (registered use per that
  pre-reg's decision rule). ~20–30 min per 100-seed policy arm →
  3 policy arms + cheap hold ≈ **1.5–2.5 h wall, ~2–3 GPU-h**.
- **Path B — oracle FAIL**: sequential arms on the GPU-compositor
  path (94 ms/tick under load, measured), ~6–9 h wall projected from
  the spot20 pace. **Gate ≤ 10 GPU-h.** Abort rule: first policy arm
  > 2.5 h wall → pause after the in-flight arm, reassess in-channel.
- Either path: arm order `er60k_v3`, `hold_v3`, `teacher80k_v3`,
  `ftrig4k_v3` (reference + floor first, then the confirmatory
  read, then the open cell).

## Finalization checklist (runs at owner unhold, before launch)

1. Owner call on the arm-set decision points (teacher80k add /
   snap30k drop / rungs stay dead).
2. Re-pin HEAD commit + asset-manifest hash in the param sheet
   (machine pin unchanged: this box, EGL, MuJoCo 3.11.0).
3. Param sheet in-channel with a stated objection window; DRAFT →
   REGISTERED stamp on this post.
4. Babysit registry entries at launch; first-poll GPU-utilization
   check per standing rule.
