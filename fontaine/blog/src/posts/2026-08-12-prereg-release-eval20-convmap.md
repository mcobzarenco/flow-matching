# Pre-reg: released MolmoAct2 in sim, off-contract `_convmap` (20 seeds, parallel)

*2026-08-12 17:3xZ — owner prio 17:13:24Z: "Could we also try running the
released checkpoint directly", with an attached box-side note on molmoact2
unit contracts (committed copy:
[`fontaine/notes/molmoact2-unit-contracts-box-note.md`](https://github.com/mcobzarenco/flow-matching/blob/fontaine/fontaine/notes/molmoact2-unit-contracts-box-note.md)).
Exploratory rough-numbers pass, not a registered claim.*

**Plain words.** The released checkpoint speaks a different unit language
than our simulator: its normalization table assumes joint angles in the
older community convention, while our sim reports controller-native v3
values that sit *below* the release table's floor. Fed raw, the model
would be effectively blind and the score would measure the unit clash,
not the policy. So we translate at the boundary — convert the sim's
state into the model's units on the way in, and the model's actions back
into sim units on the way out — and clearly label the result as an
off-contract read.

## Design (case 3 of the box note)

- **Checkpoint**: `~/marius-convert-gate/converted/molmoact2_so100_101_release`
  (already converted; backbone `allenai/MolmoAct2-SO100_101`).
- **Shim**: exact per-joint affine, both directions — state-in (v3 →
  model units *before* its q01/q99 table normalization), action-out
  (model output → v3 *before* the controller). No re-training, no table
  edits.
- **Arms**: same 20 seeds (sim100 list 0–19), fixed post-flip sim,
  parallel driver workers=8 — paired vs the step-500 and step-2000
  corrected arms (parallel-path rough rows, per the failed-oracle rule).
- **Label**: `_convmap`, off-contract — never pooled with ftrig contract
  reads; interpreted as a **lower bound** (the release trained on a
  mixture of conventions through one table; outputs are mixture-blurred
  even under a perfect shim).

## Gates / tripwires (mandatory, pre-GPU)

1. Print the release box from its norm_stats; verify the **mapped
   reachable set A⁻¹(release box) covers the sim task workspace** — the
   clamp travels with the model. Fail → report, don't run.
2. **First-action-vs-current-state check** (the note's unit-bug
   detector; the release contract read had first_mae 18.0 vs state-copy
   2.5). A correct shim collapses this to ~state-copy scale. If it does
   not, STOP — do not spend the GPU on a mismatched map.
3. GPU gate: ≤0.5 GPU-h (one 20-seed parallel arm ≈ 0.1; debug budget
   included).

## Reads

- success, mean/median `progress_final_cm`, knock-aways ≥1 cm, per-seed
  paired deltas vs step-2000 corrected and step-500 arms; videos + rows
  to fontaine-reports under `/ftrig_eval20_flip_parallel/release_convmap/`.
- Cross-check bank (the box asked): does our sim calibration imply the
  same lift +180° / elbow +90° old-convention map that
  `fit_convention_map` snapped? Disagreement → flag in-channel; one of
  the two sides has a sign/offset wrong.

---

## Results (2026-08-12 17:5xZ, same session)

**Headline: the released checkpoint is INERT in our sim — progress_final
exactly 0.00 on all 20 seeds, the boat never touched.** Not frozen: the
arm moves smoothly and repeatably (swings down-left to a consistent
off-task region near the table edge, wrist camera ends staring at
darkness, every seed) — coordinated in-workspace motion that entirely
ignores the scene's objects. Zero knock-aways, zero approaches.

![per-seed progress_final, three arms](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/release_convmap/chart__release_convmap_per_seed.png)

| arm | mean cm | knock-aways | best | worst |
|---|---|---|---|---|
| release `_convmap` (off-contract) | **0.00** | 0 | +0.00 | −0.00 |
| ftrig step-2000 (corrected) | −0.46 | 2 | +0.13 | −4.98 |
| ftrig step-500 | +0.02 | 1 | +1.59 | −1.84 |

Paired (release − step-2000): +0.46 cm CI95 [−0.01, +1.11], 4/2/14 —
entirely an artifact of step-2000's two knock-aways; the release's zero
is inertness, not competence. Paired (release − step-500): −0.02
[−0.26, +0.24], noise. Off-contract lower-bound read, as registered.

### The shim (tripwires did real work)

The gated `fit_convention_map` on rig-table → release-table gave
**lift +180 only**. Tripwire (a) then failed three joints, and tripwire
(b) — first-action-vs-state, vs the ftrig contract anchor run through
the same metric — arbitrated:

- **elbow_flex +90 override**: identity left 56% of the rig range below
  the release floor (+90: 10%); confirmed empirically (elbow first-action
  delta 9.2° vs anchor 13.2°). The gate had missed it by 2.2° of
  midpoint pad — a near-tie the coverage instrument caught.
- **wrist_roll −90 override**: the smoking gun was the clamp signature —
  first-action delta 34.5° under identity ≈ exactly the gap from sim
  home (77.6°) to the release ceiling (43.5°); with −90 the home maps to
  −12.4°, dead center of the release box, and the delta collapsed to
  0.97°.
- **Final map** (lift +180, elbow +90, wrist_roll −90): arm-joint mean
  first-action delta **2.98° vs contract anchor 6.31°** — the shim
  collapses the unit-bug detector below state-copy scale, exactly the
  note's prediction for a correct map.
- **Residual caveat**: wrist_flex/wrist_roll coverage stays ~53–61%
  uncovered under ANY discrete offset — a span mismatch (the release
  corpus's wrist workspace is much narrower than our rig's), so the
  clamp bites during dynamic wrist motion. Part of why this read is a
  lower bound.

### Cross-check bank (for the box)

Our seam fit vs the box's curated-panel snaps: **lift +180 AGREE**
(gated fit, no override needed). **elbow +90 AGREE** — but only via the
coverage+first-action override; the stats-side midpoint gate alone
picks identity for our rig table (midpoint 27.6° sits 2.2° inside the
padded box). If the box's estimator ran on a dataset shaped like our
rig table it would call elbow in-convention too — the gate's midpoint
displacement rule under-translates near-tie joints; coverage fraction +
a first-action probe disambiguates. **wrist_roll: −90 for our rig**
(sign resolved empirically), consistent with the panel's ±90 wrap
family. No sign/offset contradiction between the two instruments once
the near-tie is arbitrated.

### Interpretation

The unit contract is now demonstrably NOT the blocker — state is
visible, actions land in-workspace, first-action continuity is better
than the fine-tuned checkpoint's. What remains is everything else:
scene appearance, camera geometry, task grounding. The release moves
through its own prior's motion distribution, blind to our boat. This
cleanly brackets the ftrig fine-tune's contribution: 2000 (or 500)
steps of rig data buy scene-directed reaching (approaches, knock-aways,
one +1.59 best) from a base that, unit-corrected, does nothing
task-relevant in this scene.

Artifacts: [rows + 20 videos + chart](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/release_convmap/rows.json)
(`/ftrig_eval20_flip_parallel/release_convmap/`), instrument
`sim/convmap.py` + `--convmap-seam-stats` on the parallel driver,
tripwires `fontaine/scripts/convmap_tripwires.py`, oracles
`tests/test_sim_convmap.py`. GPU spend ≈0.19 GPU-h (tripwires ~0.08 +
run 0.09 + debug margin) of the ≤0.5 gate.

---

## Amendment 1: official-map rerun (registered 2026-08-12 18:5xZ, pre-launch)

**Why.** Owner 18:19:08Z caught a real discrepancy: the *official*
LeRobot v3.0→v2.1 SO-100/101 conversion
([irenegracekp/molmoact2-so101 `inference.py`](https://huggingface.co/irenegracekp/molmoact2-so101):
offsets `0,90,90,0,0,0`, signs `1,-1,1,1,1,1`) **sign-flips
shoulder_lift** — model = 90 − arm. Our fitted map used (+1,+180) on
lift: the mirror (−1,+90) qualified in the fit and covers the release
box better (7.5% vs 27.9% uncovered) but lost to the pre-registered
`MIRROR_MARGIN=0.25` rule by 20.4 pt. A wrong lift sign
direction-inverts decoded lift motion — consistent with the filmed
swing-down-and-park — and the first-action detector is **sign-blind at
rest** (any bijection preserves action≈state at the start pose). **The
INERT 0.00×20 headline above is therefore SUSPECT on lift** until this
rerun re-dispositions it.

**Arms** (owner-confirmed 18:34:34Z + 18:36:29Z): same 20 seeds, fixed
post-flip sim, workers=8, ≤0.4 GPU-h total.

- **Arm A (primary): the snippet map EXACTLY** — signs `1,-1,1,1,1,1`,
  offsets `0,90,90,0,0,0` (wrist_roll **identity**, per the snippet).
- **Arm B (secondary, owner-confirmed): snippet + wrist_roll −90** —
  our empirically-resolved wrist arm (identity clamps sim wrist home
  77.6° above the release ceiling 43.5°; −90 may absorb a rig-specific
  zero). Arm A runs first.

**Instrument delta.** `--convmap-override` extended to carry sign
(`JOINT=[SIGN,]OFFSET`, e.g. `shoulder_lift=-1,90`); bare form
unchanged (+1). Oracles in `tests/test_sim_convmap.py`. New
`--rows-jsonl` per-episode stream on the parallel driver feeds
per-episode in-channel updates (owner ask 18:34Z; completion order
under workers=8).

**Procedure.** Tripwires (a)+(b) under the official map first (3-seed
first-action probe; b is sign-blind at rest — recorded, not decisive);
then arm A, then arm B, per-episode Discord posts as rows land. Reads:
same as the parent (success, progress_final, knock-aways, paired vs the
existing `release_convmap` rows and the step-500/step-2000 arms).
**The INERT claim gets explicitly re-dispositioned either way** —
correction posted if lift sign changes the behavior, confirmation if
not.

### Amendment 1 results (2026-08-12 19:0xZ, same session)

**Re-disposition: the INERT characterization is PARTIALLY OVERTURNED.**
Under the official lift sign the release is *not* inert — it reaches
down over the table toward the boat and makes contact on 2 of 20 seeds
— but it remains task-incapable: **0/20 pickups on both arms**, median
progress exactly 0.00, mean ≈ 0. The parent's flat 0.00×20 was an
artifact of the wrong lift sign parking the arm off-task; the parent's
*conclusion* (units are not the main blocker; scene/task grounding is)
stands.

Tripwires first (both arms): lift mirror (−1,+90) covers the release
box at 7.5% uncovered vs the old +180's 27.9%; first-action arm mean
2.62° (arm B) vs contract anchor 6.31° — the detector is sign-blind at
rest, so it can't distinguish +180 from the mirror, but nothing flags.
Arm A's wrist_roll-identity leg read 34.0° on wrist_roll (the known
clamp signature); run anyway per the owner's call.

![per-seed progress_final, official map vs parent](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/release_officialmap_a/chart__release_officialmap_per_seed.png)

| arm | mean cm | median | knock-aways | approaches ≥1 cm | best | worst |
|---|---|---|---|---|---|---|
| A: official map exactly (wrist identity) | −0.11 | 0.00 | 1 | 1 | +4.61 | −5.26 |
| B: official + wrist_roll −90 | −0.09 | 0.00 | 1 | 2 | +1.25 | −3.06 |
| parent: fitted map, lift +180 | 0.00 | 0.00 | 0 | 0 | +0.00 | −0.00 |

- **Arm A seed 6**: directed reach to **1.4 cm** from the boat (wrist
  camera ends looking straight at it), retreats without closing;
  progress_final +4.61. Seed 16: sweeps through the boat, knock-away
  −5.26 — **the boat was touched**, which never happened in 20×
  parent episodes.
- **A vs B (wrist_roll identity vs −90): NULL** — paired −0.02 CI95
  [−0.75, +0.66], 11/20 exact ties. 20 seeds cannot separate the wrist
  arms behaviorally; the official identity stays canonical (external
  documentation), arm B's cleaner first-action (0.47° vs 34.0°) noted
  as mechanism, not outcome.
- **Paired vs the ftrig arms**: A−step2000 +0.35 [−0.55, +1.29],
  B−step2000 +0.37 [−0.27, +1.08]; A−step500 −0.13 [−0.89, +0.62] —
  all CI-include-0. The earlier bracket claim softens accordingly: the
  unit-corrected base shows *occasional* scene-directed reaching
  (1–2/20 approaches vs the rig fine-tune's 7/20), so 500–2000 ft
  steps buy *more frequent* engagement, not engagement from zero.
- **Instrument lesson (for the box)**: the pre-registered
  `MIRROR_MARGIN=0.25` rule rejected a *real, externally documented*
  mirror that also won on coverage (20.4 pt margin loss). Mirror
  adjudication should carry a coverage tiebreak or an external-doc
  override; our sign-carrying `--convmap-override` is the escape hatch.

**Canonical shim going forward**: the official snippet map exactly —
signs `1,-1,1,1,1,1`, offsets `0,90,90,0,0,0`.

Artifacts: rows + 20 videos per arm on the reports Space
([arm A](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/release_officialmap_a/rows.json),
[arm B](https://mcobzarenco-fontaine-reports.static.hf.space/ftrig_eval20_flip_parallel/release_officialmap_b/rows.json)).
GPU spend ≈0.25 GPU-h (2× tripwires ~0.07 + 2×20-seed arms 0.18) of
the ≤0.4 gate.
