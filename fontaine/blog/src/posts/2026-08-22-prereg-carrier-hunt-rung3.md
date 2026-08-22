# Pre-registration DRAFT: carrier-hunt rung 3, both contingent branches (drafted ahead of the ch0fix verdict)

*Draft cut 2026-08-22 23:3xZ (work session), riding the live
`pdnorm_ch0fix` train (rung 2, done ~11:2x–11:4xZ 08-23) so the
verdict session **executes instead of drafting** — queue item
`carrier-hunt-rung3-prereg`, the [ch0-affine
pre-reg](2026-08-22-prereg-clean-ch0-affine.md)'s two registered
branches, both drafted BEFORE the verdict exists. Exactly one branch
fires, selected mechanically by rung 2's frozen grid: **branch A** if
ch0fix lands ≥20/100, **branch B** if ≤10/100; the 11–19 band fires
neither (per-channel MAE + slices + videos to the owner, no claim, no
launch). Launch is delegated per the standing no-GO-ask rule: the
selected cell fires at the verdict session once its materializer's
oracles are green, announced in-channel. DRAFT status: neither
materializer exists yet; the reads, grids, split, and dataset names
below are frozen now and do not move.*

**Plain words.** A training run is going right now that will tell us
whether stretching one joint's recorded values fixes seven poisonous
robot episodes. This page writes down — before that answer arrives —
exactly what we do in either case, so tomorrow's session can act
instead of deliberate. If the stretch works (branch A): the fix
touched two copies of the joint's values — the *commanded* positions
and the *measured* ones — and we ask whether fixing only the commanded
half suffices, because a fix at the output side alone is something any
data pipeline could apply mechanically. If the stretch fails (branch
B): every named suspect channel is then exhausted, and we start
splitting the seven episodes themselves — train on half, see if the
poison came along. We measured tonight that the joint anomaly is
spread evenly across all seven episodes, so the split is chosen by
frame balance, not by suspicion, and it halves the search space for
the same price every earlier experiment paid (~17 GPU-hours). Either
way, nothing launches until the current run's verdict picks the
branch.

## The measured basis (banked this session, CPU, record-only)

Instrument `fontaine/scripts/carrier_rung3_basis_read.py`, report
`reports/analysis__carrier_rung3_basis.json`; sanity oracles green
(demos ch0 mean/std reproduce the rung-2 affine constants to 1e-12;
the holdout search reproduces democlean's and `clean_ch0fix_n`'s
verified `(2,)` draws).

**The ch0 anomaly is episode-uniform, not episode-concentrated.**
Per-episode ch0 action spread and KS vs demos, all 7 episodes:

| ep | frames | ch0 mean | ch0 std | ch5 max | KS ch0 vs demos |
|---|---|---|---|---|---|
| 0 | 511 | +3.93 | 8.66 | 29.96 | 0.374 |
| 1 | 509 | +0.20 | 11.43 | 30.72 | 0.319 |
| 2 (held out) | 373 | +3.21 | 9.12 | 20.91 | 0.383 |
| 3 | 380 | +12.27 | 7.03 | 19.62 | 0.469 |
| 4 | 694 | −7.11 | 8.21 | 32.30 | 0.399 |
| 5 | 484 | +2.26 | 8.39 | 18.19 | 0.354 |
| 6 | 448 | +2.03 | 5.68 | 27.40 | 0.349 |

Every episode's std sits at 5.7–11.4 vs demos' 28.0 — the compression
is a property of the whole collection, not of an outlier episode (the
gripper shortfall likewise: every ch5 max ≤ 32.3 vs the 41.69
convention). Two consequences frozen into the branches: (a) if ch0 IS
the carrier, per-episode slicing could never have found it — the
bisection ladder is correctly sequenced *behind* the channel edits;
(b) if ch0 is NOT the carrier, no measured channel ranks the episodes,
so branch B's split is frozen on **frame balance**, with the weak
anomaly prior (worst-KS ep 3, offset-mean ep 4) used only to pick
which half trains first.

**Dataset names pinned by holdout-draw search** (the gripfix
Amendment-1 class, pre-applied; `holdout_episodes` is a pure function
of the repo name):

- Branch A: **`mcobzarenco/so101_pick_place_clean_ch0fix_act_j`** —
  7 episodes, draw `(2,)`: the clean-side train split stays
  episode-identical to democlean's ({0,1,3,4,5,6}, 3026 kept frames).
- Branch B cell: **`mcobzarenco/so101_pick_place_clean_ep015_c`** — 4
  episodes (originals [0,1,2,5] in ascending order, re-indexed 0–3),
  draw `(2,)` = original episode 2, the **decoy**: episode 2 is
  democlean's own never-trained holdout, carried in each subset
  precisely so the mandatory ≥1-episode holdout lands on it and the
  cell trains its full intended subset {0,1,5} while preserving
  "episode 2 never trains" semantics.
- Branch B registered follow-up:
  **`mcobzarenco/so101_pick_place_clean_ep346_a`** — originals
  [2,3,4,6], draw `(0,)` = original episode 2, same decoy design,
  trains exactly {3,4,6}.

## Branch selection (frozen; no judgment at the verdict)

The rung-2 grid verdict on `pdnorm_ch0fix`'s sim100 battery is the
selector: **≥20/100 → branch A. ≤10/100 → branch B. 11–19 → neither**
(rung 2's ambiguous-band protocol runs; any rung-3 launch would then
need a fresh registered amendment). The verdict session's only jobs
are: bank the rung-2 verdict per its own pre-reg, build the selected
branch's materializer, run its oracles, fit-smoke, launch, announce.

---

## Branch A — the action-only cell (fires iff ch0fix ≥ 20/100)

**Question**: rung 2's affine touched ch0 in BOTH the action and
state columns. Which half carries the recovery? The queue item named
two decomposition axes; one is already settled at zero cost — the
banked constant-freeze read pre-refuted the *shift* form (KS unmoved
0.295→0.286/0.308) and the shift component of the winning affine is
0.05 of demos' std, so **scale-vs-shift is closed: the scale is the
affine**. The open axis is **action vs state**, and it is the axis
with production value: an action-only fix is output-side data hygiene
any pipeline could apply mechanically; a state-side requirement means
the fix must stay consistent with what the policy is told about the
world — a stronger, more invasive rule.

**The cell.** Mix = `grasp_demos_v2/merged` +
`so101_pick_place_clean_ch0fix_act_j` (×4), and nothing else — the 7
clean episodes with the ch0 **action column only** transformed by the
rung-2 frozen affine, constants verbatim:

**x′ = 0.0923439813196304 + (x − 1.481974338423806) × 2.755193138766973**

The ch0 **state column stays byte-identical to source** (that is the
treatment), as does every other channel, count, camera stream, and
annotation. `--recompute-stats` gives the set its own pdnorm rows —
the action row's ch0 scale must move ×2.7552 while the state row's
stays clean-like, which doubles as the live oracle that exactly one
column was touched.

Frozen transform decisions: the rung-2 affine exactly, not
re-estimated; ch0 only; **action column only**; float64 transform
cast back to source dtype.

Materializer: `fontaine/scripts/make_clean_ch0fix_act_dataset.py`
(exec session; sharing code with `make_clean_ch0fix_dataset.py` is a
file-format freedom). Oracles: action ch0 equals the affine of source
exactly; state ch0 byte-identical to source; every non-ch0 column
byte-identical; frame/episode counts identical; transformed action
range inside demos' observed support [−110.0, +79.6]; holdout draw is
`(2,)`; the rung-2 no-op guard (refuse if source ch0 std exceeds 20).

**Reads and grid** (primary sim100 flow leg at step 3000, protocol
byte-identical to the five banked batteries):

- **≥ 20/100** → **the action column alone carries**: the
  mix-hygiene rule is output-side and mechanically deployable;
  state-side consistency is not required. Mechanism cell closes the
  ch0 thread; no further rung named.
- **≤ 10/100** → **the joint edit was necessary** — the recovery
  needs the state column too. Named follow-up (delegation standing):
  the state-only complement cell, own draft, same discipline; the
  mechanism claim is capped per the honesty clause below.
- **11–19** → ambiguous (the control's band): no claim.

**Paired reads recorded alongside**: vs **ch0fix's own banked number**
(THE read — same 7 episodes, the both-columns twin), vs democlean
8/100, vs onerig 28/100, vs control 11/100.

**Registered asymmetry note (honesty clause).** The action-only edit
deliberately manufactures a *within-dataset* action/state
inconsistency on ch0: the model trains on frames whose state says
"compressed sweep" while the action commands the stretched one. A ≤10
outcome therefore cannot fully distinguish "the state half carries"
from "the inconsistency itself is an artifact that re-poisons" — the
claim caps at "action-only is not a sufficient fix; the joint edit
stands as the deployable form". The paired-progress read (the gripfix
−2.07 cm precedent) carries artifact detection.

---

## Branch B — the content-bisection cell (fires iff ch0fix ≤ 10/100)

**Question**: with the gripper amplitude struck at rung 1 and the ch0
marginal struck at rung 2, the manifold probe's named-suspect list is
exhausted — the carrier is content-level. Rung 2's ≤10 branch
mandated a design pass before committing to per-episode leave-one-out
(~17 GPU-h × 7 ≈ 119 GPU-h). **This is that design pass, and the
verdict is bisection**: one cell per rung halves the candidate set at
the standard cell price; a localized carrier is convicted to a single
episode in ≤3 rungs (~51 GPU-h worst case), and a *distributed*
carrier — which per-episode LOO cannot even represent — is detected
in two.

**The cell.** Mix = `grasp_demos_v2/merged` +
`so101_pick_place_clean_ep015_c` (×4), and nothing else — episodes
{0,1,5} of clean plus decoy episode 2, **every kept frame
byte-identical to source** (no value edits of any kind; the treatment
is subset membership). The trained clean content is exactly episodes
{0,1,5} (1,504 frames; democlean trains these plus {3,4,6}'s 1,522).

**Frozen split decisions**: the six trained episodes split
frame-balanced — **{0,1,5} (1,504) vs {3,4,6} (1,522)**; no measured
channel ranks the episodes (basis table above), so balance is the
criterion. The first cell **trains {0,1,5} and drops {3,4,6}**, the
half holding the two most marginal-anomalous episodes (ep 3: worst
per-episode KS 0.469, +12.3 mean offset; ep 4: −7.1 mean offset) —
under the weak suspects prior this maximizes the chance of the
decisive outcome (recovery ⇒ carrier localized in the dropped half).
Decoy-holdout design as pinned above; episodes re-indexed 0..3 in
ascending original order.

Materializer: `fontaine/scripts/make_clean_subset_dataset.py` (exec
session; takes the episode list + decoy). Oracles: episode set
exactly the pinned originals in ascending order; every kept frame's
every column byte-identical to source; per-episode frame counts match
the basis table; holdout draw `(2,)` for `ep015_c` / `(0,)` for
`ep346_a`; the `--dataset-repeat 'mcobzarenco/so101_pick_place*=4'`
glob matches the new name.

**Registered dose confound (honesty clause, priced in before
launch).** The cell trains 1,504 clean frames where democlean trains
3,026 — effective share ~0.34% vs 0.69% at the verbatim ×4 repeat
(one-dataset-delta discipline; no repeat-flag compensation, that
would be a second treatment variable). A **recovery is therefore
ambiguous between "carrier in the dropped half" and "half dose is
below the poisoning threshold"** — rung 2 of this sub-ladder, the
complement cell `ep346_a`, is the registered follow-up on EITHER
verdict and disambiguates:

- cell recovers, complement collapses → carrier localized in {3,4,6};
  recurse there.
- cell collapses (poison survives at half dose in {0,1,5}) →
  complement next: if it ALSO collapses, the carrier is
  **distributed** — bisection and LOO are both moot at this grain;
  escalate to the owner with content forensics, no auto-launch beyond
  the complement.
- both recover → **dose/dilution, not localized content** — a
  different, cheaper mechanism class (share-threshold sweep), own
  pre-reg.

**Reads and grid** (primary sim100 at 3000, protocol byte-identical):

- **≥ 20/100** → grasping recovers without {3,4,6} at half dose;
  complement cell next (localization vs dose per the table above).
- **≤ 10/100** → the poison rides {0,1,5} at half dose; complement
  cell next (distributedness test before any recursion).
- **11–19** → ambiguous: no claim, complement decision escalates to
  the owner.

**Paired reads recorded alongside**: vs democlean 8/100 (THE read —
same recipe, content subset), vs onerig 28/100, vs control 11/100.

---

## Shared spec (both branches)

**Command**: the democlean launcher verbatim
(`launch_local_grasp_sft_v2_joint_1gpu_pdnorm_democlean_h100.sh`)
with exactly one delta: the dataset name in `--train-data` (the
repeat glob matches unchanged). Per-dataset flow norm, joint
`--insulate-flow` ce-weight 1.0, `--recompute-stats`, eff-batch 96,
decoder-lr 5e-5 / backbone-text-lr 1e-5, image-augment 0.8, holdout
0.1, eval-250, save-500 + keep-latest-optimizer pruner from launch,
**3000 steps, seed 0** (same-seed comparability with all six
anchors). Fit smoke first; compute-app abort guard + policy-server
check carried.

**Anchors (all banked)**: ch0fix (the rung-2 number, whichever band
it lands in), democlean 8/100, gripfix 5/100, onerig 28/100, control
11/100, convicted 1/100.

**Secondary — drift guard**: Δeval(1000−500) ≤ +0.30, record-only
(the probe cannot clear these cells — decoupling banked four times).

**Tertiary — panel guard** at endpoint vs the disc-1000 banked npz,
frozen house rule (fail = worse than +0.05 CI-excl-0), truthfit
rewear alongside — unchanged from the last five cells.

**Record-only**: the recomputed pdnorm rows (branch A: action-row ch0
scale ×2.7552 with the state row pinned ≈ clean's — the
one-column-touched oracle; branch B: subset rows vs clean's banked
rows, no edit expected); eval-250 curve vs the democlean curve.

## Gates and boundaries

- **GPU-hours gate: 17 per fired cell** (train ~13.7 measured on the
  identical recipe + battery ~3); exactly one cell fires from this
  draft.
- **Launch window**: the ch0fix verdict session, immediately after
  the rung-2 verdict banks — delegated, announced in-channel, never
  gated on an owner GO. The H100 is free at that boundary (ch0fix
  train + battery complete).
- **Boundaries**: step-1000 drift read (record-only); step-3000
  endpoint → the battery script pattern (clone
  `launch_gripfix_endpoint_battery.sh`, name deltas only) → verdict
  through the branch grid.
- **Checkpoint policy**: saves under
  `~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm_ch0fix_act`
  (A) / `..._pdnorm_ep015` (B); endpoint banks weights-only on any
  decisive read.
- **Seed policy**: seed 0 (comparability); ×4 repeat carried
  unchanged; branch B's share change is the registered confound
  above, not a knob.

*Objection/decision path: branch selection is mechanical from the
rung-2 grid; reads, grids, split, affine constants, and dataset names
frozen as drafted. The exec session's only freedoms are materializer
file-format details and the in-window launch timing. Any spec change
lands as a registered amendment first.*
