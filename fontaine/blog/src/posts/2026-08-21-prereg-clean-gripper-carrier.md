# Pre-registration DRAFT: the gripper-carrier isolation cell (demos + gripper-remapped clean)

*Draft cut 2026-08-21 06:4xZ (the democlean endpoint-close session),
the [demos+clean pre-reg](2026-08-20-prereg-demos-plus-clean.md)'s
named ≤10-branch follow-up, drafted at the verdict (democlean
**8/100 — clean convicted, sufficiency proved**). Launch is delegated
per the standing no-GO-ask rule: the cell fires at the next free GPU
window once the transform tooling lands its oracles, announced
in-channel at launch. DRAFT status: the transform materializer named
below does not exist yet — the exec session freezes the final
command block in-channel before launch; the reads and grid below are
frozen now and do not move.*

**Plain words.** We now know seven episodes of real-robot data poison
grasping all by themselves — 0.7% of training is enough to drop
success from 28/100 to 8/100. We also know, from measuring those
episodes every way we could, that they are surprisingly normal except
for one thing: their gripper never opens all the way. Every other
dataset teaches "open" as roughly 40–42 units; these seven episodes
teach it as at most 32 — about 25% short. This experiment edits
exactly that one number: same seven episodes, gripper values rescaled
so "open" matches everyone else's convention, nothing else touched.
If grasping survives, one channel's amplitude convention was the
whole poison — a cheap, checkable data-hygiene rule for every future
mix. If grasping still collapses, the poison lives somewhere subtler
in those episodes, and we've eliminated the loudest suspect. Either
way the production recipe stays untouched.

## Where this cell comes from

The [democlean verdict](2026-08-20-prereg-demos-plus-clean.md)
(8/100, banked 06:3xZ 08-21) proved clean-alone sufficiency and
refuted composition-only. The banked mechanism reads localize what is
actually strange about clean:

- **Gripper amplitude compression** (manifold probe, frozen reads):
  clean's open plateau sits at q99 30.6 / max 32.3 raw where demos
  command exactly 41.69 (bang-bang {0, 41.69}) and v2 reaches 40.2 —
  a ~25–27% shortfall on the channel where the demos are binary.
  Behaviorally clean cycles normally on its own range (1.71
  transitions/ep) — the anomaly is amplitude, not missing behavior.
- **Modest ch0 shoulder-pan shift** (the only channel exceeding the
  demos↔v2 reference KS both ways; also the ×2.84 worst pdnorm
  amplification ratio) — the named residual suspect if this cell
  exonerates the gripper.

One-variable discipline, third rung: convicted→onerig subtracted
clean (28/100), convicted→democlean subtracted v2 (8/100), this cell
edits ONE CHANNEL of clean and keeps everything else byte-identical.

## The cell

**Mix = `grasp_demos_v2/merged` + `so101_pick_place_clean_gripfix`
(×4), and nothing else** — where `clean_gripfix` is the 7 clean
episodes with the gripper channel (ch5, action AND state) rescaled by
the frozen scalar **41.69 / 32.3 = 1.2907** (zero fixed point
preserved: closed = 0 stays 0; open plateau lands on the demos'
41.69 convention). All other channels byte-identical; episode count,
lengths, camera streams, annotations untouched. The transform is
output-side data hygiene — exactly what a deployment could do.

Frozen transform decisions (so the exec session has no degrees of
freedom): scalar multiply, not per-episode affine (one number, one
suspect); ch5 only; both action and state columns (they must stay
consistent for the state-conditioned policy); the scalar is pinned to
the banked manifold-probe numbers (41.69 demos open command / 32.3
clean max), not re-estimated. `--recompute-stats` then gives the
transformed set its own pdnorm row from the transformed values, as it
would any dataset.

Materializer: a new
`fontaine/scripts/make_clean_gripfix_dataset.py` (exec session) with
oracles: ch5 columns equal source × 1.2907 exactly; every non-ch5
column byte-identical; frame/episode counts identical; transformed
max ≈ 41.69 ± the bf16 class; a no-op guard that refuses if the
source already exceeds 40.

## Command

The democlean launcher verbatim
(`launch_local_grasp_sft_v2_joint_1gpu_pdnorm_democlean_h100.sh`)
with exactly one delta: `so101_pick_place_clean` →
`so101_pick_place_clean_gripfix` in `--train-data` and the
`--dataset-repeat` pattern. Per-dataset flow norm, joint
`--insulate-flow` ce-weight 1.0, `--recompute-stats`, eff-batch 96,
decoder-lr 5e-5 / backbone-text-lr 1e-5, image-augment 0.8, holdout
0.1, eval-250, save-500 + the keep-latest-optimizer pruner from
launch (44G/save class), **3000 steps, seed 0** (same-seed
comparability with all four anchors). Fit smoke first; compute-app
abort guard carried.

## Baselines and anchors (all banked)

- **democlean 8/100** — the direct paired arm: same 7 episodes,
  un-remapped. THE comparison this cell exists for.
- **Onerig 28/100** — the healthy two-dataset cell (demos + v2).
- **Demosonly control 11/100** — the never-mixed floor.
- **Convicted three-way 1/100** — the original collapse.
- Probe-curve anchors: democlean 11.82→4.6848 (collapsed cell,
  healthy-looking curve — the probe is decoupled and record-only
  here), onerig …→4.53, convicted …→6.17.

## Reads and frozen decision grid

**Primary — sim100 flow leg at step 3000**, protocol byte-identical
to the democlean/onerig/convicted batteries (unseen seeds 0–99,
euler-10, episode 30 s, execute-horizon 30, bfloat16 decoder,
`--stats-repo-id grasp_demos_v2/merged`, `--clutter-appearance
standins`).

Same numeric grid, interpretations adapted:

- **≥ 20/100** → **the gripper amplitude convention IS the carrier**:
  a one-scalar, one-channel edit de-poisons the 7 episodes. Banks
  same-session; the production implication is a concrete mix-hygiene
  rule (per-channel amplitude-convention check before any mix), and
  clean re-enters the usable pool in remapped form.
- **≤ 10/100** → **the gripper amplitude is NOT the (sole) carrier**:
  the poison survives the loudest-suspect edit. Named follow-up
  ladder (drafts, delegation standing): the ch0 shift next, then
  content-level slicing (per-episode leave-one-out at ×4 repeat is 7
  cheap cells if it comes to that).
- **11–19** → ambiguous (the control's band): per-channel MAE,
  per-slice breakdown, videos to the owner, no claim.

**Paired reads recorded alongside** (`sim100_paired_read.py`): vs
democlean 8/100 (the carrier estimate — THE read), vs onerig 28/100
(full-recovery yardstick), vs control 11/100 (does the remapped mix
at least stop hurting).

**Secondary — drift guard**: Δeval(1000−500) ≤ +0.30, same
instrument, record-only expectations reset by the democlean
decoupling: the probe canNOT clear this cell, only flag drift.

**Tertiary — panel guard** at endpoint vs the disc-1000 banked npz,
frozen house rule (fail = worse than +0.05 CI-excl-0), truthfit
rewear alongside — unchanged from the last three cells.

**Record-only**: the recomputed `clean_gripfix` pdnorm row vs clean's
banked row (ch5 scale should move ×1.29, everything else pinned —
this doubles as a live materializer oracle); eval-250 curve vs the
democlean curve (same-seed, one-channel-delta twin runs).

## Gates and boundaries

- **GPU-hours gate: 17** (class gate: train ~13.7 measured on the
  identical recipe + battery ~3).
- **Launch window**: next free GPU window after the materializer's
  oracles are green — delegated, announced in-channel, never gated on
  an owner GO. The H100 is free at this draft's cut (battery leg 2
  finishing); if the exec session is this one's chained successor,
  the window is immediate.
- **Boundaries**: step-1000 drift read (record-only per above);
  step-3000 endpoint → the same battery script pattern → verdict
  through the grid.
- **Checkpoint policy**: saves under
  `~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm_gripfix`;
  endpoint banks weights-only per the standing rule on any decisive
  read.
- **Seed policy**: seed 0 (comparability); the ×4 repeat and 0.70%
  share carried unchanged.

*Objection/decision path: reads and grid frozen as drafted; the exec
session's only freedoms are the materializer's file-format details
and the launch window. Any spec change lands as a registered
amendment first.*

## Amendment 1 (registered 08:0xZ 2026-08-21, pre-launch): dataset named `_a` to preserve the episode-identical train split

**Found at exec, before launch.** The episode holdout is a pure
function of `(repo_id, num_episodes, fraction, split_seed)`
(`bijou/data.py::holdout_episodes`). Renaming the clean set therefore
redraws which of the 7 episodes is held out: `so101_pick_place_clean`
(democlean) holds out **episode 2** (373 frames; 3026 trained, 0.69%
effective share), while the draft's `so101_pick_place_clean_gripfix`
would hold out **episode 6** — the run would train on {0,1,2,3,4,5}
where democlean trained on {0,1,3,4,5,6}. That one-episode swap is a
second treatment variable riding THE paired read: a ≥20 verdict could
mean "the gripper convention was the carrier" or "poison-heavy episode
6 left the train split" (per-episode content demonstrably matters —
per-episode LOO is this pre-reg's own named follow-up).

**The fix stays inside the frozen recipe**: the materialized dataset is
named **`so101_pick_place_clean_gripfix_a`**, chosen so its draw is
exactly `(2,)` — the clean-side train split becomes episode-identical
to democlean's ({0,1,3,4,5,6}, 3026 kept frames, 0.69% share, the same
373-frame episode held out). The demos split is untouched (its draw
keys on its own repo id) and the `--dataset-repeat
'mcobzarenco/so101_pick_place*=4'` glob matches the new name
unchanged. No flag, seed, or read moves; the launcher's single delta
is the dataset path. (The draft's "0.70% share" line was itself a
rounding of democlean's logged 0.69% — the `_a` split reproduces the
logged value exactly.)*
