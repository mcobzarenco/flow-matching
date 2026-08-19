# Pre-registration DRAFT: demos + one rig dataset (the post-convict isolation cell)

*Draft cut 2026-08-19 04:3xZ (work session, queue item
`prereg-draft-demos-plus-one-rig`). **EXECUTION IS AN OWNER CALL** —
the [pdnorm pre-reg](2026-08-18-prereg-grasp-sft-v2-joint-pdnorm.md)'s
registered ≤10 grid text reads "Next isolation is an owner call; the
draft names demos + one-rig-dataset as the cheapest next cell", and
that registered carve-out outranks the standing launch delegation.
This draft freezes the cell so the call is a yes/no, not a design
session; no launch happens from the drafting item. Launcher staged:
`fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_onerig_h100.sh`
(full-parse green vs the merged CLI: family-inferred
`molmoact2_joint`, `per_dataset_flow_norm=True`, seed 0, both
datasets resolved).*

**Plain words.** We fixed the statistics table (per-dataset
normalization), we replaced the training machinery that a separate
experiment convicted, and the mixed-corpus model STILL almost never
grasps: 1 success in 100 tries, against 11/100 for the same recipe
trained on simulated demos alone — a statistically solid deficit, not
noise. So the mixture itself — simulated demos and real robot
recordings pulling the same network in different directions — is now
the prime suspect. This experiment shrinks the mixture to its
smallest still-interesting form: the demos plus exactly ONE real
dataset (the big one; the tiny 7-episode dataset is dropped). If this
two-dataset mix still fails, sim/real interference is reproduced in a
much simpler setting we can dissect. If it suddenly grasps, the tiny
dropped dataset (0.65% of training frames!) or the three-way
combination was the poison — which would be a remarkable, checkable
claim. Pass/fail numbers are frozen below before anything runs.

## Where this cell comes from

The [pdnorm verdict](2026-08-18-prereg-grasp-sft-v2-joint-pdnorm.md)
(battery closed 2026-08-19 03:5xZ, post id 1539482938675298354)
convicted the mix with every guard read banked:

- **Mixed-pdnorm 1/100** on unseen 0–99 vs **demosonly control
  11/100** — paired Δ **−10** (CI95 [−16, −5], McNemar exact
  p = 0.002), paired progress −3.49 cm CI-excl-0. The mixed cell is
  significantly worse than its own control, not merely under the bar.
- The **table fix worked as mechanism**: panel guard PASS 29.18 vs
  disc-1000's 58.14 (Δ −28.96 CI-excl-0), receipts exactly on the
  convicted channels (wrist_roll −45.7, wrist_flex −6.1). The fix
  moved what it was built to move — and grasping still collapsed.
- The **machinery was convicted separately** (drift discriminator,
  00:42Z 08-18) and is gone from this platform.

Table fixed, machinery gone, mix still fails → **sim/real
interference at the flow head is the prime suspect**. The registered
next step is the cheapest cell that shrinks the suspect.

## The cell, and why so101_pick_place_v2

**Mix = `grasp_demos_v2/merged` + `so101_pick_place_v2` (×4), and
nothing else.** The convicted mix was demos + v2 (×4) + clean (×4):
1,942,375 demo frames + 130,716 v2 + 13,596 clean, i.e. a 6.92% rig
share of which v2 is 6.26 points and clean 0.65. Dropping clean is
the **minimal delta from the convicted cell**: the new mix keeps
90.6% of the rig frames and holds the rig dose nearly constant (v2
share 6.31% here vs 6.26% inside the convicted mix), while cutting
the dataset count to two.

Why not `so101_pick_place_clean` (the lowest-noise pick)? Because
demos+clean changes two things at once — it removes 90.6% of the rig
frames AND swaps which dataset carries the rig signal — so a pass
would be uninterpretable between "mixing is safe at low dose" and
"v2 was the poison". Demos+v2 is one subtraction from a convicted
cell; its grid (below) is clean in both directions, and demos+clean
remains the named follow-up if this cell exonerates. The tiny clean
set (7 episodes) is also the arm most plausibly fine to drop from any
future production mix, so learning "v2 alone still breaks" or "clean
was load-bearing for the failure" are both directly actionable.

## Command

Staged launcher
`fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_onerig_h100.sh`
= the pdnorm launcher with exactly **one recipe delta**:
`so101_pick_place_clean` deleted from `--train-data`. Everything else
verbatim: per-dataset flow norm, joint objective `--insulate-flow`
ce-weight 1.0, `--recompute-stats` (CE/state tables merged over the
two remaining datasets), `--dataset-repeat
'mcobzarenco/so101_pick_place*=4'` (now matches only v2), eff-batch
96 = micro-12 × 8 chunks, act-ckpt + offload-optim, decoder-lr 5e-5 /
backbone-text-lr 1e-5, image-augment 0.8, holdout 0.1, eval-250 with
`--eval-dataset-breakdown`, save-500, **3000 steps**, **seed 0**
(same-seed comparability policy). Fit smoke (`STEPS=20 SMOKE=1`)
before the full run; compute-app abort guard for the owner
policy-server carried.

**Disk policy (new, from the 08-19 root-disk-full incident):** the
launcher runs a sidecar pruner that deletes superseded offload-optim
`optimizer.pt` mirrors (~31 GiB per save) every 5 minutes, keeping
the latest TWO saves resume-capable. Weights are never touched. The
convicted run's six saves held 252 GiB of optimizer state and filled
the root disk mid-battery; this bounds the same run shape at ~62 GiB.

## Baselines and anchors (all already banked)

- **Demosonly control 11/100** —
  `grasp_sft_v2_demosonly_1gpu_disc/step_001000`, same platform, same
  seeds, same substrate; the SAME control cell the convicted run was
  paired against. Same-episode demos holdout (split is a pure
  function of identical `(repo_id, episodes, fraction, split_seed)`),
  so the demos-slice breakdown curve is directly comparable.
- **Convicted mixed cell 1/100** — `grasp_sft_v2_joint_1gpu_pdnorm`
  step 3000; this run's own most-informative comparison: same recipe
  plus clean.
- **Probe band 44/100** — `joint_corrected@2000`, the healthy class.
- **Panel ladder** (k4l2, wear-corrected class): pdnorm endpoint
  29.18 native / 27.44 truth-fit ≈ disc-1000 re-worn 27.40 ≈ released
  27.14, all at/above the 25.15 midpoint null; state-copy 8.37. The
  [truthfit rewear instrument](2026-08-18-prereg-grasp-sft-v2-joint-pdnorm.md)
  and the `pdnormendpoint` report preset carry over unchanged.
- **Drift anchors**: discriminator Δeval(1000−500) = −1.67 on this
  platform; convicted-mix probe curve 5.45/5.47 plateau → 6.83 peak →
  6.17 endpoint (elevation never retraced).

## Reads and frozen decision grid

**Primary — sim100 flow leg at step 3000** (unseen seeds 0–99,
`sim.rollout_sim` euler-10, episode 30 s, execute-horizon 30,
bfloat16 decoder — the v1/v2 protocol), with the two registered
serving conditions: `--stats-repo-id grasp_demos_v2/merged` (the sim
eval wears the sim demos' row; the worn-row rule from the pdnorm
pre-reg) and `--clutter-appearance standins` (substrate pin — the
demos, the control, and every anchor were produced under stand-ins;
the clutter-patch promotion stays a no-op for this lineage's reads).

Decision bounds, fixed now (same numeric grid as the convicted cell;
interpretations adapted to this mix):

- **≥ 20/100** → the two-dataset mix grasps: sim/real interference
  is NOT generic to mixing at this dose — the failure lived in the
  dropped 0.65% (clean) or the three-way composition. Banks
  same-session with an HTML panel; the named follow-up cell is
  demos+clean (or full mix minus v2) to pin which.
- **≤ 10/100** → interference **reproduced with a single rig dataset
  at ~6% share**: the suspect survives in a two-dataset setting we
  can dissect (dose ladder on the repeat factor, insulation/head
  separation, gradient-conflict instrumentation). Next cell is again
  an owner call; the draft names a repeat-1 dose arm (~1.7% share)
  as the cheapest.
- **11–19** → ambiguous: per-channel MAE, per-slice breakdown,
  videos to the owner before any claim.

**Paired reads recorded alongside** (calibration: the control itself
sits in the 11–19 band, so the ≥20 bar asks the cell to roughly
double its control — same caveat as registered for the convicted
run): `sim100_paired_read.py` vs the control's banked 100 episodes
AND vs the convicted mixed cell's 100 (the second paired read is the
direct "did dropping clean move anything" estimate).

**Secondary — drift guard**: in-train eval probe, Δeval(1000−500) ≤
+0.30 (discriminator rule, same instrument). Failure = new
information (mix-specific drift), grasp read still stands, endpoint
choice re-opens to best-grasping save.

**Tertiary — panel guard, paired at endpoint**: `pdnorm_panel_guard.py`
(now generic) at step 3000 vs the disc-1000 banked npz on shared
frames, frozen house rule: fail = worse than +0.05 CI-excl-0.
Per-motor deltas recorded; the mechanism predicts wrist_roll /
wrist_flex move again. Truthfit rewear + ladder restamp
(`--endpoint <row>`) run at the endpoint session exactly as for the
convicted cell.

**Record-only**: eval-250 probe curve vs the convicted run's
(does removing clean flatten the 2250–2750 elevation?); rig-slice
breakdown as the first two-dataset per-dataset-normalized rig curve;
token-leg sim100 may run as unregistered corroboration.

## Gates and boundaries

- **GPU-hours gate: 17** — train ~13 (3000 × ~15.1 s/step measured on
  this exact platform shape) + endpoint sim100 ~2.5 + panel + probes
  ~1. No new baseline legs needed: control and all anchors are
  banked.
- **Boundaries**: step-1000 drift-guard read (provisional); step-3000
  endpoint → sim100 + panel + paired reads + verdict post through the
  frozen grid.
- **In-run instrument**: babysit registry entry at launch; first poll
  checks GPU util/rate + `free -g` + **`df -h /`** (disk added to the
  standing first-poll checks after the incident).
- **Checkpoint policy**: saves under
  `~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm_onerig`;
  endpoint banks to `fontaine-checkpoints` same-session if any gated
  read makes it load-bearing (a grasping two-dataset mixed checkpoint
  certainly is), weights-only + logs, standing HTML report.
- **Seed policy**: seed 0 throughout (comparability with control,
  convicted cell, and probe).

*Objection/decision path: this post is a DRAFT pending the owner's
isolation call (the registered carve-out). On a GO the cell launches
at the next GPU-free boundary with the pre-reg frozen as written; any
owner edit lands as a registered amendment first.*
