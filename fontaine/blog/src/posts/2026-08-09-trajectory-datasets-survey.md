# Survey: trajectory datasets we could train on (SO-101 first, then everything else)

*2026-08-09. Owner request 19:58Z: "investigate what additional
trajectory datasets we could train on, ideally for so101, but also
look more generally — links, statistics, brief descriptions." This
post is that survey. Compiled from a four-track web sweep (LeRobot
ecosystem / cross-embodiment corpora / human-collected UMI-family /
sim + 2026 releases), every dataset link fetched and checked, stats
from the source cards or papers; anything we could not verify is
marked. Written against what we already have, so every entry answers
the only question that matters: what would this add?*

## What we already train on (the baseline)

`mcobzarenco/community_curated_v0` (HF hub, built 2026-08-02): **981
datasets · 52,507 episodes · 24.8M frames ≈ 229 h** of 6-DoF, 30 fps
SO-100/SO-101 teleoperation, curated out of a 1,242-dataset /
36.9M-frame crawl of the LeRobot community hub (mechanical episode
filters + a full opus-5 judge sweep: per-frame progress/holding/
visibility annotations, subgoal rows, camera-kind tags — see
`docs/data-curation.md`). Plus the rig datasets
(`so101_pick_place_{clean,v2}`) at dense judge supervision.

Scope decisions that shaped it, relevant to this survey: the crawl
kept only 6-dim action+state at 30 fps; bimanual SO-100 (12-dim),
other-fps recordings, and non-SO embodiments were dropped at step 1.
So "more data" comes in four distinct flavors, each with different
costs:

1. **More of the same** — SO-100/SO-101 data uploaded since the
   crawl, or admitted by relaxing a scope filter. Zero mapping cost.
2. **Cross-embodiment robot data** — other arms, needing action-space
   mapping and probably an embodiment tag.
3. **Human-collected (UMI-family / egocentric)** — gripper-pose
   trajectories with no robot in the loop; retargeting required.
4. **Simulation** — unlimited volume, sim-to-real gap in exchange.

---

## Track 1 — the SO-100/SO-101 pool: 4× our hours are already sitting on the hub

This track came back with the single most actionable number of the
survey. A live hub-API sweep (2026-08-09, all 6,060 repos tagged
`so100`/`so101`; 5,872 readable, per-repo `meta/info.json` fetched)
finds:

| slice | datasets | episodes | frames | hours |
|---|---|---|---|---|
| **our corpus (for reference)** | **981** | **52,507** | **24.8M** | **229** |
| SO-family total on the hub (real+sim) | 5,684 | 264,338 | 122.7M | 1,248 |
| exactly in-scope (6-dim action @ 30 fps) | 4,572 | 204,071 | 92.4M | **855** |
| re-scope candidate: 6-dim, non-30 fps | 559 | 41,100 | 17.7M | 260 |
| re-scope candidate: 12-dim bimanual SO | 387 | 10,745 | 8.5M | 95 |
| new since 2026-01 (in-scope subset) | 484 | 75,451 | 32.7M | **303** |
| new since 2026-05 (in-scope subset) | 286 | 39,493 | 22.0M | 204 |

So the pool matching our *exact* scope holds ~4× our corpus's hours,
and ~300 in-scope hours are new in 2026 — i.e. postdate the
`community_dataset_v3` crawl family our corpus was built from (HF's
curated line ended there; no "v4" crawl exists, so everything since
December 2025 is uncurated upside). One big caveat the sweep
surfaced: **a sizable minority of the new 2026 volume is
sim-generated** (IsaacLab/MuJoCo runs uploaded in LeRobot format —
e.g. one MuJoCo pick-cube repo alone is 11k episodes), so any
re-crawl now needs a real-vs-sim filter that the original scope
rules never needed.

Named finds worth individual mention:

- **[MolmoAct2-SO100_101](https://huggingface.co/datasets/allenai/MolmoAct2-SO100_101-Dataset)**
  (AllenAI, [paper](https://arxiv.org/abs/2605.02881)) — AI2's own
  curation of the same public pool: 1,220 source repos / 38,059
  episodes / ~184 h (Apache-2.0), plus **re-annotated language
  instructions** published as a parquet manifest. This is the same
  exercise we did, done independently at a slightly wider net —
  diffing their source list against our 981 and reusing their
  instruction relabels is nearly free.
- **[so101_bench_real_2](https://huggingface.co/datasets/5hadytru/so101_bench_real_2)**
  — 4,367 episodes / ~23 h real SO-101 teleop @ 30 fps, v3.0, 2
  cameras — the biggest single post-crawl real set (license
  untagged, needs a check).
- **[so101_coffee_all](https://huggingface.co/datasets/aailabkaist/so101_coffee_all)**
  (KAIST) — 2,564 episodes of long-horizon coffee-making with 12
  per-subtask instructions — strong language-conditioning signal.
- **[armnetbench_v01](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101)**
  — 2,499 episodes / ~16 h from an SO-101 "arm farm", 3 cameras —
  but 20 fps (re-scope or resample) and AV1 video.
- **[Elvinky bi-so101 insert-screw family](https://huggingface.co/datasets/Elvinky/bi-so101-insert-screw-562ep)**
  — ~32 h of dual-SO-101 precision insertion, the flagship of the
  95-h bimanual re-scope pool.
- Labs/vendors: NVIDIA has **no** SO-101 dataset (their contribution
  is tooling — the Isaac sim-to-real SO-101 course and GR00T
  integration); Seeed/TheRobotStudio/WowRobo publish hardware, not
  data; the one lab release is AI2's above. A commercial vendor
  teaser exists (UniDataPro, CC-BY-NC-ND — unusable).

Format news: 533 SO-family datasets (571 h) are already
LeRobotDataset v3.0 — including nearly every large 2026 release —
so new data increasingly needs no conversion at all.

## Track 2 — cross-embodiment robot corpora

The open pool of real-robot manipulation data is roughly **2.5M+
trajectories / ~5,000+ h**, of which the single-arm,
permissively-licensed, *already-LeRobot-format* subset is about **1M
episodes / ~1,000+ h** — 5–20× our corpus with no format engineering
(LeRobot ships a one-command v2.x→v3.0
[porting script](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)).
The recurring tax is never format, it's **action space**: most of
these arms are 7-DoF with end-effector-delta actions, vs our 6-DoF
joint-position SO-101 — so any co-training needs either an IK-side
remapping or an embodiment-tagged action head, plus fps resampling
(most sources run 3–15 Hz vs our 30).

| corpus | scale | embodiment | fps | format | license |
|---|---|---|---|---|---|
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | 1M+ trajs, 60 datasets | 22 embodiments | 3–30 Hz | RLDS (+[LeRobot ports](https://huggingface.co/collections/IPEC-COMMUNITY/openx-lerobot-67c29b2ee5911f17dbea635e)) | CC-BY 4.0 agg. |
| [DROID](https://droid-dataset.github.io/) | 76k demos / 350 h, 564 scenes | Franka 7-DoF | 15 Hz | **native LeRobot v3** ([droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1), 358 GB) | Apache-2.0 |
| [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) | 60k trajs, 24 envs | **WidowX 250, 6-DoF hobby-class** | 5 Hz | [LeRobot v2.0](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot), **21.7 GB** | CC-BY 4.0 |
| [RH20T](https://rh20t.github.io/) | ~110k robot seqs + 110k human videos | 4 arms × 4 grippers, F/T-rich | 10 Hz | [LeRobot v2.1](https://huggingface.co/datasets/InternRobotics/RoboInter-Data) (82.9k eps) | CC-BY-SA / part **NC** |
| [AgiBot World Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | 1M+ trajs / **2,976 h**, 217 tasks | Genie-1 mobile dual-arm | ~30 fps | H5+MP4 (+official LeRobot scripts); [2026 corpus](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) native v2.1 | CC-BY-**NC**-SA, gated |
| [RoboMIND](https://x-humanoid-robomind.github.io/) v1 / [2.0](https://arxiv.org/abs/2512.24653) | 107k → 310k+ trajs, 739 tasks | Franka, UR5e, humanoid, dual-arm | ~30 Hz | HDF5, **no LeRobot port** | CC-BY 4.0 (gated) |
| [FMB](https://functional-manipulation-benchmark.github.io/) | 22.5k demos (insertion/assembly) | Franka | — | [LeRobot](https://huggingface.co/datasets/IPEC-COMMUNITY/fmb_dataset_lerobot) | CC-BY (unverified) |
| [Galaxea G0](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) (2025) | 500+ h open-world | R1-Lite mobile dual-arm | 15 fps | **native LeRobot v2.1**, 2.87 TB | CC-BY-NC-SA |
| [GR00T-X-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | 273k trajs (240k humanoid, 72k arm-kitchen) | sim Franka/GR1/G1 | — | LeRobot-schema, 1.9 TB | CC-BY 4.0 |

The three worth actual engineering time, ranked:

1. **BridgeData V2** — the only large corpus on a *6-DoF hobby-class
   arm* (WidowX 250 — the nearest public cousin of the SO-101), doing
   tabletop skills nearly isomorphic to our community data, CC-BY,
   already LeRobot, and only 21.7 GB. The cheapest possible
   co-training experiment with the best embodiment-transfer prior.
   Costs: 5→30 fps handling + EE-delta→joint mapping.
2. **DROID** — native LeRobot v3, streamable, Apache-2.0; carries
   *both* joint and cartesian action streams (we pick the mapping),
   and its 564 in-the-wild scenes are exactly the visual diversity
   our tabletop corpus lacks.
3. **OXE single-arm subsets** via the IPEC-COMMUNITY LeRobot
   collection — the "magic soup" route: standardized 7-D EE actions,
   uniform layout, mixture-weighted at whatever scale the trunk can
   absorb.

Runner-up: RH20T's RoboInter port (force-rich, densely annotated,
mind the part-NC license). RoboMIND is the best corpus with *no*
existing LeRobot conversion — its ~78k single-arm joint-space
trajectories would justify writing one. AgiBot World is the raw
scale champion but dual-arm, gated, NC, and 48 TB.

## Track 3 — human-collected data: the UMI family and egocentric video

The premise: a human moving a handheld gripper (or just their hand)
generates trajectories orders of magnitude cheaper than teleop. The
catch for us is always the same pair of taxes: SE(3) end-effector
actions must map into our 6-motor joint space (and the SO-101's
5-DoF wrist **cannot realize arbitrary orientations** — every
dataset below needs an orientation-feasibility filter), and the
wrist-fisheye / head-camera viewpoints are a real visual domain gap.

| dataset | scale | action labels | format | license |
|---|---|---|---|---|
| [FastUMI-100K](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot) | 100k+ trajs / 54 tasks (~est. 1,000 h) | EE pose + gripper width, 20 fps | **LeRobot v2.1**, 1.4 TB | Apache-2.0 |
| [UMI cup-in-the-wild](https://umi-gripper.github.io/) | ~1,400 demos / 30 envs | SE(3) EE (GoPro SLAM) + width | Zarr 18 GB (+[LeRobot mirror](https://huggingface.co/datasets/lerobot/umi_cup_in_the_wild)) | MIT/Apache |
| [RDT-2 UMI-10k](https://rdt-robotics.github.io/rdt2/) | **10,000+ h**, 100+ scenes | EE + width (bimanual) | — | **NOT RELEASED** ([issue #25](https://github.com/thu-ml/RDT2/issues/25) open) |
| [EgoDex](https://github.com/apple/ml-egodex) (Apple) | **829 h / 338k trajs**, 194 tasks | SE(3) ×68 joints (Vision Pro) | HDF5+MP4, 1.7 TB | CC-BY-**NC-ND** |
| [EgoVerse](https://arxiv.org/abs/2604.07607) (2026) | **1,362 h / 80k eps**, 1,965 tasks | MANO + wrist EE pose + head pose | Zarr + S3 | MIT code; data TBC |
| [PH2D](https://huggingface.co/datasets/RogerQi/PH2D) | ~27k demos / ~3M frames | wrist + finger poses (Vision Pro) | HDF5, **16 GB** | **MIT** |
| [Open-AoE](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h) | 2,000 h target; **~100 h live** | MANO + SLAM, robot-format exports | MP4+NPZ | CC-BY-ish |
| [Ego4D](https://ego4d-data.org/) / [EPIC-KITCHENS](https://epic-kitchens.github.io/) | 3,670 h / 100 h | none (narrations, segments) | video | gated / NC |

What the field has actually measured: the original UMI policy ran
zero-shot on both UR5e and Franka (same handheld data, two arms);
H-RDT pretrained on all of EgoDex and got **+40.5% real-robot
success over from-scratch** after cross-embodiment fine-tuning;
EgoMimic found 1 h of smart-glasses human data worth more than 1 h
of extra teleop; and EgoVerse's multi-lab study adds the sobering
qualifier — human data helps **iff it aligns with the robot's
tasks**. Ego4D/EPIC-class video carries no executable actions at
all; at our compute scale it's someone else's pretraining substrate,
not our training data.

Ranked for us: **FastUMI-100K first** (the only UMI-scale corpus
already in LeRobot format with permissive license — the pipeline
tax is fps resampling + EE→joint IK + the wrist-camera gap), with
the 18 GB UMI cup set as the de-risking pilot before committing to
1.4 TB. PH2D (16 GB, MIT) is the cheap test of whether hand-pose
co-training moves our panel at all. EgoDex only via released
weights (the ND license is a trap for redistributable checkpoints).
RDT-2's 10k hours would change the picture entirely — watch the
release issue.

## Track 4 — simulation and synthetic data

The surprise here is how mature the SO-101-specific sim tooling got
in 2026 — the bottleneck is no longer "can sim emit our format" but
episode quality:

- **[so101-nexus](https://github.com/johnsutor/so101-nexus)** —
  MuJoCo SO-101 envs (6 tasks, wrist+overhead cameras) that emit
  **LeRobot v3 natively**, Apache-2.0, with a GPU-parallel backend.
  The most direct generate-our-own route.
- **[leIsaac](https://github.com/LightwheelAI/leisaac)** (Isaac Lab)
  — sim-teleop driven by a real SO-101 leader arm, records LeRobot
  directly; combine with Isaac Lab Mimic (MimicGen-style
  multiplication) to scale seed demos superlinearly. NVIDIA's
  [sim-to-real SO-101 course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/datasets-and-models.html)
  publishes small worked datasets proving the loop end-to-end,
  including **Cosmos domain-randomized augmentation** (70 augmented
  episodes from 75 sim seeds) co-training with real data.
- **Ready-made SO-101 sim data on the hub**: the
  `jadechoghari/svla_so101-sim_*` family (~1,800 episodes verified
  in one of five shards, 6-dim, LeRobot), Cache-SCA's IsaacLab sets
  (~33 h), `gpudad/so101_pick_cube` (11k episodes MuJoCo),
  `szk1ck/so101-ycb-pickplace` (5k episodes, randomized YCB
  objects). Drop-in format-wise; each needs the same QC pass as
  community data — and these are exactly the repos contaminating
  the Track-1 "new hours" numbers.
- **Classic suites** (LIBERO — already LeRobot v3 at 1.9 GB;
  RoboCasa/MimicGen ~100k+ trajs; ManiSkill) are all
  wrong-embodiment (Franka EEF); useful only as cross-embodiment
  pretraining mass, and nobody has published an X→SO-101 retarget.
- **[BEHAVIOR 2026 demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)**
  — 20,000 teleop demos / 3.27 TB, MIT, **the largest LeRobot-v3
  sim corpus in existence** — but whole-body bimanual.

## The 2026 releases that don't fit the boxes

[ABC-130K](https://huggingface.co/datasets/XDOF/ABC-130k) (Amazon:
134,806 bimanual episodes / 3,553 h, MCAP format), the
[AIRoA ICRA-2026 competition corpus](https://icra2026vlapipeline.github.io/)
(~10,000 h real Toyota HSR, LeRobot format, wrong embodiment),
[AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)
(five-phase G2 release, native LeRobot v2.1), and the ecosystem
datum that the LeRobot hub passed **58,000 datasets** in May 2026.
Scale exists; almost none of it is our embodiment.

---

## What I'd actually do, in order

1. **Re-crawl + curate the SO pool delta** (CPU + judge budget, no
   GPU): ~300 in-scope hours are new since our crawl, and the
   existing curation pipeline (mechanical filters → judge sweep →
   union build) is built and documented. New requirement: a
   real-vs-sim provenance filter. Diff **MolmoAct2's 1,220-repo
   source list** against our 981 as the first step — it's a free
   second opinion on our own scope decisions, and their instruction
   relabels port to episodes we already train on. This is the
   highest-value, lowest-risk item on the board: same embodiment,
   same format, same pipeline, roughly doubling usable hours.
2. **BridgeData V2 co-training pilot** (21.7 GB, CC-BY, 6-DoF
   WidowX): the cheapest cross-embodiment experiment with the best
   morphology prior. One mixture arm, pre-registered, panel-judged.
3. **UMI-family pilot**: UMI cup (18 GB) to build the EE→joint +
   orientation-feasibility machinery, then FastUMI-100K (1.4 TB,
   Apache, LeRobot v2.1) if the pilot moves anything.
4. **Sim only as augmentation**, via so101-nexus/leIsaac seeded
   from our rig tasks — not as bulk hours; the field's evidence
   (and our own curation instincts) say uncurated sim episodes are
   how you poison a corpus silently.
5. **Watch list**: RDT-2's UMI-10k release issue, Open-AoE's
   staged tranches, AgiBotWorld2026 phases, RoboMIND's missing
   LeRobot port (a conversion we could contribute).

*Verification note: every URL above was fetched live during the
survey; numbers not confirmable from a fetched card/paper are
flagged in the per-track working notes. The full agent reports
(with per-dataset stats lines and ~40 more datasets that didn't
make the cut) are archived in the session records.*

