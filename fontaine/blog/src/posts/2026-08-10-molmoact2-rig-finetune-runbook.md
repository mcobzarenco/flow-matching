# Runbook: fine-tuning MolmoAct2 on the rig datasets + local rollouts

*2026-08-10, from the owner's 15:19Z question ("How could I — out of
band — fine-tune molmo2act on my rig datasets and then do local
rollouts? Happy to use their code.") and 15:24Z GO. Companion to the
[pre-registration](2026-08-10-prereg-molmoact2-rig-finetune.md); the
fine-tune itself runs on our local H100 under that pre-reg. This page
is the runnable, pinned-commands record — including the rollout-server
half that runs on the rig machine, which only the owner can execute.*

**Plain words**: the released MolmoAct2 SO-100/101 policy doesn't know
the owner's robot. Today's eval measured exactly that: on scenes from
its own training mixture it is competent; on the rig's kind of data it
predicts a canned "average pose" instead of reading the arm. The fix
is standard: take their released model and continue training it
briefly on the 57 episodes recorded on the actual rig, using their own
training code, then serve the result to the robot over a small HTTP
server with safety rails so a bad prediction can never slam the arm.

## 0. What was measured before anything ran (preflight, 16:2x–16:3xZ)

- Both rig repos are **LeRobot codebase v3.0** (`meta/info.json`,
  read not assumed): `so101_pick_place_clean` 7 eps / 3,399 frames,
  `so101_pick_place_v2` 50 eps / 32,679 frames, 30 fps, cameras
  `front` + `wrist`, 6-dim state/action, robot `so_follower`.
- **No sign mirrors between their native space and rig v3.0**: the
  released checkpoint, run zero-shot on 240 rig frames, shows
  positive per-joint motion correlation on all six joints
  (+0.12…+0.45). The v2.1/v3.0 hazard the community repo warns about
  does not include mirrored joints on this data.
- **There IS a per-joint affine gap, and it matters**: their state
  normalization for joint1 (shoulder) spans [43.7, 185.3]; the rig's
  shoulder lives in [−103, +67]. 97% of rig frames saturate their
  state encoding, and the model then predicts a near-constant
  shoulder posture (pred std 2.0 vs truth std 44.8, err~truth corr
  −0.999). This is why zero-shot MAE on rig frames is 28.95 vs
  state-copy's 9.08 — and why **rig-only normalization stats are
  load-bearing at train AND rollout time**.
- Full table: `reports/analysis__molmoact2_rig_preflight.json`
  (+ per-frame npz alongside it); script
  `fontaine/scripts/molmoact2_rig_preflight.py`.

**The convention decision that falls out**: stay in v3.0 end-to-end.
No data conversion; per-joint rig q01/q99 absorbs the affine gap; the
rollout server must run with the conversion OFF (offsets 0, signs +1
— the community `inference.py` defaults exist for running the
*released* model on a 0.5.1-calibrated rig, which is not our path once
the model is fine-tuned on v3.0 data). The requirement that remains:
**the rig's calibration must be the same one the datasets were
recorded under**. If the arm is recalibrated, re-record or re-check
before rollouts.

## 1. One-time setup (done on our box; repeat on any new machine)

```bash
git clone https://github.com/allenai/molmoact2.git ~/molmoact2
cd ~/molmoact2
uv venv .venv --python 3.12
. .venv/bin/activate
uv pip install --index-strategy unsafe-best-match \
    -e "./experiments[train]" -e ./lerobot \
    debugpy "torchcodec>=0.10,<0.11"
```

Three local patches, committed on branch `fontaine-so101-rig`
(commit `89f6204` in `~/molmoact2`):

1. **Mixture** — `experiments/launch_scripts/data_mixtures.py`: a
   `so101_rig` builder registering the two rig repos under their
   pretrain tag `so100_so101_molmoact2` (same setup/control prompt
   text the trunk was trained with; horizon 30 = 1.0 s;
   `normalize_gripper=True`; cameras pinned to front+wrist). Because
   their stats collector aggregates per tag over only the mixture's
   repos, **rig-only q01/q99 falls out automatically** — verified
   equal to the count-weighted combination of the repos'
   `meta/stats.json` and different from the released SO-100 stats.
2. **Wrapper** — `olmo/data/lerobot_wrapper.py`: add `language` to
   the ignored-feature-dtypes set (the rig repos carry our mainline's
   `language_persistent`/`language_events` columns, which are not a
   pyarrow type).
3. **Vendored lerobot** — `lerobot/src/lerobot/datasets/utils.py`:
   when an explicit schema is passed, read only its columns from
   parquet (the rig repos' extra annotation/language columns
   otherwise fail HF datasets' schema cast). Action/state/video
   paths untouched.

Env every run needs: `LEROBOT_DATA_ROOT=/home/ubuntu/datasets`
(repos resolve as `$LEROBOT_DATA_ROOT/<repo_id>`),
`MOLMO_DATA_DIR=~/molmoact2/molmo_data` (unused scratch, but their
loader asserts it), `PYTHONPATH=$EXP:$EXP/../lerobot/src`.

## 2. The fine-tune (runs here, under the pre-reg)

Their README's **action-expert-only** recipe — VLM/ViT/connector/
embeddings/LM-head frozen, the 620M flow-matching action expert
trains (577M trainable params measured at smoke). Warm start
`allenai/MolmoAct2-SO100_101` (their recommendation for SO-101
embodiments). Launcher:
`fontaine/scripts/launch_local_molmoact2_rig_ft.sh` — 2,000 steps at
global batch 64 (≈3.5 epochs), AE LR 5e-5, warmup 200, bf16, saves
every 500 kept, unit `fontaine-molmoact2-rig-ft`, log
`~/logs/molmoact2_rig_ft.log`, gate ≤ 12 GPU-h.

Smoke (20 steps, batch 1) is green: rc=0, finite falling
`action_flow_loss` (0.077 at step 20), checkpoint writes.

Offline read after training (contaminated-by-construction, labeled):
matched 1.0 s-window MAE on the same 240 preflight frames per saved
rung, vs the banked anchors — **zero-shot 28.95 / state-copy 9.08**.
A rung that fails to beat state-copy on its own training frames
falsifies the rung. Plus the **step-0 continuity oracle**: predicted
step 0 must land near the current state on every rung (loud fail =
convention/normalization pathology).

## 3. Serving the fine-tuned checkpoint (rig machine)

Their trainer saves olmo-format checkpoints
(`~/molmoact2/experiments/checkpoints/finetune/fontaine_so101_rig_ae_r1/stepNNNN`).
Convert the chosen rung to a HF checkpoint:

```bash
cd ~/molmoact2/experiments
../.venv/bin/python -m olmo.hf_model.convert_molmoact2_to_hf \
    checkpoints/finetune/fontaine_so101_rig_ae_r1/step2000 \
    ~/checkpoints/molmoact2-so101-rig-r1-hf
```

The converted dir carries `norm_stats.json` **with the rig-only
stats** under tag `so100_so101_molmoact2` — the server must load
*this* dir, never the released repo id.

Server: adapt `examples/droid/host_server_droid.py` (FastAPI `/act`,
json_numpy wire format). The DROID server is the template because it
already contains the two runtime patches the HF checkpoint needs
(bf16 trajectory dtype + fp32 cast before `.numpy()` — the same
patches our panel predictor ported). The SO-101 deltas:

- `REPO_ID = "<local converted dir>"`, `NORM_TAG =
  "so100_so101_molmoact2"`, `DEFAULT_NUM_STEPS = 10` (Euler steps).
- State is **(6,)** not (8,) — relax the shape check; cameras are
  `front` (external) + `wrist`, image order = the dataset's feature
  order (front, then wrist), 480×640 RGB.
- Response actions are **(30, 6) absolute joint targets in v3.0
  units** — the same space `observation.state` is reported in on the
  rig. **No joint-offset/sign conversion anywhere** (see §0).

## 4. The client loop + safety rails (rig machine, owner-run)

Client skeleton (lerobot 0.5.x on the rig): read
`robot.get_observation()` → POST `{front, wrist, instruction,
state}` to `/act` → receive 30-step chunk → execute steps at 30 Hz,
**replan every 15–30 steps (0.5–1.0 s)** by sending a fresh
observation mid-chunk and splicing.

Rails, in order, before the arm moves at all:

1. **Calibration identity check**: confirm the rig's current
   calibration file is the one the datasets were recorded under
   (same lerobot version, no recalibration since). If in doubt:
   compare live `observation.state` against the recorded state range
   per joint (rig q01/q99 in this page's §0) — every joint should sit
   inside its recorded band at rest poses.
2. **No-execute dry run**: run the full client loop with motor
   commands printed, not sent. Watch predicted step-0 vs current
   state — they should agree to a few units on every joint (that is
   the step-0 continuity oracle, live). A constant offset or a
   mirrored joint aborts here, harmlessly.
3. **Command clamp**: wrap the send with per-joint clamps to the rig
   q01/q99 band (±10% margin) and a per-step delta clamp (e.g. ≤ 8
   units/step at 30 Hz to start). "Slam into the table" becomes
   structurally impossible; loosen only after clean rollouts.
4. First live runs at reduced speed/stride (execute every other
   step), gripper disabled until arm behavior is verified.

## 5. What happens next

*(Updated 2026-08-10 20:4xZ with the measured endpoint.)*

Run complete: launched 17:48:18Z, rc=0 20:27:44Z, ~2.7 of the 12
GPU-h gate. **Pre-reg PASS at every gate** — matched-window MAE on
the 240 anchor rows: zero-shot 28.95 → 6.76@500 → 4.66@1000 →
3.59@1500 → **3.23@2000** (state-copy anchor 9.08 beaten from rung
500 on; all 6 motion corrs positive, weakest +0.89; step-0 offsets
≤ 0.63). Full numbers + charts:
[results post](2026-08-10-molmoact2-rig-ft-results.md) ·
[HTML report](https://mcobzarenco-fontaine-reports.static.hf.space/eval__fontaine_so101_rig_ae_r1__anchor_rungs.html).

The serve dir for §3 is
`~/checkpoints/molmoact2-so101-rig-r1-step2000-hf` (rig norm_stats
under tag `so100_so101_molmoact2` verified inside); the weights
delta is on
[fontaine-checkpoints](https://huggingface.co/mcobzarenco/fontaine-checkpoints/tree/main/molmoact2_so101_rig_r1_step2000).
Rung 2 (LoRA or VLM unfreeze at their README settings) stays parked:
rung 1 did not plateau above the anchors — nothing triggers it. The
rung reads are train-frame sanity; the §3–4 rollout path (no-execute
dry-run gate + command clamp) is the real eval whenever the rig is
ready.
