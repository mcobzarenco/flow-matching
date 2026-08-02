# Rolling out Bijou on a physical SO-101

Runbook for `python -m bijou.rollout` (added `15da227`). Status: verified
hardware-free (`--check` + a synthetic robot observation through the full
predict path); **not yet run on a physical arm**.

## What it is

Closed-loop rollout using lerobot's **robot layer only** (`SOFollower`,
OpenCV cameras, calibration, safety clamps) with inference reusing
`bijou.eval.policies.BijouPolicy` verbatim — the offline-scored path is
byte-for-byte what drives the arm. Loop: observe → predict a 50-action
chunk → execute `--execute-horizon` actions at `--fps` → replan from a
fresh observation.

We deliberately did NOT (yet) implement a lerobot policy plugin
(`PreTrainedPolicy` + processor pipelines + third-party registration via
`register_third_party_plugins`), which would enable
`lerobot-rollout --policy.type=bijou` and their ecosystem features
(eval-episode recording, async/RTC inference). That remains on the roadmap.

## Prerequisites

1. A bijou checkpoint on the machine the robot is plugged into. Only two
   files are needed per checkpoint (`expert.safetensors` +
   `bijou_config.json`; `optimizer.pt` is not). The fine-tunes are public
   on HF:

   ```sh
   uv run hf download mcobzarenco/bijou-checkpoints \
       --include 'bijou_ft_marius_2k/step_001000/*' \
       --local-dir outputs/train
   ```

   (or `scp -r` from the training box's `outputs/train/`). Fine-tuned
   checkpoints for the owner's rig: `bijou_ft_marius_2k/step_{000250..002000}`
   — 2000 steps ≈ 38 epochs over the 7 recorded episodes, so later
   checkpoints are more adapted but more overfit. step_001000 is a sensible
   first try; walk toward 000250 if behavior looks memorized/brittle, toward
   002000 if it looks undertrained.
2. Rig normalization stats, one of:
   - `--stats-repo-id mcobzarenco/so101_pick_place_clean` — looked up in the
     checkpoint's `per_dataset_normalization` table (present in any
     checkpoint whose training data included the rig). NOTE: the rig
     datasets were renamed `marius/*` → `mcobzarenco/*` (2026-08-02,
     canonical hub ids); checkpoints trained BEFORE that keep the old
     key in their table — pass `--stats-repo-id marius/so101_pick_place_clean`
     for those, matching what the checkpoint recorded, not today's name.
   - `--stats-dataset <path>` — read from a local LeRobot dataset dir
     (`meta/stats.json`).
3. Follower arm calibrated under a lerobot robot id (`--robot-id`).

## The command (owner's rig, copy-paste)

Dry run first — loads the checkpoint, resolves stats, prints the plan,
exits without touching the robot:

```sh
uv run python -m bijou.rollout \
    --checkpoint outputs/train/bijou_ft_marius_2k/step_001000 \
    --stats-repo-id mcobzarenco/so101_pick_place_clean \
    --port /dev/ttyACM0 --robot-id follower0 \
    --camera front=/dev/video6 --camera wrist=/dev/video4 \
    --task "Pick up the toy boat and place it on the wooden disk." \
    --expert-dtype bfloat16 \
    --max-relative-target 20 \
    --duration 30 \
    --check
```

Then drop `--check` to drive the arm. The task string is the dataset's
recorded instruction verbatim (`meta/tasks.parquet`); camera device paths
are the owner's current enumeration (wrist = `/dev/video4`, front =
`/dev/video6`) — re-check with `uv run lerobot-find-cameras opencv` after
replugging, device indices move.

## Things that matter (first-run checklist)

- **Camera names are positional prompt slots.** Prompt order = SORTED
  camera keys, so `--camera` names must sort like the training dataset's
  keys. The owner's dataset uses `front`/`wrist` (in that sorted order);
  swapping names silently swaps views in the prompt.
- **`--max-relative-target 20`** engages lerobot's per-tick joint-motion
  clamp. Keep it on until the policy is trusted; it damps large motions.
- **`--expert-dtype bfloat16` fits the 8 GB laptop, measured**: on the
  owner's RTX 3000 Ada (8 GiB), loading step_001000 puts 4.77 GiB of
  weights on the GPU and a full predict peaks at 6.29 GiB reserved
  (outputs/probe_rollout_vram.py — synthetic 2-camera observation through
  the real predict path). fp32 expert would add ~0.8 GiB weights plus
  larger activations — possible but with no headroom.
- **Replan latency, measured on the same laptop**: ~250 ms warm (Heun-5,
  2 cameras; the first replan pays ~2 s of CUDA warmup). At
  `--execute-horizon 40` @ 30 fps that is a ~250 ms hold every 1.33 s of
  motion. Levers: horizon 50 (full chunk, fewer holds), `--sample-steps 3`.
  A smaller `--max-soft-tokens` is NOT available at rollout (the prompt
  budget is baked into the checkpoint).
- **Task string**: use the instruction wording the fine-tune data carried
  (`task` field of the episodes) — the model is conditioned on it.
- Actions are commanded in the dataset's raw units (degrees); the
  `use_degrees=True` default of `SOFollowerRobotConfig` matches the
  community-era datasets. If a rig was calibrated with the newer
  [-100,100] convention, stats and robot units must agree — check
  `--check` output's state-stats line against a live joint readout.
- Ctrl-C stops cleanly (disconnects, torque released per lerobot config).

## Known unknowns before first physical run

- Closed-loop stability of chunk handoffs (open-loop metrics don't test
  replan boundaries).
- Whether `--fps 30` execution matches the 30 fps recording cadence well
  enough (drift between commanded tick and camera exposure).
- Camera exposure/white balance vs the recordings (the fine-tune saw one
  lighting setup; big shifts degrade the vision conditioning).
