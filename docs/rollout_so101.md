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
   files are needed (`optimizer.pt` is not):

   ```sh
   scp -r ubuntu@<box>:flow-matching/outputs/train/bijou_ft_marius_2k/step_002000 /tmp/bijou_ft/
   ```

   Fine-tuned checkpoints for the owner's rig: `bijou_ft_marius_2k/step_{000250..002000}`
   (2000 steps ≈ 38 epochs over the 7 recorded episodes; later checkpoints
   are more adapted but more overfit — try 2000 first, walk back if
   behavior looks memorized/brittle).
2. Rig normalization stats, one of:
   - `--stats-repo-id marius/so101_pick_place_clean` — looked up in the
     checkpoint's `per_dataset_normalization` table (present in any
     checkpoint whose training data included the rig).
   - `--stats-dataset <path>` — read from a local LeRobot dataset dir
     (`meta/stats.json`).
3. Follower arm calibrated under a lerobot robot id (`--robot-id`).

## The command

```sh
uv run python -m bijou.rollout \
    --checkpoint /tmp/bijou_ft/step_002000 \
    --stats-repo-id marius/so101_pick_place_clean \
    --port /dev/ttyACM0 --robot-id <follower_id> \
    --camera front=/dev/video0 --camera wrist=/dev/video2 \
    --task "<instruction as recorded in the dataset>" \
    --expert-dtype bfloat16 \
    --max-relative-target 20 \
    --duration 30
```

Add `--check` first: loads the checkpoint, resolves stats, prints the plan,
exits without touching the robot.

## Things that matter (first-run checklist)

- **Camera names are positional prompt slots.** Prompt order = SORTED
  camera keys, so `--camera` names must sort like the training dataset's
  keys. The owner's dataset uses `front`/`wrist` (in that sorted order);
  swapping names silently swaps views in the prompt.
- **`--max-relative-target 20`** engages lerobot's per-tick joint-motion
  clamp. Keep it on until the policy is trusted; it damps large motions.
- **`--expert-dtype bfloat16`** on small inference GPUs (8 GB laptop:
  bf16 backbone ~5 GB + bf16 expert ~0.8 GB fits; fp32 expert is tight).
- **Replan latency**: ~200 ms on H100, ~0.5–1.5 s on a laptop GPU → visible
  hold at each replan boundary with `--execute-horizon 40`. Levers:
  horizon 50 (full chunk, fewer holds), `--sample-steps 3`, smaller
  `--max-soft-tokens` is NOT available at rollout (baked into the
  checkpoint's prompt budget).
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
- Laptop end-to-end latency (measured only on H100).
- Whether `--fps 30` execution matches the 30 fps recording cadence well
  enough (drift between commanded tick and camera exposure).
