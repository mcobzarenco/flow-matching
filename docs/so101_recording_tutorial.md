# Recording SO-101 datasets locally with LeRobot (no HF Hub required)

Everything below runs from this repo through **uv** — lerobot 0.6.0 is installed
in `.venv` with the `feetech` (SO-101 motor bus) and `smolvla` extras, plus the
`rerun` viewer. Every LeRobot utility is a console script, so the pattern is
always:

```bash
cd ~/w/flow-matching
uv run lerobot-<something> [args]
```

Available scripts (from `uv run lerobot-info`): `lerobot-find-port`,
`lerobot-setup-motors`, `lerobot-calibrate`, `lerobot-find-cameras`,
`lerobot-teleoperate`, `lerobot-record`, `lerobot-replay`,
`lerobot-dataset-viz`, `lerobot-edit-dataset`, `lerobot-train`,
`lerobot-eval`, `lerobot-rollout`, and a few more.

Config flags use draccus-style nested keys (`--robot.port=...`,
`--dataset.push_to_hub=false`). `--help` on any script prints the full tree.

## Where files end up

| What | Default location | Override |
|---|---|---|
| Datasets | `~/.cache/huggingface/lerobot/{repo_id}` | `--dataset.root=...` or `HF_LEROBOT_HOME` |
| Calibration | `~/.cache/huggingface/lerobot/calibration/` | `HF_LEROBOT_CALIBRATION` |
| Training runs | `outputs/train/...` (cwd-relative) | `--output_dir=...` |

This tutorial records into `~/w/my_datasets/<user>/<name>` so datasets sit
next to `community_dataset_v1_v3` and are servable by the same web-visualizer
file server.

**The one flag that keeps everything local is `--dataset.push_to_hub=false`**
(the default is `true`!). Nothing else in the pipeline needs the Hub except
downloading `lerobot/smolvla_base` once for fine-tuning.

---

## 1. One-time hardware setup

### 1.1 Serial permissions (Linux)

The arms enumerate as USB CDC-ACM serial devices (`/dev/ttyACM*`). Give your
user access once:

```bash
sudo usermod -aG dialout $USER   # then log out and back in
```

### 1.2 Find each arm's port

Plug in one controller board at a time, or use the interactive
unplug/replug flow:

```bash
uv run lerobot-find-port
```

It prints ports before/after you re-plug the cable, isolating the device.
Note down which `/dev/ttyACM*` is the **follower** and which is the
**leader**. (Ports can swap between reboots — if teleop ever behaves
strangely, re-check. Udev rules pinning by serial number are a nice upgrade
later.)

### 1.3 Motor IDs (only for self-assembled arms)

Servos ship with identical default IDs; each joint needs a unique one. If
your arms came pre-configured, skip this. Otherwise, with **one motor
connected at a time** (the script prompts you joint by joint):

```bash
uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
uv run lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1
```

### 1.4 Calibrate both arms

```bash
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower0
uv run lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader0
```

Follow the prompts: move the arm to the neutral middle pose, confirm, then
sweep every joint through its full range so min/max get registered.
Calibration is stored per-`id` under
`~/.cache/huggingface/lerobot/calibration/` — **keep using the same
`--robot.id`/`--teleop.id` strings in every later command**, they are the
lookup key. Recalibrate whenever you rebuild an arm or teleop feels offset.

---

## 2. Cameras

Enumerate connected cameras and grab test frames:

```bash
uv run lerobot-find-cameras opencv     # or: realsense
```

This lists each camera's index/path and saves sample images to
`outputs/captured_images/` — open them to identify which physical camera is
which. Prefer stable `/dev/videoN` paths (or `/dev/v4l/by-id/...` symlinks)
over bare indices.

Camera configs are passed inline as a dict. Two cameras (a static scene view
plus a wrist view) is the sweet spot for SO-101 + SmolVLA:

```
--robot.cameras="{
  front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
  wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}
}"
```

Practical notes:

- The key names (`front`, `wrist`) become dataset features
  `observation.images.front` / `observation.images.wrist` — choose meaningful
  names, you'll see them everywhere downstream.
- Two USB cameras can saturate a single USB controller. If frames drop, put
  them on different ports/hubs, lower the resolution, or add
  `fourcc: MJPG` to the camera config.
- 640×480@30 is plenty; SmolVLA resizes to 512×512 internally.

---

## 3. Sanity-check teleoperation

First without cameras (pure motor loop):

```bash
uv run lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower0 \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader0
```

The follower should mirror the leader smoothly at ~60 Hz. Then add cameras
and the live viewer (rerun) to check framing:

```bash
uv run lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader0 \
  --display_data=true
```

Ctrl-C to stop.

---

## 4. Record a dataset (locally)

```bash
uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader0 \
  --display_data=true \
  --dataset.repo_id=mcobzarenco/so101_pick_place \
  --dataset.single_task="Pick up the red cube and place it in the box." \
  --dataset.root=/home/marius/w/my_datasets/mcobzarenco/so101_pick_place \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.fps=30
```

Flag-by-flag:

| Flag | Meaning |
|---|---|
| `--dataset.repo_id` | Naming only when staying local, but keep the `user/name` shape — tools assume it. |
| `--dataset.single_task` | The natural-language instruction. SmolVLA conditions on this string; be specific and keep it consistent for a given behavior. |
| `--dataset.root` | Exact output directory. Omit to use `~/.cache/huggingface/lerobot/{repo_id}`. |
| `--dataset.push_to_hub=false` | **Local-only recording.** |
| `--dataset.episode_time_s` | Max seconds per episode — end early with the right-arrow key. |
| `--dataset.reset_time_s` | Countdown between episodes to reset the scene. |
| `--display_data=true` | Live rerun view of cameras + joint streams while recording. |

### Keyboard controls during recording

| Key | Effect |
|---|---|
| `→` (right arrow) | End the current episode/reset phase early and continue |
| `←` (left arrow) | Discard and re-record the current episode |
| `Esc` | Stop the session, encode videos, finalize the dataset |

### Session workflow

1. Scene reset, hands on the leader arm.
2. Perform the task at a natural pace; hit `→` when done (don't idle — pad
   frames teach the policy to do nothing).
3. During the reset countdown, randomize object positions (vary within the
   distribution you want the policy to handle).
4. Botched a demo? `←` immediately and redo it.
5. `Esc` when you're done for the day.

Add more episodes later with `--resume=true` (same `repo_id`/`root`).

Rules of thumb: 50 episodes is a workable minimum for fine-tuning on a
single task; more + more variation = better. Keep camera poses fixed across
the whole dataset. If the console warns about unstable fps, tune
`--dataset.num_image_writer_threads_per_camera` (default 4) or record with
`--dataset.streaming_encoding=true`.

### What lands on disk

A LeRobot **v3.0** dataset:

```
so101_pick_place/
├── meta/{info.json, stats.json, tasks.parquet, episodes/...}
├── data/chunk-000/file-000.parquet      # states/actions, consolidated
└── videos/observation.images.front/...  # one mp4 stream per camera
```

Exactly the format the rest of this repo's tooling already handles.

---

## 5. Inspect what you recorded

### rerun (frame-accurate, per-episode)

```bash
uv run lerobot-dataset-viz \
  --repo-id mcobzarenco/so101_pick_place \
  --root /home/marius/w/my_datasets/mcobzarenco/so101_pick_place \
  --episode-index 0
```

Opens the rerun viewer with synchronized video, joint positions, and actions
on a shared timeline. Useful extras: `--display-mode foxglove` (scrubbable
playback in Foxglove), `--save 1 --output-dir ...` (write a `.rrd` file),
`--tolerance-s` if it complains about timestamp jitter.

> **`--root` gotcha:** for all lerobot CLIs, `--root` is the *dataset
> directory itself* (the one containing `meta/`), **not** a parent tree of
> datasets. Passing the parent makes the local load fail, and lerobot then
> falls back to the HF Hub — which can surface bizarre errors about the
> *hub* copy of the dataset (e.g. a `BackwardCompatibilityError` about
> v2.1) instead of a simple "wrong path". The web visualizer's file server
> is the opposite: it takes the parent tree.

### Web visualizer (the one from `~/w/lerobot-dataset-visualizer`)

Serve the datasets root and browse at `localhost:3000`:

```bash
cd ~/w/lerobot-dataset-visualizer
bun scripts/serve-local-datasets.ts /home/marius/w/my_datasets 8000
DATASET_URL=http://localhost:8000 bun dev
# → http://localhost:3000/mcobzarenco/so101_pick_place/episode_0
```

Since the recording is v3.0 and `robot_type` is `so101_follower`, the
v3-only tabs (episode-length statistics, 3D URDF replay) light up too.

### Fixing mistakes after the fact

`uv run lerobot-edit-dataset --help` — delete episodes, split/merge datasets,
add/remove features. Deleting the episodes you flagged in the visualizer is
the typical use.

### Replay on the robot (optional sanity check)

Re-executes recorded joint trajectories on the follower — clear the workspace
first:

```bash
uv run lerobot-replay \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower0 \
  --dataset.repo_id=mcobzarenco/so101_pick_place \
  --dataset.root=/home/marius/w/my_datasets/mcobzarenco/so101_pick_place \
  --dataset.episode=0
```

---

## 6. Fine-tune SmolVLA on your dataset

```bash
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=mcobzarenco/so101_pick_place \
  --dataset.root=/home/marius/w/my_datasets/mcobzarenco/so101_pick_place \
  --output_dir=outputs/train/smolvla_so101_pick_place \
  --job_name=smolvla_so101_pick_place \
  --batch_size=4 \
  --steps=20000 \
  --save_freq=5000 \
  --wandb.enable=false
```

- `--policy.path=lerobot/smolvla_base` = start from the pretrained base
  (the only Hub download in this whole flow; cached afterwards).
- `--policy.push_to_hub=false` keeps checkpoints local (again, pushing is
  the default!).
- **8 GB VRAM reality check** (RTX 3000 Ada Laptop): the reference recipe
  uses batch 64 on a big GPU. Start with `--batch_size=4`, add
  `--policy.use_amp=true` if you hit OOM, and expect roughly a day of
  training for 20k steps. Fewer steps (5–10k) often already grasps a simple
  single task — evaluate a mid-run checkpoint before committing to more.
- Resume an interrupted run:
  `uv run lerobot-train --config_path=outputs/train/smolvla_so101_pick_place/checkpoints/last/pretrained_model/train_config.json --resume=true`

Checkpoints land in
`outputs/train/<job>/checkpoints/<step>/pretrained_model/` — each contains
the weights **plus** `policy_preprocessor.json` / `policy_postprocessor.json`
with normalization stats computed from *your* dataset, keyed by *your*
feature names (see the processor deep-dive discussion: this is what makes a
fine-tuned checkpoint self-contained, unlike `smolvla_base`).

---

## 7. Evaluate

### Offline (no robot) — with this repo's `bijou.eval`

```bash
uv run python -m bijou.eval \
  --data /home/marius/w/my_datasets/mcobzarenco/so101_pick_place \
  --smolvla outputs/train/smolvla_so101_pick_place/checkpoints/last/pretrained_model \
  --num-samples 256 --output-json eval_finetuned.json
```

This scores the policy against the state-copy baselines on seeded random
frames (add `--checkpoint` to compare a bijou checkpoint on the same
frames). Because the fine-tuned checkpoint's camera features are named
after your cameras, the rename map resolves to identity, and its saved
stats already match your dataset. Use `--episodes holdout` with the
training run's `--holdout-episodes`/`--split-seed` for honest numbers.

### On the robot — policy-driven rollouts

In lerobot 0.6, running a policy on hardware moved from `lerobot-record` to
`lerobot-rollout` (record refuses a policy and tells you so). It drives the
follower with the policy while recording the outcome as a normal dataset:

```bash
uv run lerobot-rollout \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --policy.path=outputs/train/smolvla_so101_pick_place/checkpoints/last/pretrained_model \
  --dataset.repo_id=mcobzarenco/eval_so101_pick_place \
  --dataset.root=/home/marius/w/my_datasets/mcobzarenco/eval_so101_pick_place \
  --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the red cube and place it in the box." \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=60
```

Keep a hand near the leader arm or the E-stop for the first rollouts.
Success rate over N scene randomizations is the metric that actually matters;
the offline MAE numbers are only a cheap proxy.

---

## 8. Quick reference

| Task | Command |
|---|---|
| Find serial ports | `uv run lerobot-find-port` |
| Assign motor IDs | `uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=...` |
| Calibrate | `uv run lerobot-calibrate --robot.type=so101_follower ...` / `--teleop.type=so101_leader ...` |
| List cameras | `uv run lerobot-find-cameras opencv` |
| Teleop test | `uv run lerobot-teleoperate --robot... --teleop...` |
| Record (local) | `uv run lerobot-record ... --dataset.push_to_hub=false` |
| Visualize | `uv run lerobot-dataset-viz --repo-id ... --root ... --episode-index 0` |
| Edit dataset | `uv run lerobot-edit-dataset --help` |
| Replay episode | `uv run lerobot-replay --robot... --dataset.episode=0` |
| Fine-tune | `uv run lerobot-train --policy.path=lerobot/smolvla_base ... --policy.push_to_hub=false` |
| Offline eval | `uv run python -m bijou.eval --data ... --smolvla <checkpoint>` |
| On-robot eval | `uv run lerobot-rollout --policy.path=<checkpoint> ...` |

## 9. Gotchas

- `push_to_hub` defaults to **true** in both record and train — always pass
  `false` explicitly for local-only work.
- `--robot.id` / `--teleop.id` select the calibration file; changing them
  silently means "uncalibrated arm".
- `/dev/ttyACM*` ordering can swap after re-plugging; verify before a session.
- Two cameras on one USB controller → dropped frames; watch the fps warnings
  during recording, they translate directly into jerky training data.
- The `task` string is not metadata — it's model input. Typos become part of
  the instruction distribution.
- A stale `~/.cache/huggingface/token` turns public Hub downloads into 401s
  (`RepositoryNotFoundError`); `hf auth login` or delete the file.
- Disk budget: ~130 MB per 50 episodes at 2×640×480@30 with default encoding
  (based on the community datasets) — videos dominate; parquet is negligible.
