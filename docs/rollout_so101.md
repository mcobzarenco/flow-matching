# Rolling out Bijou on a physical SO-101

Runbook for `python -m bijou.rollout` (added `15da227`). Status: **runs
on the physical arm** (owner's laptop deployment; measured mean-of-10
replan at ~576 ms with `--sample-draws 10`). This runbook covers the
core path; flags added since it was written — `--sample-draws`,
`--async-inference`, `--return-home`/`--return-home-seconds`,
`--generate [fields…]`,
`--outcome`/`--smoothness`/`--subgoal` conditioning, `--switch-blend`,
`--offload-ple`, `--target-time`, `--control-fps`, `--noise-ticket`
(fixed noise vector for every replan, npz `tickets [count, chunk,
dim]` — the eval CLI's ticket format; sha256 echoed in the banner),
`--joint-frame` (arm↔model joint-convention remap — see
`so101-joint-conventions.md` for the two axes; required as
`v30-to-v21` when a checkpoint bakes in the pre-lerobot-0.5 degrees
frame, e.g. converted MolmoAct2 releases, on an arm calibrated with
lerobot ≥ 0.5; molmo_flow checkpoints are additionally gated against
their own model-frame state band at the first observation),
and the safety envelope gate (`--unclamped`/`--skip-envelope-check`
escape hatches) — are documented in `python -m bijou.rollout --help`
and architecture.md §6.

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
    --flow-decoder-dtype bfloat16 \
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

- **`--camera` keys ARE semantic kinds** (`wrist/top/front/side/unknown`;
  anything else is refused, duplicates too). The kind drives the
  prompt's (kind, name) image order — matching training's kind-major
  sort — and the tag kind-aware prompt formats render. Asserted kinds
  are cross-checked against the `--stats-dataset`'s judged kinds
  (`meta/camera_kinds.json`) with a loud warning on mismatch or
  uncovered judged kinds; the asserted kinds always win. The owner's
  rig: `--camera top=…` (the judge voted the scene camera kind `top`)
  and `--camera wrist=…`. Swapped device paths still silently swap
  views — the kind describes the device you attach it to.
- **`--max-relative-target 20`** engages lerobot's per-tick joint-motion
  clamp. Keep it on until the policy is trusted; it damps large motions.
- **`--flow-decoder-dtype bfloat16` fits the 8 GB laptop, measured**: on
  the owner's RTX 3000 Ada (8 GiB), loading step_001000 puts 4.77 GiB of
  weights on the GPU and a full predict peaks at 6.29 GiB reserved
  (outputs/probe_rollout_vram.py — synthetic 2-camera observation through
  the real predict path; measured under the flag's old name
  `--expert-dtype`, same cast). An fp32 decoder would add ~0.8 GiB
  weights plus larger activations — possible but with no headroom.
- **Replan latency, measured on the same laptop**: ~250 ms warm (Heun-5,
  2 cameras; the first replan pays ~2 s of CUDA warmup). At
  `--execute-horizon 40` @ 30 fps that is a ~250 ms hold every 1.33 s of
  motion. Levers: horizon 50 (full chunk, fewer holds), `--sample-steps 3`.
  A smaller `--max-soft-tokens` is NOT available at rollout (the prompt
  budget is baked into the checkpoint).
- **Task string**: use the instruction wording the fine-tune data carried
  (`task` field of the episodes) — the model is conditioned on it.
- Actions are commanded in the dataset's raw units — for all our
  datasets: DEGREES under the v3.0 calibration, matching
  `SOFollowerRobotConfig`'s `use_degrees=True` default. Two separate
  mismatches are possible (`so101-joint-conventions.md`): a UNITS
  mismatch (a robot configured `use_degrees=False` speaks ±100 while
  stats are degrees) and a CALIBRATION mismatch (a checkpoint trained
  under v2.1 zeros/directions — `--joint-frame v30-to-v21`, or a
  conversion-time table remap, never both). Either way: check
  `--check` output's state-stats line against a live joint readout
  before the first motion.
- Ctrl-C stops cleanly (disconnects, torque released per lerobot config).

## Known unknowns before first physical run

- Closed-loop stability of chunk handoffs (open-loop metrics don't test
  replan boundaries).
- Whether `--fps 30` execution matches the 30 fps recording cadence well
  enough (drift between commanded tick and camera exposure).
- Camera exposure/white balance vs the recordings (the fine-tune saw one
  lighting setup; big shifts degrade the vision conditioning).

## Remote inference (policy server on a GPU box)

For checkpoints that don't fit the 8 GiB laptop (molmoact2-class: the
trunk alone is 9.7 GB), `python -m bijou.rollout --policy-server <url>`
runs the SAME sync loop with inference on a remote
`python -m bijou.policy_server`: cameras/robot stay local, each replan
ships the observation as base64 JPEG over HTTP and blocks until the
chunk comes back — the servos hold their last goal during the block,
exactly the sync path's existing semantics with a longer freeze.
Async/latency-hiding over the WAN is deliberately not implemented
(`--async-inference` + `--policy-server` is refused at the parser).
The transport is unauthenticated plain HTTP: the server binds loopback
by default and the ONLY supported remote path is an SSH tunnel.

### The commands (deployment target: fontaine_grasp_sft_joint_corrected)

```sh
# on the GPU box, in tmux:
~/.local/bin/uv run python -m bijou.policy_server \
    --checkpoint ~/checkpoints/finetune/fontaine_grasp_sft_joint_corrected/step_002000 \
    --device cuda --flow-decoder-dtype bfloat16 --port 8143

# on the laptop, terminal 1 (the tunnel — the only supported remote path):
ssh -N -L 8143:localhost:8143 ubuntu@68.209.75.143

# on the laptop, terminal 2 (note: no --checkpoint, no model dtype flags):
uv run python -m bijou.rollout \
    --policy-server http://localhost:8143 \
    --stats-repo-id mcobzarenco/so101_pick_place_v2 \
    --port /dev/ttyACM1 --robot-id follower0 \
    --camera wrist=/dev/video4 --camera top=/dev/video6 \
    --task "Pick up the toy boat and place it on the wooden disk." \
    --max-relative-target 40 --duration 300
```

Add `--check` to terminal 2 first: it exercises one synthetic predict
THROUGH the tunnel (spec fetch, JPEG encode, server inference, chunk
decode) without touching the robot.

Flag split under `--policy-server`: model-side flags
(`--flow-decoder-dtype`, `--offload-ple`) and the noise `--seed` belong
to the SERVER invocation and are refused by rollout, loudly, with the
remedy. Decode knobs (`--sample-steps`, `--sample-method`,
`--sample-draws`, `--generate`) stay on rollout and ride each request.
`--stats-repo-id` keeps its exact local behavior: the server's `/spec`
ships the checkpoint's per-dataset stats tables, the client resolves
the repo id against them and ships the vectors in every request.

Two checkpoint-specific facts for the commands above, read off the
actual step_002000 metadata (2026-08-16):

- **Stats table**: its `per_dataset_stats` contains ONLY
  `fontaine/grasp_sft_demos_v0` (the sim-collected SFT demos it was
  fine-tuned on) — `--stats-repo-id mcobzarenco/so101_pick_place_v2`
  will be refused loudly ("not in the checkpoint's stats table (1
  entries)"). Pass `--stats-repo-id fontaine/grasp_sft_demos_v0`, or
  `--stats-dataset <local rig dataset dir>` if the rig should
  normalize under its own recorded stats. Whether sim-frame stats are
  right for the physical arm is an operator call — the envelope gate
  will compare the first observation against whichever table you pick.
- **Recorded serving point**: `euler-10`. Rollout's CLI defaults are
  `heun-5` (unchanged, local and remote alike) — add
  `--sample-steps 10 --sample-method euler` to serve the recorded
  operating point.

### Camera kinds for this checkpoint

The molmoact2 prompt is POSITIONAL ("Image 1…Image 2"; it renders no
kind tags), and images are ordered by the (kind, name) sort — so the
kinds you assert control the slots. The fine-tune's dataset recorded a
scene camera (judged kind `top` on the rig data) and a wrist camera:
Image 1 = scene, Image 2 = wrist. The operator's
`--camera top=/dev/video6 --camera wrist=/dev/video4` is exactly right
(`top` sorts before `wrist`), provided `/dev/video6` IS the scene
camera — swapped device paths silently swap views in the prompt;
double-check with `uv run lerobot-find-cameras opencv`.

### Per-replan gap arithmetic

Each replan freezes the arm for:

```
gap ≈ RTT + upload(JPEGs) + server decode + inference + download(chunk)
```

- **JPEG upload**: measured with this repo's encoder (quality 95),
  640x480 rig-like smooth frames are ~23 KiB JPEG → ~31 KiB base64
  each (~62 KiB/request at two cameras; encode ~5 ms/frame CPU).
  Synthetic NOISE frames are the worst case at ~353 KiB → ~470 KiB
  each — the probe numbers below carry that inflation. On a 10 Mbit/s
  uplink, two real frames ≈ 50 ms; the 30x6 float chunk coming back is
  ~10 KiB.
- **Inference**: the server's per-request log line and the response's
  `timings` field (`decode_ms`/`infer_ms`/`total_ms`) report it —
  measure, don't assume; rollout also prints the split per replan
  (`server | infer … | wire+encode …`). Note `decode_ms` includes
  reading the request body off the socket, so slow-uplink time lands
  there.

### Measured box numbers (2026-08-16, CPU-ONLY — GPU was occupied)

Smoke run on the real step_002000 checkpoint on the H100 box, server
on `--device cpu` with 16 OMP threads because another agent's GPU job
was live (etiquette: never contend). Client on the dev laptop through
a real `ssh -N -L` tunnel, two synthetic 640x480 noise frames +
the checkpoint's recorded stats per request:

- server RSS after load: **11.2 GiB** host RAM (bf16 trunk + bf16-cast
  flow decoder; `backbone.safetensors` 9.7 GB + `flow_decoder.safetensors`
  2.5 GB fp32 on disk — `optimizer.pt` is never read);
- `GET /spec`: 6 ms loopback on the box; 282 ms through the tunnel
  including client banner work (spec payload 2.5 KiB);
- `/predict` through the tunnel: cold wall 5438 ms (server total 4942 =
  decode 1016 + infer 3926), warm wall 6111 ms (total 5549 = decode
  1913 + infer 3636). The ~1-2 s decode is the ~940 KiB noise-frame
  upload crossing the home uplink; real frames cut that ~15x. The
  ~3.6-3.9 s infer is euler-10 on CPU — **not deployment-
  representative**; expect a few hundred ms on the H100 (unmeasured:
  the GPU was busy). On CUDA the server prints its own
  "GPU memory after load" line at startup — record startup VRAM and
  warm `infer_ms` from there on the first GPU session.

### Troubleshooting

- **Timeout / connection refused**: rollout raises `SystemExit`
  immediately (no silent retries; `--policy-server-timeout`, default
  30 s). Check the tunnel terminal is still running, then the server
  tmux on the box (`pgrep -af bijou.policy_server`); a first CUDA
  predict pays kernel warmup — if it exceeds the timeout, raise
  `--policy-server-timeout` for the first run.
- **schema_version mismatch**: the client refuses at construction,
  naming both versions — the box and laptop checkouts disagree about
  the wire protocol; update the older side. A git-rev WARNING in the
  banner is the early hint (rev "unknown" means the server runs from a
  non-git scratch dir).
- **Tunnel drops mid-rollout**: the next replan raises loudly and the
  loop stops; the arm holds its last commanded position (and
  `--return-home` still runs on the way out). Restart the tunnel and
  rerun — the server keeps its loaded model; a client restart with
  different cameras or options needs no server restart.
- **Server-side OOM**: the server process dies (the request errors as
  a connection failure client-side). Restart it with
  `--flow-decoder-dtype bfloat16` (halves decoder memory), check
  nothing else occupies the GPU (`nvidia-smi`), or fall back to
  `--device cpu` to keep a slow-but-alive service. Requests themselves
  never kill the server — bad inputs return structured 400s.
