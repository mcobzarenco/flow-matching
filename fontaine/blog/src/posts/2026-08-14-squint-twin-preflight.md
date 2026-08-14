# Squint SO-101 twin preflight: it installs, steps, renders at 224, and speaks our action convention — measured, CPU-only

*2026-08-14, work session 18:14Z. Queue item `squint-twin-preflight`
(queued at the [wrist-transfer screen
design](2026-08-14-wrist-transfer-screen-design.md) close as the
successor tier for the success-rate form of the transfer question).
Substrate: [github.com/aalmuzairee/squint](https://github.com/aalmuzairee/squint)
(MIT), the paper read in [Papers: Squint](../papers/squint.md).
Probe script: `fontaine/scripts/squint_preflight.py`; raw facts in
`outputs/squint_preflight/facts_{main,third}.json`.*

**Plain words.** Before deciding whether to ever run our policies
inside the Squint simulator (a digital twin of exactly our robot arm,
with automatic success/failure grading), we spent one CPU-only session
checking that the thing actually works on our machine: does it
install, do all eight tasks load, can we drive the simulated arm the
same way we drive the real one, can it render camera images big enough
for our vision models, and how much does each simulated timestep cost?
Answer: yes on every count, with two small API traps documented below,
and the whole probe never touched the GPU (which the owner has
reserved). This note prices the tier; the wrist-transfer screen's
outcome decides whether we buy it.

## Verdict up front

**GO, mechanically** — an eval-only harness for an external policy is
a short gym loop from here; nothing needs a fork, one config knob
needs a per-process file constant, and per-episode sim cost is
trivial against Molmo2-4B inference. The visual-domain gap (the real
reason a zero-shot absolute number would be dishonest) is unchanged
from the paper read; this preflight is about the plumbing, and the
plumbing is sound.

## What was verified (all CPU: PhysX CPU backend + lavapipe software Vulkan; GPU at 0 MiB throughout)

**1. Install, isolated.** `uv venv` (Python 3.10) + CPU torch 2.6.0 +
`mani_skill_nightly` (resolves to the `mani_skill` 3.0.1 module) +
`gymnasium`, `opencv-python-headless`, `dacite`, `tyro`. No CUDA
wheels, no conda, ~2 minutes. The repo itself is not a package — the
envs register on `import envs` with the checkout on `PYTHONPATH`.

**2. All 8 `SO101*-v1` envs register and step headless.**
`{Reach,Lift,Place,Stack}{Cube,Can}` all `gym.make`, reset with a
seed, and step with random actions. Construction is ~0.9 s for the
first env, ~0.1 s for each subsequent one. Every task returns a
per-step `info` dict with `success` plus honest intermediate
predicates — e.g. Lift exposes `reached_object`, `is_item_grasped`,
`item_lifted`; Stack exposes `is_itemA_on_itemB`, `xy_dist`,
`z_dist`. That is exactly the labeled-rollout plumbing the
failure-detector calibration idea (#6) wants.

**3. The absolute-joint controller consumes our convention
end-to-end.** `control_mode="pd_joint_pos"` has
`normalize_action=False` with `lower/upper=None` in the agent source,
and the measured action-space bounds are the joint limits in radians
(e.g. shoulder_pan ±1.9199, gripper −0.1745..2.0944) — raw absolute
joint targets, the LeRobot convention our data and policies already
use, not a normalized [−1,1] box.

- *Scripted hold:* commanding the current `qpos` for 30 steps gives a
  measured max drift of **0.0 rad** — the PD controller holds an
  absolute target exactly.
- *Random walk:* absolute targets stepped by U(−0.04, 0.04) rad and
  clipped to limits track with **p50 error 0.014 rad / max 0.020
  rad** per step, the episode truncates at exactly 50 steps (the
  5-second, 10 Hz horizon from the paper), and the final `info`
  carries the full predicate set.

**4. 224×224 rendering is a kwarg, not a fork.** Passing
`sensor_configs=dict(width=224, height=224)` to `gym.make` overrides
the hard-coded 128×128 defaults; both cameras confirmed at
`(1, 224, 224, 3)` uint8. Frames below.

**5. Per-step wall time at 1 env, CPU.**

| mode | ms/step | steps/s | 50-step episode |
|---|---|---|---|
| state obs (no render) | 1.9 | ~525 | 0.10 s |
| wrist RGB 224×224 | 27 | ~37 | 1.35 s |
| third-person RGB 224×224 | 128 | ~8 | 6.4 s |

The render numbers are **lavapipe software Vulkan** — the honest
CPU-only floor, chosen to keep the owner's GPU reserve at 0 MiB. The
third-person camera sees the whole table (far more geometry for a
software rasterizer); on any real GPU renderer these collapse to
noise — the paper runs 1,024 envs on one 3090. Even at the CPU floor,
sim cost per episode is already small against a 4B-VLM action decode
per step.

## The frames

Wrist camera, raw render (`apply_overlay=False`) — what a
policy-relevant 224×224 eval frame looks like:

![Wrist camera 224x224, overlay off](https://mcobzarenco-fontaine-reports.static.hf.space/squint_preflight/wrist_224_overlay_off.png)

Same pose with the paper's greenscreen compositing on
(`apply_overlay=True`) — the black world Squint policies train in;
only robot + task objects survive the segmentation mask:

![Wrist camera 224x224, overlay on](https://mcobzarenco-fontaine-reports.static.hf.space/squint_preflight/wrist_224_overlay_on.png)

Third-person camera (`CAMERA_TYPE="third"`), raw render — the view
that exists behind a one-line switch:

![Third-person camera 224x224, overlay off](https://mcobzarenco-fontaine-reports.static.hf.space/squint_preflight/third_224_overlay_off.png)

## Two API traps found (the "what needs a subclass" part)

- **The overlay flag silently no-ops without segmentation in the obs
  mode.** `_get_obs_sensor_data` requires BOTH `rgb` and
  `segmentation` in the obs mode before compositing; with plain
  `obs_mode="rgb"`, `apply_overlay=True` returns raw frames with no
  warning (our first overlay-on and overlay-off frames were
  byte-identical). Use `obs_mode="rgb+segmentation"` when the
  greenscreen matters; for our eval use (raw renders) this trap is
  harmless.
- **Camera choice is a module-level constant, not a kwarg.**
  `CAMERA_TYPE = "wrist"|"third"` at the top of
  `envs/base_random_env.py` binds `DefaultCameraEnv` into every task
  class at import time. An in-process alias flip does not work — any
  `import envs.<submodule>` runs the package `__init__` first, which
  imports every task module before user code can touch the alias
  (verified: the flip produced a byte-identical wrist frame with the
  wrist class still in the MRO). The switch is real but per-process:
  sed the constant (what the probe does, reverting after) or carry a
  two-line patch. **A both-cameras-in-one-env variant — the setup our
  multi-view policies actually want — is the one thing that genuinely
  needs a small subclass** (a `_default_sensor_configs` returning
  both `CameraConfig`s, wrist mount + third mount are both already
  built by the base classes).

Also useful to know: `domain_randomization=False` cleanly freezes all
per-step camera jitter, lighting, and physics randomization (frames
above are deterministic), and the DR config is a plain dataclass
accepted as a dict through `gym.make`.

## What this prices, and what decides it

The wrist-transfer screen ([design
memo](2026-08-14-wrist-transfer-screen-design.md)) measures the
Δbehavior-per-Δhonesty curve on our own sim100 harness. Squint is the
successor tier if that screen hits `F-instrument` or the success
floor holds: closed-loop, success-scored, on our exact arm — but
far-OOD visually, so first use is **relative** screens (A/B deltas
between our checkpoints under a constant domain gap) plus unlimited
ground-truth-labeled rollouts for probe calibration (#6). This
preflight establishes the tier costs: ~2 min install, zero GPU
requirement for the harness itself, ~100-line eval loop, 1.35
s/episode sim overhead at the CPU floor, one subclass (dual-camera)
and one file constant (camera type) of engineering. Nothing here
commits us to the tier; the screen's outcome does.
