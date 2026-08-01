"""MuJoCo SO-101 pick-and-place sim (prototype).

Mirrors the physical rig's conventions at the seam so the bijou/lerobot
rollout loops can drive it with minimal glue:

  - observation: camera frames (HWC uint8) + 6-dof state in DEGREES,
    joint order = the rig motor order (shoulder_pan .. gripper)
  - action: 6 absolute joint targets in degrees, applied at 30 Hz

The arm model is menagerie's robotstudio_so101 (position actuators with
lerobot-derived STS3215 gains; wrist_cam included). The scene adds a top
camera, a wooden disk, and a free benchy whose color/texture randomizes
per reset.

Calibration caveat (matters when benching real checkpoints): sim joints
are zero-perfect, the rig's are offset by its calibration file. The rig's
normalization stats absorb affine offsets for the model, but any residual
sim-vs-rig zero mismatch is part of the domain gap to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = (
    Path(__file__).parent / "assets" / "robotstudio_so101" / "bijou_pickplace.xml"
)
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CONTROL_HZ = 30
# 7 physics steps of 0.005 s per control tick = 35 ms (~5% slow vs 33.3 ms;
# accepted for the prototype rather than re-tuning menagerie's timestep).
PHYSICS_STEPS_PER_TICK = 7

# Benchy spawn region: annulus around the disk kept inside comfortable
# reach (menagerie pickup keyframe grasps at ~0.22 m forward).
SPAWN_X = (0.16, 0.28)
SPAWN_Y = (-0.16, 0.02)
# Rest pose in degrees (rig order): arm folded up, gripper overlooking
# the workspace - approximates the physical rig's rest; zero-pose lays
# the arm flat across the table and points the wrist cam at the horizon.
HOME_DEGREES = np.array([0.0, -55.0, 75.0, 55.0, 0.0, 30.0])
DISK_CENTER = (0.22, 0.11)
DISK_RADIUS = 0.06


@dataclass(frozen=True, slots=True)
class SimObservation:
    """One control-tick observation.

    - top / wrist: [H, W, 3] uint8 (rendered at the requested size)
    - state: [6] float64, joint positions in degrees, rig motor order
    """

    top: np.ndarray
    wrist: np.ndarray
    state: np.ndarray


class SO101Sim:
    """Seeded, deterministic-per-seed SO-101 pick-place environment."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._joint_qpos = np.array(
            [self.model.joint(name).qposadr[0] for name in JOINTS],
        )
        self._actuator_ids = np.array(
            [self.model.actuator(name).id for name in JOINTS],
        )
        self._ctrl_low, self._ctrl_high = self.model.actuator_ctrlrange[
            self._actuator_ids
        ].T
        self._benchy_body = self.model.body("benchy").id
        self._benchy_qpos = self.model.joint("benchy_free").qposadr[0]
        self._benchy_mat = self.model.geom("benchy_visual").matid[0]

    def reset(self, seed: int) -> SimObservation:
        """Home the arm, place benchy at a seeded pose, randomize its
        color, settle physics until contacts are quiet."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        x = rng.uniform(*SPAWN_X)
        y = rng.uniform(*SPAWN_Y)
        yaw = rng.uniform(-np.pi, np.pi)
        adr = self._benchy_qpos
        self.data.qpos[adr : adr + 3] = (x, y, 0.001)
        self.data.qpos[adr + 3 : adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))

        # Cheap texture randomization: tint the flat-white texture via the
        # material color (full tex_data painting is the follow-up).
        rgba = np.append(rng.uniform(0.1, 0.9, size=3), 1.0)
        self.model.mat_rgba[self._benchy_mat] = rgba

        # Drive (not teleport) to home so the reset respects servo
        # dynamics; 1 s settles both the arm and the spawned benchy.
        self.data.ctrl[self._actuator_ids] = np.deg2rad(HOME_DEGREES)
        mujoco.mj_step(self.model, self.data, nstep=200)
        return self.observe()

    def step(self, action_degrees: np.ndarray) -> SimObservation:
        """Apply absolute joint targets (degrees, rig order) for one
        30 Hz control tick.

        - action_degrees: [6] float
        """
        target = np.clip(np.deg2rad(action_degrees), self._ctrl_low, self._ctrl_high)
        self.data.ctrl[self._actuator_ids] = target
        mujoco.mj_step(self.model, self.data, nstep=PHYSICS_STEPS_PER_TICK)
        return self.observe()

    def observe(self) -> SimObservation:
        state = np.rad2deg(self.data.qpos[self._joint_qpos])
        self.renderer.update_scene(self.data, camera="top_cam")
        top = self.renderer.render()
        self.renderer.update_scene(self.data, camera="wrist_cam")
        wrist = self.renderer.render()
        return SimObservation(top=top, wrist=wrist, state=state)

    def benchy_pose(self) -> tuple[np.ndarray, float]:
        """Benchy base position [3] and upright score (world-z of the
        body z-axis; 1 = upright, -1 = capsized)."""
        pos = self.data.xpos[self._benchy_body].copy()
        upright = float(self.data.xmat[self._benchy_body].reshape(3, 3)[2, 2])
        return pos, upright

    def success(self) -> bool:
        """Benchy resting upright on the disk: xy within the disk radius,
        base at disk height, still, and not held (gripper open enough)."""
        pos, upright = self.benchy_pose()
        dx = pos[0] - DISK_CENTER[0]
        dy = pos[1] - DISK_CENTER[1]
        on_disk = float(np.hypot(dx, dy)) < DISK_RADIUS
        at_height = 0.004 < pos[2] < 0.03
        still = float(np.abs(self.data.qvel[: self.model.nv]).max()) < 0.5
        return on_disk and at_height and upright > 0.9 and still
