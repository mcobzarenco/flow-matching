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
    Path(__file__).parents[1] / "assets" / "robotstudio_so101" / "bijou_pickplace.xml"
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

# Benchy spawn region: in front of the disk, inside comfortable reach
# (menagerie pickup keyframe grasps at ~0.22 m forward). Ranges chosen
# for mean benchy->disk distance ~9.5 cm (halved from the original
# ~18.3 cm); the near x bound keeps the hull (3 cm half-length) >=1 cm
# clear of the settled home pose's jaw tips (x=0.155 - spawns from 0.17
# used to land the boat ON the parked jaw for ~4% of seeds), and the
# hull stays clear of the 4 cm disk. Relative to the disk at (0.22, 0.11).
SPAWN_X = (0.195, 0.27)
SPAWN_Y = (-0.005, 0.04)
# Episode-initial pose: the median first-frame observation.state across
# the 50 episodes of so101_pick_place_v2 (measured from the dataset
# parquet; per-joint std 2-20 deg). Menagerie's shoulder_lift/elbow_flex
# ranges are widened at load (_widen_joint_limits) so this pose is
# representable; the settled arm still rests with the jaw tip on the
# table at elbow ~90.4 (6.6 deg shy of the rig median) - the reachable
# projection of this pose given zero-perfect sim joints vs the rig's
# calibration offsets. The eval protocol pins the SETTLED start state,
# which is seed-independent (spread <0.003 deg across seeds).
HOME_DEGREES = np.array([4.6, -102.7, 97.0, 78.7, 77.6, 3.5])
# The leader arm mirrors the follower during teleop; at episode start the
# operator holds it at the same rest pose.
LEADER_DEGREES = np.array([4.6, -102.7, 97.0, 78.7, 77.6, 3.5])


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
        self._widen_joint_limits()
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
        self._leader_actuators = np.array(
            [self.model.actuator(f"leader-{name}").id for name in JOINTS],
        )
        # Disk geometry from the model - the XML is the single source of
        # truth (a hardcoded copy here once drifted from it).
        disk = self.model.geom("disk")
        self.disk_center: tuple[float, float] = (float(disk.pos[0]), float(disk.pos[1]))
        self.disk_radius: float = float(disk.size[0])
        self._recolor_arm()

    def _widen_joint_limits(self) -> None:
        """Menagerie's shoulder_lift (+-100 deg) and elbow_flex (+-96.8)
        ranges are narrower than the rig's measured excursions: the median
        real episode STARTS at shoulder_lift -102.7 / elbow_flex 97.0, so
        the model cannot represent the recorded start state (and the
        clamped shoulder tips the forearm low enough that the elbow stalls
        on the table ~8 deg short of home). Widen at runtime rather than
        editing the vendored XML; the servo-sysid item pins final values."""
        widened = {"shoulder_lift": 110.0, "elbow_flex": 100.0}
        for prefix in ("", "leader-"):
            for name, limit in widened.items():
                bound = np.deg2rad((-limit, limit))
                self.model.jnt_range[self.model.joint(prefix + name).id] = bound
                self.model.actuator_ctrlrange[self.model.actuator(prefix + name).id] = (
                    bound
                )

    def _recolor_arm(self) -> None:
        """Menagerie ships the yellow-print arm; the rig's are black, and
        only the FOLLOWER has the bright orange moving jaw (owner-confirmed,
        and visible in the follower's own wrist view,
        outputs/sim/real/wrist_00260.png). Runtime recolor instead of
        editing the vendored XML."""
        black = (0.13, 0.13, 0.14, 1.0)
        orange = (0.95, 0.45, 0.1, 1.0)
        for index in range(self.model.nmat):
            name = self.model.mat(index).name
            if "so101" not in name:
                continue
            follower_jaw = "moving_jaw" in name and not name.startswith("leader-")
            self.model.mat_rgba[index] = orange if follower_jaw else black

    def reset(self, seed: int) -> SimObservation:
        """Home the arm, place benchy at a seeded pose, randomize its
        color, settle physics until contacts are quiet.

        The arm settles FIRST, with the benchy parked outside its sweep
        (mj_resetData lays the arm out over the workspace, and driving up
        to home used to strike an already-spawned boat on ~10% of seeds,
        displacing it up to 30 mm); the benchy is placed at its seeded
        pose only after the arm is home, then given a short settle of its
        own. `reset_strike_contacts` counts gripper-benchy contacts seen
        during the whole reset - 0 for every seed is an eval-protocol
        gate."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        x = rng.uniform(*SPAWN_X)
        y = rng.uniform(*SPAWN_Y)
        yaw = rng.uniform(-np.pi, np.pi)
        quat = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        self.reset_spawn_xy: tuple[float, float] = (float(x), float(y))

        # Cheap texture randomization: tint the flat-white texture via the
        # material color (full tex_data painting is the follow-up). Biased
        # light like the real light-gray print, with occasional color.
        base = rng.uniform(0.55, 0.95)
        rgba = np.append(
            np.clip(base + rng.uniform(-0.25, 0.1, size=3), 0.05, 1.0),
            1.0,
        )
        self.model.mat_rgba[self._benchy_mat] = rgba

        # Park the benchy far down-table while the arm settles.
        adr = self._benchy_qpos
        self.data.qpos[adr : adr + 3] = (0.9, 0.1, 0.001)
        self.data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)

        # Drive (not teleport) to home so the reset respects servo
        # dynamics; 1 s settles both arms.
        self.reset_strike_contacts = 0
        self.data.ctrl[self._actuator_ids] = np.deg2rad(HOME_DEGREES)
        self.data.ctrl[self._leader_actuators] = np.deg2rad(LEADER_DEGREES)
        self._settle_counting_strikes(200)

        # Now place the benchy at its seeded pose and let it settle onto
        # the table (spawned 1 mm up, at rest within a few steps).
        self.data.qpos[adr : adr + 3] = (x, y, 0.001)
        self.data.qpos[adr + 3 : adr + 7] = quat
        vadr = self.model.joint("benchy_free").dofadr[0]
        self.data.qvel[vadr : vadr + 6] = 0.0
        self._settle_counting_strikes(30)
        return self.observe()

    def _settle_counting_strikes(self, nstep: int) -> None:
        """Step one-by-one, tallying arm-benchy contacts into
        `reset_strike_contacts` (single steps are bit-identical to one
        batched mj_step call). Any non-world body touching the benchy
        during reset is a strike - the table is the only thing it should
        rest on before the episode starts."""
        world = 0
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                bodies = (
                    self.model.geom(contact.geom1).bodyid[0],
                    self.model.geom(contact.geom2).bodyid[0],
                )
                benchy = self._benchy_body in bodies
                arm = all(b != world for b in bodies)
                if benchy and arm and bodies != (self._benchy_body,) * 2:
                    self.reset_strike_contacts += 1

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

    def benchy_disk_distance(self) -> float:
        """XY distance from the benchy base to the disk center (meters)."""
        pos, _ = self.benchy_pose()
        return float(
            np.hypot(pos[0] - self.disk_center[0], pos[1] - self.disk_center[1]),
        )

    def success(self) -> bool:
        """Benchy resting upright on the disk: xy within the disk radius,
        base at disk height, still, and not held (gripper open enough)."""
        pos, upright = self.benchy_pose()
        on_disk = self.benchy_disk_distance() < self.disk_radius
        at_height = 0.004 < pos[2] < 0.03
        still = float(np.abs(self.data.qvel[: self.model.nv]).max()) < 0.5
        return on_disk and at_height and upright > 0.9 and still
