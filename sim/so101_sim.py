"""MuJoCo SO-101 pick-and-place sim (prototype).

Mirrors the physical rig's conventions at the seam so the bijou/lerobot
rollout loops can drive it with minimal glue:

  - observation: camera frames (HWC uint8) + 6-dof state in DEGREES,
    joint order = the rig motor order (shoulder_pan .. gripper)
  - action: 6 absolute joint targets in degrees, applied at 30 Hz

The arm model is menagerie's robotstudio_so101 (wrist_cam included), with
the STS3215 position-servo params replaced at load by the replay-identified
SERVO_SYSID set (sim.sysid_servo). The scene adds a top
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
from typing import ClassVar

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
# Servo params identified by open-loop replay of real rig episodes
# (sim.sysid_servo, outputs/sim/sysid_servo.json): held-out-episode arm
# MAE 1.76 deg vs 3.31 with the vendored menagerie gains (whose kp 998
# with forcerange 2.94 saturates at 0.17 deg of error - a force-clamped
# bang-bang servo, not the STS3215's measured response). Applied at load,
# shared by all six STS3215 actuators on BOTH arms.
SERVO_SYSID = {
    "kp": 108.18,
    "kv": 13.377,
    "forcerange": 3.478,
    "damping": 0.722,
    "frictionloss": 0.0183,
    "armature": 0.2045,
}


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

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        render_style: str = "v3",
    ) -> None:
        if render_style not in ("v0", "v1", "v2", "v3"):
            raise ValueError(
                f"render_style {render_style!r} not in ('v0', 'v1', 'v2', 'v3')",
            )
        self.render_style = render_style
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self._widen_joint_limits()
        self._apply_servo_sysid()
        self.data = mujoco.MjData(self.model)
        # Lazy: constructed on first observe() — physics-only consumers
        # (and GL-less test environments) never pay for a GL context.
        self._render_size = (height, width)
        self._renderer: mujoco.Renderer | None = None
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
        self._table_mat = self.model.geom("table").matid[0]
        # Appearance-jitter baselines: jitter multiplies/offsets these
        # stored scene values, never the (already-jittered) live ones.
        self._sun = self.model.light("sun").id
        self._fill = self.model.light("fill").id
        self._base_sun_diffuse = self.model.light_diffuse[self._sun].copy()
        self._base_sun_dir = self.model.light_dir[self._sun].copy()
        self._base_fill_diffuse = self.model.light_diffuse[self._fill].copy()
        self._base_table_rgba = self.model.mat_rgba[self._table_mat].copy()
        self._leader_actuators = np.array(
            [self.model.actuator(f"leader-{name}").id for name in JOINTS],
        )
        # Disk geometry from the model - the XML is the single source of
        # truth (a hardcoded copy here once drifted from it).
        disk = self.model.geom("disk")
        self.disk_center: tuple[float, float] = (float(disk.pos[0]), float(disk.pos[1]))
        self.disk_radius: float = float(disk.size[0])
        self._recolor_arm()
        self._repose_wrist_cam()
        self._noise_rng = np.random.default_rng(0)  # re-seeded per reset
        if self.render_style in ("v1", "v2", "v3"):
            self._init_fisheye(width, height)
        if self.render_style in ("v2", "v3"):
            self._init_inpainting(width, height)
        if self.render_style == "v3":
            self._init_bank(width, height)

    def _widen_joint_limits(self) -> None:
        """Menagerie's shoulder_lift (+-100 deg) and elbow_flex (+-96.8)
        ranges are narrower than the rig's measured excursions: the median
        real episode STARTS at shoulder_lift -102.7 / elbow_flex 97.0, so
        the model cannot represent the recorded start state (and the
        clamped shoulder tips the forearm low enough that the elbow stalls
        on the table ~8 deg short of home). Widen at runtime rather than
        editing the vendored XML."""
        widened = {"shoulder_lift": 110.0, "elbow_flex": 100.0}
        for prefix in ("", "leader-"):
            for name, limit in widened.items():
                bound = np.deg2rad((-limit, limit))
                self.model.jnt_range[self.model.joint(prefix + name).id] = bound
                self.model.actuator_ctrlrange[self.model.actuator(prefix + name).id] = (
                    bound
                )

    def _apply_servo_sysid(self) -> None:
        """Overwrite the vendored STS3215 actuator/joint params with the
        replay-identified SERVO_SYSID values (runtime, both arms - same
        convention as _widen_joint_limits: never edit the vendored XML)."""
        for prefix in ("", "leader-"):
            for name in JOINTS:
                actuator = self.model.actuator(prefix + name)
                actuator.gainprm[0] = SERVO_SYSID["kp"]
                actuator.biasprm[1] = -SERVO_SYSID["kp"]
                actuator.biasprm[2] = -SERVO_SYSID["kv"]
                actuator.forcerange[:] = (
                    -SERVO_SYSID["forcerange"],
                    SERVO_SYSID["forcerange"],
                )
                dof = self.model.joint(prefix + name).dofadr[0]
                self.model.dof_damping[dof] = SERVO_SYSID["damping"]
                self.model.dof_frictionloss[dof] = SERVO_SYSID["frictionloss"]
                self.model.dof_armature[dof] = SERVO_SYSID["armature"]

    def _repose_wrist_cam(self) -> None:
        """Menagerie's wrist_cam does not match the rig's bracket view:
        its image-right is world +y at the home pose, which puts the
        orange moving jaw on the image-RIGHT — the real wrist frames
        show it on the LEFT, looking from above/behind the wrist over
        the jaw tips at the table. Re-pose at load (runtime, vendored
        XML untouched): mount-local pose computed from the settled home
        pose as a lookat — camera over the jaw base (world ~(0.150,
        0.000, 0.150) at home), looking ~65 deg below horizontal down
        the jaws at the table, image-right = world -y — and the 16:9
        sensor model swapped for the top cam's fovy (the rig captures
        4:3 640x480 center crops on both modules).

        Re-tuned 2026-08-12 (sim-wrist-periphery-fix): the original
        pose sat at the wrist top behind the gripper (world ~(0.096,
        -0.004, 0.160), ~55 deg), which under the 72-deg v1 source
        filled the bottom ~40% of frame with the gripper body — real
        start frames show only slim jaw tips in the bottom quarter
        over full-frame table."""
        for prefix in ("", "leader-"):
            cam = self.model.camera(prefix + "wrist_cam")
            cam.pos[:] = (0.02416, -0.05504, 0.03225)
            cam.quat[:] = (-0.24345, -0.05192, 0.02663, 0.96816)
            self.model.cam_sensorsize[cam.id] = 0.0
            cam.fovy[0] = 52.0

    # v1 render style (visual matching, prereg 2026-08-12): the rig's
    # cameras are 130-deg wide-angle modules center-cropped to 4:3 —
    # straight table planks visibly bow in every real frame, and the
    # frame periphery holds content a pinhole cannot see. The sim
    # renders a wider pinhole source, then remaps it through a
    # center-matched equidistant fisheye: output radius r shows the ray
    # at angle theta = r / F_DIST, with F_DIST equal to the pinhole
    # focal of the previously-matched 52-deg view — so magnification at
    # the image center is unchanged and distortion grows toward the
    # edges like the real lens.
    V1_SRC_FOVY = 72.0
    V1_CENTER_FOVY = 52.0
    # Fixed per-channel affine grade (out = in * gain + bias), computed
    # once per camera from 25 v1 fisheye reset renders vs the pinned 150
    # real_v2 reference frames (global per-channel mean/std match). The
    # rig modules auto-white-balance the warm wood to near-neutral and
    # compress contrast; the raw render does neither.
    V1_GRADE: ClassVar[dict[str, tuple[tuple[float, ...], tuple[float, ...]]]] = {
        "top": ((0.8149, 0.8607, 0.8889), (30.61, 26.29, 29.44)),
        "wrist": ((0.7432, 0.7720, 0.8600), (57.88, 54.04, 47.57)),
    }
    # Sensor emulation (amendment to the registered post-process axes,
    # labeled in the results post): the real modules' optics soften
    # CG-crisp edges and every real frame carries sensor noise; raw
    # renders have neither. Gaussian PSF sigma in px; noise sigma in
    # 8-bit counts, drawn from the appearance RNG (deterministic per
    # reset seed, fresh each tick).
    V1_BLUR_SIGMA = 0.7
    V1_NOISE_SIGMA = 2.0

    def _init_fisheye(self, width: int, height: int) -> None:
        """Precompute the bilinear remap grid (output pixel -> source
        pinhole pixel) shared by both cameras, and widen both cameras'
        fovy to the source value."""
        for name in ("top_cam", "wrist_cam"):
            self.model.camera(name).fovy[0] = self.V1_SRC_FOVY
        f_dist = (height / 2.0) / np.tan(np.deg2rad(self.V1_CENTER_FOVY) / 2.0)
        f_src = (height / 2.0) / np.tan(np.deg2rad(self.V1_SRC_FOVY) / 2.0)
        v, u = np.mgrid[0:height, 0:width].astype(np.float64)
        x = u - (width - 1) / 2.0
        y = v - (height - 1) / 2.0
        r_out = np.hypot(x, y)
        theta = r_out / f_dist
        # Rays past the source's diagonal never occur: corner theta
        # (~47 deg) stays inside the 72-deg-fovy source diagonal (~50).
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(r_out > 0, f_src * np.tan(theta) / r_out, f_src / f_dist)
        sx = x * scale + (width - 1) / 2.0
        sy = y * scale + (height - 1) / 2.0
        x0 = np.clip(np.floor(sx).astype(np.int64), 0, width - 2)
        y0 = np.clip(np.floor(sy).astype(np.int64), 0, height - 2)
        wx = np.clip(sx - x0, 0.0, 1.0)[..., None]
        wy = np.clip(sy - y0, 0.0, 1.0)[..., None]
        self._fisheye = (x0, y0, wx, wy)

    def _remap(self, src: np.ndarray) -> np.ndarray:
        """Bilinear fisheye remap of a source pinhole image — [H, W, C]
        float in/out (any channel count)."""
        x0, y0, wx, wy = self._fisheye
        top = src[y0, x0] * (1 - wx) + src[y0, x0 + 1] * wx
        bottom = src[y0 + 1, x0] * (1 - wx) + src[y0 + 1, x0 + 1] * wx
        return top * (1 - wy) + bottom * wy

    def _apply_fisheye(self, frame: np.ndarray) -> np.ndarray:
        """Bilinear remap of a source pinhole render into the distorted
        output — [H, W, 3] uint8 in, same out."""
        remapped = self._remap(frame.astype(np.float64))
        return np.clip(remapped, 0, 255).astype(np.uint8)

    # v2 render style (real-frame inpainting, prereg 2026-08-12): the
    # background is a real photo — a per-camera clean plate mined from
    # the real_v2 reference-half episodes (fontaine/scripts/
    # make_clean_plates.py) — and only dynamic content is rendered:
    # every geom on a non-world body (both arms, benchy) plus the
    # named on-table statics whose real twins move across episodes and
    # median out of the plate (disk, clutter stand-ins). Off-table
    # statics (table, floor, pole, chairs, bag) come from the plate.
    PLATES_DIR = Path(__file__).parents[1] / "assets" / "real_plates"
    V2_DYNAMIC_STATICS = ("disk", "mouse", "mug", "laptop", "pcb")

    def _init_inpainting(self, width: int, height: int) -> None:
        """Load the per-camera clean plates and precompute the dynamic
        geom-id set for the segmentation mask."""
        from PIL import Image

        self._plates: dict[str, np.ndarray] = {}
        for camera in ("top", "wrist"):
            plate = np.asarray(
                Image.open(self.PLATES_DIR / f"{camera}_plate.png"),
                dtype=np.float64,
            )
            if plate.shape != (height, width, 3):
                raise ValueError(
                    f"{camera} plate {plate.shape} does not match the "
                    f"render size ({height}, {width}, 3)",
                )
            self._plates[camera] = plate
        dynamic = [
            index
            for index in range(self.model.ngeom)
            if self.model.geom_bodyid[index] != 0
        ] + [self.model.geom(name).id for name in self.V2_DYNAMIC_STATICS]
        self._dynamic_geoms = np.array(sorted(dynamic))

    # v3 render style (content diversity, prereg 2026-08-12): the v2
    # composite, except (a) the top plate is drawn per reset from a
    # bank of per-episode plates (each carrying that real episode's
    # lighting state; mined ghost-free by make_clean_plates.py --bank)
    # and (b) the contype-0 clutter stand-ins get per-reset poses and
    # presence drawn from the measured real between-episode spread
    # (bank_manifest.json clutter_ranges). Both draws consume the
    # appearance RNG AFTER every draw the v2 style makes, so the
    # wrist path stays bit-identical to v2 — the wrist render swaps
    # the clutter back to canonical (data-side, no mj_forward, so
    # physics never sees the swap; the stand-ins are contype 0
    # anyway).
    V3_ABSENT_POS = (2.5, 0.0, -1.0)  # outside both frusta
    V3_YAW_JITTER: ClassVar[dict[str, float]] = {"mouse": 0.3, "laptop": 0.15}

    def _init_bank(self, width: int, height: int) -> None:
        import json

        from PIL import Image

        bank_dir = self.PLATES_DIR / "bank"
        plates = sorted(bank_dir.glob("top_ep*.png"))
        if not plates:
            raise ValueError(f"no bank plates under {bank_dir}")
        manifest = json.loads((bank_dir / "bank_manifest.json").read_text())
        # Each entry: (plate, episode gain, episode bias) — the affine
        # photometric state the mining pass fitted from the global
        # plate to this episode; the composite applies it to the
        # rendered foreground too, so foreground and background share
        # the episode's lighting (iteration 2 of the registered
        # composite loop: as iteration 1 the foreground kept the fixed
        # global grade under a varying plate).
        self._bank: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for path in plates:
            plate = np.asarray(Image.open(path), dtype=np.float64)
            if plate.shape != (height, width, 3):
                raise ValueError(f"{path.name} {plate.shape} != render size")
            episode = manifest["episodes"][str(int(path.stem.removeprefix("top_ep")))]
            self._bank.append(
                (plate, np.array(episode["gain"]), np.array(episode["bias"])),
            )
        self._clutter_ranges = manifest["clutter_ranges"]
        self._clutter_base: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for name in self.V2_DYNAMIC_STATICS[1:]:  # all but the disk
            geom = self.model.geom(name)
            quat = self.model.geom_quat[geom.id].copy()
            yaw = 2.0 * float(np.arctan2(quat[3], quat[0]))
            self._clutter_base[name] = (geom.pos.copy(), quat, yaw)
        self._active_top_plate = self._plates["top"]
        self._active_gain = np.ones(3)
        self._active_bias = np.zeros(3)
        # name -> (pos, yaw) as drawn for the current reset
        self._clutter_drawn: dict[str, tuple[np.ndarray, float]] = {}

    def _draw_content(self, rng: np.random.Generator) -> None:
        """Per-reset plate + clutter draws (v3). Writes the drawn
        poses into the model (physics-inert: contype 0) and remembers
        them for the per-render canonical swap."""
        plate, gain, bias = self._bank[int(rng.integers(len(self._bank)))]
        self._active_top_plate = plate
        self._active_gain = gain
        self._active_bias = bias
        self._clutter_drawn = {}
        for name, (base_pos, _, base_yaw) in self._clutter_base.items():
            spec = self._clutter_ranges[name]
            pos = base_pos.copy()
            yaw = base_yaw
            if spec["mode"] == "fixed_canonical":
                pass
            elif rng.uniform() >= spec["presence"]:
                pos = np.array(self.V3_ABSENT_POS)
            else:
                lo, hi = spec["xy_min"], spec["xy_max"]
                if spec["mode"] == "delta_about_canonical":
                    lo, hi = spec["xy_delta_min"], spec["xy_delta_max"]
                    xy = base_pos[:2] + rng.uniform(lo, hi)
                else:
                    xy = rng.uniform(lo, hi)
                pos[:2] = xy
                jitter = self.V3_YAW_JITTER.get(name)
                if jitter is not None:
                    yaw = base_yaw + float(rng.uniform(-jitter, jitter))
            self._clutter_drawn[name] = (pos, yaw)
            gid = self.model.geom(name).id
            self.model.geom_pos[gid] = pos
            half = yaw / 2.0
            self.model.geom_quat[gid] = (np.cos(half), 0.0, 0.0, np.sin(half))

    def _set_clutter(self, *, drawn: bool) -> None:
        """Point the RENDER state (data.geom_xpos/xmat) at the drawn
        or the canonical clutter poses — the wrist view renders the
        canonical scene so its frames stay bit-identical to v2."""
        for name, (base_pos, _, base_yaw) in self._clutter_base.items():
            pos, yaw = self._clutter_drawn[name] if drawn else (base_pos, base_yaw)
            gid = self.model.geom(name).id
            self.data.geom_xpos[gid] = pos
            c, s = np.cos(yaw), np.sin(yaw)
            self.data.geom_xmat[gid] = (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)

    def _render_mask(self, camera: str) -> np.ndarray:
        """[H, W] float 0/1 dynamic-content mask in source pinhole
        space, from a segmentation pass of the same scene."""
        renderer = self.renderer
        renderer.enable_segmentation_rendering()
        renderer.update_scene(self.data, camera=camera)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        return (is_geom & np.isin(seg[..., 0], self._dynamic_geoms)).astype(
            np.float64,
        )

    def _composite(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        camera: str,
    ) -> np.ndarray:
        """Inpainting composite: graded/blurred rendered foreground over
        the real clean plate, sensor noise on the full frame (the
        median plate is denoised below single-frame noise; the plate
        already carries the real optics, so only the foreground gets
        the PSF blur)."""
        gain, bias = self.V1_GRADE[camera]
        foreground = self._remap(frame.astype(np.float64)) * gain + bias
        if self.render_style == "v3" and camera == "top":
            # The drawn plate's photometric state (global -> episode
            # affine, from the bank manifest) applies to the rendered
            # foreground too: composite lighting stays coherent and
            # varies per reset like the real episodes do.
            foreground = foreground * self._active_gain + self._active_bias
        foreground = self._blur(foreground)
        weight = self._blur(self._remap(mask[..., None]))
        weight = np.clip(weight, 0.0, 1.0)
        plate = (
            self._active_top_plate
            if self.render_style == "v3" and camera == "top"
            else self._plates[camera]
        )
        out = weight * foreground + (1.0 - weight) * plate
        out += self._noise_rng.normal(0.0, self.V1_NOISE_SIGMA, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

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

    def reset(self, seed: int, appearance_seed: int | None = None) -> SimObservation:
        """Home the arm, place benchy at a seeded pose, randomize
        appearance (benchy tint, lighting), settle physics until
        contacts are quiet.

        The arm settles FIRST, with the benchy parked outside its sweep
        (mj_resetData lays the arm out over the workspace, and driving up
        to home used to strike an already-spawned boat on ~10% of seeds,
        displacing it up to 30 mm); the benchy is placed at its seeded
        pose only after the arm is home, then given a short settle of its
        own. `reset_strike_contacts` counts gripper-benchy contacts seen
        during the whole reset - 0 for every seed is an eval-protocol
        gate.

        Appearance draws come from their own RNG stream
        (``appearance_seed``, defaulting to ``seed``): spawn draws keep
        the original stream and order, so seed -> benchy pose -> settled
        qpos is bit-identical across appearance seeds (oracle-pinned in
        tests/test_sim_appearance.py)."""
        rng = np.random.default_rng(seed)
        looks = np.random.default_rng(
            seed if appearance_seed is None else appearance_seed,
        )
        mujoco.mj_resetData(self.model, self.data)

        x = rng.uniform(*SPAWN_X)
        y = rng.uniform(*SPAWN_Y)
        yaw = rng.uniform(-np.pi, np.pi)
        quat = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        self.reset_spawn_xy: tuple[float, float] = (float(x), float(y))

        # Benchy tint: the real print is a consistent light gray; keep a
        # narrow band around it with a mild per-channel cast (the old
        # wide color draw made the boat a random-colored outlier).
        base = looks.uniform(0.72, 0.92)
        rgba = np.append(
            np.clip(base + looks.uniform(-0.06, 0.04, size=3), 0.05, 1.0),
            1.0,
        )
        self.model.mat_rgba[self._benchy_mat] = rgba
        self._jitter_appearance(looks)
        # Sensor-noise stream for this episode: seeded from the
        # appearance stream, fresh draw every observe().
        self._noise_rng = np.random.default_rng(looks.integers(2**63))
        # v3 content draws come LAST on the appearance stream: every
        # draw the v0-v2 styles make is stream-identical, so the v3
        # wrist path is bit-identical to v2 (registered guard).
        if self.render_style == "v3":
            self._draw_content(looks)

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

    def _jitter_appearance(self, rng: np.random.Generator) -> None:
        """Per-reset lighting/tone variation, matched to the real frames'
        episode-to-episode spread (window daylight: direction, intensity,
        color temperature). The encoder OOD probe measured sim renders 7x
        too homogeneous — this is the diversity axis. Multiplies the
        stored scene baselines, so repeated resets never compound."""
        # Window daylight direction: jitter the sun's tilt around its
        # baseline (straight down) — a gentle off-vertical swing moves
        # highlights/shading like the real window light does.
        tilt = rng.uniform(0.0, 0.45)
        azimuth = rng.uniform(-np.pi, np.pi)
        direction = np.array(
            [
                np.sin(tilt) * np.cos(azimuth),
                np.sin(tilt) * np.sin(azimuth),
                -np.cos(tilt),
            ],
        )
        self.model.light_dir[self._sun] = direction
        # Intensity and color temperature: warm <-> cool around baseline.
        gain = rng.uniform(0.85, 1.18)
        warmth = rng.uniform(-0.05, 0.07)
        temp = np.array([1.0 + warmth, 1.0, 1.0 - warmth])
        self.model.light_diffuse[self._sun] = self._base_sun_diffuse * gain * temp
        self.model.light_diffuse[self._fill] = self._base_fill_diffuse * rng.uniform(
            0.8,
            1.2,
        )
        # Table tone rides the same daylight variation slightly.
        table = self._base_table_rgba.copy()
        table[:3] *= rng.uniform(0.94, 1.06)
        self.model.mat_rgba[self._table_mat] = np.clip(table, 0.0, 1.0)

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

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            height, width = self._render_size
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def observe(self) -> SimObservation:
        state = np.rad2deg(self.data.qpos[self._joint_qpos])
        self.renderer.update_scene(self.data, camera="top_cam")
        top = self.renderer.render()
        if self.render_style == "v3":
            # Composite (and its segmentation mask) BEFORE the swap:
            # drawn clutter is masked at its drawn poses; the wrist
            # then renders the canonical scene (bit-identical to v2).
            top = self._composite(top, self._render_mask("top_cam"), "top")
            self._set_clutter(drawn=False)
        self.renderer.update_scene(self.data, camera="wrist_cam")
        wrist = self.renderer.render()
        if self.render_style == "v3":
            self._set_clutter(drawn=True)
            wrist = self._grade(self._apply_fisheye(wrist), "wrist")
        elif self.render_style == "v1":
            top = self._grade(self._apply_fisheye(top), "top")
            wrist = self._grade(self._apply_fisheye(wrist), "wrist")
        elif self.render_style == "v2":
            top = self._composite(top, self._render_mask("top_cam"), "top")
            # Wrist keeps the v1 render path (v3 included): the wrist
            # plate is a cross-episode mush (start-pose viewpoints
            # differ by degrees between episodes) and the wrist
            # composite READ WORSE than the v1 path on the pinned
            # probe (5-NN AUROC 0.951 vs 0.900, 08-12; pure-composite
            # read reproducible at commit f75c341). Real fix was the
            # wrist periphery re-tune (0.548, 08-12).
            wrist = self._grade(self._apply_fisheye(wrist), "wrist")
        return SimObservation(top=top, wrist=wrist, state=state)

    def _grade(self, frame: np.ndarray, camera: str) -> np.ndarray:
        gain, bias = self.V1_GRADE[camera]
        graded = frame.astype(np.float64) * gain + bias
        graded = self._blur(graded)
        graded += self._noise_rng.normal(0.0, self.V1_NOISE_SIGMA, graded.shape)
        return np.clip(graded, 0, 255).astype(np.uint8)

    def _blur(self, frame: np.ndarray) -> np.ndarray:
        """Separable Gaussian PSF, [H, W, 3] float in/out."""
        radius = max(1, int(np.ceil(2.5 * self.V1_BLUR_SIGMA)))
        taps = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (taps / self.V1_BLUR_SIGMA) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(frame, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        rows = np.zeros_like(frame)
        for i, k in enumerate(kernel):
            rows += padded[i : i + frame.shape[0]] * k
        padded = np.pad(rows, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        out = np.zeros_like(frame)
        for i, k in enumerate(kernel):
            out += padded[:, i : i + frame.shape[1]] * k
        return out

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
