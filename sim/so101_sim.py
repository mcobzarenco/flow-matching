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
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch as _torch_types

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


class _TorchPost:
    """CUDA implementation of the per-tick image post-processing
    (fisheye remap, PSF blur, grade/composite arithmetic) — the
    rollout profile is render-bound on the numpy reference path
    (~0.3 s/tick, owner-approved port 08:12Z 2026-08-12). Numerics
    are float32 on GPU vs the reference's float64; frames may differ
    by ±2/255 counts (oracle-pinned in tests/test_sim_appearance.py).
    The sensor-noise stream stays on the seeded numpy RNG."""

    def __init__(
        self,
        fisheye: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        blur_sigma: float,
        wrist_grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        import torch

        self._torch = torch
        self.device = torch.device("cuda")
        x0, y0, wx, wy = fisheye
        self.x0 = torch.from_numpy(x0).to(self.device)
        self.y0 = torch.from_numpy(y0).to(self.device)
        self.wx = torch.from_numpy(wx.astype(np.float32)).to(self.device)
        self.wy = torch.from_numpy(wy.astype(np.float32)).to(self.device)
        # Fitted-lens wrist map (cubemap source); None on the deployed
        # equidistant path, where remap() serves both cameras.
        self.wrist: (
            tuple[
                _torch_types.Tensor,
                _torch_types.Tensor,
                _torch_types.Tensor,
                _torch_types.Tensor,
            ]
            | None
        ) = None
        if wrist_grid is not None:
            gx0, gy0, gwx, gwy = wrist_grid
            self.wrist = (
                torch.from_numpy(gx0).to(self.device),
                torch.from_numpy(gy0).to(self.device),
                torch.from_numpy(gwx.astype(np.float32)).to(self.device),
                torch.from_numpy(gwy.astype(np.float32)).to(self.device),
            )
        radius = max(1, int(np.ceil(2.5 * blur_sigma)))
        taps = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (taps / blur_sigma) ** 2)
        kernel /= kernel.sum()
        self.radius = radius
        self.kernel = torch.from_numpy(kernel.astype(np.float32)).to(self.device)
        self._cache: dict[int, _torch_types.Tensor] = {}

    def upload(self, array: np.ndarray) -> _torch_types.Tensor:
        """[H, W, C] float numpy -> cached float32 CUDA tensor."""
        key = id(array)
        if key not in self._cache:
            self._cache[key] = self._torch.from_numpy(
                np.ascontiguousarray(array, dtype=np.float32),
            ).to(self.device)
        return self._cache[key]

    def frame(self, frame: np.ndarray) -> _torch_types.Tensor:
        """[H, W, C] uint8/float numpy -> float32 CUDA tensor (no cache)."""
        return self._torch.from_numpy(
            np.ascontiguousarray(frame, dtype=np.float32),
        ).to(self.device)

    def remap(self, src: _torch_types.Tensor) -> _torch_types.Tensor:
        """Bilinear fisheye remap, [H, W, C] float32 tensor in/out."""
        return self._gather(src, self.x0, self.y0, self.wx, self.wy)

    def remap_wrist(self, src: _torch_types.Tensor) -> _torch_types.Tensor:
        """Fitted-lens wrist remap — src is the vertically concatenated
        cubemap face stack [F*S, S, C]."""
        assert self.wrist is not None
        x0, y0, wx, wy = self.wrist
        return self._gather(src, x0, y0, wx, wy)

    def _gather(
        self,
        src: _torch_types.Tensor,
        x0: _torch_types.Tensor,
        y0: _torch_types.Tensor,
        wx: _torch_types.Tensor,
        wy: _torch_types.Tensor,
    ) -> _torch_types.Tensor:
        top = src[y0, x0] * (1 - wx) + src[y0, x0 + 1] * wx
        bottom = src[y0 + 1, x0] * (1 - wx) + src[y0 + 1, x0 + 1] * wx
        return top * (1 - wy) + bottom * wy

    def blur(self, image: _torch_types.Tensor) -> _torch_types.Tensor:
        """Separable Gaussian PSF with edge padding, [H, W, C]
        float32 tensor in/out — mirrors the numpy reference."""
        torch = self._torch
        channels = image.shape[-1]
        t = image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        weight_h = self.kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        weight_w = self.kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        t = torch.nn.functional.pad(
            t,
            (0, 0, self.radius, self.radius),
            mode="replicate",
        )
        t = torch.nn.functional.conv2d(t, weight_h, groups=channels)
        t = torch.nn.functional.pad(
            t,
            (self.radius, self.radius, 0, 0),
            mode="replicate",
        )
        t = torch.nn.functional.conv2d(t, weight_w, groups=channels)
        return t.squeeze(0).permute(1, 2, 0)

    def to_uint8(self, image: _torch_types.Tensor) -> np.ndarray:
        return image.clamp(0, 255).to(self._torch.uint8).cpu().numpy()


@dataclass(frozen=True, slots=True)
class _WristLens:
    """Precomputed fitted-lens wrist render state (lens_model="fitted"):
    the output->face-stack bilinear map, the camera-local face
    rotations to render (only faces the map references), and the face
    frustum geometry."""

    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    face_quats: tuple[np.ndarray, ...]
    face_size: int
    fovy: float


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
        post_backend: str = "auto",
        *,
        lens_model: str = "equidistant",
        flip_camera_mount: bool = True,
    ) -> None:
        if render_style not in ("v0", "v1", "v2", "v3", "v4"):
            raise ValueError(
                f"render_style {render_style!r} not in ('v0', 'v1', 'v2', 'v3', 'v4')",
            )
        if post_backend not in ("auto", "numpy", "torch"):
            raise ValueError(
                f"post_backend {post_backend!r} not in ('auto', 'numpy', 'torch')",
            )
        if lens_model not in ("equidistant", "fitted"):
            raise ValueError(
                f"lens_model {lens_model!r} not in ('equidistant', 'fitted')",
            )
        if lens_model == "fitted" and render_style == "v0":
            raise ValueError("lens_model 'fitted' needs a fisheye style (v1..v4)")
        self.render_style = render_style
        self.post_backend = post_backend
        self.lens_model = lens_model
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
        # flip_camera_mount=False reproduces the pre-flip (mirrored
        # Menagerie bracket) physics for paired flip-effect reads only —
        # the flipped mount is the registered geometry (d5cf9fd).
        if flip_camera_mount:
            self._flip_camera_mount()
        self._noise_rng = np.random.default_rng(0)  # re-seeded per reset
        # Fitted-lens wrist path (leg (b)); None on the deployed
        # equidistant path. Set by _init_wrist_fitted_lens.
        self._wrist_lens: _WristLens | None = None
        self._face_renderer: mujoco.Renderer | None = None
        if self.render_style in ("v1", "v2", "v3", "v4"):
            self._init_fisheye(width, height)
            if self.lens_model == "fitted":
                self._init_wrist_fitted_lens(width, height)
        if self.render_style in ("v2", "v3", "v4"):
            self._init_inpainting(width, height)
        if self.render_style in ("v3", "v4"):
            self._init_bank(width, height)
        # GPU post-processing (owner-approved 08:12Z 08-12): rollouts
        # are render-bound on the numpy path; "auto" takes CUDA when
        # available. Physics is untouched either way; frames may
        # differ from the reference by +-2/255 counts (oracle-pinned).
        self._post: _TorchPost | None = None
        if self.render_style != "v0" and post_backend != "numpy":
            try:
                import torch

                cuda = torch.cuda.is_available()
            except ImportError:
                cuda = False
            if cuda:
                self._post = _TorchPost(
                    self._fisheye,
                    self.V1_BLUR_SIGMA,
                    wrist_grid=(
                        None if self._wrist_lens is None else self._wrist_lens.grid
                    ),
                )
            elif post_backend == "torch":
                raise ValueError("post_backend='torch' needs CUDA torch")

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

    def _flip_camera_mount(self) -> None:
        """Menagerie mounts the wrist camera BRACKET mirrored, 180 deg
        from the real assembly (owner spot 2026-08-12 from rollout
        videos; the real bracket starts rotated toward the ceiling).
        Probe-measured: at the settled home the bracket hangs on the JAW
        side 40 mm over the table, its volume dips below the table on
        31.9% of the 26 reference episodes' recorded REAL poses (center
        to -46 mm), and dynamic replays grind bracket-table contact on
        up to 22% of ticks - physically blocking poses the real arm
        demonstrably reaches (~62% of the residual replay control loss).

        Fix at load (runtime, vendored XML untouched, both arms): rotate
        the mount's geoms - visual mesh + camera_box1/2 - 180 deg about
        the mount-local x axis, which lands the bracket around the
        already-re-posed camera view (that pose was fit to real frames,
        so it marks where the real bracket holds the module). The camera
        itself is posed independently above and is NOT touched. Known
        residual: body inertia was compiled with the mount mass (12 g)
        on the old side; runtime geom moves do not recompile it."""
        for prefix in ("", "leader-"):
            body = self.model.body(prefix + "camera_mount").id
            flip = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x
            for geom in range(self.model.ngeom):
                if self.model.geom_bodyid[geom] != body:
                    continue
                self.model.geom_pos[geom, 1:] *= -1.0
                rotated = np.empty(4)
                mujoco.mju_mulQuat(rotated, flip, self.model.geom_quat[geom])
                self.model.geom_quat[geom] = rotated
                # The compiler marks geoms whose frame coincides with the
                # body/inertial frame with a sameframe fast path that makes
                # mj_kinematics IGNORE geom_pos/geom_quat (the visual mesh
                # rode it, so renders kept the bracket on the old side —
                # owner spot 2026-08-12 16:07Z; physics was unaffected: the
                # mesh is non-colliding and the one affected box's skipped
                # rotation is a 180-deg self-symmetry). Clear it so the
                # edited local pose actually takes effect.
                self.model.geom_sameframe[geom] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE

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
        return self._remap_grid(src, self._fisheye)

    @staticmethod
    def _remap_grid(
        src: np.ndarray,
        grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        x0, y0, wx, wy = grid
        top = src[y0, x0] * (1 - wx) + src[y0, x0 + 1] * wx
        bottom = src[y0 + 1, x0] * (1 - wx) + src[y0 + 1, x0 + 1] * wx
        return top * (1 - wy) + bottom * wy

    def _apply_fisheye(self, frame: np.ndarray) -> np.ndarray:
        """Bilinear remap of a source pinhole render into the distorted
        output — [H, W, 3] uint8 in, same out."""
        if self._post is not None:
            return self._post.to_uint8(self._post.remap(self._post.frame(frame)))
        remapped = self._remap(frame.astype(np.float64))
        return np.clip(remapped, 0, 255).astype(np.uint8)

    # Fitted wrist lens (sim-fit-real-lens-model leg (b), lit 0823
    # papers/fisheye-lens-fitting.md, owner-adopted 22:31Z 08-12): the
    # deployed v1 warp assumes an IDEAL equidistant lens centered at
    # the image midpoint; the plumb-line fit on the 150 pinned real
    # wrist frames (fontaine/scripts/fit_lens_plumbline.py, leg (a),
    # outputs/sim/lens_fit/wrist_lens_fit.json) measures the real
    # module's optical center 22 px left / 14 px below the midpoint
    # (~5 sigma) and a stronger peripheral compression:
    #   theta = rho * (1 + k2 rho^2 + k4 rho^4),  rho = r_px / F_DIST
    # with F_DIST the V1_CENTER_FOVY pinhole focal (492.07 at 480p —
    # center magnification unchanged, same anchor as v1). The source
    # is no longer a single 72-deg pinhole (whose diagonal the fitted
    # corner rays overrun) but a pinhole CUBEMAP around the wrist
    # camera axis (2603.02139's MuJoCo recipe with the equirect stage
    # composed away): the output->face map is precomputed once, so
    # runtime cost is one bilinear gather plus one extra ~squarish
    # render per referenced face. Faces span LENS_FACE_HALF_DEG=46 deg
    # per half-axis with faces selected by dominant ray component
    # (boundary 45 deg), so every bilinear footprint sits >=1 deg
    # inside its face — no seam pixels. Face size is chosen so the
    # face focal matches the deployed 72-deg source focal: center
    # sharpness is unchanged and a fitted-vs-deployed A/B reads lens
    # geometry, not resolution. Wrist only — the top plate composite
    # already carries the real top lens (bit-identical oracle).
    #
    # Pinned params are the fit's CURVE-ONLY refit (center at the
    # image midpoint), not the full fit: the 08-13 paired gate probe
    # (20 seeds x 5 draws, reports/analysis__sim_encoder_ood_probe_
    # lensgate_*_arm.json) read the full fit at wrist knn5 AUROC 0.667
    # vs 0.560 control with the CENTER component alone reproducing the
    # whole regression (0.672) — the principal-point shift is
    # degenerate with the 08-12 wrist pose re-tune, which was fit to
    # real frames under the deployed lens and already absorbed it.
    # The curve-only refit passes the registered gate: AUROC 0.523,
    # paired dknn5 -7.6e-07 CI95 [-8.5e-07, -6.8e-07], 96/100 frames
    # closer to real. Full-fit params (plank residual 0.898 px vs
    # 0.937 curve-only) stay recorded in wrist_lens_fit.json; using
    # them requires a JOINT pose+lens refit first.
    WRIST_LENS_FIT: ClassVar[dict[str, float]] = {
        "cx": 319.5,
        "cy": 239.5,
        "k2": 0.1013541981621529,
        "k4": -0.035671567335666635,
    }
    LENS_FACE_HALF_DEG = 46.0

    # Camera-local face rotations (MuJoCo cam frame: x right, y up,
    # looking along -z; w-first quats). Rotating the camera by q makes
    # a base-frame ray d appear in the face frame as R(q)^T d.
    _COS45 = float(np.sqrt(0.5))
    LENS_FACES: ClassVar[dict[str, tuple[float, float, float, float]]] = {
        "front": (1.0, 0.0, 0.0, 0.0),
        "right": (_COS45, 0.0, -_COS45, 0.0),
        "left": (_COS45, 0.0, _COS45, 0.0),
        "up": (_COS45, _COS45, 0.0, 0.0),
        "down": (_COS45, -_COS45, 0.0, 0.0),
    }

    def _init_wrist_fitted_lens(self, width: int, height: int) -> None:
        """Precompute the output-pixel -> cubemap-face bilinear map for
        the fitted wrist lens, keep only the faces the map references,
        and point the wrist camera's fovy at the face frustum (the
        equidistant map set by _init_fisheye keeps serving the top
        cam). The fit is in 640x480 real-frame pixels; center and
        focal scale with the render size."""
        fit = self.WRIST_LENS_FIT
        cx = fit["cx"] * width / 640.0
        cy = fit["cy"] * height / 480.0
        f_dist = (height / 2.0) / np.tan(np.deg2rad(self.V1_CENTER_FOVY) / 2.0)
        f_src = (height / 2.0) / np.tan(np.deg2rad(self.V1_SRC_FOVY) / 2.0)
        half = np.deg2rad(self.LENS_FACE_HALF_DEG)
        size = 2 * round(float(f_src * np.tan(half)))  # face focal ~= f_src
        f_face = (size / 2.0) / np.tan(half)

        v, u = np.mgrid[0:height, 0:width].astype(np.float64)
        x = u - cx
        y = v - cy
        r = np.hypot(x, y)
        rho = r / f_dist
        theta = rho * (1 + fit["k2"] * rho**2 + fit["k4"] * rho**4)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(r > 0, x / r, 0.0)
            uy = np.where(r > 0, y / r, 0.0)
        # Ray in the base camera frame (image down = -y_cam; the
        # convention verified in sim/shadow.py).
        d = np.stack([np.sin(theta) * ux, -np.sin(theta) * uy, -np.cos(theta)])
        rotations = {}
        for name, quat in self.LENS_FACES.items():
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, np.array(quat))
            rotations[name] = mat.reshape(3, 3)
        names = list(self.LENS_FACES)
        # d in each face's frame, [F, 3, H*W]; the face whose axis is
        # nearest the ray (max forward component) always contains its
        # bilinear footprint: the chosen -z_face is the largest ray
        # component, so both tangents are <= tan(45 deg) < tan(half).
        face_d = np.stack(
            [rotations[name].T @ d.reshape(3, -1) for name in names],
        )
        forward = -face_d[:, 2]
        chosen = np.argmax(forward, axis=0)
        pick = face_d[chosen, :, np.arange(chosen.size)]  # [H*W, 3]
        depth = -pick[:, 2]
        center = (size - 1) / 2.0
        sx = (center + f_face * pick[:, 0] / depth).reshape(height, width)
        sy = (center - f_face * pick[:, 1] / depth).reshape(height, width)
        used = sorted(set(chosen.tolist()))
        slot = np.full(len(names), -1)
        slot[used] = np.arange(len(used))
        x0 = np.clip(np.floor(sx).astype(np.int64), 0, size - 2)
        y0 = np.clip(np.floor(sy).astype(np.int64), 0, size - 2)
        wx = np.clip(sx - x0, 0.0, 1.0)[..., None]
        wy = np.clip(sy - y0, 0.0, 1.0)[..., None]
        y0 += slot[chosen].reshape(height, width) * size
        # The vendored XML sizes the offscreen framebuffer for 640x480;
        # face renders are square and slightly wider (runtime override,
        # same convention as the servo/limit edits — set before any
        # Renderer exists, both lazy).
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, size)
        self.model.vis.global_.offheight = max(
            self.model.vis.global_.offheight,
            size,
        )
        self._wrist_lens = _WristLens(
            grid=(x0, y0, wx, wy),
            face_quats=tuple(np.array(self.LENS_FACES[names[i]]) for i in used),
            face_size=size,
            fovy=2.0 * self.LENS_FACE_HALF_DEG,
        )
        self.model.camera("wrist_cam").fovy[0] = self._wrist_lens.fovy

    @property
    def face_renderer(self) -> mujoco.Renderer:
        assert self._wrist_lens is not None
        if self._face_renderer is None:
            size = self._wrist_lens.face_size
            self._face_renderer = mujoco.Renderer(self.model, height=size, width=size)
        return self._face_renderer

    def _render_wrist_source(self) -> np.ndarray:
        """The wrist camera's source render: the single wide pinhole on
        the deployed path, or the vertically concatenated cubemap face
        stack on the fitted path (camera re-aimed per face through
        model.cam_quat + mj_camlight — pure camera state, physics and
        the RNG streams untouched; base pose restored after)."""
        if self._wrist_lens is None:
            self.renderer.update_scene(self.data, camera="wrist_cam")
            return self.renderer.render()
        cam = self.model.camera("wrist_cam")
        cam_id = self.model.camera("wrist_cam").id
        # The scene's headlight rides the render camera, so face
        # renders would each be lit from their own axis — a shading
        # seam at face boundaries. The real light of the deployed path
        # is the headlight along the BASE wrist axis: re-point every
        # face's headlight there (specular is 0 in this scene, so the
        # light direction is the only view-dependent shading term).
        base_forward = -self.data.cam_xmat[cam_id].reshape(3, 3)[:, 2].copy()
        base = cam.quat.copy()
        faces = []
        for quat in self._wrist_lens.face_quats:
            aimed = np.empty(4)
            mujoco.mju_mulQuat(aimed, base, quat)
            cam.quat[:] = aimed
            mujoco.mj_camlight(self.model, self.data)
            self.face_renderer.update_scene(self.data, camera="wrist_cam")
            scene = self.face_renderer.scene
            for index in range(scene.nlight):
                if scene.lights[index].headlight:
                    scene.lights[index].dir[:] = base_forward
            faces.append(self.face_renderer.render())
        cam.quat[:] = base
        mujoco.mj_camlight(self.model, self.data)
        return np.concatenate(faces, axis=0)

    def _apply_wrist_lens(self, frame: np.ndarray) -> np.ndarray:
        """Wrist source -> distorted output ([H, W, 3] uint8): the
        fitted-lens gather when active, else the deployed fisheye."""
        if self._wrist_lens is None:
            return self._apply_fisheye(frame)
        if self._post is not None:
            return self._post.to_uint8(
                self._post.remap_wrist(self._post.frame(frame)),
            )
        remapped = self._remap_grid(frame.astype(np.float64), self._wrist_lens.grid)
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

    # v4 render style (contact shadows, sim-composite-contact-shadows
    # leg (a), lit 0823 papers/composite-shadows.md): the v3 composite,
    # plus the one physics law every real frame obeys that no composite
    # frame does — the arm darkens the table under it. The dynamic
    # content's camera-visible pixels are slid along the fitted dominant
    # light direction onto the table plane and splatted as a soft
    # occupancy map (sim.shadow) that multiply-darkens the top plate.
    # All three constants are measured from the real episodes' own arm
    # shadows (fontaine/scripts/fit_contact_shadow.py, 200 frames of 25
    # bank episodes, reports/analysis__contact_shadow_fit.json):
    # direction zenith 30 deg / azimuth 112.5 deg (light travels toward
    # -x/+y; 85% of frame-bootstrap resamples), darkening contrast
    # +0.091 CI95 [0.081, 0.100] at the optimum vs a ring control,
    # strength 0.392 CI95 [0.364, 0.419] by least squares through the
    # origin, softness argmax of corr(map, darkening) over the sigma
    # grid. The wrist path is untouched (shadow applies to the top
    # composite only) and the pass consumes no RNG draws, so v4 wrist
    # frames are bit-identical to v3 (oracle-pinned).
    V4_LIGHT_DIR = (-0.19134172, 0.46193977, -0.8660254)
    V4_SHADOW_STRENGTH = 0.392
    V4_SHADOW_SIGMA_PX = 24.0

    def _render_shadow(self, camera: str, dynamic_mask: np.ndarray) -> np.ndarray:
        """[H, W] soft contact-shadow map in source pinhole space (one
        extra depth pass; the projection itself is numpy on both post
        backends — the map is remapped/applied inside _composite)."""
        from sim.shadow import shadow_map

        renderer = self.renderer
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        cam_id = self.model.camera(camera).id
        height, width = self._render_size
        focal = (height / 2.0) / np.tan(
            np.deg2rad(float(self.model.cam_fovy[cam_id])) / 2.0,
        )
        table = self.model.geom("table")
        plane_z = float(table.pos[2] + table.size[2])
        bounds = (
            float(table.pos[0] - table.size[0]),
            float(table.pos[0] + table.size[0]),
            float(table.pos[1] - table.size[1]),
            float(table.pos[1] + table.size[1]),
        )
        return shadow_map(
            depth,
            dynamic_mask,
            (focal, (width - 1) / 2.0, (height - 1) / 2.0),
            self.data.cam_xpos[cam_id].copy(),
            self.data.cam_xmat[cam_id].reshape(3, 3).copy(),
            np.array(self.V4_LIGHT_DIR),
            self.V4_SHADOW_SIGMA_PX,
            plane_z=plane_z,
            bounds_xy=bounds,
            max_points=20000,
        )

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
        shadow: np.ndarray | None = None,
    ) -> np.ndarray:
        """Inpainting composite: graded/blurred rendered foreground over
        the real clean plate, sensor noise on the full frame (the
        median plate is denoised below single-frame noise; the plate
        already carries the real optics, so only the foreground gets
        the PSF blur). ``shadow`` (v4, source pinhole space) multiply-
        darkens the plate contribution only — where the foreground
        covers a pixel its own render wins, like the real arm's body
        hides the table it darkens."""
        gain, bias = self.V1_GRADE[camera]
        episode_affine = self.render_style in ("v3", "v4") and camera == "top"
        plate = (
            self._active_top_plate
            if self.render_style in ("v3", "v4") and camera == "top"
            else self._plates[camera]
        )
        noise = (
            self._noise_rng.standard_normal((*mask.shape, 3), dtype=np.float32)
            * self.V1_NOISE_SIGMA
        )
        if self._post is not None:
            post = self._post
            foreground = post.remap(post.frame(frame)) * post.frame(
                np.array(gain),
            ) + post.frame(np.array(bias))
            if episode_affine:
                foreground = foreground * post.frame(
                    self._active_gain,
                ) + post.frame(self._active_bias)
            foreground = post.blur(foreground)
            weight = post.blur(post.remap(post.frame(mask[..., None])))
            weight = weight.clamp(0.0, 1.0)
            plate_term = post.upload(plate)
            if shadow is not None:
                plate_term = plate_term * (
                    1.0
                    - self.V4_SHADOW_STRENGTH
                    * post.remap(post.frame(shadow[..., None]))
                )
            out = weight * foreground + (1.0 - weight) * plate_term
            out = out + post.frame(noise)
            return post.to_uint8(out)
        foreground = self._remap(frame.astype(np.float64)) * gain + bias
        if episode_affine:
            # The drawn plate's photometric state (global -> episode
            # affine, from the bank manifest) applies to the rendered
            # foreground too: composite lighting stays coherent and
            # varies per reset like the real episodes do.
            foreground = foreground * self._active_gain + self._active_bias
        foreground = self._blur(foreground)
        weight = self._blur(self._remap(mask[..., None]))
        weight = np.clip(weight, 0.0, 1.0)
        plate_term = plate
        if shadow is not None:
            plate_term = plate * (
                1.0 - self.V4_SHADOW_STRENGTH * self._remap(shadow[..., None])
            )
        out = weight * foreground + (1.0 - weight) * plate_term
        out += noise
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
        # wrist path is bit-identical to v2 (registered guard). v4
        # adds NO draws on top of v3 (the shadow pass is deterministic),
        # so v4 spawn + content state is bit-identical to v3.
        if self.render_style in ("v3", "v4"):
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
        if self.render_style in ("v3", "v4"):
            # Composite (and its segmentation mask) BEFORE the swap:
            # drawn clutter is masked at its drawn poses; the wrist
            # then renders the canonical scene (bit-identical to v2).
            mask = self._render_mask("top_cam")
            shadow = (
                self._render_shadow("top_cam", mask)
                if self.render_style == "v4"
                else None
            )
            top = self._composite(top, mask, "top", shadow=shadow)
            self._set_clutter(drawn=False)
        wrist = self._render_wrist_source()
        if self.render_style in ("v3", "v4"):
            self._set_clutter(drawn=True)
            wrist = self._grade(self._apply_wrist_lens(wrist), "wrist")
        elif self.render_style == "v1":
            top = self._grade(self._apply_fisheye(top), "top")
            wrist = self._grade(self._apply_wrist_lens(wrist), "wrist")
        elif self.render_style == "v2":
            top = self._composite(top, self._render_mask("top_cam"), "top")
            # Wrist keeps the v1 render path (v3 included): the wrist
            # plate is a cross-episode mush (start-pose viewpoints
            # differ by degrees between episodes) and the wrist
            # composite READ WORSE than the v1 path on the pinned
            # probe (5-NN AUROC 0.951 vs 0.900, 08-12; pure-composite
            # read reproducible at commit f75c341). Real fix was the
            # wrist periphery re-tune (0.548, 08-12).
            wrist = self._grade(self._apply_wrist_lens(wrist), "wrist")
        return SimObservation(top=top, wrist=wrist, state=state)

    def _grade(self, frame: np.ndarray, camera: str) -> np.ndarray:
        gain, bias = self.V1_GRADE[camera]
        noise = (
            self._noise_rng.standard_normal(frame.shape, dtype=np.float32)
            * self.V1_NOISE_SIGMA
        )
        if self._post is not None:
            post = self._post
            graded = post.frame(frame) * post.frame(np.array(gain)) + post.frame(
                np.array(bias),
            )
            graded = post.blur(graded) + post.frame(noise)
            return post.to_uint8(graded)
        graded = frame.astype(np.float64) * gain + bias
        graded = self._blur(graded)
        graded += noise
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
