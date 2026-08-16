"""Privileged scripted expert for the benchy pick-place scene — the
grasp-rich SFT bootstrap's demonstration source (pre-reg DRAFT
posts/2026-08-14-prereg-grasp-sft-bootstrap.md, stage A).

No learning, no pixels: the expert reads the simulator's own state
(benchy pose, jaw pad frames, disk center) and drives the follower arm
through a waypoint grasp — approach above the boat, align the jaw axis
perpendicular to the hull, descend, pinch, lift, traverse to the disk,
lower, release, retreat. Position IK is damped least squares on the
``gripperframe`` site over the five arm joints, seeded from the
menagerie pickup keyframe (the pose probe P4 proved pinches hold);
jaw-axis alignment iterates ``wrist_roll`` against the FK'd pad
geoms. All kinematics run on a scratch ``MjData`` — the live physics
state is never touched by planning.

Contamination guard (pre-reg §3): demo seeds must come from
``DEMO_SEED_BASE``+ — ``run_expert_episode`` REFUSES eval seeds 0-99
so the frozen sim100 holdout can never leak into training data.

Stage-A validation (the registered ≥70% gate read) runs post-
finalization via ``run_expert_episode`` over 20 demo seeds; the CPU
oracles in tests/test_scripted_expert.py pin the IK/alignment math
kinematically without GL.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .so101_sim import HOME_DEGREES, PHYSICS_STEPS_PER_TICK, SO101Sim

#: First legal demo seed — everything below is the frozen sim100
#: eval holdout (pre-reg §3: eval seeds NEVER appear in demos).
DEMO_SEED_BASE = 1000

#: Menagerie scene_box.xml "pickup" keyframe (radians, jaw open) —
#: the IK seed; its grasp point sits ~0.22 m forward of the base,
#: inside the benchy spawn band (probe_benchy_contact P4).
PICKUP_QPOS = np.array([0.0, 0.000382, 0.4735, 1.17717, 1.58437, 0.727663])

JAW_OPEN_RAD = float(PICKUP_QPOS[5])
JAW_CLOSED_RAD = 0.0  # the P4 pinch close target
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MAX_STEP_DEG = 4.0  # per-tick joint-target slew (30 Hz)


def _quat_yaw(qpos_quat: np.ndarray) -> float:
    w, x, y, z = qpos_quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class ExpertPlanner:
    """Kinematic planning half — pure functions of a scratch MjData.

    Owns the IK and the jaw-axis alignment; never mutates the live
    ``sim.data``."""

    def __init__(self, sim: Any) -> None:
        # Any: the planner only reads ``.model`` — tests drive it with
        # a model-only stand-in (no renderer, no post pipeline).
        self.model = sim.model
        self.scratch = mujoco.MjData(self.model)
        self.site_id = self.model.site("gripperframe").id
        self.arm_qpos = np.array(
            [self.model.jnt_qposadr[self.model.joint(n).id] for n in ARM_JOINTS],
        )
        self.arm_dofs = np.array(
            [self.model.jnt_dofadr[self.model.joint(n).id] for n in ARM_JOINTS],
        )
        self.jaw_qpos = int(
            self.model.jnt_qposadr[self.model.joint("gripper").id],
        )
        low, high = self.model.jnt_range[
            [self.model.joint(n).id for n in ARM_JOINTS],
        ].T
        self.arm_low, self.arm_high = low.copy(), high.copy()
        self.fixed_pad = self.model.geom("fixed_jaw_box1").id
        self.moving_pad = self.model.geom("moving_jaw_box1").id

    def _fk(self, arm_rad: np.ndarray, jaw_rad: float) -> None:
        self.scratch.qpos[:] = 0.0
        self.scratch.qpos[self.arm_qpos] = arm_rad
        self.scratch.qpos[self.jaw_qpos] = jaw_rad
        mujoco.mj_kinematics(self.model, self.scratch)
        # Jacobians read subtree-COM state — without this, mj_jacSite
        # returns garbage and the IK silently stalls.
        mujoco.mj_comPos(self.model, self.scratch)

    def jaw_axis(self, arm_rad: np.ndarray, jaw_rad: float) -> float:
        """World-XY angle of the fixed→moving jaw pad separation."""
        self._fk(arm_rad, jaw_rad)
        fixed = self.scratch.geom_xpos[self.fixed_pad]
        moving = self.scratch.geom_xpos[self.moving_pad]
        return float(np.arctan2(moving[1] - fixed[1], moving[0] - fixed[0]))

    def grasp_point(self, arm_rad: np.ndarray, jaw_rad: float) -> np.ndarray:
        """World midpoint of the jaw pads — the point the boat's hull
        must occupy for the P4 pinch."""
        self._fk(arm_rad, jaw_rad)
        return 0.5 * (
            self.scratch.geom_xpos[self.fixed_pad]
            + self.scratch.geom_xpos[self.moving_pad]
        )

    def solve_ik(
        self,
        target: np.ndarray,
        seed_rad: np.ndarray,
        *,
        iters: int = 60,
        damping: float = 1e-3,
        tol: float = 2e-3,
        free_dofs: int = 5,
    ) -> tuple[np.ndarray, float]:
        """Damped-least-squares position IK of ``gripperframe`` to
        ``target`` over the first ``free_dofs`` arm joints — the grasp
        solve runs with the WRIST LOCKED (free_dofs=3: pan/lift/elbow
        place the pads while wrist_flex holds the P4 keyframe's proven
        jaw pitch and wrist_roll holds the aligned axis; a free-wrist
        solution tips the jaw so its tip strikes the table and the
        servos stall short — measured as descend never descending).
        Returns (arm radians, final position error in meters)."""
        arm = seed_rad.copy()
        jacp = np.zeros((3, self.model.nv))
        dofs = self.arm_dofs[:free_dofs]
        for _ in range(iters):
            self._fk(arm, JAW_OPEN_RAD)
            site_pos = self.scratch.site_xpos[self.site_id]
            error = target - site_pos
            if float(np.linalg.norm(error)) < tol:
                break
            mujoco.mj_jacSite(self.model, self.scratch, jacp, None, self.site_id)
            j = jacp[:, dofs]
            jjt = j @ j.T + damping * np.eye(3)
            j_pinv = j.T @ np.linalg.inv(jjt)
            delta = j_pinv @ error
            # Posture task in the nullspace: pull toward the seed pose.
            # Unregularized DLS wanders to straight-arm solutions whose
            # gravity moment SATURATES the sysid'd shoulder servo
            # (measured: force pinned at the 3.478 limit, arm stalled
            # 20° short) — the pickup-keyframe basin is statically
            # proven, so solutions stay near it.
            posture = seed_rad[:free_dofs] - arm[:free_dofs]
            delta = delta + 0.2 * (posture - j_pinv @ (j @ posture))
            arm[:free_dofs] = arm[:free_dofs] + delta
            arm = np.clip(arm, self.arm_low, self.arm_high)
        self._fk(arm, JAW_OPEN_RAD)
        residual = float(
            np.linalg.norm(target - self.scratch.site_xpos[self.site_id]),
        )
        return arm, residual

    def solve_ik_pads(
        self,
        pad_target: np.ndarray,
        seed_rad: np.ndarray,
        *,
        rounds: int = 3,
        free_dofs: int = 5,
    ) -> tuple[np.ndarray, float]:
        """Position IK expressed in JAW-PAD-MIDPOINT space — the grasp
        geometry's frame (the boat's hull must occupy the pad midpoint,
        not the gripperframe site, which sits a few cm away). Each
        round re-measures the site→pad offset at the current estimate
        and re-solves the site IK for (target − offset)."""
        arm = seed_rad.copy()
        for _ in range(rounds):
            self._fk(arm, JAW_OPEN_RAD)
            offset = (
                0.5
                * (
                    self.scratch.geom_xpos[self.fixed_pad]
                    + self.scratch.geom_xpos[self.moving_pad]
                )
                - self.scratch.site_xpos[self.site_id]
            )
            arm, _ = self.solve_ik(pad_target - offset, arm, free_dofs=free_dofs)
        residual = float(
            np.linalg.norm(pad_target - self.grasp_point(arm, JAW_OPEN_RAD)),
        )
        return arm, residual

    def solve_grasp(
        self,
        pad_target: np.ndarray,
        boat_yaw: float,
        seed_rad: np.ndarray,
        *,
        rounds: int = 3,
    ) -> tuple[np.ndarray, float]:
        """Grasp pose: pads AT the target AND jaw axis perpendicular to
        the hull with the P4 keyframe's jaw pitch held — alternate roll
        alignment with wrist-locked position IK (free-wrist solutions
        tip the jaw into the table; roll re-alignment after a free IK
        moves the pads centimeters — both measured as descend never
        descending)."""
        arm = seed_rad.copy()
        arm[3] = seed_rad[3]  # wrist_flex: the keyframe's proven pitch
        for _ in range(rounds):
            arm = self.align_wrist_roll(arm, boat_yaw, rounds=1)
            arm, _ = self.solve_ik_pads(pad_target, arm, rounds=2, free_dofs=4)
        residual = float(
            np.linalg.norm(pad_target - self.grasp_point(arm, JAW_OPEN_RAD)),
        )
        return arm, residual

    def align_wrist_roll(
        self,
        arm_rad: np.ndarray,
        boat_yaw: float,
        *,
        rounds: int = 3,
    ) -> np.ndarray:
        """Rotate ``wrist_roll`` so the jaw axis lands perpendicular to
        the boat's long axis (P4: the hull goes BETWEEN the pads).
        The roll→axis map is treated as locally 1:1 and iterated."""
        arm = arm_rad.copy()
        target_axis = boat_yaw + np.pi / 2
        for _ in range(rounds):
            axis = self.jaw_axis(arm, JAW_OPEN_RAD)
            delta = (target_axis - axis + np.pi / 2) % np.pi - np.pi / 2
            arm[4] = float(np.clip(arm[4] + delta, self.arm_low[4], self.arm_high[4]))
        return arm


@dataclass
class ExpertState:
    phase: str = "approach"
    ticks_in_phase: int = 0
    grasp_arm: np.ndarray | None = None
    # Wrist-roll branch: the jaw axis is a line (mod pi), so roll and
    # roll-pi pinch identically — but the gripper's overhang is NOT
    # symmetric, and for hull yaws that demand roll near 0 the default
    # branch lands the moving-jaw shell on the boat's deck and the arm
    # jams pressing on it (measured 22-40 N, pads stuck ~10 cm up).
    # The jam is detected PHYSICALLY (quiet arm, pads far from target)
    # and the flip retried once — kinematic overlap at solved poses
    # does not separate jamming from grazing branches (measured).
    roll_flip: bool = False
    # Flip BUDGET, not a one-shot: the jam-press nudges the boat, so
    # the branch that jammed can be clean on re-approach (gate read
    # 08-15: a held seed jammed on BOTH branches in sequence and the
    # exhausted one-shot left it wedged for the rest of the clock).
    flips_left: int = 3
    recover_arm: np.ndarray | None = None
    # Servo-droop compensation (descend): position servos settle a
    # few degrees short under gravity, leaving the pads ~2 cm off the
    # kinematic target — the measured pad error feeds back into the
    # IK target until the PHYSICAL pads bracket the hull.
    droop: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Servo-droop compensation (lower): the same undershoot under the
    # boat's added load stalls the radial trim short of the disk (gate
    # read 08-15: 4 held seeds parked at ~4.7 cm, arm quiet, full
    # phase clock burnt) — XY-only feedback of the measured pad error.
    place_droop: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Re-grasp recovery (loaded phases): a slipped pinch strands the
    # boat on the table mid-carry and the remaining phases run empty
    # (gate read 08-15: 3 held seeds; contacts gone, boat at z~0.3 cm).
    # One re-approach per episode, gated on a sustained no-contact
    # streak so a contact flicker never aborts a live carry.
    grip_lost_ticks: int = 0
    regrasp_tried: bool = False
    # Post-retry alignment dwell (descend): a pinch-miss retry enters
    # with the pads parked quiet at the failed pose — without a forced
    # re-measure it re-closes on tick 1 at the same pose forever. Zero
    # on a fresh approach: the dwell shifts the close pose and broke a
    # clean seed when applied unconditionally (measured, 08-15).
    descend_dwell: int = 0
    trace: list[str] = field(default_factory=list)


class ScriptedExpert:
    """The per-episode expert: ``action(sim)`` returns the next
    absolute joint target in DEGREES (rig order, [6]) and advances the
    phase machine on privileged reads. One instance serves ONE episode
    (fresh phase state), mirroring the wrist-transform convention."""

    CLEARANCE_Z = 0.055
    GRASP_Z = 0.014  # pad-midpoint height bracketing the hull (P4)
    PLACE_Z = 0.035
    # Set-down height over the disk: the grasp pad height plus the
    # disk's top surface (z 0.012) — releasing at PLACE_Z instead
    # drops the hull ~2 cm and it lands heeled (measured: upright
    # 0.88 vs the 0.9 success gate on 3 of 6 smoke misses).
    RELEASE_Z = 0.026
    SETTLE_TICKS = 60  # settle budget; keeps pressing-down bounded
    CLOSE_TICKS = 30  # 1 s pinch close+hold before lifting (P4 pace)
    OPEN_TICKS = 20
    PHASE_TIMEOUT = 200  # ticks; a stuck phase advances (never wedges)
    #: Droop/place integrator clip (m). ±0.04 was tuned on the v1
    #: spawn band (boat r_base ≤ ~0.27); a class attr so the spawn-v2
    #: robustness probe can sweep it against far-radius shoulder sag.
    DROOP_CLIP = 0.04
    #: Retreat ticks spent pulling up-and-back before slewing to HOME
    #: (clears the released boat before the big joint-space swing).
    RETREAT_UP_TICKS = 25

    def __init__(self, sim: SO101Sim) -> None:
        self.planner = ExpertPlanner(sim)
        self.state = ExpertState()
        self.disk = np.array([*sim.disk_center, 0.0])

    def _arm_now(self, sim: SO101Sim) -> np.ndarray:
        return sim.data.qpos[self.planner.arm_qpos].copy()

    def _command(self, arm_target_rad: np.ndarray, jaw_rad: float) -> np.ndarray:
        """Absolute joint target in degrees — commanded directly, the
        P4 pattern: the position servos do their own ramping (slewing
        from the MEASURED pose compounds servo lag into a crawl)."""
        return np.rad2deg(np.concatenate([arm_target_rad, [jaw_rad]]))

    def _carry(
        self,
        sim: SO101Sim,
        arm_target_rad: np.ndarray,
        jaw_rad: float,
        *,
        rate_deg: float = 1.5,
    ) -> np.ndarray:
        """Slew-limited target for the LOADED phases (traverse/lower):
        a full-speed swing flings or slips the pinched boat (P4
        measured torsional slip); a ~1.5°/tick crawl keeps the grip."""
        now = np.rad2deg(self._arm_now(sim))
        want = np.rad2deg(arm_target_rad)
        step = np.clip(want - now, -rate_deg, rate_deg)
        return np.concatenate([now + step, [np.rad2deg(jaw_rad)]])

    def _enter(self, phase: str) -> None:
        self.state.trace.append(f"{phase}@{self.state.ticks_in_phase}")
        self.state.phase = phase
        self.state.ticks_in_phase = 0

    def _grasp_seed(self) -> np.ndarray:
        seed = PICKUP_QPOS[:5].copy()
        if self.state.roll_flip:
            seed[4] -= np.pi  # the mod-pi flipped branch (clips in range)
        return seed

    def _jam_flip(self, sim: SO101Sim, err_m: float) -> bool:
        """Deck-strike recovery (approach/descend): the arm has gone
        QUIET far from its target — it is pressing on the boat's deck
        (measured 22-40 N, pads stuck ~10 cm up), and a force-clamped
        servo never pushes through. Flip the roll branch once and
        restart the approach (fresh droop: the clamped integrator is
        part of the jam state). Physical detection, not kinematic:
        solved-pose overlap does not separate jamming branches from
        grazing-but-fine ones (measured across the smoke seeds)."""
        state = self.state
        if state.flips_left <= 0 or state.ticks_in_phase <= 50 or err_m < 0.035:
            return False
        if float(np.abs(sim.data.qvel[self.planner.arm_dofs]).max()) > 0.05:
            return False
        state.flips_left -= 1
        state.roll_flip = not state.roll_flip
        state.droop[:] = 0.0
        state.trace.append("jam-flip")
        # Retreat up-and-back BEFORE re-approaching: flipping the
        # wrist 180 deg next to the hull flings the boat (measured
        # 31 cm on the retry without the retreat).
        state.recover_arm = self._arm_now(sim)
        state.recover_arm[1] -= np.deg2rad(35.0)
        self._enter("recover")
        return True

    def action(self, sim: SO101Sim) -> np.ndarray:
        planner, state = self.planner, self.state
        state.ticks_in_phase += 1
        boat_pos, _ = sim.benchy_pose()
        boat_yaw = _quat_yaw(
            sim.data.qpos[sim._benchy_qpos + 3 : sim._benchy_qpos + 7],
        )
        timeout = state.ticks_in_phase > self.PHASE_TIMEOUT

        pads_now = planner.grasp_point(self._arm_now(sim), JAW_OPEN_RAD)

        if state.phase in ("lift", "traverse", "lower"):
            fixed, moving = sim.benchy_grip_contacts()
            state.grip_lost_ticks = 0 if fixed or moving else state.grip_lost_ticks + 1
            on_disk = float(np.hypot(*(boat_pos[:2] - self.disk[:2]))) < 0.03
            if (
                state.grip_lost_ticks >= 15
                and boat_pos[2] < 0.005
                and not on_disk
                and not state.regrasp_tried
            ):
                state.regrasp_tried = True
                state.grip_lost_ticks = 0
                state.droop[:] = 0.0
                state.trace.append("regrasp")
                self._enter("approach")

        if state.phase == "recover":
            assert state.recover_arm is not None
            if state.ticks_in_phase >= 25:
                self._enter("approach")
            return self._command(state.recover_arm, JAW_OPEN_RAD)

        if state.phase == "approach":
            hover = boat_pos + np.array([0.0, 0.0, self.CLEARANCE_Z])
            hover_err = float(np.linalg.norm(pads_now - hover))
            if self._jam_flip(sim, hover_err):
                return self._command(self._arm_now(sim), JAW_OPEN_RAD)
            arm, _ = planner.solve_grasp(hover, boat_yaw, self._grasp_seed())
            if hover_err < 0.03 or timeout:
                self._enter("descend")
            return self._command(arm, JAW_OPEN_RAD)

        if state.phase == "descend":
            grasp = np.array([boat_pos[0], boat_pos[1], self.GRASP_Z])
            if self._jam_flip(sim, float(np.linalg.norm(pads_now - grasp))):
                return self._command(self._arm_now(sim), JAW_OPEN_RAD)
            # Settle-measure-correct: the sysid'd shoulder saturates
            # its force limit before the kinematic pose is reached
            # (pads float ~2 cm high and ~1.5 cm short) — so once the
            # arm is QUIET, fold the full measured pad error into the
            # target and re-solve. XY is what decides the pinch (the
            # jaw boxes span z; P4 tip-grips lift fine): close fires
            # on XY alignment, not pad-center depth.
            arm_speed = float(np.abs(sim.data.qvel[planner.arm_dofs]).max())
            if state.ticks_in_phase > 20 and arm_speed < 0.08:
                state.droop += grasp - pads_now
                state.droop[2] = min(state.droop[2], 0.0)
                state.droop = np.clip(state.droop, -self.DROOP_CLIP, self.DROOP_CLIP)
            arm, _ = planner.solve_grasp(
                grasp + state.droop,
                boat_yaw,
                self._grasp_seed(),
            )
            xy_err = float(np.hypot(*(pads_now[:2] - grasp[:2])))
            aligned = (
                xy_err < 0.008
                and arm_speed < 0.08
                and state.ticks_in_phase > state.descend_dwell
            )
            if aligned or timeout:
                state.grasp_arm = arm
                self._enter("close")
            return self._command(arm, JAW_OPEN_RAD)

        if state.phase == "close":
            assert state.grasp_arm is not None
            if state.ticks_in_phase >= self.CLOSE_TICKS:
                fixed, moving = sim.benchy_grip_contacts()
                if fixed and moving:
                    self._enter("lift")
                else:
                    # Fresh droop: the failed pinch's integrator is
                    # tuned to the pose that just missed (and a jam's
                    # clip-saturated residual poisons every later
                    # attempt — gate read 08-15).
                    state.droop[:] = 0.0
                    state.descend_dwell = 6
                    self._enter("descend")
                    state.trace.append("pinch-miss-retry")
            return self._command(state.grasp_arm, JAW_CLOSED_RAD)

        if state.phase == "lift":
            assert state.grasp_arm is not None
            lifted = state.grasp_arm.copy()
            lifted[1] -= np.deg2rad(35.0)  # the P4 lift recipe, deeper
            # The saturated servo nets ~1.5-2 cm of boat height — a
            # clean hold, so 1.2 cm IS lifted (3.5 cm never fires and
            # burnt the phase clock).
            if (boat_pos[2] > 0.012 and state.ticks_in_phase > 30) or timeout:
                self._enter("traverse")
            return self._command(lifted, JAW_CLOSED_RAD)

        if state.phase == "traverse":
            # Pure PAN arc: pan's axis is vertical so it carries no
            # gravity load (no saturation) — swinging at the lifted
            # posture preserves the carry height exactly, and the disk
            # sits at nearly the spawn band's radius. Steer the BOAT's
            # bearing (it hangs off-center in the grip) to the disk's.
            arm = self._arm_now(sim)
            boat_bearing = float(np.arctan2(boat_pos[1], boat_pos[0]))
            disk_bearing = float(np.arctan2(self.disk[1], self.disk[0]))
            # Pan and world bearing are NEGATIVELY coupled (measured:
            # pan +0.2 rad moves the pads' bearing −9.4°).
            arm[0] = float(
                np.clip(
                    arm[0] - (disk_bearing - boat_bearing),
                    planner.arm_low[0],
                    planner.arm_high[0],
                ),
            )
            aligned = abs(disk_bearing - boat_bearing) < np.deg2rad(3.0)
            if aligned or timeout:
                state.place_droop[:] = 0.0
                self._enter("lower")
            return self._carry(sim, arm, JAW_CLOSED_RAD, rate_deg=2.5)

        if state.phase == "lower":
            # Radial correction at the held bearing: steer the BOAT's
            # radius to the disk's with shoulder/elbow only (pan holds
            # the bearing; wrist holds the grip pose).
            grip_offset = pads_now - boat_pos
            place = (
                self.disk
                + np.array([0.0, 0.0, self.PLACE_Z])
                + np.array([grip_offset[0], grip_offset[1], 0.0])
            )
            # The descend lesson under the boat's added load: the
            # force-saturated arm parks the pads SHORT of the solved
            # target and the radial trim stalls (gate read 08-15:
            # 4 held seeds quiet at ~4.7 cm for the full clock) —
            # once quiet and unplaced, fold the measured pad error
            # into the target. XY only: z feedback would press the
            # hull into the disk (the settle phase owns set-down).
            arm_speed = float(np.abs(sim.data.qvel[planner.arm_dofs]).max())
            if state.ticks_in_phase > 20 and arm_speed < 0.08:
                pad_err = place - pads_now
                state.place_droop += np.array([pad_err[0], pad_err[1], 0.0])
                state.place_droop = np.clip(
                    state.place_droop,
                    -self.DROOP_CLIP,
                    self.DROOP_CLIP,
                )
            arm, _ = planner.solve_ik_pads(
                place + state.place_droop,
                self._arm_now(sim),
                free_dofs=3,
            )
            boat_placed = float(np.hypot(*(boat_pos[:2] - self.disk[:2]))) < 0.03
            if boat_placed or timeout:
                self._enter("settle")
            return self._carry(sim, arm, JAW_CLOSED_RAD, rate_deg=2.0)

        if state.phase == "settle":
            # Set the boat DOWN before letting go: keep the pinch and
            # lower the pad target to RELEASE_Z so the keel touches the
            # disk before the jaws open. Exit when the boat base is at
            # disk-top height with the arm quiet (never press past the
            # settle budget — pinning the hull tips it on retreat).
            grip_offset = pads_now - boat_pos
            place = (
                self.disk
                + np.array([0.0, 0.0, self.RELEASE_Z])
                + np.array([grip_offset[0], grip_offset[1], 0.0])
            )
            arm, _ = planner.solve_ik_pads(
                place,
                self._arm_now(sim),
                free_dofs=3,
            )
            arm_speed = float(np.abs(sim.data.qvel[planner.arm_dofs]).max())
            down = boat_pos[2] < 0.0135 and arm_speed < 0.08
            if down or state.ticks_in_phase > self.SETTLE_TICKS or timeout:
                self._enter("open")
            return self._carry(sim, arm, JAW_CLOSED_RAD, rate_deg=1.5)

        if state.phase == "open":
            if state.ticks_in_phase >= self.OPEN_TICKS:
                self._enter("retreat")
            return self._command(self._arm_now(sim), JAW_OPEN_RAD)

        # retreat: pull UP AND BACK in joint space first (an IK swing
        # back through the drop point re-contacts the released boat and
        # tips it — measured), then return to the HOME rest pose (owner
        # steering 2026-08-16 13:46Z: demos end with the arm parked and
        # the boat left standing on the disk — the collector records
        # this tail and re-verifies success after it).
        if state.ticks_in_phase <= self.RETREAT_UP_TICKS:
            parked = self._arm_now(sim)
            parked[1] = parked[1] - np.deg2rad(30.0)
            return self._command(parked, JAW_OPEN_RAD)
        return self._command(np.deg2rad(HOME_DEGREES[:5]), JAW_OPEN_RAD)


def run_expert_episode(
    sim: SO101Sim | Any,
    seed: int,
    *,
    max_ticks: int = 600,
    render: bool = False,
    on_obs: Callable[[Any], None] | None = None,
    reset_kwargs: dict[str, Any] | None = None,
) -> dict[str, object]:
    """One privileged episode: reset, run the expert to the clock,
    return the outcome row. ``render=False`` steps physics without
    calling ``observe()`` (planning is pixel-free) — the reset render
    itself is unavoidable (reset returns an observation).

    ``on_obs`` receives every SimObservation (reset + one per tick) —
    the gate read's video bank rides here; it needs ``render=True``
    because the unrendered path never produces frames.

    REFUSES eval seeds (< DEMO_SEED_BASE): the frozen sim100 holdout
    never appears in demo generation (pre-reg §3)."""
    if seed < DEMO_SEED_BASE:
        raise ValueError(
            f"seed {seed} is inside the frozen eval holdout — demo seeds "
            f"start at {DEMO_SEED_BASE} (pre-reg contamination guard)",
        )
    if on_obs is not None and not render:
        raise ValueError("on_obs needs render=True — no frames otherwise")
    obs = sim.reset(seed, **(reset_kwargs or {}))
    if on_obs is not None:
        on_obs(obs)
    expert = ScriptedExpert(sim)
    tick = 0
    for tick in range(max_ticks):  # noqa: B007 — tick is the returned count
        action = expert.action(sim)
        if render:
            obs = sim.step(action)
            if on_obs is not None:
                on_obs(obs)
        else:
            target = np.clip(
                np.deg2rad(action),
                sim._ctrl_low,
                sim._ctrl_high,
            )
            sim.data.ctrl[sim._actuator_ids] = target
            mujoco.mj_step(sim.model, sim.data, nstep=PHYSICS_STEPS_PER_TICK)
        if sim.success():
            break
    _pos, upright = sim.benchy_pose()
    return {
        "seed": seed,
        "success": bool(sim.success()),
        "ticks": tick + 1,
        "final_disk_cm": sim.benchy_disk_distance() * 100,
        "upright": upright,
        "phase_trace": [*expert.state.trace, expert.state.phase],
    }
